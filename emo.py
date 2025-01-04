import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageDraw
import mediapipe as mp
import numpy as np
import time
import sqlite3
import json
import uuid
from datetime import datetime

class SQLiteDatabase:
    def __init__(self, db_path="emotion_detection.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Create sessions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP,
                    device_id TEXT
                )
            ''')
            # Create detections table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detections (
                    detection_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TIMESTAMP,
                    emotion TEXT,
                    confidence REAL,
                    face_location TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            conn.commit()
    
    def create_session(self):
        session_id = str(uuid.uuid4())
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sessions (session_id, start_time, device_id)
                VALUES (?, ?, ?)
            ''', (session_id, datetime.now(), 'WEBCAM-0'))
            conn.commit()
        return session_id
    
    def store_detection(self, session_id, emotion, confidence, face_location):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO detections 
                (detection_id, session_id, timestamp, emotion, confidence, face_location)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                str(uuid.uuid4()),
                session_id,
                datetime.now(),
                emotion,
                float(confidence),
                json.dumps(face_location)
            ))
            conn.commit()
    
    def get_recent_detections(self, limit=10):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT timestamp, emotion, confidence
                FROM detections
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()

class EmotionRecognitionModel:
    def __init__(self):
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 7)
        self.model.eval()
        
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        try:
            with torch.no_grad():
                image = image.convert('RGB')
                input_tensor = self.transform(image).unsqueeze(0)
                output = self.model(input_tensor)
                confidence, predicted = torch.max(output, 1)
                return self.emotion_labels[predicted.item()], confidence.item()
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Unknown", 0.0

class EmotionDetectionSystem:
    def __init__(self):
        self.emotion_model = EmotionRecognitionModel()
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.2)
        self.db = SQLiteDatabase()
        self.session_id = self.db.create_session()

    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        results = self.face_detection.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x = max(0, int(bboxC.xmin * w))
                y = max(0, int(bboxC.ymin * h))
                width = min(int(bboxC.width * w), w - x)
                height = min(int(bboxC.height * h), h - y)

                face = pil_image.crop((x, y, x + width, y + height))
                emotion, confidence = self.emotion_model.predict(face)

                face_location = {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
                self.db.store_detection(
                    self.session_id, 
                    emotion, 
                    confidence, 
                    face_location
                )

                draw = ImageDraw.Draw(pil_image)
                draw.rectangle([x, y, x + width, y + height], 
                             outline="green", width=2)
                draw.text((x, y - 10), 
                         f"{emotion} ({confidence:.2f})", 
                         fill="green")

        return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def run(self):
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_frame(frame)
                cv2.imshow('Emotion Recognition', processed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.face_detection.close()

def main():
    try:
        system = EmotionDetectionSystem()
        system.run()
    except Exception as e:
        print(f"System error: {e}")

if __name__ == "__main__":
    main()
