import cv2
import faiss
import os
import pickle
import numpy as np
from datetime import datetime
from insightface.model_zoo.scrfd import SCRFD
from insightface.app import FaceAnalysis
from db_manager import load_all_user_embeddings, mark_attendance

class FaceRecognizer:
    def __init__(self):
        print("[INFO] Initializing recognizer...")

        # Model paths
        self.model_path = "models/scrfd_10g_shape1280x1280.onnx"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"[FATAL] Model not found at {self.model_path}")

        # Initialize detectors with correct input size
        self.detector = SCRFD(model_file=self.model_path)
        self.detector.prepare(ctx_id=0, input_size=(1280, 1280), providers=["CPUExecutionProvider"])
        
        self.recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=0)

        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Recognition parameters
        self.MIN_FACE_SIZE = 100
        self.THRESH = 0.55
        self.CONFIRM = 2
        self.COOLDOWN = 30

        self.index = {}
        self.recent = {}
        self.last_attendance = {}

        self._load_embeddings()
        print("[INFO] Recognizer initialized successfully")

    def _load_embeddings(self):
        print("[INFO] Loading user embeddings from DB...")
        data = load_all_user_embeddings()
        if not data:
            raise RuntimeError("[FATAL] No embeddings found in database")

        for reg_no, name, angle, blob in data:
            emb = np.array(pickle.loads(blob), dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(emb)
            if angle not in self.index:
                self.index[angle] = {'index': faiss.IndexFlatIP(emb.shape[1]), 'meta': []}
            self.index[angle]['index'].add(emb)
            self.index[angle]['meta'].append((reg_no, name))

        print(f"[INFO] Loaded {len(data)} embeddings")

    def _match(self, emb):
        best = {'reg_no': None, 'name': None, 'score': 0, 'angle': None}
        for angle, data in self.index.items():
            score, idx = data['index'].search(emb, 1)
            if score[0][0] > best['score']:
                best.update({
                    'reg_no': data['meta'][idx[0][0]][0],
                    'name': data['meta'][idx[0][0]][1],
                    'score': score[0][0],
                    'angle': angle
                })
        return best

    def _draw(self, frame, bbox, text, color):
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def recognize(self):
        print("[SYSTEM] Recognition started. Press 'q' to quit.")
        
        # Test camera connection
        if not self.cap.isOpened():
            raise RuntimeError("Could not open video stream")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] Frame not received. Retrying...")
                self.cap.release()
                self.cap = cv2.VideoCapture('http://192.168.18.120:8080/video')
                cv2.waitKey(1000)
                continue

            # Flip and resize frame to model's expected input size (1280x1280)
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            # Calculate padding to maintain aspect ratio
            target_size = 1280
            scale = target_size / max(height, width)
            resized = cv2.resize(frame, (0,0), fx=scale, fy=scale)
            h, w = resized.shape[:2]
            
            # Pad to make it square (1280x1280)
            pad_top = (target_size - h) // 2
            pad_bottom = target_size - h - pad_top
            pad_left = (target_size - w) // 2
            pad_right = target_size - w - pad_left
            padded = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, 
                                      cv2.BORDER_CONSTANT, value=[0,0,0])
            
            try:
                bboxes, _ = self.detector.detect(padded, input_size=(1280, 1280), max_num=5)
            except Exception as e:
                print(f"[WARN] Detection error: {str(e)}")
                continue

            for box in bboxes:
                x1, y1, x2, y2, _ = box.astype(int)
                # Convert coordinates back to original frame
                x1 = int((x1 - pad_left) / scale)
                y1 = int((y1 - pad_top) / scale)
                x2 = int((x2 - pad_left) / scale)
                y2 = int((y2 - pad_top) / scale)
                
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < self.MIN_FACE_SIZE or face_h < self.MIN_FACE_SIZE:
                    self._draw(frame, (x1, y1, x2, y2), "MOVE CLOSER", (0, 165, 255))
                    continue

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue
                    
                faces = self.recognizer.get(face_crop)
                if not faces:
                    self._draw(frame, (x1, y1, x2, y2), "UNKNOWN", (0, 0, 255))
                    continue

                face = faces[0]
                emb = np.array(face.embedding, dtype=np.float32).reshape(1, -1)
                faiss.normalize_L2(emb)
                best = self._match(emb)

                if best['score'] > self.THRESH:
                    reg_no = best['reg_no']
                    now = datetime.now()
                    if reg_no not in self.recent:
                        self.recent[reg_no] = {'count': 1, 'last_seen': now}
                    else:
                        self.recent[reg_no]['count'] += 1
                        self.recent[reg_no]['last_seen'] = now

                    if self.recent[reg_no]['count'] >= self.CONFIRM:
                        if not self.last_attendance.get(reg_no) or \
                                (now - self.last_attendance[reg_no]).total_seconds() > self.COOLDOWN:
                            mark_attendance(reg_no, best['name'])
                            self.last_attendance[reg_no] = now
                            status = f"{best['name']} ✔"
                            color = (0, 255, 0)
                        else:
                            status = f"{best['name']} (Cooldown)"
                            color = (0, 255, 255)
                    else:
                        status = f"{best['name']} {self.recent[reg_no]['count']}/{self.CONFIRM}"
                        color = (0, 255, 255)
                else:
                    status = "UNKNOWN"
                    color = (0, 0, 255)

                self._draw(frame, (x1, y1, x2, y2), status, color)

            # Display the resulting frame
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        FaceRecognizer().recognize()
    except Exception as e:
        print(f"[FATAL ERROR] {str(e)}")