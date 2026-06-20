import onnxruntime
import numpy as np
import torch
import cv2
import time
import torch.nn.functional as F  # for softmax

class LivenessDetector:
    def __init__(self, model_path="liveness/2.7_80x80_MiniFASNetV2.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.threshold = 0.4  # Recommended liveness threshold
        self.consecutive_frames = []
        self.buffer_size = 5
        self.min_face_size = 80
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # 200ms between inferences

    def _validate_face(self, face_img):
        """Check face image quality (size, blur, brightness)"""
        if face_img.size == 0 or min(face_img.shape[:2]) < self.min_face_size:
            return False
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        brightness = np.mean(gray)
        print(f"[DEBUG] Blur={blur:.1f}, Brightness={brightness:.1f}")
        return blur > 30 and 20 < brightness < 240

    def preprocess(self, face_img):
        """Resize and normalize"""
        face_img = cv2.resize(face_img, (80, 80))
        img = face_img.astype(np.float32)
        img = (img - 127.5) / 128.0  # Use the correct training normalization
        img = np.transpose(img, (2, 0, 1))  # CHW format
        return np.expand_dims(img, axis=0)

    def predict(self, face_img):
        current_time = time.time()

        if not self._validate_face(face_img):
            print("[DEBUG] Face validation failed.")
            return False

        if current_time - self.last_prediction_time < self.prediction_interval:
            if len(self.consecutive_frames) == self.buffer_size:
                avg_score = np.mean(self.consecutive_frames)
                print(f"[DEBUG] (buffer only) Avg Liveness Score: {avg_score:.3f}")
                return avg_score > self.threshold
            return False

        self.last_prediction_time = current_time

        try:
            input_tensor = self.preprocess(face_img)
            outputs = self.session.run(None, {self.input_name: input_tensor})
            logits = outputs[0][0]  # Raw output

            # Apply softmax
            scores = F.softmax(torch.tensor(logits), dim=0).numpy()
            prob_real = float(scores[1])  # Assuming index 1 = live class

            # Debug logs
            print(f"[DEBUG] Raw ONNX output: {logits}")
            print(f"[DEBUG] Softmax scores: {scores}")
            print(f"[DEBUG] Live Prob: {prob_real:.3f}")

            # Update buffer
            self.consecutive_frames.append(prob_real)
            if len(self.consecutive_frames) > self.buffer_size:
                self.consecutive_frames.pop(0)

            avg_score = np.mean(self.consecutive_frames)
            print(f"[DEBUG] Avg Score: {avg_score:.3f}")
            return avg_score > self.threshold

        except Exception as e:
            print("[ERROR] ONNX inference failed:", str(e))
            return False

    def reset(self):
        self.consecutive_frames = []
