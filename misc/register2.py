import cv2
import numpy as np
import time
import os
from insightface.app import FaceAnalysis
from insightface.model_zoo.scrfd import SCRFD
from db_manager import insert_user_embedding
from face_layout import load_mask, overlay_mask, check_alignment

class AutoFaceCapture:
    def __init__(self, video_source=0):
        # Initialize embedding model
        self.recognizer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=0)

        # Load SCRFD model with fixed input size
        self.model_path = "models/scrfd_10g_shape1280x1280.onnx"
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")

        self.detector = SCRFD(model_file=self.model_path)
        self.detector.prepare(ctx_id=0, input_size=(1280, 1280), providers=['CPUExecutionProvider'])

        # Camera setup
        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.embeddings = []
        self.angle_sequence = ['front', 'left', 'right', 'up', 'down']
        self.current_angle_idx = 0
        self.alignment_start_time = 0
        self.required_alignment_time = 2.0
        self.mask_rect = None

    def register_user(self, reg_no, name):
        print(f"\n[INFO] Starting registration for {name} ({reg_no})")

        while self.current_angle_idx < len(self.angle_sequence):
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            current_angle = self.angle_sequence[self.current_angle_idx]
            aligned = False
            mask = None

            # Resize + pad to 1280x1280
            resized, scale, pad = self._resize_with_padding(frame, target_size=1280)

            try:
                bboxes, _ = self.detector.detect(resized, input_size=(1280, 1280))
            except Exception as e:
                print(f"[ERROR] Detection error: {e}")
                continue

            if len(bboxes) > 0:
                bbox = bboxes[0].astype(int)
                x1, y1, x2, y2 = self._convert_coords_back(bbox[:4], scale, pad)

                if current_angle in ['front', 'left', 'right']:
                    try:
                        mask = load_mask(current_angle, frame.shape)
                    except Exception as e:
                        print(f"[ERROR] Mask load: {e}")
                        continue

                if mask is not None:
                    frame, self.mask_rect = overlay_mask(frame, mask, (x1, y1, x2, y2))
                    aligned = check_alignment((x1, y1, x2, y2), self.mask_rect)
                else:
                    aligned = True

                if aligned:
                    if self.alignment_start_time == 0:
                        self.alignment_start_time = time.time()
                        print(f"[INFO] Holding still for {current_angle.upper()}...")
                    elif time.time() - self.alignment_start_time >= self.required_alignment_time:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            faces = self.recognizer.get(face_crop)
                            if faces:
                                self.capture_embedding(faces[0], current_angle)
                        continue
                else:
                    self.alignment_start_time = 0

                cv2.rectangle(frame, (x1, y1), (x2, y2), 
                             (0, 255, 0) if aligned else (0, 0, 255), 2)
                cv2.putText(frame, f"{current_angle.upper()} - {'ALIGNED' if aligned else 'ALIGN...'}",
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                           (0, 255, 0) if aligned else (0, 0, 255), 2)
            else:
                self.alignment_start_time = 0
                cv2.putText(frame, "NO FACE DETECTED", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mask is None:
                cv2.putText(frame, f"Look {current_angle.upper()}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            display_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Scaled for laptop display
            cv2.imshow("Face Registration", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Registration cancelled.")
                break


        self.cap.release()
        cv2.destroyAllWindows()

        if self.embeddings:
            success = insert_user_embedding(reg_no, name, self.embeddings)
            print(f"\n[✅] Registration {'SUCCESS' if success else 'FAILED'} — {len(self.embeddings)} angles captured")
        else:
            print("[❌] No embeddings captured")

    def _resize_with_padding(self, frame, target_size=1280):
        h, w = frame.shape[:2]
        scale = target_size / max(w, h)
        resized = cv2.resize(frame, (int(w * scale), int(h * scale)))

        top = (target_size - resized.shape[0]) // 2
        bottom = target_size - resized.shape[0] - top
        left = (target_size - resized.shape[1]) // 2
        right = target_size - resized.shape[1] - left

        padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return padded, scale, (left, top)

    def _convert_coords_back(self, bbox, scale, pad):
        left, top = pad
        x1 = int((bbox[0] - left) / scale)
        y1 = int((bbox[1] - top) / scale)
        x2 = int((bbox[2] - left) / scale)
        y2 = int((bbox[3] - top) / scale)
        return x1, y1, x2, y2

    def capture_embedding(self, face, angle):
        self.embeddings.append({
            'angle': angle,
            'embedding': face.embedding.astype(np.float32).tolist()
        })
        print(f"[✔] Captured {angle.upper()} view")
        self.current_angle_idx += 1
        self.alignment_start_time = 0

if __name__ == "__main__":
    print("=== FACE REGISTRATION SYSTEM ===")
    name = input("Enter user's name: ").strip()
    reg_no = input("Enter registration number: ").strip()

    if name and reg_no:
        AutoFaceCapture(video_source=0).register_user(reg_no, name)
    else:
        print("❌ Name and registration number are required")
