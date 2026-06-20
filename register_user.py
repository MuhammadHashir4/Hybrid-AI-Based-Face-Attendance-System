import cv2
import numpy as np
import os
import time
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from db_manager import insert_user_embedding
from face_layout import load_mask, overlay_mask, check_alignment

class AutoFaceCapture:
    def __init__(self, video_source=0, min_side=48, min_area=4096):
        self.analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        scrfd = get_model("models/scrfd_10g_bnkps.onnx", providers=['CPUExecutionProvider'])
        scrfd.input_size = (320, 320)
        self.analyzer.det_model = scrfd
        self.analyzer.prepare(ctx_id=0, det_size=(320, 320))

        self.cap = cv2.VideoCapture(video_source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.embeddings = []
        self.angle_sequence = ['front', 'left', 'right', 'up', 'down']
        self.current_angle_idx = 0
        self.alignment_start_time = 0
        self.required_alignment_time = 2.0
        self.mask_rect = None
        self.reference_dir = None

        self.MIN_SIDE = min_side
        self.MIN_AREA = min_area

    def register_user(self, reg_no, name):
        print(f"\nStarting registration for {name} ({reg_no})")
        self.reference_dir = os.path.join("reference_images", f"{reg_no}_{name.replace(' ', '')}")
        os.makedirs(self.reference_dir, exist_ok=True)

        while self.current_angle_idx < len(self.angle_sequence):
            ret, frame = self.cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            current_angle = self.angle_sequence[self.current_angle_idx]
            aligned = False
            mask = None

            if current_angle in ['front', 'left', 'right']:
                try:
                    mask = load_mask(current_angle, frame.shape)
                except Exception as e:
                    print(f"Mask load error: {e}")
                    continue

            faces = self.analyzer.get(frame)

            if faces:
                face = faces[0]
                bbox = face.bbox.astype(int)

                if mask is not None:
                    frame, self.mask_rect = overlay_mask(frame, mask, bbox)
                    aligned = check_alignment(bbox, self.mask_rect)
                else:
                    aligned = True

                if aligned:
                    if self.alignment_start_time == 0:
                        self.alignment_start_time = time.time()
                        print(f"[INFO] Hold still for {current_angle.upper()}...")
                    elif time.time() - self.alignment_start_time >= self.required_alignment_time:
                        self.capture_embedding_and_image(face, frame, current_angle)
                        continue
                else:
                    self.alignment_start_time = 0

                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (0, 255, 0) if aligned else (0, 0, 255), 2)
                cv2.putText(frame, f"{current_angle.upper()} - {'ALIGNED' if aligned else 'ALIGN...'}",
                            (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if aligned else (0, 0, 255), 2)

            else:
                self.alignment_start_time = 0
                cv2.putText(frame, "NO FACE DETECTED", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if mask is None:
                cv2.putText(frame, f"Look {current_angle.upper()}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Face Registration", frame)
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

    def capture_embedding_and_image(self, face, frame, angle):
        x1, y1, x2, y2 = face.bbox.astype(int)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = face_crop.shape[:2]

        if crop_w < self.MIN_SIDE or crop_h < self.MIN_SIDE or (crop_w * crop_h) < self.MIN_AREA:
            print(f"[❌] Face crop too small for angle '{angle}' (size: {crop_w}x{crop_h}). Please come closer/aligned.")
            # Do not append embedding or save image if crop is too small
            return

        self.embeddings.append({
            'angle': angle,
            'embedding': face.embedding.astype(np.float32).tolist()
        })
        print(f"[✔] Captured {angle.upper()} view (size: {crop_w}x{crop_h})")

        image_path = os.path.join(self.reference_dir, f"{angle}.jpg")
        cv2.imwrite(image_path, face_crop)

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