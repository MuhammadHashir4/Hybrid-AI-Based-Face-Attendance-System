import cv2
import faiss
import numpy as np
import pickle
from datetime import datetime
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from db_manager import load_all_user_embeddings, mark_attendance
from hybrid_fallback import facepp_match_with_registered_images

class FaceRecognizer:
    def __init__(self):
        scrfd = get_model("models/scrfd_10g_bnkps.onnx", providers=["CPUExecutionProvider"])
        scrfd.input_size = (640, 640)
        self.detector = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
        self.detector.det_model = scrfd
        self.detector.prepare(ctx_id=0, det_size=(640, 640))

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.MIN_FACE_SIZE = 50
        self.THRESH = 0.48
        self.CONFIRM = 1  # for testing, mark after 1 detection
        self.COOLDOWN = 20

        self.index = {}
        self.recent = {}
        self.attendance_log = {}

        self._load_embeddings()

    def _load_embeddings(self):
        for reg_no, name, angle, blob in load_all_user_embeddings():
            emb = np.array(pickle.loads(blob), dtype=np.float32).reshape(1, -1)
            faiss.normalize_L2(emb)
            if angle not in self.index:
                self.index[angle] = {'idx': faiss.IndexFlatIP(emb.shape[1]), 'meta': []}
            self.index[angle]['idx'].add(emb)
            self.index[angle]['meta'].append((reg_no, name))

    def _draw(self, frame, bbox, text, color):
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(frame, text, (bbox[0], max(20, bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def recognize(self):
        print("[SYSTEM] Hybrid Recognition Started")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            faces = self.detector.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                fallback_reason = None

                if w < self.MIN_FACE_SIZE or h < self.MIN_FACE_SIZE:
                    fallback_reason = "TOO SMALL"
                else:
                    try:
                        emb = np.array(face.embedding, dtype=np.float32).reshape(1, -1)
                        faiss.normalize_L2(emb)
                    except Exception:
                        fallback_reason = "EMBEDDING ERROR"

                match_result = None
                if fallback_reason is None:
                    best = {'reg_no': None, 'name': None, 'score': -1}
                    for angle, data in self.index.items():
                        score, idx = data['idx'].search(emb, 1)
                        if score[0][0] > best['score']:
                            best = {
                                'reg_no': data['meta'][idx[0][0]][0],
                                'name': data['meta'][idx[0][0]][1],
                                'score': score[0][0]
                            }

                    if best['score'] >= self.THRESH:
                        reg = best['reg_no']
                        now = datetime.now()
                        print(f"[DEBUG] {best['name']} recognized with score {best['score']}")

                        if reg not in self.recent:
                            self.recent[reg] = {'count': 1, 'time': now}
                        else:
                            self.recent[reg]['count'] += 1
                            self.recent[reg]['time'] = now

                        print(f"[DEBUG] Count: {self.recent[reg]['count']}")

                        if self.recent[reg]['count'] >= self.CONFIRM:
                            last_marked = self.attendance_log.get(reg)
                            if last_marked is None or (now - last_marked).total_seconds() > self.COOLDOWN:
                                print(f"[INFO] Marking attendance for {best['name']} ({reg})")
                                mark_attendance(reg, best['name'])
                                self.attendance_log[reg] = now
                            self._draw(frame, bbox, f"{best['name']} ✔", (0,255,0))
                        else:
                            self._draw(frame, bbox,
                                       f"{best['name']} {self.recent[reg]['count']}/{self.CONFIRM}",
                                       (0,255,255))
                        continue  # skip fallback
                    else:
                        fallback_reason = "LOW CONFIDENCE"

                cropped = frame[y1:y2, x1:x2]
                if cropped is None or cropped.size == 0:
                    cropped = frame

                result = facepp_match_with_registered_images(cropped, full_frame=frame)
                if result:
                    reg_no, name = result
                    now = datetime.now()
                    if reg_no not in self.recent:
                        self.recent[reg_no] = {'count': 1, 'time': now}
                    else:
                        self.recent[reg_no]['count'] += 1
                        self.recent[reg_no]['time'] = now

                    if self.recent[reg_no]['count'] >= self.CONFIRM:
                        last_marked = self.attendance_log.get(reg_no)
                        if last_marked is None or (now - last_marked).total_seconds() > self.COOLDOWN:
                            print(f"[INFO] Marking attendance for {name} ({reg_no}) via API")
                            mark_attendance(reg_no, name)
                            self.attendance_log[reg_no] = now
                        self._draw(frame, bbox, f"{name} ✔ (API)", (255,255,0))
                    else:
                        self._draw(frame, bbox,
                                   f"{name} {self.recent[reg_no]['count']}/{self.CONFIRM} (API)",
                                   (255,255,0))
                else:
                    self._draw(frame, bbox, f"UNKNOWN ({fallback_reason or 'API'})", (0,0,255))

            cv2.imshow("Hybrid Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    FaceRecognizer().recognize()
