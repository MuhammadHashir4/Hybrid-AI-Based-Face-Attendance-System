from insightface.app import FaceAnalysis
import cv2

# Auto-download and prepare the SCRFD 10G model with keypoints
print("[INFO] Loading SCRFD 10G KPS model...")
model = FaceAnalysis(name='scrfd_10g_kps', providers=['CPUExecutionProvider'])
model.prepare(ctx_id=0, det_size=(1024, 1024))

print("[INFO] Model loaded and ready.")
print("[INFO] Press 'q' to quit.")

# Start webcam to test detection
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    faces = model.get(frame)

    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if hasattr(face, 'kps'):
            for (x, y) in face.kps:
                cv2.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)

    cv2.putText(frame, f"Faces Detected: {len(faces)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SCRFD 10G KPS Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
