from insightface.app import FaceAnalysis

# Initialize FaceAnalysis with SCRFD + ArcFace
app = FaceAnalysis(name="buffalo_l")  # buffalo_l includes SCRFD and ArcFace
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 for CPU

import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

faces = app.get(frame)

for face in faces:
    name = "Unknown"
    bbox = face.bbox.astype(int)
    embedding = face.embedding  # 512-d vector from ArcFace

    # Draw the face box and show embedding status
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imwrite("output.jpg", frame)
print("Output saved as output.jpg")
print(embedding)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
