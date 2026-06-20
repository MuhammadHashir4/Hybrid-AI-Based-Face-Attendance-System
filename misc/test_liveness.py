import cv2
from liveness_detector import LivenessDetector

# Initialize detector with your converted ONNX model
detector = LivenessDetector("liveness/output_model.onnx")

# Capture one frame from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Failed to capture image.")
    exit()

# Show cropped face area
cv2.imshow("Captured Frame", frame)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# Predict liveness
is_live = detector.predict(frame)

# Output result
print("\n✅ Liveness Result:", "LIVE ✅" if is_live else "SPOOF ❌")
print("Confidence Buffer:", detector.consecutive_frames)
