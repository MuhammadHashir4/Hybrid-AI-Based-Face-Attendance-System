import cv2
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model

class SCRFDDetector:
    def __init__(self, input_size=(640, 640)):
        print("[INFO] Initializing SCRFD face detector...")
        scrfd = get_model("models/scrfd_10g_bnkps.onnx", providers=["CPUExecutionProvider"])
        scrfd.input_size = input_size

        self.detector = FaceAnalysis(name='buffalo_l', providers=["CPUExecutionProvider"])
        self.detector.det_model = scrfd
        self.detector.prepare(ctx_id=0, det_size=input_size)

    def detect_faces(self, frame):
        return self.detector.get(frame)
