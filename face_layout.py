import cv2
import numpy as np
import os

MASKS_PATH = "assets/masks"

def load_mask(angle, frame_shape):
    """Load and preprocess mask with dynamic sizing"""
    path = os.path.join(MASKS_PATH, f"{angle}.png")
    mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {path}")
    
    # Calculate proportional scaling
    h, w = frame_shape[:2]
    mask_h, mask_w = mask.shape[:2]
    scale = min(w/mask_w, h/mask_h) * 0.8  # 80% of max possible scale
    
    # Resize maintaining aspect ratio
    return cv2.resize(mask, (int(mask_w * scale), int(mask_h * scale)))

def overlay_mask(frame, mask, face_bbox=None):
    """Intelligently position mask based on face detection"""
    if mask.shape[2] != 4:
        return frame, (0, 0, 0, 0)
    
    h, w = frame.shape[:2]
    mh, mw = mask.shape[:2]
    
    # Default center position
    offset_x = int(w/2 - mw/2)
    offset_y = int(h/2 - mh/2)
    
    # Adjust position if face is detected
    if face_bbox is not None and len(face_bbox) == 4:
        x1, y1, x2, y2 = face_bbox
        face_w = x2 - x1
        offset_x = int(x1 + face_w/2 - mw/2)
        offset_y = int(y1 - mh * 0.3)  # Position above face
    
    # Ensure mask stays within frame bounds
    offset_x = max(0, min(offset_x, w - mw))
    offset_y = max(0, min(offset_y, h - mh))
    
    # Alpha blending
    alpha = mask[:, :, 3] / 255.0
    for c in range(3):
        frame[offset_y:offset_y+mh, offset_x:offset_x+mw, c] = (
            alpha * mask[:, :, c] + 
            (1 - alpha) * frame[offset_y:offset_y+mh, offset_x:offset_x+mw, c]
        )
    return frame, (offset_x, offset_y, mw, mh)

def check_alignment(face_bbox, mask_rect):
    """Precise face-mask alignment checking"""
    if face_bbox is None or len(face_bbox) != 4:
        return False
        
    fx1, fy1, fx2, fy2 = face_bbox
    mx, my, mw, mh = mask_rect
    
    face_cx = (fx1 + fx2) // 2
    face_cy = (fy1 + fy2) // 2
    mask_center_x = mx + mw//2
    mask_center_y = my + mh//2
    
    # Allow 20% margin around mask center
    return (abs(face_cx - mask_center_x) < mw * 0.2 and 
            abs(face_cy - mask_center_y) < mh * 0.2)