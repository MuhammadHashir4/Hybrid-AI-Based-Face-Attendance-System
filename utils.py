import logging
import os

# --- Logger Setup ---
def setup_logger(name: str, log_file: str) -> logging.Logger:
    os.makedirs("logs", exist_ok=True)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(f"logs/{log_file}")
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    if not logger.hasHandlers():  # Prevent duplicate logs
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    return logger

# --- Enhanced Angle Detection ---
def detect_face_angle(face, previous_angle=None, tolerance=5):
    """
    Determine the head pose angle based on yaw and pitch.
    More user-friendly: soft detection and fallback to last stable angle.
    """
    yaw, pitch, _ = face.pose

    # Helper to handle slight jitter
    def is_close(val1, val2): return abs(val1 - val2) <= tolerance

    if -20 <= yaw <= 20 and -20 <= pitch <= 20:
        return 'front'
    elif yaw > 15:
        return 'left'
    elif yaw < -15:
        return 'right'
    elif pitch > 15:
        return 'up'
    elif pitch < -15:
        return 'down'

    # Fallback: stick with last known angle if it's still close
    return previous_angle
