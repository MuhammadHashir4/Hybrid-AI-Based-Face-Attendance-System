import os
import requests
import cv2

# CONFIGURATION
FACEPP_COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"
FACEPP_API_KEY = "oyFna8gc-_5Qo5wQ44QPlpAtbLE3X5dg"
FACEPP_API_SECRET = "pIXdADRxW7dXUc2xcNMypsG6xmBMjjLK"
REFERENCE_DIR = "reference_images"
SIMILARITY_THRESHOLD = 78  # Adjust between 75–85 depending on security
def is_api_image_valid(img):
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    return w >= 48 and h >= 48 and (w * h) >= 4096

def facepp_match_with_registered_images(live_face_img, full_frame=None):
    # Check for valid image to send
    if not is_api_image_valid(live_face_img):
        if full_frame is not None and is_api_image_valid(full_frame):
            print("[WARN] Cropped face too small for API, using full frame.")
            img_to_use = full_frame
        else:
            print("[ERROR] No valid image to send to Face++ (both crops too small).")
            return None
    else:
        img_to_use = live_face_img

    # ... existing code for encoding, API requests, etc ...

def facepp_match_with_registered_images(live_face_img, full_frame=None):
    """
    Takes a cropped face or the full frame, compares against all registered images using Face++,
    returns (reg_no, name) of the best match if above threshold, else None.
    """
    # Use fallback to full_frame if crop is bad
    if live_face_img is None or live_face_img.size == 0:
        if full_frame is not None and full_frame.size > 0:
            print("[WARN] Face crop empty, using full frame for API.")
            img_to_use = full_frame
        else:
            print("[ERROR] Both crop and frame are empty, skipping API fallback.")
            return None
    else:
        img_to_use = live_face_img

    try:
        _, live_encoded = cv2.imencode(".jpg", img_to_use)
        live_bytes = live_encoded.tobytes()
    except Exception as e:
        print(f"[ERROR] Encoding live face: {e}")
        return None

    best_match = None
    best_score = 0.0

    # Iterate through all users/folders
    for folder in os.listdir(REFERENCE_DIR):
        folder_path = os.path.join(REFERENCE_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            reg_no, name = folder.split("_", 1)
        except Exception:
            print(f"[WARN] Unexpected folder name format: {folder}")
            continue

        for img_file in os.listdir(folder_path):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(folder_path, img_file)
            with open(image_path, "rb") as f:
                reference_bytes = f.read()

            # Perform Face++ compare
            try:
                response = requests.post(
                    FACEPP_COMPARE_URL,
                    data={
                        "api_key": FACEPP_API_KEY,
                        "api_secret": FACEPP_API_SECRET,
                    },
                    files={
                        "image_file1": ("live.jpg", live_bytes, "image/jpeg"),
                        "image_file2": ("reference.jpg", reference_bytes, "image/jpeg"),
                    },
                    timeout=8
                )
            except Exception as e:
                print(f"[ERROR] Face++ API call error: {e}")
                continue

            if response.status_code != 200:
                print(f"[ERROR] Face++ API HTTP {response.status_code}: {response.text}")
                continue

            try:
                result = response.json()
                confidence = float(result.get("confidence", 0))
            except Exception as e:
                print(f"[ERROR] Could not parse Face++ response: {e}")
                continue

            print(f"[DEBUG] Compared with {img_file} ({reg_no} - {name}): confidence = {confidence}")

            if confidence > best_score:
                best_score = confidence
                best_match = (reg_no, name)

    if best_match and best_score >= SIMILARITY_THRESHOLD:
        print(f"[INFO] Face++ matched with {best_match[1]} ({best_match[0]}) — score {best_score}")
        return best_match
    else:
        print(f"[INFO] No suitable Face++ match found (best score: {best_score})")
        return None