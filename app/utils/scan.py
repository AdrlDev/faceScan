import cv2
import sqlite3
import datetime
import os
import base64
import numpy as np
from .face_utils import face_detector, recognizer, DB_PATH, TRAINER_FILE, init_db, is_scanning_active

init_db()

# Distance thresholds (LBPH: lower distance = higher confidence)
HIGH_CONF_DIST = 45
LOW_CONF_DIST = 70
MAX_DIST = 100

def scan_once(images_base64: list[str]):
    """
    Perform face recognition in API/Render mode only (base64 images).
    Returns structured result with confidence level.
    """
    if not os.path.exists(TRAINER_FILE):
        return {"status": "error", "message": "No enrolled faces found. Please enroll first."}

    recognizer.read(TRAINER_FILE)

    def classify_face(distance):
        if distance < HIGH_CONF_DIST:
            return "ok"
        elif distance < LOW_CONF_DIST:
            return "low_confidence"
        else:
            return "unknown"

    def process_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        roi = gray[y:y + h, x:x + w]
        person_id, distance = recognizer.predict(roi)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT name, id_number FROM people WHERE id=?", (person_id,))
        result = cur.fetchone()
        conn.close()

        if result:
            name, id_number = result
            status = classify_face(distance)
            message = (
                f"Recognized {name} with high confidence"
                if status == "ok"
                else f"Recognized {name} with lower confidence"
                if status == "low_confidence"
                else "Unknown face"
            )

            return {
                "status": status,
                "person_id": person_id if status != "unknown" else None,
                "name": name if status != "unknown" else None,
                "id_number": id_number if status != "unknown" else None,
                "distance": distance,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat()
            }
        else:
            return {
                "status": "unknown",
                "distance": distance,
                "message": "Unknown face",
                "timestamp": datetime.datetime.now().isoformat()
            }

    # --- Cloud/API mode ---
    if not images_base64:
        return {"status": "error", "message": "No images provided for scanning."}

    response = {"status": "unknown", "message": "No face detected"}
    for img_b64 in images_base64:
        if not is_scanning_active():
            return {"status": "canceled", "message": "Scan canceled by user"}

        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            result = process_frame(frame)
            if result:
                return result
        except Exception as e:
            response = {"status": "error", "message": str(e)}
    return response
