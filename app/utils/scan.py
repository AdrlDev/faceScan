import sqlite3
import datetime
import base64
import numpy as np
import face_recognition
import os
import cv2 
from .face_utils import DB_PATH, DATASET_DIR, is_scanning_active

# Threshold for recognition (smaller = stricter)
DIST_THRESHOLD = 0.6

def load_known_faces():
    """
    Load known faces from dataset folder.
    Assumes images are named user.<id>.<num>.jpg
    Returns: dict {person_id: {"name": str, "id_number": str, "encodings": [np.array]}}
    """
    known_faces = {}

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith((".jpg", ".png")):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        person_id = int(parts[1])

        # Fetch user info
        if person_id not in known_faces:
            cur.execute("SELECT name, id_number FROM people WHERE id=?", (person_id,))
            row = cur.fetchone()
            if not row:
                continue
            name, id_number = row
            known_faces[person_id] = {"name": name, "id_number": id_number, "encodings": []}

        # Load face encoding
        img_path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_faces[person_id]["encodings"].append(encs[0])

    conn.close()
    return known_faces

def scan_once(images_base64: list[str]):
    """
    Perform face recognition on base64 images.
    Returns first matched face or "unknown".
    """
    if not images_base64:
        return {"status": "error", "message": "No images provided"}

    known_faces = load_known_faces()
    if not known_faces:
        return {"status": "error", "message": "No enrolled faces found"}

    response = {"status": "unknown", "message": "No face detected"}

    for img_b64 in images_base64:
        if not is_scanning_active():
            return {"status": "canceled", "message": "Scan canceled by user"}

        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                best_match_id = None
                best_distance = 1.0  # start with max distance

                # Compare to each known face
                for person_id, info in known_faces.items():
                    distances = face_recognition.face_distance(info["encodings"], face_encoding)
                    if len(distances) == 0:
                        continue
                    min_dist = np.min(distances)
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_match_id = person_id

                if best_match_id is not None and best_distance <= DIST_THRESHOLD:
                    info = known_faces[best_match_id]
                    confidence = 1.0 - best_distance  # 1 = perfect match
                    if confidence > 0.8:
                        status = "ok"
                    else:
                        status = "low_confidence"

                    response = {
                        "status": status,
                        "person_id": best_match_id,
                        "name": info["name"],
                        "id_number": info["id_number"],
                        "distance": float(best_distance),
                        "message": f"Recognized {info['name']} with confidence {confidence:.2f}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    return response
                else:
                    response = {
                        "status": "unknown",
                        "distance": float(best_distance),
                        "message": "Unknown face",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
        except Exception as e:
            response = {"status": "error", "message": str(e)}

    return response
