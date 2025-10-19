import os
import base64
import cv2
import numpy as np
import face_recognition
import datetime
import sqlite3
from .face_utils import DATASET_DIR, DB_PATH

DIST_THRESHOLD = 0.6

def load_known_faces():
    known_faces = {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        person_id = int(parts[1])

        if person_id not in known_faces:
            cur.execute("SELECT name, id_number FROM people WHERE id=?", (person_id,))
            row = cur.fetchone()
            if not row:
                continue
            name, id_number = row
            known_faces[person_id] = {"name": name, "id_number": id_number, "encodings": []}

        img_path = os.path.join(DATASET_DIR, filename)
        try:
            img = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(img)
            if not encs:
                print(f"[WARN] No face found in {filename}, skipping.")
                continue
            known_faces[person_id]["encodings"].append(encs[0])
        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")
            continue

    conn.close()
    print("[DEBUG] Loaded known faces:", len(known_faces))
    return known_faces

def scan_once(images_base64: list[str]):
    if not images_base64:
        return {"status": "error", "message": "No images provided"}

    known_faces = load_known_faces()
    if not known_faces:
        return {"status": "error", "message": "No enrolled faces found"}

    for img_b64 in images_base64:
        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                best_match_id = None
                best_distance = 1.0
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
                    confidence = 1.0 - best_distance
                    status = "ok" if confidence > 0.8 else "low_confidence"
                    return {
                        "status": status,
                        "person_id": best_match_id,
                        "name": info["name"],
                        "id_number": info["id_number"],
                        "distance": float(best_distance),
                        "message": f"Recognized {info['name']} with confidence {confidence:.2f}",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return {"status": "unknown", "message": "No face recognized"}
