import os
import base64
import cv2
import numpy as np
import face_recognition
import datetime
from datetime import timezone, timedelta
import sqlite3
from .face_utils import load_known_faces, align_face

DIST_THRESHOLD = 0.5
PH_TZ = timezone(timedelta(hours=8))  # Philippine Timezone


def scan_once(images_base64: list[str]):
    """
    Perform face recognition on base64 images with face alignment.
    Returns first matched face or 'unknown'.
    """
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
            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)

            if not face_locations:
                continue

            for (top, right, bottom, left) in face_locations:
                face_img = rgb_frame[top:bottom, left:right]
                aligned_face = align_face(face_img)
                if aligned_face is None:
                    continue

                face_encodings = face_recognition.face_encodings(aligned_face)
                if not face_encodings:
                    continue

                face_encoding = face_encodings[0]
                best_match_id = None
                best_distance = 1.0

                # Compare with all known faces
                for person_id, info in known_faces.items():
                    if not info["encodings"]:
                        continue
                    distances = face_recognition.face_distance(info["encodings"], face_encoding)
                    min_dist = np.min(distances)
                    if min_dist < best_distance:
                        best_distance = min_dist
                        best_match_id = person_id

                if best_match_id is not None and best_distance <= DIST_THRESHOLD:
                    info = known_faces[best_match_id]
                    confidence = 1.0 - best_distance
                    status = "ok" if confidence > 0.6 else "low_confidence"

                    ph_time = datetime.datetime.now(PH_TZ).isoformat()

                    return {
                        "status": status,
                        "person_id": best_match_id,
                        "name": info["name"],
                        "id_number": info["id_number"],
                        "distance": float(best_distance),
                        "confidence": float(confidence),
                        "message": f"Recognized {info['name']} with confidence {confidence:.2f}",
                        "timestamp": ph_time
                    }

        except Exception as e:
            print("[ERROR]", str(e))
            return {"status": "error", "message": str(e)}

    return {"status": "unknown", "message": "No face recognized"}