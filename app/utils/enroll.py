import os
import cv2
import numpy as np
import base64
import sqlite3
import face_recognition
from .face_utils import CONFIG_DIR, DATASET_DIR, DB_PATH

# Make sure dataset dir exists
os.makedirs(DATASET_DIR, exist_ok=True)

def enroll_face(name: str, id_number: str, images_base64: list[str]):
    if not images_base64:
        return {"success": False, "message": "No images provided"}

    # Connect to DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM people WHERE id_number=?", (id_number,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": f"User {name} already enrolled"}

    face_encodings_list = []
    image_count = 0

    for idx, img_b64 in enumerate(images_base64):
        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if not encodings:
                continue

            for i, enc in enumerate(encodings):
                # Save full RGB face image
                path = os.path.join(DATASET_DIR, f"user.{id_number}.{image_count}.jpg")
                cv2.imwrite(path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
                face_encodings_list.append(enc)
                image_count += 1

        except Exception as e:
            print(f"[ERROR] Processing image {idx}: {e}")
            continue

    if image_count == 0:
        return {"success": False, "message": "No valid faces detected"}

    # Insert user into DB
    cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
    conn.commit()
    conn.close()

    # Optionally, save encodings to disk for faster scanning
    np.save(os.path.join(CONFIG_DIR, f"{id_number}_encodings.npy"), np.array(face_encodings_list))

    return {"success": True, "message": f"User {name} enrolled with {image_count} faces"}
