import os
import cv2
import numpy as np
import base64
import sqlite3
import face_recognition
from .face_utils import CONFIG_DIR, DATASET_DIR, DB_PATH, is_face_already_enrolled, align_face

# Ensure dataset folder exists
os.makedirs(DATASET_DIR, exist_ok=True)

def enroll_face(name: str, id_number: str, images_base64: list[str]):
    if not images_base64:
        return {"success": False, "message": "No images provided"}

    decoded_faces = []
    for img_b64 in images_base64:
        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            decoded_faces.append(rgb_frame)
        except Exception as e:
            print(f"[ERROR] Decoding image: {e}")
            continue

    if not decoded_faces:
        return {"success": False, "message": "No valid image frames"}

    # --- Improved duplicate check ---
    already_enrolled, matched_id, dist = is_face_already_enrolled(decoded_faces)

    # Only treat as duplicate if it's a *different* user (not same id_number)
    if already_enrolled and matched_id != id_number:
        return {
            "success": False,
            "message": f"Face already enrolled under ID {matched_id} (distance={dist:.3f})"
        }

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM people WHERE id_number=?", (id_number,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": f"User {name} already enrolled"}

    face_encodings_list = []
    image_count = 0

    for idx, rgb_frame in enumerate(decoded_faces):
        try:
            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) != 1:
                print(f"[WARN] Frame {idx}: expected 1 face, found {len(face_locations)}")
                continue

            top, right, bottom, left = face_locations[0]
            face_img = rgb_frame[top:bottom, left:right]

            # Align + resize the cropped face
            aligned_face = align_face(face_img)
            if aligned_face is None:
                print(f"[WARN] Frame {idx}: failed to align face")
                continue

            encodings = face_recognition.face_encodings(aligned_face)
            if not encodings:
                continue

            enc = encodings[0]
            face_encodings_list.append(enc)

            save_path = os.path.join(DATASET_DIR, f"user.{id_number}.{image_count}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
            image_count += 1

        except Exception as e:
            print(f"[ERROR] Processing image {idx}: {e}")
            continue

    if image_count == 0:
        return {"success": False, "message": "No valid faces detected"}

    cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
    conn.commit()
    conn.close()

    # Save encodings for faster recognition
    np.save(os.path.join(CONFIG_DIR, f"{id_number}_encodings.npy"), np.array(face_encodings_list))

    return {"success": True, "message": f"User {name} enrolled with {image_count} faces"}