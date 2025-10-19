import os
import sqlite3
import cv2
import numpy as np
import base64
from .face_utils import face_detector, DB_PATH, TRAINER_FILE, init_db

init_db()

TOTAL_SAMPLES = 20  # number of face images per user

def enroll_face(name: str, id_number: str, images_base64: list[str]):  # API only
    """
    Enroll a user using API images (base64).
    Saves face images to dataset/ and updates the recognizer trainer.
    """
    if not images_base64:
        return {"success": False, "message": "No images provided"}

    dataset_dir = os.path.join(os.path.dirname(TRAINER_FILE), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    print("[DEBUG] Dataset folder:", dataset_dir)

    # Check if user already exists in DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM people WHERE id_number=?", (id_number,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": f"User {name} already enrolled"}

    # 1️⃣ Decode and detect faces
    face_samples = []
    for img_b64 in images_base64:
        try:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                print("[DEBUG] Failed to decode one image")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_samples.append(gray[y:y+h, x:x+w])

        except Exception as e:
            print("[ERROR] Failed to process one image:", e)
            continue

    if not face_samples:
        return {"success": False, "message": "No valid faces detected in images"}

    # 2️⃣ Assign user ID
    existing_files = [f for f in os.listdir(dataset_dir) if f.startswith("user.")]
    if existing_files:
        user_id = max([int(f.split('.')[1]) for f in existing_files]) + 1
    else:
        user_id = 1
    print("[DEBUG] Assigned user_id:", user_id)

    # 3️⃣ Save face images
    for i, face in enumerate(face_samples):
        path = os.path.join(dataset_dir, f"user.{user_id}.{i}.jpg")
        if cv2.imwrite(path, face):
            print("[DEBUG] Saved face image:", path)
        else:
            print("[ERROR] Failed to save face image:", path)

    # 4️⃣ Update database
    cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
    conn.commit()
    conn.close()
    print("[DEBUG] Database updated for user:", name)

    # 5️⃣ Train recognizer
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
    faces = []
    ids = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("[DEBUG] Skipping invalid image:", path)
            continue
        uid = int(path.split('.')[1])
        faces.append(img)
        ids.append(uid)

    if faces and ids:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.write(TRAINER_FILE)
        print("[DEBUG] Trainer saved at:", TRAINER_FILE)
    else:
        print("[ERROR] No valid faces to train recognizer")
        return {"success": False, "message": "Failed to train recognizer"}

    return {"success": True, "message": f"User {name} enrolled successfully with {len(face_samples)} face images"}