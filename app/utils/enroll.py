import cv2
import os
import sqlite3
import numpy as np
from PIL import Image
from .face_utils import face_detector, DB_PATH, TRAINER_FILE, init_db

init_db()

TOTAL_SAMPLES = 20  # number of face images per user

def enroll_face(name: str, id_number: str, images_base64: list[str] = None): # type: ignore
    """
    Enroll a user into the system.
    - Saves face images to dataset/
    - Updates trainer.yml automatically
    """
    dataset_dir = os.path.join(os.path.dirname(TRAINER_FILE), "dataset")
    os.makedirs(dataset_dir, exist_ok=True)

    # Check if user already exists in DB
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM people WHERE id_number=?", (id_number,))
    if cur.fetchone():
        conn.close()
        return {"success": False, "message": f"User {name} already enrolled"}
    
    # 1️⃣ Collect face images
    face_samples = []

    if images_base64:
        # API mode
        import base64
        for img_b64 in images_base64:
            img_data = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                face_samples.append(gray[y:y+h, x:x+w])
    else:
        # Webcam mode
        import tkinter as tk
        from PIL import ImageTk

        cap = cv2.VideoCapture(0)
        count = 0

        root = tk.Tk()
        root.title("Enroll Face")
        root.geometry("640x480")
        label = tk.Label(root)
        label.pack()

        def update_frame():
            nonlocal count
            ret, frame = cap.read()
            if not ret:
                root.after(10, update_frame)
                return

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                if count < TOTAL_SAMPLES:
                    face_samples.append(gray[y:y+h, x:x+w])
                    count += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)

            if count >= TOTAL_SAMPLES:
                root.destroy()
            else:
                root.after(10, update_frame)

        update_frame()
        root.mainloop()
        cap.release()
        cv2.destroyAllWindows()

    if not face_samples:
        return {"success": False, "message": "No valid faces detected"}

    # 2️⃣ Save images to dataset
    user_id = max([int(f.split('.')[1]) for f in os.listdir(dataset_dir) if f.startswith("user.")]+[0]) + 1
    for i, face in enumerate(face_samples):
        cv2.imwrite(os.path.join(dataset_dir, f"user.{user_id}.{i}.jpg"), face)

    # 3️⃣ Update database
    cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
    conn.commit()
    conn.close()

    # 4️⃣ Train recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    image_paths = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg")]
    faces = []
    ids = []
    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        uid = int(path.split('.')[1])
        faces.append(img)
        ids.append(uid)
    recognizer.train(faces, np.array(ids))
    recognizer.write(TRAINER_FILE)

    return {"success": True, "message": f"User {name} enrolled successfully with {len(face_samples)} face images"}
