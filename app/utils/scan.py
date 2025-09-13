import cv2
import sqlite3
import datetime
import os
import base64
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk  # type: ignore
from .face_utils import face_detector, recognizer, DB_PATH, TRAINER_FILE, init_db, is_scanning_active

init_db()


def scan_once(images_base64: list[str] = None):
    """
    Face scan:
    - If images_base64 provided → process those (Render-compatible).
    - Else → fallback to webcam + Tkinter UI (local dev only).
    """

    if not os.path.exists(TRAINER_FILE):
        return {"status": "error", "message": "No enrolled faces found. Please enroll first."}

    recognizer.read(TRAINER_FILE)

    # Distance thresholds
    HIGH_CONF_DIST = 60
    LOW_CONF_DIST = 120

    # ✅ Cloud / API mode
    if images_base64:
        response = {"status": "error", "message": "No face detected"}
        for img_b64 in images_base64:
            if not is_scanning_active():
                return {"status": "canceled", "message": "Scan canceled by user"}

            try:
                img_data = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                if len(faces) == 0:
                    continue

                (x, y, w, h) = faces[0]
                roi = gray[y:y + h, x:x + w]
                id_, distance = recognizer.predict(roi)

                con = sqlite3.connect(DB_PATH)
                cur = con.cursor()
                cur.execute("SELECT name, id_number FROM people WHERE id=?", (id_,))
                result = cur.fetchone()
                con.close()

                if result:
                    name, id_number = result
                    if distance < HIGH_CONF_DIST:
                        return {
                            "status": "ok",
                            "person_id": id_,
                            "name": name,
                            "id_number": id_number,
                            "distance": distance,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    elif distance < LOW_CONF_DIST:
                        return {
                            "status": "low_confidence",
                            "name": name,
                            "id_number": id_number,
                            "distance": distance,
                            "message": "Recognition has lower confidence",
                            "timestamp": datetime.datetime.now().isoformat()
                        }

                # Unknown face
                response = {
                    "status": "unknown",
                    "message": "Unknown face",
                    "distance": distance,
                    "timestamp": datetime.datetime.now().isoformat()
                }

            except Exception as e:
                response = {"status": "error", "message": str(e)}

        return response

    # ✅ Local fallback with webcam + Tkinter
    cap = cv2.VideoCapture(0)
    response = {"status": "ok", "message": "No face detected"}  # default

    # Tkinter window
    root = tk.Tk()
    root.title("Face Recognition")
    root.overrideredirect(True)
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")

    label = tk.Label(root, bg="black")
    label.pack(expand=True)
    result_label = tk.Label(root, text="Looking for face...", font=("Arial", 18), fg="white", bg="black")
    result_label.pack(pady=10)
    instruction_label = tk.Label(root, text="Press Q to close", font=("Arial", 14), fg="gray", bg="black")
    instruction_label.pack(pady=5)

    def update_frame():
        nonlocal response

        if not is_scanning_active():
            root.destroy()
            response = {"status": "canceled", "message": "Scan canceled by user"}
            return

        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            roi = gray[y:y + h, x:x + w]
            id_, distance = recognizer.predict(roi)

            con = sqlite3.connect(DB_PATH)
            cur = con.cursor()
            cur.execute("SELECT name, id_number FROM people WHERE id=?", (id_,))
            result = cur.fetchone()
            con.close()

            if result:
                name, id_number = result
                if distance < HIGH_CONF_DIST:
                    response = {
                        "status": "ok",
                        "person_id": id_,
                        "name": name,
                        "id_number": id_number,
                        "distance": distance,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    result_label.config(text=f"{name} (High confidence)", fg="lime")
                    root.after(1500, root.destroy)
                elif distance < LOW_CONF_DIST:
                    response = {
                        "status": "low_confidence",
                        "name": name,
                        "id_number": id_number,
                        "distance": distance,
                        "message": "Recognition has lower confidence",
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    result_label.config(text=f"{name} (Low confidence)", fg="yellow")
                else:
                    response = {
                        "status": "unknown",
                        "message": "Unknown face",
                        "distance": distance,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    result_label.config(text=f"Unknown", fg="red")

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).resize((640, 480))
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)
        root.after(10, update_frame)

    def on_key(event):
        if event.keysym.lower() == "q":
            root.destroy()

    root.bind("<Key>", on_key)
    update_frame()
    root.mainloop()
    cap.release()
    cv2.destroyAllWindows()
    return response
