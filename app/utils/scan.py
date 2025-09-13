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

# Distance thresholds (LBPH: lower distance = higher confidence)
HIGH_CONF_DIST = 50      # very confident
LOW_CONF_DIST = 90       # somewhat confident
MAX_DIST = 120           # above this = unknown

def scan_once(images_base64: list[str] = None):
    """
    Perform face recognition:
    - images_base64 → API/Render mode
    - None → fallback to webcam + Tkinter
    Returns structured result with confidence level.
    """
    if not os.path.exists(TRAINER_FILE):
        return {"status": "error", "message": "No enrolled faces found. Please enroll first."}

    recognizer.read(TRAINER_FILE)

    def classify_face(distance):
        if distance < HIGH_CONF_DIST:
            return "high_confidence"
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
            if status == "high_confidence":
                message = f"Recognized {name} with high confidence"
            elif status == "low_confidence":
                message = f"Recognized {name} with lower confidence"
            else:
                message = "Unknown face"

            return {
                "status": status,
                "person_id": person_id,
                "name": name,
                "id_number": id_number,
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
    if images_base64:
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

    # --- Local webcam mode ---
    cap = cv2.VideoCapture(0)
    response = {"status": "unknown", "message": "No face detected"}

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

        result = process_frame(frame)
        if result:
            response = result
            # Update Tkinter label colors
            if response["status"] == "high_confidence":
                result_label.config(text=response["message"], fg="lime")
                root.after(1500, root.destroy)
            elif response["status"] == "low_confidence":
                result_label.config(text=response["message"], fg="yellow")
            else:
                result_label.config(text=response["message"], fg="red")

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