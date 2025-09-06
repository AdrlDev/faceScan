import cv2
import tkinter as tk
from PIL import Image, ImageTk  # type: ignore
from .face_utils import face_detector, enroll, init_db, is_user_enrolled, is_face_already_enrolled
import base64
import numpy as np

init_db()

def enroll_face(name: str, id_number: str, images_base64: list[str] = None):
    init_db()

    if is_user_enrolled(id_number):
        return {"success": False, "message": f"User with ID {id_number} is already enrolled"}

    # ========================
    # API MODE
    # ========================
    if images_base64 is not None:
        face_samples_temp = []

        for img_b64 in images_base64:
            try:
                img_data = base64.b64decode(img_b64)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    roi = gray[y:y+h, x:x+w]
                    face_samples_temp.append(roi)
            except Exception as e:
                print(f"[ERROR] Failed to process one image: {e}")
                continue

        if not face_samples_temp:
            return {"success": False, "message": "No valid faces detected in uploaded images"}

        already, existing_id, conf = is_face_already_enrolled(face_samples_temp)
        if already:
            return {"success": False, "message": f"Face already enrolled with ID {existing_id} (confidence {conf:.2f})"}

        success, msg = enroll(name, id_number, face_samples_temp)
        return {"success": success, "message": msg}

    # ========================
    # DESKTOP MODE (webcam)
    # ========================
    cap = cv2.VideoCapture(0)
    count = 0
    face_samples_temp = []
    total_samples = 20
    cancelled = False  # flag to track cancellation

    root = tk.Tk()
    root.title("Enroll Face")
    root.overrideredirect(True)
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.configure(bg="black")

    label = tk.Label(root, bg="black")
    label.place(relx=0.5, rely=0.4, anchor="center")

    progress_label = tk.Label(root, text="Captured: 0/20", font=("Arial", 18), fg="white", bg="black")
    progress_label.place(relx=0.5, rely=0.75, anchor="center")

    instruction_label = tk.Label(
        root,
        text="Scanning will close automatically.\nPress 'C' to close manually (after 100%).",
        font=("Arial", 14),
        fg="lightgray",
        bg="black"
    )
    instruction_label.place(relx=0.5, rely=0.85, anchor="center")

    # ✅ Cancel button
    def cancel_enrollment():
        nonlocal cancelled
        cancelled = True
        face_samples_temp.clear()  # discard all samples
        root.destroy()

    cancel_btn = tk.Button(root, text="Cancel", command=cancel_enrollment, font=("Arial", 14), bg="red", fg="white")
    cancel_btn.place(relx=0.5, rely=0.9, anchor="center")

    def update_frame():
        nonlocal count, face_samples_temp
        if cancelled:
            return  # stop updating frames if cancelled

        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count < total_samples:
                roi = gray[y:y+h, x:x+w]
                face_samples_temp.append(roi)
                count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        progress_label.config(
            text=f"Captured: {count}/{total_samples}",
            fg="lime" if count >= total_samples else "red"
        )

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.thumbnail((screen_width // 2, screen_height // 2))

        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        if count >= total_samples:
            root.after(1000, root.destroy)
        elif not cancelled:
            root.after(10, update_frame)

    def on_key(event):
        if event.char.lower() == "c" and count >= total_samples:
            root.destroy()

    root.bind("<Key>", on_key)
    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

    if cancelled:
        return {"success": False, "message": "Enrollment cancelled by user"}

    if len(face_samples_temp) >= total_samples:
        already, existing_id, conf = is_face_already_enrolled(face_samples_temp)
        if already:
            return {"success": False, "message": f"Face already enrolled with ID {existing_id} (confidence {conf:.2f})"}

        success, msg = enroll(name, id_number, face_samples_temp)
        return {"success": success, "message": msg}
    else:
        return {"success": False, "message": "Face not fully scanned"}
