import cv2
import tkinter as tk
from PIL import Image, ImageTk
from face_utils import face_detector, enroll, init_db

init_db()

def enroll_face(name: str, id_number: str):
    """
    Capture face samples from webcam and enroll user.
    Auto close when scan is completed.
    Camera feed is centered on screen with close instructions.
    """
    init_db()

    cap = cv2.VideoCapture(0)
    count = 0
    face_samples = []
    total_samples = 20  # number of face samples required

    print("[INFO] Starting face capture. Look at the camera...")

    # ✅ Tkinter root window (borderless fullscreen)
    root = tk.Tk()
    root.title("Enroll Face")
    root.overrideredirect(True)  # remove title bar
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    root.configure(bg="black")

    # ✅ Centered camera feed
    label = tk.Label(root, bg="black")
    label.place(relx=0.5, rely=0.4, anchor="center")  # little higher to make space for text

    # ✅ Progress label below video
    progress_label = tk.Label(root, text="Captured: 0/20", font=("Arial", 18), fg="white", bg="black")
    progress_label.place(relx=0.5, rely=0.75, anchor="center")

    # ✅ Instruction label below progress
    instruction_label = tk.Label(
        root,
        text="Scanning will close automatically.\nPress 'C' to close manually (after 100%).",
        font=("Arial", 14),
        fg="lightgray",
        bg="black"
    )
    instruction_label.place(relx=0.5, rely=0.85, anchor="center")

    def update_frame():
        nonlocal count, face_samples

        ret, frame = cap.read()
        if not ret:
            root.after(10, update_frame)
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if count < total_samples:
                roi = gray[y:y+h, x:x+w]
                face_samples.append(roi)
                count += 1
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        # ✅ Update progress label instead of drawing text on frame
        progress_label.config(
            text=f"Captured: {count}/{total_samples}",
            fg="lime" if count >= total_samples else "red"
        )

        # Convert OpenCV image to Tkinter
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        # Resize to fit neatly in center
        max_w, max_h = screen_width // 2, screen_height // 2
        img.thumbnail((max_w, max_h))

        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

        if count >= total_samples:
            root.after(1000, root.destroy)
        else:
            root.after(10, update_frame)

    def on_key(event):
        if event.char.lower() == "c" and count >= total_samples:
            root.destroy()

    root.bind("<Key>", on_key)
    update_frame()
    root.mainloop()

    cap.release()
    cv2.destroyAllWindows()

    if len(face_samples) >= total_samples:
        success, msg = enroll(name, id_number, face_samples)
        return {"success": success, "message": msg}
    else:
        return {"success": False, "message": "Face not fully scanned"}
