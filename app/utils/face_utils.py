# face_utils.py
import os
import cv2
import sqlite3
import numpy as np
import face_recognition
import datetime

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(CONFIG_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, "faces.db")

# Global flags
enrollment_active = True
scanning_active = True


# ------------------- Session control ------------------- #
def start_scan():
    global scanning_active
    scanning_active = True

def cancel_scan():
    global scanning_active
    scanning_active = False
    return {"success": True, "message": "Scan process canceled."}

def is_scanning_active():
    return scanning_active

def start_enrollment():
    global enrollment_active
    enrollment_active = True

def cancel_enrollment():
    global enrollment_active
    enrollment_active = False
    return {"success": True, "message": "Enrollment process canceled."}

def is_enrollment_active():
    return enrollment_active


# ------------------- Database ------------------- #
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS people (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        id_number TEXT NOT NULL
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER,
        name TEXT,
        id_number TEXT,
        action TEXT,
        purpose TEXT,
        timestamp TEXT
    )""")
    conn.commit()
    conn.close()


def is_user_enrolled(id_number: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM people WHERE id_number=?", (id_number,))
    exists = cur.fetchone()[0] > 0
    conn.close()
    return exists


# ------------------- Enrollment ------------------- #
def enroll(name: str, id_number: str, rgb_faces: list[np.ndarray]):
    """
    Enroll faces using face_recognition.
    rgb_faces: list of RGB images (numpy arrays) of faces
    """
    global enrollment_active
    if not enrollment_active:
        return False, "Enrollment canceled."

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Insert into DB if not exists
    if not is_user_enrolled(id_number):
        cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
        person_id = cur.lastrowid
        conn.commit()
    else:
        cur.execute("SELECT id FROM people WHERE id_number=?", (id_number,))
        person_id = cur.fetchone()[0]

    # Save face images in dataset
    count = 0
    for img in rgb_faces:
        if not enrollment_active:
            return False, "Enrollment canceled mid-process."
        path = os.path.join(DATASET_DIR, f"user.{id_number}.{count}.jpg")
        rgb_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, rgb_bgr)
        count += 1

    conn.close()
    return True, f"Enrolled {name} with {count} sample(s)."


# ------------------- Face Checking ------------------- #
def is_face_already_enrolled(new_faces: list[np.ndarray], dist_threshold: float = 0.6):
    """
    Check if a face is already enrolled using face_recognition.
    Returns (True, id_number, distance) if match found
    """
    known_faces = {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        id_number = parts[1]
        if id_number not in known_faces:
            cur.execute("SELECT name, id_number FROM people WHERE id_number=?", (id_number,))
            row = cur.fetchone()
            if not row:
                continue
            name, id_num = row
            known_faces[id_number] = {"name": name, "encodings": []}

        img_path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_faces[id_number]["encodings"].append(encs[0])
    conn.close()

    # Compare new faces to known faces
    for face_img in new_faces:
        encs = face_recognition.face_encodings(face_img)
        if not encs:
            continue
        face_encoding = encs[0]

        for id_number, info in known_faces.items():
            if not info["encodings"]:
                continue
            distances = face_recognition.face_distance(info["encodings"], face_encoding)
            min_dist = np.min(distances)
            if min_dist <= dist_threshold:
                return True, id_number, min_dist
    return False, None, None


# ------------------- Deletion ------------------- #
def delete_face_by_id(id_number: str):
    """
    Deletes all faces and DB entry for a given id_number
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, name FROM people WHERE id_number=?", (id_number,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return False, "ID not found in database."
    person_id, name = row

    # Delete dataset images
    deleted_count = 0
    for file in os.listdir(DATASET_DIR):
        if file.startswith(f"user.{id_number}."):
            os.remove(os.path.join(DATASET_DIR, file))
            deleted_count += 1

    # Delete DB record
    cur.execute("DELETE FROM people WHERE id_number=?", (id_number,))
    conn.commit()
    conn.close()

    return True, f"Deleted {name} ({id_number}) with {deleted_count} images."


# ------------------- Clear All ------------------- #
def clear_all_faces():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM people")
    conn.commit()
    conn.close()

    for file in os.listdir(DATASET_DIR):
        os.remove(os.path.join(DATASET_DIR, file))

    return {"success": True, "message": "All faces, dataset images, and database entries cleared."}
