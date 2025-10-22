# face_utils.py
import os
import cv2
import sqlite3
import numpy as np
import face_recognition
import logging
import datetime

# ------------------- Logging ------------------- #
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# ------------------- Directories ------------------- #
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATASET_DIR = os.path.join(CONFIG_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

DB_PATH = os.path.join(BASE_DIR, "data", "faces.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# ------------------- Global Flags ------------------- #
enrollment_active = True
scanning_active = True

# ------------------- Session Control ------------------- #
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
    """Create tables if not exist"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            id_number TEXT NOT NULL UNIQUE
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            name TEXT,
            id_number TEXT,
            action TEXT,
            purpose TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

# Ensure DB exists on import
init_db()

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

    if not is_user_enrolled(id_number):
        cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
        conn.commit()
    else:
        logger.info(f"User {id_number} already enrolled")

    # Save face images
    count = 0
    for img in rgb_faces:
        if not enrollment_active:
            return False, "Enrollment canceled mid-process."
        path = os.path.join(DATASET_DIR, f"user.{id_number}.{count}.jpg")
        rgb_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, rgb_bgr)
        count += 1

    conn.close()
    logger.info(f"Enrolled {name} with {count} sample(s).")
    return True, f"Enrolled {name} with {count} sample(s)."

# ------------------- Face Encoding ------------------- #
def get_face_encoding(img):
    """Return the first face encoding, or None if fails"""
    try:
        encs = face_recognition.face_encodings(img)
        if encs:
            return encs[0]
    except Exception as e:
        logger.error(f"Face encoding failed: {e}")
    return None

# ------------------- Face Checking ------------------- #
def is_face_already_enrolled(decoded_faces, threshold=0.45):
    """
    Checks if any of the given decoded faces match an already enrolled user's face.
    Returns (True, matched_id, distance) if duplicate is found.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id_number FROM people")
    ids = [row[0] for row in cur.fetchall()]
    conn.close()

    # Go through all registered users
    for existing_id in ids:
        enc_path = os.path.join(CONFIG_DIR, f"{existing_id}_encodings.npy")
        if not os.path.exists(enc_path):
            continue

        try:
            known_encodings = np.load(enc_path)
        except Exception as e:
            print(f"[WARN] Skipping {existing_id}: failed to load encodings ({e})")
            continue

        # Compare with each provided face
        for rgb in decoded_faces:
            new_encs = face_recognition.face_encodings(rgb)
            if not new_encs:
                continue
            new_enc = new_encs[0]

            distances = face_recognition.face_distance(known_encodings, new_enc)
            min_dist = np.min(distances)

            if min_dist < threshold:
                # Duplicate found
                return True, existing_id, float(min_dist)

    return False, None, 1.0

# ------------------- Deletion ------------------- #
def delete_face_by_id(id_number: str):
    """Delete all enrolled face images, encodings, and DB entry for the given id_number."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM people WHERE id_number=?", (id_number,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return False, f"ID {id_number} not found in database."

    name = row[0]
    deleted_count = 0

    # --- Delete all dataset face images ---
    for file in os.listdir(DATASET_DIR):
        # Normalize both to string for safety
        if file.startswith(f"user.{str(id_number)}."):
            file_path = os.path.join(DATASET_DIR, file)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")

    # --- Delete .npy encodings (if exist) ---
    npy_path = os.path.join(CONFIG_DIR, f"{id_number}_encodings.npy")
    if os.path.exists(npy_path):
        try:
            os.remove(npy_path)
            logger.info(f"Deleted encoding file: {npy_path}")
        except Exception as e:
            logger.warning(f"Failed to delete encoding file {npy_path}: {e}")

    # --- Remove from database ---
    cur.execute("DELETE FROM people WHERE id_number=?", (id_number,))
    conn.commit()
    conn.close()

    logger.info(f"Deleted {name} ({id_number}) with {deleted_count} images.")
    return True, f"Deleted {name} ({id_number}) with {deleted_count} image(s)."

# ------------------- Clear All ------------------- #
def clear_all_faces():
    """
    Deletes all enrolled users, dataset images, and face encoding files.
    Keeps the database schema intact but clears its data.
    """
    # 1️⃣ Clear database
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM people")
    conn.commit()
    conn.close()

    # 2️⃣ Remove all dataset images (user.ID.jpg)
    dataset_deleted = 0
    for file in os.listdir(DATASET_DIR):
        file_path = os.path.join(DATASET_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
            dataset_deleted += 1

    # 3️⃣ Remove all saved encodings (*.npy)
    encoding_deleted = 0
    for file in os.listdir(CONFIG_DIR):
        if file.endswith("_encodings.npy"):
            os.remove(os.path.join(CONFIG_DIR, file))
            encoding_deleted += 1

    logger.info(f"Cleared all faces: {dataset_deleted} images, {encoding_deleted} encodings, and all DB entries.")
    return {
        "success": True,
        "message": f"Cleared all faces — {dataset_deleted} images, {encoding_deleted} encodings, and all DB entries."
    }

def get_stored_face_encoding(id_number: str):
    """
    Load stored face encodings for a specific ID.
    Automatically normalizes encodings for consistent comparison.
    """
    encodings = []

    # 1️⃣ Check if pre-saved .npy encodings exist
    npy_path = os.path.join("config", f"{id_number}_encodings.npy")
    if os.path.exists(npy_path):
        try:
            encs = np.load(npy_path, allow_pickle=True)
            # Normalize embeddings (unit length)
            for enc in encs:
                norm = np.linalg.norm(enc)
                if norm > 0:
                    encodings.append(enc / norm)
        except Exception as e:
            print(f"[ERROR] Loading encodings for {id_number}: {e}")
        return encodings

    # 2️⃣ Fallback: reconstruct from dataset images
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM people WHERE id_number=?", (id_number,))
    if not cur.fetchone():
        conn.close()
        return []
    conn.close()

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue
        file_id = parts[1]
        if file_id != id_number:
            continue

        img_path = os.path.join(DATASET_DIR, filename)
        try:
            img = face_recognition.load_image_file(img_path)
            encs = face_recognition.face_encodings(img)
            if encs:
                enc = encs[0]
                norm = np.linalg.norm(enc)
                if norm > 0:
                    enc = enc / norm
                encodings.append(enc)
        except Exception as e:
            print(f"[ERROR] Encoding {filename}: {e}")
            continue

    return encodings

def align_face(face_img: np.ndarray):
    """
    Align the face using eye landmarks.
    Returns aligned + resized (150x150) face, or None if alignment fails.
    """
    landmarks = face_recognition.face_landmarks(face_img)
    if not landmarks:
        return None

    left_eye = np.mean(landmarks[0]['left_eye'], axis=0)
    right_eye = np.mean(landmarks[0]['right_eye'], axis=0)

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2,
                   (left_eye[1] + right_eye[1]) / 2)

    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1)
    aligned = cv2.warpAffine(face_img, M, (face_img.shape[1], face_img.shape[0]),
                             flags=cv2.INTER_CUBIC)

    # Crop and resize after alignment
    face_locations = face_recognition.face_locations(aligned)
    if not face_locations:
        return None

    top, right, bottom, left = face_locations[0]
    aligned_cropped = aligned[top:bottom, left:right]
    aligned_resized = cv2.resize(aligned_cropped, (150, 150))
    return aligned_resized