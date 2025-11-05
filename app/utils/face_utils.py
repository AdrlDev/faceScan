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

def start_enrollment():
    global enrollment_active
    enrollment_active = True

def cancel_enrollment():
    global enrollment_active
    enrollment_active = False
    return {"success": True, "message": "Enrollment process canceled."}

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

def load_known_faces():
    """Load all enrolled face encodings from dataset directory."""
    known_faces = {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for filename in os.listdir(DATASET_DIR):
        if not filename.lower().endswith(".jpg"):
            continue
        parts = filename.split(".")
        if len(parts) < 3:
            continue

        person_id = parts[1]
        if person_id not in known_faces:
            cur.execute("SELECT name, id_number FROM people WHERE id_number=?", (person_id,))
            row = cur.fetchone()
            if not row:
                continue
            name, id_number = row
            known_faces[person_id] = {"name": name, "id_number": id_number, "encodings": []}

        img_path = os.path.join(DATASET_DIR, filename)
        img = face_recognition.load_image_file(img_path)
        encs = face_recognition.face_encodings(img)
        if encs:
            known_faces[person_id]["encodings"].append(encs[0])

    conn.close()
    print("[DEBUG] Loaded known faces:", len(known_faces))
    return known_faces

# ------------------- Face Checking ------------------- #
def is_face_already_enrolled(face_encodings_list: list, current_id: str = None) -> tuple[bool, str, float]: # type: ignore
    """
    Check if face encodings match any existing enrolled users
    Returns: (is_enrolled, matched_id, best_distance)
    """
    THRESHOLD = 0.45  # stricter distance threshold to avoid false positives
    MIN_MATCHES = 2   # require at least 2 frames to match same existing ID

    matched_id = None
    best_dist = None
    matches = {}
    
    for fn in os.listdir(CONFIG_DIR):
        if not fn.endswith("_encodings.npy"):
            continue
        existing_id = fn.replace("_encodings.npy", "")
        try:
            existing = np.load(os.path.join(CONFIG_DIR, fn), allow_pickle=True)
        except Exception as e:
            print(f"[WARN] Could not load encodings {fn}: {e}")
            continue
            
        for new_enc in face_encodings_list:
            if existing.size == 0:
                continue
            dists = face_recognition.face_distance(existing, new_enc)
            if dists.size == 0:
                continue
            min_d = float(np.min(dists))
            if min_d < THRESHOLD:
                matches[existing_id] = matches.get(existing_id, 0) + 1
                if best_dist is None or min_d < best_dist:
                    best_dist = min_d

    # decide duplicate if any existing id has enough matching frames
    for eid, cnt in matches.items():
        if cnt >= MIN_MATCHES and eid != current_id:
            matched_id = eid
            break

    return (matched_id is not None, matched_id, best_dist or float('inf')) # type: ignore

# ------------------- Deletion ------------------- #
def delete_face_by_id(id_number: str):
    """
    Delete all enrolled face images, encoding files, and DB entry for a given id_number.
    """
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT name FROM people WHERE id_number=?", (id_number,))
    row = cur.fetchone()

    if not row:
        conn.close()
        return False, f"ID {id_number} not found in database."

    name = row[0]
    deleted_count = 0

    # --- Delete dataset images ---
    for file in os.listdir(DATASET_DIR):
        if file.startswith(f"user.{id_number}."):
            file_path = os.path.join(DATASET_DIR, file)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                logger.warning(f"Failed to delete image {file_path}: {e}")

    # --- Delete encoding file ---
    npy_path = os.path.join(CONFIG_DIR, f"{id_number}_encodings.npy")
    if os.path.exists(npy_path):
        try:
            os.remove(npy_path)
            logger.info(f"Deleted encoding file: {npy_path}")
        except Exception as e:
            logger.warning(f"Failed to delete encoding file {npy_path}: {e}")

    # --- Delete DB entry ---
    cur.execute("DELETE FROM people WHERE id_number=?", (id_number,))
    conn.commit()
    conn.close()

    logger.info(f"Deleted {name} ({id_number}) with {deleted_count} image(s).")
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
    Load stored face encodings for a given ID.
    Tries .npy cache first, then rebuilds from dataset images if needed.
    Returns a list of numpy arrays (encodings) or an empty list.
    """
    encodings = []
    npy_path = os.path.join(CONFIG_DIR, f"{id_number}_encodings.npy")

    # 1️⃣ Try loading from .npy cache for speed
    if os.path.exists(npy_path):
        try:
            encodings = np.load(npy_path, allow_pickle=True).tolist()
            if encodings:
                logger.info(f"Loaded {len(encodings)} encodings from {npy_path}")
                return encodings
            else:
                logger.warning(f"Encoding file {npy_path} is empty.")
        except Exception as e:
            logger.warning(f"Failed to load encodings from {npy_path}: {e}")

    # 2️⃣ Fallback — Rebuild from dataset images
    logger.info(f"Rebuilding encodings for ID {id_number} from dataset...")
    for file in os.listdir(DATASET_DIR):
        if file.startswith(f"user.{id_number}.") and file.lower().endswith(".jpg"):
            img_path = os.path.join(DATASET_DIR, file)
            try:
                img = face_recognition.load_image_file(img_path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    encodings.append(encs[0])
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")

    # 3️⃣ Cache rebuilt encodings for future scans
    if encodings:
        try:
            np.save(npy_path, np.array(encodings, dtype=object))
            logger.info(f"Cached {len(encodings)} new encodings to {npy_path}")
        except Exception as e:
            logger.warning(f"Failed to save encodings to {npy_path}: {e}")

    logger.info(f"Loaded {len(encodings)} total encodings for ID {id_number}")
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