# face_utils.py
import cv2, os, sqlite3, datetime, numpy as np

# Base directory = app/utils/
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DB_PATH = os.path.join(BASE_DIR, "data", "faces.db")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
CONFIG_DIR = os.path.join(BASE_DIR, "config")   # ✅ Add this
os.makedirs(CONFIG_DIR, exist_ok=True)          # ✅ Ensure config folder exists

TRAINER_FILE = os.path.join(CONFIG_DIR, "trainer.yml")
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

# global flag to control enrollment
enrollment_active = True
scanning_active = True

def start_scan():
    """Mark scan session as active."""
    global scanning_active
    scanning_active = True

def cancel_scan():
    """Cancel current scan process."""
    global scanning_active
    scanning_active = False
    return {"success": True, "message": "Scan process canceled."}

def is_scanning_active():
    """Check if scan is still active."""
    return scanning_active

def start_enrollment():
    """Mark enrollment session as active."""
    global enrollment_active
    enrollment_active = True

def cancel_enrollment():
    """Cancel current enrollment process."""
    global enrollment_active
    enrollment_active = False
    return {"success": True, "message": "Enrollment process canceled."}

def is_enrollment_active():
    """Check if enrollment is still active."""
    return enrollment_active

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
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
    con.commit()
    con.close()

def enroll(name, id_number, gray_faces, person_id=None):
    """
    Save detected faces (grayscale ROIs) for a person and retrain the model.
    """
    global enrollment_active
    if not enrollment_active:
        return False, "Enrollment was canceled."

    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # If no person_id given, insert into people
    if person_id is None:
        cur.execute("INSERT INTO people (name, id_number) VALUES (?, ?)", (name, id_number))
        person_id = cur.lastrowid
    con.commit()
    con.close()

    count = 0
    for roi in gray_faces:
        if not enrollment_active:  # 👈 stop mid-process if canceled
            return False, "Enrollment was canceled during process."

        path = os.path.join(DATASET_DIR, f"user.{person_id}.{count}.jpg")
        cv2.imwrite(path, roi)
        count += 1

    train_model()
    return True, f"Enrolled {name} with {count} sample(s)."

def train_model():
    paths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
    face_samples, ids = [], []
    for path in paths:
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        person_id = int(path.split(".")[1])
        face_samples.append(gray)
        ids.append(person_id)
    if face_samples:
        recognizer.train(face_samples, np.array(ids))
        recognizer.save(TRAINER_FILE)

def is_user_enrolled(id_number: str) -> bool:
    """Check if user with given ID is already in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM people WHERE id_number = ?", (id_number,))
    exists = cursor.fetchone()[0] > 0
    conn.close()
    return exists

def is_face_already_enrolled(new_face_samples: list[np.ndarray], threshold: float = 70.0):
    """
    Check if a new face matches any existing enrolled faces.
    Returns (True, id_number, confidence) if match found, else (False, None, None).
    """
    if not os.path.exists(TRAINER_FILE):
        return False, None, None  # no training data yet

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for face in new_face_samples:
        try:
            person_id, confidence = recognizer.predict(face)

            if confidence < threshold:
                # check if person_id actually exists
                cursor.execute("SELECT id_number FROM people WHERE id = ?", (person_id,))
                row = cursor.fetchone()
                if row:
                    return True, row[0], confidence  # ✅ real enrolled face found
        except Exception:
            continue

    conn.close()
    return False, None, None  # ✅ new face

def clear_all_faces():
    # ✅ Clear database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM people")  # remove all enrolled users
    conn.commit()
    conn.close()

    # ✅ Delete trainer file
    if os.path.exists(TRAINER_FILE):
        os.remove(TRAINER_FILE)

    # ✅ Clear dataset directory
    if os.path.exists(DATASET_DIR):
        for file in os.listdir(DATASET_DIR):
            file_path = os.path.join(DATASET_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    return {"success": True, "message": "All faces, dataset images, and database entries cleared"}

def delete_face_by_scan(new_face_samples: list[np.ndarray], id_number: str, threshold: float = 90):
    """
    Deletes a face strictly:
    - Only if dataset images and DB record exist.
    - Uses threshold distance for safe deletion.
    """
    if not os.path.exists(TRAINER_FILE):
        return False, "No trained faces available."

    recognizer.read(TRAINER_FILE)

    for face in new_face_samples:
        try:
            person_id, distance = recognizer.predict(face)
            if distance < threshold:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                cursor.execute("SELECT name, id_number FROM people WHERE id=? AND id_number=?", (person_id,id_number,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return False, f"No database record found for person_id {person_id}"

                name, id_number = row

                deleted_count = 0
                for file in os.listdir(DATASET_DIR):
                    if file.startswith(f"user.{person_id}."):
                        os.remove(os.path.join(DATASET_DIR, file))
                        deleted_count += 1

                if deleted_count == 0:
                    conn.close()
                    return False, f"No dataset images found for {name} ({id_number})"

                cursor.execute("DELETE FROM people WHERE id=?", (person_id,))
                conn.commit()
                conn.close()

                train_model()
                return True, f"Deleted {name} ({id_number}) with {deleted_count} image(s)."

        except Exception:
            continue

    return False, "No matching face found."

