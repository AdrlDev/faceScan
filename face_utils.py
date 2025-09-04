# face_utils.py
import cv2, os, sqlite3, datetime, numpy as np

DB_PATH = "faces.db"
DATASET_DIR = "dataset"
TRAINER_FILE = "trainer.yml"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
face_detector = cv2.CascadeClassifier(CASCADE_PATH)
recognizer = cv2.face.LBPHFaceRecognizer_create()

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
