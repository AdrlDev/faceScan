import base64
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from app.utils.enroll import enroll_face
from app.utils.scan import scan_once
from app.utils.face_utils import clear_all_faces, delete_face_by_scan

app = FastAPI()

# Allow requests from your frontend (use "*" for testing, restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000", "https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EnrollRequest(BaseModel):
    name: str
    id_number: str
    images_base64: list[str] | None = None  # 👈 accept snapshots

class ScanRequest(BaseModel):
    images_base64: list[str] | None = None   # optional, if frontend sends snapshot

@app.post("/api/enroll")
async def api_enroll(req: EnrollRequest):
    """
    Enroll a user by capturing their face through server webcam.
    """
    result = enroll_face(req.name, req.id_number, req.images_base64)
    return result

@app.post("/api/scan")
async def api_scan(req: ScanRequest):
    """
    Scan face.
    - If JSON with images_base64[] → process snapshots.
    - Else → open webcam (local only).
    """
    images = req.images_base64
    result = scan_once(images)
    return result


@app.post("/api/reset")
async def clear_faces_api():
    return clear_all_faces()

@app.post("/api/delete-face")
async def delete_face_api(req: ScanRequest):
    """
    Delete a specific enrolled user's face and data by scanning their face.
    """
    images = req.images_base64
    if not images:
        return {"success": False, "message": "No images provided for scanning."}

    # Convert base64 images to grayscale numpy arrays
    gray_faces = []
    for img_b64 in images:
        img_data = np.frombuffer(base64.b64decode(img_b64), np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_faces.append(gray)

    success, message = delete_face_by_scan(gray_faces)
    return {"success": success, "message": message}