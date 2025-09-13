import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from app.utils.enroll import enroll_face
from app.utils.scan import scan_once
from app.utils.face_utils import clear_all_faces, delete_face_by_scan, cancel_enrollment, start_enrollment, cancel_scan, start_scan

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

class ScanDeleteRequest(BaseModel):
    images_base64: list[str] | None = None   # optional, if frontend sends snapshot
    id_number: str

@app.post("/api/enroll")
async def api_enroll(req: EnrollRequest):
    """
    Enroll a user by capturing their face through server webcam.
    """
    start_enrollment()
    result = enroll_face(req.name, req.id_number, req.images_base64)
    return result

@app.post("/api/scan")
async def api_scan(req: ScanRequest):
    """
    Scan face.
    - If JSON with images_base64[] → process snapshots.
    - Else → open webcam (local only).
    """
    start_scan()
    images = req.images_base64
    result = scan_once(images)
    return result

@app.post("/api/cancel-scan")
async def cancel_scan_api():
    """Cancel the current scanning process."""
    return cancel_scan()

@app.post("/api/cancel-enroll")
async def cancel_enroll_api():
    """
    Cancel current enrollment process.
    """
    return cancel_enrollment()

@app.post("/api/reset")
async def clear_faces_api():
    return clear_all_faces()

@app.post("/api/delete-face")
async def delete_face_api(req: ScanDeleteRequest):
    if not req.images_base64:
        raise HTTPException(status_code=400, detail="No images provided for scanning.")

    gray_faces = []
    for img_b64 in req.images_base64:
        try:
            img_data = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_faces.append(gray)
        except Exception as e:
            # Skip invalid images but log
            print(f"Failed to decode an image: {e}")
            continue

    if not gray_faces:
        return {"success": False, "message": "No valid images could be decoded."}

    try:
        success, message = delete_face_by_scan(gray_faces, req.id_number)
        return {"success": success, "message": message}
    except Exception as e:
        return {"success": False, "message": f"Error during deletion: {e}"}