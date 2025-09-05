from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
from app.utils.enroll import enroll_face
from app.utils.scan import scan_once

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
