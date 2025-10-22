import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import cv2
import numpy as np
import face_recognition
from app.utils.enroll import enroll_face
from app.utils.scan import scan_once
from app.utils.face_utils import clear_all_faces, get_stored_face_encoding, logger, delete_face_by_id, cancel_enrollment, start_enrollment, cancel_scan, start_scan, align_face

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
    result = enroll_face(req.name, req.id_number, req.images_base64) # type: ignore
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
    result = scan_once(images) # type: ignore
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
    """Delete a user's enrolled face if it matches the uploaded image(s)."""
    if not req.images_base64:
        raise HTTPException(status_code=400, detail="No images provided for scanning.")

    # Step 1️⃣: Load stored encodings for the provided ID
    stored_encodings = get_stored_face_encoding(req.id_number)
    if not stored_encodings:
        raise HTTPException(status_code=404, detail=f"No stored face found for ID {req.id_number}.")

    # Step 2️⃣: Match uploaded face(s) against stored encodings
    match_found = False
    for img_b64 in req.images_base64:
        try:
            img_data = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img is None:
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_img)
            if not face_locations:
                continue

            for (top, right, bottom, left) in face_locations:
                face_img = rgb_img[top:bottom, left:right]
                aligned_face = align_face(face_img)
                if aligned_face is None:
                    continue

                encodings = face_recognition.face_encodings(aligned_face)
                if not encodings:
                    continue

                new_encoding = encodings[0]
                for stored in stored_encodings:
                    results = face_recognition.compare_faces([stored], new_encoding, tolerance=0.45)
                    if results[0]:
                        match_found = True
                        break
                if match_found:
                    break
            if match_found:
                break

        except Exception as e:
            logger.warning(f"Error processing uploaded image: {e}")
            continue

    # Step 3️⃣: Handle no match
    if not match_found:
        return {"success": False, "message": "No matching face found. Deletion cancelled."}

    # Step 4️⃣: Proceed with deletion if match found
    try:
        success, message = delete_face_by_id(req.id_number)
        return {"success": success, "message": message}
    except Exception as e:
        logger.error(f"Error during deletion: {e}")
        return {"success": False, "message": f"Error during deletion: {e}"}