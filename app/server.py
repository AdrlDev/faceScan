from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from app.utils.enroll import enroll_face
from app.utils.scan import scan_once

app = FastAPI()

class EnrollRequest(BaseModel):
    name: str
    id_number: str

@app.post("/api/enroll")
async def api_enroll(req: EnrollRequest):
    """
    Enroll a user by capturing their face through server webcam.
    """
    result = enroll_face(req.name, req.id_number)
    return result

@app.post("/api/scan")
async def api_scan():
    """Perform one scan cycle and return result"""
    result = scan_once()
    return {"status": "ok", "result": result}
