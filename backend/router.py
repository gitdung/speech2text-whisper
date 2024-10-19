from fastapi import HTTPException, APIRouter, FastAPI, Response, UploadFile, File
from service import recognition_service

router = APIRouter()

@router.get("/")
def index():
    return {"message": "Hello World"}

@router.post("/predict")
def recognition(file: UploadFile = File(...)):
    data = file.file
    result = recognition_service(data)
    return {"result": result}