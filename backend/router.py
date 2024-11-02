from fastapi import HTTPException, APIRouter, FastAPI, Response, UploadFile, File
from service import recognition_service

router = APIRouter()

@router.get("/")
def index():
    return {"message": "Hello World"}

@router.post("/predict")
async def recognition(file: UploadFile = File(...)):
    try:
        # Đọc file âm thanh dưới dạng bytes
        data = await file.read()
        
        # Gọi recognition_service để xử lý âm thanh
        result = recognition_service(io.BytesIO(data))
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))