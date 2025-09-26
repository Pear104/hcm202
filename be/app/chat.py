from fastapi import APIRouter
from rag.core import generate_response
from . import schemas

router = APIRouter()

@router.post("/chat", response_model=schemas.ChatResponse)
def chat(req: schemas.ChatRequest):
    # Không lưu lịch sử, không xác thực
    answer = generate_response(req.question, model_name=req.model_name)
    return {"answer": answer}

@router.get("/health")
def health():
    return {"status": "ok"}