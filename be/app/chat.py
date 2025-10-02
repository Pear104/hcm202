from fastapi import APIRouter
from rag.core_async import generate_response_async
from . import schemas
import asyncio

router = APIRouter()


@router.post("/chat", response_model=schemas.ChatResponse)
async def chat(req: schemas.ChatRequest):
    # Không lưu lịch sử, không xác thực
    answer = await generate_response_async(req.question, req.model_name)
    return {"answer": answer}


@router.get("/health")
def health():
    return {"status": "ok"}
