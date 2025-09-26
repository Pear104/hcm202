# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from . import chat
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="RAG Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Chỉ include router chat (không include auth)
app.include_router(chat.router)