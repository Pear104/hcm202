import asyncio
from .core import generate_response as _generate_response

async def generate_response_async(user_query: str, model_name: str = "gemini"):
    return await asyncio.to_thread(_generate_response, user_query, model_name)
