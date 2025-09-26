import requests
import os
import google.generativeai as genai
import re
import requests


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def clean_thoughts(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_with_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    try:
        res = model.generate_content(prompt)
        return res.text.strip()
    except Exception as e:
        return f"[Error Calling Gemini]: {e}"

def generate_with_groq(prompt: str, model_name: str) -> str:
    try:
        model_lookup = {
            "gpt": "openai/gpt-oss-20b",
            "gemma": "gemma2-9b-it"
        }
        model_id = model_lookup.get(model_name.lower())

        if not model_id:
            raise ValueError(f"Unsupported model name for Groq: {model_name}")

        res = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful academic advisor."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": 0.7
            }
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[Error Calling Groq]: {e}"


