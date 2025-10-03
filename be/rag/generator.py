import requests
import os
import google.generativeai as genai
import re
import requests


genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
deepseek_url = os.getenv("DEEPSEEK_URL")


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

def generate_with_lmstudio(prompt: str, model_name: str) -> str:
    model_lookup = {
        "deepseek": "deepseek-r1-distill-llama-3b"
    }
    
    model_id = model_lookup.get(model_name.lower())
    try:
        # LM Studio uses the model name as shown in its UI (check /v1/models)
        # Example: "deepseek-r1-distill-llama-3b"
        url_string = f"{deepseek_url}/v1/chat/completions"
        print(url_string)
        res = requests.post(
            url=url_string,  # LM Studio local server
            headers={
                "Authorization": "Bearer lm-studio",  # LM Studio ignores this but some clients require it
                "Content-Type": "application/json"
            },
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful academic advisor."},
                    {"role": "user", "content": prompt.strip()}
                ],
                "temperature": 0.7
            },
            timeout=60
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"].strip()

    except Exception as e:
        return f"[Error Calling LM Studio]: {e}"


