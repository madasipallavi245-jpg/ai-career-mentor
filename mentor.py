import os
import requests

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
API_URL = "https://api.groq.com/openai/v1/chat/completions"

_history = []
_is_loaded = False

def load_model():
    global _is_loaded
    _is_loaded = True

def call_groq_api(user_input: str, context: str = "") -> str:
    if not GROQ_API_KEY:
        return "GROQ_API_KEY is not set. Please add it in Streamlit Secrets."

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Build system message
    if context:
        system_msg = f"""You are Alex, an expert AI career mentor. 
Use the resume context below to give personalized, specific advice.
Keep responses to 2-3 sentences. Be practical and encouraging.

Resume Context:
{context[:1000]}"""
    else:
        system_msg = """You are Alex, an expert AI career mentor.
Give specific, actionable career advice in 2-3 sentences.
Be practical, encouraging, and professional."""

    # Build messages with history
    messages = [{"role": "system", "content": system_msg}]
    for h in _history[-4:]:
        messages.append({"role": "user", "content": h["user"]})
        messages.append({"role": "assistant", "content": h["alex"]})
    messages.append({"role": "user", "content": user_input})

    payload = {
        "model": "llama3-8b-8192",  # Free, fast model on Groq
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7,
    }

    try:
        response = requests.post(
            API_URL, headers=headers,
            json=payload, timeout=30
        )
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        elif response.status_code == 401:
            return "Invalid GROQ_API_KEY. Please check your Streamlit Secrets."
        else:
            return f"API error {response.status_code}: {response.text[:200]}"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def get_response(user_input: str) -> str:
    global _is_loaded
    if not _is_loaded:
        load_model()

    from rag import get_relevant_context, has_document
    context = get_relevant_context(user_input, k=3) if has_document() else ""
    response = call_groq_api(user_input, context)
    _history.append({"user": user_input, "alex": response})
    return response

def reset_conversation():
    global _history
    _history = []

def is_model_loaded() -> bool:
    return _is_loaded