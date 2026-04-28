import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Using Zephyr 7B - completely free, no approval needed, great for chat
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

_history = []
_is_loaded = False

def load_model():
    global _is_loaded
    _is_loaded = True

def call_hf_api(prompt: str) -> str:
    if not HF_TOKEN:
        return "HF_TOKEN is not set. Please add it in Streamlit Secrets."

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.4,
            "top_p": 0.85,
            "repetition_penalty": 1.3,
            "do_sample": True,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(
            API_URL, headers=headers,
            json=payload, timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "")
            return str(result)
        elif response.status_code == 503:
            return "Model is loading, please wait 20 seconds and try again."
        elif response.status_code == 404:
            return f"Model not found (404)."
        elif response.status_code == 401:
            return "Unauthorized — please check your HF_TOKEN in Streamlit Secrets."
        elif response.status_code == 403:
            return "Access denied — you may need to accept model terms on Hugging Face."
        else:
            return f"API error {response.status_code}: {response.text[:200]}"
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def clean_response(r: str) -> str:
    if not r or len(r) < 5:
        return "Could you give me more details about your career goals?"
    stop_phrases = [
        "User:", "Human:", "\nUser", "\nHuman",
        "<|user|>", "<|system|>", "<|assistant|>",
        "[INST]", "[/INST]", "Assistant:", "Note:"
    ]
    for stop in stop_phrases:
        if stop in r:
            r = r.split(stop)[0]
    r = r.strip()
    if r and r[-1] not in ".!?":
        last_punct = max(r.rfind("."), r.rfind("!"), r.rfind("?"))
        if last_punct > 20:
            r = r[:last_punct + 1]
    return r if len(r) > 15 else "Could you give me more details about your career goals?"

def build_prompt(user_input: str, context: str = "") -> str:
    history_str = ""
    for h in _history[-3:]:
        history_str += f"<|user|>\n{h['user']}</s>\n<|assistant|>\n{h['alex']}</s>\n"

    if context:
        system = f"You are Alex, a helpful career mentor. Use the resume context to give personalized advice in 2-3 sentences.\n\nResume:\n{context}"
    else:
        system = "You are Alex, a helpful career mentor. Give specific, actionable advice in 2-3 sentences."

    prompt = f"<|system|>\n{system}</s>\n{history_str}<|user|>\n{user_input}</s>\n<|assistant|>\n"
    return prompt

def get_response(user_input: str) -> str:
    global _is_loaded
    if not _is_loaded:
        load_model()

    from rag import get_relevant_context, has_document
    context = get_relevant_context(user_input, k=3) if has_document() else ""
    prompt = build_prompt(user_input, context)
    raw = call_hf_api(prompt)
    response = clean_response(raw)
    _history.append({"user": user_input, "alex": response})
    return response

def reset_conversation():
    global _history
    _history = []

def is_model_loaded() -> bool:
    return _is_loaded