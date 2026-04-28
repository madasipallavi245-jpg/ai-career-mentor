import os
import requests

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# Simple conversation history without langchain
_history = []
_is_loaded = False

def load_model():
    global _is_loaded
    _is_loaded = True

def call_hf_api(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.3,
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
            return "Model is loading, please try again in 20 seconds."
        else:
            return f"API error {response.status_code}. Please try again."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Error: {str(e)}"

def clean_response(r: str) -> str:
    stop_phrases = [
        "User:", "Human:", "\nUser", "\nHuman",
        "[INST]", "[/INST]", "Alex (", "Assistant:",
        "<|", "|>", "Generated Output:", "Question:", "Note:"
    ]
    for stop in stop_phrases:
        if stop in r:
            r = r.split(stop)[0]
    r = r.strip()
    if r and r[-1] not in ".!?":
        last_punct = max(r.rfind("."), r.rfind("!"), r.rfind("?"))
        if last_punct > 20:
            r = r[:last_punct+1]
    return r if len(r) > 15 else "Could you give me more details about your career goals?"

def build_prompt(user_input: str, context: str = "") -> str:
    # Build history string (last 5 exchanges)
    history_str = ""
    for h in _history[-5:]:
        history_str += f"User: {h['user']}\nAlex: {h['alex']}\n"

    if context:
        return f"""You are Alex, a career mentor.
Read the resume below and answer in 2-3 short sentences.

Resume:
{context}

{history_str}
User: {user_input}
Alex:"""
    else:
        return f"""You are Alex, a career mentor.
Answer in 2-3 short complete sentences. Be helpful and specific.

{history_str}
User: {user_input}
Alex:"""

def get_response(user_input: str) -> str:
    global _is_loaded
    if not _is_loaded:
        load_model()

    from rag import get_relevant_context, has_document
    context = get_relevant_context(user_input, k=3) if has_document() else ""
    prompt = build_prompt(user_input, context)
    raw = call_hf_api(prompt)
    response = clean_response(raw)

    # Save to history
    _history.append({"user": user_input, "alex": response})
    return response

def reset_conversation():
    global _history
    _history = []

def is_model_loaded() -> bool:
    return _is_loaded