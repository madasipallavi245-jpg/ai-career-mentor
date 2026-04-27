import os, requests
from langchain.memory import ConversationBufferWindowMemory
from prompts import rag_prompt, chat_prompt
from rag import get_relevant_context, has_document

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = "https://api-inference.huggingface.co/models/microsoft/phi-2"

_memory = None
_is_loaded = False

def load_model():
    global _memory, _is_loaded
    _memory = ConversationBufferWindowMemory(
        k=5, human_prefix="User", ai_prefix="Alex", input_key="input"
    )
    _is_loaded = True

def query_api(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "")
        return "I am having trouble connecting. Please try again."
    except Exception as e:
        return f"API error: {str(e)}"

def clean_response(r):
    for stop in ["User:","Human:","\nUser","\nHuman","[INST]","[/INST]","Alex ("]:
        if stop in r:
            r = r.split(stop)[0]
    r = r.strip()
    if r and r[-1] not in ".!?":
        last_punct = max(r.rfind("."), r.rfind("!"), r.rfind("?"))
        if last_punct > 20:
            r = r[:last_punct+1]
    return r if len(r) > 15 else "Could you give me more details about your career goals?"

def get_response(user_input):
    global _memory, _is_loaded
    if not _is_loaded:
        load_model()
    history = _memory.load_memory_variables({}).get("history", "")
    if has_document():
        context = get_relevant_context(user_input, k=3)
        prompt = rag_prompt.format(context=context, history=history, input=user_input)
    else:
        prompt = chat_prompt.format(history=history, input=user_input)
    response = clean_response(query_api(prompt))
    _memory.save_context({"input": user_input}, {"output": response})
    return response

def reset_conversation():
    global _memory
    if _memory:
        _memory.clear()

def is_model_loaded():
    return _is_loaded
