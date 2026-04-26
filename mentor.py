import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from prompts import rag_prompt, chat_prompt
from rag import get_relevant_context, has_document

MODEL_PATH = "/kaggle/working/phi2_model"
_llm = None
_memory = None
_is_loaded = False

def load_model():
    global _llm, _memory, _is_loaded
    print("⏳ Loading Phi-2...")
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    mdl.eval()
    print(f"✅ Phi-2 loaded! GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    pipe = pipeline(
        "text-generation",
        model=mdl,
        tokenizer=tok,
        max_new_tokens=120,   # short → always complete!
        temperature=0.3,      # low → focused answers
        top_p=0.85,
        repetition_penalty=1.3,
        do_sample=True,
        return_full_text=False
    )
    _llm = HuggingFacePipeline(pipeline=pipe)
    _memory = ConversationBufferWindowMemory(
        k=10, human_prefix="User", ai_prefix="Alex", input_key="input"
    )
    _is_loaded = True
    print("✅ Ready!")

def clean_response(r):
    stop_phrases = [
        "User:", "Human:", "\nUser", "\nHuman",
        "[INST]", "[/INST]", "[INSERT", "Alex (",
        "Assistant:", "<|", "|>", "Generated Output:",
        "Question:", "Note:", "Disclaimer:"
    ]
    for stop in stop_phrases:
        if stop in r:
            r = r.split(stop)[0]
    r = r.strip()
    # Ensure response ends at a complete sentence
    if r and r[-1] not in ".!?":
        last_punct = max(r.rfind("."), r.rfind("!"), r.rfind("?"))
        if last_punct > 20:
            r = r[:last_punct+1]
    return r if len(r) > 15 else "Could you give me more details about your career goals?"

def get_response(user_input):
    global _llm, _memory, _is_loaded
    if not _is_loaded:
        load_model()
    history = _memory.load_memory_variables({}).get("history", "")
    if has_document():
        context = get_relevant_context(user_input, k=3)
        prompt = rag_prompt.format(
            context=context, history=history, input=user_input
        )
    else:
        prompt = chat_prompt.format(history=history, input=user_input)
    response = clean_response(_llm(prompt))
    _memory.save_context({"input": user_input}, {"output": response})
    return response

def reset_conversation():
    global _memory
    if _memory: _memory.clear()

def is_model_loaded():
    return _is_loaded
