import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from rag import process_uploaded_file, reset_vector_store, has_document
    from mentor import get_response, reset_conversation, is_model_loaded
    from prompts import WELCOME_MESSAGE
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

st.set_page_config(page_title="AI Career Mentor", page_icon="🎓", layout="wide")
st.markdown("""<style>
.mentor-header{background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);padding:20px 30px;border-radius:15px;color:white;margin-bottom:20px;}
.file-success{background:#d4edda;border:1px solid #c3e6cb;border-radius:8px;padding:10px;color:#155724;font-size:14px;}
.block-container{padding-top:1rem;}
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="mentor-header">
<h1 style="margin:0;font-size:28px;">🎓 AI Career Mentor</h1>
<p style="margin:5px 0 0 0;opacity:0.9;">Powered by Phi-2 + RAG + FAISS — 100% Free</p>
</div>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"assistant","content":WELCOME_MESSAGE}]
if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "pending_input" not in st.session_state:
    st.session_state.pending_input = None

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    st.markdown("### 📎 Upload Resume")
    st.markdown("*PDF, TXT, DOCX*")
    uploaded_file = st.file_uploader("Drop resume", type=["pdf","txt","docx"], label_visibility="collapsed")
    if uploaded_file is not None:
        if uploaded_file.name != st.session_state.uploaded_filename:
            with st.spinner(f"Reading {uploaded_file.name}..."):
                result = process_uploaded_file(uploaded_file)
            if "✅" in result:
                st.session_state.file_uploaded = True
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.messages.append({"role":"assistant","content":f"📄 Read **{uploaded_file.name}**! Ask me to review it or identify skill gaps."})
                st.rerun()
            else:
                st.error(result)
    if st.session_state.file_uploaded:
        st.markdown(f'''<div class="file-success">✅ <strong>{st.session_state.uploaded_filename}</strong><br>Resume ready!</div>''', unsafe_allow_html=True)
    else:
        st.info("💡 Upload resume for personalized advice")
    st.markdown("---")
    st.markdown("### 💡 Quick Topics")
    for q in ["📄 Review my resume","🎯 Transition to Data Science","💼 Software engineer interview prep","📈 AI/ML skill roadmap","💰 Salary negotiation tips","🔍 Job search strategy"]:
        if st.button(q, use_container_width=True):
            st.session_state.pending_input = q
    st.markdown("---")
    if st.button("🔄 Reset Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = [{"role":"assistant","content":WELCOME_MESSAGE}]
        st.session_state.file_uploaded = False
        st.session_state.uploaded_filename = None
        st.session_state.pending_input = None
        reset_conversation()
        reset_vector_store()
        st.rerun()
    st.markdown("---")
    if is_model_loaded():
        st.success("🟢 Phi-2 Active")
    else:
        st.warning("🟡 Loads on first message (~2 mins)")

if st.session_state.file_uploaded:
    st.success(f"🔍 RAG Active — **{st.session_state.uploaded_filename}**")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🎓" if msg["role"]=="assistant" else "👤"):
        st.markdown(msg["content"])

user_input = None
if st.session_state.pending_input:
    user_input = st.session_state.pending_input
    st.session_state.pending_input = None
chat_input = st.chat_input("Ask Alex about your career...")
if chat_input:
    user_input = chat_input

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    with st.chat_message("assistant", avatar="🎓"):
        with st.spinner("Alex is thinking..."):
            response = get_response(user_input)
        st.markdown(response)
    st.session_state.messages.append({"role":"assistant","content":response})
    st.rerun()
