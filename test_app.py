import streamlit as st

st.title("Test App")
st.write("Hello! App is working!")

try:
    from prompts import WELCOME_MESSAGE, rag_prompt, chat_prompt
    st.success("✅ prompts.py OK")
except Exception as e:
    st.error(f"❌ prompts.py error: {e}")

try:
    from rag import has_document, get_relevant_context, process_uploaded_file, reset_vector_store
    st.success("✅ rag.py OK")
except Exception as e:
    st.error(f"❌ rag.py error: {e}")

try:
    from mentor import get_response, reset_conversation, is_model_loaded
    st.success("✅ mentor.py OK")
except Exception as e:
    st.error(f"❌ mentor.py error: {e}")

st.write("All imports tested!")