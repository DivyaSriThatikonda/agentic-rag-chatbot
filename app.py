import streamlit as st
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
import uuid

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🎨", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
body { background-color: #FAF3E0; font-family: 'Helvetica Neue', sans-serif; }
.header-box {
    background: #FFFFFF; border: 1px solid #E5DCC3; padding: 18px; border-radius: 12px;
    text-align: center; font-size: 24px; font-weight: 600; color: #4E423D;
    margin-bottom: 25px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_agent" not in st.session_state:
    st.session_state.retrieval_agent = RetrievalAgent()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = False
if "status_message" not in st.session_state:
    st.session_state.status_message = None

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 📂 Upload Your Documents")
    st.info("Note: Processing new documents will start a new session and **delete all previously uploaded data**.")
    uploaded_files = st.file_uploader("Upload files to start a new session.", type=["pdf", "docx", "pptx", "csv", "txt"], accept_multiple_files=True)
    if st.button("🚀 Process Documents"):
        if uploaded_files:
            with st.spinner("Clearing old data and processing new documents..."):
                all_text_content = [IngestionAgent(file) for file in uploaded_files if IngestionAgent(file)]
                if all_text_content:
                    st.session_state.retrieval_agent.ingest_and_embed("\n\n".join(all_text_content))
                    st.session_state.processed_files = True
                    st.session_state.chat_history = []
                    st.session_state.status_message = {"msg": "✅ New documents processed and ready!", "type": "success"}
                    st.rerun()
                else:
                    st.session_state.status_message = {"msg": "No text could be extracted from the documents.", "type": "error"}
                    st.rerun()
        else:
            st.session_state.status_message = {"msg": "Please upload at least one file to process.", "type": "warning"}
            st.rerun()
    st.markdown("---")
    st.markdown("## 🕑 Recent Questions")
    if st.session_state.chat_history:
        user_questions = [msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user']
        for question in user_questions[-5:]:
            st.markdown(f"- {question}")
    else:
        st.info("No questions have been asked in this session yet.")

# --- App Header & Status ---
st.markdown("<div class='header-box'>🎨 Agentic RAG Chatbot</div>", unsafe_allow_html=True)
if st.session_state.status_message:
    msg_type = st.session_state.status_message["type"]
    if msg_type == "success": st.success(st.session_state.status_message["msg"])
    elif msg_type == "error": st.error(st.session_state.status_message["msg"])
    elif msg_type == "warning": st.warning(st.session_state.status_message["msg"])

# --- Intro Message ---
if not st.session_state.processed_files:
    st.markdown("""<div style="text-align: center; color: #6F6259; font-size: 14px; margin-bottom: 20px;"><b>How it works:</b> Upload your documents in the sidebar and click "Process Documents" to begin!</div>""", unsafe_allow_html=True)

# --- Chat Display ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Sources"):
                    for i, doc in enumerate(message["sources"]):
                        st.info(f"**Source {i + 1}:**\n\n{doc.page_content}")
            if "mcp_messages" in message:
                with st.expander("🤖 View Agent Messages (MCP)"):
                    st.json(message["mcp_messages"])

# --- User Input & Agent Logic ---
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.status_message = None
    if not st.session_state.processed_files:
        st.warning("Please process your documents before asking a question.")
    else:
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"): st.markdown(user_query)
        
        with st.spinner("Thinking..."):
            # --- MCP WORKFLOW CAPTURE ---
            trace_id = str(uuid.uuid4())
            retrieval_request = {"sender": "Coordinator", "receiver": "RetrievalAgent", "type": "RETRIEVAL_REQUEST", "trace_id": trace_id, "payload": {"query": user_query}}
            retrieved_context = st.session_state.retrieval_agent.retrieve_context(retrieval_request["payload"]["query"])
            context_response = {"sender": "RetrievalAgent", "receiver": "LLMResponseAgent", "type": "CONTEXT_RESPONSE", "trace_id": trace_id, "payload": {"top_chunks_count": len(retrieved_context), "query": user_query}}
            ai_response = LLMResponseAgent(query=context_response["payload"]["query"], context_chunks=retrieved_context)
        
        mcp_data = {
            "Message 1: Retrieval Request": retrieval_request,
            "Message 2: Context Response": context_response
        }
        assistant_message = {"role": "assistant", "content": ai_response, "sources": retrieved_context, "mcp_messages": mcp_data}
        st.session_state.chat_history.append(assistant_message)
        st.rerun()




