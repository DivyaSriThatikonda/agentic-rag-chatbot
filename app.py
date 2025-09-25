# import streamlit as st
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- UI Configuration ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖")
# st.title("🤖 Agentic RAG Chatbot")
#
# # --- Session State Initialization ---
# # This is to keep track of variables as the user interacts with the app
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     # We initialize the retrieval agent once and store it in the session state
#     st.session_state.retrieval_agent = RetrievalAgent()
#
# # --- Sidebar for Document Upload ---
# with st.sidebar:
#     st.header("Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Upload your documents here (PDF, DOCX, PPTX, CSV, TXT)",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files:
#         if st.button("Process Documents"):
#             with st.spinner("Processing documents..."):
#                 for file in uploaded_files:
#                     # 1. IngestionAgent is called here
#                     text_content = IngestionAgent(file)
#
#                     # 2. RetrievalAgent's ingest_and_embed is called here
#                     st.session_state.retrieval_agent.ingest_and_embed(text_content)
#
#                 st.success("Documents processed successfully!")
#
# # --- Main Chat Interface ---
# st.write("Welcome to the Chatbot! Ask any question about your documents.")
#
# # Display previous chat messages
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # Handle new user input
# if user_query := st.chat_input("What is your question?"):
#     # Add user's message to chat history and display it
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     with st.chat_message("user"):
#         st.markdown(user_query)
#
#     # Get the AI's response
#     with st.spinner("Thinking..."):
#         # 3. RetrievalAgent's retrieve_context is called here
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(user_query)
#
#         # 4. LLMResponseAgent is called here
#         ai_response = LLMResponseAgent(query=user_query, context_chunks=retrieved_context)
#
#         # Display AI's response and add to chat history
#         with st.chat_message("assistant"):
#             st.markdown(ai_response)
#         st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

# import streamlit as st
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- UI Configuration ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
#
# # --- Sidebar Settings ---
# with st.sidebar:
#     st.header("⚙️ Settings")
#     theme = st.radio("Choose Theme", ["Light", "Dark"], index=0)
#     st.markdown("---")
#     st.header("📂 Upload Documents")
#     uploaded_files = st.file_uploader(
#         "Upload your documents (PDF, DOCX, PPTX, CSV, TXT)",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )
#
# # --- Session State Initialization ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()
# if "retrieved_sources" not in st.session_state:
#     st.session_state.retrieved_sources = []
# if "agent_trace" not in st.session_state:
#     st.session_state.agent_trace = []
#
# # --- Process Documents ---
# if uploaded_files and st.sidebar.button("🚀 Process Documents"):
#     with st.spinner("IngestionAgent parsing documents... 🔄"):
#         for file in uploaded_files:
#             text_content = IngestionAgent(file)
#             st.session_state.retrieval_agent.ingest_and_embed(text_content)
#     st.sidebar.success("✅ Documents processed successfully!")
#
# # --- Main Tabs ---
# tab_chat, tab_sources, tab_trace = st.tabs(["💬 Chat", "📖 Sources", "🔍 Agent Trace"])
#
# # ---------------- CHAT TAB ----------------
# with tab_chat:
#     st.subheader("🤖 Agentic RAG Chatbot")
#     st.markdown("Ask questions about your uploaded documents.")
#
#     # Display previous chat
#     for message in st.session_state.chat_history:
#         role = "user" if message["role"] == "user" else "assistant"
#         with st.chat_message(role):
#             st.markdown(message["content"])
#
#     # Handle user input
#     if user_query := st.chat_input("Type your question..."):
#         # Save user query
#         st.session_state.chat_history.append({"role": "user", "content": user_query})
#         with st.chat_message("user"):
#             st.markdown(user_query)
#
#         with st.spinner("🤖 Agents are working..."):
#             # Trace: RetrievalAgent
#             st.session_state.agent_trace.append("🔍 RetrievalAgent: Searching relevant chunks")
#             retrieved_context = st.session_state.retrieval_agent.retrieve_context(user_query)
#             st.session_state.retrieved_sources = retrieved_context
#
#             # Trace: LLMResponseAgent
#             st.session_state.agent_trace.append("✨ LLMResponseAgent: Generating response")
#             ai_response = LLMResponseAgent(query=user_query, context_chunks=retrieved_context)
#
#         # Show AI response
#         with st.chat_message("assistant"):
#             st.markdown(ai_response)
#         st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
#
# # ---------------- SOURCES TAB ----------------
# with tab_sources:
#     st.subheader("📖 Retrieved Sources")
#     if st.session_state.retrieved_sources:
#         for i, src in enumerate(st.session_state.retrieved_sources, 1):
#             with st.expander(f"Source {i}"):
#                 st.write(src)
#     else:
#         st.info("No sources retrieved yet. Ask a question first!")
#
# # ---------------- TRACE TAB ----------------
# with tab_trace:
#     st.subheader("🔍 Agent Trace (MCP Messages)")
#     if st.session_state.agent_trace:
#         for step in st.session_state.agent_trace:
#             st.markdown(f"- {step}")
#     else:
#         st.info("No trace yet. Try asking a question!")

# import streamlit as st
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="centered")
#
# # --- Custom CSS for Stylish UI ---
# st.markdown("""
# <style>
# body {
#     background: linear-gradient(135deg, #f8f9fa, #e0f7fa);
# }
# .chat-bubble-user {
#     background-color: #1976d2;
#     color: white;
#     padding: 12px 16px;
#     border-radius: 18px 18px 0px 18px;
#     margin: 5px 0;
#     max-width: 70%;
#     align-self: flex-end;
# }
# .chat-bubble-assistant {
#     background-color: #f1f1f1;
#     color: black;
#     padding: 12px 16px;
#     border-radius: 18px 18px 18px 0px;
#     margin: 5px 0;
#     max-width: 70%;
#     align-self: flex-start;
#     box-shadow: 0px 2px 6px rgba(0,0,0,0.1);
# }
# .source-link {
#     font-size: 12px;
#     color: #00796b;
#     cursor: pointer;
# }
# </style>
# """, unsafe_allow_html=True)
#
# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()
# if "retrieved_sources" not in st.session_state:
#     st.session_state.retrieved_sources = []
#
# # --- Header ---
# st.markdown("<h1 style='text-align: center;'>🤖 Agentic RAG Chatbot</h1>", unsafe_allow_html=True)
#
# # --- Upload Floating Box ---
# st.markdown("### 📂 Upload your documents")
# uploaded_files = st.file_uploader(
#     "Drag and drop or select files",
#     type=["pdf", "docx", "pptx", "csv", "txt"],
#     accept_multiple_files=True
# )
#
# if uploaded_files and st.button("🚀 Process Documents"):
#     with st.spinner("Processing documents..."):
#         for file in uploaded_files:
#             text_content = IngestionAgent(file)
#             st.session_state.retrieval_agent.ingest_and_embed(text_content)
#     st.success("✅ Documents processed!")
#
# # --- Chat History Display ---
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             st.markdown(f"<div class='chat-bubble-user'>{message['content']}</div>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<div class='chat-bubble-assistant'>{message['content']}</div>", unsafe_allow_html=True)
#             if message.get("sources"):
#                 with st.expander("🔎 View Sources"):
#                     for src in message["sources"]:
#                         st.write(src)
#
# # --- User Input ---
# if user_query := st.chat_input("Ask a question..."):
#     # Save user query
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)
#
#     with st.spinner("🤖 Thinking..."):
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(user_query)
#         ai_response = LLMResponseAgent(query=user_query, context_chunks=retrieved_context)
#
#     # Save assistant response with sources
#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": ai_response,
#         "sources": retrieved_context
#     })
#     st.markdown(f"<div class='chat-bubble-assistant'>{ai_response}</div>", unsafe_allow_html=True)
#     with st.expander("🔎 View Sources"):
#         for src in retrieved_context:
#             st.write(src)



# import streamlit as st
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
#
# # --- Custom CSS ---
# st.markdown("""
# <style>
# /* Background pastel gradient */
# body {
#     background: linear-gradient(135deg, #fbc2eb, #a6c1ee);
#     background-attachment: fixed;
#     font-family: 'Segoe UI', sans-serif;
# }
#
# /* Heading Box */
# .header-box {
#     background: #ffffffcc;
#     border: 2px solid #a6c1ee;
#     padding: 15px;
#     border-radius: 12px;
#     text-align: center;
#     font-size: 28px;
#     font-weight: bold;
#     color: #333333;
#     margin-bottom: 20px;
#     box-shadow: 0px 3px 8px rgba(0,0,0,0.1);
# }
#
# /* Chat bubbles */
# .chat-bubble-user {
#     background-color: #1976d2;
#     color: white;
#     padding: 12px 16px;
#     border-radius: 18px 18px 0px 18px;
#     margin: 5px 0;
#     max-width: 70%;
#     align-self: flex-end;
# }
# .chat-bubble-assistant {
#     background-color: #ffffffcc;
#     color: black;
#     padding: 12px 16px;
#     border-radius: 18px 18px 18px 0px;
#     margin: 5px 0;
#     max-width: 70%;
#     align-self: flex-start;
#     box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
# }
# </style>
# """, unsafe_allow_html=True)
#
# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()
# if "retrieved_sources" not in st.session_state:
#     st.session_state.retrieved_sources = []
#
# # --- Sidebar (flexible: can collapse/expand) ---
# with st.sidebar:
#     st.markdown("## 📂 Drop Your Documents")
#     uploaded_files = st.file_uploader(
#         "Upload files here",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files and st.button("🚀 Process Documents"):
#         with st.spinner("Processing documents..."):
#             for file in uploaded_files:
#                 text_content = IngestionAgent(file)
#                 st.session_state.retrieval_agent.ingest_and_embed(text_content)
#         st.success("✅ Documents processed successfully!")
#
#     st.markdown("---")
#     st.markdown("## 🕑 Recent Questions")
#     if st.session_state.chat_history:
#         for msg in st.session_state.chat_history[-5:]:
#             if msg["role"] == "user":
#                 st.markdown(f"- ❓ {msg['content']}")
#     else:
#         st.info("No questions yet!")
#
# # --- App Header ---
# st.markdown("<div class='header-box'>🤖 Agentic RAG Chatbot</div>", unsafe_allow_html=True)
#
# # --- Intro Message ---
# st.markdown("""
# ### 📘 How it works?
# 1. **Drop your documents** in the left sidebar.
# 2. **Ask questions** in the chat box.
# 3. The bot will answer **based on your documents** with references.
# """)
#
# # --- Chat Section ---
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             st.markdown(f"<div class='chat-bubble-user'>{message['content']}</div>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<div class='chat-bubble-assistant'>{message['content']}</div>", unsafe_allow_html=True)
#             if message.get("sources"):
#                 with st.expander("🔎 View Sources"):
#                     for src in message["sources"]:
#                         st.write(src)
#
# # --- User Input ---
# if user_query := st.chat_input("Type your question here..."):
#     # Save user query
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)
#
#     with st.spinner("🤖 Thinking..."):
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(user_query)
#         ai_response = LLMResponseAgent(query=user_query, context_chunks=retrieved_context)
#
#     # Save assistant response with sources
#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": ai_response,
#         "sources": retrieved_context
#     })
#     st.markdown(f"<div class='chat-bubble-assistant'>{ai_response}</div>", unsafe_allow_html=True)
#     with st.expander("🔎 View Sources"):
#         for src in retrieved_context:
#             st.write(src)


#@



#
# import streamlit as st
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
#
# # --- Custom CSS (Claude-style theme) ---
# st.markdown("""
# <style>
# /* Keep clean white background */
# [data-testid="stAppViewContainer"] {
#     background: white;
#     font-family: 'Segoe UI', sans-serif;
# }
#
# /* Sidebar soft background */
# [data-testid="stSidebar"] {
#     background: #fafafa;
#     border-right: 1px solid #eee;
# }
#
# /* Heading Box (pastel lavender) */
# .header-box {
#     background: #ede7f6;  /* pastel lavender */
#     border: 1px solid #d1c4e9;
#     padding: 15px;
#     border-radius: 12px;
#     text-align: center;
#     font-size: 26px;
#     font-weight: bold;
#     color: #4a148c;
#     margin-bottom: 20px;
#     box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
# }
#
# /* User chat bubble (pastel mint) */
# .chat-bubble-user {
#     background-color: #e0f7fa; /* pastel mint */
#     color: #004d40;
#     padding: 12px 16px;
#     border-radius: 16px 16px 0px 16px;
#     margin: 6px 0;
#     max-width: 70%;
#     align-self: flex-end;
#     border: 1px solid #b2ebf2;
# }
#
# /* Assistant chat bubble (soft gray/neutral) */
# .chat-bubble-assistant {
#     background-color: #f5f5f5; /* soft neutral gray */
#     color: #212121;
#     padding: 14px 18px;
#     border-radius: 16px 16px 16px 0px;
#     margin: 6px 0;
#     max-width: 75%;
#     align-self: flex-start;
#     border: 1px solid #e0e0e0;
#     box-shadow: 0px 2px 4px rgba(0,0,0,0.08);
# }
# </style>
# """, unsafe_allow_html=True)
#
# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()
#
# # --- Sidebar ---
# with st.sidebar:
#     st.markdown("## 📂 Drop Your Documents")
#     uploaded_files = st.file_uploader(
#         "Upload files here",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files and st.button("🚀 Process Documents"):
#         with st.spinner("Processing documents..."):
#             for file in uploaded_files:
#                 text_content = IngestionAgent(file)
#                 st.session_state.retrieval_agent.ingest_and_embed(text_content)
#         st.success("✅ Documents processed successfully!")
#
#     st.markdown("---")
#     st.markdown("## 🕑 Recent Questions")
#     if st.session_state.chat_history:
#         for msg in st.session_state.chat_history[-5:]:
#             if msg["role"] == "user":
#                 st.markdown(f"- ❓ {msg['content']}")
#     else:
#         st.info("No questions yet!")
#
# # --- App Header ---
# st.markdown("<div class='header-box'>🤖 Agentic RAG Chatbot</div>", unsafe_allow_html=True)
#
# # --- Intro Message ---
# st.markdown("""
# ### 📘 How it works?
# 1. **Drop your documents** in the left sidebar.
# 2. **Ask questions** in the chat box.
# 3. The bot will answer **based on your documents**.
# """)
#
# # --- Chat Section ---
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         if message["role"] == "user":
#             st.markdown(f"<div class='chat-bubble-user'>{message['content']}</div>", unsafe_allow_html=True)
#         else:
#             st.markdown(f"<div class='chat-bubble-assistant'>{message['content']}</div>", unsafe_allow_html=True)
#
# # --- User Input ---
# if user_query := st.chat_input("Type your question here..."):
#     # Save user query
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)
#
#     with st.spinner("🤖 Thinking..."):
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(user_query)
#         ai_response = LLMResponseAgent(query=user_query, context_chunks=retrieved_context)
#
#     # Save assistant response
#     st.session_state.chat_history.append({
#         "role": "assistant",
#         "content": ai_response,
#     })
#     st.markdown(f"<div class='chat-bubble-assistant'>{ai_response}</div>", unsafe_allow_html=True)






#
# import streamlit as st
# import uuid
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🎨", layout="wide")
#
# # --- NEW CSS: Warm Pastel Orange & Cream Theme ---
# st.markdown("""
# <style>
# /* Core body and font styles */
# body {
#     background-color: #FAF3E0; /* Soft Cream Background */
#     font-family: 'Helvetica Neue', sans-serif;
# }
#
# /* Header Box */
# .header-box {
#     background: #FFFFFF;
#     border: 1px solid #E5DCC3;
#     padding: 18px;
#     border-radius: 12px;
#     text-align: center;
#     font-size: 24px;
#     font-weight: 600;
#     color: #4E423D; /* Warm dark brown text */
#     margin-bottom: 25px;
#     box-shadow: 0 2px 5px rgba(0,0,0,0.05);
# }
#
# /* Chat Bubbles - Warm & Friendly Design */
# .chat-bubble-user {
#     background-color: #FAB488; /* Pastel Orange */
#     color: #4E423D;
#     padding: 14px 20px;
#     border-radius: 18px 18px 4px 18px;
#     margin: 6px 0;
#     max-width: 70%;
#     float: right;
#     clear: both;
#     line-height: 1.6;
#     border: 1px solid #F8A56E;
# }
# .chat-bubble-assistant {
#     background-color: #FFFFFF; /* Clean white for contrast */
#     color: #4E423D;
#     border: 1px solid #E5DCC3;
#     padding: 14px 20px;
#     border-radius: 18px 18px 18px 4px;
#     margin: 6px 0;
#     max-width: 80%;
#     float: left;
#     clear: both;
#     box-shadow: 0 1px 3px rgba(0,0,0,0.05);
#     line-height: 1.6;
# }
#
# /* Expander for sources */
# .stExpander {
#     border: none !important;
#     box-shadow: none !important;
# }
# .st-emotion-cache-1h9usn1 { /* Targets the expander header */
#     font-size: 14px;
#     color: #6F6259;
# }
# </style>
# """, unsafe_allow_html=True)
#
# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()
#
# # --- Sidebar ---
# with st.sidebar:
#     st.markdown("## 📂 Drop Your Documents")
#     uploaded_files = st.file_uploader(
#         "Upload files here",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )
#
#     if uploaded_files and st.button("🚀 Process Documents"):
#         with st.spinner("Processing documents..."):
#             for file in uploaded_files:
#                 text_content = IngestionAgent(file)
#                 if text_content:
#                     st.session_state.retrieval_agent.ingest_and_embed(text_content)
#         st.success("✅ Documents processed successfully!")
#
#     st.markdown("---")
#     st.markdown("## 🕑 Recent Questions")
#     if st.session_state.chat_history:
#         user_questions = [msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user']
#         for question in user_questions[-5:]:
#             st.markdown(f"- {question}")
#     else:
#         st.info("No questions yet!")
#
# # --- App Header ---
# st.markdown("<div class='header-box'>🎨 Agentic RAG Chatbot</div>", unsafe_allow_html=True)
#
# # --- Intro Message ---
# st.markdown("""
# <div style="text-align: center; color: #6F6259; font-size: 14px; margin-bottom: 20px;">
#     <b>How it works:</b> Drop your documents in the sidebar, then ask questions about them!
# </div>
# """, unsafe_allow_html=True)
#
# # --- Chat Section ---
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         role = message["role"]
#         content = message["content"]
#         st.markdown(f"<div class='chat-bubble-{role}'>{content}</div>", unsafe_allow_html=True)
#         if role == "assistant" and "sources" in message:
#             with st.expander("🔍 View Sources"):
#                 for i, doc in enumerate(message["sources"]):
#                     st.info(f"**Source {i + 1}:**\n\n{doc.page_content}")
#
# # --- User Input & Agent Logic ---
# if user_query := st.chat_input("Ask a question about your documents..."):
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     with chat_container:
#         st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)
#
#     with st.spinner("Thinking..."):
#         # --- MCP WORKFLOW ---
#         trace_id = str(uuid.uuid4())
#         retrieval_request = {
#             "sender": "Coordinator", "receiver": "RetrievalAgent", "type": "RETRIEVAL_REQUEST",
#             "trace_id": trace_id, "payload": {"query": user_query}
#         }
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(
#             retrieval_request["payload"]["query"]
#         )
#         context_response = {
#             "sender": "RetrievalAgent", "receiver": "LLMResponseAgent", "type": "CONTEXT_RESPONSE",
#             "trace_id": trace_id, "payload": {"top_chunks": retrieved_context, "query": user_query}
#         }
#         ai_response = LLMResponseAgent(
#             query=context_response["payload"]["query"],
#             context_chunks=context_response["payload"]["top_chunks"]
#         )
#
#     assistant_message = {
#         "role": "assistant",
#         "content": ai_response,
#         "sources": retrieved_context
#     }
#     st.session_state.chat_history.append(assistant_message)
#
#     st.rerun()

# before final code
# import streamlit as st
# import uuid
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent

# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🎨", layout="wide")

# # --- NEW CSS: Warm Pastel Orange & Cream Theme ---
# st.markdown("""
# <style>
# /* Core body and font styles */
# body {
#     background-color: #FAF3E0; /* Soft Cream Background */
#     font-family: 'Helvetica Neue', sans-serif;
# }

# /* Header Box */
# .header-box {
#     background: #FFFFFF;
#     border: 1px solid #E5DCC3;
#     padding: 18px;
#     border-radius: 12px;
#     text-align: center;
#     font-size: 24px;
#     font-weight: 600;
#     color: #4E423D; /* Warm dark brown text */
#     margin-bottom: 25px;
#     box-shadow: 0 2px 5px rgba(0,0,0,0.05);
# }

# /* Chat Bubbles - Warm & Friendly Design */
# .chat-bubble-user {
#     background-color: #FAB488; /* Pastel Orange */
#     color: #4E423D;
#     padding: 14px 20px;
#     border-radius: 18px 18px 4px 18px;
#     margin: 6px 0;
#     max-width: 70%;
#     float: right;
#     clear: both;
#     line-height: 1.6;
#     border: 1px solid #F8A56E;
# }
# .chat-bubble-assistant {
#     background-color: #FFFFFF; /* Clean white for contrast */
#     color: #4E423D;
#     border: 1px solid #E5DCC3;
#     padding: 14px 20px;
#     border-radius: 18px 18px 18px 4px;
#     margin: 6px 0;
#     max-width: 80%;
#     float: left;
#     clear: both;
#     box-shadow: 0 1px 3px rgba(0,0,0,0.05);
#     line-height: 1.6;
# }

# /* Expander for sources */
# .stExpander {
#     border: none !important;
#     box-shadow: none !important;
# }
# .st-emotion-cache-1h9usn1 { /* Targets the expander header */
#     font-size: 14px;
#     color: #6F6259;
# }
# </style>
# """, unsafe_allow_html=True)

# # --- Session State ---
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
# if "retrieval_agent" not in st.session_state:
#     st.session_state.retrieval_agent = RetrievalAgent()

# # --- Sidebar ---
# with st.sidebar:
#     st.markdown("## 📂 Drop Your Documents")
#     uploaded_files = st.file_uploader(
#         "Upload files here",
#         type=["pdf", "docx", "pptx", "csv", "txt"],
#         accept_multiple_files=True
#     )

#     if uploaded_files and st.button("🚀 Process Documents"):
#         with st.spinner("Processing documents..."):
#             for file in uploaded_files:
#                 text_content = IngestionAgent(file)
#                 if text_content:
#                     st.session_state.retrieval_agent.ingest_and_embed(text_content)
#         st.success("✅ Documents processed successfully!")

#     st.markdown("---")
#     st.markdown("## 🕑 Recent Questions")
#     if st.session_state.chat_history:
#         user_questions = [msg['content'] for msg in st.session_state.chat_history if msg['role'] == 'user']
#         for question in user_questions[-5:]:
#             st.markdown(f"- {question}")
#     else:
#         st.info("No questions yet!")

# # --- App Header ---
# st.markdown("<div class='header-box'>🎨 Agentic RAG Chatbot</div>", unsafe_allow_html=True)

# # --- Intro Message ---
# st.markdown("""
# <div style="text-align: center; color: #6F6259; font-size: 14px; margin-bottom: 20px;">
#     <b>How it works:</b> Drop your documents in the sidebar, then ask questions about them!
# </div>
# """, unsafe_allow_html=True)

# # --- Chat Section ---
# chat_container = st.container()
# with chat_container:
#     for message in st.session_state.chat_history:
#         role = message["role"]
#         content = message["content"]
#         st.markdown(f"<div class='chat-bubble-{role}'>{content}</div>", unsafe_allow_html=True)
#         if role == "assistant" and "sources" in message:
#             with st.expander("🔍 View Sources"):
#                 for i, doc in enumerate(message["sources"]):
#                     st.info(f"**Source {i + 1}:**\n\n{doc.page_content}")

# # --- User Input & Agent Logic ---
# if user_query := st.chat_input("Ask a question about your documents..."):
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     with chat_container:
#         st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)

#     with st.spinner("Thinking..."):
#         # --- MCP WORKFLOW ---
#         trace_id = str(uuid.uuid4())
#         retrieval_request = {
#             "sender": "Coordinator", "receiver": "RetrievalAgent", "type": "RETRIEVAL_REQUEST",
#             "trace_id": trace_id, "payload": {"query": user_query}
#         }
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(
#             retrieval_request["payload"]["query"]
#         )
#         context_response = {
#             "sender": "RetrievalAgent", "receiver": "LLMResponseAgent", "type": "CONTEXT_RESPONSE",
#             "trace_id": trace_id, "payload": {"top_chunks": retrieved_context, "query": user_query}
#         }
#         ai_response = LLMResponseAgent(
#             query=context_response["payload"]["query"],
#             context_chunks=context_response["payload"]["top_chunks"]
#         )

#     assistant_message = {
#         "role": "assistant",
#         "content": ai_response,
#         "sources": retrieved_context
#     }
#     st.session_state.chat_history.append(assistant_message)


#     st.rerun()



import streamlit as st
import uuid
from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent

# --- Page Config ---
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")

# --- App Header ---
st.markdown("## 🤖 Agentic RAG Chatbot")
st.markdown("---")


# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "retrieval_agent" not in st.session_state:
    st.session_state.retrieval_agent = RetrievalAgent()


# --- Sidebar for Document Upload ---
with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, PPTX, CSV, or TXT files",
        type=["pdf", "docx", "pptx", "csv", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("🚀 Process Documents"):
        with st.spinner("Processing documents... This may take a moment."):
            all_text_content = []
            for file in uploaded_files:
                text_content = IngestionAgent(file)
                if text_content:
                    all_text_content.append(text_content)
            
            # Combine all text and ingest at once
            if all_text_content:
                combined_text = "\n\n".join(all_text_content)
                st.session_state.retrieval_agent.ingest_and_embed(combined_text)
                st.success("✅ Documents processed successfully!")
            else:
                st.error("No text could be extracted from the uploaded documents.")


# --- REVISED CHAT DISPLAY using st.chat_message ---
# This loop now correctly handles and displays the chat history.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        # Render the main content with markdown support
        st.markdown(message["content"])
        
        # Display sources if they exist for an assistant message
        if message["role"] == "assistant" and "sources" in message and message["sources"]:
            with st.expander("🔍 View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.info(f"**Source {i + 1}:**\n\n{doc.page_content}")


# --- User Input & Agent Logic ---
if user_query := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history and display it
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process the query and get the AI response
    with st.spinner("Thinking..."):
        # --- MCP WORKFLOW ---
        trace_id = str(uuid.uuid4())
        
        # 1. Retrieval Request
        retrieval_request = {
            "sender": "Coordinator", "receiver": "RetrievalAgent", "type": "RETRIEVAL_REQUEST",
            "trace_id": trace_id, "payload": {"query": user_query}
        }
        retrieved_context = st.session_state.retrieval_agent.retrieve_context(
            retrieval_request["payload"]["query"]
        )
        
        # 2. Context Response to LLM
        context_response = {
            "sender": "RetrievalAgent", "receiver": "LLMResponseAgent", "type": "CONTEXT_RESPONSE",
            "trace_id": trace_id, "payload": {"top_chunks": retrieved_context, "query": user_query}
        }
        ai_response = LLMResponseAgent(
            query=context_response["payload"]["query"],
            context_chunks=context_response["payload"]["top_chunks"]
        )

    # Add AI response to chat history and display it
    assistant_message = {
        "role": "assistant",
        "content": ai_response,
        "sources": retrieved_context  # Attach the sources here
    }
    st.session_state.chat_history.append(assistant_message)
    
    # Rerun the script to display the new assistant message immediately
    st.rerun()
