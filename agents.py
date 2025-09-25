# import pandas as pd
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
#
# # Agent 1: IngestionAgent
# def IngestionAgent(uploaded_file):
#     """
#     Parses and preprocesses uploaded documents of various formats.
#     Returns the extracted text content as a single string.
#     """
#     # Get the file extension
#     file_extension = uploaded_file.name.split('.')[-1].lower()
#
#     # Create a temporary file path to save the uploaded file
#     temp_file_path = uploaded_file.name
#     with open(temp_file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#
#     documents_text = ""
#     try:
#         if file_extension == "pdf":
#             loader = PyPDFLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "docx":
#             loader = Docx2txtLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "pptx":
#             loader = UnstructuredPowerPointLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "csv":
#             df = pd.read_csv(temp_file_path)
#             documents_text = df.to_string()
#         elif file_extension in ["txt", "md"]:
#             with open(temp_file_path, "r", encoding="utf-8") as f:
#                 documents_text = f.read()
#         else:
#             return None # Unsupported file type
#
#     except Exception as e:
#         print(f"Error processing file {uploaded_file.name}: {e}")
#         return None
#
#     return documents_text
#
# import os
# from dotenv import load_dotenv
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
#
# # Load environment variables from .env file
# load_dotenv()
#
# # Agent 2: RetrievalAgent
# class RetrievalAgent:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         """Initializes the agent with an embedding model and a Pinecone index."""
#         self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#
#         # --- Pinecone Initialization ---
#         # NOTE: You must have created an index in your Pinecone project beforehand.
#         # We'll name our index 'agentic-rag-chatbot' for this example.
#         self.pinecone_index_name = "agentic-rag-chatbot"
#
#         # Initialize the PineconeVectorStore. This will be our vector store.
#         # It will create a new index if one doesn't exist, or use an existing one.
#         self.vector_store = PineconeVectorStore(
#             index_name=self.pinecone_index_name,
#             embedding=self.embeddings
#         )
#         print("Pinecone vector store initialized.")
#
#     def ingest_and_embed(self, text_content: str):
#         """Splits text, creates embeddings, and stores them in Pinecone."""
#         if not text_content:
#             print("No text content to ingest.")
#             return
#
#         # Split the text into manageable chunks
#         chunks = self.text_splitter.split_text(text_content)
#
#         # Add the text chunks to the Pinecone index
#         self.vector_store.add_texts(chunks)
#         print(f"Successfully embedded and stored {len(chunks)} chunks in Pinecone.")
#
#     def retrieve_context(self, query: str):
#         """Retrieves relevant context chunks for a given query from Pinecone."""
#         print(f"Retrieving context for query: '{query}'")
#         # Use Pinecone's similarity search to find the most relevant documents
#         retrieved_docs = self.vector_store.similarity_search(query, k=3)
#         return retrieved_docs
#
#
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser
#
# # Agent 3: LLMResponseAgent
# # def LLMResponseAgent(query: str, context_chunks: list):
# #     """
# #     Forms a final prompt using retrieved context and generates an answer using Google Gemini.
# #     """
# #     if not context_chunks:
# #         return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."
# #
# #     # Format the retrieved context chunks into a single string
# #     context_str = "\n".join([chunk.page_content for chunk in context_chunks])
# #
# #     # Create a prompt template
# #     template = """
# #     You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
# #     If the answer is not found in the context, respond with "I'm sorry, I couldn't find the answer in the provided documents."
# #
# #     Context:
# #     {context}
# #
# #     Question: {question}
# #
# #     Answer:
# #     """
# #
# #     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# #
# #     # Initialize the LLM (Google Gemini Pro)
# #     llm = ChatGoogleGenerativeAI(
# #         model="gemini-pro",
# #         google_api_key=os.getenv("GOOGLE_API_KEY")
# #     )
# #
# #     # Create the processing chain
# #     chain = prompt | llm | StrOutputParser()
# #
# #     # Invoke the chain with the query and context
# #     response = chain.invoke({"context": context_str, "question": query})
# #
# #     return response
#
# # from langchain_openai import ChatOpenAI
# # from langchain.prompts import PromptTemplate
# # from langchain.schema.output_parser import StrOutputParser
# #
# # # Agent 3: LLMResponseAgent
# # def LLMResponseAgent(query: str, context_chunks: list):
# #     """
# #     Forms a final prompt using retrieved context and generates an answer using Groq via OpenRouter.
# #     """
# #     if not context_chunks:
# #         return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."
# #
# #     context_str = "\n".join([chunk.page_content for chunk in context_chunks])
# #
# #     template = """
# #     You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
# #     If the answer is not found in the context, respond with "I'm sorry, I couldn't find the answer in the provided documents."
# #
# #     Context:
# #     {context}
# #
# #     Question: {question}
# #
# #     Answer:
# #     """
# #
# #     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# #
# #     # Initialize the LLM (Groq Llama3 70b via OpenRouter)
# #     llm = ChatOpenAI(
# #         model="groq/llama-3-8b-8192", # Model name from OpenRouter
# #         api_key=os.getenv("OPENROUTER_API_KEY"),
# #         base_url="https://openrouter.ai/api/v1",
# #         default_headers={
# #             "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL"),
# #         }
# #     )
# #
# #     chain = prompt | llm | StrOutputParser()
# #
# #     response = chain.invoke({"context": context_str, "question": query})
# #
# #     return response
#
# # import os
# # from langchain_openai import ChatOpenAI
# # from langchain.prompts import PromptTemplate
# # from langchain.schema.output_parser import StrOutputParser
# #
# #
# # # Agent 3: LLMResponseAgent
# # def LLMResponseAgent(query: str, context_chunks: list):
# #     """
# #     Forms a final prompt using retrieved context and generates an answer using Grok-4-Fast via OpenRouter.
# #     """
# #     if not context_chunks:
# #         return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."
# #
# #     # Format the retrieved context chunks into a single string
# #     context_str = "\n".join([chunk.page_content for chunk in context_chunks])
# #
# #     # Create a prompt template
# #     template = """
# #     You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
# #     If the answer is not found in the context, respond with "I'm sorry, I couldn't find the answer in the provided documents."
# #
# #     Context:
# #     {context}
# #
# #     Question: {question}
# #
# #     Answer:
# #     """
# #
# #     prompt = PromptTemplate(template=template, input_variables=["context", "question"])
# #
# #     # Initialize the LLM with the correct model ID for Grok-4-Fast
# #     llm = ChatOpenAI(
# #         model="x-ai/grok-4-fast:free",
# #         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
# #         openai_api_base="https://openrouter.ai/api/v1"
# #     )
# #
# #     # Create the processing chain
# #     chain = prompt | llm | StrOutputParser()
# #
# #     # Invoke the chain with the query and context
# #     response = chain.invoke({"context": context_str, "question": query})
# #
# #     return response
#
#
# import streamlit as st
# import uuid
# from agents import IngestionAgent, RetrievalAgent, LLMResponseAgent
#
# # --- Page Config ---
# st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
#
# # --- NEW: Claude-style CSS ---
# st.markdown("""
# <style>
# /* Core body and font styles */
# body {
#     background-color: #f8f7f5; /* Light beige background */
#     font-family: 'Helvetica Neue', sans-serif;
# }
#
# /* Header Box */
# .header-box {
#     background: #ffffff;
#     border: 1px solid #e0e0e0;
#     padding: 20px;
#     border-radius: 12px;
#     text-align: center;
#     font-size: 26px;
#     font-weight: 600;
#     color: #1f1f1f; /* Dark gray text */
#     margin-bottom: 25px;
#     box-shadow: 0 2px 8px rgba(0,0,0,0.05);
# }
#
# /* Chat Bubbles */
# .chat-bubble-user {
#     background-color: #6251dd; /* Claude's signature purple */
#     color: white;
#     padding: 12px 18px;
#     border-radius: 20px 20px 4px 20px; /* Slightly adjusted radius */
#     margin: 6px 0;
#     max-width: 65%;
#     float: right;
#     clear: both;
#     line-height: 1.5;
# }
# .chat-bubble-assistant {
#     background-color: #ffffff;
#     color: #1f1f1f;
#     border: 1px solid #e0e0e0;
#     padding: 12px 18px;
#     border-radius: 20px 20px 20px 4px;
#     margin: 6px 0;
#     max-width: 75%;
#     float: left;
#     clear: both;
#     box-shadow: 0 1px 4px rgba(0,0,0,0.05);
#     line-height: 1.5;
# }
#
# /* Expander for sources */
# .stExpander {
#     border: none !important;
#     box-shadow: none !important;
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
#                 # IngestionAgent now returns text and cleans up its temp file
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
# st.markdown("<div class='header-box'>🤖 Agentic RAG Chatbot</div>", unsafe_allow_html=True)
#
# # --- Intro Message ---
# st.markdown("""
# <div style="text-align: center; margin-bottom: 20px;">
#     <b>How it works:</b> Drop your documents in the sidebar, then ask questions about them!
# </div>
# """, unsafe_allow_html=True)
#
# # --- Chat Section ---
# chat_container = st.container()
# with chat_container:
#     # We store the full message object now, including sources
#     for message in st.session_state.chat_history:
#         role = message["role"]
#         content = message["content"]
#         st.markdown(f"<div class='chat-bubble-{role}'>{content}</div>", unsafe_allow_html=True)
#         # If the message is from the assistant and has sources, display them
#         if role == "assistant" and "sources" in message:
#             with st.expander("🔍 View Sources"):
#                 for i, doc in enumerate(message["sources"]):
#                     st.info(f"**Source {i + 1}:**\n\n{doc.page_content}")
#
# # --- User Input & Agent Logic ---
# if user_query := st.chat_input("Ask a question about your documents..."):
#     # Append user message to history and display it
#     st.session_state.chat_history.append({"role": "user", "content": user_query})
#     with chat_container:
#         st.markdown(f"<div class='chat-bubble-user'>{user_query}</div>", unsafe_allow_html=True)
#
#     with st.spinner("Thinking..."):
#         # --- MCP WORKFLOW ---
#         trace_id = str(uuid.uuid4())
#
#         # 1. Coordinator -> RetrievalAgent
#         retrieval_request = {
#             "sender": "Coordinator", "receiver": "RetrievalAgent", "type": "RETRIEVAL_REQUEST",
#             "trace_id": trace_id, "payload": {"query": user_query}
#         }
#         retrieved_context = st.session_state.retrieval_agent.retrieve_context(
#             retrieval_request["payload"]["query"]
#         )
#
#         # 2. RetrievalAgent -> LLMResponseAgent
#         context_response = {
#             "sender": "RetrievalAgent", "receiver": "LLMResponseAgent", "type": "CONTEXT_RESPONSE",
#             "trace_id": trace_id, "payload": {"top_chunks": retrieved_context, "query": user_query}
#         }
#         ai_response = LLMResponseAgent(
#             query=context_response["payload"]["query"],
#             context_chunks=context_response["payload"]["top_chunks"]
#         )
#
#     # Create the full assistant message with sources
#     assistant_message = {
#         "role": "assistant",
#         "content": ai_response,
#         "sources": retrieved_context  # Store the sources
#     }
#     st.session_state.chat_history.append(assistant_message)
#
#     # Rerun the script to display the new message and sources
#     st.rerun()

### correct code
# import os
# import tempfile
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser

# # Load environment variables from .env file
# load_dotenv()


# # Agent 1: IngestionAgent (Updated for safer file handling)
# def IngestionAgent(uploaded_file):
#     """
#     Parses documents by writing them to a temporary file and then cleaning up.
#     Returns the extracted text content as a single string.
#     """
#     file_extension = uploaded_file.name.split('.')[-1].lower()

#     # Use tempfile to create a secure temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
#         temp_file.write(uploaded_file.getbuffer())
#         temp_file_path = temp_file.name

#     documents_text = ""
#     try:
#         # Process the file based on its extension
#         if file_extension == "pdf":
#             loader = PyPDFLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "docx":
#             loader = Docx2txtLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "pptx":
#             loader = UnstructuredPowerPointLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "csv":
#             df = pd.read_csv(temp_file_path)
#             documents_text = df.to_string()
#         elif file_extension in ["txt", "md"]:
#             with open(temp_file_path, "r", encoding="utf-8") as f:
#                 documents_text = f.read()
#         else:
#             print(f"Unsupported file type: {file_extension}")
#             return None

#     except Exception as e:
#         print(f"Error processing file {uploaded_file.name}: {e}")
#         return None
#     finally:
#         # Ensure the temporary file is always removed
#         os.remove(temp_file_path)

#     return documents_text


# # Agent 2: RetrievalAgent
# class RetrievalAgent:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         """Initializes the agent with an embedding model and a Pinecone index."""
#         self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#         # --- Pinecone Initialization ---
#         self.pinecone_index_name = "agentic-rag-chatbot"

#         self.vector_store = PineconeVectorStore(
#             index_name=self.pinecone_index_name,
#             embedding=self.embeddings
#         )
#         print("Pinecone vector store initialized.")

#     def ingest_and_embed(self, text_content: str):
#         """Splits text, creates embeddings, and stores them in Pinecone."""
#         if not text_content:
#             print("No text content to ingest.")
#             return

#         chunks = self.text_splitter.split_text(text_content)
#         self.vector_store.add_texts(chunks)
#         print(f"Successfully embedded and stored {len(chunks)} chunks in Pinecone.")

#     def retrieve_context(self, query: str):
#         """Retrieves relevant context chunks for a given query from Pinecone."""
#         print(f"Retrieving context for query: '{query}'")
#         retrieved_docs = self.vector_store.similarity_search(query, k=7)
#         return retrieved_docs


# # Agent 3: LLMResponseAgent
# def LLMResponseAgent(query: str, context_chunks: list):
#     """
#     Forms a final prompt using retrieved context and generates an answer using an LLM via OpenRouter.
#     """
#     if not context_chunks:
#         return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."

#     context_str = "\n".join([chunk.page_content for chunk in context_chunks])

#     template = """
#     You are a helpful AI assistant. Answer the user's question based ONLY on the following context.
#     If the answer is not found in the context, respond with "I'm sorry, I couldn't find the answer in the provided documents."

#     Context:
#     {context}

#     Question: {question}

#     Answer:
#     """

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#     llm = ChatOpenAI(
#         model="x-ai/grok-4-fast:free",
#         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#         openai_api_base="https://openrouter.ai/api/v1"
#     )

#     chain = prompt | llm | StrOutputParser()

#     response = chain.invoke({"context": context_str, "question": query})


#     return response


# import os
# import tempfile
# import pandas as pd
# from dotenv import load_dotenv
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langchain_openai import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.schema.output_parser import StrOutputParser

# # Load environment variables from .env file
# load_dotenv()


# # # Agent 1: IngestionAgent
# # def IngestionAgent(uploaded_file):
# #     """
# #     Parses documents by writing them to a temporary file and then cleaning up.
# #     Returns the extracted text content as a single string.
# #     """
# #     file_extension = uploaded_file.name.split('.')[-1].lower()

# #     # Use tempfile to create a secure temporary file
# #     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
# #         temp_file.write(uploaded_file.getbuffer())
# #         temp_file_path = temp_file.name

# #     documents_text = ""
# #     try:
# #         # Process the file based on its extension
# #         if file_extension == "pdf":
# #             loader = PyPDFLoader(temp_file_path)
# #             documents = loader.load()
# #             documents_text = "\n".join([doc.page_content for doc in documents])
# #         elif file_extension == "docx":
# #             loader = Docx2txtLoader(temp_file_path)
# #             documents = loader.load()
# #             documents_text = "\n".join([doc.page_content for doc in documents])
# #         elif file_extension == "pptx":
# #             loader = UnstructuredPowerPointLoader(temp_file_path)
# #             documents = loader.load()
# #             documents_text = "\n".join([doc.page_content for doc in documents])
# #         elif file_extension == "csv":
# #             df = pd.read_csv(temp_file_path)
# #             documents_text = df.to_string()
# #         elif file_extension in ["txt", "md"]:
# #             with open(temp_file_path, "r", encoding="utf-8") as f:
# #                 documents_text = f.read()
# #         else:
# #             print(f"Unsupported file type: {file_extension}")
# #             return None

# #     except Exception as e:
# #         print(f"Error processing file {uploaded_file.name}: {e}")
# #         return None
# #     finally:
# #         # Ensure the temporary file is always removed
# #         os.remove(temp_file_path)

# #     return documents_text

# def IngestionAgent(uploaded_file):
#     """
#     Parses documents by writing them to a temporary file and then cleaning up.
#     Returns the extracted text content as a single string.
#     """
#     file_extension = uploaded_file.name.split('.')[-1].lower()

#     with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
#         temp_file.write(uploaded_file.getbuffer())
#         temp_file_path = temp_file.name

#     documents_text = ""
#     try:
#         if file_extension == "pdf":
#             loader = PyPDFLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "docx":
#             loader = Docx2txtLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
#         elif file_extension == "pptx":
#             loader = UnstructuredPowerPointLoader(temp_file_path)
#             documents = loader.load()
#             documents_text = "\n".join([doc.page_content for doc in documents])
        
#         # --- NEW & IMPROVED CSV HANDLING ---
#         elif file_extension == "csv":
#             df = pd.read_csv(temp_file_path)
#             # Convert each row into a descriptive sentence
#             sentences = []
#             for index, row in df.iterrows():
#                 row_str = ", ".join([f"{col}: {val}" for col, val in row.dropna().items()])
#                 sentences.append(f"Row {index+1} contains the following data: {row_str}.")
#             documents_text = "\n".join(sentences)
#         # --- END OF CSV HANDLING UPDATE ---

#         elif file_extension in ["txt", "md"]:
#             with open(temp_file_path, "r", encoding="utf-8") as f:
#                 documents_text = f.read()
#         else:
#             print(f"Unsupported file type: {file_extension}")
#             return None

#     except Exception as e:
#         print(f"Error processing file {uploaded_file.name}: {e}")
#         return None
#     finally:
#         os.remove(temp_file_path)

#     return documents_text


# # Agent 2: RetrievalAgent
# class RetrievalAgent:
#     def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#         """Initializes the agent with an embedding model and a Pinecone index."""
#         self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
#         self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         self.pinecone_index_name = "agentic-rag-chatbot"

#         self.vector_store = PineconeVectorStore(
#             index_name=self.pinecone_index_name,
#             embedding=self.embeddings
#         )
#         print("Pinecone vector store initialized.")

#     def ingest_and_embed(self, text_content: str):
#         """Splits text, creates embeddings, and stores them in Pinecone."""
#         if not text_content:
#             print("No text content to ingest.")
#             return

#         chunks = self.text_splitter.split_text(text_content)
#         self.vector_store.add_texts(chunks)
#         print(f"Successfully embedded and stored {len(chunks)} chunks in Pinecone.")

#     def retrieve_context(self, query: str):
#         """Retrieves relevant context chunks for a given query from Pinecone."""
#         print(f"Retrieving context for query: '{query}'")
#         # --- CHANGE 1: Increased k from 3 to 5 to retrieve more context ---
#         retrieved_docs = self.vector_store.similarity_search(query, k=5)
#         return retrieved_docs


# # Agent 3: LLMResponseAgent
# def LLMResponseAgent(query: str, context_chunks: list):
#     """
#     Forms a final prompt using retrieved context and generates an answer using an LLM.
#     """
#     if not context_chunks:
#         return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."

#     context_str = "\n".join([chunk.page_content for chunk in context_chunks])

#     # --- CHANGE 2: Replaced the old prompt with the new stricter one ---
#     template = """
#     **ROLE:** You are a hyper-focused information extraction engine.

#     **TASK:** Your sole purpose is to find and extract the answer to the QUESTION from the provided CONTEXT.

#     **RULES:**
#     1.  You must analyze the CONTEXT meticulously. Your answer must be extracted directly from this text and nothing else.
#     2.  Your response must be the answer itself, direct and concise.
#     3.  **DO NOT** provide any explanations, interpretations, summaries, or conversational filler.
#     4.  If the answer is not explicitly stated in the CONTEXT, you **MUST** respond with the exact phrase: "I'm sorry, I couldn't find the answer in the provided documents."

#     ---
#     **CONTEXT:**
#     {context}
#     ---
#     **QUESTION:**
#     {question}
#     ---
#     **ANSWER:**
#     """

#     prompt = PromptTemplate(template=template, input_variables=["context", "question"])

#     llm = ChatOpenAI(
#         model="x-ai/grok-4-fast:free",
#         openai_api_key=os.getenv("OPENROUTER_API_KEY"),
#         openai_api_base="https://openrouter.ai/api/v1"
#     )

#     chain = prompt | llm | StrOutputParser()
#     response = chain.invoke({"context": context_str, "question": query})

#     return response


import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

def IngestionAgent(uploaded_file):
    """
    Parses documents by writing them to a temporary file and then cleaning up.
    Returns the extracted text content as a single string.
    """
    file_extension = uploaded_file.name.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name
    documents_text = ""
    try:
        if file_extension == "pdf":
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            documents_text = "\n".join([doc.page_content for doc in documents])
        elif file_extension == "docx":
            loader = Docx2txtLoader(temp_file_path)
            documents = loader.load()
            documents_text = "\n".join([doc.page_content for doc in documents])
        elif file_extension == "pptx":
            loader = UnstructuredPowerPointLoader(temp_file_path)
            documents = loader.load()
            documents_text = "\n".join([doc.page_content for doc in documents])
        elif file_extension == "csv":
            df = pd.read_csv(temp_file_path)
            sentences = []
            for index, row in df.iterrows():
                row_str = ", ".join([f"{col.strip()}: {str(val).strip()}" for col, val in row.dropna().items()])
                if row_str:
                    sentences.append(f"Row {index + 1} of the CSV contains the following data: {row_str}.")
            documents_text = "\n".join(sentences)
        elif file_extension in ["txt", "md"]:
            with open(temp_file_path, "r", encoding="utf-8") as f:
                documents_text = f.read()
    except Exception as e:
        print(f"Error processing file {uploaded_file.name}: {e}")
        return None
    finally:
        os.remove(temp_file_path)
    return documents_text

class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.pinecone_index_name = "agentic-rag-chatbot"
        self.vector_store = PineconeVectorStore(index_name=self.pinecone_index_name, embedding=self.embeddings)
        print("Pinecone vector store initialized.")

    def clear_index(self):
        """Deletes all vectors from the Pinecone index."""
        self.vector_store.delete(delete_all=True)
        print("Pinecone index cleared successfully.")

    def ingest_and_embed(self, text_content: str):
        """Clears the index, then splits text, creates embeddings, and stores them."""
        if not text_content:
            print("No text content to ingest.")
            return
        
        # --- CRITICAL STEP: WIPE THE OLD DATA ---
        self.clear_index()
        
        chunks = self.text_splitter.split_text(text_content)
        self.vector_store.add_texts(chunks)
        print(f"Successfully embedded and stored {len(chunks)} new chunks.")

    def retrieve_context(self, query: str):
        """Retrieves relevant context chunks for a given query from Pinecone."""
        print(f"Retrieving context for query: '{query}'")
        retrieved_docs = self.vector_store.similarity_search(query, k=5)
        return retrieved_docs

def LLMResponseAgent(query: str, context_chunks: list):
    """Forms a final prompt using retrieved context and generates an answer using an LLM."""
    if not context_chunks:
        return "I'm sorry, I couldn't find any relevant information in the documents to answer your question."
    context_str = "\n".join([chunk.page_content for chunk in context_chunks])
    template = """
    **ROLE:** You are a hyper-focused information extraction engine.
    **TASK:** Your sole purpose is to find and extract the answer to the QUESTION from the provided CONTEXT.
    **RULES:**
    1.  You must analyze the CONTEXT meticulously. Your answer must be extracted directly from this text and nothing else.
    2.  Your response must be the answer itself, direct and concise.
    3.  **DO NOT** provide any explanations, interpretations, summaries, or conversational filler.
    4.  If the answer is not explicitly stated in the CONTEXT, you **MUST** respond with the exact phrase: "I'm sorry, I couldn't find the answer in the provided documents."
    ---
    **CONTEXT:**
    {context}
    ---
    **QUESTION:**
    {question}
    ---
    **ANSWER:**
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    llm = ChatOpenAI(model="x-ai/grok-4-fast:free", openai_api_key=os.getenv("OPENROUTER_API_KEY"), openai_api_base="https://openrouter.ai/api/v1")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_str, "question": query})
    return response



