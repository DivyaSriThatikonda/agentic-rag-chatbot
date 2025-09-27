# Agentic RAG Chatbot for Multi-Format Document Q&A
This project is a sophisticated, agent-based Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions from a variety of uploaded documents. The architecture is built around a multi-agent system that uses the Model Context Protocol (MCP) for internal communication, ensuring a modular and scalable design.

## 🚀 Core Features
**Multi-Format Document Support:** Ingests and processes a wide range of document types, including .pdf, .pptx, .csv, .docx, and .txt.

**Agentic Architecture:** Utilizes a specialized three-agent system for a clear separation of concerns:

**IngestionAgent:** Handles document parsing and text extraction.

**RetrievalAgent:** Manages text embedding and semantic retrieval from a vector database.

**LLMResponseAgent:** Synthesizes the final answer using retrieved context.

**Model Context Protocol (MCP):** All communication between agents is handled via structured MCP messages, which can be inspected in the UI for demonstration purposes.

**Session-Based Memory:** Each time new documents are processed, the vector database is automatically cleared to ensure answers are based only on the most recently uploaded files.

**Interactive UI:** A user-friendly interface built with Streamlit that allows for easy document uploading, multi-turn conversations, and viewing of source context for each answer.

## 🛠️ Tech Stack

**Application Framework:** Streamlit

**Language:** Python

**Core Libraries:** LangChain, python-dotenv, pandas

**LLM Provider:** OpenRouter.ai

**LLM Model:** x-ai/grok-4-fast:free (A high-performance free model)

**Embeddings Model:** sentence-transformers/all-MiniLM-L6-v2 (from Hugging Face)

**Vector Database:** Pinecone

## ⚙️ Setup and Installation
Follow these steps to set up and run the project locally.

**1. Prerequisites**
   
a. Python 3.9 or higher

b. An active Pinecone account

c. An active OpenRouter.ai account

**2. Clone the Repository**
   
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

**3. Create a Virtual Environment**

It's highly recommended to use a virtual environment to manage dependencies.

**For Windows**

python -m venv .venv
.\.venv\Scripts\activate

**For macOS/Linux**

python3 -m venv .venv
source .venv/bin/activate

**4. Install Dependencies**

Install all the required Python packages using the requirements.txt file.

pip install -r requirements.txt

**5. Set Up Environment Variables**

You need to provide your API keys for Pinecone and OpenRouter.

a. Create a new file named .env in the root of the project.

b. Add the following content to the .env file, replacing the placeholder text with your actual keys:

**Get this from your Pinecone project dashboard**

PINECONE_API_KEY="your_pinecone_api_key_here"

**Get this from your OpenRouter.ai account keys page**

OPENROUTER_API_KEY="sk-or-your_openrouter_api_key_here"

**6. Create a Pinecone Index**

Before running the app for the first time, you must create a vector index in your Pinecone account.

a. Log in to your Pinecone dashboard.
b. Create a new index with the following specifications:
-   Index Name: agentic-rag-chatbot
-   Dimensions: 384 (This is required for the all-MiniLM-L6-v2 model)
-   Metric: cosine

## ▶️ How to Run the Application

Once the setup is complete, run the following command in your terminal from the project's root directory:

streamlit run app.py

## Live app : 

https://agentic-rag-chatbot-m7l2atbtcnj6hwhyrvhgnx.streamlit.app/

## Video:

https://www.loom.com/share/44edd6d0e6a046cb87ed3759477f5837?sid=1c3a976c-27fc-45f5-89b9-fa24e9729368
