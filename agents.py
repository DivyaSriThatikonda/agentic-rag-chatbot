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
from pinecone import Pinecone

# Load environment variables from .env file
load_dotenv()

# Agent 1: IngestionAgent 
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
                row_str = ", ".join([f"{col}: {val}" for col, val in row.dropna().items()])
                sentences.append(f"Row {index+1} of the CSV contains the following data: {row_str}.")
            documents_text = "\n".join(sentences)
        elif file_extension in ["txt", "md"]:
            with open(temp_file_path, "r", encoding="utf-8") as f:
                documents_text = f.read()
        else:
            print(f"Unsupported file type: {file_extension}")
            return None

    except Exception as e:
        print(f"Error processing file {uploaded_file.name}: {e}")
        return None
    finally:
        os.remove(temp_file_path)

    return documents_text

# Agent 2: RetrievalAgent 
class RetrievalAgent:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initializes the agent with an embedding model and a Pinecone index."""
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.pinecone_index_name = "agentic-rag-chatbot"
        
        # Initialize Pinecone client
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.pinecone_index = self.pinecone.Index(self.pinecone_index_name)

        self.vector_store = PineconeVectorStore(
            index_name=self.pinecone_index_name,
            embedding=self.embeddings
        )
        print("Pinecone vector store initialized.")

    def clear_index(self):
        """Deletes all vectors from the Pinecone index to start a fresh session."""
        print("Clearing all vectors from the Pinecone index...")
        self.pinecone_index.delete(delete_all=True)
        print("Index cleared successfully.")

    def ingest_and_embed(self, text_content: str):
        """Clears the index, then splits text, creates embeddings, and stores them in Pinecone."""
        self.clear_index() # <-- THIS IS THE CRUCIAL STEP
        
        if not text_content:
            print("No text content to ingest.")
            return

        chunks = self.text_splitter.split_text(text_content)
        self.vector_store.add_texts(chunks)
        print(f"Successfully embedded and stored {len(chunks)} chunks in Pinecone.")

    def retrieve_context(self, query: str):
        """Retrieves relevant context chunks for a given query from Pinecone."""
        print(f"Retrieving context for query: '{query}'")
        retrieved_docs = self.vector_store.similarity_search(query, k=5)
        return retrieved_docs

# Agent 3: LLMResponseAgent
def LLMResponseAgent(query: str, context_chunks: list):
    """
    Forms a final prompt using retrieved context and generates an answer using an LLM.
    """
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
    llm = ChatOpenAI(
        model="x-ai/grok-4-fast:free",
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1"
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context_str, "question": query})
    return response





