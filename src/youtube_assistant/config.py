import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

GROQ_API_KEY =os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY or GROQ_API_KEY == "":
    raise ValueError("GROQ_API_KEY is missing.")

llm = ChatGroq(model = GROQ_MODEL, api_key = GROQ_API_KEY, temperature = 0.2)

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"