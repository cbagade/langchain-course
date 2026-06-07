import os

from dotenv import load_dotenv


load_dotenv()

MODEL_NAME = "ollama:llama3.1:8b"  # "gpt-5-codex"
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2-codex")
EMBEDDING_MODEL = "text-embedding-3-small"
INGESTION_FILE_PATH = os.getenv("INGESTION_FILE_PATH", "data/rca_data_vdb.json")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", os.getenv("INDEX_NAME", "incidents"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
