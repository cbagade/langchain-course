import json

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from config import EMBEDDING_MODEL


def _clean_metadata(metadata: dict) -> dict:
    return {key: value for key, value in metadata.items() if value is not None}


def _load_rca_documents(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8") as file:
        records = json.load(file)

    documents = []
    for record in records:
        metadata = {
            **record.get("metadata", {}),
            "id": record.get("id"),
            "product": record.get("product"),
            "cbc": record.get("cbc"),
            "date_occurred": record.get("date_occurred"),
        }
        documents.append(
            Document(
                page_content=record["document"],
                metadata=_clean_metadata(metadata),
            )
        )

    return documents


def ingest_documents(file_path: str, index_name: str, pinecone_api_key: str, openai_api_key: str):
    documents = _load_rca_documents(file_path)
    print(f"Loaded {len(documents)} RCA documents.")

    # Create embeddings for the chunks
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=openai_api_key)

    print("Created embeddings for the RCA documents...")
    # Store the embeddings in Pinecone
    PineconeVectorStore.from_documents(documents, embeddings, pinecone_api_key=pinecone_api_key, index_name=index_name)
