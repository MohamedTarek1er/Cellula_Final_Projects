from langchain_core.documents import Document
from knowledge_base.text_splitter import split_documents
from knowledge_base.vector_store import store_chunks

def ingest_documents(documents: list[Document]) -> int:
    if not documents:
        raise ValueError("No documents to ingest.")

    chunks = split_documents(documents)
    return store_chunks(chunks)