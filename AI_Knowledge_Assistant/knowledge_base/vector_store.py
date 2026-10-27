import hashlib
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vector_store() -> Chroma:
    return Chroma(
        collection_name="knowledge_base",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

def store_chunks(chunks: list[Document]) -> int:
    chunks = [
        chunk for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise ValueError("No valid chunks were provided for storage.")

    vector_store = get_vector_store()
    chunk_ids = []

    for index, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        start_index = chunk.metadata.get("start_index", index)

        unique_text = f"{source}|{start_index}|{chunk.page_content}"
        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    return len(chunks)

def search_documents(query: str, k: int = 4) -> list[Document]:
    if k < 1:
        raise ValueError("k must be at least 1.")

    vector_store = get_vector_store()

    return vector_store.similarity_search(
        query=query.strip(),
        k=k,
    )
