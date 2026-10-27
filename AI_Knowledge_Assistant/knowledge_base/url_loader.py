from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

def load_url(url: str) -> list[Document]:
    url = url.strip()

    if not url:
        raise ValueError("URL cannot be empty.")

    loader = WebBaseLoader(web_paths=(url,))
    documents = loader.load()

    for document in documents:
        document.metadata["source"] = url
        document.metadata["file_type"] = "url"

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]
