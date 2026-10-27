from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    UnstructuredPowerPointLoader,
)

SUPPORTED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".py", ".js",
    ".html", ".css", ".json", ".csv",
    ".docx", ".pptx",
}

def load_document(file_path: str) -> list[Document]:

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {extension}")

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    elif extension == ".pptx":
        loader = UnstructuredPowerPointLoader(str(path))

    else:
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = str(path)
        document.metadata["file_type"] = extension

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]