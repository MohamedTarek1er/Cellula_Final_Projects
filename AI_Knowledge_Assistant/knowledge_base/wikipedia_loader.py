from langchain_community.document_loaders import WikipediaLoader
from langchain_core.documents import Document
import wikipedia

wikipedia.set_user_agent(
    "AIKnowledgeAssistant/1.0 (contact: mohamedeleraky1907@gmail.com)"
)

def load_wikipedia(topic: str) -> list[Document]:
    topic = topic.strip()

    if not topic:
        raise ValueError("Wikipedia topic cannot be empty.")

    loader = WikipediaLoader(
        query=topic,
        load_max_docs=1,
        doc_content_chars_max=10000,
    )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = document.metadata.get(
            "source",
            "Wikipedia",
        )
        document.metadata["file_type"] = "wikipedia"

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]
