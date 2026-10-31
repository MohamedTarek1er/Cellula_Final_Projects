import logging
import uuid
from pathlib import Path

import streamlit as st

from agents.feedback_loop import run_feedback_loop, clear_session_memory
from knowledge_base.audio_loader import load_wav
from knowledge_base.document_loader import load_document
from knowledge_base.ingestion import ingest_documents
from knowledge_base.url_loader import load_url
from knowledge_base.vector_store import search_documents
from knowledge_base.wikipedia_loader import load_wikipedia

# Create a folder for uploaded files.
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = [
    "pdf", "txt", "md", "py", "js", "html",
    "css", "json", "csv", "docx", "pptx"
]

st.set_page_config(
    page_title="AI Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

logging.basicConfig(level=logging.INFO)

# Streamlit re-runs this script on every click; session_state survives that.
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []

st.title("📚 AI Knowledge Assistant")
st.write(
    "Upload knowledge, store it in Chroma, and ask questions. "
    "Answers are checked by an Evaluator before they are returned."
)


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def save_uploaded_file(uploaded_file):
    """Save an uploaded file locally and return its path."""
    file_path = UPLOAD_DIR / Path(uploaded_file.name).name
    file_path.write_bytes(uploaded_file.getvalue())
    return str(file_path)


def ingest_and_report(documents):
    """Store loaded documents in the knowledge base."""
    chunk_count = ingest_documents(documents)
    st.success(f"Knowledge stored successfully. Chunks stored: {chunk_count}")


STATUS_BANNERS = {
    "accepted": (st.success, "✅ The evaluator accepted this answer."),
    "max_iterations": (st.warning, "⚠️ Not validated: maximum attempts reached."),
    "no_knowledge": (st.info, "ℹ️ No relevant information was found in the knowledge base."),
    "evaluation_failed": (st.warning, "⚠️ Not validated: the evaluator had a technical problem."),
    "error": (st.error, "❌ A technical error occurred."),
}


def render_result(result: dict):
    """Show the answer, status banner, sources and evaluator feedback."""
    st.write("### Answer")
    st.write(result.get("answer", "No answer was generated."))

    banner, default_text = STATUS_BANNERS.get(
        result.get("status"), (st.warning, "Unknown status.")
    )
    banner(result.get("disclaimer") or default_text)

    st.write("Iterations:", result.get("iterations", 0))

    sources = result.get("sources", [])
    if sources:
        st.write("### Sources")
        for source in sources:
            st.write(f"- {source}")

    history = result.get("history", [])
    if history:
        with st.expander("Evaluator feedback and attempts"):
            for item in history:
                st.markdown(f"**Attempt {item['attempt']}: {item['decision']}**")
                st.write(f"Evaluator feedback: {item.get('feedback') or '(none)'}")
                st.caption(item["answer"])
                st.divider()


# --------------------------------------------------
# Sidebar: conversation memory controls
# --------------------------------------------------

with st.sidebar:
    st.subheader("Conversation")
    st.caption(f"Session: {st.session_state.session_id[:8]}")

    if st.button("🗑️ Clear conversation memory"):
        clear_session_memory(st.session_state.session_id)
        st.session_state.chat = []
        st.success("Generator and Evaluator memory cleared.")


# --------------------------------------------------
# Tabs
# --------------------------------------------------

file_tab, url_tab, wiki_tab, audio_tab, question_tab = st.tabs([
    "📁 Upload Files",
    "🌐 Web URL",
    "📖 Wikipedia",
    "🎙️ WAV Audio",
    "💬 Ask Questions"
])


# --------------------------------------------------
# 1. Upload local files
# --------------------------------------------------

with file_tab:
    st.subheader("Upload Knowledge Files")

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
        key="local_files"
    )

    if st.button("Load and Store Files", key="ingest_files"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            documents = []

            try:
                with st.spinner("Loading files..."):
                    for uploaded_file in uploaded_files:
                        file_path = save_uploaded_file(uploaded_file)
                        loaded = load_document(file_path)
                        documents.extend(loaded)

                if documents:
                    ingest_and_report(documents)
                    st.write(f"Documents loaded: {len(documents)}")
                else:
                    st.warning("No readable content was found.")

            except Exception as error:
                st.error(f"File ingestion failed: {error}")


# --------------------------------------------------
# 2. Load a webpage
# --------------------------------------------------

with url_tab:
    st.subheader("Load a Webpage")

    url = st.text_input(
        "Enter the webpage URL",
        placeholder="https://example.com"
    )

    if st.button("Load and Store URL"):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                with st.spinner("Loading webpage..."):
                    documents = load_url(url.strip())
                    ingest_and_report(documents)
            except Exception as error:
                st.error(f"URL ingestion failed: {error}")


# --------------------------------------------------
# 3. Load Wikipedia
# --------------------------------------------------

with wiki_tab:
    st.subheader("Load a Wikipedia Topic")

    topic = st.text_input(
        "Enter a Wikipedia topic",
        placeholder="Python programming"
    )

    if st.button("Load and Store Wikipedia"):
        if not topic.strip():
            st.warning("Please enter a topic.")
        else:
            try:
                with st.spinner("Loading Wikipedia..."):
                    documents = load_wikipedia(topic.strip())
                    ingest_and_report(documents)
            except Exception as error:
                st.error(f"Wikipedia ingestion failed: {error}")


# --------------------------------------------------
# 4. Load WAV audio
# --------------------------------------------------

with audio_tab:
    st.subheader("Transcribe and Store Audio")

    audio_file = st.file_uploader(
        "Upload a WAV file",
        type=["wav"],
        key="wav_file"
    )

    if st.button("Transcribe and Store Audio"):
        if audio_file is None:
            st.warning("Please upload a WAV file.")
        else:
            try:
                audio_path = save_uploaded_file(audio_file)

                with st.spinner("Transcribing audio..."):
                    documents = load_wav(audio_path)

                if documents:
                    st.write("### Transcription")
                    st.write(documents[0].page_content)
                    ingest_and_report(documents)
                else:
                    st.warning("No transcription was produced.")

            except Exception as error:
                st.error(f"Audio ingestion failed: {error}")


# --------------------------------------------------
# 5. Retrieve knowledge and ask questions
# --------------------------------------------------

with question_tab:
    st.subheader("Ask a Question")

    question = st.text_area(
        "Your question",
        placeholder="Ask something related to your uploaded knowledge..."
    )

    if st.button("Search Knowledge and Generate Answer"):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching the knowledge base..."):
                    retrieved_docs = search_documents(
                        question.strip(),
                        k=4
                    )

                st.write("### Retrieved Documents")

                if retrieved_docs:
                    for index, document in enumerate(retrieved_docs, start=1):
                        with st.expander(f"Document {index}"):
                            st.write("Source:", document.metadata.get("source", "Unknown"))
                            st.write(document.page_content)
                else:
                    st.info("No documents were retrieved.")

                with st.spinner("Generating and evaluating answer..."):
                    result = run_feedback_loop(
                        question.strip(),
                        session_id=st.session_state.session_id,
                    )

                st.session_state.chat.append(
                    {"question": question.strip(), "result": result}
                )
                render_result(result)

            except Exception as error:
                st.error(f"Question answering failed: {error}")

    # Earlier questions of this conversation (outside the button block,
    # so it is still visible after Streamlit re-runs).
    if len(st.session_state.chat) > 1:
        with st.expander("Previous questions in this conversation"):
            for turn in reversed(st.session_state.chat[:-1]):
                st.markdown(f"**You:** {turn['question']}")
                st.write(turn["result"].get("answer", ""))
                st.divider()