# AI Knowledge Assistant

## 1. Project Overview

The **AI Knowledge Assistant** is an AI-powered application that allows users to add information from different sources and ask questions about it.

The assistant searches its knowledge base for relevant information and uses an LLM to generate an answer based on the retrieved content.

The project uses **RAG (Retrieval-Augmented Generation)** to improve answers by providing the model with relevant information from the knowledge base.

## 2. How Does It Work?

The system follows these steps:

1. **Add Knowledge:** The user uploads a document, enters a webpage URL, provides a Wikipedia topic, or uploads a WAV audio file.
2. **Extract Text:** The appropriate loader extracts text from the selected source.
3. **Split Documents:** The extracted text is divided into smaller chunks.
4. **Store Knowledge:** The chunks are converted into embeddings and stored in Chroma.
5. **Retrieve Information:** When the user asks a question, the system searches for relevant chunks.
6. **Generate an Answer:** The LLM uses the question and retrieved information to generate an answer.
7. **Evaluate the Answer:** Another LLM call checks whether the answer is supported by the retrieved context.
8. **Improve the Answer:** If the evaluator rejects the answer, its feedback is sent to the generator for another attempt, up to four attempts.

## 3. Project Modules

| File / Module                        | Responsibility                                                                             |
| ------------------------------------ | ------------------------------------------------------------------------------------------ |
| `main.py`                            | Provides the Streamlit interface for testing the application.                              |
| `llm.py`                             | Connects the application to an LLM through OpenRouter.                                     |
| `knowledge_base/document_loader.py`  | Loads supported local documents, including PDF, TXT, Markdown, code files, DOCX, and PPTX. |
| `knowledge_base/url_loader.py`       | Extracts content from webpage URLs.                                                        |
| `knowledge_base/wikipedia_loader.py` | Retrieves information from Wikipedia.                                                      |
| `knowledge_base/audio_loader.py`     | Converts WAV audio into text using Faster Whisper.                                         |
| `knowledge_base/text_splitter.py`    | Splits documents into smaller chunks.                                                      |
| `knowledge_base/ingestion.py`        | Coordinates document splitting and storage.                                                |
| `knowledge_base/vector_store.py`     | Stores and retrieves document chunks using Chroma.                                         |
| `agents/generator.py`                | Generates answers using the user's question and retrieved context.                         |
| `agents/evaluator.py`                | Evaluates generated answers and returns a decision and feedback.                           |
| `agents/feedback_loop.py`            | Repeats generation and evaluation when an answer is rejected, up to four attempts.         |
| `requirements.txt`                   | Contains the Python dependencies required by the project.                                  |

## 4. Technologies Used

- **Python:** Main programming language.
- **Streamlit:** User interface.
- **LangChain:** Integration with language models and document loaders.
- **OpenRouter:** Access to the selected LLM.
- **ChromaDB:** Vector database for storing and searching knowledge.
- **Sentence Transformers:** Generates text embeddings.
- **Faster Whisper:** Transcribes WAV audio.
- **Redis:** Planned caching feature; integration still needs to be verified.

## 5. How to Run the Project

### Step 1: Create a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=your_model_id
```

Replace the example values with your actual OpenRouter API key and a model ID available to your account.

### Step 4: Run the application

```bash
streamlit run main.py
```

Open the local URL displayed in the terminal.

## 6. Current Implementation Status

The following modules are present in the current implementation described by the project code:

- OpenRouter LLM configuration.
- Loading documents from supported file formats.
- Loading webpage and Wikipedia content.
- Transcribing WAV audio.
- Splitting documents into chunks.
- Storing and retrieving information using Chroma.
- Generating answers from retrieved context.
- Evaluating generated answers.
- Retrying rejected answers using evaluator feedback.
- A Streamlit interface for testing these features together.

**Note:** The presence of a module does not guarantee that every feature works correctly. Run and test the application to verify the complete workflow.

## 7. Remaining Work

### Task 1: Implement Redis Caching

Redis is included in the dependencies, but the caching workflow still needs to be implemented or verified.

- [ ] Set up and connect to a Redis server.
- [ ] Decide which results should be cached.
- [ ] Add cache lookup and storage to the appropriate workflow.
- [ ] Configure expiration times.
- [ ] Test cache hits, cache misses, and Redis connection failures.
- [ ] Consider how changes to the knowledge base affect cached answers.

### Task 2: Implement or Verify Conversation Memory

- [ ] Decide what information the assistant should remember between messages.
- [ ] Store and retrieve conversation history using appropriate conversation IDs.
- [ ] Decide whether different agents require separate memory.
- [ ] Keep conversation history separate from verified knowledge-base sources.
- [ ] Add a way to clear conversation history if needed.

### Task 3: Improve the Streamlit Interface

- [ ] Display clear loading and error messages.
- [ ] Make retrieved documents and their sources easy to inspect.
- [ ] Show evaluator feedback when an answer is rejected.
- [ ] Provide a way to add missing or corrected knowledge if required.
- [ ] Test large files and repeated uploads.
- [ ] Improve the interface based on user testing.

## 8. Important Notes for the Team

- **Understand the existing code before changing it.** Avoid rewriting modules that already work.
- **Work on one task at a time.** Test each change before combining it with other work.
- **Keep function names and parameters consistent.** If a function changes, update the modules that call it.
- **Test both known and unknown questions.** The assistant should not confidently invent information that is missing from the knowledge base.
- **Do not assume Redis caching or conversation memory is complete** until it has been connected to the application and tested.
- **Avoid duplicate work.** Coordinate with teammates before implementing a feature another person may already be working on.

## 9. Project Summary

The main goal is to build a knowledge assistant that can ingest information from different sources, retrieve relevant content, generate answers, and evaluate those answers before returning them to the user.

The main next steps are to **test the existing modules, complete or verify Redis caching, complete or verify conversation memory, improve the interface, and prepare the project for deployment**.

Update this README as tasks are completed so that it always reflects the actual state of the project.
