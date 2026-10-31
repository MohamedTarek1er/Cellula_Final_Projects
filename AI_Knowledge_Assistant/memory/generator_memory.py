import logging
import threading

logger = logging.getLogger(__name__)

MAX_TURNS = 10          # conversation turns kept per session
MAX_CHARS_PER_ITEM = 700  # keeps prompts from growing too large


def _clip(text: str, limit: int = MAX_CHARS_PER_ITEM) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


class GeneratorMemory:
    """
    Private memory of the Generator ONLY.
    Stores: conversation turns, retrieved context, previous attempts
    and the feedback received for them.
    It never stores Evaluator decisions or rules.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self._max_turns = max_turns
        self._lock = threading.Lock()
        self._turns: dict[str, list[dict]] = {}     # finished questions
        self._attempts: dict[str, list[dict]] = {}  # attempts of current question
        self._context: dict[str, str] = {}          # context of current question

    # ---- current question ----
    def start_question(self, session_id: str, context: str) -> None:
        """Call once per new question: resets attempts, stores the context."""
        with self._lock:
            self._attempts[session_id] = []
            self._context[session_id] = context

    def add_attempt(
        self, session_id: str, attempt: int, answer: str, feedback_received: str = ""
    ) -> None:
        """feedback_received = the feedback that led to this attempt ('' for the first)."""
        with self._lock:
            self._attempts.setdefault(session_id, []).append(
                {
                    "attempt": attempt,
                    "answer": answer,
                    "feedback_received": feedback_received,
                }
            )

    def get_attempts(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._attempts.get(session_id, []))

    def get_context(self, session_id: str) -> str:
        with self._lock:
            return self._context.get(session_id, "")

    # ---- finished questions (conversation history) ----
    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
        sources: list,
        validated: bool,
    ) -> None:
        with self._lock:
            turns = self._turns.setdefault(session_id, [])
            turns.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": sources,
                    "validated": validated,
                }
            )
            # keep only the most recent turns
            del turns[: max(0, len(turns) - self._max_turns)]

    def get_turns(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._turns.get(session_id, []))

    # ---- text blocks for the prompt ----
    def format_conversation(self, session_id: str, last_n: int = 3) -> str:
        turns = self.get_turns(session_id)[-last_n:]
        if not turns:
            return "None. This is the start of the conversation."
        return "\n\n".join(
            f"User: {_clip(t['question'])}\nAssistant: {_clip(t['answer'])}"
            for t in turns
        )

    def format_attempts(self, session_id: str) -> str:
        attempts = self.get_attempts(session_id)
        if not attempts:
            return "None. This is the first attempt."
        return "\n\n".join(
            f"Attempt {a['attempt']}:\n{_clip(a['answer'])}\n"
            f"Feedback received before this attempt: "
            f"{_clip(a['feedback_received']) or 'none'}"
            for a in attempts
        )

    # ---- cleanup ----
    def clear(self, session_id: str) -> None:
        with self._lock:
            self._turns.pop(session_id, None)
            self._attempts.pop(session_id, None)
            self._context.pop(session_id, None)
        logger.info("Generator memory cleared for session %s", session_id)


# One instance for the whole app. Data is separated per session_id.
generator_memory = GeneratorMemory()