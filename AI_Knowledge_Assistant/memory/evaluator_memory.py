import logging
import threading

logger = logging.getLogger(__name__)

MAX_TURNS = 10
MAX_CHARS_PER_ITEM = 700

# Fixed rules the Evaluator always works with (the spec's criteria).
EVALUATION_CRITERIA = [
    "Accuracy",
    "Relevance to the question",
    "Completeness",
    "Consistency with the external knowledge",
    "Grounding in retrieved information",
    "No unsupported claims",
    "Overall answer quality",
]


def _clip(text: str, limit: int = MAX_CHARS_PER_ITEM) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "..."


class EvaluatorMemory:
    """
    Private memory of the Evaluator ONLY.
    Stores: evaluation history, decisions and the feedback it gave.
    It never stores the Generator's conversation or its reasoning.
    """

    def __init__(self, max_turns: int = MAX_TURNS):
        self._max_turns = max_turns
        self._lock = threading.Lock()
        self._evaluations: dict[str, list[dict]] = {}  # current question
        self._past: dict[str, list[dict]] = {}         # finished questions

    @property
    def criteria(self) -> list[str]:
        return list(EVALUATION_CRITERIA)

    # ---- current question ----
    def start_question(self, session_id: str) -> None:
        with self._lock:
            self._evaluations[session_id] = []

    def add_evaluation(
        self, session_id: str, attempt: int, decision: str, feedback: str
    ) -> None:
        with self._lock:
            self._evaluations.setdefault(session_id, []).append(
                {"attempt": attempt, "decision": decision, "feedback": feedback}
            )

    def get_evaluations(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._evaluations.get(session_id, []))

    # ---- finished questions ----
    def add_turn_summary(
        self, session_id: str, question: str, final_decision: str, iterations: int
    ) -> None:
        with self._lock:
            past = self._past.setdefault(session_id, [])
            past.append(
                {
                    "question": question,
                    "final_decision": final_decision,
                    "iterations": iterations,
                }
            )
            del past[: max(0, len(past) - self._max_turns)]

    def get_past(self, session_id: str) -> list[dict]:
        with self._lock:
            return list(self._past.get(session_id, []))

    # ---- text block for the prompt ----
    def format_feedback_history(self, session_id: str) -> str:
        evals = self.get_evaluations(session_id)
        if not evals:
            return "None. This is the first evaluation of this question."
        return "\n\n".join(
            f"Attempt {e['attempt']}: {e['decision']}\n"
            f"Feedback you gave: {_clip(e['feedback']) or 'none'}"
            for e in evals
        )

    # ---- cleanup ----
    def clear(self, session_id: str) -> None:
        with self._lock:
            self._evaluations.pop(session_id, None)
            self._past.pop(session_id, None)
        logger.info("Evaluator memory cleared for session %s", session_id)


evaluator_memory = EvaluatorMemory()