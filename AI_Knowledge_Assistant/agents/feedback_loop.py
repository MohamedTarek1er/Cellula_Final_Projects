import logging

from agents.evaluator import evaluate_answer
from agents.generator import generate_answer, retrieve_context

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 4

NO_KNOWLEDGE_MESSAGE = (
    "I could not find relevant information in the available knowledge base."
)

UNVALIDATED_DISCLAIMER = (
    "Note: this answer could not be fully validated by the evaluator "
    f"after {MAX_ITERATIONS} attempts. Please verify it against the sources."
)

EVALUATION_FAILED_DISCLAIMER = (
    "Note: the evaluator could not check this answer because of a technical "
    "problem, so it is not validated. Please verify it against the sources."
)

ERROR_MESSAGE = "Something went wrong while generating the answer. Please try again."


def _result(
    answer: str,
    status: str,
    validated: bool,
    iterations: int,
    sources: list,
    history: list,
    disclaimer: str = "",
) -> dict:
    """Build the final response in one consistent shape."""
    return {
        "answer": answer,
        "decision": "accept" if validated else "reject",
        "status": status,  # accepted | max_iterations | no_knowledge | evaluation_failed | error
        "validated": validated,
        "iterations": iterations,
        "sources": sources,
        "history": history,
        "disclaimer": disclaimer,
    }


def run_feedback_loop(question: str, top_k: int = 4) -> dict:
    history = []

    # Step 1: search the knowledge base ONCE (the question never changes).
    try:
        retrieval = retrieve_context(question, top_k=top_k)
    except ValueError:
        raise  # empty question: let the UI show a validation message
    except Exception:
        logger.exception("Retrieval failed.")
        return _result(ERROR_MESSAGE, "error", False, 0, [], history)

    sources = retrieval["sources"]

    # Nothing found: an honest "I don't know", not a failed answer.
    if not retrieval["context"].strip():
        logger.info("No knowledge found; skipping the evaluation loop.")
        return _result(NO_KNOWLEDGE_MESSAGE, "no_knowledge", False, 0, [], history)

    context = retrieval["context"]
    answer = ""
    feedback = ""

    # Step 2: generate -> evaluate, at most MAX_ITERATIONS times.
    for attempt in range(1, MAX_ITERATIONS + 1):
        logger.info("Attempt %s/%s", attempt, MAX_ITERATIONS)

        # Generator gets: question, context, its own previous answer, feedback.
        try:
            generated = generate_answer(
                question=question,
                top_k=top_k,
                feedback=feedback,
                previous_answer=answer,
                retrieval=retrieval,
            )
        except Exception:
            logger.exception("Generator failed on attempt %s.", attempt)
            # Keep the last good answer if we have one.
            return _result(
                answer or ERROR_MESSAGE,
                "error",
                False,
                attempt,
                sources,
                history,
                disclaimer="Note: a technical error stopped the process." if answer else "",
            )

        answer = generated["answer"]

        # Evaluator gets: question, context, the new answer. Nothing else.
        evaluation = evaluate_answer(
            question=question,
            answer=answer,
            context=context,
        )

        # Technical failure of the evaluator (not a real rejection).
        if evaluation.get("evaluation_failed"):
            history.append(
                {
                    "attempt": attempt,
                    "answer": answer,
                    "decision": "evaluation_failed",
                    "feedback": "",
                }
            )
            return _result(
                answer,
                "evaluation_failed",
                False,
                attempt,
                sources,
                history,
                EVALUATION_FAILED_DISCLAIMER,
            )

        decision = evaluation["decision"]
        evaluator_feedback = evaluation["feedback"]

        history.append(
            {
                "attempt": attempt,
                "answer": answer,
                "decision": decision,
                "feedback": evaluator_feedback,
            }
        )

        # Step 3a: accepted -> exit and return.
        if decision == "accept":
            logger.info("Answer accepted on attempt %s.", attempt)
            return _result(answer, "accepted", True, attempt, sources, history)

        # Step 3b: rejected -> the feedback goes back to the Generator.
        feedback = evaluator_feedback

    # Step 4: out of attempts -> return the latest answer with a disclaimer.
    logger.warning("Max iterations reached without acceptance.")
    return _result(
        answer,
        "max_iterations",
        False,
        MAX_ITERATIONS,
        sources,
        history,
        UNVALIDATED_DISCLAIMER,
    )