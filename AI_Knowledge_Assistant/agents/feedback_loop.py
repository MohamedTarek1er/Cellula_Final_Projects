import logging

from agents.evaluator import evaluate_answer
from agents.generator import generate_answer, retrieve_context
from memory.evaluator_memory import evaluator_memory
from memory.generator_memory import generator_memory

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


def _finish(session_id: str, question: str, result: dict) -> dict:
    """
    Close the question: write the finished turn into each agent's OWN memory,
    then return the result. Technical errors are not remembered.
    """
    status = result["status"]

    if status != "error":
        generator_memory.add_turn(
            session_id,
            question,
            result["answer"],
            result["sources"],
            result["validated"],
        )

    if status in ("accepted", "max_iterations"):
        evaluator_memory.add_turn_summary(
            session_id,
            question,
            result["decision"],
            result["iterations"],
        )

    return result


def clear_session_memory(session_id: str) -> None:
    """Wipe both memories of one session (for the 'clear history' button)."""
    generator_memory.clear(session_id)
    evaluator_memory.clear(session_id)


def run_feedback_loop(
    question: str,
    top_k: int = 4,
    session_id: str = "default",
) -> dict:
    session_id = (session_id or "").strip() or "default"
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
    context = retrieval["context"]

    # Nothing found: an honest "I don't know", not a failed answer.
    if not context.strip():
        logger.info("No knowledge found; skipping the evaluation loop.")
        result = _result(NO_KNOWLEDGE_MESSAGE, "no_knowledge", False, 0, [], history)
        return _finish(session_id, question, result)

    # Step 2: open a fresh page in each agent's own notebook.
    generator_memory.start_question(session_id, context)
    evaluator_memory.start_question(session_id)

    answer = ""
    feedback = ""

    # Step 3: generate -> evaluate, at most MAX_ITERATIONS times.
    for attempt in range(1, MAX_ITERATIONS + 1):
        logger.info("Attempt %s/%s (session=%s)", attempt, MAX_ITERATIONS, session_id)

        # Generator reads ONLY its own memory.
        try:
            generated = generate_answer(
                question=question,
                top_k=top_k,
                feedback=feedback,
                retrieval=retrieval,
                conversation_history=generator_memory.format_conversation(session_id),
                attempt_history=generator_memory.format_attempts(session_id),
            )
        except Exception:
            logger.exception("Generator failed on attempt %s.", attempt)
            # Keep the last good answer if we have one.
            result = _result(
                answer or ERROR_MESSAGE,
                "error",
                False,
                attempt,
                sources,
                history,
                disclaimer="Note: a technical error stopped the process."
                if answer
                else "",
            )
            return _finish(session_id, question, result)

        answer = generated["answer"]

        # Generator writes to its own memory (answer + the feedback that led to it).
        generator_memory.add_attempt(
            session_id,
            attempt,
            answer,
            feedback_received=feedback,
        )

        # Evaluator reads ONLY its own memory plus question, context, new answer.
        evaluation = evaluate_answer(
            question=question,
            answer=answer,
            context=context,
            feedback_history=evaluator_memory.format_feedback_history(session_id),
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
            result = _result(
                answer,
                "evaluation_failed",
                False,
                attempt,
                sources,
                history,
                EVALUATION_FAILED_DISCLAIMER,
            )
            return _finish(session_id, question, result)

        decision = evaluation["decision"]
        evaluator_feedback = evaluation["feedback"]

        # Evaluator writes to its own memory.
        evaluator_memory.add_evaluation(
            session_id,
            attempt,
            decision,
            evaluator_feedback,
        )

        history.append(
            {
                "attempt": attempt,
                "answer": answer,
                "decision": decision,
                "feedback": evaluator_feedback,
            }
        )

        # Step 4a: accepted -> exit and return.
        if decision == "accept":
            logger.info("Answer accepted on attempt %s.", attempt)
            result = _result(answer, "accepted", True, attempt, sources, history)
            return _finish(session_id, question, result)

        # Step 4b: rejected -> the feedback goes back to the Generator.
        feedback = evaluator_feedback

    # Step 5: out of attempts -> return the latest answer with a disclaimer.
    logger.warning("Max iterations reached without acceptance.")
    result = _result(
        answer,
        "max_iterations",
        False,
        MAX_ITERATIONS,
        sources,
        history,
        UNVALIDATED_DISCLAIMER,
    )
    return _finish(session_id, question, result)
