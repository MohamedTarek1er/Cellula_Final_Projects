from agents.generator import generate_answer
from agents.evaluator import evaluate_answer

MAX_ITERATIONS = 4

def run_feedback_loop(question: str, top_k: int = 4) -> dict:
    feedback = ""
    history = []

    for attempt in range(1, MAX_ITERATIONS + 1):
        generated = generate_answer(
            question=question,
            top_k=top_k,
            feedback=feedback
        )

        answer = generated["answer"]
        context = generated.get("context", "")
        sources = generated.get("sources", [])

        if not context.strip():
            return {
                "answer": answer,
                "decision": "reject",
                "iterations": attempt,
                "sources": sources,
                "history": history,
                "validated": False
            }

        evaluation = evaluate_answer(
            question=question,
            answer=answer,
            context=context
        )

        decision = evaluation.get("decision", "reject")
        evaluator_feedback = evaluation.get("feedback", "")

        history.append({
            "attempt": attempt,
            "answer": answer,
            "decision": decision,
            "feedback": evaluator_feedback
        })

        if decision == "accept":
            return {
                "answer": answer,
                "decision": "accept",
                "iterations": attempt,
                "sources": sources,
                "history": history,
                "validated": True
            }

        feedback = evaluator_feedback

    # The maximum number of attempts was reached.
    return {
        "answer": answer,
        "decision": "reject",
        "iterations": MAX_ITERATIONS,
        "sources": sources,
        "history": history,
        "validated": False
    }
