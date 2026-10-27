import json
from langchain_core.messages import SystemMessage, HumanMessage
from llm import llm

def evaluate_answer(question: str, answer: str, context: str) -> dict:
    system_prompt = """
You are an Evaluator for an AI Knowledge Assistant.

Evaluate the generated answer using the user's question
and the retrieved context.

Accept the answer if it answers the question correctly
and is supported by the context.

Reject the answer if it contains incorrect information,
unsupported claims, or does not answer the question.

Return JSON only:
{
    "decision": "accept" or "reject",
    "feedback": "Explain why and how to improve the answer if rejected."
}
"""

    human_prompt = f"""
            Question:
            {question}

            Retrieved Context:
            {context}

            Generated Answer:
            {answer}
            """

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ])

    response_text = response.content.strip()

    # Remove Markdown code fences if present.
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1]
        response_text = response_text.rsplit("```", 1)[0].strip()

    try:
        result = json.loads(response_text)
    except json.JSONDecodeError:
        return {
            "decision": "reject",
            "feedback": "The evaluator returned invalid JSON. Please try again.",
        }

    # Validate the expected response structure.
    if (
        not isinstance(result, dict)
        or result.get("decision") not in {"accept", "reject"}
        or not isinstance(result.get("feedback"), str)
    ):
        return {
            "decision": "reject",
            "feedback": "The evaluator returned an invalid response format.",
        }

    return result
