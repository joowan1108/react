from __future__ import annotations

from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

AOP_SUMMARIZE_DESCRIPTION = (
    "condense lengthy text into shorter summaries to optimize context consumption and readability"
)


@dataclass(frozen=True, slots=True)
class SummarizeRequest:
    text: str
    focus: str | None = None
    max_input_chars: int = 12000


def _truncate_text(text: str, max_input_chars: int) -> tuple[str, bool]:
    if max_input_chars <= 0 or len(text) <= max_input_chars:
        return text, False
    return text[:max_input_chars], True


def build_aop_summarize_messages(request: SummarizeRequest) -> tuple[list[ModelMessage], bool]:
    normalized_text = request.text.strip()
    if not normalized_text:
        raise ValueError("summarize.text must be a non-empty string.")

    truncated_text, was_truncated = _truncate_text(normalized_text, request.max_input_chars)

    instructions = [
        "The next operator is Summarize.",
        f"This operator is to {AOP_SUMMARIZE_DESCRIPTION}.",
        "Please execute it following the instructions and output the results.",
        "Return only the summary text.",
        "Preserve key entities, numbers, facts, constraints, and relationships.",
        "Do not add information that is not present in the input context.",
    ]
    if request.focus:
        instructions.append(f"Focus especially on: {request.focus}")

    user_prompt = "\n".join(instructions) + f"\n\nInput context:\n{truncated_text}"
    return (
        [
            ModelMessage(
                role="system",
                content=(
                    "You are executing a semantic operator in an AOP-style LLM pipeline. "
                    "Follow the operator instructions exactly and return only the operator result."
                ),
            ),
            ModelMessage(role="user", content=user_prompt),
        ],
        was_truncated,
    )


def summarize_text_with_model(model: ModelAdapter, request: SummarizeRequest) -> dict[str, object]:
    messages, was_truncated = build_aop_summarize_messages(request)
    summary = model.complete(messages).strip()
    if not summary:
        raise RuntimeError("Summarize operator returned an empty response.")

    return {
        "operator": "Summarize",
        "operator_description": AOP_SUMMARIZE_DESCRIPTION,
        "summary": summary,
        "input_char_count": len(request.text),
        "used_input_char_count": min(len(request.text), request.max_input_chars),
        "truncated": was_truncated,
        "focus": request.focus,
    }
