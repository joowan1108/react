from __future__ import annotations

from dataclasses import dataclass

from data_agent_baseline.agents.model import ModelAdapter, ModelMessage

SUMMARIZE_TOOL_DESCRIPTION = (
    "condense long context into a shorter fact-preserving summary for the next reasoning step"
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


def build_summarize_messages(request: SummarizeRequest) -> tuple[list[ModelMessage], bool]:
    normalized_text = request.text.strip()
    if not normalized_text:
        raise ValueError("summarize.text must be a non-empty string.")

    truncated_text, was_truncated = _truncate_text(normalized_text, request.max_input_chars)

    instructions = [
        "You are helping a ReAct-style agent summarize context before the next tool or reasoning step.",
        "Return only a concise summary.",
        "Preserve key entities, numbers, facts, constraints, and relationships.",
        "Keep only information that may matter for answering the task correctly.",
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
                    "You are a summarization helper for a ReAct-style data agent. "
                    "Summarize only the provided context and return only the summary text."
                ),
            ),
            ModelMessage(role="user", content=user_prompt),
        ],
        was_truncated,
    )


def summarize_text_with_model(model: ModelAdapter, request: SummarizeRequest) -> dict[str, object]:
    messages, was_truncated = build_summarize_messages(request)
    summary = model.complete(messages).strip()
    if not summary:
        raise RuntimeError("summarize returned an empty response.")

    return {
        "tool": "summarize",
        "tool_description": SUMMARIZE_TOOL_DESCRIPTION,
        "summary": summary,
        "input_char_count": len(request.text),
        "used_input_char_count": min(len(request.text), request.max_input_chars),
        "truncated": was_truncated,
        "focus": request.focus,
    }
