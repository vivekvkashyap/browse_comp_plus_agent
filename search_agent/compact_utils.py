"""
Shared utilities for the compact (context summarization) tool.

The compact tool allows an LLM agent to voluntarily summarize its conversation
history at any point, replacing the full history with a concise summary to free
up context space.
"""

import json
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Compact / summarization prompt
# ---------------------------------------------------------------------------

COMPACT_PROMPT = """\
You are a research-session summarizer. You will receive the full conversation \
history of a deep-research agent that has been searching a document corpus to \
answer a question. Your job is to produce a **concise, structured summary** \
that preserves every piece of information the agent would need to continue \
its research effectively.

Your summary MUST include the following sections:

1. **Original Question** – Restate the question exactly as it was asked.

2. **Key Findings** – Bullet list of facts, claims, and evidence discovered \
so far. For each finding, include the document ID(s) in square brackets \
(e.g. [12345]) so the agent can refer back to them.

3. **Search Queries Tried** – List every search query the agent has already \
issued, so it does not repeat them.

4. **Current Hypothesis** – The agent's best current answer or leading \
hypothesis, if any.

5. **Open Questions / Next Steps** – What the agent still needs to find out.

Be concise but DO NOT drop any factual detail, document ID, or search query. \
The agent will lose access to the original messages after this summary, so \
anything you omit is lost forever.
""".strip()

# ---------------------------------------------------------------------------
# History serialization
# ---------------------------------------------------------------------------


def format_history_for_compact(messages: List[Dict[str, Any]]) -> str:
    """Serialize a conversation message list into readable text for the
    summarizer.

    Handles message formats from:
    - OpenAI Responses API  (type-based: function_call, function_call_output, …)
    - OpenAI Chat Completions / GLM  (role-based: system, user, assistant, tool)
    - Anthropic Messages API  (role-based with content blocks)
    """
    lines: list[str] = []

    for msg in messages:
        # --- OpenAI Responses API (type-based items) -----------------------
        msg_type = msg.get("type")
        if msg_type == "function_call":
            name = msg.get("name", "unknown")
            arguments = msg.get("arguments", "")
            lines.append(f"[Tool Call: {name}] Arguments: {arguments}")
            continue
        if msg_type == "function_call_output":
            output = msg.get("output", "")
            # Truncate very long outputs to keep the summary prompt manageable
            if len(output) > 2000:
                output = output[:2000] + " … (truncated)"
            lines.append(f"[Tool Result] {output}")
            continue
        if msg_type == "message":
            # Responses-API message wrapper
            content = msg.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        if text:
                            lines.append(f"[Assistant] {text}")
            elif isinstance(content, str) and content:
                lines.append(f"[Assistant] {content}")
            continue
        if msg_type == "reasoning":
            summary = msg.get("summary")
            if summary:
                lines.append(f"[Reasoning] {summary}")
            continue

        # --- Role-based messages (Chat Completions / Anthropic) ------------
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            lines.append(f"[System] {content}")
        elif role == "user":
            if isinstance(content, str):
                lines.append(f"[User] {content}")
            elif isinstance(content, list):
                # Anthropic content blocks
                for block in content:
                    if isinstance(block, dict) and block.get("text"):
                        lines.append(f"[User] {block['text']}")
        elif role == "assistant":
            if isinstance(content, str) and content:
                lines.append(f"[Assistant] {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type", "")
                        if btype == "text" and block.get("text"):
                            lines.append(f"[Assistant] {block['text']}")
                        elif btype == "tool_use":
                            name = block.get("name", "unknown")
                            inp = json.dumps(block.get("input", {}))
                            lines.append(f"[Tool Call: {name}] Arguments: {inp}")
            # Also handle tool_calls list (Chat Completions format)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "unknown")
                    arguments = fn.get("arguments", "")
                    lines.append(f"[Tool Call: {name}] Arguments: {arguments}")
        elif role == "tool":
            name = msg.get("name", "")
            if isinstance(content, str):
                output = content
            else:
                output = json.dumps(content)
            if len(output) > 2000:
                output = output[:2000] + " … (truncated)"
            lines.append(f"[Tool Result ({name})] {output}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summarizer calls
# ---------------------------------------------------------------------------


def call_compact_openai(client, model: str, history_text: str, compact_prompt: str | None = None) -> tuple[str, dict]:
    """Call an OpenAI-compatible API to produce a summary of the conversation
    history.

    Works for both OpenAI and GLM (which uses the OpenAI SDK).

    Returns:
        (summary_text, usage_dict) where usage_dict has input_tokens,
        output_tokens, total_tokens from the summarizer call.
    """
    prompt = compact_prompt or COMPACT_PROMPT

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Here is the full conversation history to summarize:\n\n{history_text}"},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    summary = response.choices[0].message.content.strip()

    usage: dict = {}
    if hasattr(response, "usage") and response.usage:
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(response.usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(response.usage, "total_tokens", 0) or 0,
        }

    return summary, usage


def call_compact_anthropic(client, model: str, history_text: str, compact_prompt: str | None = None) -> tuple[str, dict]:
    """Call the Anthropic Messages API to produce a summary of the
    conversation history.

    Returns:
        (summary_text, usage_dict) where usage_dict has input_tokens,
        output_tokens, total_tokens from the summarizer call.
    """
    prompt = compact_prompt or COMPACT_PROMPT

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=prompt,
        messages=[
            {"role": "user", "content": f"Here is the full conversation history to summarize:\n\n{history_text}"},
        ],
        temperature=0.3,
    )

    # Extract text from content blocks
    parts = []
    for block in response.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    summary = "\n".join(parts).strip()

    usage: dict = {}
    if hasattr(response, "usage") and response.usage:
        in_tok = getattr(response.usage, "input_tokens", 0) or 0
        out_tok = getattr(response.usage, "output_tokens", 0) or 0
        usage = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
        }

    return summary, usage
