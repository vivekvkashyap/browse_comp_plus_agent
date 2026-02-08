import argparse
import csv
import datetime
import json
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime as dt
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from rich import print as rprint
from tqdm import tqdm

load_dotenv(override=False)

script_env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=script_env_path, override=False)

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from anthropic import Anthropic, APIError, RateLimitError
from compact_utils import call_compact_anthropic, format_history_for_compact
from prompts import format_query
from transformers import AutoTokenizer
from utils import extract_retrieved_docids_from_result

from searcher.searchers import SearcherType


# ---------------------------------------------------------------------------
# SearchToolHandler -- local tool execution (Anthropic tool schema format)
# ---------------------------------------------------------------------------


class SearchToolHandler:
    """Manages tool definitions (Anthropic format) and local execution of
    search, get_document, and compact tools."""

    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
        include_compact: bool = False,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document
        self.include_compact = include_compact

        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions in Anthropic's native format
        (using ``input_schema``, not the OpenAI ``parameters`` wrapper)."""
        tools: list[dict[str, Any]] = [
            {
                "name": "search",
                "description": self.searcher.search_description(self.k),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query string",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]

        if self.include_get_document:
            tools.append(
                {
                    "name": "get_document",
                    "description": self.searcher.get_document_description(),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "docid": {
                                "type": "string",
                                "description": "Document ID to retrieve",
                            }
                        },
                        "required": ["docid"],
                    },
                }
            )

        if self.include_compact:
            tools.append(
                {
                    "name": "compact",
                    "description": (
                        "Summarize and compact your conversation history to free up context space. "
                        "Call this proactively when you have accumulated many search results and your "
                        "context is getting long. This does NOT end your research — continue searching "
                        "after compacting. Takes no arguments."
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                }
            )

        return tools

    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        if tool_name == "search":
            return self._search(arguments["query"])
        elif tool_name == "get_document":
            return self._get_document(arguments["docid"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str) -> str:
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                if len(tokens) > self.snippet_max_tokens:
                    truncated_tokens = tokens[: self.snippet_max_tokens]
                    cand["snippet"] = self.tokenizer.decode(
                        truncated_tokens, skip_special_tokens=True
                    )
                else:
                    cand["snippet"] = text
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str) -> str:
        result = self.searcher.get_document(docid)
        if result is None:
            return json.dumps({"error": f"Document with docid '{docid}' not found"})
        return json.dumps(result, indent=2)


# ---------------------------------------------------------------------------
# Rate-limit / retry helpers (kept from original)
# ---------------------------------------------------------------------------


def _make_request_with_retry(client: Anthropic, max_retries: int = 5, delay: float = 60.0, **kwargs):
    """Call client.messages.create with automatic rate-limit retry."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(**kwargs)

            if hasattr(response, "_headers") and response._headers:
                wait_time = _calculate_wait_time_from_headers(response._headers)
                if wait_time > 0:
                    print(
                        f"[Info] Waiting {wait_time}s for input token rate limit to fully replenish"
                    )
                    time.sleep(wait_time)

            return response

        except RateLimitError as e:
            if attempt == max_retries - 1:
                print(f"[Error] Rate limit exceeded after {max_retries} attempts")
                raise e

            print(e)

            retry_after = _extract_retry_after_from_error(e)
            input_token_wait = _extract_input_token_reset_from_error(e)

            wait_times = [
                t for t in [retry_after, input_token_wait, delay] if t is not None
            ]
            wait_time = max(wait_times) if wait_times else delay

            if retry_after and input_token_wait:
                print(
                    f"[Info] Rate limit hit, retrying after {wait_time}s (max of retry-after: {retry_after}s, input token reset: {input_token_wait}s) (attempt {attempt + 1}/{max_retries})"
                )
            elif retry_after:
                print(
                    f"[Info] Rate limit hit, retrying after {wait_time}s as specified by retry-after header (attempt {attempt + 1}/{max_retries})"
                )
            elif input_token_wait:
                print(
                    f"[Info] Rate limit hit, retrying after {wait_time}s until input tokens replenish (attempt {attempt + 1}/{max_retries})"
                )
            else:
                print(
                    f"[Info] Rate limit hit, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                )

            time.sleep(wait_time)

        except APIError as e:
            print(f"[Error] API error: {e}")
            raise e
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")
            raise e


def _calculate_wait_time_from_headers(headers):
    input_token_reset = headers.get("anthropic-ratelimit-input-tokens-reset")
    if input_token_reset:
        try:
            reset_time = datetime.datetime.fromisoformat(
                input_token_reset.replace("Z", "+00:00")
            )
            current_time = datetime.datetime.now(datetime.timezone.utc)
            wait_seconds = (reset_time - current_time).total_seconds()
            return max(0, wait_seconds)
        except Exception as e:
            print(f"[Warning] Could not parse input token reset time: {e}")

    return 0


def _extract_retry_after_from_error(rate_limit_error):
    try:
        if hasattr(rate_limit_error, "response") and rate_limit_error.response:
            headers = rate_limit_error.response.headers
            retry_after = headers.get("retry-after")
            if retry_after:
                return int(retry_after)
    except Exception:
        pass

    return None


def _extract_input_token_reset_from_error(rate_limit_error):
    try:
        if hasattr(rate_limit_error, "response") and rate_limit_error.response:
            headers = rate_limit_error.response.headers
            input_token_reset = headers.get("anthropic-ratelimit-input-tokens-reset")
            if input_token_reset:
                reset_time = datetime.datetime.fromisoformat(
                    input_token_reset.replace("Z", "+00:00")
                )
                current_time = datetime.datetime.now(datetime.timezone.utc)
                wait_seconds = (reset_time - current_time).total_seconds()
                return max(0, wait_seconds)
    except Exception as e:
        print(f"[Warning] Could not parse input token reset time from error: {e}")

    return None


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _serialize_content_block(block) -> dict:
    """Serialize an Anthropic content block to a JSON-safe dict."""
    if isinstance(block, dict):
        return block
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="python")
    if hasattr(block, "to_dict"):
        return block.to_dict()
    return str(block)


# ---------------------------------------------------------------------------
# Core conversation loop with native tool calling
# ---------------------------------------------------------------------------


def run_conversation_with_tools(
    client: Anthropic,
    *,
    query: str,
    model: str,
    max_tokens: int,
    tool_handler: SearchToolHandler,
    system_prompt: str | None = None,
    query_template: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_iterations: int = 100,
    max_rate_limit_retries: int = 5,
    rate_limit_delay: float = 60.0,
    compact_model: str | None = None,
    thinking_enabled: bool = False,
    thinking_budget: int = 8192,
):
    """Run a multi-turn conversation with native Anthropic tool calling.

    Returns (normalized_results, cumulative_usage, stop_reason, trajectory).
    """
    tools = tool_handler.get_tool_definitions()

    # Resolve compact model: default to the agent model
    _compact_model = compact_model or model

    formatted_query = format_query(query, query_template)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": formatted_query},
    ]

    cumulative_usage = {
        "input_tokens": 0,
        "input_tokens_cached": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }

    summarizer_usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "num_calls": 0,
    }

    normalized_results: list[dict[str, Any]] = []
    stop_reason: Optional[str] = None

    # --- Trajectory: flat conversation log ---
    trajectory: list[dict[str, Any]] = []
    if system_prompt:
        trajectory.append({"role": "system", "content": system_prompt})
    trajectory.append({"role": "user", "content": formatted_query})

    # Treat max_tokens as a global output budget across the entire conversation.
    global_max_tokens = max_tokens

    for iteration in range(max_iterations):
        remaining_tokens = global_max_tokens - cumulative_usage["output_tokens"]
        if remaining_tokens <= 0:
            print(f"Warning: Reached global max_tokens output budget ({global_max_tokens})")
            break

        # Build API call kwargs
        create_kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": min(remaining_tokens, global_max_tokens),
            "messages": messages,
            "tools": tools,
        }

        if system_prompt:
            create_kwargs["system"] = system_prompt

        if thinking_enabled:
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }

        if temperature is not None:
            create_kwargs["temperature"] = temperature

        if top_p is not None:
            create_kwargs["top_p"] = top_p

        response = _make_request_with_retry(
            client,
            max_retries=max_rate_limit_retries,
            delay=rate_limit_delay,
            **create_kwargs,
        )

        stop_reason = response.stop_reason

        # Accumulate usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            cumulative_usage["input_tokens"] += getattr(usage, "input_tokens", 0)
            cumulative_usage["output_tokens"] += getattr(usage, "output_tokens", 0)
            cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
            cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
            cumulative_usage["input_tokens_cached"] += cache_create + cache_read
            cumulative_usage["total_tokens"] += (
                getattr(usage, "input_tokens", 0) + getattr(usage, "output_tokens", 0)
            )

        # Serialize content blocks for trajectory
        serialized_blocks = [_serialize_content_block(b) for b in response.content]
        trajectory.append({"role": "assistant", "content": serialized_blocks})

        # Process content blocks into normalized results
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                normalized_results.append({
                    "type": "reasoning",
                    "tool_name": None,
                    "arguments": None,
                    "output": [getattr(block, "thinking", "")],
                })
            elif btype == "text":
                text = getattr(block, "text", "")
                if text.strip():
                    normalized_results.append({
                        "type": "output_text",
                        "tool_name": None,
                        "arguments": None,
                        "output": text,
                    })
            elif btype == "tool_use":
                normalized_results.append({
                    "type": "tool_call",
                    "tool_name": getattr(block, "name", None),
                    "arguments": json.dumps(getattr(block, "input", {})),
                    "output": None,
                })

        # If no tool calls, we are done
        if stop_reason != "tool_use":
            break

        # --- Append assistant response to messages (contains tool_use blocks) ---
        messages.append({"role": "assistant", "content": response.content})

        # --- Execute tool calls and collect results ---
        tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        tool_results: list[dict[str, Any]] = []
        did_compact = False

        for block in tool_use_blocks:
            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            try:
                # ---- Handle compact tool ----
                if tool_name == "compact":
                    print(f"[Compact] Summarizing conversation history (iter {iteration + 1})…")

                    # Serialize messages for the summarizer
                    serialized_messages = []
                    for msg in messages:
                        if isinstance(msg, dict):
                            content = msg.get("content")
                            if isinstance(content, list):
                                # Content might contain SDK objects; serialize them
                                serialized_content = []
                                for item in content:
                                    serialized_content.append(_serialize_content_block(item))
                                serialized_messages.append({**msg, "content": serialized_content})
                            else:
                                serialized_messages.append(msg)
                        else:
                            serialized_messages.append(msg)

                    history_text = format_history_for_compact(serialized_messages)
                    summary, compact_usage = call_compact_anthropic(client, _compact_model, history_text)
                    print(f"[Compact] Summary produced ({len(summary)} chars). Replacing history.")

                    # Accumulate summarizer token usage
                    if compact_usage:
                        summarizer_usage["input_tokens"] += compact_usage.get("input_tokens", 0)
                        summarizer_usage["output_tokens"] += compact_usage.get("output_tokens", 0)
                        summarizer_usage["total_tokens"] += compact_usage.get("total_tokens", 0)
                        summarizer_usage["num_calls"] += 1

                    result_text = "Context compacted successfully. Your conversation history has been summarized. Continue your research."

                    # Patch the last tool_call in normalized_results
                    for nr in reversed(normalized_results):
                        if nr.get("type") == "tool_call" and nr.get("tool_name") == "compact":
                            nr["output"] = result_text
                            break

                    # Record compact in trajectory
                    trajectory.append({
                        "role": "compact_summary",
                        "input": history_text,
                        "content": summary,
                    })
                    trajectory.append({
                        "role": "tool",
                        "tool_name": "compact",
                        "call_id": tool_use_id,
                        "content": result_text,
                    })

                    # Replace messages: original query + summary + continue prompt
                    messages = [
                        {"role": "user", "content": formatted_query},
                        {
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        "Here is a summary of my prior research:\n\n" + summary
                                    ),
                                }
                            ],
                        },
                        {
                            "role": "user",
                            "content": "Continue your research based on the summary above.",
                        },
                    ]

                    did_compact = True
                    # Stop processing other tool calls from this response
                    # (messages were reset, so any other pending calls are orphaned)
                    break

                # ---- Handle search / get_document tools ----
                output = tool_handler.execute_tool(tool_name, tool_input)

                # Patch the matching tool_call in normalized_results
                for nr in reversed(normalized_results):
                    if (
                        nr.get("type") == "tool_call"
                        and nr.get("tool_name") == tool_name
                        and nr.get("output") is None
                    ):
                        nr["output"] = output
                        break

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": output,
                })

                # Record tool result in trajectory
                trajectory.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "call_id": tool_use_id,
                    "content": output,
                })

            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(error_msg)

                # Patch error into normalized_results
                for nr in reversed(normalized_results):
                    if (
                        nr.get("type") == "tool_call"
                        and nr.get("tool_name") == tool_name
                        and nr.get("output") is None
                    ):
                        nr["output"] = error_msg
                        break

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": error_msg,
                    "is_error": True,
                })

                # Record tool error in trajectory
                trajectory.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "call_id": tool_use_id,
                    "content": error_msg,
                    "error": True,
                })

        # Append tool results to messages (unless compact reset messages)
        if not did_compact and tool_results:
            messages.append({"role": "user", "content": tool_results})

    if stop_reason is None:
        print(f"Warning: Conversation hit max iterations ({max_iterations}) without final response")

    return normalized_results, cumulative_usage, stop_reason, trajectory, summarizer_usage


# ---------------------------------------------------------------------------
# Persist response to JSON
# ---------------------------------------------------------------------------


def _persist_response(
    out_dir: str,
    *,
    model: str,
    query_id: str | None,
    system_prompt: str | None,
    max_tokens: int,
    normalized_results: list[dict[str, Any]],
    cumulative_usage: dict,
    stop_reason: Optional[str],
    trajectory: list[dict[str, Any]] | None = None,
    thinking_config: dict | None = None,
    summarizer_usage: dict | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    tool_call_counts: dict[str, int] = {}
    for item in normalized_results:
        if item.get("type") == "tool_call" and item.get("tool_name"):
            name = item["tool_name"]
            tool_call_counts[name] = tool_call_counts.get(name, 0) + 1

    normalized_usage = {
        "input_tokens": cumulative_usage.get("input_tokens", 0),
        "input_tokens_cached": cumulative_usage.get("input_tokens_cached", 0),
        "output_tokens": cumulative_usage.get("output_tokens", 0),
        "included_reasoning_tokens": None,
        "total_tokens": cumulative_usage.get("total_tokens", 0),
    }

    status = "completed" if stop_reason == "end_turn" else stop_reason

    ts = dt.utcnow().strftime("%Y%m%dT%H%M%S%fZ")
    filename = os.path.join(out_dir, f"run_{ts}.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "model": model,
                    "output_dir": str(out_dir),
                    "max_tokens": max_tokens,
                    "thinking": thinking_config,
                },
                "query_id": query_id,
                "tool_call_counts": tool_call_counts,
                "usage": normalized_usage,
                "status": status,
                "retrieved_docids": extract_retrieved_docids_from_result(normalized_results),
                "result": normalized_results,
                "trajectory": trajectory or [],
                "summarizer_usage": summarizer_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "num_calls": 0},
            },
            f,
            indent=2,
            default=str,
        )

    print("Saved response to", filename, "| tool call counts:", tool_call_counts)


# ---------------------------------------------------------------------------
# TSV dataset processing
# ---------------------------------------------------------------------------


def _process_tsv_dataset(
    tsv_path: str, client: Anthropic, args, tool_handler: SearchToolHandler
):
    dataset_path = Path(tsv_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    out_dir = Path(args.output_dir).expanduser().resolve()

    queries: list[tuple[str, str]] = []
    with dataset_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            queries.append((row[0].strip(), row[1].strip()))

    processed_ids: set[str] = set()
    if out_dir.exists():
        for json_path in out_dir.glob("run_*.json"):
            try:
                with json_path.open("r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    qid_saved = meta.get("query_id")
                    if qid_saved:
                        processed_ids.add(str(qid_saved))
            except Exception:
                continue

    remaining = [(qid, qtext) for qid, qtext in queries if qid not in processed_ids]

    print(
        f"Processing {len(remaining)} remaining queries (skipping {len(processed_ids)}) from {dataset_path} …"
    )

    thinking_enabled = not args.no_thinking
    thinking_config = None
    if thinking_enabled:
        thinking_config = {
            "type": "enabled",
            "budget_tokens": args.thinking_budget,
        }

    completed_lock = threading.Lock()
    completed_count = [0]

    def _handle_single_query(qid: str, qtext: str, pbar=None):
        try:
            normalized_results, cumulative_usage, finish_reason, traj, sum_usage = run_conversation_with_tools(
                client,
                query=qtext,
                model=args.model,
                max_tokens=args.max_tokens,
                tool_handler=tool_handler,
                system_prompt=args.system,
                query_template=args.query_template,
                temperature=args.temperature,
                top_p=args.top_p,
                max_iterations=args.max_iterations,
                max_rate_limit_retries=args.max_rate_limit_retries,
                rate_limit_delay=args.rate_limit_delay,
                compact_model=getattr(args, "compact_model", None),
                thinking_enabled=thinking_enabled,
                thinking_budget=args.thinking_budget,
            )

            with completed_lock:
                completed_count[0] += 1
                if pbar:
                    pbar.set_postfix(completed=completed_count[0])

            _persist_response(
                str(out_dir),
                model=args.model,
                query_id=qid,
                system_prompt=args.system,
                max_tokens=args.max_tokens,
                normalized_results=normalized_results,
                cumulative_usage=cumulative_usage,
                stop_reason=finish_reason,
                trajectory=traj,
                thinking_config=thinking_config,
                summarizer_usage=sum_usage,
            )

        except Exception as exc:
            print(f"[Error] Query id={qid} failed: {exc}")
            sys.exit(1)

    if args.num_threads <= 1:
        with tqdm(remaining, desc="Queries", unit="query") as pbar:
            for qid, qtext in pbar:
                _handle_single_query(qid, qtext, pbar)
    else:
        with (
            ThreadPoolExecutor(max_workers=args.num_threads) as executor,
            tqdm(total=len(remaining), desc="Queries", unit="query") as pbar,
        ):
            futures = [
                executor.submit(_handle_single_query, qid, qtext, pbar)
                for qid, qtext in remaining
            ]

            for _ in as_completed(futures):
                pbar.update(1)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Call Anthropic Messages API with native tool calling and local search."
    )
    parser.add_argument(
        "--query",
        default="topics-qrels/queries.tsv",
        help="User query text or path to TSV. Wrap in quotes if contains spaces.",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model name (default: %(default)s)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=10000,
        help="Maximum tokens to generate (default: %(default)s)",
    )
    parser.add_argument("--system", default=None, help="Optional system prompt")

    # Thinking configuration
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Claude's extended thinking feature",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=8192,
        help="Token budget for extended thinking (min 1024, default: %(default)s)",
    )

    parser.add_argument(
        "--output-dir",
        default="runs/bm25/claude",
        help="Directory to save request/response logs (default: %(default)s)",
    )
    parser.add_argument(
        "--query-template",
        choices=[
            "QUERY_TEMPLATE",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT",
            "QUERY_TEMPLATE_NO_GET_DOCUMENT_NO_CITATION",
            "QUERY_TEMPLATE_WITH_COMPACT",
            "QUERY_TEMPLATE_WITH_COMPACT_AND_GET_DOCUMENT",
        ],
        default="QUERY_TEMPLATE_NO_GET_DOCUMENT",
        help="Specify the query template to use (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for the model (default: use model defaults)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Top P for the model (default: use model defaults)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="Number of parallel threads for dataset processing (default: %(default)s)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=100,
        help="Maximum number of conversation rounds with function calls (default: %(default)s)",
    )
    parser.add_argument(
        "--max-rate-limit-retries",
        type=int,
        default=5,
        help="Maximum retries for rate limit errors (default: %(default)s)",
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=60.0,
        help="Fixed delay in seconds between rate limit retries (default: %(default)s)",
    )

    # Searcher configuration
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help=f"Type of searcher to use: {', '.join(SearcherType.get_choices())}",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Number of tokens to include for each document snippet in search results using Qwen/Qwen3-0.6B tokenizer (default: 512).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Fixed number of search results to return for all queries in this session (default: 5).",
    )
    parser.add_argument(
        "--get-document",
        action="store_true",
        help="If set, register both the search tool and the get_document tool.",
    )
    parser.add_argument(
        "--compact-model",
        type=str,
        default=None,
        help="Model to use for the compact (summarization) tool. Defaults to the same as --model.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        help="Hugging Face token for accessing private datasets/models.",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        help="Hugging Face home directory for caching models and datasets.",
    )

    # Dynamic searcher args
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    if args.hf_token:
        print(f"[DEBUG] Setting HF token from CLI argument: {args.hf_token[:10]}...")
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.hf_token

    if args.hf_home:
        print(f"[DEBUG] Setting HF home from CLI argument: {args.hf_home}")
        os.environ["HF_HOME"] = args.hf_home

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in environment or .env file")

    client = Anthropic(api_key=api_key, timeout=1200)

    # Validate thinking budget
    thinking_enabled = not args.no_thinking
    if thinking_enabled and args.thinking_budget < 1024:
        raise ValueError(
            "thinking_budget must be at least 1024 tokens when thinking is enabled."
        )

    thinking_config = None
    if thinking_enabled:
        thinking_config = {
            "type": "enabled",
            "budget_tokens": args.thinking_budget,
        }

    # Initialize searcher
    searcher = searcher_class(args)

    # Determine if compact tool should be enabled based on query template
    enable_compact = args.query_template in (
        "QUERY_TEMPLATE_WITH_COMPACT",
        "QUERY_TEMPLATE_WITH_COMPACT_AND_GET_DOCUMENT",
    ) or args.compact_model is not None

    tool_handler = SearchToolHandler(
        searcher=searcher,
        snippet_max_tokens=args.snippet_max_tokens,
        k=args.k,
        include_get_document=args.get_document,
        include_compact=enable_compact,
    )

    tools_registered = ["search"]
    if args.get_document:
        tools_registered.append("get_document")
    if enable_compact:
        tools_registered.append("compact")
    tools_str = ", ".join(tools_registered)

    print(
        f"Search agent started with {searcher.search_type} search (snippet_max_tokens={args.snippet_max_tokens}, k={args.k})"
    )
    print(f"Registered tools: {tools_str}")

    # If --query looks like a TSV path, process dataset
    if isinstance(args.query, str):
        qstr = args.query.strip()
        if qstr.lower().endswith(".tsv"):
            potential_path = Path(qstr)
            try:
                if potential_path.is_file():
                    _process_tsv_dataset(str(potential_path), client, args, tool_handler)
                    return
            except OSError:
                pass

    print("Sending request to Anthropic with native tool calling...")
    normalized_results, cumulative_usage, finish_reason, trajectory, sum_usage = run_conversation_with_tools(
        client,
        query=args.query,
        model=args.model,
        max_tokens=args.max_tokens,
        tool_handler=tool_handler,
        system_prompt=args.system,
        query_template=args.query_template,
        temperature=args.temperature,
        top_p=args.top_p,
        max_iterations=args.max_iterations,
        max_rate_limit_retries=args.max_rate_limit_retries,
        rate_limit_delay=args.rate_limit_delay,
        compact_model=args.compact_model,
        thinking_enabled=thinking_enabled,
        thinking_budget=args.thinking_budget,
    )

    _persist_response(
        args.output_dir,
        model=args.model,
        query_id=None,
        system_prompt=args.system,
        max_tokens=args.max_tokens,
        normalized_results=normalized_results,
        cumulative_usage=cumulative_usage,
        stop_reason=finish_reason,
        trajectory=trajectory,
        thinking_config=thinking_config,
        summarizer_usage=sum_usage,
    )

    # Print final output text if present
    final_texts = [item["output"] for item in normalized_results if item.get("type") == "output_text"]
    if final_texts:
        rprint(final_texts[-1])


if __name__ == "__main__":
    main()
