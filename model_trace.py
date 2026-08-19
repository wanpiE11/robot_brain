"""Readable console logging for the demo: full model input/output per call.

Wiring: attach `ModelTraceCallbackHandler` to the ChatOpenAI at construction in
main.py (``callbacks=[...]``). LangChain propagates constructor callbacks through
``bind_tools`` / ``with_structured_output``, so every planner / executor /
replanner model call fires ``on_chat_model_start`` / ``on_chat_model_end`` and
we can log the real, unparsed input and output.

Call ``setup_logging()`` after rai is imported (rai installs coloredlogs at
import time) to strip the ``hostname[pid]`` prefix and silence noisy loggers.
"""

import itertools
import json
import logging
import sys
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatResult, LLMResult

TRACE_LOGGER = "robot_trace"
LINE = "━" * 72

NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "openai", "langchain_openai", "urllib3")


def _force_utf8() -> None:
    """Emit UTF-8 so Chinese text and box-drawing chars never crash the console."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def setup_logging() -> None:
    """Reset root logging to a clean console format and silence noisy loggers."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    _force_utf8()

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    )
    root.addHandler(stream)
    root.setLevel(logging.INFO)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    trace = logging.getLogger(TRACE_LOGGER)
    trace.handlers.clear()
    trace.setLevel(logging.INFO)
    trace.propagate = False
    trace_handler = logging.StreamHandler(sys.stdout)
    trace_handler.setFormatter(logging.Formatter("%(message)s"))
    trace.addHandler(trace_handler)


def _content_to_text(content: Any) -> str:
    """Flatten a message content (str or multimodal content blocks) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type == "text":
                    parts.append(block.get("text", ""))
                elif block_type == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _role_label(msg: BaseMessage) -> str:
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, AIMessage):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool"
    return type(msg).__name__


def _tool_calls_repr(msg: BaseMessage) -> str:
    lines = []
    for tc in getattr(msg, "tool_calls", None) or []:
        name = tc.get("name", "?")
        args = tc.get("args", {})
        if isinstance(args, dict):
            try:
                args_json = json.dumps(args, ensure_ascii=False)
            except TypeError:
                args_json = str(args)
        else:
            args_json = str(args)
        lines.append(f"[调用工具] {name}({args_json})")
    return "\n".join(lines)


def _classify(messages: list[BaseMessage]) -> str:
    blob = "\n".join(_content_to_text(m.content) for m in messages)
    if "针对给定的目标" in blob:
        return "planner"
    if "当前计划按顺序的步骤依次为：" in blob:
        return "replanner"
    return "executor"


class ModelTraceCallbackHandler(BaseCallbackHandler):
    """Log the full raw input and output of every model call as a readable block."""

    def __init__(self) -> None:
        self._counter = itertools.count(1)
        self._starts: dict[UUID, float] = {}
        self._logged_start: set[UUID] = set()
        self._logged_end: set[UUID] = set()

    def _log(self, text: str) -> None:
        logging.getLogger(TRACE_LOGGER).info(text)

    # --- input ---

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        batch = messages[0] if messages else []
        self._on_start(run_id, batch)

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        self._on_start(run_id, [HumanMessage(content=p) for p in prompts])

    def _on_start(self, run_id: UUID, messages: list[BaseMessage]) -> None:
        if run_id in self._logged_start:
            return
        self._logged_start.add(run_id)
        number = next(self._counter)
        role = _classify(messages)
        self._starts[run_id] = time.monotonic()

        lines = [f"\n{LINE}", f"  LLM #{number} · {role}"]
        for i, msg in enumerate(messages, 1):
            text = _content_to_text(msg.content)
            if text:
                lines.append(f"· 输入 [{i}] {_role_label(msg)}")
                lines.append(text)
            tool_repr = _tool_calls_repr(msg)
            if tool_repr:
                lines.append(tool_repr)
            elif not text:
                lines.append(f"· 输入 [{i}] {_role_label(msg)}（空）")
        self._log("\n".join(lines))

    # --- output ---

    def on_chat_model_end(self, response: ChatResult, *, run_id: UUID, **kwargs: Any) -> Any:
        self._on_end(run_id, response)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Any:
        self._on_end(run_id, response)

    def _on_end(self, run_id: UUID, response: ChatResult | LLMResult) -> None:
        if run_id in self._logged_end:
            return
        self._logged_end.add(run_id)
        start = self._starts.pop(run_id, None)
        elapsed = f"{time.monotonic() - start:.1f}s" if start else "?"

        try:
            first = response.generations[0]
        except (IndexError, AttributeError):
            raw = str(response)
        else:
            # ChatResult.generations is flat; LLMResult.generations is nested.
            generation = first[0] if isinstance(first, list) else first
            raw = self._render_generation(generation)
        usage = self._usage(response)

        meta = " · ".join(part for part in (f"用时 {elapsed}", usage) if part)
        header = f"  · 输出（原始） · {meta}" if meta else "  · 输出（原始）"
        self._log(f"{header}\n{raw}\n{LINE}")

    @staticmethod
    def _render_generation(generation: Any) -> str:
        message = getattr(generation, "message", None)
        text = getattr(generation, "text", "") or ""
        if message is not None:
            content = _content_to_text(getattr(message, "content", "")) or ""
            tool_repr = _tool_calls_repr(message)
        else:
            content = text
            tool_repr = ""
        parts = [p for p in (content, tool_repr) if p]
        return "\n".join(parts) if parts else "(空输出)"

    @staticmethod
    def _usage(response: ChatResult | LLMResult) -> str | None:
        llm_output = getattr(response, "llm_output", None)
        if not isinstance(llm_output, dict):
            return None
        usage = llm_output.get("token_usage") or llm_output.get("usage")
        if isinstance(usage, dict) and usage.get("total_tokens"):
            return f"{usage['total_tokens']} tokens"
        return None
