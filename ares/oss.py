"""Helpers for gpt-oss served behind an OpenAI-compatible endpoint (e.g. sglang).

gpt-oss emits Harmony-formatted text:
  thinking   <|channel|>analysis<|message|>...<|end|>
  tool call  <|channel|>commentary to=TOOL_NAME <|message|>{JSON}<|call|>
  reply      <|channel|>final<|message|>...<|end|>
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

OSS_PROVIDER = "oss"
THINK_START_TOKEN = "<|channel|>analysis<|message|>"
_TAG = re.compile(r"<\|.*?\|>")


def oss_base_url() -> str:
    return os.getenv("OSS_OPENAI_BASE_URL", "http://localhost:30000/v1")


def oss_api_key() -> str:
    return os.getenv("OSS_OPENAI_API_KEY", "EMPTY")


def litellm_kwargs(provider: str) -> Dict[str, Any]:
    """Route `provider="oss"` to the OpenAI-compatible gpt-oss server; pass other litellm providers through."""
    if provider == OSS_PROVIDER:
        return {"custom_llm_provider": "openai", "api_base": oss_base_url(), "api_key": oss_api_key()}
    return {"custom_llm_provider": provider}


def strip_oss_reasoning(content: str) -> Tuple[str, str]:
    """Split a gpt-oss response into (final_content, thinking)."""
    if not content:
        return "", ""

    reasoning_raw = ""
    final_raw = content
    if THINK_START_TOKEN in content:
        after_start = content.split(THINK_START_TOKEN, 1)[1]
        if "<|end|>" in after_start:
            reasoning_raw, final_raw = after_start.split("<|end|>", 1)
        else:
            reasoning_raw, final_raw = after_start, ""

    # The final message body is the last <|message|> segment, up to <|call|> or <|end|>.
    msg_idx = content.rfind("<|message|>")
    if msg_idx != -1:
        after_msg = content[msg_idx + len("<|message|>") :]
        final_raw = after_msg.split("<|call|>", 1)[0] if "<|call|>" in after_msg else after_msg.split("<|end|>", 1)[0]

    return _TAG.sub("", final_raw).strip(), _TAG.sub("", reasoning_raw).strip()


def sanitize_messages_for_oss(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove Harmony tags and thinking from message contents before sending them back to gpt-oss."""
    sanitized = []
    for msg in messages:
        msg = dict(msg)
        content = msg.get("content")
        if isinstance(content, str):
            cleaned, _ = strip_oss_reasoning(content)
            msg["content"] = _TAG.sub("", cleaned) or _TAG.sub("", content)
        sanitized.append(msg)
    return sanitized


def parse_oss_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Recover a tool call that the server left in the text, e.g. `commentary to=functions.NAME json{...}`."""
    to_idx = text.rfind("to=")
    if to_idx == -1:
        return None
    func_segment = text[to_idx + 3 :].split("<|", 1)[0].strip()
    func_token = func_segment.split()[0].split("{", 1)[0] if func_segment else ""
    if not func_token:
        return None
    name = func_token.split(".")[-1]

    start = text.find("{", to_idx)
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    args = json.loads(text[start : i + 1])
                except Exception:
                    return None
                return {
                    "id": "oss-tool-call",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
    return None
