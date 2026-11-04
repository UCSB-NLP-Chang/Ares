"""The ARES router: prompt, input formatting, output parsing, and an inference client."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

EFFORTS = ("low", "medium", "high")

ROUTER_SYSTEM_PROMPT = (
    "You are a router that selects the reasoning_effort (low, medium, or high) for the OSS model's *next* response.\n\n"
    "Context: The conversation involves an agent helping a user with tool calls. The OSS model charges more cost and latency at higher reasoning levels.\n"
    "- Choose HIGH if the next step likely needs careful reasoning, or complex tool sequencing.\n"
    "- Choose MEDIUM if the next step is moderately complex.\n"
    "- Choose LOW if the next step is straightforward and the risk is low.\n\n"
    "Produce exactly one word: low, medium, or high."
)


ROLE_PREFIX = {"human": "User", "gpt": "Assistant", "function_call": "AssistantToolCall", "observation": "Tool"}


def _loads(text: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return text


def render_turn(role: str, value: Any) -> str:
    """One history line; turns use the APIGen-MT roles (human / gpt / function_call / observation)."""
    if role == "function_call":
        call = _loads(value)
        if isinstance(call, dict):
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                arguments = _loads(arguments)
            value = f"{call.get('name', 'unknown_tool')} arguments={json.dumps(arguments, sort_keys=True, ensure_ascii=False)}"
    elif role == "observation":
        try:
            value = json.dumps(json.loads(value), sort_keys=True, ensure_ascii=False)
        except Exception:
            pass
    return f"{ROLE_PREFIX.get(role, role or 'Unknown')}: {value}".strip()


def render_history(turns: Iterable[Tuple[str, Any]], max_chars: int = 8000) -> str:
    """Router input: the task, interaction history, and latest observation, keeping the last `max_chars` characters."""
    return "\n".join(render_turn(role, value) for role, value in turns).strip()[-max_chars:]


def message_turns(messages: List[Dict[str, Any]]) -> List[Tuple[str, Any]]:
    """Convert OpenAI-style agent messages to APIGen-MT turns; the system prompt is dropped."""
    turns = []
    for msg in messages:
        role = msg.get("role")
        if role == "user":
            turns.append(("human", msg.get("content") or ""))
        elif role == "assistant" and msg.get("tool_calls"):
            for call in msg["tool_calls"]:
                function = call.get("function") or {}
                turns.append(("function_call", {"name": function.get("name"), "arguments": function.get("arguments")}))
        elif role == "assistant":
            turns.append(("gpt", msg.get("content") or ""))
        elif role == "tool":
            turns.append(("observation", msg.get("content") or ""))
    return turns


def format_router_input(messages: List[Dict[str, Any]], max_chars: int = 8000) -> str:
    return render_history(message_turns(messages), max_chars)


def build_router_messages(history: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
        {"role": "user", "content": history},
    ]


def parse_router_output(text: Optional[str], require_rationale: bool = False) -> Optional[str]:
    """Return the effort level of a router response, or None if it is malformed.

    The router responds with "<think>rationale</think> effort". With `require_rationale`, a response
    without the closed <think> block is malformed (the RL format reward); otherwise it is only dropped.
    """
    text = (text or "").replace("<|im_end|>", "")
    if require_rationale:
        match = re.fullmatch(r"\s*<think>.*?</think>(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        text = match.group(1)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
    text = text.strip().lower()
    return text if text in EFFORTS else None


class EffortRouter:
    """Queries a router served behind an OpenAI-compatible endpoint (e.g. sglang or vLLM)."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        api_key: str = "EMPTY",
        temperature: float = 0.6,
        max_tokens: int = 2048,
        max_retries: int = 2,
        fallback: str = "low",
    ) -> None:
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback = fallback

    def respond(self, messages: List[Dict[str, Any]]) -> str:
        """The router's raw response (rationale and effort) for the next step of the agent `messages`."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=build_router_messages(format_router_input(messages)),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def __call__(self, messages: List[Dict[str, Any]]) -> str:
        """Pick the effort for the next agent step; fall back to `fallback` on errors or malformed outputs."""
        try:
            effort = parse_router_output(self.respond(messages))
        except Exception as exc:
            print(f"Router call failed ({exc}); using {self.fallback} effort.")
            return self.fallback
        if effort is None:
            print(f"Router output invalid; using {self.fallback} effort.")
            return self.fallback
        return effort
