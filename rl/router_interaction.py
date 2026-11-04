"""verl interaction for RL on the ARES router.

At every turn the router's output selects the reasoning effort for one gpt-oss agent step in a
tau-bench environment (APIGen-MT tasks, LLM user simulator). When the episode ends, success is
checked tau-bench style (final database state and required outputs) and the reward is

    R = R_out + mean_t c(e_t)   on success
    R = R_fail                  on failure
    R = R_form + R_fail         if the router output is malformed (the episode is terminated).

Per-turn rewards are c(e_t) and the final turn absorbs the remainder, so they sum to R.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Optional
from uuid import uuid4

from litellm import completion
from litellm import exceptions as litellm_exceptions
from verl.interactions.base import BaseInteraction

from ares.oss import OSS_PROVIDER, litellm_kwargs, parse_oss_tool_call, sanitize_messages_for_oss, strip_oss_reasoning
from ares.router import format_router_input, parse_router_output
from tau_bench.envs.airline.data import load_data as load_airline_data
from tau_bench.envs.airline.tools import ALL_TOOLS as AIRLINE_TOOLS
from tau_bench.envs.retail.data import load_data as load_retail_data
from tau_bench.envs.retail.tools import ALL_TOOLS as RETAIL_TOOLS

DOMAINS = {
    "airline": (load_airline_data, {t.get_info()["function"]["name"]: t for t in AIRLINE_TOOLS}),
    "retail": (load_retail_data, {t.get_info()["function"]["name"]: t for t in RETAIL_TOOLS}),
}

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=8)


def call_llm(call_kwargs: dict[str, Any]):
    """litellm completion with a hard wall-clock timeout of `call_kwargs["timeout"]` seconds."""
    timeout = call_kwargs.get("timeout")
    if not timeout:
        return completion(**call_kwargs)
    future = _EXECUTOR.submit(completion, **call_kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise litellm_exceptions.Timeout(f"completion timed out after {timeout}s") from exc


def infer_domain(tools: list[dict[str, Any]]) -> str:
    names = {t.get("function", {}).get("name") for t in tools}
    if names & (DOMAINS["airline"][1].keys() - DOMAINS["retail"][1].keys()):
        return "airline"
    if names & (DOMAINS["retail"][1].keys() - DOMAINS["airline"][1].keys()):
        return "retail"
    return "airline"


def data_hash(data: Any) -> str:
    def hashable(item):
        if isinstance(item, dict):
            return tuple((k, hashable(v)) for k, v in sorted(item.items()))
        if isinstance(item, list):
            return tuple(hashable(v) for v in item)
        if isinstance(item, set):
            return tuple(sorted(hashable(v) for v in item))
        return item

    return sha256(str(hashable(data)).encode("utf-8")).hexdigest()


def normalize_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def clean(value):
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if v is not None}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    tools = [t if "type" in t and "function" in t else {"type": "function", "function": t} for t in tools]
    for tool in tools:
        if "parameters" in tool["function"]:
            tool["function"]["parameters"] = clean(tool["function"]["parameters"])
    return tools


class LLMUserSimulator:
    def __init__(self, model: str, provider: str, instruction: str, initial_message: Optional[str],
                 temperature: float, max_tokens: Optional[int], timeout: Optional[float]) -> None:
        self.model, self.provider = model, provider
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        instruction_display = f"\n\nInstruction: {instruction}\n" if instruction else ""
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": (
                "You are a user interacting with an agent."
                f"{instruction_display}"
                "\nRules:\n"
                "- Just generate one line at a time to simulate the user's message.\n"
                "- Do not give away all the instruction at once. Only provide the information that is necessary for the current step.\n"
                "- Do not hallucinate information that is not provided in the instruction. For example, if the agent asks for the order id but it is not mentioned in the instruction, do not make up an order id, just say you do not remember or have it.\n"
                "- If the instruction goal is satisified, generate '###STOP###' as a standalone message without anything else to end the conversation.\n"
                "- Do not repeat the exact instruction in the conversation. Instead, use your own words to convey the same information.\n"
                "- Try to make the conversation as natural as possible, and stick to the personalities in the instruction."
            )},
            {"role": "user", "content": "Hi! How can I help you today?"},
        ]
        if initial_message:
            self.messages.append({"role": "assistant", "content": initial_message})
            self.initial_message = initial_message
        else:
            self.initial_message = self._generate()

    def _generate(self) -> str:
        call_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": sanitize_messages_for_oss(self.messages) if self.provider == OSS_PROVIDER else self.messages,
            "temperature": self.temperature,
            **litellm_kwargs(self.provider),
        }
        if self.max_tokens is not None:
            call_kwargs["max_tokens"] = self.max_tokens
        if self.timeout:
            call_kwargs["timeout"] = self.timeout
        try:
            message = call_llm(call_kwargs).choices[0].message
        except litellm_exceptions.ContentPolicyViolationError:
            self.messages.append({"role": "assistant", "content": "###STOP###"})
            return "###STOP###"
        content, _ = strip_oss_reasoning(message.content or "")
        reply = message.model_dump()
        reply["content"] = content or reply.get("content")
        self.messages.append(reply)
        return reply["content"]

    def step(self, agent_message: str) -> str:
        self.messages.append({"role": "user", "content": agent_message})
        return self._generate()


@dataclass
class Episode:
    instruction: str
    tools: list[dict[str, Any]]
    domain: str
    data: dict[str, Any]
    user_sim: LLMUserSimulator
    gt_actions: list[dict[str, Any]]
    gt_outputs: list[str]
    agent_messages: list[dict[str, Any]]
    conversation_log: list[dict[str, Any]]
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False


class RouterInteraction(BaseInteraction):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self._instances: dict[str, Episode] = {}
        self.oss_model = config.get("oss_model", "openai/gpt-oss-20b")
        self.oss_temperature = float(config.get("oss_temperature", 1.0))
        self.oss_max_tokens = config.get("oss_max_tokens")
        self.user_model = config.get("user_model", "gpt-4o")
        self.user_provider = config.get("user_provider", "openai")
        self.user_temperature = float(config.get("user_temperature", 1.0))
        self.user_max_tokens = config.get("user_max_tokens")
        self.timeout = config.get("request_timeout")
        self.max_total_steps = int(config.get("max_total_steps", 30))
        self.max_history_chars = int(config.get("max_history_chars", 8000))
        self.success_reward = float(config.get("success_reward", 5.0))
        self.failure_reward = float(config.get("failure_reward", 0.0))
        self.format_penalty = float(config.get("format_penalty", -1.0))
        self.penalties = dict(config.get("penalties", {"low": -0.2, "medium": -0.5, "high": -1.0}))
        self.terminate_tools = set(config.get("terminate_tools", ["transfer_to_human_agents"]))
        self.log_path = None
        if config.get("log_dir"):
            os.makedirs(config["log_dir"], exist_ok=True)
            self.log_path = os.path.join(config["log_dir"], "router_rollouts.jsonl")

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        *,
        instruction: str,
        tools: list[dict[str, Any]] | str,
        system: str,
        initial_user_message: Optional[str] = None,
        actions: Optional[list[dict[str, Any]] | str] = None,
        outputs: Optional[list[str] | str] = None,
        **kwargs,
    ) -> str:
        instance_id = instance_id or str(uuid4())
        tools = normalize_tools(json.loads(tools) if isinstance(tools, str) else list(tools))
        # Actions/outputs are stored as JSON strings: parquet would pad argument dicts with None-valued keys.
        actions = json.loads(actions) if isinstance(actions, str) else actions
        outputs = json.loads(outputs) if isinstance(outputs, str) else outputs
        domain = infer_domain(tools)
        user_sim = LLMUserSimulator(
            self.user_model, self.user_provider, instruction, initial_user_message,
            self.user_temperature, self.user_max_tokens, self.timeout,
        )
        first_message = initial_user_message or user_sim.initial_message
        self._instances[instance_id] = Episode(
            instruction=instruction,
            tools=tools,
            domain=domain,
            data=copy.deepcopy(DOMAINS[domain][0]()),
            user_sim=user_sim,
            gt_actions=[{"name": a["name"], "arguments": a.get("arguments") or {}} for a in (actions or [])],
            gt_outputs=[str(o) for o in (outputs or []) if o is not None],
            agent_messages=[{"role": "system", "content": system}, {"role": "user", "content": first_message}],
            conversation_log=[{"from": "human", "value": first_message}],
        )
        return instance_id

    async def generate_response(
        self, instance_id: str, messages: list[dict[str, Any]], **kwargs
    ) -> tuple[bool, str, float, dict[str, Any]]:
        episode = self._instances[instance_id]
        if episode.done:
            return True, self._observation(episode), 0.0, {}

        response = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "assistant"), "")
        effort = parse_router_output(response, require_rationale=True)
        if effort is None:
            reward = self._final_reward(episode, self.format_penalty + self.failure_reward)
            self._record(episode, "invalid", self.penalties["low"], reward, done=True)
            self._log(episode, success=False, detail="invalid_format")
            return True, self._observation(episode), reward, self._info("invalid", False, False)

        penalty = self.penalties[effort]
        reply, called_tool, terminated = self._agent_step(episode, effort)
        if called_tool:
            done = terminated
        else:
            user_reply = episode.user_sim.step(reply)
            episode.conversation_log.append({"from": "human", "value": user_reply})
            episode.agent_messages.append({"role": "user", "content": user_reply})
            done = "###STOP###" in user_reply
        done = done or len(episode.turn_records) + 1 >= self.max_total_steps

        reward, success, detail = penalty, None, None
        if done:
            success, detail = self._verify(episode)
            if success:
                mean_penalty = (penalty + sum(r["penalty"] for r in episode.turn_records)) / (len(episode.turn_records) + 1)
                total = self.success_reward + mean_penalty
            else:
                total = self.failure_reward
            reward = self._final_reward(episode, total)
        self._record(episode, effort, penalty, reward, done)
        if done:
            self._log(episode, success, detail)
        return done, self._observation(episode), reward, self._info(effort, True, success)

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)

    def _final_reward(self, episode: Episode, total: float) -> float:
        """Reward of the last turn, chosen so that the turn rewards of the episode sum to `total`."""
        return total - sum(r["reward"] for r in episode.turn_records)

    def _record(self, episode: Episode, effort: str, penalty: float, reward: float, done: bool) -> None:
        episode.turn_records.append({"effort": effort, "penalty": penalty, "reward": reward, "done": done})
        episode.done = done

    def _info(self, effort: str, format_valid: bool, success: Optional[bool]) -> dict[str, Any]:
        return {
            "effort": effort,
            "format_valid": format_valid,
            "format_penalty": 0.0 if format_valid else self.format_penalty,
            "verify_success": success,
        }

    def _observation(self, episode: Episode) -> str:
        return format_router_input(episode.agent_messages, self.max_history_chars)

    def _agent_step(self, episode: Episode, effort: str) -> tuple[str, bool, bool]:
        """Run one gpt-oss step at `effort`. Returns (reply, called_tool, terminated)."""
        call_kwargs: dict[str, Any] = {
            "model": self.oss_model,
            "messages": sanitize_messages_for_oss(episode.agent_messages),
            "tools": episode.tools,
            "temperature": self.oss_temperature,
            "extra_body": {"reasoning_effort": effort},
            **litellm_kwargs(OSS_PROVIDER),
        }
        if self.oss_max_tokens is not None:
            call_kwargs["max_tokens"] = self.oss_max_tokens
        if self.timeout:
            call_kwargs["timeout"] = self.timeout
        message = call_llm(call_kwargs).choices[0].message.model_dump()
        raw = message.get("content") or ""
        tool_call = message["tool_calls"][0] if message.get("tool_calls") else parse_oss_tool_call(raw)

        if tool_call is None:
            reply = strip_oss_reasoning(raw)[0] or raw or "I'm unable to continue with this request."
            episode.conversation_log.append({"from": "gpt", "value": reply})
            episode.agent_messages.append({"role": "assistant", "content": reply})
            return reply, False, False

        name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", "{}")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        else:
            args = args if isinstance(args, dict) else {}
            tool_call = {**tool_call, "function": {**tool_call["function"], "arguments": json.dumps(args)}}
        tool = DOMAINS[episode.domain][1].get(name)
        try:
            result = tool.invoke(episode.data, **args) if tool else f"Error: tool not found: {name}"
        except Exception as exc:
            result = f"Error: {exc}"
        episode.conversation_log.append({"from": "function_call", "value": json.dumps({"name": name, "arguments": args})})
        episode.conversation_log.append({"from": "observation", "value": result})
        episode.agent_messages.append({"role": "assistant", "tool_calls": [tool_call]})
        episode.agent_messages.append({"role": "tool", "tool_call_id": tool_call.get("id", "tool-call"), "name": name, "content": result})
        return "", True, name in self.terminate_tools

    def _verify(self, episode: Episode) -> tuple[bool, str]:
        """tau-bench check: same final database as replaying the reference actions, and all required outputs said."""
        gt_data = copy.deepcopy(DOMAINS[episode.domain][0]())
        for action in episode.gt_actions:
            tool = DOMAINS[episode.domain][1].get(action["name"])
            if tool is None or action["name"] in self.terminate_tools:
                continue
            try:
                tool.invoke(gt_data, **action["arguments"])
            except Exception:
                pass
        actions_ok = data_hash(episode.data) == data_hash(gt_data)
        replies = [str(t["value"]).lower().replace(",", "") for t in episode.conversation_log if t["from"] == "gpt"]
        missing = [o for o in episode.gt_outputs if o and not any(o.lower() in r for r in replies)]
        detail = json.dumps({"actions_ok": actions_ok, "outputs_ok": not missing, "missing_outputs": missing})
        return actions_ok and not missing, detail

    def _log(self, episode: Episode, success: Optional[bool], detail: str) -> None:
        total = sum(r["reward"] for r in episode.turn_records)
        print(f"[router_rollout] turns={len(episode.turn_records)} reward={total:.2f}")
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "instruction": episode.instruction,
                    "verify_success": success,
                    "verify_detail": detail,
                    "total_reward": total,
                    "turn_records": episode.turn_records,
                    "conversation": episode.conversation_log,
                }) + "\n")
