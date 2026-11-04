# Copyright Sierra

import json
from litellm import completion
from typing import Callable, List, Optional, Dict, Any

from ares.oss import OSS_PROVIDER, litellm_kwargs, parse_oss_tool_call, sanitize_messages_for_oss, strip_oss_reasoning
from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME


class ToolCallingAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        router: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        # A fixed effort ("low" / "medium" / "high"), or "router" to let `router` pick it at every step.
        self.reasoning_effort = reasoning_effort
        self.router = router

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.wiki},
            {"role": "user", "content": obs},
        ]
        is_oss = self.provider == OSS_PROVIDER
        for _ in range(max_num_steps):
            effort = None
            if is_oss and self.reasoning_effort:
                effort = self.router(messages) if self.reasoning_effort == "router" else self.reasoning_effort
            call_kwargs: Dict[str, Any] = dict(
                messages=sanitize_messages_for_oss(messages) if is_oss else messages,
                model=self.model,
                tools=self.tools_info,
                temperature=self.temperature,
                **litellm_kwargs(self.provider),
            )
            if effort:
                call_kwargs["extra_body"] = {"reasoning_effort": effort}
            if self.max_tokens is not None:
                call_kwargs["max_tokens"] = self.max_tokens
            res = completion(**call_kwargs)
            raw_content = res.choices[0].message.content or ""
            next_message = res.choices[0].message.model_dump()
            if is_oss and not next_message.get("tool_calls"):
                tool_call = parse_oss_tool_call(raw_content)
                if tool_call:
                    next_message["tool_calls"] = [tool_call]
            if next_message.get("content"):
                content, thinking = strip_oss_reasoning(raw_content)
                next_message["oss_reasoning"] = thinking
                next_message["content"] = content or next_message["content"]
            if is_oss:
                next_message["oss_reasoning_effort"] = effort
            total_cost += res._hidden_params["response_cost"] or 0
            action = message_to_action(next_message)
            env_response = env.step(action)
            reward = env_response.reward
            info = {**info, **env_response.info.model_dump()}
            if action.name != RESPOND_ACTION_NAME:
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                messages.extend(
                    [
                        next_message,
                        {
                            "role": "tool",
                            "tool_call_id": next_message["tool_calls"][0]["id"],
                            "name": next_message["tool_calls"][0]["function"]["name"],
                            "content": env_response.observation,
                        },
                    ]
                )
            else:
                messages.extend(
                    [
                        next_message,
                        {"role": "user", "content": env_response.observation},
                    ]
                )
            if env_response.done:
                break
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})
