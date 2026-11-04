"""verl agent loop that trains the router on one sample per routing decision.

Each episode is unrolled step by step: the router sees [router prompt, current history], its
response is handed to `RouterInteraction`, and every step becomes its own training sample whose
reward is the total episode reward (GRPO advantages are computed over episodes of the same prompt).
"""

from __future__ import annotations

import asyncio
import copy
import sys
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import ray
from omegaconf import OmegaConf

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopManager,
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    _agent_loop_registry,
    get_trajectory_info,
)
from verl.experimental.agent_loop.utils import resolve_config_path
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.protocol import DataProto
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import RolloutTraceConfig, rollout_trace_attr


class RouterStepAgentLoop(AgentLoopBase):
    def __init__(self, trainer_config: DictConfigWrap, server_manager, tokenizer, processor,
                 dataset_cls, dataset_config, **kwargs):
        super().__init__(trainer_config, server_manager, tokenizer, processor, dataset_cls, dataset_config, **kwargs)
        config = trainer_config.config
        self.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        self.response_length = config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        self.interaction_map = initialize_interactions_from_config(
            config.actor_rollout_ref.rollout.multi_turn.interaction_config_path
        )

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        request_id = uuid4().hex
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")

        interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
        interaction = self.interaction_map[interaction_kwargs["name"]]
        await interaction.start_interaction(request_id, **interaction_kwargs)

        outputs: list[AgentLoopOutput] = []
        total_reward = 0.0
        done = False
        while not done:
            metrics: dict[str, Any] = {}
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            prompt_ids = prompt_ids[-self.prompt_length :]
            step_sampling = {**sampling_params, "max_tokens": sampling_params.get("max_tokens", self.response_length)}
            with simple_timer("generate_sequences", metrics):
                token_output = await self.server_manager.generate(
                    request_id=f"{request_id}_{len(outputs)}",
                    prompt_ids=prompt_ids,
                    sampling_params=step_sampling,
                    image_data=None,
                )
            response_ids = token_output.token_ids[: self.response_length]
            response_logprobs = None
            if token_output.log_probs is not None:
                response_logprobs = list(token_output.log_probs)[: len(response_ids)]
            response = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
            )

            done, observation, reward, info = await interaction.generate_response(
                request_id, messages + [{"role": "assistant", "content": response}], **interaction_kwargs
            )
            info = info or {}
            total_reward += float(reward or 0.0)
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_mask=[1] * len(response_ids),
                    response_logprobs=response_logprobs,
                    multi_modal_data={},
                    metrics=AgentLoopMetrics(**metrics),
                    extra_fields={"effort": info.get("effort"), "verify_success": info.get("verify_success")},
                )
            )
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": str(observation)}]

        await interaction.finalize_interaction(request_id)

        # Every step of the episode is rewarded with the total episode reward.
        for output in outputs:
            output.reward_score = total_reward
            output.num_turns = len(outputs)
        return outputs


@ray.remote
class RouterStepAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker that flattens the per-step outputs of each episode into one batch."""

    def _ensure_agent_loop_registry(self, agent_name: str) -> None:
        if agent_name in _agent_loop_registry:
            return
        config_path = self.config.actor_rollout_ref.rollout.agent.agent_loop_config_path
        agent_loop_configs = OmegaConf.load(resolve_config_path(config_path))
        if OmegaConf.is_dict(agent_loop_configs):
            agent_loop_configs = [agent_loop_configs]
        for agent_loop_config in agent_loop_configs:
            _agent_loop_registry[agent_loop_config.name] = agent_loop_config

    async def generate_sequences(self, batch: DataProto) -> DataProto:
        config = self.config.actor_rollout_ref.rollout
        validate = batch.meta_info.get("validate", False)
        sampling = config.val_kwargs if validate else config
        temperature = batch.meta_info.get("temperature", sampling.temperature)
        top_p = batch.meta_info.get("top_p", sampling.top_p)
        top_k = batch.meta_info.get("top_k", sampling.top_k)
        if not batch.meta_info.get("do_sample", sampling.do_sample):
            temperature, top_p, top_k = 0.0, 1.0, -1
        sampling_params = dict(
            temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=1.0, logprobs=config.calculate_log_probs
        )

        if "agent_name" not in batch.non_tensor_batch:
            batch.non_tensor_batch["agent_name"] = np.array([config.agent.default_agent_loop] * len(batch), dtype=object)
        index = batch.non_tensor_batch.get("index", np.arange(len(batch)))

        max_traced = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker
        traced = set(range(len(batch)))
        if max_traced is not None and max_traced < len(np.unique(index)):
            selected = set(np.random.choice(np.unique(index), max_traced, replace=False).tolist())
            traced = {i for i in range(len(batch)) if index[i] in selected}
        trajectory_info = await get_trajectory_info(batch.meta_info.get("global_steps", -1), index.tolist(), validate)

        drop_failed = bool((config.get("custom") or {}).get("drop_failed_samples", False))

        async def run_one(i: int):
            kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
            try:
                return await self._run_agent_loop(sampling_params, trajectory_info[i], trace=i in traced, **kwargs)
            except Exception as exc:
                if not drop_failed:
                    raise
                print(f"[router_rollout] dropped sample index={index[i]} error={type(exc).__name__}: {exc}", file=sys.stderr)
                return None

        results = await asyncio.gather(*(run_one(i) for i in range(len(batch))))
        results = [r for r in results if r]
        if not results:
            raise RuntimeError("All rollouts failed; no samples to process.")
        if validate:  # validation scores one sample per episode
            return self._postprocess([r[0] for r in results])
        return self._postprocess([step for r in results for step in r])

    async def _run_agent_loop(self, sampling_params, trajectory, *, agent_name: str, trace: bool = True, **kwargs):
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            self._ensure_agent_loop_registry(agent_name)
            agent_loop = hydra.utils.instantiate(
                config=_agent_loop_registry[agent_name],
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            outputs = await agent_loop.run(sampling_params, **kwargs)
            processed = []
            for output in outputs:
                output.extra_fields["uid"] = kwargs["uid"]  # all steps of all rollouts of a prompt form one GRPO group
                result = await self._agent_loop_postprocess(output, **kwargs)
                result.extra_fields.pop("raw_prompt", None)
                processed.append(result)
            return processed


class RouterStepAgentLoopManager(AgentLoopManager):
    def __init__(self, config: OmegaConf, worker_group=None, rm_resource_pool=None):
        self.agent_loop_workers_class = RouterStepAgentLoopWorker
        super().__init__(config, worker_group, rm_resource_pool)
