"""Roll out the SFT router N times per task to score prompts for RL (step 3 of the RL data pipeline).

The router is served behind an OpenAI-compatible endpoint; episodes run in the same environment
as RL training (rl/router_interaction.py). Writes one JSON line per rollout and resumes from an
existing output file.
"""

import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ares.router import EffortRouter  # noqa: E402
from router_interaction import RouterInteraction  # noqa: E402


async def rollout(interaction: RouterInteraction, router: EffortRouter, index: int, rollout_id: int, record: dict) -> dict:
    instance_id = uuid4().hex
    first_user_message = next((t["value"] for t in record["conversations"] if t.get("from") == "human"), None)
    await interaction.start_interaction(
        instance_id,
        instruction=record["instruction"],
        tools=record["tools"],
        system=record["system"],
        initial_user_message=first_user_message,
        actions=record["actions"],
        outputs=record["outputs"],
    )
    try:
        episode = interaction._instances[instance_id]
        done, info = False, {}
        while not done:
            response = router.respond(episode.agent_messages)
            done, _, _, info = await interaction.generate_response(instance_id, [{"role": "assistant", "content": response}])
        return {
            "index": index,
            "rollout_id": rollout_id,
            "success": bool(info.get("verify_success")),
            "efforts": [r["effort"] for r in episode.turn_records],
            "total_reward": sum(r["reward"] for r in episode.turn_records),
            "trajectory": episode.conversation_log,
        }
    finally:
        await interaction.finalize_interaction(instance_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON written by extract_gt.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n", type=int, default=8, help="Rollouts per task.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--interaction-config", default=os.path.join(os.path.dirname(__file__), "..", "config", "interaction.yaml"))
    parser.add_argument("--router-base-url", default="http://localhost:30001/v1")
    parser.add_argument("--router-model", default="default")
    parser.add_argument("--router-temperature", type=float, default=0.0)
    parser.add_argument("--router-max-tokens", type=int, default=512)
    args = parser.parse_args()

    with open(args.interaction_config) as f:
        config = yaml.safe_load(f)["interaction"][0]["config"]
    config.update(log_dir=None)
    interaction = RouterInteraction(config)
    router = EffortRouter(
        base_url=args.router_base_url,
        model=args.router_model,
        temperature=args.router_temperature,
        max_tokens=args.router_max_tokens,
        max_retries=10,
    )

    with open(args.input) as f:
        records = json.load(f)
    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {(r["index"], r["rollout_id"]) for r in map(json.loads, f) if "error" not in r}
    jobs = [(i, k) for i in range(len(records)) for k in range(args.n) if (i, k) not in done]

    def work(job):
        index, rollout_id = job
        try:
            return asyncio.run(rollout(interaction, router, index, rollout_id, records[index]))
        except Exception as exc:
            return {"index": index, "rollout_id": rollout_id, "error": f"{type(exc).__name__}: {exc}"}

    with open(args.output, "a") as out, ThreadPoolExecutor(args.workers) as pool:
        for result in tqdm(pool.map(work, jobs), total=len(jobs)):
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
