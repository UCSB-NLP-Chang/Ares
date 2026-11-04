# Copyright Sierra

import argparse
from tau_bench.types import RunConfig
from tau_bench.run import run
from litellm import provider_list
from tau_bench.envs.user import UserStrategy
from ares.oss import OSS_PROVIDER
from ares.router import EFFORTS


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument(
        "--env", type=str, choices=["retail", "airline"], default="retail"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-20b",
        help="The model to use for the agent",
    )
    parser.add_argument(
        "--model-provider",
        type=str,
        default=OSS_PROVIDER,
        choices=provider_list + [OSS_PROVIDER],
        help="The model provider for the agent; `oss` targets the server at $OSS_OPENAI_BASE_URL",
    )
    parser.add_argument(
        "--user-model",
        type=str,
        default="gpt-4o",
        help="The model to use for the user simulator",
    )
    parser.add_argument(
        "--user-model-provider",
        type=str,
        default="openai",
        choices=provider_list,
        help="The model provider for the user simulator",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="The sampling temperature for the action model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens per agent completion",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        choices=[*EFFORTS, "router"],
        help="A fixed reasoning effort for gpt-oss, or `router` to select it with the ARES router at every step",
    )
    parser.add_argument("--router-base-url", type=str, default="http://localhost:30001/v1")
    parser.add_argument("--router-model", type=str, default="default")
    parser.add_argument("--router-temperature", type=float, default=0.6)
    parser.add_argument("--router-max-tokens", type=int, default=2048)
    parser.add_argument(
        "--task-split",
        type=str,
        default="test",
        choices=["train", "test", "dev"],
        help="The split of tasks to run (only applies to the retail domain for now",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=-1, help="Run all tasks if -1")
    parser.add_argument("--task-ids", type=int, nargs="+", help="(Optional) run only the tasks with the given IDs")
    parser.add_argument("--log-dir", type=str, default="results")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of tasks to run in parallel",
    )
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=0)
    parser.add_argument("--user-strategy", type=str, default="llm", choices=[item.value for item in UserStrategy])
    args = parser.parse_args()
    print(args)
    return RunConfig(
        model_provider=args.model_provider,
        user_model_provider=args.user_model_provider,
        model=args.model,
        user_model=args.user_model,
        num_trials=args.num_trials,
        env=args.env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        router_base_url=args.router_base_url,
        router_model=args.router_model,
        router_temperature=args.router_temperature,
        router_max_tokens=args.router_max_tokens,
        task_split=args.task_split,
        start_index=args.start_index,
        end_index=args.end_index,
        task_ids=args.task_ids,
        log_dir=args.log_dir,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        shuffle=args.shuffle,
        user_strategy=args.user_strategy,
    )


def main():
    config = parse_args()
    run(config)


if __name__ == "__main__":
    main()
