"""Select RL prompts from SFT-router rollouts (step 4 of the RL data pipeline).

Per domain: drop tasks that never succeed, keep tasks with 0 < success rate < 1, and among
always-solved tasks keep the top `--top-pct` percent by the variance of the total reward across
rollouts.
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from statistics import pvariance

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from router_interaction import infer_domain, normalize_tools  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSON written by extract_gt.py.")
    parser.add_argument("--rollouts", required=True, help="JSONL written by rollout.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-pct", type=float, default=30.0)
    args = parser.parse_args()

    with open(args.data) as f:
        records = json.load(f)
    rollouts = {}
    with open(args.rollouts) as f:
        for r in map(json.loads, f):
            if "error" not in r:
                rollouts[r["index"], r["rollout_id"]] = r
    by_task = defaultdict(list)
    for (index, _), r in rollouts.items():
        by_task[index].append(r)

    partial, solved = defaultdict(list), defaultdict(list)
    for index, rs in sorted(by_task.items()):
        domain = infer_domain(normalize_tools(json.loads(records[index]["tools"])))
        num_success = sum(r["success"] for r in rs)
        if num_success == len(rs):
            rewards = [r["total_reward"] for r in rs]
            solved[domain].append((pvariance(rewards) if len(rewards) > 1 else 0.0, index))
        elif num_success:
            partial[domain].append(index)

    kept = []
    for domain in sorted(set(partial) | set(solved)):
        ranked = sorted(solved[domain], key=lambda item: -item[0])
        top = ranked[: math.ceil(len(ranked) * args.top_pct / 100)]
        kept += partial[domain] + [index for _, index in top]
        print(f"{domain}: kept {len(partial[domain])} partially solved + {len(top)} / {len(ranked)} always solved tasks")
    with open(args.output, "w") as f:
        json.dump([records[i] for i in sorted(kept)], f, ensure_ascii=True)
    print(f"Wrote {len(kept)} tasks to {args.output}")


if __name__ == "__main__":
    main()
