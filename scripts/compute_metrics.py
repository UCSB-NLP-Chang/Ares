"""Accuracy and reasoning-token metrics for TAU-Bench result files written by run.py.

Each result file is one run; metrics are reported per run and averaged over runs:
  Acc      success rate (%)
  S        agent steps per task
  T_total  reasoning tokens over the whole benchmark
  T_task   reasoning tokens per task
  T_step   reasoning tokens per agent step

Usage: python scripts/compute_metrics.py results/run1.json results/run2.json ...
"""

import argparse
import json
from statistics import mean

from transformers import AutoTokenizer


def run_metrics(results, tokenizer):
    steps = reasoning_steps = tokens = 0
    for result in results:
        for msg in result["traj"]:
            if msg.get("role") == "assistant":
                steps += 1
            if msg.get("oss_reasoning") is not None:
                reasoning_steps += 1
                tokens += len(tokenizer.encode(msg["oss_reasoning"], add_special_tokens=False))
    n = len(results)
    return {
        "Acc": 100 * sum(abs(r["reward"] - 1) <= 1e-6 for r in results) / n,
        "S": steps / n,
        "T_total": tokens,
        "T_task": tokens / n,
        "T_step": tokens / max(reasoning_steps, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    runs = []
    for path in args.paths:
        with open(path) as f:
            runs.append(run_metrics(json.load(f), tokenizer))
        print(path, {k: round(v, 1) for k, v in runs[-1].items()})
    if len(runs) > 1:
        print("mean", {k: round(mean(r[k] for r in runs), 1) for k in runs[0]})


if __name__ == "__main__":
    main()
