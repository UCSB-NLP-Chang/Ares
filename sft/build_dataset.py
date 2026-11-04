"""Turn labeled trajectories into router SFT examples (one per labeled agent step).

Each example is [system: router prompt, user: history before the step, assistant: effort label].
"""

import argparse
import json
import random
from pathlib import Path

from ares.router import ROUTER_SYSTEM_PROMPT, render_history


def build_examples(path: Path):
    conversations = json.loads(path.read_text())["conversations"]
    for idx, turn in enumerate(conversations):
        label = turn.get("minimal_reasoning_effort_needed")
        history = render_history((t.get("from", ""), t.get("value", "")) for t in conversations[:idx])
        if not label or not history:
            continue
        yield {
            "id": f"{path.stem}_turn_{idx}",
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": history},
                {"role": "assistant", "content": label},
            ],
            "metadata": {"source_file": str(path), "turn_index": idx, "label_type": turn.get("label_type")},
        }


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/labeled"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = [r for path in sorted(args.data_dir.glob("*.json")) for r in build_examples(path)]
    random.Random(args.seed).shuffle(records)
    eval_size = min(max(1, int(len(records) * args.eval_ratio)), len(records) - 1)
    write_jsonl(args.output_dir / "eval.jsonl", records[:eval_size])
    write_jsonl(args.output_dir / "train.jsonl", records[eval_size:])
    print(f"Wrote {len(records) - eval_size} train and {eval_size} eval examples to {args.output_dir}")


if __name__ == "__main__":
    main()
