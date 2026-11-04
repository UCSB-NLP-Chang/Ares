"""Convert selected tasks into verl parquet files (step 5 of the RL data pipeline)."""

import argparse
import json
import os

import datasets

from ares.router import ROUTER_SYSTEM_PROMPT, format_router_input


def build_row(record, index: int, split: str) -> dict:
    tools = json.loads(record["tools"]) if isinstance(record["tools"], str) else record["tools"]
    tools = [t if "type" in t and "function" in t else {"type": "function", "function": t} for t in tools]
    first_user_message = next((t["value"] for t in record["conversations"] if t.get("from") in ("human", "user")), "")
    history = format_router_input([{"role": "user", "content": first_user_message}])
    return {
        "data_source": "apigen_router",
        "prompt": [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}, {"role": "user", "content": history}],
        "extra_info": {
            "split": split,
            "index": index,
            "interaction_kwargs": {
                "name": "ares_router",
                "instruction": record["instruction"],
                "system": record["system"],
                "initial_user_message": first_user_message,
                # Nested structures are JSON-encoded: parquet would pad dicts with None-valued keys.
                "tools": json.dumps(tools, ensure_ascii=True),
                "actions": json.dumps(record["actions"], ensure_ascii=True),
                "outputs": json.dumps(record["outputs"], ensure_ascii=True),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON written by filter_prompts.py.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-count", type=int, default=10)
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    val_count = max(1, min(len(records) - 1, args.val_count))
    os.makedirs(args.out_dir, exist_ok=True)
    for split, rows in (("train", records[:-val_count]), ("val", records[-val_count:])):
        data = [build_row(r, i, split) for i, r in enumerate(rows)]
        datasets.Dataset.from_list(data).to_parquet(os.path.join(args.out_dir, f"{split}.parquet"))


if __name__ == "__main__":
    main()
