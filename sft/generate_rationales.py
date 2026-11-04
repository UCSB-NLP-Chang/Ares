"""Add a teacher-written rationale ("thinking") to every router SFT example."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from litellm import completion
from tqdm import tqdm

RATIONALE_PROMPT = """You are generating chain-of-thought rationales for a router dataset. This router decides the minimal reasoning effort for a retail / airline assistant model to issue the next correct action, in order to achieve the reasoning efficiency.

You will receive:
- Conversation history up to the next action.
- The assistant guidance/policy for the domain (airline or retail).
- The gold minimal reasoning effort label: low, medium, or high.

Write a concise internal reasoning that justifies why the given label is the minimal required effort for the next action.
Guidelines:
- Ground the reasoning in the conversation context and the guidance.
- Keep it short (3-5 sentences).
- Do not change the label or propose a different one.
- Do not include any other keys or text.

Return only valid JSON:
{"thinking": "..."}"""


@lru_cache(maxsize=None)
def load_policy(source_file: str) -> str:
    try:
        with open(source_file) as f:
            return json.load(f).get("system", "")
    except Exception:
        return ""


def build_prompt(record) -> str:
    history = next(m["content"] for m in record["messages"] if m["role"] == "user")
    label = next(m["content"] for m in reversed(record["messages"]) if m["role"] == "assistant")
    policy = load_policy(record.get("metadata", {}).get("source_file", "")).strip()
    prompt = f"Conversation history:\n{history}\n\nMinimal reasoning effort label: {label}"
    return f"Assistant guidance:\n{policy}\n\n{prompt}" if policy else prompt


def parse_thinking(raw: str) -> str:
    raw = (raw or "").strip()
    for candidate in [raw, *raw.split("```")]:
        try:
            return json.loads(candidate.strip()).get("thinking") or raw
        except Exception:
            continue
    return raw


def generate(record, model: str, max_retries: int) -> str:
    messages = [{"role": "system", "content": RATIONALE_PROMPT}, {"role": "user", "content": build_prompt(record)}]
    for _ in range(max_retries):
        try:
            thinking = parse_thinking(completion(model=model, messages=messages).choices[0].message.content).strip()
            if thinking:
                return thinking
        except Exception as exc:
            print(f"retrying after error: {exc}")
        time.sleep(20)
    raise RuntimeError(f"failed to generate a rationale for {record.get('id')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL written by build_dataset.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5", help="litellm model name of the teacher.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]
    with ThreadPoolExecutor(args.workers) as pool:
        thinkings = list(tqdm(pool.map(lambda r: generate(r, args.model, args.max_retries), records), total=len(records)))
    with open(args.output, "w") as f:
        for record, thinking in zip(records, thinkings):
            f.write(json.dumps({**record, "thinking": thinking}) + "\n")


if __name__ == "__main__":
    main()
