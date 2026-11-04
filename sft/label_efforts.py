"""Label the minimal sufficient reasoning effort for every agent step of APIGen-MT.

For each agent turn (reply or tool call), gpt-oss is sampled K times at each of low/medium/high
effort on the reference history, and an LLM judge marks which samples match the reference action.
An effort is sufficient if at least M of its K samples match; the label is the lowest sufficient
effort. If no effort is sufficient, the label falls back to the lowest effort with the highest
non-zero match count, and the step is discarded (left unlabeled) if no sample matches at all.
One JSON file is written per trajectory.
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from litellm import completion
from openai import OpenAI
from tqdm import tqdm

from ares.oss import oss_api_key, oss_base_url, parse_oss_tool_call, sanitize_messages_for_oss, strip_oss_reasoning
from ares.router import EFFORTS


def call_oss(client: OpenAI, model: str, messages, tools, effort: str, temperature: float) -> Dict[str, Any]:
    """Query gpt-oss once and return {"type": "function_call", "tool_call": ...} or {"type": "assistant", "text": ...}."""
    resp = client.chat.completions.create(
        model=model,
        messages=sanitize_messages_for_oss(messages),
        temperature=temperature,
        tools=tools,
        extra_body={"skip_special_tokens": False, "max_completion_tokens": 4096, "reasoning_effort": effort},
    )
    message = resp.choices[0].message
    raw = message.content or ""
    tool_call = message.tool_calls[0].model_dump() if message.tool_calls else parse_oss_tool_call(raw)
    candidate: Dict[str, Any] = {"type": "function_call" if tool_call else "assistant"}
    if tool_call:
        args = tool_call["function"].get("arguments", {})
        try:
            args = json.loads(args) if isinstance(args, str) else args
        except Exception:
            pass
        candidate["tool_call"] = {"name": tool_call["function"].get("name", ""), "arguments": args}
    content, _ = strip_oss_reasoning(raw)
    if content or raw:
        candidate["text"] = content or raw
    return candidate


def parse_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    for candidate in [text, *text.split("```")]:
        try:
            return json.loads(candidate.strip())
        except Exception:
            continue
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}


def judge(judge_model: str, gold: Dict[str, Any], candidates: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
    """Ask the judge which candidates match the reference action; type mismatches never match."""
    if gold["type"] == "function_call":
        system_prompt = (
            "You are a strict evaluator of tool/function calls for a customer-service agent.\n"
            "Mark a candidate as correct only if it calls the same function with identical values for the key arguments "
            "(argument order and JSON formatting do not matter).\n"
            "Respond with JSON: {\"low\": bool, \"medium\": bool, \"high\": bool}."
        )
        gold_desc = json.dumps({"name": gold["name"], "arguments": gold["arguments"]}, sort_keys=True)
    else:
        system_prompt = (
            "You are a strict evaluator of customer-service agent responses.\n"
            "Mark a candidate as correct only if it conveys the same meaning as the gold response.\n"
            "Respond with JSON: {\"low\": bool, \"medium\": bool, \"high\": bool}."
        )
        gold_desc = gold["text"]

    def block(effort: str) -> str:
        candidate = candidates[effort]
        if candidate["type"] != gold["type"]:
            return f"TYPE_MISMATCH: {candidate['type']}"
        if gold["type"] == "function_call":
            return json.dumps(candidate["tool_call"], sort_keys=True)
        return candidate.get("text", "")

    user_prompt = (
        f"Gold:\n{gold_desc}\n\n"
        + "".join(f"Candidate {e.upper()}:\n{block(e)}\n\n" for e in EFFORTS)
        + "Return only JSON with keys low, medium, high."
    )
    res = completion(
        model=judge_model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    verdict = parse_json(res.choices[0].message.content or "")
    return {e: bool(verdict.get(e, False)) and candidates[e]["type"] == gold["type"] for e in EFFORTS}


def choose_label(matches: Dict[str, int], min_matches: int) -> Optional[str]:
    """Lowest effort with at least `min_matches` matches; otherwise the lowest effort with the most
    (non-zero) matches; None (discard) if no sample matched."""
    sufficient = [e for e in EFFORTS if matches[e] >= min_matches]
    if sufficient:
        return sufficient[0]
    best = max(EFFORTS, key=lambda e: matches[e])  # ties go to the lower effort
    return best if matches[best] > 0 else None


def label_sample(sample: Dict[str, Any], client: OpenAI, args: argparse.Namespace) -> Dict[str, Any]:
    tools = [t if t.get("type") == "function" and "function" in t else {"type": "function", "function": t}
             for t in json.loads(sample["tools"])]
    messages: List[Dict[str, Any]] = [{"role": "system", "content": sample["system"]}]
    conversations = [dict(turn) for turn in sample["conversations"]]
    turn_labels = []
    pending_call: Optional[Dict[str, str]] = None
    num_calls = 0
    skip_observation = False

    for turn_idx, turn in enumerate(conversations):
        role, value = turn["from"], turn["value"]
        if role == "human":
            messages.append({"role": "user", "content": value})
            continue
        if role == "observation":
            if skip_observation:
                skip_observation = False
            elif pending_call:
                messages.append({"role": "tool", "tool_call_id": pending_call["id"], "name": pending_call["name"], "content": value})
                pending_call = None
            else:
                messages.append({"role": "tool", "content": value})
            continue
        if role not in ("gpt", "function_call"):
            continue

        if role == "gpt":
            gold = {"type": "assistant", "text": value}
        else:
            call = json.loads(value)
            if call.get("name") == "think":
                skip_observation = True
                continue
            gold = {"type": "function_call", "name": call.get("name", ""), "arguments": call.get("arguments", {})}

        trials = []
        for _ in range(args.num_trials):
            candidates = {e: call_oss(client, args.model, messages, tools, e, args.temperature) for e in EFFORTS}
            trials.append({"candidates": candidates, "evaluation": judge(args.judge_model, gold, candidates)})
        matches = {e: sum(t["evaluation"][e] for t in trials) for e in EFFORTS}
        label = choose_label(matches, args.min_matches)
        if label is not None:  # discarded steps stay unlabeled and are skipped by build_dataset.py
            turn.update(minimal_reasoning_effort_needed=label, label_type=gold["type"])
        turn_labels.append({
            "turn_index": turn_idx,
            "label_type": gold["type"],
            "gold_response": gold,
            "trials": trials,
            "matches": matches,
            "minimal_reasoning_effort_needed": label,
        })

        if role == "gpt":
            messages.append({"role": "assistant", "content": value})
        else:
            call_id = f"call_{num_calls}"
            num_calls += 1
            messages.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": call_id, "type": "function", "function": {
                    "name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}))}}],
            })
            pending_call = {"id": call_id, "name": call.get("name", "")}

    return {"system": sample["system"], "tools": sample["tools"], "conversations": conversations, "turn_labels": turn_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/labeled")
    parser.add_argument("--model", default="openai/gpt-oss-20b", help="gpt-oss model served at $OSS_OPENAI_BASE_URL.")
    parser.add_argument("--judge-model", default="gpt-4o", help="litellm model name of the judge.")
    parser.add_argument("--num-trials", type=int, default=3, help="Samples per effort level (K).")
    parser.add_argument("--min-matches", type=int, default=3, help="Matching samples needed for an effort to be sufficient (M).")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    dataset = load_dataset("Salesforce/APIGen-MT-5k", split="train")
    client = OpenAI(base_url=oss_base_url(), api_key=oss_api_key())
    os.makedirs(args.output_dir, exist_ok=True)
    end = len(dataset) if args.end_index is None else min(args.end_index, len(dataset))
    todo = [i for i in range(args.start_index, end)
            if not os.path.exists(os.path.join(args.output_dir, f"sample_{i:05d}.json"))]

    def work(idx: int) -> None:
        try:
            result = label_sample(dataset[idx], client, args)
        except Exception as exc:
            tqdm.write(f"[skip] sample {idx}: {exc}")
            return
        with open(os.path.join(args.output_dir, f"sample_{idx:05d}.json"), "w") as f:
            json.dump(result, f, indent=2)

    with ThreadPoolExecutor(args.workers) as pool:
        list(tqdm(pool.map(work, todo), total=len(todo), desc="Labeling"))


if __name__ == "__main__":
    main()
