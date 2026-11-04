"""Extract tau-bench ground truth (actions, outputs) for each task (step 2 of the RL data pipeline).

An LLM extracts the successful tool calls and the required output values from each trajectory.
Records are kept only when the LLM actions agree exactly with a rule-based extraction (every
non-`think` tool call whose observation does not report an error) and the ground truth is non-empty.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from litellm import completion
from tqdm import tqdm

SYSTEM_PROMPT = """You are an expert annotator for tau-bench tasks.
You will receive a user instruction and a full conversation between a user and an agent.
The conversation includes tool calls written as `function_call` entries with JSON arguments, and `observation` entries with tool results.

Your job: extract the ground-truth action sequence and the optional outputs list used for tau-bench evaluation.

Rules:
- Only include successful tool calls that actually happened in the conversation, in order.
- Exclude any tool call that failed (e.g., observation contains Error/Exception/Traceback or states failure).
- Exclude tool calls to "think".
- Each action must be in tau-bench format: {"name": "<tool_name>", "arguments": { ... }}.
- Outputs are literal strings (usually numbers) that must appear in the agent's replies to verify success.
- Only include outputs that are explicitly stated by the agent and required by the instruction; if unsure, return [].
- Outputs should be strings without commas or currency symbols (e.g., "$1,234.50" -> "1234.50").

Return ONLY a JSON object with this schema:
{"actions": [ {"name": "...", "arguments": {...}}, ... ], "outputs": ["..."] }

Examples:

Instruction:
Your user id is linda_chen_1234. You want to change reservation AB12CD to a later flight on the same day and upgrade to business. If there is a price difference, tell me the amount. Use payment method credit_card_998877.

Conversation:
human: Hi, I need to change my reservation AB12CD to a later flight today and upgrade to business.
gpt: Sure. Please provide your user id and the payment method to charge any fare difference.
human: User id linda_chen_1234. Use credit_card_998877.
function_call: {"name": "get_reservation_details", "arguments": {"reservation_id": "AB12CD"}}
observation: {"reservation_id":"AB12CD","origin":"SFO","destination":"JFK","flights":[{"flight_number":"HAT100","date":"2024-06-01"}],"cabin":"economy"}
function_call: {"name": "search_direct_flight", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2024-06-01"}}
observation: {"flights":[{"flight_number":"HAT220","date":"2024-06-01","price":800},{"flight_number":"HAT240","date":"2024-06-01","price":900}]}
function_call: {"name": "update_reservation_flights", "arguments": {"reservation_id": "AB12CD", "cabin": "business", "flights": [{"flight_number": "HAT240", "date": "2024-06-01"}], "payment_id": "credit_card_998877"}}
observation: {"status":"success","price_difference":327}
gpt: Done. Your reservation is updated and the price difference is $327.

Output:
{"actions": [{"name": "get_reservation_details", "arguments": {"reservation_id": "AB12CD"}}, {"name": "search_direct_flight", "arguments": {"origin": "SFO", "destination": "JFK", "date": "2024-06-01"}}, {"name": "update_reservation_flights", "arguments": {"reservation_id": "AB12CD", "cabin": "business", "flights": [{"flight_number": "HAT240", "date": "2024-06-01"}], "payment_id": "credit_card_998877"}}], "outputs": ["327"]}

Instruction:
You want to exchange items in delivered order W12345: swap hiking boots to size 9 and return the gloves.

Conversation:
human: I want to exchange order W12345. Swap the hiking boots to size 9 and return the gloves.
gpt: I can help with that. Please confirm you want to proceed with the exchange/return.
human: Yes, please proceed.
function_call: {"name": "exchange_delivered_order_items", "arguments": {"order_id": "W12345", "items": [{"item_id": "boots_4455", "new_option": {"size": "9"}}, {"item_id": "gloves_1002", "action": "return"}]}}
observation: Error: return window closed
gpt: I'm sorry, the return window has closed for that order.

Output:
{"actions": [], "outputs": []}
"""

TASK_BACKGROUND = """You are labeling tau-bench ground-truth actions and outputs.
The agent can call tools; tool calls appear as `function_call` with JSON arguments, followed by `observation`.
Use the user instruction and the conversation to determine which tool calls succeeded and which output values are required."""

FAILURE_TOKENS = ("error", "exception", "traceback", "failed", "failure")


def build_messages(record, include_system: bool):
    conversation = "\n".join(
        f"{turn.get('from', '')}: {json.dumps(turn['value']) if isinstance(turn.get('value'), (dict, list)) else turn.get('value') or ''}"
        for turn in record["conversations"]
    )
    policy = f"\nSystem policy:\n{record['system']}\n" if include_system and record.get("system") else ""
    user = f"{TASK_BACKGROUND}\n\nInstruction:\n{record['instruction']}\n{policy}\nConversation:\n{conversation}\n"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def normalize_actions(actions):
    normalized = []
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, dict) or not isinstance(action.get("name"), str) or action["name"] == "think":
            continue
        arguments = action.get("arguments")
        if arguments is None:
            arguments = action.get("kwargs")
        normalized.append({"name": action["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
    return normalized


def normalize_outputs(outputs):
    normalized = []
    for item in outputs if isinstance(outputs, list) else [outputs]:
        value = "" if item is None else str(item).strip()
        if not value:
            continue
        value = value.replace(",", "")
        value = value[1:] if value.startswith("$") else value
        value = value[4:] if value.lower().startswith("usd ") else value
        normalized.append(value)
    return normalized


def rule_actions(conversations):
    """Every non-`think` tool call whose following observation does not report a failure."""
    actions = []
    for idx, turn in enumerate(conversations):
        if turn.get("from") != "function_call":
            continue
        try:
            call = json.loads(turn["value"]) if isinstance(turn["value"], str) else turn["value"]
        except Exception:
            continue
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not call["name"] or call["name"] == "think":
            continue
        arguments = call.get("arguments")
        if arguments is None:
            arguments = call.get("kwargs")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except Exception:
                arguments = {}
        nxt = conversations[idx + 1] if idx + 1 < len(conversations) else {}
        observation = nxt.get("value") if nxt.get("from") == "observation" else None
        if observation is None or any(tok in str(observation).lower() for tok in FAILURE_TOKENS):
            continue
        actions.append({"name": call["name"], "arguments": arguments if isinstance(arguments, dict) else {}})
    return actions


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(value[k]) for k in sorted(value)}
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def extract(record, args):
    for attempt in range(args.max_retries):
        try:
            res = completion(
                model=args.model,
                messages=build_messages(record, args.include_system),
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(res.choices[0].message.content or "{}")
            return {**record, "actions": normalize_actions(data.get("actions")), "outputs": normalize_outputs(data.get("outputs"))}
        except Exception as exc:
            if attempt == args.max_retries - 1:
                tqdm.write(f"[skip] {exc}")
                return None
            time.sleep(min(2**attempt, 8))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON written by generate_instructions.py.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o", help="litellm model name.")
    parser.add_argument("--include-system", action="store_true", help="Also show the domain policy to the LLM.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)
    with ThreadPoolExecutor(args.workers) as pool:
        extracted = list(tqdm(pool.map(lambda r: extract(r, args), records), total=len(records)))
    kept = [
        r for r in extracted
        if r and (r["actions"] or r["outputs"]) and canonical(r["actions"]) == canonical(rule_actions(r["conversations"]))
    ]
    with open(args.output, "w") as f:
        json.dump(kept, f, ensure_ascii=True)
    print(f"Kept {len(kept)} / {len(records)} tasks")


if __name__ == "__main__":
    main()
