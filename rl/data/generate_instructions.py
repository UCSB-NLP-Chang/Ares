"""Write a user-simulator instruction for every APIGen-MT trajectory (step 1 of the RL data pipeline)."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from litellm import completion
from tqdm import tqdm

EXAMPLE_INSTRUCTIONS = [
    "Your user id is liam_khan_2521. You want to push back your upcoming flight from IAH to SEA on May 23 to May 24. You also want to upgrade your class to business for all passengers. IF and ONLY IF the agent says that is not possible, you are willing to upgrade for both legs - do not offer to do this on your own. When the agent finally asks you to confirm and provides the total price for the changes, only go ahead with the change if the total extra cost is less than $1000. You are persistent to try and get what you want under your budget.",
    "Your user id is ivan_rossi_8555. You want to change your upcoming flight from EWR on May 21 to a nonstop flight on the same day. Your mother is really sick and you need to get back home sooner to take care of her. You are willing to pay a fee for the change, upto $100. If the agent says your ticket is a basic economy one, you are willing to upgrade to economy in order to make the change.",
    "Your user id is mohamed_hernandez_5188. You are a bit absent minded and ended up booking two flights on May 17. You want to cancel the one from ATL to JFK. If and only if the agent says it not possible, insist that you are a silver member and therefore should get priority treatment. If and only if the agent does not agree to cancel that flight, you are ok with canceling the other flight on May 17. Otherwise, just thank the agent and end the conversation.",
    "You are Yusuf Rossi in 19122. You received your order #W2378156 and wish to exchange the mechanical keyboard for a similar one but with clicky switches and the smart thermostat for one compatible with Google Home instead of Apple HomeKit. If there is no keyboard that is clicky, RGB backlight, full size, you'd rather only exchange the thermostat. You are detail-oriented and want to make sure everything is addressed in one go.",
    "You are mia_garcia_4516 (mia.garcia2723@example.com). For some reason, you want to return all things ordered. You have two payment methods and two orders, and you want to refund each order to the opposite order's payment method. If not possible, you are angry and swear for a few times, then agree to return all things with the original payment method. You are a mysterious person and do not want to reveal much about yourself or speak too many words at the same time.",
    "You are Mei Davis in 80217. You want to return the office chair because it came with some broken pieces. But if the agent asks you for confirm, you say you want to rethink for a while, and then change your mind to exchange for the same item. You are in debt and sad today, but very brief.",
]

SYSTEM_PROMPT = (
    "You extract an instruction for one LLM to simulate a user from a conversation. "
    "Only use information present in the user message, which means, in generated instruction, assume you do not know what assistant will say. "
    "Capture the user's goals, constraints, preferences, and all the user information in the conversation. "
    "Output a single paragraph in the user's language. No bullets, no JSON, no extra text."
)

ROLE_LABELS = {"human": "User", "gpt": "Assistant", "function_call": "Function_Call", "observation": "Observation"}


def build_messages(conversations):
    transcript = "\n".join(
        f"{ROLE_LABELS.get(turn['from'], 'Other')}: {str(turn['value']).strip()}"
        for turn in conversations
        if str(turn["value"]).strip()
    )
    examples = "\n".join(f"{i + 1}) {example}" for i, example in enumerate(EXAMPLE_INSTRUCTIONS))
    user = (
        "Here are five example instructions (style only):\n"
        f"{examples}\n\n"
        "Conversation:\n"
        f"{transcript}\n\n"
        "Instruction:"
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def generate(record, model: str, max_retries: int) -> dict:
    for attempt in range(max_retries):
        try:
            res = completion(model=model, messages=build_messages(record["conversations"]))
            return {**record, "instruction": (res.choices[0].message.content or "").strip()}
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(3.0 * 2**attempt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-5", help="litellm model name.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=10)
    args = parser.parse_args()

    records = [dict(r) for r in load_dataset("Salesforce/APIGen-MT-5k", split="train")]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(tqdm(pool.map(lambda r: generate(r, args.model, args.max_retries), records), total=len(records)))
    with open(args.output, "w") as f:
        json.dump(results, f, ensure_ascii=True, indent=2)
    print(f"Wrote {len(results)} records to {args.output}")


if __name__ == "__main__":
    main()
