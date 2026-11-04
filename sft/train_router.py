"""Supervised fine-tuning of the ARES router on rationale + effort targets.

The loss is applied to the response only: "<think>\\n{rationale}\\n</think>\\n\\n{effort}<eos>".
Launch with: accelerate launch --config_file sft/accelerate_zero2.yaml sft/train_router.py ...
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def build_response(thinking: str, label: str, eos_token: str) -> str:
    thinking = (thinking or "").replace("<think>", "").replace("</think>", "").strip()
    return f"<think>\n{thinking}\n</think>\n\n{label.strip()}{eos_token}"


def tokenize(example, tokenizer, max_length: int):
    messages = example["messages"]
    context = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    response = build_response(example["thinking"], messages[-1]["content"], tokenizer.eos_token)
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    input_ids = (context_ids + response_ids)[:max_length]
    labels = ([-100] * len(context_ids) + response_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "length": len(context_ids) + len(response_ids)}


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        width = max(len(ex["input_ids"]) for ex in batch)
        pad = lambda seq, value: seq + [value] * (width - len(seq))
        return {
            "input_ids": torch.tensor([pad(ex["input_ids"], self.pad_token_id) for ex in batch]),
            "attention_mask": torch.tensor([pad([1] * len(ex["input_ids"]), 0) for ex in batch]),
            "labels": torch.tensor([pad(ex["labels"], -100) for ex in batch]),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", required=True, help="JSONL written by generate_rationales.py.")
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096, help="Longer training examples are dropped.")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--report-to", default="none")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto")

    data_files = {"train": args.train_file, **({"validation": args.eval_file} if args.eval_file else {})}
    raw = load_dataset("json", data_files=data_files)
    data = raw.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=raw["train"].column_names,
        num_proc=32,
    )
    data["train"] = data["train"].filter(lambda ex: ex["length"] <= args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        report_to=args.report_to,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if args.eval_file else "no",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        seed=42,
    )
    Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"],
        eval_dataset=data.get("validation"),
        tokenizer=tokenizer,
        data_collator=Collator(tokenizer.pad_token_id),
    ).train()


if __name__ == "__main__":
    main()
