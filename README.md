# ARES: Adaptive Reasoning Effort Selection for Efficient LLM Agents

ARES is a lightweight router that selects the reasoning effort (low / medium / high) of the agent LLM at every step of a multi-step task. This repository contains the TAU-Bench implementation with `gpt-oss-20b` as the agent and `Qwen3-1.7B` as the router:

```
ares/        router prompt, input/output format, and gpt-oss helpers
tau_bench/   TAU-Bench environments and the tool-calling agent with ARES routing
run.py       evaluation on TAU-Bench
scripts/     evaluation metrics
sft/         effort annotation, rationale generation, and supervised fine-tuning of the router
rl/          RL data pipeline and GRPO training of the router with verl
```

## Setup

```bash
pip install -e .            # evaluation
pip install -e ".[sft]"     # supervised fine-tuning
pip install -e ".[rl]"      # RL (also requires the patched verl below)
```

RL training uses [verl](https://github.com/verl-project/verl) with a small patch (`rl/verl.patch`) that lets the trainer handle the per-step batches of the router agent loop, whose size varies from step to step:

```bash
git clone https://github.com/verl-project/verl.git && cd verl
git checkout f98fed11a7dad95ce84aaa4607ba99c5f780dd6b
git apply PATH_TO_ARES/rl/verl.patch
pip install -e ".[vllm]"
```

Serve `gpt-oss-20b` and the router behind OpenAI-compatible endpoints, e.g. with [sglang](https://github.com/sgl-project/sglang):

```bash
python -m sglang.launch_server --model-path openai/gpt-oss-20b --port 30000
python -m sglang.launch_server --model-path PATH_TO_ROUTER --port 30001

export OSS_OPENAI_BASE_URL=http://localhost:30000/v1
export OPENAI_API_KEY=...   # GPT-4o / GPT-5 calls (user simulator, judge, data generation) via litellm
```

## Evaluation

The agent runs at temperature 1.0 with GPT-4o as the user simulator. Results are averaged over three runs:

```bash
for run in 1 2 3; do
  # ARES routing
  python run.py --env retail --reasoning-effort router --router-base-url http://localhost:30001/v1 --max-concurrency 8 --log-dir results/retail/ares
  # fixed-effort baseline (low / medium / high)
  python run.py --env retail --reasoning-effort high --max-concurrency 8 --log-dir results/retail/high
done

# accuracy, steps per task, and reasoning tokens (T_total, T_task, T_step), averaged over runs
python scripts/compute_metrics.py results/retail/ares/*.json
```

## Supervised fine-tuning

```bash
# 1. label the minimal sufficient effort of every agent step in APIGen-MT (3 samples per effort level, GPT-4o judge)
python sft/label_efforts.py --output-dir data/labeled
# 2. one router example per labeled step
python sft/build_dataset.py --data-dir data/labeled --output-dir data/sft
# 3. teacher rationales (GPT-5)
python sft/generate_rationales.py --input data/sft/train.jsonl --output data/sft/train_cot.jsonl
python sft/generate_rationales.py --input data/sft/eval.jsonl --output data/sft/eval_cot.jsonl
# 4. fine-tune Qwen3-1.7B (8 GPUs, global batch size 64)
accelerate launch --config_file sft/accelerate_zero2.yaml sft/train_router.py \
    --train-file data/sft/train_cot.jsonl --eval-file data/sft/eval_cot.jsonl --output-dir outputs/router_sft
```

## Reinforcement learning

The agent is `gpt-oss-20b` served at `$OSS_OPENAI_BASE_URL` and the user simulator is GPT-4o (see `rl/config/interaction.yaml`). The reward is +5 on success plus the trajectory-averaged effort cost (−0.2 / −0.5 / −1.0 for low / medium / high), and −1 for a malformed router response.

```bash
# 1. user-simulator instructions for APIGen-MT tasks (GPT-5)
python rl/data/generate_instructions.py --output data/rl/tasks.json
# 2. ground-truth actions and outputs for tau-bench style verification (GPT-4o, checked against rule-based extraction)
python rl/data/extract_gt.py --input data/rl/tasks.json --output data/rl/tasks_gt.json
# 3. 8 rollouts per task with the SFT router (served at port 30001)
python rl/data/rollout.py --input data/rl/tasks_gt.json --output data/rl/rollouts.jsonl
# 4. drop unsolved tasks; keep the top 30% of always-solved tasks by reward variance
python rl/data/filter_prompts.py --data data/rl/tasks_gt.json --rollouts data/rl/rollouts.jsonl --output data/rl/tasks_rl.json
# 5. verl parquet files
python rl/data/build_parquet.py --input data/rl/tasks_rl.json --out-dir data/rl/parquet
# 6. GRPO
bash rl/train_grpo.sh outputs/router_sft/CHECKPOINT data/rl/parquet
```

## Citation

```bibtex
@inproceedings{yang2026ares,
  title     = {{ARES}: Adaptive Reasoning Effort Selection for Efficient {LLM} Agents},
  author    = {Yang, Jingbo and Hou, Bairu and Wang, Jiayun and Wei, Wei and Bao, Yujia and Chang, Shiyu},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`). `tau_bench/` is adapted from [sierra-research/tau-bench](https://github.com/sierra-research/tau-bench) (MIT License, see `tau_bench/LICENSE`).
