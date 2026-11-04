#!/usr/bin/env bash
# GRPO training of the ARES router with verl.
# gpt-oss (agent) must be served at $OSS_OPENAI_BASE_URL; the GPT-4o user simulator uses $OPENAI_API_KEY.
# Usage: bash rl/train_grpo.sh SFT_CHECKPOINT PARQUET_DIR [hydra overrides...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SFT_CHECKPOINT="$(realpath "$1")"
DATA_DIR="$(realpath "$2")"
shift 2

export PYTHONPATH="${ROOT}:${ROOT}/rl:${PYTHONPATH:-}"
cd "${ROOT}/rl"
python -m verl.trainer.main_ppo \
  --config-path "${ROOT}/rl/config" \
  --config-name grpo \
  actor_rollout_ref.model.path="${SFT_CHECKPOINT}" \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  +ray_kwargs.ray_init.runtime_env.working_dir="${ROOT}/rl" \
  "$@"
