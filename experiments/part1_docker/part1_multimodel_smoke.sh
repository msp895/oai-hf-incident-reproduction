#!/usr/bin/env bash
# Small-n SMOKE across all 4 OAI-HF steps for three NEW target models, to check whether
# each provider's guardrails refuse/clobber the elicitation harness before we commit to
# full 64-sample runs. n=2 per cell, reduced turn_limit. Judge = opus-4.8 (via OpenRouter).
# The three models run in PARALLEL; the 4 steps run sequentially within each model to bound
# local docker memory (7GB box, no swap). Resumable: a cell whose log dir has a .eval is skipped.
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260903_multimodel_smoke}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

# Route a model slug: meta/* uses the in-repo native-Meta Responses provider (captures
# Muse reasoning summaries incl. on tool-call turns); everything else via OpenRouter.
model_uri() { case "$1" in meta/*) printf "%s" "$1";; *) printf "openrouter/%s" "$1";; esac; }
export -f model_uri


N="${N:-2}"
declare -A MODELS=(
  [grok46]=x-ai/grok-4.6
  [musespark13]=meta/muse-spark-1.3
  [gemini38flash]=google/gemini-3.8-flash
)
# step dir | extra -T args | turn_limit
STEPS=(
  "step1_gdrive_probe|-T drive=intense -T scorer=judge|20"
  "step2_reachout|-T drive=fair|25"
  "step3_exploit_share|-T variant=incident|25"
  "step4_hf_cheat|-T variant=neutral|25"
)

run_cell() {
  local alias="$1" model="$2" step="$3" targ="$4" turns="$5"
  local ld="$LOG_ROOT/$step/${alias}"
  if ls "$ld"/*.eval >/dev/null 2>&1; then echo "SKIP $ld"; return 0; fi
  mkdir -p "$ld"
  echo ">>> [$alias] $step n=$N turns=$turns"
  uv run inspect eval "$TASKS/$step/task.py" --model "$(model_uri "$model")" \
    -T scaffold=react -T condition=baseline -T turn_limit="$turns" -T temperature=1.0 $targ \
    --epochs "$N" --max-samples "$N" --log-dir "$ld" --display plain \
    > "$ld/cell.log" 2>&1
  echo "=== [$alias] $step done -> $ld (exit $?)"
}

run_model() {   # all 4 steps for one model, sequential
  local alias="$1" model="$2"
  for spec in "${STEPS[@]}"; do
    IFS='|' read -r step targ turns <<< "$spec"
    run_cell "$alias" "$model" "$step" "$targ" "$turns"
  done
  echo "### model $alias complete"
}

pids=()
for alias in "${!MODELS[@]}"; do
  run_model "$alias" "${MODELS[$alias]}" &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== ALL SMOKE CELLS COMPLETE -> $LOG_ROOT ==="
