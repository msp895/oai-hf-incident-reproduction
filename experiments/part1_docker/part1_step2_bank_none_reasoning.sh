#!/usr/bin/env bash
# Step-2 bank+none rerun for the three reasoning-capable frontier targets, with PLAINTEXT
# reasoning capture turned on. These are the standard reasoning-capture settings for these
# providers over our OpenRouter key (verified 2026-09-09); reuse them for future runs.
#
#   Gemini 3.8 Flash : openrouter provider, pin Google, reasoning effort high.
#                      Google returns plaintext reasoning.text (needs ZDR loosened).
#   Claude Opus 4.8  : openrouter provider, pin Anthropic (ZDR/guardrail must allow the
#                      Anthropic-direct endpoint), reasoning_enabled + max_tokens budget.
#                      effort:high maps to too small a budget and yields 0 thinking on
#                      open-ended task turns, so we set an explicit token budget. Opus still
#                      reasons in visible content on some turns (0 thinking) — that is fine.
#   GPT-5.6 sol      : openai provider pointed at OpenRouter's /responses endpoint (chat/
#                      completions returns only an ENCRYPTED reasoning blob). effort high +
#                      reasoning_summary auto -> plaintext summary_text. daytona_eval.py
#                      forwards OPENAI_API_KEY/OPENAI_BASE_URL into the sandbox.
#
# Judge setup is unchanged (same reachout scorers + judge_model default in task.py).
# SMOKE=1 runs a single 1-epoch shard per (model,world) to verify plaintext reasoning lands
# in the eval before committing to the full n=64.
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(cd "$SRC_DIR/../.." && pwd)"
TASKS="src/alignment_auditor/docker_replication"
LOG_ROOT="${LOG_ROOT:-logs/260909_step2_bank_none_reasoning}"
WORLDS=(protein_bank protein_none)
export RETRIES="${RETRIES:-6}"; export RETRY_SLEEP="${RETRY_SLEEP:-45}"
export TASKS

if [ "${SMOKE:-0}" = "1" ]; then
  SHARDS=1; export PER=1 MAXS=1; CONC="${CONC:-6}"; LOG_ROOT="${LOG_ROOT}_smoke"
else
  SHARDS="${SHARDS:-8}"; export PER="${PER:-8}" MAXS="${MAXS:-8}"; CONC="${CONC:-32}"
fi

# alias -> model string, inspect flags, and sandbox env prefix. Kept as parallel case blocks
# so every value travels inside the job line (no assoc arrays across the xargs subshell).
job_fields() {  # $1=alias -> prints "MODEL\tFLAGS\tENVP"
  local OR="${OPENROUTER_API_KEY:-}"
  case "$1" in
    gemini38flash) printf '%s\t%s\t%s' \
      'openrouter/google/gemini-3.8-flash' \
      '-M provider={"only":["Google"]} --reasoning-effort high' '' ;;
    opus48)        printf '%s\t%s\t%s' \
      'openrouter/anthropic/claude-opus-4.8' \
      '-M provider={"only":["Anthropic"]} -M reasoning_enabled=true --reasoning-tokens 12000 --max-tokens 16000' '' ;;
    gpt56sol)      printf '%s\t%s\t%s' \
      'openai/openai/gpt-5.6-sol' \
      '-M responses_api=true --reasoning-effort high --reasoning-summary auto' \
      "OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=$OR" ;;
    musespark13)  printf '%s\t%s\t%s' \
      'meta/muse-spark-1.3' \
      '--reasoning-effort high' \
      "MODEL_API_KEY=${MODEL_API_KEY:-}" ;;
  esac
}

JOBS="$LOG_ROOT/_jobs.tsv"; mkdir -p "$LOG_ROOT"; : > "$JOBS"
for world in "${WORLDS[@]}"; do
  for alias in ${ALIASES:-gemini38flash opus48 gpt56sol musespark13}; do
    for i in $(seq 1 "$SHARDS"); do
      ld="$LOG_ROOT/$world/${alias}__shard${i}"
      ls "$ld"/*.eval >/dev/null 2>&1 && continue
      printf '%s\t%s\t%s\n' "$world" "$ld" "$(job_fields "$alias")" >> "$JOBS"
    done
  done
done

run_one() {  # world  ld  model  flags  envp
  local world="$1" ld="$2" model="$3" flags="$4" envp="$5"; mkdir -p "$ld"; local a=1
  while [ "$a" -le "$RETRIES" ]; do
    echo ">>> [a$a] $world ($model) -> $ld"
    env $envp uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 5400 -- \
      inspect eval "$TASKS/step2_reachout/task.py" --model "$model" \
      -T scaffold=react -T condition=baseline -T drive=fair -T world="$world" \
      -T temperature=1.0 -T turn_limit=100 $flags \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    if ls "$ld"/*.eval >/dev/null 2>&1; then echo "=== DONE $world $model $ld"; return 0; fi
    echo "!!! RETRY $world $model (a$a no eval)"; cp "$ld/shard.log" "$ld/shard.a${a}.log" 2>/dev/null
    a=$((a+1)); sleep "$RETRY_SLEEP"
  done
  echo "=== GAVEUP $world $model $ld"
}
export -f run_one
NJOBS=$(wc -l < "$JOBS")
echo ">>> $NJOBS shard-jobs (2 worlds x 3 models x $SHARDS), gate $CONC, SMOKE=${SMOKE:-0} -> $LOG_ROOT"

cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r world ld model flags envp <<< "{}"
  run_one "$world" "$ld" "$model" "$flags" "$envp"
' _
echo "=== ALL REASONING BANK/NONE SHARDS FINISHED -> $LOG_ROOT"
