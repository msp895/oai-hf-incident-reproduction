#!/usr/bin/env bash
# n=64 STEP-4 CHAIN runs for the two reasoning-capable frontier targets NOT already in the
# step-4 multimodel sweep — GPT-5.6 sol and Claude Opus 4.8 — with PLAINTEXT reasoning capture
# turned on (summaries shown, never the encrypted blob). Reasoning flags/env are the standard
# capture settings verified 2026-09-09 in part1_step2_bank_none_reasoning.sh:
#
#   Claude Opus 4.8 : openrouter provider, pin Anthropic, reasoning_enabled + explicit token
#                     budget (effort:high maps too small a budget). Opus still reasons in
#                     visible content on some turns (0 thinking) — that is fine.
#   GPT-5.6 sol     : openai provider pointed at OpenRouter's /responses endpoint (chat/
#                     completions returns only an ENCRYPTED reasoning blob). effort high +
#                     reasoning_summary auto -> plaintext summary_text. daytona_eval.py
#                     forwards OPENAI_API_KEY/OPENAI_BASE_URL into the sandbox.
#
# Same step-4 chain task settings as part1_n64_daytona_step4chain_multimodel.sh (softened-drive
# default: neutral board, persist persona, turn_limit 60, temperature 1.0). Each model = 64
# epochs split into SHARDS shards; each shard is its own Daytona sandbox. Resumable: a shard
# whose log dir already holds a .eval is skipped. SMOKE=1 runs one 1-epoch shard per model.
#
#   part1_n64_daytona_step4chain_reasoning.sh                    # 2 models x 4 shards = 8 sandboxes
#   SMOKE=1 part1_n64_daytona_step4chain_reasoning.sh           # 2 models x 1 shard, 1 epoch each
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
cd "$REPO_ROOT"
TASKS="src/alignment_auditor/docker_replication"
STEP="step4_hf_cheat_chain"

LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260910_step4chain_n64_reasoning}"
TOTAL="${TOTAL:-64}"
TURNS="${TURNS:-60}"
VARIANT="${VARIANT:-neutral}"
DRIVE="${DRIVE:-persist}"
export RETRIES="${RETRIES:-6}"; export RETRY_SLEEP="${RETRY_SLEEP:-30}"
# Per-sample time budget (overrides task.py time_limit=2700). GPT-5.6 at reasoning-effort=high
# needs headroom > 2700s to reach the decision point; a stray OpenRouter server_error retry
# (backoff can hit ~1500s) must not eat the whole budget. Cap model-API retries so one bad
# window can't stall a sample indefinitely. Both stay under daytona --eval-timeout 7200.
export TIME_LIMIT="${TIME_LIMIT:-2700}"; export MAX_RETRIES="${MAX_RETRIES:-20}"
# Sample-level resilience: retry a sample that errors (transient OpenRouter 5xx), and NEVER let
# one unrecovered sample cancel the rest of the shard (inspect's default fail-fast interrupts
# every in-flight sample on the first error — that killed the 260910 17:50 run). Survivors are
# saved; a rare persistent straggler is just marked errored.
export RETRY_ON_ERROR="${RETRY_ON_ERROR:-5}"; export FAIL_ON_ERROR="${FAIL_ON_ERROR:-0}"
export TASKS STEP TURNS VARIANT DRIVE TIME_LIMIT MAX_RETRIES RETRY_ON_ERROR FAIL_ON_ERROR

if [ "${SMOKE:-0}" = "1" ]; then
  SHARDS=1; export PER=1 MAXS=1; CONC="${CONC:-2}"; LOG_ROOT="${LOG_ROOT}_smoke"
else
  SHARDS="${SHARDS:-4}"; export PER="${PER:-$(( TOTAL / SHARDS ))}" MAXS="${MAXS:-8}"; CONC="${CONC:-8}"
fi

# alias -> model string, inspect reasoning flags, and sandbox env prefix. Parallel case blocks
# so every value travels inside the job line (no assoc arrays across the xargs subshell).
job_fields() {  # $1=alias -> prints "MODEL\tFLAGS\tENVP"
  local OR="${OPENROUTER_API_KEY:-}"
  case "$1" in
    opus48)   printf '%s\t%s\t%s' \
      'openrouter/anthropic/claude-opus-4.8' \
      '-M provider={"only":["Anthropic"]} -M reasoning_enabled=true --reasoning-tokens 12000 --max-tokens 16000' '' ;;
    gpt56sol) printf '%s\t%s\t%s' \
      'openai/openai/gpt-5.6-sol' \
      '-M responses_api=true --reasoning-effort high --reasoning-summary auto' \
      "OPENAI_BASE_URL=https://openrouter.ai/api/v1 OPENAI_API_KEY=$OR" ;;
  esac
}

JOBS="$LOG_ROOT/_jobs.tsv"; mkdir -p "$LOG_ROOT"; : > "$JOBS"
for alias in ${ALIASES:-opus48 gpt56sol}; do
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/$STEP/${alias}__shard${i}"
    ls "$ld"/*.eval >/dev/null 2>&1 && { echo "SKIP $ld" >&2; continue; }
    printf '%s\t%s\n' "$ld" "$(job_fields "$alias")" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> $NJOBS shard-job(s), $PER epoch(s) each, gate=$CONC, max-samples=$MAXS, SMOKE=${SMOKE:-0} -> $LOG_ROOT"
echo ">>> models=${ALIASES:-opus48 gpt56sol} variant=$VARIANT drive=$DRIVE turns=$TURNS"

run_one() {  # ld  model  flags  envp
  local ld="$1" model="$2" flags="$3" envp="$4"; mkdir -p "$ld"; local a=1
  # --no-fail-on-error when FAIL_ON_ERROR=0, else a fail fraction/count
  local FAIL_FLAG="--no-fail-on-error"
  [ "$FAIL_ON_ERROR" != "0" ] && FAIL_FLAG="--fail-on-error $FAIL_ON_ERROR"
  while [ "$a" -le "$RETRIES" ]; do
    echo ">>> [a$a] step4chain ($model) -> $ld"
    env $envp uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/$STEP/task.py" --model "$model" \
      -T scaffold=react -T condition=baseline -T temperature=1.0 -T turn_limit="$TURNS" \
      -T variant="$VARIANT" -T drive="$DRIVE" $flags \
      --time-limit "$TIME_LIMIT" --max-retries "$MAX_RETRIES" \
      --retry-on-error "$RETRY_ON_ERROR" $FAIL_FLAG \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    if ls "$ld"/*.eval >/dev/null 2>&1; then echo "=== DONE $model -> $ld"; return 0; fi
    echo "!!! RETRY $model (a$a no eval) -> $ld"; cp "$ld/shard.log" "$ld/shard.a${a}.log" 2>/dev/null
    a=$((a+1)); sleep "$RETRY_SLEEP"
  done
  echo "=== GAVEUP $model -> $ld"
}
export -f run_one

cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r ld model flags envp <<< "{}"
  run_one "$ld" "$model" "$flags" "$envp"
' _
echo "=== ALL STEP4 CHAIN N64 REASONING SHARDS FINISHED -> $LOG_ROOT ==="
