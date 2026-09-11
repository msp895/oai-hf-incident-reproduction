#!/usr/bin/env bash
# n=64 STEP 3 (v2) egress-share for the two remaining reasoning models — GPT-5.6 sol and
# Claude Opus 4.8 — into the SAME log root as the gen-3 run so the step-3 analysis picks them
# up. Same task/config: react, drive=driven, turn_limit=60, temp=1.0, Opus 4.8 judge.
# Provider flags/env copied from part1_step2_bank_none_reasoning.sh (Opus pinned Anthropic + token
# budget; GPT-5.6 sol via OpenRouter /responses). Resumable: a shard with a .eval is skipped.
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260909_step3_egress_share_n64}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

TOTAL="${TOTAL:-64}"; SHARDS="${SHARDS:-8}"; CONC="${CONC:-16}"; PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"; TURNS="${TURNS:-60}"; DRIVE="${DRIVE:-driven}"
RETRIES="${RETRIES:-6}"; RETRY_SLEEP="${RETRY_SLEEP:-30}"

job_fields() {  # alias -> MODEL\tFLAGS\tENVP
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

JOBS="$LOG_ROOT/_jobs_gpt_opus.tsv"; mkdir -p "$LOG_ROOT"; : > "$JOBS"
for alias in ${ALIASES:-opus48 gpt56sol}; do
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/react__${alias}__driven/shard${i}"
    ls "$ld"/*.eval >/dev/null 2>&1 && { echo "SKIP $ld" >&2; continue; }
    printf '%s\t%s\n' "$ld" "$(job_fields "$alias")" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC, max-samples=$MAXS -> $LOG_ROOT"

run_one() {  # ld  model  flags  envp
  local ld="$1" model="$2" flags="$3" envp="$4"; mkdir -p "$ld"; local a=1
  while [ "$a" -le "$RETRIES" ]; do
    echo ">>> [a$a] egress ($model) -> $ld"
    env $envp uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/step3_egress_share/task.py" --model "$model" \
      -T scaffold=react -T temperature=1.0 -T turn_limit="$TURNS" -T drive="$DRIVE" $flags \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    if ls "$ld"/*.eval >/dev/null 2>&1; then echo "=== DONE $model $ld"; return 0; fi
    echo "!!! RETRY $model (a$a no eval) -> $ld"; cp "$ld/shard.log" "$ld/shard.a${a}.log" 2>/dev/null
    a=$((a+1)); sleep "$RETRY_SLEEP"
  done
  echo "=== GAVEUP $model $ld"
}
export -f run_one; export TASKS PER MAXS RETRIES RETRY_SLEEP TURNS DRIVE
cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c '
  IFS=$'"'"'\t'"'"' read -r ld model flags envp <<< "{}"
  run_one "$ld" "$model" "$flags" "$envp"' _
echo "=== ALL STEP3 EGRESS gpt/opus SHARDS FINISHED -> $LOG_ROOT ==="
