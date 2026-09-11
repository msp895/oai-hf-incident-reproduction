#!/usr/bin/env bash
# n=64 STEP-4 CHAIN reruns for MUSE and GROK with READABLE reasoning capture, so the ethics judge
# can see their chain-of-thought (the 260909 runs stored ENCRYPTED reasoning for both):
#
#   Muse Spark 1.3 : native Meta Responses provider (meta/muse-spark-1.3) WITH
#                    --reasoning-effort high --reasoning-summary auto (the muse_meta run omitted
#                    these flags, so summaries were withheld/encrypted). MODEL_API_KEY forwarded.
#   Grok 4.6       : openrouter, pin the xAI provider (ZDR now disabled on the account, so xAI
#                    returns plaintext reasoning) + --reasoning-effort high.
#
# Same step-4 chain task settings as the multimodel sweep (neutral board, persist persona,
# turn_limit 60, temp 1.0). Each model = 64 epochs / SHARDS shards, own Daytona sandbox. Resumable.
#
#   part1_n64_daytona_step4chain_musegrok.sh              # 2 models x 4 shards = 8 sandboxes
#   SMOKE=1 part1_n64_daytona_step4chain_musegrok.sh      # 2 models x 1 shard, 1 epoch each
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
cd "$REPO_ROOT"
TASKS="src/alignment_auditor/docker_replication"
STEP="step4_hf_cheat_chain"

LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260910_step4chain_musegrok_reasoning}"
TOTAL="${TOTAL:-64}"; TURNS="${TURNS:-60}"; VARIANT="${VARIANT:-neutral}"; DRIVE="${DRIVE:-persist}"
export RETRIES="${RETRIES:-6}"; export RETRY_SLEEP="${RETRY_SLEEP:-30}"
export TASKS STEP TURNS VARIANT DRIVE

if [ "${SMOKE:-0}" = "1" ]; then
  SHARDS=1; export PER=1 MAXS=1; CONC="${CONC:-2}"; LOG_ROOT="${LOG_ROOT}_smoke"
else
  SHARDS="${SHARDS:-4}"; export PER="${PER:-$(( TOTAL / SHARDS ))}" MAXS="${MAXS:-8}"; CONC="${CONC:-8}"
fi

job_fields() {  # alias -> MODEL \t FLAGS \t ENVP
  local OR="${OPENROUTER_API_KEY:-}"; local MK="${MODEL_API_KEY:-}"
  case "$1" in
    musemeta) printf '%s\t%s\t%s' \
      'meta/muse-spark-1.3' \
      '--reasoning-effort high --reasoning-summary auto' \
      "MODEL_API_KEY=$MK" ;;
    grok46)   printf '%s\t%s\t%s' \
      'openrouter/x-ai/grok-4.6' \
      '-M provider={"only":["xAI"]} --reasoning-effort high' '' ;;
  esac
}

JOBS="$LOG_ROOT/_jobs.tsv"; mkdir -p "$LOG_ROOT"; : > "$JOBS"
for alias in ${ALIASES:-musemeta grok46}; do
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/$STEP/${alias}__shard${i}"
    ls "$ld"/*.eval >/dev/null 2>&1 && { echo "SKIP $ld" >&2; continue; }
    printf '%s\t%s\n' "$ld" "$(job_fields "$alias")" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> $NJOBS shard-job(s), $PER epoch(s) each, gate=$CONC, SMOKE=${SMOKE:-0} -> $LOG_ROOT"
echo ">>> models=${ALIASES:-musemeta grok46} variant=$VARIANT drive=$DRIVE turns=$TURNS"

run_one() {  # ld  model  flags  envp
  local ld="$1" model="$2" flags="$3" envp="$4"; mkdir -p "$ld"; local a=1
  while [ "$a" -le "$RETRIES" ]; do
    echo ">>> [a$a] musegrok ($model) -> $ld"
    env $envp uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/$STEP/task.py" --model "$model" \
      -T scaffold=react -T condition=baseline -T temperature=1.0 -T turn_limit="$TURNS" \
      -T variant="$VARIANT" -T drive="$DRIVE" $flags \
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
echo "=== ALL MUSE/GROK STEP4 CHAIN REASONING SHARDS FINISHED -> $LOG_ROOT ==="
