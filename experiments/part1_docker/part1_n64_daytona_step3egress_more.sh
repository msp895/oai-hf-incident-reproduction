#!/usr/bin/env bash
# Extra models for STEP 3 (v2) egress-share, added to the 260909 gen-3 set. Provider routing
# per the verified 2026-09-09 settings:
#   qwen3.8-max     : OpenRouter, default routing.
#   gemini-3.8-flash: OpenRouter, pin Google + reasoning effort high (the new settings).
#   muse-spark-1.3  : the in-repo native-Meta Responses provider directly (NOT OpenRouter);
#                     needs MODEL_API_KEY / META_API_KEY (daytona_eval forwards them).
# Same task/config as the gen-3 run: react, drive=driven, turn_limit=60, n=64, Opus 4.8 judge.
# Writes into the SAME log root so the plot/ingest pick all seven models up together.
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260909_step3_egress_share_n64}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

TOTAL="${TOTAL:-64}"; SHARDS="${SHARDS:-4}"; CONC="${CONC:-12}"; PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"; TURNS="${TURNS:-60}"; DRIVE="${DRIVE:-driven}"
RETRIES="${RETRIES:-6}"; RETRY_SLEEP="${RETRY_SLEEP:-30}"

job_fields() {  # alias -> "MODEL\tFLAGS"
  case "$1" in
    qwen38max)     printf '%s\t%s' 'openrouter/qwen/qwen3.8-max' '' ;;
    gemini38flash) printf '%s\t%s' 'openrouter/google/gemini-3.8-flash' '-M provider={"only":["Google"]} --reasoning-effort high' ;;
    musespark13)   printf '%s\t%s' 'meta/muse-spark-1.3' '--reasoning-effort high' ;;
  esac
}

JOBS="$LOG_ROOT/_jobs_more.tsv"; mkdir -p "$LOG_ROOT"; : > "$JOBS"
for alias in ${ALIASES:-qwen38max gemini38flash musespark13}; do
  IFS=$'\t' read -r model flags <<< "$(job_fields "$alias")"
  for i in $(seq 1 "$SHARDS"); do
    ld="$LOG_ROOT/react__${alias}__driven/shard${i}"
    ls "$ld"/*.eval >/dev/null 2>&1 && { echo "SKIP $ld" >&2; continue; }
    # field order: MODEL <tab> LD <tab> FLAGS  (FLAGS last so an empty value can't shift fields)
    printf '%s\t%s\t%s\n' "$model" "$ld" "$flags" >> "$JOBS"
  done
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC -> $LOG_ROOT"

run_one() {  # model  ld  flags
  local model="$1" ld="$2" flags="$3"; mkdir -p "$ld"; local a=1
  while [ "$a" -le "$RETRIES" ]; do
    echo ">>> [a$a] step3egress ($model) -> $ld"
    uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/step3_egress_share/task.py" --model "$model" $flags \
      -T scaffold=react -T temperature=1.0 -T turn_limit="$TURNS" -T drive="$DRIVE" \
      --epochs "$PER" --max-samples "$MAXS" --display plain > "$ld/shard.log" 2>&1
    local rc=$?
    ls "$ld"/*.eval >/dev/null 2>&1 && { echo "=== [DONE $rc] $model -> $ld"; return 0; }
    echo "!!! [RETRY] $model a$a rc=$rc -> $ld"; cp "$ld/shard.log" "$ld/shard.attempt${a}.log" 2>/dev/null
    a=$((a+1)); sleep "$RETRY_SLEEP"
  done
  echo "=== [GAVEUP] $model -> $ld"
}
export -f run_one; export TASKS PER MAXS RETRIES RETRY_SLEEP TURNS DRIVE
cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r m l fl <<< "{}"; run_one "$m" "$l" "$fl"'
echo "=== STEP3 EGRESS EXTRA MODELS FINISHED -> $LOG_ROOT ==="
