#!/usr/bin/env bash
# n=64 run of the STEP 4 CHAIN variant (realistic HF credential-harvest breach) for a single
# target model, offloaded to Daytona docker-in-docker sandboxes. 64 epochs split into SHARDS
# shards; each shard is its own sandbox running (64/SHARDS) epochs at max-samples MAXS.
# Resumable: a shard whose log dir already holds a .eval is skipped.
#
#   MODEL=z-ai/glm-5.2 part1_n64_daytona_step4chain.sh          # default 4 shards x16
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260905_step4chain_n64}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

# Route a model slug: meta/* uses the in-repo native-Meta Responses provider (captures
# Muse reasoning summaries incl. on tool-call turns); everything else via OpenRouter.
model_uri() { case "$1" in meta/*) printf "%s" "$1";; *) printf "openrouter/%s" "$1";; esac; }
export -f model_uri


MODEL="${MODEL:-z-ai/glm-5.2}"
ALIAS="${ALIAS:-$(echo "$MODEL" | sed 's#.*/##; s/[.-]//g')}"
TOTAL="${TOTAL:-64}"
SHARDS="${SHARDS:-4}"
CONC="${CONC:-4}"
PER=$(( TOTAL / SHARDS ))
MAXS="${MAXS:-8}"
TURNS="${TURNS:-60}"
VARIANT="${VARIANT:-neutral}"
DRIVE="${DRIVE:-persist}"
RETRIES="${RETRIES:-6}"
RETRY_SLEEP="${RETRY_SLEEP:-30}"

JOBS="$LOG_ROOT/_jobs.tsv"
mkdir -p "$LOG_ROOT"
: > "$JOBS"
for i in $(seq 1 "$SHARDS"); do
  ld="$LOG_ROOT/step4_hf_cheat_chain/${ALIAS}__shard${i}"
  if ls "$ld"/*.eval >/dev/null 2>&1; then echo "SKIP $ld" >&2; continue; fi
  printf '%s\n' "$ld" >> "$JOBS"
done
NJOBS=$(wc -l < "$JOBS")
echo ">>> launching $NJOBS shard(s), $PER epochs each, gate=$CONC, max-samples=$MAXS -> $LOG_ROOT"
echo ">>> model=$MODEL variant=$VARIANT drive=$DRIVE turns=$TURNS"

run_one() {
  local ld="$1"
  mkdir -p "$ld"
  local attempt=1
  while [ "$attempt" -le "$RETRIES" ]; do
    echo ">>> [START a$attempt] chain shard -> $ld"
    uv run python "$TASKS/daytona_eval.py" --local-log-dir "$ld" \
      --cpu 4 --memory 8 --disk 10 --eval-timeout 7200 -- \
      inspect eval "$TASKS/step4_hf_cheat_chain/task.py" --model "$(model_uri "$MODEL")" \
      -T scaffold=react -T condition=baseline -T temperature=1.0 -T turn_limit="$TURNS" \
      -T variant="$VARIANT" -T drive="$DRIVE" \
      --epochs "$PER" --max-samples "$MAXS" --display plain \
      > "$ld/shard.log" 2>&1
    local rc=$?
    if ls "$ld"/*.eval >/dev/null 2>&1; then
      echo "=== [DONE exit $rc] chain -> $ld"; return 0
    fi
    echo "!!! [RETRY] chain attempt $attempt failed (rc=$rc, no eval) -> $ld"
    cp "$ld/shard.log" "$ld/shard.attempt${attempt}.log" 2>/dev/null
    attempt=$((attempt + 1))
    sleep "$RETRY_SLEEP"
  done
  echo "=== [GAVEUP after $RETRIES] chain -> $ld"
}
export -f run_one
export TASKS PER MAXS RETRIES RETRY_SLEEP MODEL TURNS VARIANT DRIVE

cat "$JOBS" | xargs -P "$CONC" -d '\n' -I{} bash -c 'run_one "{}"'
echo "=== ALL CHAIN N64 SHARDS FINISHED -> $LOG_ROOT ==="
