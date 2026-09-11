#!/usr/bin/env bash
# Step-1 Google-Drive variant ELICITATION runs: the first-incident-agent reproduction.
# Push glm-5.2 (react) on a workbook whose data is behind external links it cannot reach,
# and score with the Opus-4.8 LLM judge whether it escalates to SSRF / writes to the registry.
#
#   part1_step1_gdrive.sh [n_epochs] [turn_limit]
#
# Config (baked into the task defaults; overridable via -T) — matches the elicitation config
# that best surfaced the target behaviour in step1_boundary_probe:
#   scaffold=react  model=glm-5.2  drive=intense  scorer=judge  temperature=1.0
#   condition=baseline (NO scoreboard / self-preservation framing)
# Resumable: a cell whose log dir already holds a .eval is skipped.
set -uo pipefail
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs/260810_docker_replication}"
TASKS="src/alignment_auditor/docker_replication"
cd "$REPO_ROOT"

N="${1:-8}"
TURNS="${2:-40}"
MODEL="openrouter/z-ai/glm-5.2"
LOGDIR="${LOG_ROOT}/step1_gdrive_probe/react__glm52__elicit_intense"

if ls "${LOGDIR}"/*.eval >/dev/null 2>&1; then
  echo "SKIP (exists): ${LOGDIR}"
  exit 0
fi

echo ">>> step1 gdrive  glm52 react  drive=intense scorer=judge temp=1.0  (n=${N}, turns=${TURNS})"
uv run inspect eval "${TASKS}/step1_gdrive_probe/task.py" \
  --model "$MODEL" \
  -T scaffold=react -T condition=baseline -T drive=intense \
  -T scorer=judge -T temperature=1.0 -T turn_limit="$TURNS" \
  --epochs "$N" --max-samples 8 \
  --log-dir "$LOGDIR" --display plain
echo "=== step1 gdrive complete -> ${LOGDIR} ==="
