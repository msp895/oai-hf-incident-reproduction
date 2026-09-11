---
name: petri-to-docent
description: Upload Petri audit .eval logs to Docent/Transluce with the repo's standard shape — one AgentRun per audit whose TranscriptGroup holds separate auditor, target, and per-judge transcripts. Use whenever uploading, ingesting, or importing Petri/inspect audit transcripts (from logs/<run>/scored/*.eval) into Docent for this project. Prefer this over the generic docent ingestion skill for Petri data.
---

# Petri → Docent ingestion (repo convention)

When uploading Petri audits to Docent for this project, ALWAYS use this shape. Do not fall back to
the plain target-only `load_inspect_log` upload.

## The shape (required)

- **One `AgentRun` per audit** (per sample/epoch).
- **One `TranscriptGroup` per AgentRun**, named `audit`, containing a **separate `Transcript`
  per participant**:
  - `target` — the agent under test, from its OWN ModelEvents. If the auditor restarted or
    rolled back, each branch is its own `target:branchN` transcript (chronological).
  - `auditor` — the model that role-played the environment
  - `judge:<label>` — **one transcript per judge** that scored the audit (e.g. `judge:custom`,
    `judge:validity`, and any extra behaviour block such as `judge:reachout_broad`)
- **Flat, filterable metadata** on the AgentRun: `step`, `rung_level` (0–4; `-1` = judge parse
  fail), `signature`, `scenario_valid`, `signature_and_valid`, the validity sub-fields
  (`unsanctioned`, `pushiness`, `realism`, `honest_path_available`, `peers_undirected`,
  `persona_pressure`), each judge's `*_headline`/`*_rationale`, plus any extra-behaviour
  `<label>_signature`/`<label>_level`, and `tags`.

## How to do it

The converter already implements all of this. Use it — do not re-derive:

```python
from alignment_auditor.petri.docent.petri_to_docent import build_agent_runs, upload
from docent.data_models.chat.checks import check_agent_runs

runs = build_agent_runs(
    cell_dir="logs/<run>/scored",
    behaviour_file="glm52__glm52__custom_opus48.eval",     # carries auditor+target events + rung judge
    judge_files={"validity": "glm52__glm52__validity_opus48.eval",
                 # add any extra behaviour judges, e.g.:
                 "reachout_broad": "glm52__glm52__reachout_broad_opus48.eval"},
    extra_meta={"step": "3"},
)
print(check_agent_runs(runs).counts_by_code)   # sanity
upload(runs, collection_name="<slug>", description="...")   # creates fresh collection, verifies count
```

`src/alignment_auditor/petri/docent/ingest_steps23.py` is a worked driver (steps 2 & 3). Copy its
`STEPS` block for a new run.

## Gotchas the converter already handles (know them, don't re-hit them)

- **All judge blocks share the inspect scorer name `seed_custom_judge`.** Distinguish judges by
  the **filename**, never the scorer name.
- **ModelEvents carry `.role`** (`auditor`/`target`); the judge model call has `role=None` + the
  judge model. The judge transcript is the judge's single call (`input` = prompt with the
  transcript, `output` = verdict).
- **`sample.messages` is the AUDITOR's thread, never the target's.** The target's replies appear
  in it only as `<target_response>` tool results. Use it for the `auditor` transcript; build the
  `target` transcript from the last `role='target'` ModelEvent's `input` + `output`. Using
  `sample.messages` for both uploads the auditor twice under two names — that was a real bug
  (fixed 260821); if two transcripts in a collection look identical, this is why.
- **One audit holds SEVERAL target conversations.** A restart/rollback truncates the target's
  history, so a ModelEvent whose `input` did not grow starts a new branch (2–4 per audit is
  typical). The judged behaviour can sit in any branch — the 260821 reach-out hits were in
  "Branch 3" and "Branch 2" — so upload every branch, not just the last.
- **Resolve attachments** (`resolve_sample_attachments`) or reasoning shows as `attachment://…`.
- **`scoring_metadata` holds inspect `Score` OBJECTS** whose `.value` is `NaN` on judge
  parse-failure. A recursive float-clean does NOT reach into objects, so the converter DROPS
  `scoring_metadata`; leaving it causes a server-side 400 (`non-finite float`). Verdicts are kept
  as flat fields instead.
- **Docent reads are eventually-consistent.** After `add_agent_runs`, poll `list_agent_run_ids`
  with short sleeps before trusting a count (the converter's `upload()` does this).
- **`check_agent_runs` returns a `TranscriptCheckReport`** (`.warnings`, `.warning_count`,
  `.counts_by_code`), not a list. Import from `docent.data_models.chat.checks`.

## Expected, benign warnings (do NOT try to "fix" these)

- `missing_tool_response` (~1/run): the auditor's terminal tool call (e.g. `end_conversation`)
  legitimately gets no response because the audit ends there. Faithful, not a bug.
- `empty_message`: a judge that returned no parseable verdict (also `rung_level=-1`).
- `reasoning_embedded_as_text` (a few): a message with a `thinking:`-style marker.

## Don't

- Don't edit the vendored `docent` plugin skill (plugin cache, overwritten on update).
- Don't upload target-only runs for Petri data — always include auditor + judges.
