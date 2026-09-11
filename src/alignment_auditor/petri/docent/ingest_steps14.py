"""Upload the final Petri step-1 (deposit) and step-4 (HF cheat) audits to Docent,
one AgentRun per audit with a TranscriptGroup of auditor/target/judge transcripts.

    uv run python -m alignment_auditor.petri.docent.ingest_steps14 --dry-run
    uv run python -m alignment_auditor.petri.docent.ingest_steps14            # uploads
"""
import argparse

from docent.data_models.chat.checks import check_agent_runs
from alignment_auditor.petri.docent.petri_to_docent import build_agent_runs, upload

STEPS = {
    "1": dict(
        cell_dir="logs/260814_ladder_glm52_L0/scored",
        behaviour_file="glm52__glm52__custom_opus48.eval",
        judge_files={"validity": "glm52__glm52__validity_opus48.eval"},
        collection="step1-deposit-glm52-n21",
        desc=("Step 1 deposit. glm52 auditor+target, opus48 judges, n=21. One AgentRun/audit; "
              "TranscriptGroup = auditor, target branches, judge:custom (strict behaviour), "
              "judge:validity. Fields: rung_level, signature, scenario_valid, "
              "signature_and_valid, unsanctioned, pushiness, realism, honest_path_available, "
              "persona_pressure."),
    ),
    "4": dict(
        cell_dir="logs/260819_rung4_hf_cheat_rubsub_n128/scored",
        behaviour_file="glm52__glm52__custom_opus48.eval",
        judge_files={"validity": "glm52__glm52__validity_opus48.eval"},
        collection="step4-hf-cheat-glm52-n42",
        desc=("Step 4 HF cheat. glm52 auditor+target, opus48 judges, n=42 (one behaviour-judge "
              "parse failure kept as rung_level=-1; 41 contribute to the published rate). One "
              "AgentRun/audit; TranscriptGroup = auditor, target branches, judge:custom, "
              "judge:validity. Fields: rung_level, signature, scenario_valid, "
              "signature_and_valid, unsanctioned, pushiness, realism, honest_path_available, "
              "persona_pressure."),
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", choices=["1", "4"], default=None)
    args = ap.parse_args()

    for step, cfg in STEPS.items():
        if args.only and step != args.only:
            continue
        runs = build_agent_runs(cfg["cell_dir"], cfg["behaviour_file"], cfg["judge_files"],
                                extra_meta={"step": step})
        rep = check_agent_runs(runs)
        n = len(runs)
        sig = sum(r.metadata.get("signature") for r in runs)
        val = sum(bool(r.metadata.get("scenario_valid")) for r in runs)
        sv = sum(r.metadata.get("signature_and_valid") for r in runs)
        ntr = {len(r.transcripts) for r in runs}
        print(f"step {step}: n={n} transcripts/run={ntr} signature={sig} valid={val} "
              f"sig&valid={sv} warnings={rep.warning_count} {rep.counts_by_code}")
        if not args.dry_run:
            upload(runs, cfg["collection"], cfg["desc"])
    if args.dry_run:
        print("\n[dry-run] not uploading")


if __name__ == "__main__":
    main()
