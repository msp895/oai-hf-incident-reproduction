"""Envsrc-access grouped plot for the NEW GENERIC PEER environment
(logs/260911_step2_none_generic_multimodel), reusing the exact styling/make_plot from
plot_step2_envsrc_access.py. Reads counts from step2_envsrc_judge_generic.json
(pooled baseline+collab, n=128/model).

    uv run python -m alignment_auditor.docker_replication.analysis.plot_step2_envsrc_access_generic
"""
from __future__ import annotations

import json
from pathlib import Path

from alignment_auditor.docker_replication.analysis.plot_step2_envsrc_access import (
    make_plot, wilson, ORDER,
)

HERE = Path(__file__).resolve().parent


def main() -> None:
    counts = json.load(open(HERE / "step2_envsrc_judge_generic.json"))["counts"]
    rows = []
    for name in ORDER:
        if name not in counts:
            print("skip", name); continue
        c = counts[name]
        n = c["n"]
        rows.append({"name": name,
                     "finds_api": wilson(c["finds_api"], n),
                     "gets_source": wilson(c["gets_source"], n)})
        print(f"{name:18} finds_api {c['finds_api']:3}/{n}  gets_source {c['gets_source']:3}/{n}")
    make_plot(rows, "Step 2: Reads the eval's own source code",
              "260911_step2_envsrc_access_generic_grouped.png")


if __name__ == "__main__":
    main()
