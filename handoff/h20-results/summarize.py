# SPDX-License-Identifier: Apache-2.0
"""Recompute the recorded 2A2F results: python summarize.py EXTRACTED_RUN_DIR.

The directory must contain B/E/G sample files. When profiler/ is present, also
recompute the four original traces under profiler/{attention,ffn}. Run locally,
not on a shared cluster login node. Outputs are written beside the raw evidence.
"""

import json
import sys
from pathlib import Path


def merged(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def overlap(left, right):
    i = j = 0
    total = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def main():
    root = Path(sys.argv[1])
    cases = {}
    for phase in ["B", "E", "G"]:
        path = next((root / phase).glob("*0/**/samples*.jsonl"))
        cases[phase] = {
            s["doc_id"]: s
            for line in path.read_text().splitlines()
            if (s := json.loads(line))["filter"] == "strict-match"
        }
    comparisons = {}
    for a, b in [("B", "E"), ("B", "G"), ("E", "G")]:
        assert all(
            cases[a][i]["arguments"] == cases[b][i]["arguments"] for i in cases[a]
        )
        ids = [
            i
            for i in cases[a]
            if cases[a][i]["exact_match"] != cases[b][i]["exact_match"]
        ]
        comparisons[a + b] = {
            "correctness_differences": ids,
            "answer_differences": [
                i
                for i in cases[a]
                if cases[a][i]["filtered_resps"] != cases[b][i]["filtered_resps"]
            ],
            "text_difference_count": sum(
                cases[a][i]["resps"] != cases[b][i]["resps"] for i in cases[a]
            ),
        }
    (root / "comparison.json").write_text(json.dumps(comparisons, indent=2))
    if not (root / "profiler").exists():
        print("Comparison written; no profiler traces in this run.")
        return
    traces = {}
    stats = {}
    for p in (root / "profiler").glob("*/*"):
        d = json.loads(p.read_text())
        events = d["traceEvents"]
        kernels = [e for e in events if e.get("cat") == "kernel"]
        compute = merged(
            [
                (e["ts"], e["ts"] + e["dur"])
                for e in kernels
                if "nccl" not in e["name"].lower()
            ]
        )
        comm = merged(
            [
                (e["ts"], e["ts"] + e["dur"])
                for e in kernels
                if "nccl" in e["name"].lower()
            ]
        )
        key = p.parent.name + "-" + str(kernels[0]["args"]["device"])
        traces[key] = compute
        stats[key] = {
            "base_ns": d["baseTimeNanoseconds"],
            "kernels": len(kernels),
            "graph_kernels": sum(bool(e["args"].get("graph id")) for e in kernels),
            "compute_us": sum(b - a for a, b in compute),
            "nccl_compute_overlap_us": overlap(compute, comm),
            "steps": sorted(
                {
                    e["name"]
                    for e in events
                    if e.get("cat") == "user_annotation"
                    and e["name"].startswith("ProfilerStep")
                }
            ),
        }
    assert len({s["base_ns"] for s in stats.values()}) == 1
    for rank in [0, 1]:
        stats["cross_role_" + str(rank)] = {
            "compute_overlap_us": overlap(
                traces["attention-" + str(rank)], traces["ffn-" + str(rank)]
            )
        }
    (root / "profile-summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
