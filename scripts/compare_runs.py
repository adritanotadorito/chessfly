#!/usr/bin/env python3
"""Aggregate every Harbor trial in harbor_runs/ into a reference-vs-agent comparison.

Writes reports/harbor_comparison.json and reports/harbor_comparison.md.
Only observed trials are reported; nothing is estimated or filled in.
"""
import json, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "harbor_runs"
SCEN = ["scenario_1_poisoned_material", "scenario_2_acute_threat_defense",
        "scenario_3_tactical_counterplay", "scenario_4_sound_vs_unsound"]


def label(cfg, task):
    a = cfg["agent"]
    name = a.get("name") or a.get("import_path") or "?"
    if task != "harbor_task":
        return f"baseline:{task}"
    if name == "oracle":
        return "reference (oracle)"
    if name == "nop":
        return "do-nothing (nop)"
    return f"{name}:{a.get('model_name')}" if a.get("model_name") else name


def main():
    rows = []
    for res in sorted(RUNS.glob("*/*/result.json")):
        r = json.loads(res.read_text())
        vr = r.get("verifier_result") or {}
        reward = (vr.get("rewards") or {}).get("reward")
        det_p = res.parent / "verifier" / "details.json"
        if not det_p.exists():
            det_p = res.parent / "artifacts" / "logs" / "verifier" / "details.json"
        det = json.loads(det_p.read_text()) if det_p.exists() else {}
        per = {s["scenario_id"]: s for s in det.get("scenarios", [])}
        stdout = res.parent / "verifier" / "test-stdout.txt"
        if not per and stdout.exists():  # fall back to parsing verifier stdout
            for line in stdout.read_text().splitlines():
                for s in SCEN:
                    if line.startswith(f"[{s}"):
                        parts = {k.strip(): v.strip() for k, v in (p.split(":", 1) for p in line.split("]", 1)[1].split("|") if ":" in p)}
                        per[s] = {"score": float(parts.get("Score", 0)), "cp": parts.get("CP"), "reason": parts.get("Reason")}
        rows.append({
            "job": res.parent.parent.name, "trial": r["trial_name"],
            "agent": label(r["config"], r["task_name"]),
            "reward": reward,
            "exception": (r.get("exception_info") or {}).get("exception_type"),
            "scenarios": {s: {"score": per.get(s, {}).get("score"), "cp": per.get(s, {}).get("cp"),
                              "reason": per.get(s, {}).get("reason")} for s in SCEN},
        })
    groups = {}
    for row in rows:
        groups.setdefault(row["agent"], []).append(row)
    summary = []
    for agent, rs in groups.items():
        vals = [x["reward"] for x in rs if x["reward"] is not None]
        summary.append({"agent": agent, "trials": len(rs), "scored": len(vals),
                        "mean": round(statistics.mean(vals), 4) if vals else None,
                        "min": min(vals) if vals else None, "max": max(vals) if vals else None,
                        "rewards": vals})
    order = lambda s: (0 if s["agent"].startswith("reference") else 2 if s["agent"].startswith(("do-nothing", "baseline")) else 1, s["agent"])
    summary.sort(key=order)
    out = {"summary": summary, "trials": rows}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "harbor_comparison.json").write_text(json.dumps(out, indent=2))
    md = ["# ChessFly: Counterplay — Harbor comparison (observed trials only)", "",
          "| Agent | Trials | Mean reward | Min | Max |", "|---|---|---|---|---|"]
    for s in summary:
        md.append(f"| {s['agent']} | {s['scored']}/{s['trials']} | {s['mean']} | {s['min']} | {s['max']} |")
    md += ["", "## Per-scenario (score / final cp from agent's side)", "",
           "| Agent | Trial | " + " | ".join(f"S{i+1}" for i in range(4)) + " | Reward |", "|---|---|---|---|---|---|---|"]
    for row in sorted(rows, key=lambda r: order({"agent": r["agent"]})):
        cells = []
        for s in SCEN:
            d = row["scenarios"][s]
            cells.append("—" if d["score"] is None else f"{d['score']} / {d['cp']}")
        md.append(f"| {row['agent']} | {row['trial']} | " + " | ".join(cells) + f" | {row['reward']} |")
    (ROOT / "reports" / "harbor_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:4 + len(summary)]))


if __name__ == "__main__":
    main()
