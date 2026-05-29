"""Phase 3-A: Failure pattern analysis from existing logs (v2).

Reads baseline.jsonl + optimization_report.json + degradation_results.json
+ phase3c_results.json (if exist) and produces:
  1. Overall failure-type distribution
  2. Per-tool accuracy table
  3. Pairwise confusion matrix (which tools get mistakenly chosen for which)
  4. Optimizer before/after table (from optimization_report)
  5. Degradation summary (from degradation_results)
  6. Adversarial pair findings (from phase3c_results)

Zero new LLM calls — pure aggregation of existing data.

Output:
    data/phase3a_analysis.json      — machine-readable tables
    Console: formatted tables for slides
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(l) for l in lines if l.strip()]


def pct(n: int, total: int) -> str:
    if total == 0:
        return "  —  "
    return f"{100*n/total:5.1f}%"


def main() -> int:
    baseline_path = ROOT / "data" / "logs" / "baseline.jsonl"
    if not baseline_path.exists():
        print("ERROR: data/logs/baseline.jsonl not found. Run run_baseline.py first.")
        return 1

    records = load_jsonl(baseline_path)
    print(f"Loaded {len(records)} baseline records from {baseline_path.name}")

    # ── 1. Overall failure-type breakdown ─────────────────────────────────────
    ft_counter: Counter = Counter(r["failure_type"] for r in records)
    total = len(records)
    failures = [r for r in records if r["failure_type"] != "correct"]

    print(f"\n{'='*80}")
    print(f"  SECTION 1: Overall Failure-Type Breakdown  (n={total})")
    print(f"{'='*80}")
    print(f"  {'failure_type':<25} {'count':>6}  {'of_total':>9}  {'of_failures':>11}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*9}  {'-'*11}")
    for ft, cnt in ft_counter.most_common():
        marker = "" if ft == "correct" else "  <-- failure"
        f_pct = pct(cnt, len(failures)) if ft != "correct" else "     —"
        print(f"  {ft:<25} {cnt:>6}  {pct(cnt, total):>9}  {f_pct:>11}{marker}")

    # ── 2. Per-tool accuracy ──────────────────────────────────────────────────
    tool_total: Counter = Counter()
    tool_correct: Counter = Counter()
    for rec in records:
        tool_total[rec["tool_name"]] += 1
        if rec["failure_type"] == "correct":
            tool_correct[rec["tool_name"]] += 1

    print(f"\n{'='*80}")
    print(f"  SECTION 2: Per-Tool Accuracy")
    print(f"{'='*80}")
    print(f"  {'tool_name':<42} {'correct':>9}  {'total':>6}  {'accuracy':>9}")
    print(f"  {'-'*42}  {'-'*9}  {'-'*6}  {'-'*9}")
    for tool in sorted(tool_total.keys()):
        c = tool_correct[tool]
        t = tool_total[tool]
        print(f"  {tool:<42} {c:>9}  {t:>6}  {pct(c, t):>9}")

    # ── 3. Confusion matrix: who gets confused with whom ──────────────────────
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    for rec in records:
        target = rec["tool_name"]
        actual = rec.get("actual_call", {})
        if not actual:
            continue
        actual_name = actual.get("name")
        if actual_name and actual_name != target:
            pair = tuple(sorted([target, actual_name]))
            confusion[pair] += 1

    print(f"\n{'='*80}")
    print(f"  SECTION 3: Cross-Tool Confusion (target -> actual)")
    print(f"{'='*80}")
    if confusion:
        print(f"  {'pair':<70} {'count':>6}")
        print(f"  {'-'*70}  {'-'*6}")
        for (a, b), cnt in sorted(confusion.items(), key=lambda x: -x[1]):
            print(f"  {a} <-> {b:<{max(1, 70 - len(a) - 5)}}  {cnt:>6}")
    else:
        print("  No cross-tool confusion in baseline (all failures are not wrong_tool)")

    # ── 4. Optimizer impact ───────────────────────────────────────────────────
    opt_path = ROOT / "data" / "optimization_report.json"
    if opt_path.exists():
        report = json.loads(opt_path.read_text(encoding="utf-8"))
        print(f"\n{'='*80}")
        print(f"  SECTION 4: Optimizer Impact (n={len(report)})")
        print(f"{'='*80}")
        if report:
            avg_before = sum(r["before_accuracy"] for r in report) / len(report)
            avg_after = sum(r["after_accuracy"] for r in report) / len(report)
            improved = sum(1 for r in report if r["delta"] > 0)
            same = sum(1 for r in report if r["delta"] == 0)
            regressed = sum(1 for r in report if r["delta"] < 0)
            changed_spec = sum(1 for r in report if r.get("spec_changed"))
            print(f"  Before avg: {avg_before:.1%}   After avg: {avg_after:.1%}   Delta: {(avg_after-avg_before)*100:+.1f}pp")
            print(f"  Spec changed: {changed_spec}/{len(report)}")
            print(f"  Per-tool: {improved} improved, {same} unchanged, {regressed} regressed")
            print(f"\n  {'spec_name':<42} {'before':>7} {'after':>7} {'delta':>7} {'reason'}")
            print(f"  {'-'*42}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*20}")
            for r in report:
                arrow = "UP" if r["delta"] > 0 else ("==" if r["delta"] == 0 else "DN")
                print(f"  {r['spec_name']:<42} {r['before_accuracy']:>7.1%} {r['after_accuracy']:>7.1%} "
                      f"{r['delta']*100:>+6.0f}pp  {r['terminated_reason']}")

    # ── 5. Degradation summary ────────────────────────────────────────────────
    deg_path = ROOT / "data" / "degradation_results.json"
    if deg_path.exists():
        deg = json.loads(deg_path.read_text(encoding="utf-8"))
        print(f"\n{'='*80}")
        print(f"  SECTION 5: Degradation + Recovery Summary (n={len(deg)})")
        print(f"{'='*80}")
        # v3 format uses 'scenario'; v2 used 'bug_type'. Support both.
        key = "scenario" if deg and "scenario" in deg[0] else "bug_type"
        by_grp: dict[str, list] = defaultdict(list)
        for r in deg:
            by_grp[r.get(key, "unknown")].append(r)
        print(f"  {key:<25} {'cells':>5} {'clean':>7} {'degraded':>9} {'recovered':>10} {'damaged':>8} {'full_rec':>9}")
        print(f"  {'-'*25}  {'-'*5}  {'-'*7}  {'-'*9}  {'-'*10}  {'-'*8}  {'-'*9}")
        for grp, rows in by_grp.items():
            avg_c = sum(r["clean_acc"] for r in rows) / len(rows)
            avg_d = sum(r["degraded_acc"] for r in rows) / len(rows)
            avg_r = sum(r["recovered_acc"] for r in rows) / len(rows)
            damaged = [r for r in rows if r["damage"] > 0]
            n_dmg = len(damaged)
            n_full = sum(1 for r in damaged if r["recovered_acc"] >= r["clean_acc"])
            print(f"  {grp:<25} {len(rows):>5} {avg_c:>7.0%} {avg_d:>9.0%} {avg_r:>10.0%} "
                  f"{n_dmg:>3}/{len(rows):>3} {n_full:>3}/{n_dmg:>3}")

    # ── 6. Phase 3-C summary ──────────────────────────────────────────────────
    p3c_path = ROOT / "data" / "phase3c_results.json"
    if p3c_path.exists():
        p3c = json.loads(p3c_path.read_text(encoding="utf-8"))
        print(f"\n{'='*80}")
        print(f"  SECTION 6: Adversarial Pair Findings (Phase 3-C)")
        print(f"{'='*80}")
        for pair in p3c.get("discovered_pairs", []):
            a, b = pair["pair"]
            print(f"\n  Auto-discovered pair: {a} <-> {b}")
            print(f"    Baseline confusion count: {pair['baseline_confusion']}")
            pair_key = f"{a}__vs__{b}"
            exps = p3c["experiments"].get(pair_key, {})
            for exp_name, exp_data in exps.items():
                if isinstance(exp_data, dict) and "label" in exp_data:
                    print(f"    {exp_data['label']:<55} correct={exp_data['correct']}/{exp_data['n']}  confused={exp_data.get('confused_with_partner', 0)}")

    # ── Save machine-readable ─────────────────────────────────────────────────
    output = {
        "failure_type_counts": dict(ft_counter),
        "per_tool_accuracy": {t: {"correct": tool_correct[t], "total": tool_total[t]} for t in tool_total},
        "cross_tool_confusion": {f"{a}__vs__{b}": c for (a, b), c in confusion.items()},
    }
    (ROOT / "data" / "phase3a_analysis.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: data/phase3a_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
