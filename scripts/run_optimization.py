"""Run the Optimizer on all 11 discovered specs, then fairly compare
before/after accuracy on the SAME prompts.

Design (critical for fair comparison):
    - Optimizer iterations use ADAPTIVE prompts (paper's design)
    - Final eval re-tests the optimized spec on the ORIGINAL baseline prompts
      so before/after is apples-to-apples
    - Competing specs are the other 10 discovered specs (constant control)

Output:
    data/optimized_specs/{spec.name}.json       - final spec per tool
    data/logs/optimization.jsonl                - all iteration records
    data/logs/final_eval.jsonl                  - final spec re-tested on baseline prompts
    data/optimization_report.json               - before/after summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop  # noqa: E402
from src.schema import ToolSpec, append_record  # noqa: E402


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    prompts_dir = ROOT / "data" / "test_prompts"
    if not (specs_dir.exists() and prompts_dir.exists()):
        print("ERROR: Run scripts/run_baseline.py first (need specs + prompts).")
        return 1

    out_specs = ROOT / "data" / "optimized_specs"
    out_specs.mkdir(parents=True, exist_ok=True)

    opt_log = ROOT / "data" / "logs" / "optimization.jsonl"
    fin_log = ROOT / "data" / "logs" / "final_eval.jsonl"
    opt_log.parent.mkdir(parents=True, exist_ok=True)
    opt_log.write_text("", encoding="utf-8")
    fin_log.write_text("", encoding="utf-8")

    llm = get_client("openai")

    spec_files = sorted(specs_dir.glob("*.json"))
    all_specs: list[ToolSpec] = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8")) for p in spec_files
    ]

    baseline_prompts: dict[str, list[str]] = {}
    for spec in all_specs:
        p = prompts_dir / f"{spec.name}.json"
        if p.exists():
            baseline_prompts[spec.name] = json.loads(p.read_text(encoding="utf-8"))

    tester = InvocationTester(llm)
    report: list[dict] = []

    print(f"Optimization run: {len(all_specs)} tools, model={llm.model}")
    print(f"  Per tool: up to 3 iterations x 5 adaptive prompts")
    print(f"  Final eval: each tool's optimized spec re-tested on its baseline prompts")
    print("-" * 80)

    t0 = time.time()
    for i, spec in enumerate(all_specs, start=1):
        competing = [s for s in all_specs if s.name != spec.name]

        loop = OptimizerLoop(
            llm=llm, log_path=opt_log,
            n_prompts=5, max_iterations=3, target_accuracy=1.0,
        )
        opt_result = loop.optimize(spec, competing_specs=competing)

        baseline_p = baseline_prompts.get(spec.name, [])
        if not baseline_p:
            print(f"[{i:2d}/{len(all_specs)}] {spec.name:42s} (no baseline prompts, skipped)")
            continue

        before_results = tester.test_batch(spec, baseline_p, competing_specs=competing)
        after_results = tester.test_batch(opt_result.final_spec, baseline_p, competing_specs=competing)
        before_acc = sum(1 for r in before_results if r.failure_type == "correct") / len(before_results)
        after_acc = sum(1 for r in after_results if r.failure_type == "correct") / len(after_results)

        for r in after_results:
            rec = r.to_failure_record(opt_result.final_spec, iteration=99, model=llm.model)
            append_record(fin_log, rec)

        (out_specs / f"{spec.name}.json").write_text(
            opt_result.final_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        delta = after_acc - before_acc
        arrow = "UP" if delta > 0 else ("==" if delta == 0 else "DN")
        print(f"[{i:2d}/{len(all_specs)}] {spec.name:42s} "
              f"{before_acc:.0%} -> {after_acc:.0%} {arrow}  "
              f"({opt_result.terminated_reason}, iters={len(opt_result.iterations)})")

        report.append({
            "spec_name": spec.name,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "delta": delta,
            "iterations_run": len(opt_result.iterations),
            "terminated_reason": opt_result.terminated_reason,
            "spec_changed": opt_result.final_spec.spec_hash() != spec.spec_hash(),
        })

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Done in {elapsed:.1f}s\n")

    if report:
        avg_before = sum(r["before_accuracy"] for r in report) / len(report)
        avg_after = sum(r["after_accuracy"] for r in report) / len(report)
        improved = sum(1 for r in report if r["delta"] > 0)
        same = sum(1 for r in report if r["delta"] == 0)
        regressed = sum(1 for r in report if r["delta"] < 0)
        print(f"OVERALL: {avg_before:.1%} -> {avg_after:.1%} ({(avg_after - avg_before)*100:+.1f}pp)")
        print(f"  Per-tool: {improved} improved, {same} unchanged, {regressed} regressed")

    (ROOT / "data" / "optimization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
