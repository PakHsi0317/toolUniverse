"""Phase 3-B v3: Comprehensive controlled degradation + recovery experiment.

Expands from v2's 5-tool single-bug design to ALL 11 tools with TWO bug
scenarios per tool (22 cells total):
  - Single bug: wrong_type only (the known-damaging bug from v2)
  - Compound bug: wrong_type + empty_description (tests multi-field repair)

This gives the Optimizer 22 chances to recover specs (vs 20 in v2 of which
only 2 cells actually showed damage). Per-tool damage profiles reveal which
specs are robust vs fragile, and the single-vs-compound comparison tests
whether bugs damage additively.

Bug types (from v2, kept for reference):
    EMPTY_DESCRIPTION    : spec.description = 'Tool.'
    NO_PARAM_DESCRIPTIONS: every parameters[i].description = 'A parameter.'
    WRONG_TYPE           : flip the first parameter's type (number<->string)
    REQUIRED_TO_OPTIONAL : every parameters[i].required = False

Scenarios actually run in v3:
    "wrong_type_only"          : [WRONG_TYPE]
    "wrong_type_plus_empty"    : [WRONG_TYPE, EMPTY_DESCRIPTION]

Each (tool, scenario) cell:
    1. Inject all bugs in scenario → degraded spec
    2. Measure invocation accuracy on baseline prompts → degraded_acc
    3. Run OptimizerLoop on degraded spec → recovered spec
    4. Measure recovered spec on same baseline prompts → recovered_acc
    5. Report: clean_acc, degraded_acc, recovered_acc, recovery_rate

Output:
    data/logs/degradation.jsonl         — all FailureRecords
    data/degradation_results.json       — per-(tool, scenario) summary
    Console: table + per-scenario aggregation
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop  # noqa: E402
from src.schema import ToolSpec  # noqa: E402


BugType = Literal[
    "empty_description",
    "no_param_descriptions",
    "wrong_type",
    "required_to_optional",
]


# v3 design: 2 scenarios applied to ALL 11 tools = 22 cells
SCENARIOS: dict[str, list[BugType]] = {
    "wrong_type_only":       ["wrong_type"],
    "wrong_type_plus_empty": ["wrong_type", "empty_description"],
}


_TYPE_FLIP = {
    "string": "integer",
    "integer": "string",
    "number": "string",
    "boolean": "integer",
    "array": "string",
    "object": "string",
}


def apply_bug(spec: ToolSpec, bug: BugType) -> ToolSpec:
    """Apply a single bug to a spec. Returns new spec (immutable)."""
    if bug == "empty_description":
        return spec.set_field("description", "Tool.")

    if bug == "no_param_descriptions":
        out = spec
        for i in range(len(spec.parameters)):
            out = out.set_field(f"parameters[{i}].description", "A parameter.")
        return out

    if bug == "wrong_type":
        if not spec.parameters:
            return spec
        original = spec.parameters[0].type
        flipped = _TYPE_FLIP[original]
        return spec.set_field("parameters[0].type", flipped)

    if bug == "required_to_optional":
        out = spec
        for i in range(len(spec.parameters)):
            out = out.set_field(f"parameters[{i}].required", False)
        return out

    raise ValueError(f"Unknown bug: {bug}")


def apply_scenario(spec: ToolSpec, bugs: list[BugType]) -> ToolSpec:
    """Apply multiple bugs sequentially."""
    out = spec
    for bug in bugs:
        out = apply_bug(out, bug)
    return out


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    prompts_dir = ROOT / "data" / "test_prompts"

    # Load all 11 discovered specs (used as both targets AND competing pool)
    all_specs: list[ToolSpec] = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(specs_dir.glob("*.json"))
    ]

    # Build prompts map keyed by spec.name
    baseline_prompts: dict[str, list[str]] = {}
    for spec in all_specs:
        path = prompts_dir / f"{spec.name}.json"
        if path.exists():
            baseline_prompts[spec.name] = json.loads(path.read_text(encoding="utf-8"))

    llm = get_client("openai")
    tester = InvocationTester(llm)
    log_path = ROOT / "data" / "logs" / "degradation.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    results: list[dict] = []
    n_cells = len(all_specs) * len(SCENARIOS)
    print(f"Degradation experiment v3: {len(all_specs)} tools x {len(SCENARIOS)} scenarios = {n_cells} cells")
    print(f"  Model: {llm.model}")
    print(f"  Scenarios:")
    for name, bugs in SCENARIOS.items():
        print(f"    {name:<25} → {' + '.join(bugs)}")
    print("=" * 110)
    print(f"{'Tool':<40} {'Scenario':<24} {'Clean':>8} {'Degraded':>10} {'Recovered':>10} {'Net':>8}")
    print("=" * 110)

    t0 = time.time()
    for spec in all_specs:
        prompts = baseline_prompts.get(spec.name, [])
        if not prompts:
            print(f"  {spec.name}: no baseline prompts, skipping")
            continue
        competing = [s for s in all_specs if s.name != spec.name]

        # Clean baseline (reference for all scenarios on this tool)
        clean_results = tester.test_batch(spec, prompts, competing)
        clean_acc = sum(1 for r in clean_results if r.failure_type == "correct") / len(clean_results)

        for scenario_name, bugs in SCENARIOS.items():
            try:
                degraded = apply_scenario(spec, bugs)
            except Exception as e:
                print(f"  {spec.name} / {scenario_name}: inject failed: {e}")
                continue

            # Measure degraded
            deg_results = tester.test_batch(degraded, prompts, competing)
            degraded_acc = sum(1 for r in deg_results if r.failure_type == "correct") / len(deg_results)

            # Run Optimizer
            loop = OptimizerLoop(
                llm=llm, log_path=log_path,
                n_prompts=5, max_iterations=3, target_accuracy=1.0,
            )
            opt_result = loop.optimize(degraded, competing_specs=competing)

            # Measure recovered
            rec_results = tester.test_batch(opt_result.final_spec, prompts, competing)
            recovered_acc = sum(1 for r in rec_results if r.failure_type == "correct") / len(rec_results)

            damage = clean_acc - degraded_acc
            recovered = recovered_acc - degraded_acc
            recovery_rate = (recovered / damage) if damage > 0 else 0.0

            results.append({
                "tool_name": spec.name,
                "scenario": scenario_name,
                "bugs": list(bugs),
                "clean_acc": clean_acc,
                "degraded_acc": degraded_acc,
                "recovered_acc": recovered_acc,
                "damage": damage,
                "recovery": recovered,
                "recovery_rate": recovery_rate,
                "iterations_run": len(opt_result.iterations),
                "terminated_reason": opt_result.terminated_reason,
            })

            print(f"{spec.name:<40} {scenario_name:<24} {clean_acc:>7.0%} "
                  f"{degraded_acc:>9.0%} {recovered_acc:>9.0%} "
                  f"{(recovered_acc-degraded_acc)*100:>+7.0f}pp")

    elapsed = time.time() - t0
    print("=" * 110)
    print(f"Done in {elapsed:.1f}s\n")

    # Aggregate by scenario
    print("Per-scenario summary:")
    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)
    for scenario, rows in by_scenario.items():
        avg_clean = sum(r["clean_acc"] for r in rows) / len(rows)
        avg_deg = sum(r["degraded_acc"] for r in rows) / len(rows)
        avg_rec = sum(r["recovered_acc"] for r in rows) / len(rows)
        avg_dmg = sum(r["damage"] for r in rows) / len(rows)
        # Only count tools where damage actually manifested
        actually_damaged = [r for r in rows if r["damage"] > 0]
        n_damaged = len(actually_damaged)
        if actually_damaged:
            rec_rate = sum(r["recovery_rate"] for r in actually_damaged) / n_damaged
            full_recovery = sum(1 for r in actually_damaged if r["recovered_acc"] >= r["clean_acc"])
        else:
            rec_rate = 0.0
            full_recovery = 0
        print(f"  {scenario:<25}")
        print(f"    n={len(rows)} cells | avg clean={avg_clean:.0%} | avg degraded={avg_deg:.0%} | avg recovered={avg_rec:.0%}")
        print(f"    avg damage={avg_dmg:+.0%} | tools actually damaged={n_damaged}/{len(rows)}")
        print(f"    recovery_rate on damaged={rec_rate:.0%} | full_recovery={full_recovery}/{n_damaged}")

    (ROOT / "data" / "degradation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWritten: data/degradation_results.json, data/logs/degradation.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
