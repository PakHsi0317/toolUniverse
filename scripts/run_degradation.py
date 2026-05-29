"""Phase 3-B v4: Multi-dimensional controlled degradation experiment.

Expands v3 (22 cells: 11 tools x 2 scenarios) to 66 cells (11 tools x
6 scenarios) testing 4 different "accuracy dimensions" of spec quality.

Scenarios test atomic bug dimensions PLUS compound combinations:

  Atomic dimensions (single bug each):
    S1: wrong_type_only           - type system damage
    S2: wrong_param_name_only     - parameter naming damage
    S3: misleading_description    - tool selection damage (description-driven)
    S4: add_fake_required_param   - extra/phantom parameter

  Compound (multiple bugs):
    S5: wrong_type_plus_empty           - type + description-empty
    S6: wrong_type_plus_misleading      - type + description-swap

Each (tool, scenario) cell:
    1. Inject all bugs in scenario → degraded spec
    2. Measure invocation accuracy → degraded_acc
    3. Run OptimizerLoop → recovered spec
    4. Measure recovered_acc
    5. Report damage and recovery_rate

The 4 atomic dimensions test different fields the Optimizer must repair:
    wrong_type           → parameters[i].type
    wrong_param_name     → parameters[i].name
    misleading_desc      → description
    add_fake_required    → parameters list (must remove a param)

Output:
    data/logs/degradation.jsonl
    data/degradation_results.json
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
from src.schema import Parameter, ToolSpec  # noqa: E402


BugType = Literal[
    "empty_description",
    "wrong_type",
    "wrong_param_name",
    "misleading_description",
    "add_fake_required_param",
]


SCENARIOS: dict[str, list[BugType]] = {
    "wrong_type_only":            ["wrong_type"],
    "wrong_param_name_only":      ["wrong_param_name"],
    "misleading_description":     ["misleading_description"],
    "add_fake_required_param":    ["add_fake_required_param"],
    "wrong_type_plus_empty":      ["wrong_type", "empty_description"],
    "wrong_type_plus_misleading": ["wrong_type", "misleading_description"],
}


_TYPE_FLIP = {
    "string": "integer",
    "integer": "string",
    "number": "string",
    "boolean": "integer",
    "array": "string",
    "object": "string",
}


def apply_bug(spec: ToolSpec, bug: BugType, context: dict | None = None) -> ToolSpec:
    """Apply a single bug to a spec. Some bugs need context (e.g., other specs)."""
    context = context or {}

    if bug == "empty_description":
        return spec.set_field("description", "Tool.")

    if bug == "wrong_type":
        if not spec.parameters:
            return spec
        original = spec.parameters[0].type
        flipped = _TYPE_FLIP[original]
        return spec.set_field("parameters[0].type", flipped)

    if bug == "wrong_param_name":
        if not spec.parameters:
            return spec
        # Rename first parameter to an opaque name
        return spec.set_field("parameters[0].name", "input_x")

    if bug == "misleading_description":
        # Replace description with a partner tool's description (deterministic swap)
        all_specs = context.get("all_specs", [])
        others = [s for s in all_specs if s.name != spec.name]
        if not others:
            return spec
        # Deterministic: pick the spec whose name comes next alphabetically
        sorted_others = sorted(others, key=lambda s: s.name)
        sorted_all = sorted(all_specs, key=lambda s: s.name)
        idx = next(i for i, s in enumerate(sorted_all) if s.name == spec.name)
        partner = sorted_all[(idx + 1) % len(sorted_all)]
        if partner.name == spec.name:  # safety
            partner = others[0]
        return spec.set_field("description", partner.description)

    if bug == "add_fake_required_param":
        # Add a fake plausible-sounding parameter that doesn't belong
        fake = Parameter(
            name="auth_token",
            type="string",
            description="Authentication token for the request.",
            required=True,
        )
        new_params = list(spec.parameters) + [fake]
        return spec.model_copy(update={"parameters": new_params})

    raise ValueError(f"Unknown bug: {bug}")


def apply_scenario(spec: ToolSpec, bugs: list[BugType], context: dict) -> ToolSpec:
    out = spec
    for bug in bugs:
        out = apply_bug(out, bug, context)
    return out


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    prompts_dir = ROOT / "data" / "test_prompts"

    all_specs: list[ToolSpec] = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(specs_dir.glob("*.json"))
    ]

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
    print(f"Degradation experiment v4: {len(all_specs)} tools x {len(SCENARIOS)} scenarios = {n_cells} cells")
    print(f"  Model: {llm.model}")
    print(f"  Scenarios:")
    for name, bugs in SCENARIOS.items():
        print(f"    {name:<28} → {' + '.join(bugs)}")
    print("=" * 110)
    print(f"{'Tool':<40} {'Scenario':<28} {'Clean':>7} {'Deg':>7} {'Rec':>7} {'Net':>7}")
    print("=" * 110)

    context = {"all_specs": all_specs}

    t0 = time.time()
    for spec in all_specs:
        prompts = baseline_prompts.get(spec.name, [])
        if not prompts:
            print(f"  {spec.name}: no baseline prompts, skipping")
            continue
        competing = [s for s in all_specs if s.name != spec.name]

        clean_results = tester.test_batch(spec, prompts, competing)
        clean_acc = sum(1 for r in clean_results if r.failure_type == "correct") / len(clean_results)

        for scenario_name, bugs in SCENARIOS.items():
            try:
                degraded = apply_scenario(spec, bugs, context)
            except Exception as e:
                print(f"  {spec.name} / {scenario_name}: inject failed: {e}")
                continue

            deg_results = tester.test_batch(degraded, prompts, competing)
            degraded_acc = sum(1 for r in deg_results if r.failure_type == "correct") / len(deg_results)

            loop = OptimizerLoop(
                llm=llm, log_path=log_path,
                n_prompts=5, max_iterations=3, target_accuracy=1.0,
            )
            opt_result = loop.optimize(degraded, competing_specs=competing)

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

            print(f"{spec.name:<40} {scenario_name:<28} {clean_acc:>6.0%} "
                  f"{degraded_acc:>6.0%} {recovered_acc:>6.0%} "
                  f"{(recovered_acc-degraded_acc)*100:>+6.0f}pp")

    elapsed = time.time() - t0
    print("=" * 110)
    print(f"Done in {elapsed:.1f}s\n")

    # Aggregate by scenario
    print("Per-scenario summary:")
    by_scenario: dict[str, list[dict]] = {}
    for r in results:
        by_scenario.setdefault(r["scenario"], []).append(r)
    print(f"  {'scenario':<28} {'cells':>5} {'clean':>7} {'degr':>7} {'rec':>7} {'damaged':>10} {'full_rec':>10}")
    print(f"  {'-'*28}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*10}  {'-'*10}")
    for scenario, rows in by_scenario.items():
        avg_clean = sum(r["clean_acc"] for r in rows) / len(rows)
        avg_deg = sum(r["degraded_acc"] for r in rows) / len(rows)
        avg_rec = sum(r["recovered_acc"] for r in rows) / len(rows)
        damaged = [r for r in rows if r["damage"] > 0]
        full_rec = [r for r in damaged if r["recovered_acc"] >= r["clean_acc"]]
        n_dmg = len(damaged)
        n_full = len(full_rec)
        print(f"  {scenario:<28} {len(rows):>5} {avg_clean:>7.0%} {avg_deg:>7.0%} {avg_rec:>7.0%} "
              f"{n_dmg:>3}/{len(rows):<3} {' '*3}{n_full:>3}/{n_dmg:<3}")

    (ROOT / "data" / "degradation_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWritten: data/degradation_results.json, data/logs/degradation.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
