"""B2 experiment: weaken the DISCOVERER (not the input) — does the Optimizer help?

B1 weakened the *input* (vague descriptions). B2 weakens the *Discoverer itself*:
same rich descriptions as the main line, but the spec is generated WITHOUT the
few-shot seed templates (zero-shot). This removes the pattern-guidance that made
the main-line specs so good, and tests whether the Optimizer can recover specs
produced by a weaker generator.

This is the cross-validation of B1: B1 showed a strong Discoverer + weak input
yields structural defects the Optimizer can't fix. B2 asks the complementary
question — strong input + weak Discoverer. If both land at ~+0pp for the same
reason (defects are structural, not field-level), the capability boundary is
confirmed from two independent directions.

Pipeline (isolated under data/noseed/):
    1. Generate specs ZERO-SHOT (SpecGenerator with empty seeds) from the SAME
       11 main-line descriptions
    2. Baseline: each spec competes against the 11 main-line (rich) specs
    3. Optimize (guard on, redesign-detection on, held-out = main-line prompts)
    4. Report before/after + how many hit needs_redesign

Reads main-line files READ-ONLY (descriptions, rich specs as distractors,
baseline prompts as held-out). Writes only under data/noseed/.

Output:
    data/noseed/discovered_specs/{name}.json
    data/noseed/noseed_report.json
    data/noseed/logs/{baseline,optimization}.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.discoverer.spec_generator import SpecGenerator  # noqa: E402
from src.discoverer.stub_generator import generate_stub  # noqa: E402
from src.discoverer.static_validator import StaticValidator  # noqa: E402
from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop, TestPromptGenerator  # noqa: E402
from src.schema import ToolSpec, append_record  # noqa: E402

N_PROMPTS = 5


def main() -> int:
    base = ROOT / "data" / "noseed"
    specs_dir = base / "discovered_specs"
    logs_dir = base / "logs"
    for d in (specs_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    base_log = logs_dir / "baseline.jsonl"
    opt_log = logs_dir / "optimization.jsonl"
    base_log.write_text("", encoding="utf-8")
    opt_log.write_text("", encoding="utf-8")

    # Same rich descriptions as the main line.
    descs = json.loads(
        (ROOT / "data" / "tool_descriptions.json").read_text(encoding="utf-8")
    )["tools"]

    llm = get_client("openai")
    spec_gen = SpecGenerator(llm)
    validator = StaticValidator()
    prompt_gen = TestPromptGenerator(llm)
    tester = InvocationTester(llm)

    # Main-line rich specs = competition distractors; their baseline prompts =
    # held-out set for needs_redesign (real, external distribution).
    main_specs = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "data" / "discovered_specs").glob("*.json"))
    ]
    main_base = [json.loads(l) for l in (ROOT / "data" / "logs" / "baseline.jsonl").open()]

    def held_out_for(name: str) -> list[str]:
        return [r["test_prompt"] for r in main_base if r["tool_name"] == name]

    print(f"B2 zero-shot Discoverer experiment: {len(descs)} tools, model={llm.model}")
    print(f"  Specs generated WITHOUT few-shot seeds (weakened Discoverer)")
    print("-" * 100)
    print(f"{'Tool':<40}{'valid?':<8}{'before':>8}{'after':>8}{'delta':>8}  reason")
    print("-" * 100)

    report = []
    t0 = time.time()
    for desc in descs:
        # ZERO-SHOT generation: empty seeds → no pattern guidance
        spec = spec_gen.generate(desc, seeds=[])
        stub = generate_stub(spec)
        errors = validator.validate(spec, stub)
        (specs_dir / f"{spec.name}.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )

        competing = [s for s in main_specs if s.name != spec.name]
        held = held_out_for(spec.name)  # may be empty if name differs from main line

        prompts = prompt_gen.generate(spec, n=N_PROMPTS)
        before = tester.test_batch(spec, prompts, competing_specs=competing)
        before_acc = sum(1 for r in before if r.failure_type == "correct") / len(before)
        for r in before:
            append_record(base_log, r.to_failure_record(spec, iteration=0, model=llm.model))

        loop = OptimizerLoop(llm=llm, log_path=opt_log, n_prompts=N_PROMPTS,
                             max_iterations=3, target_accuracy=1.0,
                             do_no_harm=True, detect_redesign=True)
        opt = loop.optimize(spec, competing_specs=competing,
                            eval_prompts=held or None)
        after = tester.test_batch(opt.final_spec, prompts, competing_specs=competing)
        after_acc = sum(1 for r in after if r.failure_type == "correct") / len(after)

        delta = after_acc - before_acc
        arrow = "UP" if delta > 0 else ("==" if delta == 0 else "DN")
        valid = "ok" if not errors else f"{len(errors)}err"
        print(f"{spec.name:<40}{valid:<8}{before_acc:>7.0%}{after_acc:>8.0%}"
              f"{delta*100:>+7.0f}pp  {arrow} {opt.terminated_reason[:34]}")

        report.append({
            "description": desc,
            "spec_name": spec.name,
            "valid": not errors,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "delta": delta,
            "needs_redesign": opt.needs_redesign,
            "terminated_reason": opt.terminated_reason,
        })

    elapsed = time.time() - t0
    print("-" * 100)
    n = len(report)
    avg_b = sum(r["before_accuracy"] for r in report) / n
    avg_a = sum(r["after_accuracy"] for r in report) / n
    imp = sum(1 for r in report if r["delta"] > 0)
    reg = sum(1 for r in report if r["delta"] < 0)
    redesign = sum(1 for r in report if r["needs_redesign"])
    print(f"OVERALL (zero-shot Discoverer): {avg_b:.1%} -> {avg_a:.1%} ({(avg_a-avg_b)*100:+.1f}pp)")
    print(f"  {imp} improved, {n-imp-reg} unchanged, {reg} regressed, "
          f"{redesign} flagged needs_redesign   (done in {elapsed:.1f}s)")
    print(f"\n  Compare: B1 (weak input)   89%->64% baseline, +0.0pp")
    print(f"           main (few-shot)    89.1%->90.9%, +1.8pp")

    (base / "noseed_report.json").write_text(
        json.dumps({
            "n_tools": n, "avg_before": avg_b, "avg_after": avg_a,
            "delta": avg_a - avg_b, "improved": imp, "regressed": reg,
            "needs_redesign_count": redesign, "per_tool": report,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten: data/noseed/noseed_report.json + discovered_specs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
