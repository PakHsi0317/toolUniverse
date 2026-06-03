"""B1 experiment: does the Optimizer help when specs are NATURALLY flawed?

Motivation: on the main 11 tools the Optimizer only gained +1.8pp, because the
Discoverer's specs were already near-perfect (8/11 at 100% baseline). That makes
the Optimizer look weak — but it's really a ceiling effect. B1 removes the
ceiling HONESTLY: feed the Discoverer deliberately VAGUE descriptions so it
produces genuinely flawed specs (wrong types, vague names), then measure whether
the Optimizer recovers accuracy. Unlike Phase 3-B, the bugs here are NOT
injected — they emerge naturally from weak input.

Pipeline (fully isolated under data/lowqual/):
    1. Discover specs from the vague descriptions  → data/lowqual/discovered_specs/
    2. Baseline: each spec competes against the SAME 11 main-line specs as
       distractors (realistic competition), measure accuracy
    3. Optimize each flawed spec (do-no-harm guard ON), re-test on same prompts
    4. Report before/after

Does NOT touch any main-line file. Reads data/discovered_specs/ READ-ONLY just
to use the 11 rich specs as competition distractors.

Output:
    data/lowqual/discovered_specs/{name}.json
    data/lowqual/optimized_specs/{name}.json
    data/lowqual/lowqual_report.json
    data/lowqual/logs/{baseline,optimization}.jsonl
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.discoverer import Discoverer, load_seed_templates  # noqa: E402
from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop, TestPromptGenerator  # noqa: E402
from src.schema import ToolSpec, append_record  # noqa: E402

N_PROMPTS = 5


def main() -> int:
    base = ROOT / "data" / "lowqual"
    src_descs = json.loads(
        (base / "tool_descriptions_lowqual.json").read_text(encoding="utf-8")
    )["tools"]

    specs_dir = base / "discovered_specs"
    opt_dir = base / "optimized_specs"
    logs_dir = base / "logs"
    for d in (specs_dir, opt_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    base_log = logs_dir / "baseline.jsonl"
    opt_log = logs_dir / "optimization.jsonl"
    base_log.write_text("", encoding="utf-8")
    opt_log.write_text("", encoding="utf-8")

    llm = get_client("openai")
    seeds = load_seed_templates()  # same seeds as main line — fairness
    discoverer = Discoverer(llm=llm, seed_templates=seeds)
    prompt_gen = TestPromptGenerator(llm)
    tester = InvocationTester(llm)

    # Main-line rich specs serve as realistic competition distractors (READ-ONLY).
    main_specs = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "data" / "discovered_specs").glob("*.json"))
    ]

    print(f"B1 low-quality experiment: {len(src_descs)} vague tools, model={llm.model}")
    print(f"  Each competes against {len(main_specs)} main-line specs as distractors")
    print("-" * 92)
    print(f"{'Tool (vague desc)':<46}{'before':>8}{'after':>8}{'delta':>8}  reason")
    print("-" * 92)

    report = []
    t0 = time.time()
    for item in src_descs:
        vague = item["lowqual"]
        # 1. Discover a spec from the VAGUE description
        result = discoverer.discover(vague)
        spec = result.spec
        (specs_dir / f"{spec.name}.json").write_text(
            spec.model_dump_json(indent=2), encoding="utf-8"
        )

        # competition = this flawed spec + all main-line specs (minus any name clash)
        competing = [s for s in main_specs if s.name != spec.name]

        # 2. Generate prompts + baseline accuracy
        prompts = prompt_gen.generate(spec, n=N_PROMPTS)
        before = tester.test_batch(spec, prompts, competing_specs=competing)
        before_acc = sum(1 for r in before if r.failure_type == "correct") / len(before)
        for r in before:
            append_record(base_log, r.to_failure_record(spec, iteration=0, model=llm.model))

        # 3. Optimize (guard ON), re-test on SAME prompts (fair before/after)
        loop = OptimizerLoop(llm=llm, log_path=opt_log, n_prompts=N_PROMPTS,
                             max_iterations=3, target_accuracy=1.0, do_no_harm=True)
        opt = loop.optimize(spec, competing_specs=competing)
        after = tester.test_batch(opt.final_spec, prompts, competing_specs=competing)
        after_acc = sum(1 for r in after if r.failure_type == "correct") / len(after)
        (opt_dir / f"{opt.final_spec.name}.json").write_text(
            opt.final_spec.model_dump_json(indent=2), encoding="utf-8"
        )

        delta = after_acc - before_acc
        arrow = "UP" if delta > 0 else ("==" if delta == 0 else "DN")
        print(f"{item['id']+' ('+vague[:24]+'..)':<46}"
              f"{before_acc:>7.0%}{after_acc:>8.0%}{delta*100:>+7.0f}pp  {arrow} {opt.terminated_reason[:30]}")

        report.append({
            "id": item["id"],
            "vague_description": vague,
            "rich_reference": item["rich_reference"],
            "spec_name": spec.name,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "delta": delta,
            "spec_changed": opt.final_spec.spec_hash() != spec.spec_hash(),
            "terminated_reason": opt.terminated_reason,
        })

    elapsed = time.time() - t0
    print("-" * 92)
    n = len(report)
    avg_b = sum(r["before_accuracy"] for r in report) / n
    avg_a = sum(r["after_accuracy"] for r in report) / n
    imp = sum(1 for r in report if r["delta"] > 0)
    reg = sum(1 for r in report if r["delta"] < 0)
    print(f"OVERALL (low-quality specs): {avg_b:.1%} -> {avg_a:.1%} ({(avg_a-avg_b)*100:+.1f}pp)")
    print(f"  {imp} improved, {n-imp-reg} unchanged, {reg} regressed   (done in {elapsed:.1f}s)")
    print(f"\n  Compare: main-line RICH specs were 89.1% -> 90.9% (+1.8pp, ceiling effect)")

    (base / "lowqual_report.json").write_text(
        json.dumps({
            "n_tools": n,
            "avg_before": avg_b,
            "avg_after": avg_a,
            "delta": avg_a - avg_b,
            "improved": imp,
            "regressed": reg,
            "per_tool": report,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten: data/lowqual/lowqual_report.json + discovered_specs/ + optimized_specs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
