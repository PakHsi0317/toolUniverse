"""Phase 4: Five-Dimension Invocation Evaluation (activates dimension ⑤ values_ok).

Where run_baseline.py reports ONE headline number (a call is correct or not),
this script breaks every call into 5 INDEPENDENT correctness dimensions and
reports a per-dimension pass rate. It additionally activates dimension ⑤
(values_ok): whether the literal value in the prompt (e.g. a PMID) actually
landed in an argument — a semantic check the single-label classifier never did.

Pipeline per tool (multi-tool competition, same as baseline):
    1. TestPromptGenerator.generate_with_salient() → prompts + ground-truth values
    2. For each prompt, present ALL specs, capture the chosen call
    3. diagnose_dimensions(spec, call, salient_values) → 5-D vector
    4. Aggregate per-dimension pass rates + co-occurring-defect stats

This is ADDITIVE and isolated — it reads discovered_specs/ but writes only:
    data/test_prompts_gt/{spec.name}.json   - prompts WITH salient_values
    data/dimension_eval.json                - per-dimension + per-tool report
    data/logs/dimension_eval.jsonl          - one row per call (full 5-D vector)

It does NOT touch test_prompts/, baseline.jsonl, optimization, or degradation
data, so all existing Phase 1–3 results remain byte-for-byte reproducible.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, TestPromptGenerator  # noqa: E402
from src.schema import ToolSpec  # noqa: E402


N_PROMPTS_PER_TOOL = 5
DIMENSIONS = ["tool_name", "required_present", "no_hallucination", "types_ok", "values_ok"]


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    if not specs_dir.exists() or not any(specs_dir.glob("*.json")):
        print(f"ERROR: {specs_dir} empty. Run scripts/run_discoverer.py first.")
        return 1

    gt_dir = ROOT / "data" / "test_prompts_gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / "data" / "logs" / "dimension_eval.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    llm = get_client("openai")
    prompt_gen = TestPromptGenerator(llm)
    tester = InvocationTester(llm)

    spec_files = sorted(specs_dir.glob("*.json"))
    all_specs: list[ToolSpec] = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8")) for p in spec_files
    ]

    print(f"Dimension eval: {len(all_specs)} tools x {N_PROMPTS_PER_TOOL} prompts, model={llm.model}")
    print(f"  5 dimensions, multi-tool competition, dimension ⑤ values_ok ACTIVE")
    print("-" * 96)
    header = f"{'Tool':<38}" + "".join(f"{d[:6]:>8}" for d in DIMENSIONS) + f"{'allOK':>8}{'#gt':>5}"
    print(header)
    print("-" * 96)

    # dim -> [passes, applicable_total]
    dim_tally: dict[str, list[int]] = {d: [0, 0] for d in DIMENSIONS}
    per_tool: list[dict] = []
    n_all_correct = 0
    n_total = 0
    n_multi_defect = 0
    log_f = log_path.open("a", encoding="utf-8")

    t0 = time.time()
    for i, spec in enumerate(all_specs, start=1):
        gt_items = prompt_gen.generate_with_salient(spec, n=N_PROMPTS_PER_TOOL)
        (gt_dir / f"{spec.name}.json").write_text(
            json.dumps([it.model_dump() for it in gt_items], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        competing = [s for s in all_specs if s.name != spec.name]

        tool_dim_pass: dict[str, int] = {d: 0 for d in DIMENSIONS}
        tool_dim_appl: dict[str, int] = {d: 0 for d in DIMENSIONS}
        tool_all_correct = 0
        n_gt = 0

        for it in gt_items:
            sal = it.salient_values or None
            if sal:
                n_gt += 1
            res = tester.test_one(spec, it.prompt, competing_specs=competing,
                                  salient_values=sal)
            dims = res.dimensions
            n_total += 1
            if dims.all_correct:
                n_all_correct += 1
                tool_all_correct += 1
            if len(dims.failed_dimensions()) >= 2:
                n_multi_defect += 1

            dd = dims.as_dict()
            for d in DIMENSIONS:
                val = dd[d]
                if val is None:
                    continue  # dimension not applicable (values_ok w/o ground truth)
                tool_dim_appl[d] += 1
                dim_tally[d][1] += 1
                if val:
                    tool_dim_pass[d] += 1
                    dim_tally[d][0] += 1

            log_f.write(json.dumps({
                "tool_name": spec.name,
                "prompt": it.prompt,
                "salient_values": it.salient_values,
                "actual_call": res.actual_call,
                "failure_type": res.failure_type,
                "dimensions": dd,
            }, ensure_ascii=False) + "\n")

        # format row: each dim shows pass/appl
        cells = ""
        for d in DIMENSIONS:
            appl = tool_dim_appl[d]
            cells += f"{(str(tool_dim_pass[d])+'/'+str(appl)) if appl else '  -':>8}"
        print(f"{spec.name:<38}{cells}{str(tool_all_correct)+'/'+str(len(gt_items)):>8}{n_gt:>5}")

        per_tool.append({
            "tool_name": spec.name,
            "all_correct": tool_all_correct,
            "n_prompts": len(gt_items),
            "n_with_ground_truth": n_gt,
            "dim_pass": tool_dim_pass,
            "dim_applicable": tool_dim_appl,
        })

    log_f.close()
    elapsed = time.time() - t0
    print("-" * 96)
    print(f"Done in {elapsed:.1f}s\n")

    print(f"OVERALL all-5-correct: {n_all_correct}/{n_total} = {100*n_all_correct/n_total:.1f}%")
    print(f"Calls with >=2 co-occurring defects: {n_multi_defect} "
          f"(single-label would hide the secondary ones)\n")
    print("Per-dimension pass rate (independent, no short-circuit):")
    for d in DIMENSIONS:
        p, t = dim_tally[d]
        rate = f"{100*p/t:.0f}%" if t else "n/a (no ground truth)"
        print(f"  {d:20s} {p:3d}/{t:<3d}  {rate}")

    report = {
        "model": llm.model,
        "n_total_calls": n_total,
        "n_all_correct": n_all_correct,
        "n_multi_defect_calls": n_multi_defect,
        "per_dimension": {
            d: {"pass": dim_tally[d][0], "applicable": dim_tally[d][1]}
            for d in DIMENSIONS
        },
        "per_tool": per_tool,
    }
    (ROOT / "data" / "dimension_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nWritten: data/dimension_eval.json, data/logs/dimension_eval.jsonl, data/test_prompts_gt/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
