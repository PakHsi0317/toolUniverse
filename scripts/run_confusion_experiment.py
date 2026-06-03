"""Phase 5: Confusion-pair experiment — does the Optimizer fix tool-selection
failures caused by semantically overlapping tools?

Motivation: in the natural 11-tool set, only ONE confusable pair exists
(compute_tanimoto_similarity vs find_similar_molecules), so the "right tool
name" dimension of invocation accuracy is barely exercised and baseline is
high (89%). This experiment adds 4 deliberately-overlapping tools to create
4 confusion pairs, then measures:

    1. BEFORE: baseline accuracy on the 8 pair-member tools, under 15-tool
       competition (confusion should drag accuracy DOWN via wrong_tool).
    2. AFTER:  run the Optimizer on each, re-test on the SAME prompts
       (it should sharpen descriptions and recover accuracy).

This is the assignment's Q2 (before/after invocation accuracy) made
discriminative — the original 11-tool before/after was nearly flat (+1.8pp)
because there was almost nothing to confuse.

Confusion pairs (new tool  <->  existing tool, shared semantics):
    search_clinical_trials                    <-> search_biomedical_articles
    get_gene_info                             <-> convert_gene_symbol_to_ensembl_id
    compute_molecular_fingerprint_similarity  <-> compute_tanimoto_similarity
    get_drug_interactions                     <-> get_drug_side_effects

Isolated outputs (existing Phase 0-4 data is untouched):
    data/confusion_specs/*.json              - 15-tool set (11 + 4), pre-seeded
    data/confusion_optimized/{name}.json     - optimized specs (pair members)
    data/logs/confusion.jsonl                - all optimizer iteration records
    data/confusion_report.json               - per-tool + overall before/after
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop, TestPromptGenerator  # noqa: E402
from src.schema import ToolSpec  # noqa: E402


# The 8 tools that belong to a confusion pair (only these are affected by the
# added overlap, so we measure before/after on them).
PAIR_MEMBERS = [
    "search_clinical_trials", "search_biomedical_articles",
    "get_gene_info", "convert_gene_symbol_to_ensembl_id",
    "compute_molecular_fingerprint_similarity", "compute_tanimoto_similarity",
    "get_drug_interactions", "get_drug_side_effects",
]

N_PROMPTS = 5


def main() -> int:
    specs_dir = ROOT / "data" / "confusion_specs"
    if not specs_dir.exists() or len(list(specs_dir.glob("*.json"))) < 15:
        print(f"ERROR: {specs_dir} should contain the 15-tool set. "
              f"Found {len(list(specs_dir.glob('*.json')))}.")
        return 1

    out_specs = ROOT / "data" / "confusion_optimized"
    out_specs.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "data" / "logs" / "confusion.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    llm = get_client("openai")
    prompt_gen = TestPromptGenerator(llm)
    tester = InvocationTester(llm)

    all_specs = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(specs_dir.glob("*.json"))
    ]
    by_name = {s.name: s for s in all_specs}

    print(f"Confusion experiment: {len(all_specs)}-tool competition "
          f"({len(all_specs) - 4} original + 4 added overlap), model={llm.model}")
    print(f"  Measuring before/after on {len(PAIR_MEMBERS)} pair-member tools")
    print("-" * 88)
    print(f"{'Tool':<44}{'before':>8}{'after':>8}{'delta':>8}")
    print("-" * 88)

    report = []
    t0 = time.time()
    for name in PAIR_MEMBERS:
        spec = by_name[name]
        competing = [s for s in all_specs if s.name != name]

        # Fixed prompts (generated once from the ORIGINAL spec) used for BOTH
        # before and after — apples-to-apples, matching run_optimization.py.
        prompts = prompt_gen.generate(spec, n=N_PROMPTS)

        before = tester.test_batch(spec, prompts, competing_specs=competing)
        before_acc = sum(1 for r in before if r.failure_type == "correct") / len(before)

        loop = OptimizerLoop(llm=llm, log_path=log_path,
                             n_prompts=N_PROMPTS, max_iterations=3, target_accuracy=1.0)
        opt = loop.optimize(spec, competing_specs=competing)

        after = tester.test_batch(opt.final_spec, prompts, competing_specs=competing)
        after_acc = sum(1 for r in after if r.failure_type == "correct") / len(after)

        (out_specs / f"{name}.json").write_text(
            opt.final_spec.model_dump_json(indent=2), encoding="utf-8")

        delta = after_acc - before_acc
        arrow = "UP" if delta > 0 else ("==" if delta == 0 else "DN")
        print(f"{name:<44}{before_acc:>7.0%}{after_acc:>8.0%}{delta*100:>+7.0f}pp {arrow}")

        # Which tools did the failures get misrouted to? (confusion evidence)
        misroutes = {}
        for r in before:
            if r.failure_type == "wrong_tool" and r.actual_call:
                tgt = r.actual_call["name"]
                misroutes[tgt] = misroutes.get(tgt, 0) + 1

        report.append({
            "tool_name": name,
            "before_accuracy": before_acc,
            "after_accuracy": after_acc,
            "delta": delta,
            "spec_changed": opt.final_spec.spec_hash() != spec.spec_hash(),
            "terminated_reason": opt.terminated_reason,
            "baseline_misroutes": misroutes,
        })

    elapsed = time.time() - t0
    print("-" * 88)
    avg_before = sum(r["before_accuracy"] for r in report) / len(report)
    avg_after = sum(r["after_accuracy"] for r in report) / len(report)
    improved = sum(1 for r in report if r["delta"] > 0)
    regressed = sum(1 for r in report if r["delta"] < 0)
    print(f"OVERALL (8 pair members): {avg_before:.1%} -> {avg_after:.1%} "
          f"({(avg_after - avg_before)*100:+.1f}pp)")
    print(f"  {improved} improved, {len(report) - improved - regressed} unchanged, "
          f"{regressed} regressed")
    print(f"Done in {elapsed:.1f}s")

    (ROOT / "data" / "confusion_report.json").write_text(
        json.dumps({
            "n_tools_in_competition": len(all_specs),
            "pair_members": PAIR_MEMBERS,
            "avg_before": avg_before,
            "avg_after": avg_after,
            "per_tool": report,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWritten: data/confusion_report.json, data/confusion_optimized/, data/logs/confusion.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
