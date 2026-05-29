"""Run baseline invocation accuracy on all 11 discovered specs.

Multi-tool competition setting: for each test prompt, the LLM sees ALL 11
specs and must pick the right one. This matches how real AI agents operate
and exposes failures (wrong tool, ambiguous descriptions) that single-tool
testing hides.

For each spec:
    1. Generate 5 hard test prompts (TestPromptGenerator)
    2. For each prompt, present ALL 11 specs to LLM, capture chosen tool call
    3. Mechanically classify -> FailureRecord -> append to baseline.jsonl

Output:
    data/test_prompts/{spec.name}.json    - generated prompts
    data/logs/baseline.jsonl              - all FailureRecord entries
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, TestPromptGenerator  # noqa: E402
from src.schema import ToolSpec, append_record  # noqa: E402


N_PROMPTS_PER_TOOL = 5


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    if not specs_dir.exists() or not any(specs_dir.glob("*.json")):
        print(f"ERROR: {specs_dir} empty. Run scripts/run_discoverer.py first.")
        return 1

    prompts_dir = ROOT / "data" / "test_prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    log_path = ROOT / "data" / "logs" / "baseline.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")

    llm = get_client("openai")
    prompt_gen = TestPromptGenerator(llm)
    tester = InvocationTester(llm)

    spec_files = sorted(specs_dir.glob("*.json"))
    all_specs: list[ToolSpec] = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8")) for p in spec_files
    ]

    print(f"Baseline run: {len(all_specs)} tools x {N_PROMPTS_PER_TOOL} prompts each, model={llm.model}")
    print(f"  Multi-tool competition: each prompt sees all {len(all_specs)} specs")
    print("-" * 80)

    per_tool_correct: dict[str, int] = {}
    per_tool_total: dict[str, int] = {}
    global_failure_counts: Counter = Counter()

    t0 = time.time()
    for i, spec in enumerate(all_specs, start=1):
        prompts = prompt_gen.generate(spec, n=N_PROMPTS_PER_TOOL)
        (prompts_dir / f"{spec.name}.json").write_text(
            json.dumps(prompts, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        competing = [s for s in all_specs if s.name != spec.name]
        results = tester.test_batch(spec, prompts, competing_specs=competing)

        correct = sum(1 for r in results if r.failure_type == "correct")
        per_tool_correct[spec.name] = correct
        per_tool_total[spec.name] = len(results)

        for r in results:
            global_failure_counts[r.failure_type] += 1
            rec = r.to_failure_record(spec, iteration=0, model=llm.model)
            append_record(log_path, rec)

        bar = "#" * correct + "." * (len(results) - correct)
        print(f"[{i:2d}/{len(all_specs)}] {spec.name:42s} {correct}/{len(results)} [{bar}]")

    elapsed = time.time() - t0
    total_correct = sum(per_tool_correct.values())
    total = sum(per_tool_total.values())

    print("-" * 80)
    print(f"Done in {elapsed:.1f}s\n")
    print(f"OVERALL ACCURACY: {total_correct}/{total} = {100*total_correct/total:.1f}%")
    print(f"\nFailure-type breakdown:")
    for ft, count in global_failure_counts.most_common():
        marker = "" if ft == "correct" else "  <-- failure"
        print(f"  {ft:22s} {count:3d}{marker}")
    print(f"\nLog written to: {log_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
