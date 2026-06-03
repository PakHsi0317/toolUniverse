"""Phase 7: needs_redesign — does the Optimizer KNOW what it can't fix?

B1 (Phase 6) showed the Optimizer can't repair structural defects (vague,
over-broad tool identities). The honest next step isn't to force a fix — it's
to make the system DETECT the boundary and report it, so the defect can be
routed back to the Discoverer for redesign instead of silently burning 3 rounds.

Key methodology: structural defects HIDE under the loop's self-generated prompts
(those prompts are written for the spec, so they flatter it). We therefore judge
needs_redesign on a HELD-OUT prompt set (the real, external B1 baseline prompts).
A spec that aces its own prompts but fails held-out ones has a scope that "only
fits its own prompts" — the signature of a structural defect.

Runs every B1 low-quality tool through OptimizerLoop with held-out evaluation,
plus a healthy control tool that must NOT be flagged.

Output:
    data/redesign_detection.json   - per-tool: flagged? + reason + accuracies
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester, OptimizerLoop  # noqa: E402
from src.schema import ToolSpec  # noqa: E402


def main() -> int:
    llm = get_client("openai")
    tester = InvocationTester(llm)

    main_specs = [
        ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted((ROOT / "data" / "discovered_specs").glob("*.json"))
    ]
    lq_base = [json.loads(l) for l in (ROOT / "data" / "lowqual" / "logs" / "baseline.jsonl").open()]
    main_base = [json.loads(l) for l in (ROOT / "data" / "logs" / "baseline.jsonl").open()]

    def held_out_for(name: str, source: list) -> list[str]:
        return [r["test_prompt"] for r in source if r["tool_name"] == name]

    # Structural-defect candidates (B1 low-quality) + one healthy control.
    cases = []
    for p in sorted((ROOT / "data" / "lowqual" / "discovered_specs").glob("*.json")):
        spec = ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        cases.append(("lowqual", spec, held_out_for(spec.name, lq_base)))
    healthy = next(s for s in main_specs if s.name == "get_pdb_structure")
    cases.append(("healthy_control", healthy, held_out_for("get_pdb_structure", main_base)))

    print(f"Phase 7: needs_redesign detection, model={llm.model}")
    print(f"  Judging on HELD-OUT prompts (real distribution), not self-generated")
    print("-" * 100)
    print(f"{'Tool':<40}{'kind':<18}{'held-out':>9}  flagged?  reason")
    print("-" * 100)

    report = []
    for kind, spec, held in cases:
        competing = [s for s in main_specs if s.name != spec.name]
        if not held:
            continue
        held_acc = sum(1 for r in tester.test_batch(spec, held, competing)
                       if r.failure_type == "correct") / len(held)
        loop = OptimizerLoop(llm=llm, log_path=ROOT / "data" / "logs" / "redesign.jsonl",
                             n_prompts=5, max_iterations=3,
                             do_no_harm=True, detect_redesign=True)
        result = loop.optimize(spec, competing_specs=competing, eval_prompts=held)
        flagged = result.needs_redesign
        mark = "YES" if flagged else "no"
        print(f"{spec.name:<40}{kind:<18}{held_acc:>8.0%}  {mark:<8}  {result.terminated_reason[:46]}")
        report.append({
            "tool_name": spec.name,
            "kind": kind,
            "held_out_accuracy": held_acc,
            "needs_redesign": flagged,
            "terminated_reason": result.terminated_reason,
        })

    structural = [r for r in report if r["kind"] == "lowqual"]
    flagged = [r for r in structural if r["needs_redesign"]]
    controls = [r for r in report if r["kind"] == "healthy_control"]
    false_pos = [r for r in controls if r["needs_redesign"]]

    print("-" * 100)
    print(f"Structural defects flagged: {len(flagged)}/{len(structural)}")
    print(f"Healthy controls false-flagged: {len(false_pos)}/{len(controls)}")

    (ROOT / "data" / "redesign_detection.json").write_text(
        json.dumps({
            "structural_flagged": f"{len(flagged)}/{len(structural)}",
            "control_false_positives": f"{len(false_pos)}/{len(controls)}",
            "per_tool": report,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWritten: data/redesign_detection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
