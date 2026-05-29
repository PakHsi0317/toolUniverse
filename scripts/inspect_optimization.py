"""Show the diagnosis trajectory for tools that didn't improve."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import load_records  # noqa: E402


def main() -> int:
    log = ROOT / "data" / "logs" / "optimization.jsonl"
    records = load_records(log)

    tools_of_interest = ["convert_dna_to_rna", "compute_two_sample_ttest_pvalue"]

    for tool in tools_of_interest:
        print(f"\n{'=' * 70}\n=== {tool} ===\n{'=' * 70}")
        tool_records = [r for r in records if r.tool_name == tool]
        failures = [r for r in tool_records if r.failure_type != "correct"]
        correct = [r for r in tool_records if r.failure_type == "correct"]
        print(f"Total records: {len(tool_records)} ({len(correct)} correct, {len(failures)} failures)")

        for f in failures:
            print(f"\n  -- Iter {f.iteration} : {f.failure_type} --")
            print(f"  Prompt:        {f.test_prompt}")
            print(f"  Actual call:   {f.actual_call}")
            print(f"  Blamed field:  {f.blamed_field}")
            if f.root_cause:
                print(f"  Root cause:    {f.root_cause}")
            if f.suggested_rewrite:
                preview = str(f.suggested_rewrite.new_value)[:200]
                print(f"  Suggested:     [{f.suggested_rewrite.field}]")
                print(f"                 new_value: {preview}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
