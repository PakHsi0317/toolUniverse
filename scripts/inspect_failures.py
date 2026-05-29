"""Inspect failures in a FailureRecord jsonl log."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.schema import load_records  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--include-correct", action="store_true")
    args = parser.parse_args()

    records = load_records(args.log_path)
    failures = [r for r in records if r.failure_type != "correct" or args.include_correct]
    print(f"Total records: {len(records)}")
    print(f"Showing:       {len(failures)}\n")

    for i, r in enumerate(failures, 1):
        print(f"--- Record {i} ---")
        print(f"Target tool:   {r.tool_name}")
        print(f"Iteration:     {r.iteration}")
        print(f"Prompt:        {r.test_prompt}")
        print(f"Failure type:  {r.failure_type}")
        print(f"Actual call:   {json.dumps(r.actual_call, ensure_ascii=False, indent=2)}")
        if r.blamed_field:
            print(f"Blamed field:  {r.blamed_field}")
            print(f"Root cause:    {r.root_cause}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
