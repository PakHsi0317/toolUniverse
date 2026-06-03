"""Export degraded (bug-injected) spec copies for the presentation.

The degradation experiment (run_degradation.py) injects bugs into in-memory
spec copies and discards them. For the talk it's useful to SHOW the actual
broken spec next to the clean one and the optimizer-recovered one.

This script regenerates the `wrong_type` degraded specs deterministically
(reusing run_degradation's own apply_bug) for the 7 tools that wrong_type
actually damaged, and writes them to data/degraded_specs/ alongside a
three-state comparison (clean / degraded / recovered accuracy).

Output:
    data/degraded_specs/{tool}.wrong_type.json   - the broken spec
    data/degraded_specs/_README.md               - what changed + accuracies

No LLM calls, no cost — pure regeneration from existing files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_degradation import apply_bug  # reuse the EXACT injection logic
from src.schema import ToolSpec  # noqa: E402


def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    out_dir = ROOT / "data" / "degraded_specs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The 7 tools wrong_type actually damaged (from degradation_results.json)
    deg = json.loads((ROOT / "data" / "degradation_results.json").read_text())
    damaged = {
        r["tool_name"]: r
        for r in deg
        if r["scenario"] == "wrong_type_only" and r["damage"] > 0
    }

    all_specs = {
        p.stem: ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(specs_dir.glob("*.json"))
    }

    readme_rows = []
    print(f"Exporting wrong_type degraded specs for {len(damaged)} damaged tools\n")
    print(f"{'Tool':<42} {'field flipped':<28} {'clean→deg→rec'}")
    print("-" * 96)

    for name, result in damaged.items():
        clean = all_specs[name]
        degraded = apply_bug(clean, "wrong_type")  # deterministic, same as experiment

        # what exactly changed
        before_type = clean.parameters[0].type
        after_type = degraded.parameters[0].type
        field = f"parameters[0].type ({before_type}→{after_type})"

        out_path = out_dir / f"{name}.wrong_type.json"
        out_path.write_text(degraded.model_dump_json(indent=2), encoding="utf-8")

        acc = f"{result['clean_acc']:.0%}→{result['degraded_acc']:.0%}→{result['recovered_acc']:.0%}"
        print(f"{name:<42} {field:<28} {acc}")
        readme_rows.append((name, before_type, after_type, result))

    # Write a human-readable comparison README
    lines = [
        "# Degraded Spec Copies (wrong_type) — for presentation",
        "",
        "These are the EXACT broken specs the degradation experiment injected,",
        "regenerated deterministically. Each shows what a single type-flip does.",
        "",
        "| Tool | param[0].type before → after | clean | degraded | recovered |",
        "|------|------------------------------|-------|----------|-----------|",
    ]
    for name, bt, at, r in readme_rows:
        lines.append(
            f"| `{name}` | {bt} → **{at}** | {r['clean_acc']:.0%} | "
            f"**{r['degraded_acc']:.0%}** | **{r['recovered_acc']:.0%}** |"
        )
    lines += [
        "",
        "## How to read these in the talk",
        "",
        "For any tool, show three states side by side:",
        "",
        "```bash",
        "# 1. CLEAN (original, correct type)",
        "cat data/discovered_specs/convert_gene_symbol_to_ensembl_id.json",
        "",
        "# 2. DEGRADED (this folder — type flipped, accuracy crashed to 0%)",
        "cat data/degraded_specs/convert_gene_symbol_to_ensembl_id.wrong_type.json",
        "",
        "# 3. RECOVERED (optimizer fixed it back — note: optimized_specs holds the",
        "#    natural-data optimization run; the degradation recovery is in the logs)",
        "```",
        "",
        "The single line that differs between #1 and #2 is `parameters[0].type`.",
        "That one-character class of change is what crashes 7/11 tools — and what",
        "the Optimizer recovers 7/7.",
    ]
    (out_dir / "_README.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nWritten {len(damaged)} degraded specs + _README.md to data/degraded_specs/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
