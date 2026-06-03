"""Export the full 3-state spec trio for ONE concrete degradation example:
convert_gene_symbol_to_ensembl_id under wrong_type.

Re-runs the real pipeline (inject bug -> Optimizer) so the recovered spec is a
genuine optimization product. Writes all three states to
data/example_convert_gene/ for the presentation.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.run_degradation import apply_bug
from src.llm_client import get_client
from src.optimizer import OptimizerLoop
from src.schema import ToolSpec

TOOL = "convert_gene_symbol_to_ensembl_id"


def main() -> int:
    out = ROOT / "data" / "example_convert_gene"
    out.mkdir(parents=True, exist_ok=True)

    specs_dir = ROOT / "data" / "discovered_specs"
    all_specs = [ToolSpec.model_validate_json(p.read_text(encoding="utf-8"))
                 for p in sorted(specs_dir.glob("*.json"))]
    clean = next(s for s in all_specs if s.name == TOOL)
    competing = [s for s in all_specs if s.name != TOOL]

    (out / "1_clean.json").write_text(clean.model_dump_json(indent=2), encoding="utf-8")

    degraded = apply_bug(clean, "wrong_type")
    (out / "2_degraded.json").write_text(degraded.model_dump_json(indent=2), encoding="utf-8")

    llm = get_client("openai")
    loop = OptimizerLoop(llm=llm, log_path=out / "optimize.jsonl",
                         n_prompts=5, max_iterations=3, target_accuracy=1.0, do_no_harm=True)
    result = loop.optimize(degraded, competing_specs=competing)
    recovered = result.final_spec
    (out / "3_recovered.json").write_text(recovered.model_dump_json(indent=2), encoding="utf-8")

    print(f"clean     param[0].type = {clean.parameters[0].type}")
    print(f"degraded  param[0].type = {degraded.parameters[0].type}")
    print(f"recovered param[0].type = {recovered.parameters[0].type}")
    print(f"optimizer ended: {result.terminated_reason}")
    print("\nWritten to data/example_convert_gene/ (1_clean / 2_degraded / 3_recovered)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
