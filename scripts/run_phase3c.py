"""Phase 3-C: Adversarial pair disambiguation experiment (auto-discovered pairs).

Improvement over the original design: instead of pre-declaring which tool pair
is adversarial, this script DISCOVERS adversarial pairs from baseline confusion
data. This makes the experiment more rigorous — the detection is blind to
designer intent and lets data surface true adversarial relationships.

Experiment flow:
    1. Load baseline.jsonl, compute confusion matrix across all tool pairs
    2. Identify top-N most-confused pairs (auto-discovered adversarial pairs)
    3. For each discovered pair, run three experiments:
       A. Baseline disambiguation (naturally generated prompts)
       B. Confounding prompts (LLM-generated queries that blend both tools' semantics)
       C. Spec ablation (remove "Do NOT use" clauses, re-test confounding)

Output:
    data/logs/phase3c.jsonl       - all test records
    data/phase3c_results.json     - summary with discovered pairs
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.llm_client import get_client  # noqa: E402
from src.optimizer import InvocationTester  # noqa: E402
from src.schema import ToolSpec  # noqa: E402


# ------------------------------------------------------------------------------
# Auto-discover adversarial pairs from baseline confusion data
# ------------------------------------------------------------------------------

def discover_adversarial_pairs(
    baseline_jsonl: Path, top_n: int = 2, min_confusion: int = 1
) -> list[tuple[tuple[str, str], int]]:
    """Find tool pairs with highest mutual confusion in baseline."""
    confusion: Counter = Counter()
    for line in baseline_jsonl.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        target = rec.get("tool_name")
        actual_call = rec.get("actual_call")
        if not actual_call:
            continue
        actual = actual_call.get("name")
        if not actual or target == actual:
            continue
        # symmetric pair key
        pair = tuple(sorted([target, actual]))
        confusion[pair] += 1

    sorted_pairs = [(p, c) for p, c in confusion.most_common() if c >= min_confusion]
    return sorted_pairs[:top_n]


# ------------------------------------------------------------------------------
# Confounding prompt generation (tailored to each discovered pair)
# ------------------------------------------------------------------------------

CONFOUNDING_SYSTEM_TMPL = """\
You are a biomedical researcher writing realistic queries.
Generate AMBIGUOUS queries that could plausibly be answered by EITHER of these two tools:

Tool A: {desc_a}
Tool B: {desc_b}

Write queries that BLEND the semantics of both tools so it is genuinely unclear
which the user wants. Do NOT use exact phrases from either description.

Return a JSON array of exactly {n} short, natural-language queries (strings).
Output ONLY the JSON array, no prose.
"""


def generate_confounding_prompts(
    llm, spec_a: ToolSpec, spec_b: ToolSpec, n: int = 5
) -> list[str]:
    system = CONFOUNDING_SYSTEM_TMPL.format(
        desc_a=spec_a.description, desc_b=spec_b.description, n=n,
    )
    prompt = f"Generate {n} ambiguous queries that blend Tool A and Tool B."
    raw = llm.complete(prompt, system=system, temperature=0.7, max_tokens=512)
    text = raw.text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


# ------------------------------------------------------------------------------
# Ablation: strip "Do NOT use" sentences
# ------------------------------------------------------------------------------

def strip_do_not_use(spec: ToolSpec) -> ToolSpec:
    sentences = spec.description.split(". ")
    kept = [s for s in sentences if "Do NOT" not in s and "do not" not in s.lower()]
    new_desc = ". ".join(kept).strip()
    if not new_desc.endswith("."):
        new_desc += "."
    return spec.set_field("description", new_desc)


# ------------------------------------------------------------------------------
# Measurement
# ------------------------------------------------------------------------------

def measure(
    tester: InvocationTester,
    spec: ToolSpec,
    prompts: list[str],
    competing: list[ToolSpec],
    label: str,
) -> dict:
    results = tester.test_batch(spec, prompts, competing)
    correct = sum(1 for r in results if r.failure_type == "correct")
    actual_calls = [r.actual_call for r in results]
    confused = sum(
        1 for c in actual_calls
        if c and c.get("name") and c["name"] != spec.name and
        any(s.name == c["name"] for s in competing)
    )
    picks = [c["name"] if c else None for c in actual_calls]
    print(f"    {label:<55} correct={correct}/{len(prompts)}  "
          f"confused_with_partner={confused}")
    return {
        "label": label,
        "n": len(prompts),
        "correct": correct,
        "confused_with_partner": confused,
        "picks": picks,
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> int:
    specs_dir = ROOT / "data" / "discovered_specs"
    baseline_path = ROOT / "data" / "logs" / "baseline.jsonl"
    prompts_dir = ROOT / "data" / "test_prompts"

    if not baseline_path.exists():
        print("ERROR: baseline.jsonl not found. Run run_baseline.py first.")
        return 1

    # ---- 1. Auto-discover adversarial pairs ----
    pairs = discover_adversarial_pairs(baseline_path, top_n=3, min_confusion=1)
    print("=" * 80)
    print("AUTO-DISCOVERED ADVERSARIAL PAIRS (from baseline confusion data)")
    print("=" * 80)
    if not pairs:
        print("No tool pairs with cross-confusion found in baseline.")
        print("Falling back: pick the two most semantically similar tools by name.")
        # Fallback: use rank_drug_compounds vs rank_therapeutic_targets if both exist
        all_files = {p.stem for p in specs_dir.glob("*.json")}
        if "rank_drug_compounds" in all_files and "rank_therapeutic_targets" in all_files:
            pairs = [(("rank_drug_compounds", "rank_therapeutic_targets"), 0)]
            print(f"  Fallback pair: rank_drug_compounds <-> rank_therapeutic_targets")
        else:
            print("ERROR: no fallback pair available. Aborting.")
            return 1
    else:
        for (a, b), c in pairs:
            print(f"  ({a}, {b}) — confused {c} times in baseline")

    # ---- 2. Run experiments on each discovered pair ----
    llm = get_client("openai")
    tester = InvocationTester(llm)

    all_results: dict = {"discovered_pairs": [], "experiments": {}}

    for (name_a, name_b), confusion_count in pairs:
        path_a = specs_dir / f"{name_a}.json"
        path_b = specs_dir / f"{name_b}.json"
        if not (path_a.exists() and path_b.exists()):
            print(f"  SKIP: spec files missing for ({name_a}, {name_b})")
            continue

        spec_a = ToolSpec.model_validate_json(path_a.read_text(encoding="utf-8"))
        spec_b = ToolSpec.model_validate_json(path_b.read_text(encoding="utf-8"))

        all_results["discovered_pairs"].append({
            "pair": [name_a, name_b],
            "baseline_confusion": confusion_count,
            "description_a": spec_a.description,
            "description_b": spec_b.description,
        })

        pair_key = f"{name_a}__vs__{name_b}"
        all_results["experiments"][pair_key] = {}

        print(f"\n{'='*80}")
        print(f"PAIR: {name_a}  <->  {name_b}")
        print(f"{'='*80}")

        # ---- Experiment A: baseline prompts ----
        print("\nEXPERIMENT A: Baseline disambiguation (naturally generated prompts)")
        prompts_a = json.loads((prompts_dir / f"{name_a}.json").read_text(encoding="utf-8"))
        prompts_b = json.loads((prompts_dir / f"{name_b}.json").read_text(encoding="utf-8"))
        r_aa = measure(tester, spec_a, prompts_a, [spec_b], f"[A] {name_a} spec, its prompts")
        r_bb = measure(tester, spec_b, prompts_b, [spec_a], f"[A] {name_b} spec, its prompts")
        all_results["experiments"][pair_key]["A_target_a"] = r_aa
        all_results["experiments"][pair_key]["A_target_b"] = r_bb

        # ---- Experiment B: confounding prompts ----
        print("\nEXPERIMENT B: Confounding prompts (blend semantics of both tools)")
        confounding = generate_confounding_prompts(llm, spec_a, spec_b, n=5)
        print(f"  Generated {len(confounding)} confounding prompts:")
        for j, p in enumerate(confounding, 1):
            print(f"    {j}. {p}")
        r_ba = measure(tester, spec_a, confounding, [spec_b], f"[B] {name_a} spec, confounding")
        r_bb_conf = measure(tester, spec_b, confounding, [spec_a], f"[B] {name_b} spec, confounding")
        all_results["experiments"][pair_key]["B_target_a"] = r_ba
        all_results["experiments"][pair_key]["B_target_b"] = r_bb_conf
        all_results["experiments"][pair_key]["confounding_prompts"] = confounding

        # ---- Experiment C: ablation ----
        print("\nEXPERIMENT C: Ablation - remove 'Do NOT use' clauses, retest confounding")
        spec_a_ab = strip_do_not_use(spec_a)
        spec_b_ab = strip_do_not_use(spec_b)
        r_ca = measure(tester, spec_a_ab, confounding, [spec_b_ab], f"[C] {name_a} ABLATED, confounding")
        r_cb = measure(tester, spec_b_ab, confounding, [spec_a_ab], f"[C] {name_b} ABLATED, confounding")
        all_results["experiments"][pair_key]["C_target_a"] = r_ca
        all_results["experiments"][pair_key]["C_target_b"] = r_cb

    # ---- 3. Save results ----
    (ROOT / "data" / "phase3c_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: data/phase3c_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
