"""Run the Discoverer on all NL tool descriptions and save results.

Reads tool_descriptions.json (a flat list of description strings — no ids,
no metadata) and runs the Discoverer on each. The Discoverer's generated
spec.name becomes the canonical identifier used everywhere downstream:
file names, baseline keys, optimization tracking, degradation reps.

Outputs:
    data/discovered_specs/{spec.name}.json     — generated ToolSpec
    data/discovered_stubs/{spec.name}.py       — generated Python stub
    data/discovery_manifest.json               — index → description → spec.name
    data/discovery_report.json                 — per-tool validation summary
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


def main() -> int:
    raw = json.loads((ROOT / "data" / "tool_descriptions.json").read_text(encoding="utf-8"))
    descs: list[str] = raw["tools"]
    seeds = load_seed_templates()

    out_specs = ROOT / "data" / "discovered_specs"
    out_stubs = ROOT / "data" / "discovered_stubs"
    out_specs.mkdir(parents=True, exist_ok=True)
    out_stubs.mkdir(parents=True, exist_ok=True)

    llm = get_client("openai")
    discoverer = Discoverer(llm=llm, seed_templates=seeds)

    manifest: list[dict] = []
    report: list[dict] = []
    used_names: set[str] = set()

    print(f"Running Discoverer on {len(descs)} tool descriptions using {llm.__class__.__name__} / {llm.model}")
    print("-" * 80)

    t0 = time.time()
    for i, desc in enumerate(descs, start=1):
        short = desc[:55] + ("..." if len(desc) > 55 else "")
        print(f"[{i:2d}/{len(descs)}] {short:60s} ", end="", flush=True)
        try:
            result = discoverer.discover(desc)
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            manifest.append({"index": i, "description": desc, "spec_name": None, "error": str(e)})
            report.append({"index": i, "description": desc, "status": "error",
                           "error": f"{type(e).__name__}: {e}"})
            continue

        # Handle (rare) name collision by appending index suffix
        spec_name = result.spec.name
        if spec_name in used_names:
            spec_name = f"{spec_name}_{i:02d}"
            print(f"(collision → renamed to {spec_name}) ", end="")
            # We can't change result.spec.name (it's immutable), but we can save under new file name
        used_names.add(spec_name)

        # Persist outputs
        (out_specs / f"{spec_name}.json").write_text(
            result.spec.model_dump_json(indent=2), encoding="utf-8"
        )
        (out_stubs / f"{spec_name}.py").write_text(result.stub_source, encoding="utf-8")

        status = "OK" if result.is_valid else f"WARN({len(result.validation_errors)})"
        print(f"-> {spec_name:42s} [{status}]")

        manifest.append({
            "index": i,
            "description": desc,
            "spec_name": spec_name,
            "num_params": len(result.spec.parameters),
        })
        report.append({
            "index": i,
            "description": desc,
            "spec_name": spec_name,
            "status": "valid" if result.is_valid else "warn",
            "validation_errors": [str(e) for e in result.validation_errors],
            "seeds_used": result.seeds_used,
            "num_params": len(result.spec.parameters),
        })

    elapsed = time.time() - t0
    print("-" * 80)
    print(f"Done in {elapsed:.1f}s")
    valid = sum(1 for r in report if r.get('status') == 'valid')
    errors = sum(1 for r in report if r.get('status') == 'error')
    print(f"  Valid:  {valid}/{len(descs)}")
    print(f"  Errors: {errors}")

    (ROOT / "data" / "discovery_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "data" / "discovery_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nSaved: data/discovery_manifest.json, data/discovery_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
