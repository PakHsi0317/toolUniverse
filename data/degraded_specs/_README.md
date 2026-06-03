# Degraded Spec Copies (wrong_type) — for presentation

These are the EXACT broken specs the degradation experiment injected,
regenerated deterministically. Each shows what a single type-flip does.

| Tool | param[0].type before → after | clean | degraded | recovered |
|------|------------------------------|-------|----------|-----------|
| `compute_tanimoto_similarity` | string → **integer** | 60% | **20%** | **80%** |
| `convert_gene_symbol_to_ensembl_id` | string → **integer** | 100% | **0%** | **100%** |
| `get_drug_side_effects` | string → **integer** | 100% | **0%** | **100%** |
| `get_protein_tissue_expression` | string → **integer** | 100% | **0%** | **100%** |
| `predict_admet_properties` | string → **integer** | 100% | **0%** | **100%** |
| `rank_therapeutic_targets` | string → **integer** | 100% | **0%** | **100%** |
| `search_biomedical_articles` | string → **integer** | 100% | **0%** | **100%** |

## How to read these in the talk

For any tool, show three states side by side:

```bash
# 1. CLEAN (original, correct type)
cat data/discovered_specs/convert_gene_symbol_to_ensembl_id.json

# 2. DEGRADED (this folder — type flipped, accuracy crashed to 0%)
cat data/degraded_specs/convert_gene_symbol_to_ensembl_id.wrong_type.json

# 3. RECOVERED (optimizer fixed it back — note: optimized_specs holds the
#    natural-data optimization run; the degradation recovery is in the logs)
```

The single line that differs between #1 and #2 is `parameters[0].type`.
That one-character class of change is what crashes 7/11 tools — and what
the Optimizer recovers 7/7.