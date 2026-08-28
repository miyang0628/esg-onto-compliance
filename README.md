# ESG-Onto-Compliance

Ontology-anchored regulatory-compliance mapping for multilingual ESG disclosures
in low-resource languages (Vietnamese and Malay).

> **Anonymous release for double-blind review.**
> This repository is anonymized. It contains no author names, affiliations, or
> identifying information. Please do not attempt to deanonymize the authors.

---

## Overview

This repository reproduces the empirical analysis of an ontology-anchored
pipeline for ESG regulatory-compliance mapping over a trilingual
(Korean / English / local) legal-and-regulatory corpus. The pipeline:

1. parses the source corpus and validates its distribution,
2. builds a typed ontology (class instances + relations) from the raw fields,
3. derives compliance gold labels from legal-case outcome fields,
4. evaluates retrieval/linking with a flat baseline (**B2**) and an
   ontology-anchored re-ranker (**B3**), and
5. evaluates cross-lingual (English vs. local) LLM compliance judgment.

Key empirical findings are reproduced by the notebooks and written to
`results/tables/` and `results/figures/`.

---

## Repository structure

```
esg-onto-compliance/
├── data/
│   ├── raw/          # place the source dataset here (not distributed; see below)
│   ├── interim/      # parsed parquet (produced by notebook 01)
│   └── processed/    # ontology nodes/edges, compliance gold (nb 02-03)
├── notebooks/
│   ├── 01_data_parsing.ipynb          # parse + validate distributions
│   ├── 02_ontology_build.ipynb        # classes, properties, relations
│   ├── 03_compliance_labeling.ipynb   # legal-case -> compliance gold
│   ├── 04_retrieval_b2_b3.ipynb       # B2 flat vs B3 ontology-anchored
│   └── 05_crosslingual_eval.ipynb     # cross-lingual LLM compliance judgment
├── src/
│   ├── config.py         # paths, seed, constants (single source of truth)
│   ├── data_loader.py    # deterministic zip parsing / on-demand content
│   ├── labeling.py       # compliance labeling rule
│   └── metrics.py        # retrieval + cross-lingual metrics
├── results/
│   ├── figures/      # fig01-05 (png + pdf, grayscale, dpi 600)
│   └── tables/       # table01-05 (csv)
├── requirements.txt
└── README.md
```

---

## Data

This study uses a public ESG dataset for Vietnam and Malaysia (regulations,
legislation, legal cases, news, and ESG reports, with parallel
Korean/English/local text plus QA and benchmark labels). The dataset is **not
redistributed here** due to its license; it must be obtained from the original
provider and placed under `data/raw/`.

- Source zips (regulations, legislation, legal cases, news, gov/corp ESG,
  intl ESG — for both countries) and the label zip go directly under
  `data/raw/`.
- Filenames are **auto-discovered** by the loader, so separator/spacing
  variants (e.g. `". "` vs `"_"`) do not need manual renaming.
- Notebook 01 asserts the expected distribution after parsing; if your local
  counts differ, the run stops so you can check for missing files.

Expected counts for the provided split (asserted in notebook 01):

| Field | Value |
|---|---|
| Source records | 40,078 |
| Categories | Regulation 9,614 / Legislation 7,234 / Legal case 2,529 / News 4,764 / Gov·Corp ESG 7,951 / Intl ESG 7,986 |
| Country | Vietnam 20,079 / Malaysia 19,999 |
| ESG | E 26,342 / S 9,104 / G 4,632 |
| QA / Benchmark | 7,208 / 808 |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Notebook 05 (cross-lingual LLM judgment) additionally requires an LLM API key.
Create a `.env` file in the project root (this file is git-ignored and must
**not** be committed):

```
OPENAI_API_KEY=...
LLM_MODEL_WEAK=...
LLM_MODEL_MID=...
LLM_MODEL_STRONG=...
```

Notebooks 01–04 run without any API key or internet access.

---

## Reproducing the results

Run the notebooks in order (each consumes the previous stage's output):

```bash
cd notebooks
for nb in 01_data_parsing 02_ontology_build 03_compliance_labeling \
          04_retrieval_b2_b3 05_crosslingual_eval; do
  jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb" \
    --ExecutePreprocessor.timeout=1800
done
```

Notebooks 01–04 are fully deterministic and require no network. Notebook 05
caches every LLM call under `artifacts/llm_cache/`, so re-runs are free and
reproducible once the cache is populated.

### What each stage produces

| Stage | Main outputs |
|---|---|
| 01 | `data/interim/{source,qa,benchmark}.parquet`, `table01*`, `fig01` |
| 02 | `data/processed/{nodes,edges}.parquet`, `table02*`, `fig02` |
| 03 | `data/processed/compliance_gold.parquet`, `table03*`, `fig03` |
| 04 | `table04*` (B2 vs B3), `fig04` |
| 05 | `data/processed/crosslingual_pred_*.parquet`, `table05*`, `fig05` |

---

## Reproducibility notes

- All paths are computed relative to the project root; no absolute paths.
- Global seed is fixed (42); sampling and hashing are deterministic.
- Zip entries are read in sorted order and frames sorted by id, so row order and
  aggregates are identical across machines/OSes.
- Content text (large, trilingual) is loaded on demand rather than held in the
  main table, keeping memory bounded on a typical machine.
- The B3 anchor uses only the query's own country/ESG signal (never the gold
  label), so it does not leak the answer.
- All figures are grayscale, untitled, dpi 600, saved as both PNG and PDF.

---

## License

Code is released for review under an open-source license (see `LICENSE`).
The dataset is **not** included and is subject to its provider's terms.

---

## Anonymity

This repository is prepared for **double-blind peer review**. It intentionally
omits author names, affiliations, funding, acknowledgments, and any links that
could identify the authors. A non-anonymized version with full attribution will
be released upon acceptance.
