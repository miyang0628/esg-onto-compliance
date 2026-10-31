# ESG-Onto-Compliance

**Ontology-Anchored Evidence Retrieval and Cross-Lingual Compliance Judgment for ESG Regulation in Low-Resource Languages**

Reproducible code for the study: a trilingual (Korean / English / local) ESG
corpus for Vietnam and Malaysia, a field-derived typed ontology, a transparent
compliance-labeling rule, and a two-axis evaluation that **separates evidence
retrieval from cross-lingual compliance judgment**.

> Anonymized for double-blind review. Data is **not** redistributed; see
> [Data](#data).

---

## What is new in this revision (R1)

| Area | Added | Notebook |
|---|---|---|
| Cross-lingual retrieval | Language-controlled (ko/en/local) + **multilingual dense** (mE5, BGE-M3). The sparse near-zero en/local was a script artifact; BGE-M3 largely closes the gap. | `07` |
| Methodological contribution | **Qualified Ontology Anchoring (QOA)**: triple → quad (+validTime) → pentuple (+severity/doctype), held-out α, **oracle vs predicted** query attributes. | `06`, `08` |
| Bottleneck (causal) | Same cases, same judge, evidence = gold / flat-retrieved / anchored (0.580 → 0.311 → 0.505). | `09` |
| Selective prediction | Risk–coverage, AURC, ECE, **TOST equivalence** test, OTHER-vs-abstention separation. | `10` |
| Validity | Verdict-leakage rate (79.8%), label audit sheet, OTHER-bias, count reconciliation. | `11` |

---

## Repository layout

```
notebooks/   01..11  pipeline (single self-contained cells; see run order)
src/         config, data_loader, labeling, metrics, ontology, attributes,
             retrieval, llm   (shared, tested modules)
data/
  raw/       <- put the AI Hub source/label zips here (auto-discovered)
  interim/   parsed parquet (produced by nb01)
  processed/ ontology, gold labels, predictions
results/
  tables/    table01..table11 (CSV)
  figures/   grayscale PNG + PDF (600 dpi)
artifacts/
  llm_cache/ cached LLM judgments (weak tier shipped)
```

> The single-cell notebooks embed the `src/` modules, so each notebook runs on
> its own; `src/` is also provided for direct import/testing.

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas scikit-learn matplotlib seaborn pyarrow openai python-dotenv
pip install "sentence-transformers>=2.7"      # dense retrieval (nb07, nb08)
cp .env.example .env                           # then add your OPENAI_API_KEY
```

`.env`:

```
OPENAI_API_KEY=sk-...
LLM_MODEL_WEAK=gpt-4o-mini
LLM_MODEL_MID=gpt-5.4-mini
LLM_MODEL_STRONG=gpt-5.4
```

---

## Data

Place the AI Hub **Vietnam/Malaysia trilingual ESG** archives in `data/raw/`.
Filenames are **auto-discovered** and parsed tolerantly, so the real spacing
variants all work, e.g.:

```
TS_1. 베트남_01. 규제.zip      TS_1_베트남_02. 입법 동향.zip
TS_1._베트남_01_규제.zip       TS_2. 말레이시아_03. 판례.zip
02.라벨링데이터.zip            (label archive)
```

Category/country values are normalized to English after parsing. The trilingual
fields are the source dataset's own parallel content (not machine-translated by
us). The corpus is cited in the paper and **not redistributed**.

---

## Run order

| # | Notebook | Needs | Key outputs |
|---|---|---|---|
| 01 | `01_data_parsing` | raw zips | `data/interim/*.parquet`, `table01*`, `fig01` |
| 02 | `02_ontology_build` | interim | `nodes/edges`, `table02*`, `fig02` |
| 03 | `03_compliance_labeling` | interim | `compliance_gold.parquet`, `table03*`, `fig03` |
| 04 | `04_graphrag_retrieval` | interim + raw | `table04*` (B2 baseline) |
| 05 | `05_crosslingual_eval` | processed + **API** | `crosslingual_pred_*`, `table05*` |
| 06 | `06_ontology_qualified` | interim | `anchors`, qualified statements, `table06*` |
| 07 | `07_retrieval_crosslingual` | interim + raw (+ dense) | `table07*`, cross-lingual + dense |
| 08 | `08_retrieval_methods` | interim + raw (+ dense) | `table08*` (QOA ablation, dense) |
| 09 | `09_judgment_evidence` | processed + raw + **API** | `table09_bottleneck` |
| 10 | `10_selective_prediction` | nb05/nb09 preds | `table10*` (risk–coverage, equivalence) |
| 11 | `11_validity_consistency` | interim + processed + raw | `table11*` (leakage, audit, bias) |

**GPU / dense note.** `nb07`/`nb08` default to **mE5** on CPU; BGE-M3 is heavy,
so it is **off by default**. On a GPU set `RUN_BGE = True` in the dense cell.
The dense step caps `max_seq_length=384` and encodes only the document head, and
saves after each language, so it finishes quickly and never loses progress.

**Reproducibility.** A single global seed is fixed; archive entries are read in
sorted order; LLM judgments are cached by input hash (the weak-tier cache is
shipped, so weak-tier numbers reproduce at no cost). mid/strong tiers and any
new prompt require live API calls.

---

## Figures

All figures are grayscale, 16:9, opaque background, 600 dpi, saved as PNG + PDF,
with captions set in LaTeX (not embedded in the image). Series are distinguished
by gray value plus hatch (bars) or dash (lines) for color-blind / grayscale
printing.

---

## Citation & license

- **Data:** AI Hub Vietnam/Malaysia trilingual ESG dataset (cited in the paper);
  not redistributed here.
- **Code:** released for reproducibility (license: TBD upon de-anonymization).
- Please cite the paper when using this code or the ontology/labeling rule.
