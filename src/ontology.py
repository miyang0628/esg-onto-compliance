"""
ontology.py — Qualified Ontology Anchoring (QOA): triple -> quad -> pentuple.

Motivation (reviewer 2, M3 / originality)
-----------------------------------------
The submitted "ontology-anchored" ranker (B3) only added a score bonus for
matching two metadata fields (jurisdiction + ESG topic). A reviewer rightly
noted this is a *metadata boost*, not a methodological contribution, and that
a conventional metadata baseline must be compared against.

We keep that exact B3 as the **triple** level (the required metadata baseline)
and extend the ontology from binary triples (s, p, o) to **n-ary qualified
statements** that encode the real compliance semantics the flat triple cannot:

  * triple   (s, p, o)                         e.g. (Case, governedBy, Regulation)
  * quad     (s, p, o | validTime)             + the regulation must be IN FORCE
                                                 at the violation/publication time
  * pentuple (s, p, o | validTime, severity)   + the sanction severity / ESG
                                                 subtopic must be consistent

Each statement is reified (RDF-star style) into a statement node carrying
qualifiers, so the graph is a true n-ary knowledge graph rather than a bag of
triples. The retriever (src/retrieval.py) scores a candidate by a weighted sum
of per-qualifier alignment, with the weight vector tuned on a held-out split.
This yields an ablatable method (triple -> quad -> pentuple) whose gain over
the metadata baseline is attributable to the *qualifiers*, not to the fact that
an ontology exists.

Everything here is derived 1:1 from source fields (defensible, no hand curation)
and is deterministic.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from config import (
    SEVERITY_MAP, SEVERITY_ORDER, QOA_TEMPORAL_TAU,
    SANCTION_TOKENS, ACQUIT_TOKENS,
)

# --------------------------------------------------------------------------
# 0. field normalisation helpers
# --------------------------------------------------------------------------
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def extract_year(*values) -> Optional[int]:
    """First 4-digit year found across the given date-ish strings, else None."""
    for v in values:
        if v is None:
            continue
        m = _YEAR_RE.search(str(v))
        if m:
            y = int(m.group(0))
            if 1950 <= y <= 2100:
                return y
    return None


def severity_bucket(sanction_type: str, sanction_level: str) -> str:
    """Map a case result to an ordinal severity bucket (pentuple qualifier)."""
    s = (str(sanction_type or "") + " " + str(sanction_level or ""))
    if any(tok in s for tok in ACQUIT_TOKENS) and not any(tok in s for tok in SANCTION_TOKENS):
        return "NONE"
    best = "NONE"
    for tok, bucket in SEVERITY_MAP.items():
        if tok in s and SEVERITY_ORDER.index(bucket) > SEVERITY_ORDER.index(best):
            best = bucket
    # a generic sanction token with no severity map still counts as at least MINOR
    if best == "NONE" and any(tok in s for tok in SANCTION_TOKENS):
        best = "MINOR"
    return best


def esg_subtopic(pollutant_category: str, esg: str) -> str:
    """A finer topic qualifier: pollutant_category when present, else ESG type."""
    pc = str(pollutant_category or "").strip()
    return pc if pc else str(esg or "").strip()


# --------------------------------------------------------------------------
# 1. document anchor profiles  (one row per document)
# --------------------------------------------------------------------------
ANCHOR_COLS = ["id", "doc_id", "country", "region", "esg", "subtopic",
               "year", "doctype", "severity", "has_reg_ref"]


def build_anchor_profiles(source: pd.DataFrame) -> pd.DataFrame:
    """Derive the per-document qualified anchor profile from source metadata.

    Parameters
    ----------
    source : the interim source DataFrame from nb01 (data/interim/source.parquet)

    Returns
    -------
    DataFrame[ANCHOR_COLS]
    """
    df = source.copy()
    years = [
        extract_year(a, b, c)
        for a, b, c in zip(df.get("effective_date", ""),
                           df.get("violation_date", ""),
                           df.get("published_at", ""))
    ]
    out = pd.DataFrame({
        "id":        df["id"],
        "doc_id":    df.get("doc_id", df["id"]),
        "country":   df["country"].fillna("").astype(str),
        "region":    df.get("region", "").fillna("").astype(str),
        "esg":       df["esg"].fillna("").astype(str),
        "subtopic":  [esg_subtopic(pc, e)
                      for pc, e in zip(df.get("pollutant_category", ""), df.get("esg", ""))],
        "year":      years,
        "doctype":   df.get("document_type", df.get("category", "")).fillna("").astype(str),
        "severity":  [severity_bucket(t, l)
                      for t, l in zip(df.get("sanction_type", ""), df.get("sanction_level", ""))],
        "has_reg_ref": (df.get("reg_reference", "").fillna("").astype(str).str.strip() != ""),
    })
    return out[ANCHOR_COLS].reset_index(drop=True)


# --------------------------------------------------------------------------
# 2. reified qualified statements  (the n-ary KG itself)
# --------------------------------------------------------------------------
def build_qualified_statements(source: pd.DataFrame) -> pd.DataFrame:
    """Emit reified statements with qualifiers.

    We materialise the compliance-relevant relation `governedBy`
    (Case/Document -> Jurisdiction+Topic context) as a statement node carrying
    the validTime (year) and severity qualifiers. The point is not the specific
    triple but that each statement is *n-ary*: it is (s,p,o) PLUS qualifiers,
    which the triple store cannot represent without reification.

    Returns
    -------
    DataFrame[stmt_id, subject, predicate, object, q_jurisdiction, q_topic,
              q_validtime, q_severity, q_doctype, arity]
    """
    anc = build_anchor_profiles(source)
    rows = []
    for i, r in anc.iterrows():
        juris = f"{r['country']}/{r['region']}".strip("/")
        rows.append({
            "stmt_id":   f"S{i:07d}",
            "subject":   r["id"],
            "predicate": "hasComplianceContext",
            "object":    f"CTX::{juris}::{r['esg']}",
            "q_jurisdiction": juris,
            "q_topic":        r["subtopic"] or r["esg"],
            "q_validtime":    r["year"],
            "q_severity":     r["severity"],
            "q_doctype":      r["doctype"],
        })
    stmts = pd.DataFrame(rows)
    # arity = 3 (triple) + number of non-empty qualifiers present
    qcols = ["q_jurisdiction", "q_topic", "q_validtime", "q_severity", "q_doctype"]

    def _arity(row):
        present = sum(1 for c in qcols if (row[c] not in ("", None) and not (isinstance(row[c], float) and np.isnan(row[c]))))
        return 3 + present
    stmts["arity"] = stmts.apply(_arity, axis=1)
    return stmts


def statement_summary(stmts: pd.DataFrame) -> pd.DataFrame:
    """Counts of statements by effective arity (triple/quad/pentuple+)."""
    vc = stmts["arity"].value_counts().sort_index()
    rows = [{"arity": int(a), "count": int(c)} for a, c in vc.items()]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. per-qualifier alignment functions  (used by the QOA reranker)
# --------------------------------------------------------------------------
def match_jurisdiction(q: dict, a: dict) -> float:
    """1.0 same country+region, 0.6 same country, 0 otherwise. Unknown -> 0."""
    qc, ac = q.get("country"), a.get("country")
    if not qc or not ac:
        return 0.0
    if qc != ac:
        return 0.0
    qr, ar = (q.get("region") or ""), (a.get("region") or "")
    return 1.0 if (qr and qr == ar) else 0.6


def match_topic(q: dict, a: dict) -> float:
    """Exact subtopic 1.0; same ESG letter 0.5; else 0. Unknown -> 0."""
    qs, as_ = (q.get("subtopic") or ""), (a.get("subtopic") or "")
    if qs and qs == as_:
        return 1.0
    qe, ae = (q.get("esg") or ""), (a.get("esg") or "")
    if qe and qe == ae:
        return 0.5
    return 0.0


def match_temporal(q: dict, a: dict, tau: float = QOA_TEMPORAL_TAU) -> float:
    """exp(-|dy|/tau) year proximity. Unknown on either side -> 0 (neutral)."""
    qy, ay = q.get("year"), a.get("year")
    if qy is None or ay is None:
        return 0.0
    try:
        return float(np.exp(-abs(int(qy) - int(ay)) / float(tau)))
    except (TypeError, ValueError):
        return 0.0


def match_severity(q: dict, a: dict) -> float:
    """Ordinal severity closeness in [0,1]. Unknown -> 0."""
    qs, as_ = q.get("severity"), a.get("severity")
    if not qs or not as_ or qs not in SEVERITY_ORDER or as_ not in SEVERITY_ORDER:
        return 0.0
    d = abs(SEVERITY_ORDER.index(qs) - SEVERITY_ORDER.index(as_))
    return 1.0 - d / (len(SEVERITY_ORDER) - 1)


def match_doctype(q: dict, a: dict) -> float:
    """1.0 if same document type, else 0. Unknown -> 0."""
    qd, ad = (q.get("doctype") or ""), (a.get("doctype") or "")
    return 1.0 if (qd and qd == ad) else 0.0


MATCHERS = {
    "jurisdiction": match_jurisdiction,
    "topic":        match_topic,
    "temporal":     match_temporal,
    "severity":     match_severity,
    "doctype":      match_doctype,
}


def alignment_vector(q_profile: dict, a_profile: dict, dims: list[str]) -> np.ndarray:
    """Per-dimension alignment scores for the requested qualifier dims."""
    return np.array([MATCHERS[d](q_profile, a_profile) for d in dims], dtype=float)
