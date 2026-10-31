"""
attributes.py — predict a query's anchor attributes FROM THE QUERY TEXT.

Reviewer 2 (M3): "explain how query attributes were obtained. If they come from
the gold document, add a condition that predicts them from query text."

We therefore support two attribute sources for the QOA reranker:

  * ORACLE     : attributes read from the gold document's anchor profile.
                 This is an upper bound and deliberately leaks; we report it only
                 to quantify the headroom.
  * PREDICTED  : attributes inferred from the query string alone, with NO access
                 to the gold document. This is the honest operating condition.

The predictor is intentionally light and reproducible:
  - jurisdiction (country): rule-based keyword detection (country names in ko/en/vi/ms)
  - topic / subtopic / year: 1-NN over TRAINING (query -> gold anchor) pairs using
    the same char n-gram TF-IDF as the retriever; year also tried via regex.
No external model or internet required.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ontology import extract_year

_COUNTRY_KEYWORDS = {
    "Vietnam":  ["베트남", "vietnam", "viet nam", "việt", "viet"],
    "Malaysia": ["말레이시아", "말레이", "malaysia", "malay"],
}


def detect_country(text: str) -> str:
    t = str(text or "").lower()
    for country, kws in _COUNTRY_KEYWORDS.items():
        if any(kw in t for kw in kws):
            return country
    return ""


class QueryAttributePredictor:
    """Predict {country, esg, subtopic, year} for a query from its text only."""

    def __init__(self, ngram=(3, 4), max_features=100_000, min_df=2):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram,
                                   min_df=min_df, max_features=max_features,
                                   dtype=np.float32)
        self._fit = False

    def fit(self, train_queries: list[str], train_anchors: list[dict]):
        """train_anchors: gold-doc anchor dicts aligned to train_queries."""
        self.train_mat = self.vec.fit_transform([str(q) for q in train_queries])
        self.train_anchors = list(train_anchors)
        self._fit = True
        return self

    def predict_one(self, query: str) -> dict:
        assert self._fit, "call fit() first"
        q = str(query or "")
        # 1-NN over training queries
        qv = self.vec.transform([q])
        sims = cosine_similarity(qv, self.train_mat)[0]
        nn = int(np.argmax(sims)) if len(sims) else 0
        nn_anchor = self.train_anchors[nn] if self.train_anchors else {}
        # jurisdiction: prefer explicit keyword in the query, else neighbour
        country = detect_country(q) or nn_anchor.get("country", "")
        # year: regex on the query first, else neighbour
        year = extract_year(q)
        if year is None:
            year = nn_anchor.get("year")
        return {
            "country":  country,
            "region":   "",                      # region rarely stated in a query
            "esg":      nn_anchor.get("esg", ""),
            "subtopic": nn_anchor.get("subtopic", ""),
            "year":     year,
            "doctype":  nn_anchor.get("doctype", ""),
            "severity": nn_anchor.get("severity", ""),
        }

    def predict(self, queries: list[str]) -> list[dict]:
        return [self.predict_one(q) for q in queries]


def anchor_row_to_dict(row) -> dict:
    """Convert an anchor-profile row (Series/dict) to the matcher dict."""
    g = row.get if isinstance(row, dict) else (lambda k, d=None: row[k] if k in row else d)
    return {
        "country":  g("country", ""),
        "region":   g("region", ""),
        "esg":      g("esg", ""),
        "subtopic": g("subtopic", ""),
        "year":     g("year", None),
        "doctype":  g("doctype", ""),
        "severity": g("severity", ""),
    }
