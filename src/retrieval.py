"""
retrieval.py — retrievers + evaluation harness for the revision experiments.

Retrievers
----------
* TfidfRetriever : char n-gram TF-IDF (reproducible, offline) — the B2 baseline.
* DenseRetriever : multilingual sentence embeddings (mE5 / BGE-M3) — the
                   multilingual dense baseline reviewer 2 (M3, Lit) asked for.
                   Needs `sentence-transformers`; degrades gracefully if absent.
* qoa_rerank     : Qualified Ontology Anchoring reranker (src/ontology.py). It
                   reorders a base retriever's top-N pool by a weighted sum of
                   per-qualifier alignment. Weights are tuned on a held-out split.

Levels (ablation): triple (= metadata baseline B3) -> quad -> pentuple.

Everything returns PER-QUERY arrays (hit@k, reciprocal rank) so we can attach
bootstrap confidence intervals to every number (reviewer 2, Results Q4 / M5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (DOC_CHAR_LIMIT, RETRIEVAL_K_LIST, BOOTSTRAP_N, BOOTSTRAP_CI,
                    QOA_ALPHA_GRID, SPLIT_FRACS)
from ontology import alignment_vector
from attributes import anchor_row_to_dict


# ==========================================================================
# corpus construction
# ==========================================================================
def build_corpus(content: pd.DataFrame, anchors: pd.DataFrame, lang: str):
    """Return (doc_ids, doc_text, anchor_by_docid) for a given language.

    content : DataFrame with columns id, doc_id, content_{ko,en,local}
    anchors : anchor-profile DataFrame (ontology.build_anchor_profiles)
    lang    : 'ko' | 'en' | 'local'
    """
    col = f"content_{lang}"
    df = content.sort_values("doc_id").reset_index(drop=True)
    doc_ids = df["doc_id"].to_numpy()
    doc_text = df[col].fillna("").astype(str).str.slice(0, DOC_CHAR_LIMIT).tolist()
    anc = anchors.set_index("doc_id")
    anchor_by_docid = {}
    for did in doc_ids:
        if did in anc.index:
            row = anc.loc[did]
            if isinstance(row, pd.DataFrame):      # dup doc_id -> take first
                row = row.iloc[0]
            anchor_by_docid[did] = anchor_row_to_dict(row)
        else:
            anchor_by_docid[did] = anchor_row_to_dict({})
    return doc_ids, doc_text, anchor_by_docid


# ==========================================================================
# retrievers
# ==========================================================================
class TfidfRetriever:
    """Reproducible char n-gram TF-IDF retriever (identical config to nb04)."""

    def __init__(self, ngram=(3, 4), min_df=3, max_features=100_000):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram,
                                   min_df=min_df, max_features=max_features,
                                   dtype=np.float32)

    def fit(self, doc_text):
        self.doc_mat = self.vec.fit_transform(doc_text)
        return self

    def search(self, queries, topn=100, batch=256):
        """Return (cand_idx[n_q,topn], cand_score[n_q,topn]) by base similarity."""
        n_doc = self.doc_mat.shape[0]
        topn = min(topn, n_doc)
        all_idx, all_score = [], []
        for s in range(0, len(queries), batch):
            qv = self.vec.transform(queries[s:s + batch])
            sims = cosine_similarity(qv, self.doc_mat)        # (b, n_doc)
            part = np.argpartition(-sims, kth=topn - 1, axis=1)[:, :topn]
            for i in range(sims.shape[0]):
                cand = part[i]
                cand = cand[np.argsort(-sims[i, cand])]
                all_idx.append(cand)
                all_score.append(sims[i, cand])
        return np.vstack(all_idx), np.vstack(all_score)


class DenseRetriever:
    """Multilingual dense retriever (mE5 / BGE-M3). Needs sentence-transformers."""

    def __init__(self, model_name, e5_prefixes=None, normalize=True, device=None):
        from sentence_transformers import SentenceTransformer   # lazy import
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.normalize = normalize
        # e5 family expects "query:"/"passage:" prefixes; bge-m3 does not.
        if e5_prefixes is None:
            e5_prefixes = "e5" in model_name.lower()
        self.e5 = e5_prefixes

    def _encode(self, texts, kind, batch_size=64):
        if self.e5:
            pre = "query: " if kind == "query" else "passage: "
            texts = [pre + t for t in texts]
        emb = self.model.encode(texts, batch_size=batch_size,
                                convert_to_numpy=True, show_progress_bar=False,
                                normalize_embeddings=self.normalize)
        return emb.astype(np.float32)

    def fit(self, doc_text):
        self.doc_emb = self._encode(doc_text, kind="passage")
        return self

    def search(self, queries, topn=100, batch=256):
        n_doc = self.doc_emb.shape[0]
        topn = min(topn, n_doc)
        all_idx, all_score = [], []
        for s in range(0, len(queries), batch):
            qe = self._encode(queries[s:s + batch], kind="query")
            sims = qe @ self.doc_emb.T                          # (b, n_doc)
            part = np.argpartition(-sims, kth=topn - 1, axis=1)[:, :topn]
            for i in range(sims.shape[0]):
                cand = part[i]
                cand = cand[np.argsort(-sims[i, cand])]
                all_idx.append(cand)
                all_score.append(sims[i, cand])
        return np.vstack(all_idx), np.vstack(all_score)


# ==========================================================================
# QOA reranking
# ==========================================================================
def _minmax(x):
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def qoa_rerank(cand_idx, cand_score, query_profiles, doc_ids, anchor_by_docid,
               dims, alpha):
    """Reorder each query's candidate pool by base + sum_k alpha_k * align_k.

    Parameters
    ----------
    cand_idx     : (n_q, topn) base-retriever candidate doc indices
    cand_score   : (n_q, topn) base similarity scores
    query_profiles : list of n_q matcher dicts (oracle or predicted)
    doc_ids      : (n_doc,) array mapping index -> doc_id
    anchor_by_docid : dict doc_id -> anchor matcher dict
    dims         : qualifier dimensions to use (QOA level)
    alpha        : array aligned to `dims`

    Returns
    -------
    reranked_idx : (n_q, topn) candidate indices in new order
    """
    alpha = np.asarray(alpha, dtype=float)
    n_q, topn = cand_idx.shape
    reranked = np.empty_like(cand_idx)
    for i in range(n_q):
        base = _minmax(cand_score[i])                      # scale base to [0,1]
        qp = query_profiles[i]
        bonus = np.zeros(topn)
        for j in range(topn):
            did = doc_ids[cand_idx[i, j]]
            av = alignment_vector(qp, anchor_by_docid.get(did, {}), dims)
            bonus[j] = float(np.dot(alpha, av))
        final = base + bonus
        order = np.argsort(-final)
        reranked[i] = cand_idx[i][order]
    return reranked


# ==========================================================================
# evaluation (with per-query arrays for bootstrap CIs)
# ==========================================================================
def gold_positions(ranked_idx, doc_ids, gold_ids):
    """0-based rank of each gold doc within the ranked candidate list; None if absent."""
    pos = []
    for i, gid in enumerate(gold_ids):
        row = doc_ids[ranked_idx[i]]
        hit = np.where(row == gid)[0]
        pos.append(int(hit[0]) if len(hit) else None)
    return pos


def metrics_from_positions(positions, k_list=RETRIEVAL_K_LIST, n_boot=BOOTSTRAP_N,
                           ci=BOOTSTRAP_CI, seed=42):
    """Aggregate retrieval metrics + bootstrap CIs from per-query gold positions."""
    from metrics import bootstrap_ci
    n = len(positions)
    hit = {k: np.array([1.0 if (p is not None and p < k) else 0.0 for p in positions])
           for k in k_list}
    rr = np.array([1.0 / (p + 1) if p is not None else 0.0 for p in positions])
    maxk = max(k_list)
    ndcg = np.array([1.0 / np.log2(p + 2) if (p is not None and p < maxk) else 0.0
                     for p in positions])
    out = {"n": n}
    for k in k_list:
        ci_k = bootstrap_ci(hit[k], n_boot=n_boot, ci=ci, seed=seed)
        out[f"Recall@{k}"] = round(ci_k["point"], 4)
        out[f"Recall@{k}_lo"] = round(ci_k["lo"], 4)
        out[f"Recall@{k}_hi"] = round(ci_k["hi"], 4)
    mrr_ci = bootstrap_ci(rr, n_boot=n_boot, ci=ci, seed=seed)
    out["MRR"] = round(mrr_ci["point"], 4)
    out["MRR_lo"] = round(mrr_ci["lo"], 4)
    out["MRR_hi"] = round(mrr_ci["hi"], 4)
    out[f"nDCG@{maxk}"] = round(float(ndcg.mean()), 4)
    return out, {"hit": hit, "rr": rr, "ndcg": ndcg}


# ==========================================================================
# deterministic splits + alpha tuning
# ==========================================================================
def make_splits(n, fracs=SPLIT_FRACS, seed=42):
    """Deterministic index split into train/val/test."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_tr = int(fracs["train"] * n)
    n_va = int(fracs["val"] * n)
    return {"train": np.sort(idx[:n_tr]),
            "val":   np.sort(idx[n_tr:n_tr + n_va]),
            "test":  np.sort(idx[n_tr + n_va:])}


def tune_alpha(cand_idx, cand_score, query_profiles, doc_ids, anchor_by_docid,
               gold_ids, dims, val_idx, grid=QOA_ALPHA_GRID, target="Recall@10",
               seed=42):
    """Coordinate-ascent over per-dimension alpha on the VALIDATION split.

    Returns the alpha vector (aligned to `dims`) maximising `target` on val.
    """
    alpha = np.zeros(len(dims))

    def score(alpha_vec):
        rr = qoa_rerank(cand_idx[val_idx], cand_score[val_idx],
                        [query_profiles[i] for i in val_idx],
                        doc_ids, anchor_by_docid, dims, alpha_vec)
        pos = gold_positions(rr, doc_ids, [gold_ids[i] for i in val_idx])
        m, _ = metrics_from_positions(pos, n_boot=1, seed=seed)  # point only
        return m[target]

    best = score(alpha)
    improved = True
    while improved:
        improved = False
        for d in range(len(dims)):
            cur = alpha[d]
            for g in grid:
                if g == cur:
                    continue
                trial = alpha.copy()
                trial[d] = g
                s = score(trial)
                if s > best + 1e-9:
                    best, alpha, improved = s, trial, True
    return alpha, best
