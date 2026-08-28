"""
metrics.py — 교차언어 일관성 및 준수판정 평가 지표

기획서 C-2 구현. 외부 의존 최소(numpy만). LLM 예측 결과가 주어지면
언어 간 일치·정확도 격차·보정오차를 계산한다.
"""
from __future__ import annotations
import numpy as np


def agreement(pred_a: list, pred_b: list) -> float:
    """두 언어 예측의 단순 일치율."""
    a = np.asarray(pred_a); b = np.asarray(pred_b)
    assert len(a) == len(b)
    return float((a == b).mean()) if len(a) else 0.0


def cohen_kappa(pred_a: list, pred_b: list, labels: list | None = None) -> float:
    """Cohen's κ (우연 보정 일치). 두 rater = 두 언어 예측."""
    a = np.asarray(pred_a); b = np.asarray(pred_b)
    if labels is None:
        labels = sorted(set(a.tolist()) | set(b.tolist()))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(a)
    if n == 0:
        return 0.0
    K = len(labels)
    conf = np.zeros((K, K))
    for x, y in zip(a, b):
        conf[idx[x], idx[y]] += 1
    po = np.trace(conf) / n
    row = conf.sum(axis=1) / n
    col = conf.sum(axis=0) / n
    pe = float((row * col).sum())
    return float((po - pe) / (1 - pe)) if (1 - pe) > 1e-12 else 1.0


def accuracy(pred: list, gold: list) -> float:
    p = np.asarray(pred); g = np.asarray(gold)
    return float((p == g).mean()) if len(p) else 0.0


def accuracy_gap(pred_en: list, pred_local: list, gold: list) -> dict:
    """Acc(en) - Acc(local): H2 핵심 지표."""
    acc_en = accuracy(pred_en, gold)
    acc_local = accuracy(pred_local, gold)
    return {
        "acc_en": round(acc_en, 4),
        "acc_local": round(acc_local, 4),
        "gap": round(acc_en - acc_local, 4),
    }


def expected_calibration_error(confidences: list, correct: list, n_bins: int = 10) -> float:
    """ECE: 확신도와 실제 정답률의 괴리(과신 진단)."""
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    if len(conf) == 0:
        return 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(conf)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc_bin = corr[mask].mean()
        conf_bin = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc_bin - conf_bin)
    return float(ece)


def retrieval_overlap(topk_a: list, topk_b: list) -> float:
    """언어별 top-k 검색결과 Jaccard 중첩(과제 B 일관성)."""
    sa, sb = set(topk_a), set(topk_b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)
