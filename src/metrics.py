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


# ==========================================================================
# REVISION ADD-ONS (R1): uncertainty, equivalence testing, selective prediction
# --------------------------------------------------------------------------
# Reviewer 2: M5 (risk-coverage curves, ECE, uncertainty on main comparisons;
# justify "negligible gap" via an equivalence test with a stated margin) and
# Results Q4 (CIs needed; an observed 0.008 does not establish negligibility).
# ==========================================================================


def bootstrap_ci(values, stat_fn=np.mean, n_boot: int = 1000,
                 ci: float = 0.95, seed: int = 42) -> dict:
    """Nonparametric bootstrap CI for a statistic over a 1-D sample."""
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return {"point": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = stat_fn(x[idx])
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {"point": float(stat_fn(x)), "lo": lo, "hi": hi, "n": n}


def paired_diff_ci(correct_a, correct_b, n_boot: int = 1000,
                   ci: float = 0.95, seed: int = 42) -> dict:
    """Bootstrap CI for the PAIRED difference in accuracy (acc_a - acc_b).

    correct_a / correct_b are 0/1 correctness arrays on the SAME items
    (e.g. English vs local on the same cases). Paired resampling keeps the
    dependence structure, which an unpaired CI would ignore.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    assert len(a) == len(b), "paired arrays must be equal length"
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = a[idx].mean() - b[idx].mean()
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return {"diff": float(a.mean() - b.mean()), "lo": lo, "hi": hi, "n": n}


def tost_equivalence(correct_a, correct_b, margin: float = 0.05,
                     n_boot: int = 2000, seed: int = 42) -> dict:
    """Two One-Sided Tests (TOST) for equivalence of paired accuracies.

    H0 (non-equivalence): |acc_a - acc_b| >= margin.
    We reject H0 (i.e. conclude EQUIVALENCE within +/- margin) when the
    (1-2*alpha) bootstrap CI of the paired difference lies entirely inside
    (-margin, +margin). Returns the 90% CI (alpha=0.05 each side) and a verdict.

    This is what lets the paper say "negligible gap" defensibly, instead of
    inferring it from a single point estimate of 0.008.
    """
    res = paired_diff_ci(correct_a, correct_b, n_boot=n_boot, ci=0.90, seed=seed)
    equivalent = (res["lo"] > -margin) and (res["hi"] < margin)
    return {"diff": res["diff"], "ci90_lo": res["lo"], "ci90_hi": res["hi"],
            "margin": margin, "equivalent": bool(equivalent), "n": res["n"]}


def risk_coverage_curve(confidences, correct):
    """Selective-prediction risk-coverage curve.

    Sort items by DESCENDING confidence; at each prefix (coverage = fraction
    answered) compute risk = 1 - accuracy on the answered subset.
    Returns arrays (coverage, risk) plus AURC (area under risk-coverage).
    """
    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(correct, dtype=float)
    n = len(conf)
    if n == 0:
        return {"coverage": np.array([]), "risk": np.array([]), "aurc": 0.0}
    order = np.argsort(-conf)
    corr_sorted = corr[order]
    cum_correct = np.cumsum(corr_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = 1.0 - cum_correct / k
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    aurc = float(_trapz(risk, coverage))
    return {"coverage": coverage, "risk": risk, "aurc": aurc}


def coverage_at_risk(confidences, correct, target_risk: float) -> float:
    """Maximum coverage achievable while keeping selective risk <= target."""
    rc = risk_coverage_curve(confidences, correct)
    cov, risk = rc["coverage"], rc["risk"]
    ok = cov[risk <= target_risk]
    return float(ok.max()) if len(ok) else 0.0


def selective_accuracy(preds, gold, confidences, threshold: float):
    """Accuracy on items whose confidence >= threshold, plus coverage."""
    p = np.asarray(preds)
    g = np.asarray(gold)
    c = np.asarray(confidences, dtype=float)
    mask = c >= threshold
    cov = float(mask.mean()) if len(c) else 0.0
    acc = float((p[mask] == g[mask]).mean()) if mask.sum() else 0.0
    return {"threshold": threshold, "coverage": cov, "selective_acc": acc,
            "n_answered": int(mask.sum())}
