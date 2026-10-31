"""
labeling.py — 판례 case_info → 준수판정 골드 라벨 유도 규칙

기획서 (b)절 구현. 라벨은 '공시 준수'가 아니라 **'법적 책임 성립'**의 프록시임을
명시적으로 한정한다(과대해석 차단).

규칙 요지:
  - result(type+level) 텍스트에 제재 신호만 있으면 NON_COMPLIANT
  - 면책 신호만 있으면 COMPLIANT
  - 둘 다/혼재/불충분 → OTHER (인적 검수)
  - result 공란 → UNLABELED (detail 기반 수동 또는 제외)

주의: 화폐 단위 '동'(VND)은 '1심 소송비용 부담' 등 중립 문구에도 흔히 나와
      제재 신호로 쓰면 과다 포획된다. 따라서 SANCTION_TOKENS에서 제외한다.
"""
from __future__ import annotations
import pandas as pd

from config import (
    SANCTION_TOKENS, ACQUIT_TOKENS,
    LABEL_NON_COMPLIANT, LABEL_COMPLIANT, LABEL_OTHER, LABEL_UNLABELED,
)


def derive_label(sanction_type: str, sanction_level: str) -> str:
    """단일 판례 result → 준수판정 라벨."""
    s = (str(sanction_type or "") + " " + str(sanction_level or "")).strip()
    if not s:
        return LABEL_UNLABELED
    has_sanction = any(tok in s for tok in SANCTION_TOKENS)
    has_acquit = any(tok in s for tok in ACQUIT_TOKENS)
    if has_sanction and not has_acquit:
        return LABEL_NON_COMPLIANT
    if has_acquit and not has_sanction:
        return LABEL_COMPLIANT
    return LABEL_OTHER


def label_cases(case_df: pd.DataFrame) -> pd.DataFrame:
    """
    판례 DataFrame(id, sanction_type, sanction_level, ...)에 label 열을 추가해 반환.
    입력 순서를 보존하며 결정론적.
    """
    out = case_df.copy()
    out["label"] = [
        derive_label(t, l)
        for t, l in zip(out["sanction_type"], out["sanction_level"])
    ]
    return out


def reliability_summary(labeled: pd.DataFrame) -> dict:
    """자동 라벨 신뢰도 요약(신뢰 자동 = NON+COMP)."""
    vc = labeled["label"].value_counts().to_dict()
    n = len(labeled)
    reliable = vc.get(LABEL_NON_COMPLIANT, 0) + vc.get(LABEL_COMPLIANT, 0)
    return {
        "counts": vc,
        "n": n,
        "reliable_auto": reliable,
        "reliable_ratio": round(reliable / n, 4) if n else 0.0,
    }


# ==========================================================================
# REVISION ADD-ONS (R1): label validity, leakage, audit, reconciliation (M4)
# ==========================================================================
from config import SANCTION_TOKENS as _SANC, ACQUIT_TOKENS as _ACQ  # noqa: E402


def verdict_leakage(content_text: str) -> dict:
    """Does the case *body* contain explicit verdict/sanction wording?

    Reviewer 2 (M4): if the judged text already states the outcome, the task may
    reduce to outcome extraction rather than compliance reasoning. We flag, per
    case, whether sanction/acquittal tokens appear in the text actually sent to
    the judge, so the manuscript can report the leakage rate and (optionally)
    re-run on verdict-masked text.
    """
    s = str(content_text or "")
    hit_sanction = [t for t in _SANC if t in s]
    hit_acquit = [t for t in _ACQ if t in s]
    return {
        "leaks_verdict": bool(hit_sanction or hit_acquit),
        "n_sanction_tokens": len(hit_sanction),
        "n_acquit_tokens": len(hit_acquit),
        "tokens": (hit_sanction + hit_acquit)[:10],
    }


def mask_verdict(content_text: str, placeholder: str = "[REDACTED]") -> str:
    """Replace sanction/acquittal tokens so the judge cannot copy the outcome."""
    s = str(content_text or "")
    for t in sorted(set(_SANC + _ACQ), key=len, reverse=True):
        s = s.replace(t, placeholder)
    return s


def audit_sample(labeled, n_per_label: int = 30, seed: int = 42):
    """Deterministic stratified sample of auto-labeled cases for human audit.

    Produces a review sheet (id, label, sanction_type, sanction_level, detail)
    the authors fill with a human judgment, so inter-annotator agreement with
    the rule can be reported.
    """
    import pandas as pd
    cols = [c for c in ["id", "label", "sanction_type", "sanction_level",
                        "case_detail", "country"] if c in labeled.columns]
    parts = []
    for lab, g in labeled.groupby("label"):
        parts.append(g.sample(n=min(n_per_label, len(g)), random_state=seed)[cols])
    out = pd.concat(parts).sort_values(["label", "id"]).reset_index(drop=True)
    out["human_label"] = ""        # to be filled by an annotator
    out["agrees_with_rule"] = ""   # to be filled by an annotator
    return out


def reconcile_counts(source) -> dict:
    """Reconcile the case-count discrepancies reviewer 2 (minor 2) flagged.

    Explains: total cases vs populated outcomes vs UNLABELED vs
    resultsInSanction edges, all from one source of truth.
    """
    import pandas as pd
    case = source[source["category"] == "판례"].copy()
    total = len(case)
    result_filled = (case["sanction_type"].fillna("").astype(str).str.strip() != "").sum()
    labeled = label_cases(case)
    vc = labeled["label"].value_counts().to_dict()
    return {
        "total_cases": int(total),
        "result_field_filled": int(result_filled),
        "result_field_empty(UNLABELED)": int(total - result_filled),
        "label_counts": {k: int(v) for k, v in vc.items()},
        "note": ("resultsInSanction edges count only cases whose rule label is "
                 "NON_COMPLIANT; OTHER/COMPLIANT/UNLABELED carry no sanction edge."),
    }
