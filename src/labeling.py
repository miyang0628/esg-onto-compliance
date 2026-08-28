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
