"""
data_loader.py — 원천/라벨 zip을 결정론적으로 파싱해 DataFrame으로 반환

설계 원칙(재현성):
  - zip을 디스크에 풀지 않고 메모리에서 직접 읽는다(중간 파일 오염 방지).
  - 파일 순회는 항상 정렬된 순서로(sorted) — OS/파일시스템에 무관하게 동일 순서.
  - 결과 DataFrame은 'id' 기준 정렬 후 반환 — 행 순서까지 재현.
"""
from __future__ import annotations
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator

import pandas as pd

from config import RAW_DIR, SOURCE_ZIPS, LABEL_ZIP


# ==========================================================================
# 원천 데이터
# ==========================================================================
def _iter_json_in_zip(zf: zipfile.ZipFile) -> Iterator[dict]:
    """zip 내 .json을 파일명 정렬 순서로 yield."""
    for name in sorted(zf.namelist()):
        if not name.endswith(".json"):
            continue
        with zf.open(name) as fh:
            try:
                yield json.load(fh)
            except json.JSONDecodeError:
                continue


def _flatten_meta_record(d: dict) -> dict:
    """
    원천 JSON 1건 → **메타데이터만** 평탄화(경량).
    3중 병렬 본문(content)은 여기 포함하지 않는다 — 문서당 평균 5만자 × 3언어이므로
    메인 테이블에 실으면 수 GB로 불어나 로컬 재현이 불가능해진다.
    본문은 load_content() 로 필요할 때만 국가·카테고리 단위로 로드한다.
    """
    data = d.get("data", {}) or {}
    m = d.get("metadata", {}) or {}
    reg = m.get("regulation", {}) or {}
    ci = m.get("case_info", {}) or {}
    result = ci.get("result", {}) or {}
    content = data.get("content", {}) or {}

    return {
        "id":            m.get("id", ""),
        "doc_id":        data.get("id", ""),
        "file_name":     m.get("file_name", ""),
        "category":      m.get("category", ""),
        "country":       m.get("country", ""),
        "region":        m.get("region", ""),
        "title":         m.get("title", ""),
        "alias":         m.get("alias", ""),
        "esg":           m.get("esg", ""),
        "document_type": m.get("document_type", ""),
        "publisher":     m.get("publisher", ""),
        "entity":        m.get("entity", ""),
        "published_at":  m.get("published_at", ""),
        "source":        m.get("source", ""),
        "copyrighter":   m.get("copyrighter", ""),
        # 본문은 길이만 보관(분포 분석·채움율 판정용). 원문은 별도 저장.
        "len_ko":    len(content.get("ko", "") or ""),
        "len_en":    len(content.get("en", "") or ""),
        "len_local": len(content.get("local", "") or ""),
        # 규제 조문
        "reg_name":      reg.get("regulation_name", "") or "",
        "reg_reference": reg.get("regulation_reference", "") or "",
        "effective_date": reg.get("effective_date", "") or "",
        # 판례 case_info
        "case_detail":       ci.get("detail", "") or "",
        "pollutant_category": ci.get("pollutant_category", "") or "",
        "sanction_type":     result.get("type", "") or "",
        "sanction_level":    result.get("level", "") or "",
        "violation_date":    result.get("violation_date", "") or "",
        "sanction_date":     result.get("sanction_date", "") or "",
        "previous_violations": json.dumps(ci.get("previous_violations", []), ensure_ascii=False),
    }


def load_source(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    원천 12개 zip 전체를 파싱해 **메타데이터 DataFrame**을 반환(경량, ~40k행).
    본문(content)은 포함하지 않는다 → load_content() 사용.
    """
    rows = []
    for zip_name in SOURCE_ZIPS:  # config에 고정된 순서
        zpath = raw_dir / zip_name
        if not zpath.exists():
            raise FileNotFoundError(
                f"원천 zip을 찾을 수 없습니다: {zpath}\n"
                f"→ AI Hub 원본 zip을 data/raw/ 아래에 넣어주세요."
            )
        with zipfile.ZipFile(zpath) as zf:
            for d in _iter_json_in_zip(zf):
                rows.append(_flatten_meta_record(d))
    df = pd.DataFrame(rows)
    df = df.sort_values("id", kind="mergesort").reset_index(drop=True)
    return df


# 파일명 → (국가, 카테고리번호) 매핑: 본문을 선택적으로 로드할 때 사용
def _zip_country_cat(zip_name: str) -> tuple[str, str]:
    """
    zip 파일명 → (국가, 카테고리코드2자리).
    예: 'TS_1__베트남_03__판례.zip' → ('베트남', '03')
    """
    import re
    stem = zip_name.replace(".zip", "")
    country = "베트남" if "베트남" in stem else "말레이시아"
    # '_03__' 형태의 2자리 코드 추출
    m = re.search(r"_(\d{2})__", stem)
    catcode = m.group(1) if m else ""
    return country, catcode


def load_content(ids: set[str] | None = None,
                 country: str | None = None,
                 category_code: str | None = None,
                 raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """
    3중 병렬 본문을 **선택적으로** 로드한다(메모리 안전).

    Parameters
    ----------
    ids : 특정 문서 id 집합만 로드(교차언어 평가셋 등). None이면 필터 안 함.
    country : "베트남"/"말레이시아"로 zip 범위 축소.
    category_code : "01"(규제)..."06"(국제) zip 범위 축소.

    Returns
    -------
    DataFrame[id, content_ko, content_en, content_local]
    """
    rows = []
    for zip_name in SOURCE_ZIPS:
        zc, cc = _zip_country_cat(zip_name)
        if country and zc != country:
            continue
        if category_code and cc != category_code:
            continue
        with zipfile.ZipFile(raw_dir / zip_name) as zf:
            for d in _iter_json_in_zip(zf):
                _id = d.get("metadata", {}).get("id", "")
                if ids is not None and _id not in ids:
                    continue
                c = d.get("data", {}).get("content", {}) or {}
                rows.append({
                    "id": _id,
                    "content_ko":    c.get("ko", "") or "",
                    "content_en":    c.get("en", "") or "",
                    "content_local": c.get("local", "") or "",
                })
    return pd.DataFrame(rows).sort_values("id", kind="mergesort").reset_index(drop=True)


# ==========================================================================
# 라벨 데이터 (중첩 zip: 국가 × {QA, 벤치마크} × 카테고리)
# ==========================================================================
def _iter_inner_label_zips(outer: zipfile.ZipFile) -> Iterator[tuple[str, bytes]]:
    """바깥 라벨 zip 안의 내부 .zip들을 (내부이름, 바이트)로 yield."""
    for name in sorted(outer.namelist()):
        if not name.endswith(".zip"):
            continue
        with outer.open(name) as fh:
            yield name, fh.read()


def _classify_label_zip(inner_name: str) -> str:
    """내부 zip 파일명으로 QA / 벤치마크 구분."""
    base = Path(inner_name).name
    if "QA" in base:
        return "qa"
    if "벤치마크" in base or "benchmark" in base.lower():
        return "benchmark"
    return "unknown"


def load_labels(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    라벨 zip을 파싱해 (qa_df, benchmark_df) 반환.
    QA: 멀티턴 대화 / 벤치마크: 객관식(a_option, a_index).
    """
    zpath = raw_dir / LABEL_ZIP
    if not zpath.exists():
        raise FileNotFoundError(f"라벨 zip을 찾을 수 없습니다: {zpath}")

    qa_rows, bm_rows = [], []
    with zipfile.ZipFile(zpath) as outer:
        for inner_name, inner_bytes in _iter_inner_label_zips(outer):
            kind = _classify_label_zip(inner_name)
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner_zf:
                for d in _iter_json_in_zip(inner_zf):
                    qa = d.get("qa", {}) or {}
                    base = {
                        "id":        d.get("id", ""),
                        "category":  qa.get("category", ""),
                        "country":   qa.get("country", ""),
                        "esg":       qa.get("ESG", ""),
                        "source_id": json.dumps(qa.get("source_id", []), ensure_ascii=False),
                    }
                    if kind == "qa":
                        turns = qa.get("qa_turns", [])
                        base["n_turns"] = len(turns)
                        base["qa_turns"] = json.dumps(turns, ensure_ascii=False)
                        qa_rows.append(base)
                    elif kind == "benchmark":
                        turns = qa.get("qa_turns", [])
                        t0 = turns[0] if turns else {}
                        base["question"]    = t0.get("q_content", "")
                        base["options"]     = json.dumps(t0.get("a_option", []), ensure_ascii=False)
                        base["answer_index"] = t0.get("a_index", -1)
                        base["explanation"] = t0.get("a_explanation", "")
                        bm_rows.append(base)

    qa_df = pd.DataFrame(qa_rows).sort_values("id", kind="mergesort").reset_index(drop=True)
    bm_df = pd.DataFrame(bm_rows).sort_values("id", kind="mergesort").reset_index(drop=True)
    return qa_df, bm_df
