"""
config.py — 프로젝트 전역 설정 (경로 · 시드 · 상수)

재현성 원칙:
  - 모든 경로는 이 파일의 위치를 기준으로 계산한다(하드코딩 절대경로 금지).
  - 모든 무작위 시드는 SEED 하나로 통일한다.
  - 카테고리/라벨 상수는 여기서만 정의하고 노트북은 import 해서 쓴다.
"""
from __future__ import annotations
import os
import random
from pathlib import Path

# --------------------------------------------------------------------------
# 경로 (이 파일: <PROJECT_ROOT>/src/config.py 기준)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR      = PROJECT_ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"          # AI Hub 원본 zip (읽기 전용, 수정 금지)
INTERIM_DIR   = DATA_DIR / "interim"      # 파싱·정제 중간 산출물
PROCESSED_DIR = DATA_DIR / "processed"    # 온톨로지 노드/엣지, 준수판정 골드

RESULTS_DIR   = PROJECT_ROOT / "results"
FIGURES_DIR   = RESULTS_DIR / "figures"
TABLES_DIR    = RESULTS_DIR / "tables"

for _d in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# 재현성: 시드
# --------------------------------------------------------------------------
SEED = 42

def set_seed(seed: int = SEED) -> None:
    """모든 라이브러리 시드를 한 번에 고정. 노트북 첫 셀에서 호출."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

# --------------------------------------------------------------------------
# 원천 데이터 상수
# --------------------------------------------------------------------------
# AI Hub 원천 zip 파일명 (data/raw/ 아래에 그대로 위치)
SOURCE_ZIPS = [
    "TS_1__베트남_01__규제.zip",
    "TS_1__베트남_02__입법_동향.zip",
    "TS_1__베트남_03__판례.zip",
    "TS_1__베트남_04__기사.zip",
    "TS_1__베트남_05__정부_및_기업_ESG_자료.zip",
    "TS_1__베트남_06__국제_ESG_자료.zip",
    "TS_2__말레이시아_01__규제.zip",
    "TS_2__말레이시아_02__입법_동향.zip",
    "TS_2__말레이시아_03__판례.zip",
    "TS_2__말레이시아_04__기사.zip",
    "TS_2__말레이시아_05__정부_및_기업_ESG_자료.zip",
    "TS_2__말레이시아_06__국제_ESG_자료.zip",
]
LABEL_ZIP = "02_라벨링데이터.zip"

CATEGORIES = ["규제", "입법 동향", "판례", "기사", "정부 및 기업 ESG 자료", "국제 ESG 자료"]
COUNTRIES  = ["베트남", "말레이시아"]
ESG_TYPES  = ["E", "S", "G"]

# 실측 기준값 (parsing 검증용 — 로컬에서 이 값과 달라지면 데이터 누락 경고)
EXPECTED_SOURCE_TOTAL = 40078          # 업로드 제공분 (공식 전체 50,098)
EXPECTED_CATEGORY_COUNTS = {
    "규제": 9614, "국제 ESG 자료": 7986, "정부 및 기업 ESG 자료": 7951,
    "입법 동향": 7234, "기사": 4764, "판례": 2529,
}
EXPECTED_COUNTRY_COUNTS = {"베트남": 20079, "말레이시아": 19999}
EXPECTED_ESG_COUNTS = {"E": 26342, "S": 9104, "G": 4632}

# --------------------------------------------------------------------------
# 준수판정 라벨 상수 (compliance labeling)
# --------------------------------------------------------------------------
LABEL_NON_COMPLIANT = "NON_COMPLIANT"
LABEL_COMPLIANT     = "COMPLIANT"
LABEL_OTHER         = "OTHER"      # PARTIAL/MIXED/불명 → 인적 검수
LABEL_UNLABELED     = "UNLABELED"  # result 공란

# 제재 신호: 명확한 처벌 결과 어휘.
# 주의) 화폐 단위 "동"(VND)은 '판결비용 부담' 등 중립 문구에도 등장하여
#       과다 포획을 유발하므로 SANCTION 토큰에서 제외한다(§labeling 주석 참조).
SANCTION_TOKENS = [
    "징역", "벌금", "태형", "손해배상", "배상책임", "해고",
    "과태료", "몰수", "영업정지", "등록취소", "취소", "환수",
]
ACQUIT_TOKENS = [
    "무죄", "기각", "각하", "면제", "책임 없음", "해당 없음", "없음",
]
