# 전처리 — 종목 교차 학습/평가셋 생성
#   창 선택 = eda.py 그대로 재사용(fig1_highlight_*와 동일 구간)
#   클리닝: URL 제거 → strip → 3자 이하 제거 (종목 관련성 필터는 하지 않음)
#   길이: 512 truncation은 토크나이즈 시점 처리(여기선 원문 보존, 길이 제외 안 함)
#   평가셋: 상대 종목 학습셋 작성자(닉네임_ID) 댓글 전부 제거 (user-disjoint)
import os
import re
import pandas as pd

import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
from exploration import (load_comments, _best_var_window, _best_eval_window,
                         _window_users, ROOT)

OUTDIR = os.path.join(ROOT, "data", "unlabeled", "splits")
os.makedirs(OUTDIR, exist_ok=True)

URL_RE = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|io|ly|kr|co\.kr)(?:/\S*)?",
                    re.IGNORECASE)
WS_RE = re.compile(r"\s+")
MIN_CHARS = 4          # 3자 이하 제거 → 4자 이상만 유지

# 출력에 보존할 컬럼 (가명 ID·주주여부 등 후속 단계 필수)
KEEP = ["commentId", "닉네임_ID", "종목명", "주주_여부", "뱃지",
        "작성일", "us_date", "내용", "text"]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """URL 제거 → 공백 정리 → 4자 미만 제거. 원문(내용)은 그대로 두고 text 컬럼 신설."""
    t = df["내용"].fillna("").astype(str)
    t = t.str.replace(URL_RE, " ", regex=True)
    t = t.str.replace(WS_RE, " ", regex=True).str.strip()
    out = df.copy()
    out["text"] = t
    keep = out["text"].str.len() >= MIN_CHARS
    return out[keep].copy()


def slice_window(df, stock, w):
    return df[(df["종목명"] == stock)
             & (df["us_date"] >= w["start"]) & (df["us_date"] <= w["end"])].copy()


def main():
    df = load_comments()
    print(f"로드: {len(df):,}건")

    # 1) 학습 창(20k, 균형 up-down) + 학습셋 유저 집합
    train_w = {s: _best_var_window(df, s, 20000) for s in ["TSLA", "NVDA"]}
    tusers = {s: _window_users(df, s, train_w[s]) for s in ["TSLA", "NVDA"]}

    # 2) 평가 창(10k) — 상대 종목 학습셋 유저와 최소 겹침
    eval_w = {"NVDA": _best_eval_window(df, "NVDA", 10000, tusers["TSLA"]),  # Fold A
              "TSLA": _best_eval_window(df, "TSLA", 10000, tusers["NVDA"])}  # Fold B

    rows = []
    # ── 학습셋 (TSLA→A, NVDA→B) ─────────────────────────────
    for fold, s in [("A", "TSLA"), ("B", "NVDA")]:
        w = train_w[s]
        raw = slice_window(df, s, w)
        cl = clean(raw)
        cl[KEEP].to_csv(os.path.join(OUTDIR, f"train_{fold}_{s}.csv"),
                        index=False, encoding="utf-8-sig")
        rows.append((f"{fold}-train", s, f'{w["start"]}~{w["end"]}',
                     len(raw), len(cl), "-", "-"))

    # ── 평가셋 (NVDA→A, TSLA→B) + user-disjoint ─────────────
    #   상대 종목 학습 유저: A평가(NVDA)는 TSLA 학습유저, B평가(TSLA)는 NVDA 학습유저
    for fold, s, other in [("A", "NVDA", "TSLA"), ("B", "TSLA", "NVDA")]:
        w = eval_w[s]
        raw = slice_window(df, s, w)
        cl = clean(raw)
        n_clean = len(cl)
        bad = cl["닉네임_ID"].isin(tusers[other])
        rm_users = cl.loc[bad, "닉네임_ID"].nunique()
        rm_cmt = int(bad.sum())
        cl = cl[~bad].copy()
        cl[KEEP].to_csv(os.path.join(OUTDIR, f"test_{fold}_{s}.csv"),
                        index=False, encoding="utf-8-sig")
        rows.append((f"{fold}-test", s, f'{w["start"]}~{w["end"]}',
                     len(raw), n_clean, f"{rm_users}명/{rm_cmt}건", len(cl)))

    print("\n구분        종목   구간                      raw     clean   제거(user)      최종")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:<5} {r[2]:<24} {r[3]:>7,} {r[4]:>7,}  {str(r[5]):<14} {str(r[6]):>7}")
    print(f"\n저장 -> {OUTDIR}/{{train,test}}_{{A,B}}_{{종목}}.csv")


if __name__ == "__main__":
    main()
