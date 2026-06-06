# (3-2) 주가 결과 결정적 계산 + (3-3) Decision Logic → Class 0~3 부여.
#   LLM 출력(labeled_*.csv) + 주가(1d) → 우리가 코드로 적중여부 계산(누수/재현성).
#   정책 상세: 프로젝트 루트 labeling.md
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import exploration  # noqa: E402  (load_prices, μ·σ)

DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "unlabeled", "splits")
LABELED = os.path.join(DATA, "labeled")
FILES = ["tsla_train", "nvda_eval", "expert_control"]

# ── 파라미터 (labeling.md 확정) ─────────────────────────────
TAU = 0.5          # 확신도 컷오프
K = 1.0            # 판정폭 계수
DIR_FWD = 3        # W_dir forward 거래일
HIT_HALF = 1       # W_hit = D ± HIT_HALF
LABEL_COLS = ["시점관계", "예측_방향", "예측_날짜", "확신도", "근거_유형", "Class0_사유"]


def load_stock(stock: str) -> dict:
    p, mu, sig = exploration.load_prices(stock)
    p = p.sort_values("Date").reset_index(drop=True)
    dates = pd.to_datetime(p["Date"]).to_numpy()                      # datetime64[D] 자정
    closes = p["Close"].to_numpy(float)
    # 세션 마감시각 = 해당 거래일 16:00 ET → UTC naive 로 환산해 비교용
    close_et = (pd.to_datetime(p["Date"]) + pd.Timedelta(hours=16)).dt.tz_localize(
        "America/New_York")
    close_utc = close_et.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    return dict(dates=dates, closes=closes, close_utc=close_utc, mu=mu, sig=sig, n=len(p))


def anchor_idx(sd, comment_iso) -> int:
    """작성시각 기준 마지막으로 '마감된' 세션 인덱스(없으면 -1)."""
    t = pd.to_datetime(comment_iso).tz_convert("UTC").tz_localize(None).to_numpy()
    return int(np.searchsorted(sd["close_utc"], t, side="right") - 1)


def session_for_date(sd, D):
    """예측 날짜 D 이상인 첫 거래세션 인덱스(범위 밖이면 None)."""
    d = np.datetime64(pd.to_datetime(D).normalize(), "ns")
    i = int(np.searchsorted(sd["dates"].astype("datetime64[ns]"), d, side="left"))
    return i if i < sd["n"] else None


def crossed(direction, R, mu, sig, h, k=K) -> bool:
    """누적수익률 R(%)이 예측 방향으로 판정선 μh±kσ√h 를 넘었는가."""
    up = mu * h + k * sig * np.sqrt(h)
    dn = mu * h - k * sig * np.sqrt(h)
    if direction == "상승":
        return R >= up
    if direction == "하락":
        return R <= dn
    return False


def classify(row, sd) -> dict:
    direction = row["예측_방향"]
    conf = float(row["확신도"]) if pd.notna(row["확신도"]) else 0.0
    rel = row["시점관계"]
    D = row["예측_날짜"] if pd.notna(row["예측_날짜"]) else None

    A = anchor_idx(sd, row["작성일"])
    D_idx = session_for_date(sd, D) if D else None
    date_valid = (D_idx is not None) and (A >= 0) and (D_idx > A)   # 과거날짜 무효

    dir_realized = False
    date_hit = False
    evaluable = A >= 0
    R_dir = np.nan
    if evaluable:
        end = (D_idx if date_valid else A) + DIR_FWD
        if end >= sd["n"]:
            evaluable = False                      # 미래봉 부족 → 보수적 미실현
        else:
            h = end - A
            R_dir = (sd["closes"][end] / sd["closes"][A] - 1) * 100
            dir_realized = crossed(direction, R_dir, sd["mu"], sd["sig"], h)
        if date_valid:
            hb, he = D_idx - (HIT_HALF + 1), D_idx + HIT_HALF
            if hb >= 0 and he < sd["n"]:
                Rh = (sd["closes"][he] / sd["closes"][hb] - 1) * 100
                date_hit = crossed(direction, Rh, sd["mu"], sd["sig"], he - hb)

    # (3-3) Decision Logic — 위에서부터
    if direction == "없음" or conf < TAU or rel == "리액션":
        cls = 0
    elif not dir_realized:
        cls = 1
    elif not date_valid or not date_hit:
        cls = 2
    else:
        cls = 3
    return dict(Class=cls, dir_realized=dir_realized, date_valid=date_valid,
                date_hit=date_hit, evaluable=evaluable, R_dir=round(R_dir, 2)
                if pd.notna(R_dir) else np.nan)


def run(name):
    lab = pd.read_csv(os.path.join(LABELED, f"{name}_labeled.csv"), encoding="utf-8-sig",
                      on_bad_lines="skip", engine="python")
    src = pd.read_csv(os.path.join(SPLITS, f"{name}.csv"), encoding="utf-8-sig")

    df = src.merge(lab, on="commentId", how="inner")
    n_err = int(df["시점관계"].isna().sum())
    df = df[df["시점관계"].notna()].copy()          # 라벨 실패행 제외(재라벨 대상)

    sds = {s: load_stock(s) for s in df["종목명"].unique()}   # 종목 섞여도 행별 처리
    stock = "+".join(sds)
    res = df.apply(lambda r: classify(r, sds[r["종목명"]]), axis=1, result_type="expand")
    out = pd.concat([df, res], axis=1)
    keep = (["commentId", "닉네임_ID", "종목명", "grp", "주주_여부", "작성일", "text"]
            + LABEL_COLS + ["Class", "dir_realized", "date_valid", "date_hit",
                            "evaluable", "R_dir"])
    out[[c for c in keep if c in out.columns]].to_csv(
        os.path.join(LABELED, f"{name}_final.csv"), index=False, encoding="utf-8-sig")

    dist = out["Class"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    n = len(out)
    print(f"[{name}] {stock} | {n:,}건 (라벨실패 제외 {n_err}) | "
          f"미평가(데이터부족) {int((~out['evaluable']).sum())}")
    print("  Class " + " ".join(f"{c}:{dist[c]:,}({dist[c]/n*100:.1f}%)" for c in [0, 1, 2, 3]))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="단일 파일명")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    for t in (FILES if a.all else [a.file]):
        assert t, "--file 또는 --all 필요"
        if not os.path.exists(os.path.join(LABELED, f"{t}_labeled.csv")):
            print(f"[{t}] {t}_labeled.csv 없음 — 건너뜀(이번 범위 아님)")
            continue
        run(t)
