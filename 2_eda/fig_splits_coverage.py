# unlabeled split별 'fig1 위에 추출 구간 표시' 그림 생성.
#   처음 highlight 방식 그대로: fig1(종가 + 빨강 상승사건/파랑 하락사건 + 누적패널)에
#   우리가 추출한 split 구간을 점선 박스로 얹는다. (fig_price_story(highlights=...) 재사용)
#   그림은 데이터 옆(splits/)에 split명으로 저장 → splits/README.md 와 짝(설계 설명서).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import exploration as E

SPL = os.path.join(E.ROOT, "data", "unlabeled", "splits")
BOX = "#1B7837"                                   # 추출 구간 박스 색(녹색 — 빨강/파랑 사건과 구분)
ROLE = {"tsla_train": "학습셋 (Fold A: TSLA→NVDA)",
        "nvda_eval": "평가셋 (Fold A: TSLA→NVDA)",
        "expert_control": "2차 보강·실력검정 (고수후보+대조군, 전기간)",
        "nvda_train": "학습셋 (역방향 Fold B · 미사용)",
        "tsla_eval": "평가셋 (역방향 Fold B · 미사용)"}


def window_hl(stock, d0, d1, n_comments, tag):
    """추출 구간 [d0,d1]의 상승/하락 사건일·누적변동을 fig1과 동일 기준으로 계산 → highlights dict."""
    cw0, cw1 = E.WINDOWS[stock]
    s, e = max(d0, cw0), min(d1, cw1)                       # WINDOWS 범위로 클램프
    p, mu, sig = E.load_prices(stock)
    p = p.sort_values("Date").reset_index(drop=True)
    p["RW"] = p["Close"].pct_change(E.H_DEFAULT) * 100
    up = mu * E.H_DEFAULT + E.K_DEFAULT * sig * np.sqrt(E.H_DEFAULT)
    dn = mu * E.H_DEFAULT - E.K_DEFAULT * sig * np.sqrt(E.H_DEFAULT)
    w = p[(p["Date"] >= s) & (p["Date"] <= e)]
    st = np.where(w["RW"] >= up, 1, np.where(w["RW"] <= dn, -1, 0))
    cum = (w["Close"].iloc[-1] / w["Close"].iloc[0] - 1) * 100 if len(w) else 0.0
    return dict(start=s, end=e, color=BOX, tag=tag, comments=int(n_comments),
                up=int((st == 1).sum()), dn=int((st == -1).sum()), cum=float(cum))


def make(split_name, stock, sub_csv, outdir):
    df = pd.read_csv(sub_csv, encoding="utf-8-sig")
    df = df[df["종목명"] == stock].copy()
    dt = pd.to_datetime(df["작성일"], format="ISO8601", utc=True, errors="coerce")
    df["d"] = dt.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    us = dt.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    d0, d1 = us.min(), us.max()
    title = f"{split_name} — {stock} 추출 구간  ·  {ROLE.get(split_name,'')}"

    if split_name == "expert_control":   # 사람 기반·전기간 → 추출 사용자 댓글을 그룹 막대로
        bars = [("고수후보", df[df["grp"] == "고수후보"].groupby("d").size(), "#8E24AA"),
                ("대조군",  df[df["grp"] == "대조군"].groupby("d").size(), "#607D8B")]
        E.fig_price_story(
            stock, bars=bars, save_as=f"{split_name}_{stock.lower()}.png", outdir=outdir,
            title_override=title,
            subtitle_override=("fig1(빨강=상승·파랑=하락 변동사건) 위에 추출 사용자(고수후보+대조군) "
                               f"일별 댓글 막대  ·  총 {len(df):,}건 · {d0}~{d1}"))
    else:                                # 연속 구간 → 추출구간 박스
        hl = window_hl(stock, d0, d1, len(df), f"{split_name} ({stock})")
        E.fig_price_story(
            stock, highlights=[hl], save_as=f"{split_name}.png", outdir=outdir,
            title_override=title,
            subtitle_override=("fig1(빨강=상승·파랑=하락 변동사건) 위에 추출 구간(녹색 점선) 표시  ·  "
                               f"댓글 {len(df):,}건 · {d0}~{d1}"))


JOBS = [
    ("tsla_train",     ["TSLA"],         os.path.join(SPL, "tsla_train.csv"),     SPL),
    ("nvda_eval",      ["NVDA"],         os.path.join(SPL, "nvda_eval.csv"),      SPL),
    ("expert_control", ["TSLA", "NVDA"], os.path.join(SPL, "expert_control.csv"), SPL),
    ("nvda_train",     ["NVDA"],         os.path.join(SPL, "역방향_미사용", "nvda_train.csv"),
     os.path.join(SPL, "역방향_미사용")),
    ("tsla_eval",      ["TSLA"],         os.path.join(SPL, "역방향_미사용", "tsla_eval.csv"),
     os.path.join(SPL, "역방향_미사용")),
]

if __name__ == "__main__":
    print("split별 fig1+추출구간 그림 생성 → data/unlabeled/splits/")
    for name, stocks, csv, outdir in JOBS:
        for stk in stocks:
            make(name, stk, csv, outdir)
