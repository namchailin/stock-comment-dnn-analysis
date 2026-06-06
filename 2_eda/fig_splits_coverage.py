# unlabeled split별 '종가 그래프 위 구간' 커버리지 그림 생성.
#   현재 data/unlabeled/splits/ 에 있는 모든 split에 대해, 그 댓글이 종가 차트의
#   어느 구간/시점에 분포하는지 한눈에 보이게 그린다. 그림은 데이터 옆(splits/)에 저장 →
#   누가 봐도 추출 설계 설명서처럼 읽히도록(= splits/README.md 와 짝).
#   표현: 종가 라인 + 추출구간 음영 + 일별 댓글 밀도(5일 이동평균 영역). 막대 난립 회피.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exploration as E

SPL = os.path.join(E.ROOT, "data", "unlabeled", "splits")
ROLE = {"tsla_train": "학습셋 (Fold A: TSLA→NVDA)",
        "nvda_eval": "평가셋 (Fold A: TSLA→NVDA)",
        "expert_control": "2차 보강·실력검정 (고수후보+대조군, 전기간)",
        "nvda_train": "학습셋 (역방향 Fold B · 미사용)",
        "tsla_eval": "평가셋 (역방향 Fold B · 미사용)"}


def coverage_fig(split_name, stock, sub_csv, out_png):
    df = pd.read_csv(sub_csv, encoding="utf-8-sig")
    df = df[df["종목명"] == stock].copy()
    dt = pd.to_datetime(df["작성일"], format="ISO8601", utc=True, errors="coerce")
    df["d"] = dt.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    p, _, _ = E.load_prices(stock)
    p = p.copy(); p["d"] = pd.to_datetime(p["Date"])
    daily = df.groupby("d").size().reindex(p["d"], fill_value=0)
    dens = daily.rolling(5, center=True, min_periods=1).mean()      # 5일 이동평균 → 매끄러운 밀도

    x = p["d"].to_numpy(); close = p["Close"].to_numpy(); y = np.asarray(dens)
    d0, d1 = df["d"].min(), df["d"].max()
    col = E.STOCK_C[stock]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x, close, color="black", lw=1.7, zorder=3, label="종가")
    ax.axvspan(d0, d1, color=col, alpha=0.07, zorder=1)            # 추출 구간 음영(은은하게)
    ax.set_ylabel("종가(USD)", fontsize=11); ax.set_xlabel("날짜", fontsize=11)
    ax.grid(False)
    ax2 = ax.twinx()
    ax2.fill_between(x, y, color=col, alpha=0.45, lw=0, zorder=2,
                     label=f"일별 댓글 밀도 (5일 평균 · 총 {len(df):,}건)")
    ax2.set_ylabel("일별 댓글 수 (5일 평균)", fontsize=10)
    ax2.set_ylim(0, max(1, y.max() * 1.15)); ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, framealpha=0.92)
    fig.suptitle(f"{split_name}  —  {stock} · {ROLE.get(split_name, '')}",
                 fontsize=14, fontweight="bold", y=0.98)
    ax.set_title(f"댓글 {len(df):,}건 · 추출구간 {d0.date()} ~ {d1.date()} "
                 f"(음영=구간, 영역=일별 댓글 밀도)", fontsize=10, color="0.4", pad=7)
    fig.autofmt_xdate()
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  저장: {os.path.relpath(out_png, E.ROOT)}  ({stock} {len(df):,}건)")


# split → (stock들, csv경로, 출력폴더). 파일명 stock 접미사는 소문자로 통일.
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
    print("unlabeled split 커버리지 그림 생성 → data/unlabeled/splits/")
    for name, stocks, csv, outdir in JOBS:
        for stk in stocks:
            suffix = f"_{stk.lower()}" if len(stocks) > 1 else ""
            coverage_fig(name, stk, csv, os.path.join(outdir, f"{name}{suffix}.png"))
