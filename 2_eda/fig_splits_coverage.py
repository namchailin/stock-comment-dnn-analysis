# unlabeled split별 '종가 그래프 위 구간' 커버리지 그림 생성.
#   설계 의도(fig1)를 유지: 빨강=상승사건 / 파랑=하락사건(변동 판정선 μh±kσ√h 초과 구간)을
#   배경에 깔고, 그 위에 split의 추출구간(점선 박스) + 일별 댓글 밀도(회색 영역)를 겹쳐
#   "이 split이 어떤 변동구간에서 뽑혔나"가 한눈에 보이게 한다.
#   그림은 데이터 옆(splits/)에 저장 → splits/README.md 와 짝(설계 설명서).
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import exploration as E

SPL = os.path.join(E.ROOT, "data", "unlabeled", "splits")
DENS = "#607D8B"                              # 댓글 밀도(회색 계열) — 빨강/파랑 사건과 구분
ROLE = {"tsla_train": "학습셋 (Fold A: TSLA→NVDA)",
        "nvda_eval": "평가셋 (Fold A: TSLA→NVDA)",
        "expert_control": "2차 보강·실력검정 (고수후보+대조군, 전기간)",
        "nvda_train": "학습셋 (역방향 Fold B · 미사용)",
        "tsla_eval": "평가셋 (역방향 Fold B · 미사용)"}


def _events(stock):
    """fig1과 동일 로직: 전 구간에서 상승(+1)/하락(-1) 변동 사건 추출 → axvspan용 리스트."""
    p, mu, sig = E.load_prices(stock)
    p = p.reset_index(drop=True)
    rw = (p["Close"].pct_change(E.H_DEFAULT) * 100).to_numpy()
    up = mu * E.H_DEFAULT + E.K_DEFAULT * sig * np.sqrt(E.H_DEFAULT)
    dn = mu * E.H_DEFAULT - E.K_DEFAULT * sig * np.sqrt(E.H_DEFAULT)
    state = np.where(rw >= up, 1, np.where(rw <= dn, -1, 0))
    d = pd.to_datetime(p["Date"])
    out = []
    for sign, i, j in E._episodes(state, gap=2):
        out.append((sign, d.iloc[max(0, i - E.H_DEFAULT)], d.iloc[j]))
    return p, d, out


def coverage_fig(split_name, stock, sub_csv, out_png):
    df = pd.read_csv(sub_csv, encoding="utf-8-sig")
    df = df[df["종목명"] == stock].copy()
    dt = pd.to_datetime(df["작성일"], format="ISO8601", utc=True, errors="coerce")
    df["d"] = dt.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    p, pdate, events = _events(stock)
    daily = df.groupby("d").size().reindex(pd.to_datetime(p["Date"]), fill_value=0)
    dens = daily.rolling(5, center=True, min_periods=1).mean()        # 5일 이동평균 밀도

    x = pdate.to_numpy(); close = p["Close"].to_numpy(); y = np.asarray(dens)
    d0, d1 = df["d"].min(), df["d"].max()
    half = pd.Timedelta(hours=12)

    fig, ax = plt.subplots(figsize=(13, 5.2))
    # 배경: 빨강/파랑 변동 사건 (설계 의도)
    for sign, ds, de in events:
        ax.axvspan(ds - half, de + half, color=(E.UP if sign > 0 else E.DOWN),
                   alpha=0.16, lw=0, zorder=1)
    ax.plot(x, close, color="black", lw=1.7, zorder=4, label="종가")
    # 추출 구간 경계(점선 박스)
    ax.axvspan(d0, d1, facecolor="none", edgecolor="0.25", lw=1.5, ls="--", zorder=5)
    ax.set_ylabel("종가(USD)", fontsize=11); ax.set_xlabel("날짜", fontsize=11)
    ax.grid(True, axis="both")                                        # 그리드 유지(흐린 회색)

    ax2 = ax.twinx()                                                  # 댓글 밀도(회색 영역)
    ax2.fill_between(x, y, color=DENS, alpha=0.35, lw=0, zorder=2)
    ax2.plot(x, y, color=DENS, lw=1.2, zorder=3)
    ax2.set_ylabel("일별 댓글 수 (5일 평균)", fontsize=10)
    ax2.set_ylim(0, max(1, y.max() * 1.2)); ax2.grid(False)

    leg = [Line2D([], [], color="black", lw=1.7, label="종가"),
           Patch(color=E.UP, alpha=0.5, label="상승 사건"),
           Patch(color=E.DOWN, alpha=0.5, label="하락 사건"),
           Patch(color=DENS, alpha=0.5, label=f"댓글 밀도(총 {len(df):,}건)"),
           Line2D([], [], color="0.25", lw=1.5, ls="--", label="추출 구간")]
    ax.legend(handles=leg, loc="upper left", fontsize=9.5, ncol=2, framealpha=0.93)
    fig.suptitle(f"{split_name}  —  {stock} · {ROLE.get(split_name, '')}",
                 fontsize=14, fontweight="bold", y=0.99)
    ax.set_title(f"댓글 {len(df):,}건 · 추출구간 {d0.date()} ~ {d1.date()}  "
                 f"(빨강=상승·파랑=하락 변동사건 / 점선=추출구간 / 회색=댓글 밀도)",
                 fontsize=9.5, color="0.4", pad=7)
    fig.autofmt_xdate()
    fig.savefig(out_png, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  저장: {os.path.relpath(out_png, E.ROOT)}  ({stock} {len(df):,}건)")


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
