# 2차 라벨링 커버리지 — fig1 스타일로 '고수후보+대조군'이 6개월 전체에 퍼진 분포 표시.
#   (1차 fig1_highlight 4개는 그대로 두고, 이번 라운드 대상을 복제 스타일로 시각화)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import exploration as E

fin = pd.concat([pd.read_csv(os.path.join(E.ROOT, "data", "labeled", f"{f}_final.csv"),
                             encoding="utf-8-sig")
                 for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
g = fin.groupby("닉네임_ID")["Class"]
hit = g.apply(lambda s: ((s == 2) | (s == 3)).sum())
npred = g.apply(lambda s: s.isin([1, 2, 3]).sum())
cand = set(hit[hit >= 2].index)                                   # 고수후보 130
ctrl_all = npred[(npred >= 2) & (~npred.index.isin(cand))].index  # 대조군 후보 323
ctrl = set(np.random.default_rng(0).choice(list(ctrl_all), 130, replace=False))

df = E.load_comments()
df["grp"] = np.where(df["닉네임_ID"].isin(cand), "고수후보",
                     np.where(df["닉네임_ID"].isin(ctrl), "대조군", None))
sub = df[df["grp"].notna()].copy()
sub["day"] = pd.to_datetime(sub["us_date"])

for stock in ["TSLA", "NVDA"]:
    p, _, _ = E.load_prices(stock)
    cw0, cw1 = E.WINDOWS[stock]
    pw = p[(p["Date"] >= cw0) & (p["Date"] <= cw1)].copy()
    pw["d"] = pd.to_datetime(pw["Date"])
    s = sub[sub["종목명"] == stock]
    gc = (s.groupby(["day", "grp"]).size().unstack(fill_value=0)
          .reindex(pw["d"], fill_value=0))
    go = gc.get("고수후보", pd.Series(0, index=pw["d"]))
    ct = gc.get("대조군", pd.Series(0, index=pw["d"]))

    xd = pw["d"].to_numpy(); yc = pw["Close"].to_numpy()
    go_a = np.asarray(go); ct_a = np.asarray(ct)
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(xd, yc, color="black", lw=1.8, zorder=3, label="종가")
    ax.set_ylabel("종가(USD)", fontsize=12)
    ax.set_xlabel("날짜", fontsize=12)
    ax2 = ax.twinx()
    ax2.bar(xd, go_a, width=1.0, color=E.UP, alpha=0.55, label=f"고수후보({len(cand)}명)")
    ax2.bar(xd, ct_a, bottom=go_a, width=1.0, color=E.NEUTRAL, alpha=0.55,
            label=f"대조군({len(ctrl)}명)")
    ax2.set_ylabel("일별 라벨링 대상 댓글 수", fontsize=11)
    n_total = int((go.sum() + ct.sum()))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, ncol=3)
    fig.suptitle(f"{stock} — 2차 라벨링 커버리지 (held-out 일관성 검증)",
                 fontsize=16, fontweight="bold", y=0.98)
    ax.set_title(f"고수후보·대조군의 6개월 전체 댓글 {n_total:,}건  "
                 f"— 1차의 좁은 윈도우(fig1)와 달리 전 기간에 분포 → 시점 다양성 확보",
                 fontsize=11, pad=8)
    fig.autofmt_xdate()
    E._save(fig, f"fig1_highlight_{stock}_2차.png")
