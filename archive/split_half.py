# split-half 지속성 검정 — "잘하는 사람이 계속 잘하나?"
#   유저 예측을 시간순 전/후반 분할 → 각 반의 '종목기저 대비 초과 적중률' 계산
#   → 전반 초과 ↔ 후반 초과 상관. 양의 상관 = 지속적 실력(타이밍/종목효과는 기저로 통제).
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
from exploration import UP, NEUTRAL, FIGDIR, _save, ROOT

DATA = os.path.join(ROOT, "data", "labeled")
MIN_PRED = 10      # 반으로 갈라 각 ≥5 되도록


def main():
    df = pd.concat([pd.read_csv(os.path.join(DATA, f"{f}_final.csv"), encoding="utf-8-sig")
                    for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
    pred = df[df["Class"].isin([1, 2, 3])].copy()
    pred["hit"] = pred["Class"].isin([2, 3]).astype(int)
    pred["p"] = pred["종목명"].map(pred.groupby("종목명")["hit"].mean())
    pred["dt"] = pd.to_datetime(pred["작성일"])

    rows = []
    for uid, gp in pred.groupby("닉네임_ID"):
        if len(gp) < MIN_PRED:
            continue
        gp = gp.sort_values("dt")
        h = len(gp) // 2
        a, b = gp.iloc[:h], gp.iloc[h:]
        # 종목기저 대비 초과 적중률 (시장효과 통제)
        ex_a = a["hit"].mean() - a["p"].mean()
        ex_b = b["hit"].mean() - b["p"].mean()
        rows.append((uid, ex_a, ex_b))
    d = pd.DataFrame(rows, columns=["uid", "전반_초과", "후반_초과"])
    print(f"분석 대상: 예측 {MIN_PRED}건 이상 유저 {len(d)}명")

    r, p = stats.pearsonr(d["전반_초과"], d["후반_초과"])
    rho, prho = stats.spearmanr(d["전반_초과"], d["후반_초과"])
    print(f"전반 초과적중 ↔ 후반 초과적중:")
    print(f"  Pearson r = {r:.3f} (p={p:.3f}) / Spearman ρ = {rho:.3f} (p={prho:.3f})")
    print("  → 양의 유의 상관이면 **지속적 실력**(전반 잘한 사람이 후반도). 0이면 일회성 운/타이밍.")

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(d["전반_초과"]*100, d["후반_초과"]*100, s=40, alpha=0.7, color=UP, edgecolor="white")
    ax.axhline(0, color=NEUTRAL, lw=0.8); ax.axvline(0, color=NEUTRAL, lw=0.8)
    # 추세선
    z = np.polyfit(d["전반_초과"]*100, d["후반_초과"]*100, 1)
    xs = np.array([d["전반_초과"].min()*100, d["전반_초과"].max()*100])
    ax.plot(xs, z[0]*xs + z[1], color="#333", ls="--", lw=1.5)
    ax.set_xlabel("전반 초과 적중률 (기저 대비, %)"); ax.set_ylabel("후반 초과 적중률 (%)")
    ax.set_title(f"split-half 지속성 — r={r:.2f}(p={p:.3f})\n우상향=지속적 실력 / 무상관=일회성")
    _save(fig, "fig12_split_half.png")


if __name__ == "__main__":
    main()
