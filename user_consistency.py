# 유저 일관성 검정 — "운이면 꾸준한 고수가 없다"
#   예측을 여러 번 한 유저들의 적중률이, 모두 동일(운)일 때보다 더 흩어지나(과분산)?
#   시장효과 통제: 기대 적중은 '종목별 기저 적중률'로 계산.
#   과분산 검정(sum z^2 ~ chi2) + 시뮬레이션 + 그림.
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from exploration import UP, NEUTRAL, FIGDIR, _save, ROOT   # 1차 EDA 폰트·색 재사용

DATA = os.path.join(ROOT, "data", "labeled")
MIN_PRED = 5      # 예측 N건 이상 유저만(통계 의미)


def main():
    df = pd.concat([pd.read_csv(os.path.join(DATA, f"{f}_final.csv"), encoding="utf-8-sig")
                    for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
    pred = df[df["Class"].isin([1, 2, 3])].copy()       # 결정적 예측만
    pred["hit"] = pred["Class"].isin([2, 3]).astype(int)
    # 종목별 기저 적중률(시장효과 통제)
    base = pred.groupby("종목명")["hit"].mean()
    pred["p"] = pred["종목명"].map(base)
    print("종목별 기저 적중률:", {k: round(v, 3) for k, v in base.items()})

    g = pred.groupby("닉네임_ID").agg(k=("hit", "size"), hits=("hit", "sum"),
                                    exp=("p", "sum"), var=("p", lambda s: (s*(1-s)).sum()))
    g = g[g["k"] >= MIN_PRED]
    print(f"분석 대상: 예측 {MIN_PRED}건 이상 유저 {len(g)}명")

    # 과분산 검정: z=(hits-exp)/sqrt(var) ~ N(0,1) if 운. sum z^2 ~ chi2(df=n)
    z = (g["hits"] - g["exp"]) / np.sqrt(g["var"])
    chi2 = (z**2).sum(); dfree = len(g)
    p_over = stats.chi2.sf(chi2, dfree)        # 관측 분산 > 기대(운)일 확률
    print(f"\n과분산 검정: sum z^2 = {chi2:.1f} (df={dfree}, 기대 ≈ {dfree})")
    print(f"  분산비(관측/기대) = {chi2/dfree:.2f}")
    print(f"  p-value(과분산) = {p_over:.4f} → " +
          ("**유의: 일부 유저가 꾸준히 잘함 = 실력 존재**" if p_over < 0.05
           else "유의X: 유저별 적중률이 우연 흩어짐과 같음 = **꾸준한 고수 없음(운)**"))

    # 시뮬레이션: 운일 때 '적중률 60%+ 꾸준한 유저' 몇 명? vs 관측
    obs_hi = int((g["hits"]/g["k"] >= 0.6).sum())
    rng = np.random.default_rng(42)
    sims = []
    pe = pred.groupby("닉네임_ID")["p"].apply(list)
    users = g.index
    for _ in range(2000):
        cnt = 0
        for u in users:
            ps = np.array(pe[u])
            cnt += (rng.random(len(ps)) < ps).mean() >= 0.6
        sims.append(cnt)
    sims = np.array(sims)
    p_hi = (sims >= obs_hi).mean()
    print(f"\n'적중률 60%+ 유저' 관측 {obs_hi}명 vs 우연 평균 {sims.mean():.1f}명 (95%상한 {np.quantile(sims,0.95):.0f}) | p={p_hi:.3f}")

    # 그림: 유저 적중률 vs 종목 기저선
    fig, ax = plt.subplots(figsize=(8, 5))
    rate = g["hits"]/g["k"]
    ax.scatter(g["k"], rate*100, s=30, alpha=0.6, color=UP, edgecolor="white")
    for stk, b in base.items():
        ax.axhline(b*100, ls="--", color=NEUTRAL, lw=1)
        ax.text(g["k"].max(), b*100, f" {stk} 기저 {b*100:.0f}%", va="center", fontsize=9)
    ax.set_xlabel(f"유저의 예측 횟수 (≥{MIN_PRED})"); ax.set_ylabel("그 유저의 적중률 (%)")
    ax.set_title(f"유저별 적중률 — 기저선 주변 흩어짐(운) vs 위쪽 쏠림(실력)\n과분산 p={p_over:.3f}")
    _save(fig, "fig11_user_consistency_test.png")


if __name__ == "__main__":
    main()
