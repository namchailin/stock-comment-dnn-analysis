# 사용자 일관성 강화 검증 — 운 vs 진짜 실력 (결론 보류, 증거만 제시)
#   ① split-half: 전반 초과적중 ↔ 후반 (시장·종목효과는 '종목기저 대비 초과'로 통제)
#      여러 임계값 + 부트스트랩 95% CI → 검정력/불확실성 정직하게.
#   ② leave-one-out: 유저의 '나머지' 초과적중률이 '이' 예측 적중을 예측하나(로지스틱).
import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from exploration import ROOT

DATA = os.path.join(ROOT, "data", "labeled")


def load():
    df = pd.concat([pd.read_csv(os.path.join(DATA, f"{f}_final.csv"), encoding="utf-8-sig")
                    for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
    p = df[df["Class"].isin([1, 2, 3])].copy()
    p["hit"] = p["Class"].isin([2, 3]).astype(int)
    p["p"] = p["종목명"].map(p.groupby("종목명")["hit"].mean())   # 종목 기저(시장통제)
    p["dt"] = pd.to_datetime(p["작성일"])
    return p


def split_half(p, min_pred, nboot=2000, seed=42):
    rows = []
    for uid, g in p.groupby("닉네임_ID"):
        if len(g) < min_pred:
            continue
        g = g.sort_values("dt"); h = len(g) // 2
        a, b = g.iloc[:h], g.iloc[h:]
        rows.append((a["hit"].mean() - a["p"].mean(), b["hit"].mean() - b["p"].mean()))
    d = np.array(rows)
    if len(d) < 5:
        return None
    r = stats.pearsonr(d[:, 0], d[:, 1])[0]
    rng = np.random.default_rng(seed)
    boots = [stats.pearsonr(*d[rng.integers(0, len(d), len(d))].T)[0] for _ in range(nboot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return len(d), r, lo, hi


def leave_one_out(p, min_pred=3):
    g = p.groupby("닉네임_ID")
    p = p.join(g["hit"].transform("sum").rename("uhit"))
    p = p.join(g["p"].transform("sum").rename("up"))
    p = p.join(g["hit"].transform("size").rename("un"))
    p = p[p["un"] >= min_pred].copy()
    # 나머지(자기 제외) 초과 적중률
    p["loo"] = ((p["uhit"] - p["hit"]) - (p["up"] - p["p"])) / (p["un"] - 1)
    X = sm.add_constant(p[["loo", "p"]])
    m = sm.Logit(p["hit"], X).fit(disp=0)
    return len(p), m.params["loo"], m.pvalues["loo"]


def main():
    p = load()
    print(f"예측(결정적) 총 {len(p):,}건 / 유저 {p['닉네임_ID'].nunique():,}명")
    print("종목 기저적중:", {k: round(v, 3) for k, v in p.groupby('종목명')['hit'].mean().items()})

    print("\n=== ① split-half (전반↔후반 초과적중, 부트스트랩 95%CI) ===")
    for m in [6, 8, 10, 12]:
        res = split_half(p, m)
        if res:
            n, r, lo, hi = res
            sig = "유의(+)" if lo > 0 else ("유의(−)" if hi < 0 else "0 포함→불확정")
            print(f"  ≥{m}건: n={n:>3}명 | r={r:+.3f} [95%CI {lo:+.2f},{hi:+.2f}] → {sig}")

    print("\n=== ② leave-one-out (나머지 초과적중 → 이 예측 적중, 로지스틱) ===")
    for m in [3, 5]:
        n, coef, pv = leave_one_out(p, m)
        sig = "유의(+)=실력" if (pv < 0.05 and coef > 0) else "유의X=운"
        print(f"  ≥{m}건: n={n:,}예측 | 계수={coef:+.3f} p={pv:.3f} → {sig}")
    print("\n[해석 보류] 위 지표들이 일관되게 + 유의면 실력, 0 근처/불확정이면 '운 또는 검정력 부족'.")


if __name__ == "__main__":
    main()
