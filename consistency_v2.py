# 사용자 일관성 v2 — 검정력 보존 + 자기상관 통제 (LOO 폐기)
#   사전등록 주 분석: 군집 순열검정.
#     - 기대적중 p_i = (종목×주차) 평균 → '같은 주에 몰린 예측이 같이 맞는' 자기상관을 baseline이 흡수
#     - 통계량 T = Σ_u z_u²  (z_u = 유저 잔차합/표준편차) — 데이터 안 쪼갬(검정력 보존)
#     - 귀무: 유저 라벨만 순열(예측의 주차 baseline은 고정) → T 분포. 실제 T가 크면 '주차 통제 후에도 유저차 존재'
#   보조: 시간순 split-half 효과크기 + 부트스트랩 CI (추정 프레이밍)
import os
import numpy as np
import pandas as pd
from scipy import stats
from exploration import ROOT

DATA = os.path.join(ROOT, "data", "labeled")


def load():
    df = pd.concat([pd.read_csv(os.path.join(DATA, f"{f}_final.csv"), encoding="utf-8-sig")
                    for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
    p = df[df["Class"].isin([1, 2, 3])].copy()
    p["hit"] = p["Class"].isin([2, 3]).astype(int)
    p["dt"] = pd.to_datetime(p["작성일"]).dt.tz_localize(None)
    p["week"] = p["dt"].dt.to_period("W").astype(str)
    p["block"] = p["종목명"] + "|" + p["week"]            # 종목×주차
    p["p"] = p.groupby("block")["hit"].transform("mean")  # 자기상관 흡수 baseline(전체로 추정)
    return p


def cluster_perm(p, min_pred, B=5000, seed=42):
    n = p.groupby("닉네임_ID")["hit"].transform("size")
    d = p[n >= min_pred].copy()
    users = d["닉네임_ID"].to_numpy()
    hit = d["hit"].to_numpy(float); pp = d["p"].to_numpy(float)

    def T(uarr):
        t = pd.DataFrame({"u": uarr, "o": hit, "e": pp, "v": pp*(1-pp)})
        g = t.groupby("u").sum()
        g = g[g["v"] > 0]
        return float((((g["o"]-g["e"])**2)/g["v"]).sum()), len(g)

    T_obs, nu = T(users)
    rng = np.random.default_rng(seed)
    null = np.array([T(rng.permutation(users))[0] for _ in range(B)])
    pval = (null >= T_obs).mean()
    return dict(users=nu, n=len(d), T=T_obs, ratio=T_obs/nu,
               null_mean=null.mean(), p=pval, multiweek=int((pp*(1-pp) > 0).sum()))


def split_half_es(p, min_pred, nboot=3000, seed=7):
    rows = []
    for _, g in p.groupby("닉네임_ID"):
        if len(g) < min_pred:
            continue
        g = g.sort_values("dt"); h = len(g)//2
        a, b = g.iloc[:h], g.iloc[h:]
        rows.append((a["hit"].mean()-a["p"].mean(), b["hit"].mean()-b["p"].mean()))
    d = np.array(rows)
    r = stats.pearsonr(d[:, 0], d[:, 1])[0]
    rng = np.random.default_rng(seed)
    bs = [stats.pearsonr(*d[rng.integers(0, len(d), len(d))].T)[0] for _ in range(nboot)]
    return len(d), r, np.percentile(bs, [2.5, 97.5])


def main():
    p = load()
    print(f"예측 {len(p):,}건 / 유저 {p['닉네임_ID'].nunique():,} / 종목×주차 블록 {p['block'].nunique()}")
    print("종목 기저:", {k: round(v, 3) for k, v in p.groupby('종목명')['hit'].mean().items()})

    print("\n===== [주 분석·사전등록] 군집 순열검정 (주차×종목 baseline) =====")
    for m in [5, 10]:
        r = cluster_perm(p, m)
        sig = "**유의: 주차 통제 후에도 유저별 지속차 존재(실력)**" if r["p"] < 0.05 else "유의X: 유저차가 운(주차효과)과 무차이"
        print(f"  ≥{m}건: 유저 {r['users']}명 / 예측 {r['n']} (정보유효 {r['multiweek']}) | "
              f"T={r['T']:.0f} (기대~{r['users']}, 분산비 {r['ratio']:.2f}) | p={r['p']:.4f} → {sig}")

    print("\n===== [보조] 시간순 split-half 효과크기 + 95%CI =====")
    for m in [6, 8, 10]:
        n, r, ci = split_half_es(p, m)
        print(f"  ≥{m}건: n={n} | r={r:+.3f} [95%CI {ci[0]:+.2f},{ci[1]:+.2f}]")
    print("\n[판정은 주 분석(군집 순열) 기준. split-half는 효과크기 참고. 결론 보류 가능.]")


if __name__ == "__main__":
    main()
