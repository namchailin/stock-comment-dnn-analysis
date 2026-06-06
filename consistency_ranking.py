# 유저 일관성/실력 순위 — round-2(expert_control)만 = out-of-sample(편향 없음).
#   순위 기준: '예측 방향으로 실현된 누적변동(R_dir)'의 시장대비 초과 + 볼륨보정(EB).
#     → 적중 여부(0/1)가 아니라 '얼마나 크게 맞췄나'를 점수화. 큰 변동 적중 = 가산점.
#   적중률(전체/TSLA/NVDA)은 참고 컬럼. 예측 10회↑.
#   출력: data/labeled/user_consistency.csv
import os
import numpy as np
import pandas as pd
from exploration import ROOT

LAB = os.path.join(ROOT, "data", "labeled")
EB_K, MIN_PRED = 20, 5

r2 = pd.read_csv(os.path.join(LAB, "expert_control_final.csv"), encoding="utf-8-sig")
p = r2[(r2["Class"].isin([1, 2, 3])) & (r2["R_dir"].notna())].copy()
p["hit"] = p["Class"].isin([2, 3]).astype(int)
# 예측 방향으로 실현된 누적변동(%): 상승예측이면 +R_dir, 하락예측이면 -R_dir (맞으면 +, 틀리면 -)
p["실현변동"] = np.where(p["예측_방향"] == "상승", p["R_dir"], -p["R_dir"])
p["기저변동"] = p.groupby("종목명")["실현변동"].transform("mean")   # 같은 종목 평균(시장통제)


def by_stock(stk):
    s = p[p["종목명"] == stk].groupby("닉네임_ID")["hit"].agg(["size", "sum"])
    return s.rename(columns={"size": f"{stk}_예측", "sum": f"{stk}_적중"})


g = p.groupby("닉네임_ID").agg(grp=("grp", "first"), 예측수=("hit", "size"),
                             적중수=("hit", "sum"), 평균실현변동=("실현변동", "mean"),
                             초과합=("실현변동", lambda s: (s - p.loc[s.index, "기저변동"]).sum()))
g = g.join(by_stock("TSLA")).join(by_stock("NVDA")).fillna(0)
for s in ["TSLA", "NVDA"]:
    g[f"{s}_적중률"] = np.where(g[f"{s}_예측"] > 0,
                              (g[f"{s}_적중"] / g[f"{s}_예측"] * 100).round(0), np.nan)
g["적중률"] = (g["적중수"] / g["예측수"] * 100).round(0)
g["평균실현변동"] = g["평균실현변동"].round(1)
g["실력점수"] = (g["초과합"] / (g["예측수"] + EB_K)).round(2)   # 시장대비 변동초과·볼륨보정

g = g[g["예측수"] >= MIN_PRED].drop(columns="초과합").sort_values("실력점수", ascending=False)
g.insert(0, "순위", range(1, len(g) + 1))
cols = ["순위", "grp", "예측수", "TSLA_예측", "NVDA_예측", "적중률", "TSLA_적중률",
        "NVDA_적중률", "평균실현변동", "실력점수"]
g[cols].to_csv(os.path.join(LAB, "user_consistency.csv"), encoding="utf-8-sig")

print(f"저장: data/labeled/user_consistency.csv (round-2, 예측{MIN_PRED}회↑ {len(g)}명)")
print("적중률 = 전체적중률(TSLA적중률 / NVDA적중률)  ·  실력점수 = 예측방향 누적변동의 시장대비 초과(볼륨보정)\n")
print("=== 실력 TOP 5 (큰 변동 적중에 가산점) ===")
for u, r in g.head(5).iterrows():
    t = f"{int(r.TSLA_적중률)}%" if r.TSLA_예측 > 0 else "-"
    n = f"{int(r.NVDA_적중률)}%" if r.NVDA_예측 > 0 else "-"
    print(f"  {r.순위}위 {u[:10]} [{r.grp}] 예측{int(r.예측수)}(TSLA {int(r.TSLA_예측)}/NVDA {int(r.NVDA_예측)}) | "
          f"적중률 {int(r.적중률)}%({t}/{n}) | 평균실현변동 {r.평균실현변동:+.1f}% | 실력점수 {r.실력점수}")
