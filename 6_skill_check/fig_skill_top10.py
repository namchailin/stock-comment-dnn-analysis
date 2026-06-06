# 실력순위 TOP10 표 → 그림(png). consistency_ranking과 동일 로직(round-2, MIN_PRED=5).
#   그룹(고수후보/대조군) 색으로 'TOP10 5:5 → 사전 선정 우위 없음'을 시각화.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import exploration as E

LAB = os.path.join(E.ROOT, "data", "labeled")
EB_K, MIN_PRED = 20, 5
r2 = pd.read_csv(os.path.join(LAB, "expert_control_final.csv"), encoding="utf-8-sig")
p = r2[(r2["Class"].isin([1, 2, 3])) & (r2["R_dir"].notna())].copy()
p["hit"] = p["Class"].isin([2, 3]).astype(int)
p["실현변동"] = np.where(p["예측_방향"] == "상승", p["R_dir"], -p["R_dir"])
p["기저변동"] = p.groupby("종목명")["실현변동"].transform("mean")


def by_stock(stk):
    s = p[p["종목명"] == stk].groupby("닉네임_ID")["hit"].agg(["size", "sum"])
    return s.rename(columns={"size": f"{stk}_예측", "sum": f"{stk}_적중"})


g = p.groupby("닉네임_ID").agg(grp=("grp", "first"), 예측수=("hit", "size"),
                             적중수=("hit", "sum"),
                             초과합=("실현변동", lambda s: (s - p.loc[s.index, "기저변동"]).sum()))
g = g.join(by_stock("TSLA")).join(by_stock("NVDA")).fillna(0)
for s in ["TSLA", "NVDA"]:
    g[f"{s}_적중률"] = np.where(g[f"{s}_예측"] > 0, (g[f"{s}_적중"] / g[f"{s}_예측"] * 100).round(0), np.nan)
g["적중률"] = (g["적중수"] / g["예측수"] * 100).round(0)
g["실력점수"] = (g["초과합"] / (g["예측수"] + EB_K)).round(3)
g = g[g["예측수"] >= MIN_PRED].sort_values("실력점수", ascending=False).head(10).reset_index()

GO, CT = "#F3D9E2", "#D6E4F0"            # 고수후보(분홍) / 대조군(파랑) 행 색
rows = []
colors = []
for i, r in g.iterrows():
    t = f"{int(r.TSLA_적중률)}" if r.TSLA_예측 > 0 else "-"
    n = f"{int(r.NVDA_적중률)}" if r.NVDA_예측 > 0 else "-"
    rows.append([f"{i+1}", r["닉네임_ID"][:10], r["grp"],
                 f"{int(r.예측수)}/{int(r.적중수)}",
                 f"{int(r.적중률)}% ({t}/{n})", f"{r.실력점수:.3f}"])
    c = GO if r["grp"] == "고수후보" else CT
    colors.append(["white", "white", c, "white", "white", "white"])

cols = ["순위", "닉네임_ID", "그룹", "예측/적중", "적중률 (전체·T/N)", "실력점수"]
nGO = int((g["grp"] == "고수후보").sum()); nCT = int((g["grp"] == "대조군").sum())

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axis("off")
fig.subplots_adjust(top=0.80, bottom=0.08)       # 상단 제목 영역 확보(겹침 방지)
tbl = ax.table(cellText=rows, colLabels=cols, cellColours=colors,
               cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.7)
for j in range(len(cols)):                       # 헤더 스타일
    c = tbl[0, j]; c.set_facecolor("#37474F"); c.set_text_props(color="white", fontweight="bold")
fig.suptitle("유저 실력순위 TOP 10  (round-2 out-of-sample · 예측 5회↑ 147명 중)",
             fontsize=14, fontweight="bold", y=0.965)
fig.text(0.5, 0.895, f"실력점수 = 예측방향 누적변동의 시장대비 초과(볼륨보정)   ·   "
         f"TOP10 = 고수후보 {nGO} : 대조군 {nCT}  → 사전 선정 집단의 우위 없음",
         ha="center", fontsize=10.5, color="0.30")
fig.text(0.5, 0.03, "적중률 = 전체%(TSLA%/NVDA%)  ·  분홍=고수후보, 파랑=대조군",
         ha="center", fontsize=9.5, color="0.45")

out = os.path.join(E.ROOT, "5_train2_retrain", "skill_top10.png")
fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
print(f"저장: {os.path.relpath(out, E.ROOT)}  (고수후보 {nGO} : 대조군 {nCT})")
