# C2/C3 보강 시각화 — 1차 vs enriched 클래스 건수 + 적중 출처(1차/고수후보/대조군)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import exploration as E

LAB = os.path.join(E.ROOT, "data", "labeled")
tt = pd.read_csv(os.path.join(LAB, "tsla_train_final.csv"), encoding="utf-8-sig")
ec = pd.read_csv(os.path.join(LAB, "expert_control_final.csv"), encoding="utf-8-sig")
en = pd.read_csv(os.path.join(LAB, "tsla_train_enriched_final.csv"), encoding="utf-8-sig")

NAMES = {0: "C0\n예측없음", 1: "C1\n실패", 2: "C2\n방향적중", 3: "C3\n날짜적중"}
d1 = tt["Class"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
de = en["Class"].value_counts().reindex([0, 1, 2, 3], fill_value=0)

# 적중(C2+C3) 출처 추적: 1차 / 2차-고수후보 / 2차-대조군
tt_ids = set(tt["commentId"])
grp = ec[ec["종목명"] == "TSLA"].set_index("commentId")["grp"].to_dict()
h = en[en["Class"].isin([2, 3])].copy()
h["출처"] = np.where(h["commentId"].isin(tt_ids), "1차",
                   h["commentId"].map(grp).map({"고수후보": "2차 고수후보", "대조군": "2차 대조군"}))
src = h["출처"].value_counts()
S1, Sg, Sc = src.get("1차", 0), src.get("2차 고수후보", 0), src.get("2차 대조군", 0)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6.3), gridspec_kw={"width_ratios": [1.4, 1]})

# ── 좌: 클래스별 건수 1차 vs enriched (log y) ──
x = np.arange(4); w = 0.38
b1 = axL.bar(x - w / 2, d1.values, w, color=E.NEUTRAL, label=f"1차만 ({len(tt):,}건)")
b2 = axL.bar(x + w / 2, de.values, w, color=E.UP, label=f"enriched ({len(en):,}건)")
axL.set_yscale("log")
axL.set_xticks(x); axL.set_xticklabels([NAMES[i] for i in range(4)], fontsize=10)
axL.set_ylabel("댓글 수 (log)", fontsize=11)
axL.set_title("클래스별 건수 — 희소클래스 C2·C3 보강 확인", fontsize=13, fontweight="bold", pad=8)
for xi, (a, b) in enumerate(zip(d1.values, de.values)):
    axL.annotate(f"{a:,}", (xi - w / 2, a), ha="center", va="bottom", fontsize=8.5, color="0.35")
    inc = (b - a) / a * 100 if a else 0
    axL.annotate(f"{b:,}\n(+{inc:.0f}%)", (xi + w / 2, b), ha="center", va="bottom",
                 fontsize=8.5, color=E.UP, fontweight="bold")
axL.legend(fontsize=10, loc="upper right")
axL.margins(y=0.18)

# ── 우: 적중(C2+C3) 출처별 누적 ──
tot = S1 + Sg + Sc
axR.bar(0, S1, 0.55, color=E.NEUTRAL, label=f"1차 {S1}")
axR.bar(0, Sg, 0.55, bottom=S1, color=E.UP, label=f"2차 고수후보 +{Sg}")
axR.bar(0, Sc, 0.55, bottom=S1 + Sg, color=E.DOWN, label=f"2차 대조군 +{Sc}")
for y0, v, c in [(S1 / 2, S1, "white"), (S1 + Sg / 2, Sg, "white"), (S1 + Sg + Sc / 2, Sc, "white")]:
    axR.annotate(f"{v}", (0, y0), ha="center", va="center", fontsize=11, fontweight="bold", color=c)
axR.annotate(f"총 {tot}건", (0, tot), ha="center", va="bottom", fontsize=12, fontweight="bold")
axR.set_xlim(-0.7, 0.7); axR.set_xticks([]); axR.set_ylabel("적중(C2+C3) 댓글 수", fontsize=11)
axR.set_title(f"적중 표본 출처\n643 → {tot}건 (+{(tot-S1)/S1*100:.0f}%)", fontsize=13, fontweight="bold", pad=8)
axR.legend(fontsize=10, loc="upper left")
axR.margins(y=0.12)

# 제목·소제목을 그래프 위로 띄우되, 제목↔소제목 간격은 좁게
fig.subplots_adjust(top=0.80)
fig.suptitle("2차 라벨링으로 희소클래스(C2·C3) 학습표본 보강 — TSLA 학습셋",
             fontsize=15.5, fontweight="bold", y=0.99)
fig.text(0.5, 0.918, "비중은 ~3.5%로 유지(C0도 함께 증가) · 보강의 본질은 적중 '절대량'↑ → 러닝커브 데이터부족 완화",
         ha="center", fontsize=9.5, color="0.45")
_out = os.path.join(E.ROOT, "results", "fig_enrich_c2c3.png")
fig.savefig(_out, dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"저장: {_out}")
print(f"  1차 Class {dict(d1)} | enriched {dict(de)}")
print(f"  적중 출처: 1차 {S1} / 2차고수후보 {Sg} / 2차대조군 {Sc} = {tot}")

# ── 최종 학습셋(enriched) 커버리지: fig1 위에 학습 댓글 일별 분포(1차/2차 보강) ──
dt = pd.to_datetime(en["작성일"], format="ISO8601", utc=True, errors="coerce")
en["d"] = dt.dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
b1s = en[en["commentId"].isin(tt_ids)].groupby("d").size()       # 1차(고변동 윈도우)
b2s = en[~en["commentId"].isin(tt_ids)].groupby("d").size()      # 2차(round-2 보강)
HERE = os.path.dirname(os.path.abspath(__file__))
E.fig_price_story(
    "TSLA",
    bars=[("1차 (고변동 윈도우)", b1s, "#546E7A"), ("2차 (round-2 보강)", b2s, "#2E7D32")],
    save_as="train_enriched_coverage.png", outdir=HERE,
    title_override="최종 학습셋(enriched) — TSLA · Fold A 학습 (1차 + 2차 보강)",
    subtitle_override=("fig1(빨강=상승·파랑=하락 변동사건) 위 학습 댓글 일별 분포  ·  "
                       f"1차 고변동 윈도우 + 2차 round-2 전기간 보강 · 총 {len(en):,}건"))
