# 라벨링 후(後) EDA — plan 3-2. final_*.csv(Class 부여본)에서 5종 차트.
#   1차 EDA(exploration.py)의 스타일/색/_save를 그대로 재사용해 발표 일관성 유지.
#   실행: python3 eda_post.py   (finalize_labels.py 실행 후)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1차 EDA 스타일 그대로 재사용 (폰트 rcParams·색·_save·경로는 import 시 적용됨)
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
from exploration import UP, DOWN, NEUTRAL, FIGDIR, _save, ROOT

DATA = os.path.join(ROOT, "data")
SPLITS = os.path.join(DATA, "unlabeled", "splits")
LABELED = os.path.join(DATA, "labeled")
# Fold A 단방향(현재 범위). 라벨된 것만 자동 사용.
FOLD = {"tsla_train": "TSLA", "nvda_eval": "NVDA"}

CLASS_NAME = {0: "C0 예측없음", 1: "C1 예측실패", 2: "C2 방향적중", 3: "C3 날짜적중"}
CLASS_C = {0: NEUTRAL, 1: "#5B8DEF", 2: "#FF9800", 3: UP}      # C3=빨강 강조
DIR_C = {"상승": UP, "하락": DOWN, "없음": NEUTRAL}             # 1차 팔레트 동일
REL_C = {"예측": UP, "리액션": NEUTRAL}


def load_final() -> pd.DataFrame:
    parts = []
    for name in FOLD:
        fp = os.path.join(LABELED, f"{name}_final.csv")
        if not os.path.exists(fp):
            print(f"  (스킵: final_{name}.csv 없음)")
            continue
        d = pd.read_csv(fp, encoding="utf-8-sig")
        src = os.path.join(SPLITS, f"{name}.csv")          # 뱃지 복원용
        if os.path.exists(src):
            s = pd.read_csv(src, encoding="utf-8-sig")[["commentId", "뱃지"]]
            d = d.merge(s, on="commentId", how="left")
        d["_set"] = name
        parts.append(d)
    if not parts:
        raise SystemExit("final_*.csv 가 없습니다. 먼저 finalize_labels.py 실행.")
    df = pd.concat(parts, ignore_index=True)
    df["has_badge"] = df.get("뱃지", "").fillna("").astype(str).str.strip().ne("")
    df["주주"] = df["주주_여부"].astype(str).str.lower().eq("true")
    return df


def _annot_bar(ax, bars, total, pct=True):
    for b in bars:
        h = b.get_height()
        txt = f"{int(h):,}" + (f"\n{h/total*100:.1f}%" if pct else "")
        ax.text(b.get_x() + b.get_width() / 2, h, txt, ha="center", va="bottom",
                fontsize=10, fontweight="bold")


# ── fig6: 4-Class 분포 ───────────────────────────────────────
def fig_class_dist(df):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    vc = df["Class"].value_counts().reindex([0, 1, 2, 3], fill_value=0)
    bars = ax.bar([CLASS_NAME[c] for c in [0, 1, 2, 3]], vc.values,
                  color=[CLASS_C[c] for c in [0, 1, 2, 3]], edgecolor="white")
    _annot_bar(ax, bars, len(df))
    ax.set_ylabel("댓글 수")
    ax.set_title(f"4-Class 라벨 분포 (n={len(df):,})  —  C3(날짜적중) 희소성 주목")
    ax.set_ylim(0, vc.max() * 1.18)
    _save(fig, "fig6_class_dist.png")


# ── fig7: Class0 사유 구성 ───────────────────────────────────
def fig_class0_reasons(df):
    sub = df[df["Class"] == 0]
    vc = sub["Class0_사유"].fillna("(미분류)").value_counts()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.barh(vc.index[::-1], vc.values[::-1], color="#7E57C2", edgecolor="white")
    for b in bars:
        w = b.get_width()
        ax.text(w, b.get_y() + b.get_height() / 2, f" {int(w):,} ({w/len(sub)*100:.1f}%)",
                va="center", fontsize=9, fontweight="bold")
    ax.set_xlabel("댓글 수")
    ax.set_title(f"Class 0 구성 사유 (n={len(sub):,})  —  사전필터 생략 정당성 점검")
    ax.set_xlim(0, vc.max() * 1.22)
    _save(fig, "fig7_class0_reasons.png")


# ── fig8: 시점관계 + 예측_방향 (1차 팔레트) ──────────────────
def fig_timing_direction(df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    rc = df["시점관계"].value_counts().reindex(["예측", "리액션"], fill_value=0)
    b1 = a1.bar(rc.index, rc.values, color=[REL_C[k] for k in rc.index], edgecolor="white")
    _annot_bar(a1, b1, len(df)); a1.set_title("시점관계 (예측 vs 리액션)")
    a1.set_ylim(0, rc.max() * 1.18); a1.set_ylabel("댓글 수")

    dc = df["예측_방향"].value_counts().reindex(["상승", "하락", "없음"], fill_value=0)
    b2 = a2.bar(dc.index, dc.values, color=[DIR_C[k] for k in dc.index], edgecolor="white")
    _annot_bar(a2, b2, len(df)); a2.set_title("예측_방향 (상승=빨강·하락=파랑·없음=회색)")
    a2.set_ylim(0, dc.max() * 1.18)
    fig.suptitle("댓글 시점관계·방향 분포 — 예측/리액션 분리 근거", fontweight="bold")
    _save(fig, "fig8_timing_direction.png")


# ── fig9: 뱃지 ↔ 고수(C2·3) 경향 ─────────────────────────────
def fig_badge_expert(df):
    if "뱃지" not in df:
        print("  (스킵 fig9: 뱃지 컬럼 없음)"); return
    df = df.copy()
    df["고수"] = df["Class"].isin([2, 3])
    g = df.groupby("has_badge")["고수"].mean() * 100
    n = df.groupby("has_badge").size()
    labels = {False: f"뱃지無\n(n={n.get(False,0):,})", True: f"뱃지有\n(n={n.get(True,0):,})"}
    fig, ax = plt.subplots(figsize=(6, 4.4))
    xs = [labels[k] for k in [False, True]]
    ys = [g.get(False, 0), g.get(True, 0)]
    bars = ax.bar(xs, ys, color=[NEUTRAL, UP], edgecolor="white")
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{b.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("고수(C2·C3) 비율 (%)")
    ax.set_title("뱃지 보유 ↔ 고수 라벨 경향 (sanity check, 인과 아님)")
    ax.set_ylim(0, max(ys) * 1.25 if max(ys) else 1)
    _save(fig, "fig9_badge_expert.png")


# ── fig10: 사용자별 고수비율 분포 ────────────────────────────
def fig_user_consistency(df, min_cmt=5):
    g = df.groupby("닉네임_ID")
    ratio = (g["Class"].apply(lambda s: s.isin([2, 3]).mean()) * 100)
    cnt = g.size()
    keep = ratio[cnt >= min_cmt]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.hist(keep.values, bins=20, color="#26A69A", edgecolor="white")
    ax.set_xlabel("유저별 고수(C2·C3) 댓글 비율 (%)")
    ax.set_ylabel(f"유저 수 (댓글 {min_cmt}건 이상, n={len(keep):,})")
    ax.set_title("사용자별 일관성 — 우연 적중 vs 꾸준한 고수 구분(해석·보조용)")
    _save(fig, "fig10_user_consistency.png")


def main():
    df = load_final()
    print(f"로드: {len(df):,}건 / Class 분포 {dict(df['Class'].value_counts().sort_index())}")
    fig_class_dist(df)
    fig_class0_reasons(df)
    fig_timing_direction(df)
    fig_badge_expert(df)
    fig_user_consistency(df)
    print(f"→ 그림 5종 저장: {FIGDIR}/fig6~fig10")


if __name__ == "__main__":
    main()
