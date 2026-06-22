# 클래스별 F1 막대그래프 — 표 4(1단계, TSLA학습→NVDA평가) 값으로 결과 시각화.
#   인코더 3종 + TF-IDF 기준선. 다수클래스(=0)는 C1~C3이 0이라 생략.
#   exploration 스타일(Noto 폰트·색·_save) 재사용.
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lib"))
import numpy as np
import matplotlib.pyplot as plt
from exploration import _save  # 폰트 rcParams·저장 경로(results/figures) 적용

CLASSES = ["Class 0", "Class 1", "Class 2", "Class 3"]
# 표 4 (1단계 원본 라벨 학습) 클래스별 F1
ROWS = {
    "KLUE-RoBERTa":        [0.925, 0.217, 0.184, 0.232],
    "KcELECTRA":           [0.930, 0.223, 0.214, 0.000],
    "KR-FinBERT":          [0.920, 0.211, 0.175, 0.069],
    "TF-IDF (기준선)":      [0.882, 0.123, 0.116, 0.170],
}
COLORS = ["#1f5fbf", "#4f9d4f", "#e08a1e", "#9aa3af"]  # 마지막=기준선 회색

x = np.arange(len(CLASSES))
n = len(ROWS)
w = 0.8 / n
fig, ax = plt.subplots(figsize=(7.6, 4.3))
for i, (name, vals) in enumerate(ROWS.items()):
    bars = ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=name,
                  color=COLORS[i], edgecolor="white", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x); ax.set_xticklabels(CLASSES)
ax.set_ylabel("F1"); ax.set_ylim(0, 1.05)
ax.set_title("클래스별 F1 — 인코더 vs 기준선 (1단계, NVDA 교차 평가)")
ax.legend(ncol=2, fontsize=8.5, loc="upper right", framealpha=0.9)
ax.axhline(0, color="0.6", lw=0.6)
_save(fig, "fig_perclass_f1.png")
print("저장: results/figures/fig_perclass_f1.png")
