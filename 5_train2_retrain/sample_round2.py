# 2차 라벨링 대상 샘플링 — 고수후보(C2/C3≥2) + 대조군(예측≥2·C2/C3≤1) 의 미라벨 댓글.
#   정책: labeling/labeling_consistency.md
import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LABELED = os.path.join(DATA, "labeled")
URL_RE = re.compile(r"https?://\S+|www\.\S+|\b\S+\.(?:com|net|org|io|ly|kr|co\.kr)(?:/\S*)?",
                    re.IGNORECASE)
WS_RE = re.compile(r"\s+")
KEEP = ["commentId", "닉네임_ID", "종목명", "주주_여부", "뱃지", "작성일", "내용", "text", "grp"]

# 1) 고수후보 / 대조군 유저
fin = pd.concat([pd.read_csv(os.path.join(DATA, f"{f}_final.csv"), encoding="utf-8-sig")
                 for f in ["tsla_train", "nvda_eval"]], ignore_index=True)
g = fin.groupby("닉네임_ID")["Class"]
hit = g.apply(lambda s: ((s == 2) | (s == 3)).sum())
npred = g.apply(lambda s: s.isin([1, 2, 3]).sum())
cand = set(hit[hit >= 2].index)
ctrl_all = npred[(npred >= 2) & (~npred.index.isin(cand))].index
ctrl = set(np.random.default_rng(0).choice(list(ctrl_all), 130, replace=False))

# 2) 이들의 미라벨 댓글 + 클리닝
raw = pd.read_csv(os.path.join(ROOT, "data", "unlabeled", "raw", "merged_comments.csv"), encoding="utf-8-sig")
raw["grp"] = np.where(raw["닉네임_ID"].isin(cand), "고수후보",
                      np.where(raw["닉네임_ID"].isin(ctrl), "대조군", None))
raw = raw[raw["grp"].notna()].copy()
t = raw["내용"].fillna("").astype(str).str.replace(URL_RE, " ", regex=True)
raw["text"] = t.str.replace(WS_RE, " ", regex=True).str.strip()
raw = raw[raw["text"].str.len() >= 4]
labeled = set(fin["commentId"])
raw = raw[~raw["commentId"].isin(labeled)]

raw[KEEP].to_csv(os.path.join(ROOT, "data", "unlabeled", "splits", "expert_control.csv"), index=False, encoding="utf-8-sig")
print(f"저장: data/round2_consistency.csv  {len(raw):,}건")
print("  그룹:", dict(raw["grp"].value_counts()), "/ 종목:", dict(raw["종목명"].value_counts()))
print(f"  유저: 고수후보 {raw[raw.grp=='고수후보']['닉네임_ID'].nunique()}명 / "
      f"대조군 {raw[raw.grp=='대조군']['닉네임_ID'].nunique()}명")
