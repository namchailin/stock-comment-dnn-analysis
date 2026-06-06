# C3(날짜+방향 적중) 특징 해부 — 분포·horizon·확신도·근거·텍스트
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
import os,sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
from exploration import ROOT

LAB = os.path.join(ROOT, "data", "labeled")
fin = pd.concat([pd.read_csv(os.path.join(LAB, f"{f}_final.csv"), encoding="utf-8-sig")
                 for f in ["tsla_train", "nvda_eval", "expert_control"]],
                ignore_index=True).drop_duplicates("commentId")
raw = pd.read_csv(os.path.join(ROOT, "data", "unlabeled", "raw", "merged_comments.csv"),
                  encoding="utf-8-sig")
bmap = raw.groupby("닉네임_ID")["뱃지"].first().fillna("").astype(str).str.strip().ne("")
fin["badge"] = fin["닉네임_ID"].map(bmap).fillna(False)

c3 = fin[fin["Class"] == 3].copy()
pred = fin[fin["Class"].isin([1, 2, 3])].copy()      # 비교군: 전체 예측
other = pred[pred["Class"] != 3]
print(f"=== C3 {len(c3)}건 vs 전체예측 {len(pred):,} ===\n")

# 1) 구성비
print("[구성]")
print("  종목:", dict(c3["종목명"].value_counts()), "| 전체예측:", dict(pred["종목명"].value_counts()))
print("  방향:", dict(c3["예측_방향"].value_counts()))
print(f"  주주: {c3['주주_여부'].astype(str).str.lower().eq('true').mean()*100:.0f}% | 뱃지: {c3['badge'].mean()*100:.0f}%")

# 2) 확신도
print("\n[확신도]")
print(f"  C3 {c3['확신도'].mean():.2f} (median {c3['확신도'].median():.2f}) vs 비-C3예측 {other['확신도'].mean():.2f}")

# 3) 근거유형
print("\n[근거유형 비율 %]")
cg = c3["근거_유형"].value_counts(normalize=True) * 100
og = other["근거_유형"].value_counts(normalize=True) * 100
for k in ["단순감정", "차트", "뉴스", "실적", "기타", "없음"]:
    print(f"  {k:5}: C3 {cg.get(k,0):4.0f}% | 비-C3 {og.get(k,0):4.0f}%")

# 4) horizon (예측날짜 - 작성일)
base = pd.to_datetime(c3["작성일"]).dt.tz_localize(None).dt.normalize()
predd = pd.to_datetime(c3["예측_날짜"], errors="coerce").dt.normalize()
off = (predd - base).dt.days
print("\n[예측 horizon(일)]")
print(f"  중앙값 {off.median():.0f} / 분포 p10~90 {off.quantile(.1):.0f}~{off.quantile(.9):.0f} / 당일(0일) {int((off==0).sum())}건")

# 5) 작성 시점 분포 (월별)
c3["월"] = pd.to_datetime(c3["작성일"]).dt.tz_localize(None).dt.strftime("%Y-%m")
print("\n[작성 월별 분포]")
for m, v in c3["월"].value_counts().sort_index().items():
    print(f"  {m}: {'█'*v} {v}")

# 6) 텍스트
print("\n[텍스트]")
print(f"  평균 길이: C3 {c3['text'].str.len().mean():.0f}자 vs 비-C3 {other['text'].str.len().mean():.0f}자")
# C3에서 두드러진 토큰 (C3 빈도/전체예측 빈도)
def toks(s): return [w for w in re.findall(r"[가-힣A-Za-z0-9]+", str(s)) if len(w) >= 2]
c3c = Counter(t for x in c3["text"] for t in toks(x))
allc = Counter(t for x in pred["text"] for t in toks(x))
lift = {w: (c3c[w]/len(c3))/((allc[w]/len(pred))+1e-9) for w in c3c if c3c[w] >= 5}
top = sorted(lift.items(), key=lambda x: -x[1])[:15]
print("  C3에 두드러진 단어(lift):", ", ".join(f"{w}({c3c[w]})" for w, _ in top))

# 7) 예시
print("\n[C3 예시 8건]")
for _, r in c3.sample(min(8, len(c3)), random_state=1).iterrows():
    print(f"  [{r['종목명']}/{r['예측_방향']}/확{r['확신도']}/{r['근거_유형']}] {str(r['text'])[:50]}")
