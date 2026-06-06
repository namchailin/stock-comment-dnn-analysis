# 추론 ON vs OFF 라벨 일치율 — "추론 끄면 성능 떨어지나" 검증
import os
import json
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from price_context import context_text
from prompt import build_messages

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
c = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
M = os.environ["MODEL"]
J = re.compile(r"\{.*\}", re.DOTALL)


def lab(r, think):
    msgs = build_messages(r["종목명"], r["작성일"], bool(r["주주_여부"]), r["text"],
                          context_text(r["종목명"], r["작성일"]))
    kw = {"model": M, "messages": msgs, "temperature": 0,
          "response_format": {"type": "json_object"}}
    if not think:
        kw["extra_body"] = {"reasoning": {"enabled": False}}
    for _ in range(3):
        try:
            ct = c.chat.completions.create(**kw).choices[0].message.content
            if ct:
                o = json.loads(J.search(ct).group(0))
                return o.get("시점관계"), o.get("예측_방향"), float(o.get("확신도") or 0)
        except Exception:
            pass
    return None, None, None


df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data",
                              "round2_consistency.csv"), encoding="utf-8-sig")
kw = "간다|오를|빠질|내릴|떡상|손절|상승|하락|불기둥|파란|빨간|살듯|팔"
pred = df[df["text"].str.contains(kw, na=False)].sample(10, random_state=2)
samp = pd.concat([pred, df.sample(8, random_state=2)]).drop_duplicates("commentId").head(18)

at = ad = ok = 0
cdiff = []
dis = []
for _, r in samp.iterrows():
    t1, d1, c1 = lab(r, True)
    t2, d2, c2 = lab(r, False)
    if t1 is None or t2 is None:
        continue
    ok += 1
    at += (t1 == t2)
    ad += (d1 == d2)
    cdiff.append(abs(c1 - c2))
    if t1 != t2 or d1 != d2:
        dis.append((str(r["text"])[:28], f"{t1}/{d1}", f"{t2}/{d2}"))
print(f"=== 추론 ON vs OFF 일치 (유효 {ok}건) ===")
print(f"  시점관계 일치 {at}/{ok} ({at/max(ok,1)*100:.0f}%)")
print(f"  예측방향 일치 {ad}/{ok} ({ad/max(ok,1)*100:.0f}%)")
print(f"  확신도 평균차 {sum(cdiff)/max(len(cdiff),1):.2f}")
for x in dis:
    print("  불일치:", x)
