# LLM 4-Class 라벨링 러너 (OpenAI 호환 엔드포인트)
#   사용:
#     python run_label.py --file train_A_TSLA --limit 50      # 소규모 검증
#     python run_label.py --all                                # 4개 전체
#   특징: 동시성(ThreadPool) · 재개(이미 라벨된 commentId 스킵) · JSON 검증.
#   출력: data/labeled_{name}.csv  (commentId + 6컬럼). 원본과 commentId로 머지.
import os
import re
import json
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from price_context import context_text, ROOT
from prompt import build_messages, ALLOWED

DATADIR = os.path.join(ROOT, "data"); SPLITS = os.path.join(DATADIR, "unlabeled", "splits"); LBL = os.path.join(DATADIR, "labeled")
FILES = ["tsla_train", "nvda_eval", "expert_control"]
COLS = ["시점관계", "예측_방향", "예측_날짜", "확신도", "근거_유형", "Class0_사유"]
OUTCOLS = ["commentId"] + COLS + ["_error"]   # 모든 행 동일 컬럼(에러행 포함) → CSV 안 깨짐

load_dotenv(os.path.join(ROOT, ".env"))
client = OpenAI(api_key=os.environ["API_KEY"], base_url=os.environ["BASE_URL"])
MODEL = os.environ["MODEL"]
_lock = threading.Lock()
_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse(txt: str) -> dict:
    m = _JSON.search(txt)
    if not m:
        raise ValueError(f"JSON 없음: {txt[:80]}")
    o = json.loads(m.group(0))
    # 검증·정규화
    for k in ("시점관계", "예측_방향", "근거_유형", "Class0_사유"):
        if o.get(k) not in ALLOWED[k]:
            o[k] = None if k == "Class0_사유" else "없음"
    o["확신도"] = max(0.0, min(1.0, float(o.get("확신도", 0) or 0)))
    o["예측_날짜"] = o.get("예측_날짜") or None
    return {k: o.get(k) for k in COLS}


_json_mode = True          # 모델이 response_format 지원하면 유지, 거부하면 자동 해제


def label_one(row) -> dict:
    global _json_mode
    ctx = context_text(row["종목명"], row["작성일"])
    msgs = build_messages(row["종목명"], row["작성일"], bool(row["주주_여부"]),
                          row["text"], ctx)
    for attempt in range(3):
        try:
            kw = {"model": MODEL, "messages": msgs, "temperature": 0,
                  "extra_body": {"reasoning": {"enabled": False}}}   # 추론 끄기(출력토큰 43배↓)
            if _json_mode:
                kw["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kw)
            out = parse(r.choices[0].message.content)
            out["commentId"] = row["commentId"]
            return out
        except Exception as e:
            es = str(e).lower()
            if _json_mode and ("response_format" in es or "json" in es
                               or "not support" in es):
                _json_mode = False          # JSON 모드 미지원 → 끄고 재시도
                continue
            if attempt == 2:
                return {"commentId": row["commentId"], "_error": str(e)[:120],
                        **{c: None for c in COLS}}


def run(name, limit, workers):
    src = pd.read_csv(os.path.join(SPLITS, f"{name}.csv"), encoding="utf-8-sig")
    if limit:
        src = src.head(limit)
    out_path = os.path.join(LBL, f"{name}_labeled.csv")
    done = set()
    if os.path.exists(out_path):
        prev = pd.read_csv(out_path, encoding="utf-8-sig",
                           on_bad_lines="skip", engine="python")   # 레거시 깨진 파일도 견고히
        ok = prev[prev["시점관계"].notna()] if "시점관계" in prev.columns else prev.iloc[:0]
        ok = ok.drop_duplicates("commentId")
        ok.reindex(columns=OUTCOLS).to_csv(out_path, index=False, encoding="utf-8-sig")
        done = set(ok["commentId"])                             # 성공한 것만 스킵 → 에러는 재시도
    todo = src[~src["commentId"].isin(done)]
    print(f"[{name}] 대상 {len(src):,} / 이미됨 {len(done):,} / 할것 {len(todo):,}")
    if todo.empty:
        return
    header_needed = not os.path.exists(out_path)
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(label_one, r) for _, r in todo.iterrows()]
        for fu in as_completed(futs):
            res = fu.result()
            with _lock:
                pd.DataFrame([res]).reindex(columns=OUTCOLS).to_csv(
                    out_path, mode="a", index=False, header=header_needed,
                    encoding="utf-8-sig")     # 항상 동일 컬럼 → CSV 안 깨짐
                header_needed = False
            n += 1
            if n % 50 == 0:
                print(f"  {name}: {n:,}/{len(todo):,}")
    err = pd.read_csv(out_path, encoding="utf-8-sig", on_bad_lines="skip", engine="python")
    if "_error" in err:
        print(f"  ⚠️ 실패 {int(err['_error'].notna().sum())}건 (재실행 시 자동 재시도)")
    print(f"  saved -> data/labeled_{name}.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="단일 파일명(확장자 제외)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    targets = FILES if a.all else [a.file]
    assert targets[0], "--file 또는 --all 필요"
    for t in targets:
        run(t, a.limit, a.workers)
