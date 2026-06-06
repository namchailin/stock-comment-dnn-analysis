# 가격맥락 추출 — 각 댓글 작성시각(KST) '직전' 시간별 봉을 잘라 프롬프트용 텍스트로.
#   미래 구간은 절대 포함하지 않음(누수 차단). 직전 48h 내 봉, 부족하면 직전 8봉 fallback.
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "unlabeled", "raw")

LOOKBACK_H = 48        # 직전 몇 시간을 볼지
MIN_BARS = 4           # 48h 내 봉이 이보다 적으면(주말 등) fallback
FALLBACK_BARS = 8      # fallback 시 직전 N봉
MAX_BARS = 16          # 너무 길면 컷


def load_1h(stock: str) -> pd.DataFrame:
    p = pd.read_csv(os.path.join(DATA, f"{stock}_prices_1h.csv"), encoding="utf-8-sig")
    p["dt"] = pd.to_datetime(p["Datetime_KST"])      # naive KST
    return p.sort_values("dt").reset_index(drop=True)


_CACHE: dict[str, pd.DataFrame] = {}


def _prices(stock: str) -> pd.DataFrame:
    if stock not in _CACHE:
        _CACHE[stock] = load_1h(stock)
    return _CACHE[stock]


def context_text(stock: str, comment_iso: str) -> str:
    """댓글 작성시각 직전 시간별 흐름을 한국어 한 줄 요약 + 봉 나열로 반환."""
    t = pd.to_datetime(comment_iso).tz_convert("Asia/Seoul").tz_localize(None)
    p = _prices(stock)
    past = p[p["dt"] < t]
    if past.empty:
        return "(직전 거래 데이터 없음)"
    win = past[past["dt"] >= t - pd.Timedelta(hours=LOOKBACK_H)]
    if len(win) < MIN_BARS:
        win = past.tail(FALLBACK_BARS)
    win = win.tail(MAX_BARS)

    first, last = win.iloc[0], win.iloc[-1]
    chg = (last["Close"] / first["Close"] - 1) * 100
    arrow = "상승(빨강)" if chg > 0.3 else ("하락(파랑)" if chg < -0.3 else "보합")
    head = (f"[직전 가격흐름 {win.iloc[0]['dt']:%m-%d %H:%M}~{last['dt']:%m-%d %H:%M}] "
            f"종가 {first['Close']:.2f}→{last['Close']:.2f} ({chg:+.1f}%, {arrow})")
    bars = " | ".join(f"{r.dt:%m-%d %H시} {r.Close:.2f}" for r in win.itertuples())
    return head + "\n" + bars


if __name__ == "__main__":   # 단독 실행 = 동작 점검
    import pandas as pd
    for s, f in [("TSLA", "train_A_TSLA"), ("NVDA", "test_A_NVDA")]:
        c = pd.read_csv(os.path.join(ROOT, "data", f"{f}.csv"), encoding="utf-8-sig", nrows=2)
        for _, r in c.iterrows():
            print(f"=== {s} | {r['작성일']} ===")
            print(context_text(s, r["작성일"]))
            print()
