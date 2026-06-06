"""TSLA·NVDA 주가 수집 — 라벨 생성용 (일봉 + 1시간봉).

plan.md 2절: '유의미한 변동'은 지난 6개월 일일 수익률의 μ, σ로 정의하고,
기준 구간 W는 예측일 ±1일 또는 작성일+3거래일까지 본다.
또한 댓글 timestamp가 변동 구간보다 앞설 때만 '예측'으로 인정(시점관계).

=> 일봉(1d): 6개월 μ/σ 통계 + 일 단위 변동 판정용
   1시간봉(1h): 댓글 작성시각(KST) vs 변동시각의 정밀 선후 판정용
   (yfinance 제약상 과거 1년 구간은 1h가 최소 단위, 분봉은 불가)

주가는 모델 입력이 아니라 **라벨 생성 전용**이다(누수 차단).
출력: raw_data/<TICKER>_prices_1d.csv , raw_data/<TICKER>_prices_1h.csv
"""

import os

import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "raw_data")   # 주가는 수집 원본 → raw_data/
KST = "Asia/Seoul"

# (티커, 댓글구간, 일봉수집구간[앞6개월·뒤버퍼])
JOBS = [
    {"ticker": "TSLA",
     "comment_window": ("2025-10-01", "2026-03-31"),
     "fetch": ("2025-04-01", "2026-04-10")},
    {"ticker": "NVDA",
     "comment_window": ("2025-12-01", "2026-05-31"),
     "fetch": ("2025-06-01", "2026-06-10")},
]


def _flatten(df):
    if hasattr(df.columns, "nlevels") and df.columns.nlevels > 1:
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_daily(job):
    t = job["ticker"]
    df = yf.download(t, start=job["fetch"][0], end=job["fetch"][1],
                     interval="1d", auto_adjust=True, progress=False)
    df = _flatten(df).reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].round(2)
    df["daily_return_pct"] = (df["Close"].pct_change() * 100).round(3)
    df["Volume"] = df["Volume"].astype("int64")
    df.to_csv(os.path.join(DATA, f"{t}_prices_1d.csv"), index=False, encoding="utf-8-sig")

    cw0, cw1 = job["comment_window"]
    win = df[(df["Date"] >= cw0) & (df["Date"] <= cw1)]["daily_return_pct"].dropna()
    mu, sig = win.mean(), win.std()
    print(f"[{t} 1d] {len(df)}행  {df['Date'].iloc[0]}~{df['Date'].iloc[-1]}  "
          f"-> raw_data/{t}_prices_1d.csv")
    print(f"        댓글구간 일일수익률 μ={mu:+.3f}% σ={sig:.3f}%  "
          f"(상승>{mu+sig:+.2f}%, 하락<{mu-sig:+.2f}%)")


def fetch_hourly(job):
    t = job["ticker"]
    # 1h는 과거 ~730일 제약. 댓글구간 + 앞뒤 며칠 버퍼.
    cw0, cw1 = job["comment_window"]
    start = (pd.Timestamp(cw0) - pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(cw1) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    df = yf.download(t, start=start, end=end, interval="1h",
                     auto_adjust=True, progress=False)
    df = _flatten(df).reset_index()
    dtcol = "Datetime" if "Datetime" in df.columns else df.columns[0]
    ts = pd.to_datetime(df[dtcol], utc=True)          # 거래소 tz -> UTC -> KST
    df["Datetime_KST"] = ts.dt.tz_convert(KST).dt.strftime("%Y-%m-%d %H:%M")
    out = df[["Datetime_KST", "Open", "High", "Low", "Close", "Volume"]].copy()
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c].round(2)
    out["return_pct"] = (out["Close"].pct_change() * 100).round(3)
    out["Volume"] = out["Volume"].astype("int64")
    out.to_csv(os.path.join(DATA, f"{t}_prices_1h.csv"), index=False, encoding="utf-8-sig")
    print(f"[{t} 1h] {len(out)}행  {out['Datetime_KST'].iloc[0]}~"
          f"{out['Datetime_KST'].iloc[-1]} (KST)  -> raw_data/{t}_prices_1h.csv")


if __name__ == "__main__":
    for job in JOBS:
        fetch_daily(job)
        fetch_hourly(job)
    print("\n수집 완료.")
