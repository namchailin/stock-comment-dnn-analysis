"""[Step 2] EDA — 3-1. 라벨링 전(前) 단계 (plan.md 개정판).

핵심: '유의미한 변동'을 **다중일 누적** 기준으로 정의한다.
  6개월 일일 수익률의 μ, σ로부터, h거래일 누적수익률 R_W 가
    R_W >= μ·h + k·σ·√h  → 상승
    R_W <= μ·h − k·σ·√h  → 하락
    그 사이                → 무변동   (k = σ 임계 배수)

3-1 산출물:
  A. 종목별 주가 추이 + 변동사건 음영(band) + R_W 패널  -> fig1_price_story_<T>.png
  B. 변동 판정선 μh±kσ√h의 (k, h) 스윕(지속율·변동 사건 수)  -> fig2_threshold_sweep.png
  C. 댓글 길이·단어 수 분포                              -> fig3_length_dist.png
  D. 유저 메타데이터 비율(주주/뱃지)                     -> fig4_user_meta.png
  E. 주가 변동성(|일일 등락률|) vs 일일 댓글량            -> fig5_volatility_vs_volume.png
  F. 일일 등락률(빨강=상승/파랑=하락) vs 댓글량           -> fig6_updown_vs_volume.png

색상 관례(한국): 빨강=상승, 파랑=하락 (plan 3-0 최우선).
보류(3-2, 라벨 필요): 4-Class 분포 / Class0 사유 / 시점관계 / 뱃지↔고수 / 일관성.
"""

import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

# 전역 스타일: 흰 배경 + 아주 흐린 그리드
sns.set_theme(style="white", context="notebook")
# 한글 폰트는 seaborn 테마 적용 *뒤*에 재설정해야 먹는다.
# Regular+Bold 둘 다 등록해야 fontweight="bold"가 진짜 굵게 렌더된다.
_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
_FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
for _fp in (_FONT, _FONT_BOLD):
    if os.path.exists(_fp):
        fm.fontManager.addfont(_fp)
if os.path.exists(_FONT):
    matplotlib.rcParams["font.family"] = fm.FontProperties(fname=_FONT).get_name()
matplotlib.rcParams.update({
    "axes.unicode_minus": False,
    "axes.titleweight": "bold",        # 모든 축 제목 볼드
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.grid": True,                 # 그리드 켜되
    "axes.axisbelow": True,
    "grid.color": "0.9",               # 아주 흐리게
    "grid.linewidth": 0.7,
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 프로젝트 루트(이 파일이 루트에 위치)
# 수집 원본·파생 캐시는 raw_data/, 학습/평가셋은 data/ 에 분리 보관
DATA = os.path.join(ROOT, "data", "unlabeled", "raw")
FIGDIR = os.path.join(ROOT, "results", "figures")
os.makedirs(FIGDIR, exist_ok=True)

UP, DOWN, NEUTRAL = "#E31A1C", "#1F6FE5", "#9E9E9E"   # 상승 / 하락 / 무변동
WINDOWS = {"TSLA": ("2025-10-01", "2026-03-31"),
           "NVDA": ("2025-12-01", "2026-05-31")}
STOCK_C = {"TSLA": UP, "NVDA": "#76B900"}

# 변동 정의 파라미터 — 임계 스윕(B)으로 정당화, 스토리 차트(A)엔 이 기본값 사용
H_DEFAULT, K_DEFAULT = 5, 1.0
H_SWEEP, K_SWEEP = [2, 3, 4, 5, 6, 7, 8, 10], [1.0, 1.5]
SMALL_N = 10            # 이 미만이면 지속율이 표본 노이즈 — 빨강 경고 표기


# ── 로드 ────────────────────────────────────────────────────
def load_comments() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA, "merged_comments.csv"), encoding="utf-8-sig")
    dt = pd.to_datetime(df["작성일"], format="ISO8601", utc=True, errors="coerce")
    df = df[dt.notna()].copy()
    dt = dt[dt.notna()]
    df["dt_kst"] = dt.dt.tz_convert("Asia/Seoul")
    # 댓글-일봉 매칭용: 미국 거래 세션(개장일) 기준 날짜 (KST 밤~새벽 세션 귀속 보정)
    df["us_date"] = dt.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    df["내용"] = df["내용"].fillna("").astype(str)
    df["char_len"] = df["내용"].str.len()
    df["word_cnt"] = df["내용"].str.split().str.len()
    df["주주"] = df["주주_여부"].astype(str).str.lower().eq("true")
    df["뱃지"] = df["뱃지"].fillna("").astype(str)
    df["has_badge"] = df["뱃지"].str.strip().ne("")
    return df


def load_prices(stock: str):
    """전체 일봉 + 댓글구간 μ·σ(%) 반환. (누적계산용으로 버퍼 포함 전체 반환)"""
    p = pd.read_csv(os.path.join(DATA, f"{stock}_prices_1d.csv"), encoding="utf-8-sig")
    cw0, cw1 = WINDOWS[stock]
    win = p[(p["Date"] >= cw0) & (p["Date"] <= cw1)]["daily_return_pct"]
    return p, win.mean(), win.std()


def _save(fig, name):
    fig.savefig(os.path.join(FIGDIR, name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> results/figures/{name}")


# ── 변동 사건 에피소드 추출 (연속 구간 묶기 + 짧은 틈 병합) ──
def _episodes(state, gap=2):
    """state(-1/0/1 배열)에서 같은 부호 연속구간을 사건으로 추출.
    부호가 같고 사이의 0구간이 gap 이하면 하나로 병합."""
    s = state.copy()
    n = len(s)
    i = 0
    while i < n:                      # 짧은 틈 메우기
        if s[i] == 0:
            j = i
            while j < n and s[j] == 0:
                j += 1
            left = s[i - 1] if i > 0 else 0
            right = s[j] if j < n else 0
            if left != 0 and left == right and (j - i) <= gap:
                s[i:j] = left
            i = j
        else:
            i += 1
    runs = []                         # 연속구간 추출
    i = 0
    while i < n:
        if s[i] != 0:
            sign, j = s[i], i
            while j + 1 < n and s[j + 1] == sign:
                j += 1
            runs.append((sign, i, j))
            i = j + 1
        else:
            i += 1
    return runs


# ── A. 주가 스토리 (굵은 라벨 밴드 + 거래량 + 누적수익률) ────
def fig_price_story(stock, h=H_DEFAULT, k=K_DEFAULT, max_labels=8,
                    highlights=None, suffix="", htitle=""):
    p, mu, sig = load_prices(stock)
    p = p.reset_index(drop=True)
    p["RW"] = p["Close"].pct_change(h) * 100        # 최근 h거래일 누적수익률(%)
    up_thr = mu * h + k * sig * np.sqrt(h)
    dn_thr = mu * h - k * sig * np.sqrt(h)
    cw0, cw1 = WINDOWS[stock]
    widx = p.index[(p["Date"] >= cw0) & (p["Date"] <= cw1)]
    W0, W1 = widx[0], widx[-1]
    w = p.loc[W0:W1].reset_index(drop=True)
    x = pd.to_datetime(w["Date"]).to_numpy()
    rw = w["RW"].to_numpy()
    state = np.where(rw >= up_thr, 1, np.where(rw <= dn_thr, -1, 0))

    dates_f = pd.to_datetime(p["Date"])
    close_f = p["Close"].to_numpy()
    eps = []
    for sign, i, j in _episodes(state, gap=2):
        fi, fj = W0 + i, W0 + j
        start = max(0, fi - h)                       # 실제 가격이 움직이기 시작한 지점(룩백 h)
        cum = (close_f[fj] / close_f[start] - 1) * 100
        ds = dates_f.iloc[max(start, W0)]            # 표시 시작은 구간 시작으로 클램프
        de = dates_f.iloc[fj]
        eps.append((sign, ds, de, cum))
    eps = sorted(eps, key=lambda e: -abs(e[3]))[:max_labels]  # 큰 사건 우선
    eps = sorted(eps, key=lambda e: e[1])                     # 그린 뒤엔 날짜순

    fig, (axP, axR) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.7]})
    # 1단: 종가 + 굵은 사건 밴드 + 라벨
    lo, hi = w["Close"].min(), w["Close"].max()
    rng = hi - lo
    axP.set_ylim(lo - rng * 0.03, hi + rng * 0.16)   # 위 공백 최소화
    half = pd.Timedelta(hours=12)        # 짧은(1~2일) 사건도 보이게 양옆 패딩
    for n_e, (sign, ds, de, cum) in enumerate(eps):
        c = UP if sign > 0 else DOWN
        axP.axvspan(ds - half, de + half, color=c, alpha=0.22, zorder=1)
        xm = ds + (de - ds) / 2
        ylab = hi + rng * (0.045 if n_e % 2 == 0 else 0.105)
        axP.annotate(f"{'▲' if sign>0 else '▼'}{cum:+.0f}%\n{ds:%m/%d}~{de:%m/%d}",
                     xy=(xm, ylab), ha="center", va="center", fontsize=9,
                     color=c, fontweight="bold")
    axP.plot(x, w["Close"].to_numpy(), color="black", lw=1.8, zorder=3)
    axP.set_ylabel("종가(USD)", fontsize=12)
    axP.legend(handles=[Patch(color=UP, alpha=0.4, label="상승 사건"),
                        Patch(color=DOWN, alpha=0.4, label="하락 사건")],
               loc="lower left", fontsize=11, ncol=2)

    # 2단: 최근 h일 누적수익률 막대 (사건=빨강/파랑, 무변동=회색)
    bar_c = [UP if v >= up_thr else DOWN if v <= dn_thr else NEUTRAL for v in rw]
    axR.bar(x, rw, color=bar_c, width=1.0)
    axR.axhline(0, color="gray", lw=0.6)
    axR.axhline(up_thr, color=UP, ls="--", lw=1.4, label=f"상승 판정선 +{up_thr:.1f}%")
    axR.axhline(dn_thr, color=DOWN, ls="--", lw=1.4, label=f"하락 판정선 {dn_thr:.1f}%")
    axR.set_ylabel(f"{h}일 누적(%)", fontsize=10)
    axR.set_xlabel("날짜", fontsize=12)
    axR.legend(loc="upper left", fontsize=9, ncol=2)
    axR.annotate("※ 막대가 비는 구간 = 비거래일(주말·미국 공휴일)", xy=(0.995, 0.06),
                 xycoords="axes fraction", ha="right", va="bottom",
                 fontsize=9, color="dimgray")

    fig.autofmt_xdate()
    axP.tick_params(axis="x", labelbottom=True)       # 위 패널도 날짜 눈금 표시
    plt.setp(axP.get_xticklabels(), rotation=30, ha="right")

    fname = f"fig1_price_story_{stock}.png"
    title = f"{stock} 주가 추이와 유의미 변동 사건"
    subtitle = (f"{h}일치 누적 수익률(h={h})이 평소 변동폭(±{k}σ)을 크게 넘은 구간을 "
                f"유의미한 상승/하락 사건으로 표시   (변동 판정선 μh ± kσ√h)")
    if highlights:                                    # 누적변동 큰 연속 구간 표시(복제본)
        for hl in highlights:
            s = pd.to_datetime(hl["start"]); e = pd.to_datetime(hl["end"])
            axP.axvspan(s, e, facecolor=hl["color"], alpha=0.12,
                        edgecolor=hl["color"], lw=2.4, ls="--", zorder=4)
            extra = (f"\n학습셋 유저 겹침 {hl['overlap']:,}건" if "overlap" in hl else "")
            axP.annotate(f"{hl['tag']}\n댓글 {hl['comments']:,}건 · "
                         f"상승 {hl['up']}일·하락 {hl['dn']}일 · 누적 {hl['cum']:+.1f}%{extra}",
                         xy=(s + (e - s) / 2, lo + rng * 0.02), ha="center", va="bottom",
                         fontsize=9.5, fontweight="bold", color=hl["color"], zorder=6,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                   alpha=0.9, edgecolor=hl["color"]))
        fname = f"fig1_highlight_{stock}{suffix}.png"
        title = f"{stock} — 누적변동 큰 연속 구간  {htitle}"
        subtitle = ("색칠 구간 = 댓글이 표적 수에 도달하는 연속 거래일 중 "
                    "누적변동(|종가 변화율|)이 가장 큰 구간")

    fig.tight_layout()
    fig.subplots_adjust(top=0.87, hspace=0.30)       # 위 패널 날짜 라벨 + 패널 간격
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.918, subtitle, ha="center", fontsize=10, color="0.45")
    _save(fig, fname)


# ── A-2. 누적변동 큰 연속 구간(댓글량 표적)을 fig1 복제본에 표시 ──
# 댓글≥target 중 (상승 사건일, 하락 사건일)이 '골고루 + 많이'인 구간 선택
#   key = (min(상승,하락) 최대 → 적은 쪽도 많게, 불균형 |상승−하락| 최소, 총 사건일 최대)
#   → fig1의 빨강·파랑 밴드가 한쪽 쏠림 없이 빽빽하게 섞인 = 업다운 심한 구간
def _best_var_window(df, stock, target):
    cw0, cw1 = WINDOWS[stock]
    d = df[(df["종목명"] == stock) & (df["us_date"] >= cw0) & (df["us_date"] <= cw1)]
    cc = d.groupby("us_date").size()
    idx = pd.date_range(cw0, cw1, freq="D").strftime("%Y-%m-%d")
    cumv = cc.reindex(idx, fill_value=0).cumsum().to_numpy()
    pos = {dt: i for i, dt in enumerate(idx)}
    p, mu, sig = load_prices(stock)
    p = p.sort_values("Date").reset_index(drop=True)
    p["RW"] = p["Close"].pct_change(H_DEFAULT) * 100
    up_thr = mu * H_DEFAULT + K_DEFAULT * sig * np.sqrt(H_DEFAULT)
    dn_thr = mu * H_DEFAULT - K_DEFAULT * sig * np.sqrt(H_DEFAULT)
    pw = p[(p["Date"] >= cw0) & (p["Date"] <= cw1)].reset_index(drop=True)
    td = pw["Date"].tolist(); close = pw["Close"].to_numpy(); rw = pw["RW"].to_numpy()
    state = np.where(rw >= up_thr, 1, np.where(rw <= dn_thr, -1, 0))  # fig1과 동일
    n = len(td)

    def comments(i, j):
        pi, pj = pos[td[i]], pos[td[j]]
        return int(cumv[pj] - (cumv[pi - 1] if pi > 0 else 0))

    bw = None
    for i in range(n):
        j = i
        while j < n and comments(i, j) < target:
            j += 1
        if j >= n:
            continue
        seg = state[i:j + 1]
        up = int((seg == 1).sum()); dn = int((seg == -1).sum())
        net = (close[j] / close[i] - 1) * 100
        key = (min(up, dn), -abs(up - dn), up + dn)    # 적은쪽 최대 → 균형 → 총수
        if bw is None or key > bw["key"]:
            bw = dict(start=td[i], end=td[j], cum=net, up=up, dn=dn, key=key,
                      comments=comments(i, j))
    return bw


def _window_users(df, stock, w):                      # 구간 내 댓글 작성 유저 집합
    return set(df[(df["종목명"] == stock) & (df["us_date"] >= w["start"])
                  & (df["us_date"] <= w["end"])]["닉네임_ID"])


# 평가 구간: 겹침이 (최소 + slack) 이내인 후보 중 균형 up-down 최대
#   1순위 학습셋 유저 겹침 최소(평가 형평성), 단 slack만큼 양보해 2순위 균형도 확보
def _best_eval_window(df, stock, target, train_users, slack_frac=0.05):
    cw0, cw1 = WINDOWS[stock]
    d = df[(df["종목명"] == stock) & (df["us_date"] >= cw0) & (df["us_date"] <= cw1)]
    idx = pd.date_range(cw0, cw1, freq="D").strftime("%Y-%m-%d")
    cumv = d.groupby("us_date").size().reindex(idx, fill_value=0).cumsum().to_numpy()
    ovv = (d[d["닉네임_ID"].isin(train_users)].groupby("us_date").size()
           .reindex(idx, fill_value=0).cumsum().to_numpy())
    pos = {dt: i for i, dt in enumerate(idx)}
    p, mu, sig = load_prices(stock)
    p = p.sort_values("Date").reset_index(drop=True)
    p["RW"] = p["Close"].pct_change(H_DEFAULT) * 100
    up_thr = mu * H_DEFAULT + K_DEFAULT * sig * np.sqrt(H_DEFAULT)
    dn_thr = mu * H_DEFAULT - K_DEFAULT * sig * np.sqrt(H_DEFAULT)
    pw = p[(p["Date"] >= cw0) & (p["Date"] <= cw1)].reset_index(drop=True)
    td = pw["Date"].tolist(); close = pw["Close"].to_numpy(); rw = pw["RW"].to_numpy()
    state = np.where(rw >= up_thr, 1, np.where(rw <= dn_thr, -1, 0))
    n = len(td)

    def cnt(cum_, i, j):
        pi, pj = pos[td[i]], pos[td[j]]
        return int(cum_[pj] - (cum_[pi - 1] if pi > 0 else 0))

    cands = []
    for i in range(n):
        j = i
        while j < n and cnt(cumv, i, j) < target:
            j += 1
        if j >= n:
            continue
        seg = state[i:j + 1]
        up = int((seg == 1).sum()); dn = int((seg == -1).sum())
        cands.append(dict(start=td[i], end=td[j], up=up, dn=dn, overlap=cnt(ovv, i, j),
                          cum=(close[j] / close[i] - 1) * 100, comments=cnt(cumv, i, j)))
    if not cands:
        return None
    thr = min(c["overlap"] for c in cands) + slack_frac * target   # 겹침 최소 + 여유
    elig = [c for c in cands if c["overlap"] <= thr]
    # 여유 안에서 균형 최대(적은쪽↑ → |차|↓ → 총수↑), 동률이면 겹침 최소
    return max(elig, key=lambda c: (min(c["up"], c["dn"]), -abs(c["up"] - c["dn"]),
                                    c["up"] + c["dn"], -c["overlap"]))


def fig_price_highlight(df):
    GREEN = "#1B7837"
    # 학습(20K): 균형 up-down. 평가(10K): 상대 종목 학습셋 유저와 최소 겹침 → 균형
    train = {s: _best_var_window(df, s, 20000) for s in ["TSLA", "NVDA"]}
    tusers = {s: _window_users(df, s, train[s]) for s in ["TSLA", "NVDA"]}
    test = {"NVDA": _best_eval_window(df, "NVDA", 10000, tusers["TSLA"]),  # Fold A 평가
            "TSLA": _best_eval_window(df, "TSLA", 10000, tusers["NVDA"])}  # Fold B 평가
    for s in ["TSLA", "NVDA"]:
        w = train[s]
        fig_price_story(s, suffix="_20k", htitle="(학습셋 ~2만건 · 균형 up-down)",
                        highlights=[dict(start=w["start"], end=w["end"], cum=w["cum"],
                                         up=w["up"], dn=w["dn"], comments=w["comments"],
                                         color=GREEN, tag="~2만 댓글 (학습)")])
        w = test[s]
        ou = len(_window_users(df, s, w) & tusers["NVDA" if s == "TSLA" else "TSLA"])
        fig_price_story(s, suffix="_10k",
                        htitle=f"(평가셋 ~1만건 · 학습셋 유저 최소 겹침: {ou:,}명)",
                        highlights=[dict(start=w["start"], end=w["end"], cum=w["cum"],
                                         up=w["up"], dn=w["dn"], comments=w["comments"],
                                         overlap=w["overlap"], color=GREEN,
                                         tag="~1만 댓글 (평가)")])


# ── B 공통 계산: (h, k)별 변동 지속율 · 유의미한 변동일 비율 ──
def _hk_curves(stock, k):
    """H_SWEEP 각 h에 대한 (변동 지속율 %, 유의미한 변동일 비율 %, 사건 수 n)."""
    p, mu, sig = load_prices(stock)
    p = p.reset_index(drop=True)
    cw0, cw1 = WINDOWS[stock]
    widx = p.index[(p["Date"] >= cw0) & (p["Date"] <= cw1)]
    close = p["Close"].to_numpy()
    N = len(p)
    cov, persist, counts = [], [], []
    for h in H_SWEEP:
        rw = (p["Close"].pct_change(h) * 100).to_numpy()
        u, d = mu * h + k * sig * np.sqrt(h), mu * h - k * sig * np.sqrt(h)
        flagged = []                           # (방향, 사후 h일 수익률)
        for i in widx:
            if np.isnan(rw[i]):
                continue
            sgn = 1 if rw[i] >= u else (-1 if rw[i] <= d else 0)
            if sgn == 0 or i + h >= N:
                continue
            flagged.append((sgn, close[i + h] / close[i] - 1))
        wvalid = (((p["Date"] >= cw0) & (p["Date"] <= cw1)) & ~np.isnan(rw)).sum()
        counts.append(len(flagged))
        cov.append(100 * len(flagged) / wvalid if wvalid else 0)
        if flagged:                            # 지속율 = 사후 h일 방향이 판정과 같은 비율
            cont = sum(1 for s, f in flagged if np.sign(f) == s)
            persist.append(100 * cont / len(flagged))
        else:
            persist.append(np.nan)
    return np.array(H_SWEEP), persist, cov, counts


# n 라벨 배치 코드 → (dx, dy, ha, va).  9pt=한 칸 위/아래, 7pt=옆
_LBL_OFF = {
    "above":       (0, 9, "center", "bottom"),
    "below":       (0, -9, "center", "top"),
    "right":       (8, 0, "left", "center"),     # 점 기준 오른쪽 동일 높이
    "right-above": (6, 7, "left", "bottom"),     # 대각선은 점에 살짝 가깝게
    "right-below": (6, -7, "left", "top"),
    "left-above":  (-6, 7, "right", "bottom"),
    "left-below":  (-6, -7, "right", "top"),
}
# 자동 규칙으로 안 맞는 점은 (종목, k, h)별로 수동 지정 (없으면 자동 규칙 사용)
_LBL_POS = {
    # 왼쪽 위 = TSLA·k=1.0σ
    ("TSLA", 1.0, 5): "left-below", ("TSLA", 1.0, 6): "left-above",
    ("TSLA", 1.0, 8): "right-above",
    # 오른쪽 위 = NVDA·k=1.0σ
    ("NVDA", 1.0, 5): "right", ("NVDA", 1.0, 6): "right-above",
    ("NVDA", 1.0, 7): "right-above", ("NVDA", 1.0, 8): "right-above",
    # 왼쪽 아래 = TSLA·k=1.5σ
    ("TSLA", 1.5, 3): "right-above", ("TSLA", 1.5, 4): "right",
    ("TSLA", 1.5, 5): "right-above", ("TSLA", 1.5, 6): "right-above",
    # 오른쪽 아래 = NVDA·k=1.5σ
    ("NVDA", 1.5, 5): "right-above", ("NVDA", 1.5, 6): "below",
    ("NVDA", 1.5, 7): "right-above",
}


# ── B. 변동 판정선 μh±kσ√h의 (k, h) 스윕 ──
#   위 행=k=1.0σ(채택), 아래 행=k=1.5σ(엄격) / 열=TSLA, NVDA
def fig_threshold_sweep():
    stocks = ["TSLA", "NVDA"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for r, k in enumerate(K_SWEEP):            # 행 = σ배수 k
        for c, stock in enumerate(stocks):     # 열 = 종목
            ax = axes[r][c]
            x, persist, cov, counts = _hk_curves(stock, k)
            l1, = ax.plot(x, persist, "o-", color="#D55E00", lw=2.3, ms=8,
                          label="변동 지속율 (↑)")
            ax2 = ax.twinx()
            l2, = ax2.plot(x, cov, "s--", color="#0072B2", lw=2.0, ms=7,
                           label="유의미한 변동일 비율 (≳)")
            # 사건 수 n = 유의미한 변동일 수 = 변동일 비율(파란)의 분자 → 파란 점에 표기
            # (작은 표본=빨강 경고: 그 n으로 계산한 지속율이 노이즈라는 뜻)
            # 선과 안 겹치게: 골(앞뒤↑)=아래, 마루(앞뒤↓)=위, 단조=낮은 이웃 쪽으로 수평
            cov_arr = np.array(cov, dtype=float)
            for idx, (xi, yi, n) in enumerate(zip(x, cov, counts)):
                left = cov_arr[idx - 1] if idx > 0 else None
                right = cov_arr[idx + 1] if idx < len(cov_arr) - 1 else None
                neigh = [u for u in (left, right) if u is not None]
                pos = _LBL_POS.get((stock, k, int(xi)))
                if pos is not None:                     # 수동 지정 우선
                    dx, dy, ha, va = _LBL_OFF[pos]
                elif all(u > yi for u in neigh):        # 골(앞뒤↑) → 아래
                    dx, dy, ha, va = 0, -9, "center", "top"
                elif all(u < yi for u in neigh):        # 마루(앞뒤↓) → 위
                    dx, dy, ha, va = 0, 9, "center", "bottom"
                elif left < right:                      # 상승세 → 위 + 상승(오른)쪽
                    dx, dy, ha, va = 7, 9, "left", "bottom"
                else:                                   # 하락세 → 아래 + 하락(오른)쪽
                    dx, dy, ha, va = 7, -9, "left", "top"
                small = n < SMALL_N
                ax2.annotate(f"n={n}", xy=(xi, yi), xytext=(dx, dy),
                             textcoords="offset points", ha=ha, va=va, fontsize=8,
                             color=("#C00000" if small else "0.35"),
                             fontweight=("bold" if small else "normal"))
            ax2.grid(False)                    # 트윈축 그리드 끔(이중 격자 방지)
            ax.set_axisbelow(True)             # 그리드는 선/마커 뒤로
            ax2.margins(y=0.15)                # n 라벨 들어갈 위쪽 여백(파란 축)
            if k == K_DEFAULT:                 # h=5 점선·라벨은 채택 행(k=1.0)에만
                ax.axvline(H_DEFAULT, color="0.4", ls="--", lw=1.3)
                h5_y = {"TSLA": 25, "NVDA": 32}[stock]   # 왼쪽 y축(지속율) 데이터 높이
                ax.text(H_DEFAULT + 0.25, h5_y, "h = 5 채택", ha="left", va="center",
                        fontsize=10, color="0.25", zorder=5)
            ax.set_title(f"{stock} (k={k})", fontsize=13, pad=8)
            ax.set_xticks(x)
            ax.set_ylabel("변동 지속율 (%)")           # 모든 칸에 양쪽 축 라벨
            ax2.set_ylabel("유의미한 변동일 비율 (%)")
            ax.set_xlabel("누적기간 길이 h (거래일)")   # 모든 행에 x축 라벨
            if r == 0 and c == 1:              # 범례는 한 번만
                # 빈 3번째 행으로 박스 높이만 확보 → 메모 텍스트는 루프 밖에서 왼쪽 끝 정렬로 얹음
                blank = Line2D([], [], linestyle="", marker="", label="\n ")
                # 가운데 라벨 꼬리 공백으로 박스 폭 확보(메모가 안 잘리게)
                note_leg = ax.legend(
                    handles=[l1, l2, blank],
                    labels=["변동 지속율 (↑)", "유의미한 변동일 비율 (≳)", " "],
                    fontsize=9.5, loc="upper right", framealpha=0.92)
                note_ax = ax
    fig.tight_layout()
    fig.subplots_adjust(top=0.835, hspace=0.32, wspace=0.22)  # 소제목↔그래프/행/열 간격
    fig.suptitle("변동 판정선 (μh ± kσ√h)의 Threshold Sweep (판정폭 계수 k, 누적기간 길이 h)",
                 fontsize=17, fontweight="bold", y=0.98)
    fig.text(0.5, 0.925,                             # 제목과는 더 벌리고
             "판정폭 계수 k를 1.5로 하면 유의미한 변동 사건 수가 급감하여 "
             "고수 라벨 거의 안 생기므로 k = 1.0 채택",
             ha="center", fontsize=10.5, color="0.45")
    fig.text(0.5, 0.902,                             # 소제목 두 줄 간격 살짝
             "변동 지속율(높을수록 진짜 사건)이 높으면서 유의미한 변동일 비율도 "
             "너무 적지 않은 h = 5 채택",
             ha="center", fontsize=10.5, color="0.45")
    # 메모를 범례 박스 왼쪽 끝(마커 열)에 맞춰 직접 얹기 — 핸들 칸 공백 없이 정렬
    note_leg.set_zorder(4)             # 메모 텍스트(아래)가 프레임 위로 오게
    fig.canvas.draw()
    binv = note_leg.get_window_extent().transformed(note_ax.transAxes.inverted())
    note_ax.text(binv.x0 + 0.012, binv.y0 + 0.008,
                 "n : 유의미한 변동 사건 수",
                 transform=note_ax.transAxes, fontsize=9.5, va="bottom", ha="left",
                 zorder=6, clip_on=False)
    _save(fig, "fig2_threshold_sweep.png")


# KcELECTRA 토큰 길이 (URL 제거 후). 캐시: raw_data/token_len_kcelectra.npy
def _token_lengths():
    cache = os.path.join(DATA, "token_len_kcelectra.npy")
    if os.path.exists(cache):
        return np.load(cache)
    from transformers import AutoTokenizer       # 토큰화 시에만 import
    txt = (load_comments()["내용"].astype(str)
           .str.replace(r"https?://\S+|www\.\S+", "", regex=True).str.strip())
    txt = txt[txt.str.len() >= 4].tolist()        # URL 제거 후 3자 이하 제외
    tok = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
    # 이 토크나이저는 post-processor가 없어 add_special_tokens가 [CLS]/[SEP]를
    # 안 붙임 → 내용 토큰만 세고 +2([CLS],[SEP]) 직접 보정
    lens = []
    for i in range(0, len(txt), 20000):
        enc = tok(txt[i:i + 20000], add_special_tokens=False, truncation=False,
                  padding=False, return_attention_mask=False,
                  return_token_type_ids=False)
        lens.extend(len(x) + 2 for x in enc["input_ids"])
    lens = np.array(lens)
    np.save(cache, lens)
    return lens


# ── C. 토큰 수 분포 (KcELECTRA, 선형 + broken axis로 긴 꼬리 연결) ──
def fig_length(df):
    s = _token_lengths()
    mn, med, p95, p99, mx = (s.min(), np.median(s), np.percentile(s, 95),
                             np.percentile(s, 99), s.max())
    BAR, INK, RED = "#7A9CC6", "#33404D", "#C0392B"
    BRK = 80                               # 본문/꼬리 분할 지점(토큰)
    fig, (axL, axR) = plt.subplots(
        1, 2, sharey=True, figsize=(15, 5),
        gridspec_kw={"width_ratios": [3, 1.25], "wspace": 0.05})

    # 좌: 본문(0~80) 세밀, 우: 꼬리(80~최대) 거친 bin
    axL.hist(s, bins=np.arange(0, BRK + 2, 2), color=BAR, edgecolor="white", lw=0.4)
    axR.hist(s, bins=np.arange(BRK, mx + 40, 40), color=BAR, edgecolor="white", lw=0.4)
    RXMAX = mx * 1.05                         # 우측 패널 끝(최대 토큰 + 여백)
    axL.set_xlim(0, BRK); axR.set_xlim(BRK, RXMAX)
    axR.set_xticks([250, 500, 1000])

    # 분위 점선 + 가로 라벨 (전부 기본 잉크색)
    specs = [(axL, mn, "최소", "left"), (axL, med, "중앙값", "left"),
             (axL, p95, "P95", "left"), (axR, p99, "P99", "left"),
             (axR, mx, "최대", "right")]
    for ax, v, name, side in specs:
        ax.axvline(v, color=INK, ls=(0, (4, 3)), lw=1.0, alpha=0.7, zorder=2)
        ax.annotate(f"{name} {v:.0f}개", xy=(v, 1.0),
                    xytext=(4 if side == "left" else -4, -5),
                    textcoords="offset points", xycoords=("data", "axes fraction"),
                    ha=side, va="top", fontsize=9, fontweight="bold", color=INK)

    # 512(KcELECTRA 한계) 초과 구간 — 투명 빨강 박스(얇은 실선 테두리) + 빨강 글씨
    from matplotlib.colors import to_rgba
    n512 = int((s > 512).sum()); p512 = 100 * n512 / len(s)
    axR.axvspan(512, RXMAX, facecolor=to_rgba(RED, 0.12), edgecolor=RED,
                lw=1.0, zorder=1.5)
    axR.annotate(f"Max Token 512개 초과\n({n512:,}건, {p512:.2f}%)",
                 xy=((512 + RXMAX) / 2, 0.55), xycoords=("data", "axes fraction"),
                 ha="center", va="center", fontsize=12.5, fontweight="bold",
                 color=RED, zorder=4)

    # 맞닿은 안쪽 축선 숨기고 // 사선 단절 표시 (matplotlib 공식 레시피)
    axL.spines["right"].set_visible(False)
    axR.spines["left"].set_visible(False)
    axR.tick_params(left=False)
    dd = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=10, linestyle="none",
              color="0.4", mec="0.4", mew=1, clip_on=False)
    axL.plot([1, 1], [0, 1], transform=axL.transAxes, **dd)
    axR.plot([0, 0], [0, 1], transform=axR.transAxes, **dd)

    axL.set_ylabel("댓글 수", fontsize=11)
    fig.supxlabel("댓글 토큰 수 (KcELECTRA base Tokenizer, [CLS]·[SEP] 포함)",
                  fontsize=11)
    fig.suptitle("댓글 토큰 수 분포 (TSLA + NVDA)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); _save(fig, "fig3_length_dist.png")


# ── D. 유저 메타 ────────────────────────────────────────────
def _user_meta_panel(df, label, fname):
    # 유저(닉네임_ID) 단위 집계 — 댓글 수 편향 제거
    g = df.groupby("닉네임_ID")
    u_share = g["주주"].max()                       # 한 번이라도 주주면 주주
    u_badge = g["has_badge"].max()
    nU = len(u_share)
    badged = df[df["has_badge"]]
    u_btype = badged.groupby("닉네임_ID")["뱃지"].agg(lambda x: x.mode().iloc[0])
    btype = u_btype.value_counts()[::-1]            # 뱃지 종류별 유저 수

    fig = plt.figure(figsize=(17, 5))
    # 파이 둘은 붙이고(좁은 wspace), 빈 스페이서 컬럼으로 막대그래프만 따로 넓게
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.18, 1.7], wspace=0.12,
                          top=0.82, bottom=0.1)   # 상단 제목과 그래프 간격 확보
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
            fig.add_subplot(gs[3])]
    axes[0].grid(False); axes[1].grid(False)       # 파이엔 그리드 X
    s1 = int(u_share.sum())
    axes[0].pie([s1, nU - s1], labels=["주주", "비주주"], autopct="%1.1f%%",
                colors=["#4C9F70", "#D0D0D0"], startangle=90)
    axes[0].set_title("유저 내 주주 점유율")
    b1 = int(u_badge.sum())
    axes[1].pie([b1, nU - b1], labels=["뱃지 보유자", "없음"], autopct="%1.1f%%",
                colors=["#E6A817", "#D0D0D0"], startangle=90)
    axes[1].set_title("유저 내 뱃지 보유자 점유율")
    axes[2].barh(btype.index, btype.values, color="#E6A817")
    tot_b = int(btype.sum())                       # 뱃지 보유 유저 수
    for yi, v in enumerate(btype.values):          # 막대 끝에 유저수·비율
        axes[2].text(v + tot_b * 0.012, yi, f"{v:,}명 ({100*v/tot_b:.1f}%)",
                     va="center", fontsize=9.5, color="#8A6500", fontweight="bold")
    axes[2].set_xlim(0, btype.max() * 1.32)
    axes[2].set_title("뱃지 종류별 유저 수")
    axes[2].set_xlabel("유저 수")
    fig.suptitle(f"유저 메타데이터 분포 ({label}, 유저 {nU:,}명)", fontweight="bold")
    _save(fig, fname)


def fig_user_meta(df):
    _user_meta_panel(df, "TSLA + NVDA", "fig4_user_meta.png")
    for stock in ["TSLA", "NVDA"]:
        _user_meta_panel(df[df["종목명"] == stock], stock,
                         f"fig4_user_meta_{stock}.png")


# ── 댓글-주가 일별 병합(미국 거래일 기준) ───────────────────
def _daily_merge(df, stock):
    # 비거래일(주말·공휴일) 댓글을 '다음 거래일(개장일)'에 합산 — 누락 0%
    cw0, cw1 = WINDOWS[stock]
    p, _, _ = load_prices(stock)
    p = (p[(p["Date"] >= cw0) & (p["Date"] <= cw1)][["Date", "daily_return_pct"]]
         .sort_values("Date").reset_index(drop=True))
    tdays = pd.to_datetime(p["Date"]).to_numpy()           # 거래일(오름차순)
    c = df[(df["종목명"] == stock) & (df["us_date"] >= cw0) & (df["us_date"] <= cw1)]
    cdate = pd.to_datetime(c["us_date"], errors="coerce").dropna().to_numpy()
    # 각 댓글 작성일 → 그 이상인 첫 거래일(=다음 개장일). 거래일이면 그대로.
    idx = np.clip(np.searchsorted(tdays, cdate, side="left"), 0, len(tdays) - 1)
    cnt = pd.Series(idx).value_counts()
    p["comments"] = [int(cnt.get(i, 0)) for i in range(len(p))]
    p["x"] = tdays
    return p


# ── E. 변동성 vs 댓글량 (시간축 오버레이 + 상관 통계) ────────
def fig_volatility_vs_volume(df):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, stock in zip(axes, ["TSLA", "NVDA"]):
        m = _daily_merge(df, stock); xx = m["x"].to_numpy()
        absret = m["daily_return_pct"].abs()
        ax.plot(xx, absret.to_numpy(), color="purple", lw=1.4, label="|일일 등락률|(%)")
        ax2 = ax.twinx()
        ax2.bar(xx, m["comments"].to_numpy(), color="gray", alpha=0.30, width=1.0,
                label="일일 댓글량")
        ax2.set_ylabel("일일 댓글량"); ax2.grid(False)
        from scipy import stats
        xv = absret.to_numpy(); yv = m["comments"].to_numpy()
        pr, pp = stats.pearsonr(xv, yv); sr, _ = stats.spearmanr(xv, yv)
        pf = "p < .001" if pp < 1e-3 else f"p = {pp:.3f}"
        ax.set_title(f"{stock} 변동성 vs 댓글량  "
                     f"(거래일 n={len(xv)}, Pearson r={pr:.2f}, "
                     f"Spearman ρ={sr:.2f}, {pf})",
                     fontweight="bold")
        ax.set_ylabel("|일일 등락률|(%)"); ax.legend(loc="upper left", fontsize=9)
    axes[1].set_xlabel("날짜")
    for ax in axes:                                # 위·아래 둘 다 날짜 눈금 표시
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.text(0.5, 0.005, "※ 주말·공휴일 댓글은 다음 거래일(개장일)에 합산 — "
             "개장일 댓글량이 누적 효과로 높을 수 있음",
             ha="center", fontsize=9, color="0.45")
    fig.tight_layout()
    # top: 헤더(제목+부제) 영역 충분히 확보 / hspace: subplot 간격 / bottom: 하단 주석 여백
    fig.subplots_adjust(top=0.83, hspace=0.55, bottom=0.16)
    fig.suptitle("주가 변동성과 댓글량의 상관관계", fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.93, "고변동 구간 분석 정당화", ha="center", fontsize=11, color="0.45")
    _save(fig, "fig5_volatility_timeseries.png")


# ── F. 등락률(빨강/파랑) vs 댓글량 ──────────────────────────
def fig_updown_vs_volume(df):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    for ax, stock in zip(axes, ["TSLA", "NVDA"]):
        m = _daily_merge(df, stock); xx = m["x"].to_numpy()
        colors = [UP if v >= 0 else DOWN for v in m["daily_return_pct"].fillna(0)]
        ax.bar(xx, m["daily_return_pct"].to_numpy(), color=colors, width=1.0)
        ax.set_ylabel("일일 등락률(%)\n(빨강=상승/파랑=하락)")
        ax2 = ax.twinx()
        ax2.plot(xx, m["comments"].to_numpy(), color="black", lw=1.2, label="일일 댓글량")
        ax2.set_ylabel("일일 댓글량"); ax2.grid(False)
        ax.set_title(f"{stock} 일일 등락률 vs 댓글량 — 등락 폭이 클수록 댓글 급증?",
                     fontweight="bold")
        ax2.legend(loc="upper left", fontsize=9)
    axes[1].set_xlabel("날짜")
    for ax in axes:                                # 위·아래 둘 다 날짜 눈금 표시
        ax.tick_params(axis="x", labelbottom=True)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    # fig5와 동일: 헤더 영역 확보(top) + subplot 간격(hspace)
    fig.subplots_adjust(top=0.83, hspace=0.55)
    fig.suptitle("일일 등락(전일 대비 %)과 댓글량·리액션 비중의 관계",
                 fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.93, "예측 vs 리액션 분리 정당화", ha="center", fontsize=11, color="0.45")
    _save(fig, "fig6_updown_vs_volume.png")


def main():
    df = load_comments()
    print(f"댓글 {len(df):,}건 | 작성자 {df['닉네임_ID'].nunique():,}명 | "
          f"주주 {df['주주'].mean():.1%} | 뱃지 {df['has_badge'].mean():.1%}")
    for stock in ["TSLA", "NVDA"]:
        fig_price_story(stock)
    # split 구간 커버리지는 2_eda/fig_splits_coverage.py 로 분리(→ data/unlabeled/splits/)
    fig_threshold_sweep()
    fig_length(df)
    fig_user_meta(df)
    fig_volatility_vs_volume(df)
    fig_updown_vs_volume(df)
    print("\n[보류·3-2] 4-Class 분포 / Class0 사유 / 시점관계 / 뱃지↔고수 / 일관성 — 라벨링 후")


if __name__ == "__main__":
    main()
