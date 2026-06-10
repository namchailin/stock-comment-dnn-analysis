# =============================================================================
#  Ablation arm B — 2-Tower(텍스트 + 과거 주가) 4-Class 분류기
#  arm A(lib/train_colab.py, 텍스트만)와 fold·loss·metric을 동일하게 맞춰 직접 비교.
#
#  설계 핵심(누수 방어):
#   - 가격은 "댓글 작성 시각 *직전*"까지의 시간봉만 입력(미래 컷오프 strict assert).
#   - 정답(Class 2·3)을 만든 *미래* 주가는 절대 입력에 넣지 않음 — 과거 가격은 누수 아님.
#   - arm A와 동일한 class-weighted focal loss / macro-F1 → A vs B 사과-대-사과 비교.
#
#  모델: KcELECTRA([CLS] pooled) ⊕ 가격 GRU(masked mean) → concat → MLP head.
#
#  [Colab 사용법] arm A와 동일. 추가로 시간별 가격 CSV 4개 필요:
#    TSLA_prices_1h.csv  NVDA_prices_1h.csv  (+ 학습/평가 final CSV)
#    !pip install -q -U transformers datasets accelerate scikit-learn
#    !python train_2tower.py
# =============================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModel, Trainer, TrainingArguments,
                          EarlyStoppingCallback)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── CONFIG (arm A와 공유하는 값은 동일하게) ──────────────────────────────────
DATA_DIR   = "data/labeled"
PRICE_DIR  = "data/unlabeled/raw"
FOLDS = {
    "A": dict(train="tsla_train_enriched_final.csv", train_ticker="TSLA",
              test="nvda_eval_final.csv",            test_ticker="NVDA",
              note="TSLA학습(보강)→NVDA평가"),
}
ENCODER = "beomi/KcELECTRA-base"     # 주력(arm A 벤치 최고와 동일 인코더 사용)

NUM_LABELS, LABELS, NAMES = 4, [0, 1, 2, 3], ["C0예측없음", "C1실패", "C2방향적중", "C3날짜적중"]

MAX_LEN = 512
VAL_SIZE = 0.15
EPOCHS = 6
BATCH = 16            # 2-tower라 메모리 여유 위해 A(32)보다↓
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
FOCAL_GAMMA = 2.0
WEIGHT_TEMPER = 0.5
SEED = 42
HOLDER_TOKENS = ["[주주]", "[비주주]"]

# 가격 타워 설정
PRICE_WIN   = 48     # 댓글 직전 시간봉 개수(≈7 거래일). 부족하면 앞쪽 zero-pad+mask.
PRICE_FEATS = 4      # [return, (C-O)/O, (H-L)/O, vol z-score(window-local)]
PRICE_HID   = 128    # GRU hidden(bi → 256 → proj 128)

OUT_DIR = "runs_2tower"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 가격 윈도우 추출기 (인과적, 미래 컷오프) ─────────────────────────────────
class PriceWindows:
    """종목별 시간봉을 1회 로드해두고, (작성일 직전) 윈도우를 O(logN)로 슬라이스."""
    def __init__(self, price_dir=PRICE_DIR, win=PRICE_WIN):
        self.win = win
        self.store = {}                      # ticker -> (dt_ns[np], feat_src DataFrame)
        for tk in ("TSLA", "NVDA"):
            p = pd.read_csv(os.path.join(price_dir, f"{tk}_prices_1h.csv"),
                            encoding="utf-8-sig")
            dt = pd.to_datetime(p["Datetime_KST"])           # KST naive
            p = p.assign(_dt=dt).sort_values("_dt").reset_index(drop=True)
            self.store[tk] = (p["_dt"].values.astype("datetime64[ns]"),
                              p[["Open", "High", "Low", "Close", "Volume", "return_pct"]])

    @staticmethod
    def _to_kst_naive(ts: str) -> np.datetime64:
        # 작성일은 tz-aware(+09:00). UTC 경유로 통일 후 KST로 변환, tz 제거.
        return np.datetime64(pd.Timestamp(ts).tz_convert("Asia/Seoul").tz_localize(None))

    def _features(self, rows: pd.DataFrame) -> np.ndarray:
        o, h, l, c = (rows[k].to_numpy(float) for k in ("Open", "High", "Low", "Close"))
        ret = np.nan_to_num(rows["return_pct"].to_numpy(float) / 100.0)
        co  = np.nan_to_num((c - o) / np.where(o == 0, np.nan, o))
        hl  = np.nan_to_num((h - l) / np.where(o == 0, np.nan, o))
        v   = np.log1p(np.clip(rows["Volume"].to_numpy(float), 0, None))
        vz  = (v - v.mean()) / (v.std() + 1e-6)              # window-local z (누수 없음)
        return np.stack([ret, co, hl, vz], axis=1).astype("float32")  # (n, 4)

    def get(self, ticker: str, written_at: str):
        """반환: seq (win, PRICE_FEATS) float32, mask (win,) float32. 미래봉 미포함 보장."""
        dts, src = self.store[ticker]
        t = self._to_kst_naive(written_at)
        end = int(np.searchsorted(dts, t, side="left"))      # t '미만' 개수 == strict 과거
        assert end == 0 or dts[end - 1] < t, "미래 컷오프 위반(과거봉만 허용)"
        start = max(0, end - self.win)
        rows = src.iloc[start:end]
        seq = np.zeros((self.win, PRICE_FEATS), dtype="float32")
        mask = np.zeros((self.win,), dtype="float32")
        if len(rows) > 0:
            f = self._features(rows)                         # (n, 4), 최신이 마지막
            seq[self.win - len(f):] = f                      # 뒤쪽 정렬, 앞쪽 zero-pad
            mask[self.win - len(f):] = 1.0
        return seq, mask


# ── 데이터셋 / 콜레이터 ──────────────────────────────────────────────────────
def make_inputs(df: pd.DataFrame) -> list:
    holder = df["주주_여부"].astype(str).str.lower().eq("true").map(
        {True: "[주주]", False: "[비주주]"})
    return (holder + " " + df["text"].astype(str)).tolist()


class TwoTowerDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tickers, written, tok, pw: PriceWindows):
        self.enc = tok(texts, truncation=True, max_length=MAX_LEN)
        self.labels = list(labels)
        self.seqs, self.masks = [], []
        for tk, ts in zip(tickers, written):
            s, m = pw.get(tk, ts)
            self.seqs.append(s); self.masks.append(m)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["price_seq"] = torch.tensor(self.seqs[i])
        item["price_mask"] = torch.tensor(self.masks[i])
        item["labels"] = int(self.labels[i])
        return item


class TwoTowerCollator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        price_seq = torch.stack([f.pop("price_seq") for f in feats])
        price_mask = torch.stack([f.pop("price_mask") for f in feats])
        labels = torch.tensor([f.pop("labels") for f in feats], dtype=torch.long)
        batch = self.tok.pad(feats, return_tensors="pt")     # input_ids/attention_mask 패딩
        batch["price_seq"] = price_seq
        batch["price_mask"] = price_mask
        batch["labels"] = labels
        return batch


# ── 토크나이저(특수토큰 보정) — arm A와 동일 ────────────────────────────────
def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": HOLDER_TOKENS})
    ids = tok("테스트")["input_ids"]
    if tok.cls_token_id is not None and (len(ids) == 0 or ids[0] != tok.cls_token_id):
        from tokenizers.processors import TemplateProcessing
        tok._tokenizer.post_processor = TemplateProcessing(
            single=f"{tok.cls_token} $A {tok.sep_token}",
            pair=f"{tok.cls_token} $A {tok.sep_token} $B:1 {tok.sep_token}:1",
            special_tokens=[(tok.cls_token, tok.cls_token_id),
                            (tok.sep_token, tok.sep_token_id)])
        chk = tok("테스트")["input_ids"]
        assert chk[0] == tok.cls_token_id and chk[-1] == tok.sep_token_id, "CLS/SEP 보정 실패"
        print(f"    [특수토큰 보정] {model_name}: TemplateProcessing 적용")
    return tok


# ── 2-Tower 모델 (loss를 forward에 내장 → Trainer 그대로 사용) ───────────────
class TwoTowerClassifier(nn.Module):
    def __init__(self, encoder_name, num_labels, class_weights, focal_gamma):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size
        self.price_gru = nn.GRU(PRICE_FEATS, PRICE_HID, batch_first=True, bidirectional=True)
        self.price_proj = nn.Linear(PRICE_HID * 2, PRICE_HID)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Sequential(
            nn.Linear(h + PRICE_HID, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_labels))
        self.register_buffer("class_weights", class_weights)
        self.focal_gamma = focal_gamma
        self.num_labels = num_labels

    def resize_token_embeddings(self, n):
        self.encoder.resize_token_embeddings(n)

    def _text_emb(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output                          # [CLS] pooled
        return out.last_hidden_state[:, 0]                    # pooler 없으면 [CLS] 토큰

    def _price_emb(self, price_seq, price_mask):
        seq, _ = self.price_gru(price_seq)                    # (B, win, 2H)
        m = price_mask.unsqueeze(-1)                          # (B, win, 1)
        summed = (seq * m).sum(1)
        denom = m.sum(1).clamp(min=1.0)                       # 빈 윈도우 보호
        return self.price_proj(summed / denom)                # masked mean → (B, H)

    def forward(self, input_ids=None, attention_mask=None,
                price_seq=None, price_mask=None, labels=None, **kw):
        t = self._text_emb(input_ids, attention_mask)
        p = self._price_emb(price_seq, price_mask)
        logits = self.head(self.dropout(torch.cat([t, p], dim=-1)))
        loss = None
        if labels is not None:
            w = self.class_weights.to(logits.device)
            ce = F.cross_entropy(logits, labels, weight=w, reduction="none")
            if self.focal_gamma > 0:
                pt = torch.exp(-F.cross_entropy(logits, labels, reduction="none"))
                loss = ((1 - pt) ** self.focal_gamma * ce).mean()
            else:
                loss = ce.mean()
        return SequenceClassifierOutput(loss=loss, logits=logits)


def metrics_fn(p):
    preds = p.predictions.argmax(-1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "acc": accuracy_score(p.label_ids, preds)}


def report(name, y_true, y_pred):
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n  ── {name} ──  macro-F1 = {mf1:.4f}  acc = {accuracy_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, labels=LABELS, target_names=NAMES,
                                digits=3, zero_division=0))
    print("  혼동행렬 (행=실제, 열=예측):\n", confusion_matrix(y_true, y_pred, labels=LABELS))
    return mf1


# ── 학습 + 교차평가 ──────────────────────────────────────────────────────────
def load_fold(fold):
    f = FOLDS[fold]
    tr = pd.read_csv(os.path.join(DATA_DIR, f["train"]), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(DATA_DIR, f["test"]), encoding="utf-8-sig")
    for d in (tr, te):
        d.dropna(subset=["Class", "text", "작성일"], inplace=True)
        d["Class"] = d["Class"].astype(int)
    return tr, te, f


def run_fold(fold, pw: PriceWindows):
    tr_df, te_df, f = load_fold(fold)
    tok = build_tokenizer(ENCODER)
    Xtr, ytr = make_inputs(tr_df), tr_df["Class"].to_numpy()
    wtr = tr_df["작성일"].tolist();  tkr_tr = [f["train_ticker"]] * len(tr_df)
    Xte, yte = make_inputs(te_df), te_df["Class"].to_numpy()
    wte = te_df["작성일"].tolist();  tkr_te = [f["test_ticker"]] * len(te_df)

    idx = np.arange(len(Xtr))
    it, iv = train_test_split(idx, test_size=VAL_SIZE, stratify=ytr, random_state=SEED)
    g = lambda L, ix: [L[i] for i in ix]
    ds_tr = TwoTowerDS(g(Xtr, it), ytr[it], g(tkr_tr, it), g(wtr, it), tok, pw)
    ds_va = TwoTowerDS(g(Xtr, iv), ytr[iv], g(tkr_tr, iv), g(wtr, iv), tok, pw)
    ds_te = TwoTowerDS(Xte, yte, tkr_te, wte, tok, pw)

    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=ytr[it])
    cw = torch.tensor(cw, dtype=torch.float) ** WEIGHT_TEMPER

    model = TwoTowerClassifier(ENCODER, NUM_LABELS, cw, FOCAL_GAMMA)
    model.resize_token_embeddings(len(tok))

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, fold),
        num_train_epochs=EPOCHS, learning_rate=LR,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=32,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, seed=SEED, report_to="none",
        remove_unused_columns=False,                          # price_seq/mask 보존 필수
        fp16=torch.cuda.is_available())
    trainer = Trainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=TwoTowerCollator(tok), compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    pred = trainer.predict(ds_te).predictions.argmax(-1)
    return report(f"[B/2-tower] {f['note']}", yte, pred)


def main():
    print(f"device={DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    pw = PriceWindows()
    print("\n===== [arm B] 2-Tower(텍스트+과거주가) — Fold A =====")
    mf1 = run_fold("A", pw)
    print(f"\n완료. arm B macro-F1 = {mf1:.4f}")
    print("※ arm A(텍스트만, train_colab.py) 결과와 같은 fold/loss/metric으로 비교.")


if __name__ == "__main__":
    main()
