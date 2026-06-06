# =============================================================================
#  4-Class 주가-예측-언어 분류기 — Colab(A100) 독립 실행 스크립트
#  이 파일 + final CSV 4개만 있으면 됨. 로컬 레포 다른 모듈에 의존하지 않음.
#
#  [Colab 사용법]
#   1) 런타임 → GPU(A100) 선택
#   2) final CSV 4개를 Colab에 업로드(또는 Drive 마운트) 후 DATA_DIR 지정:
#        final_train_A_TSLA.csv  final_test_A_NVDA.csv   (Fold A: TSLA학습→NVDA평가)
#        final_train_B_NVDA.csv  final_test_B_TSLA.csv   (Fold B: NVDA학습→TSLA평가)
#   3) !pip install -q -U transformers datasets accelerate scikit-learn
#   4) !python train_colab.py            (또는 셀에 붙여넣고 main() 실행)
#
#  설계(plan Step4·5):
#   - 입력 = 텍스트 + 주주여부([주주]/[비주주] 토큰 prepend). 주가·뱃지·미래정보 미포함.
#   - KcELECTRA 특수토큰 함정 보정(TemplateProcessing으로 [CLS]/[SEP] 강제).
#   - 불균형: class weight + macro-F1 기준 early stopping(학습셋서 val 분리 → 평가셋 누수 차단).
#   - 벤치마크: Fold A 단방향(TSLA학습→NVDA평가)에서 4모델 비교 → macro-F1 최고 선택.
#   - 베이스라인: 다수클래스 / TF-IDF+LogReg.
#   ※ 비용상 Fold A 단방향만 진행(양방향 교차 생략). Fold B 돌리려면 main()에 run_model(best,...,"B") 추가.
# =============================================================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          EarlyStoppingCallback)

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_DIR = "."                  # 업로드한 CSV 위치
# 2차(expert_control)로 C2/C3 보강한 enriched 학습셋 사용. 평가셋은 자연분포 유지(누수 없음).
FOLDS = {
    "A": dict(train="tsla_train_enriched_final.csv", test="nvda_eval_final.csv",
              note="TSLA학습(보강)→NVDA평가"),
}
# 벤치마크 후보. ModernBERT-ko 체크포인트는 분류 적합성 확인 후 교체 가능(plan 주의).
MODELS = {
    "KcELECTRA":    "beomi/KcELECTRA-base",        # 주력(댓글 도메인)
    "KLUE-RoBERTa": "klue/roberta-base",           # 한국어 NLU 표준 레퍼런스
    "KR-FinBERT":   "snunlp/KR-FinBert",           # 금융 도메인(ablation 성격)
    # "ModernBERT-ko": "<MLM 계열 한국어 ModernBERT 체크포인트>",  # 확인 후 주석 해제
}
# ── F1 개선 노브 ──────────────────────────────────────────
#  MERGE_C1C2=True  → C1(실패)·C2(적중) 병합 = 3-class(C0 / 예측함 / 날짜적중).
#    텍스트로 적중/실패 구분은 본질적으로 불가(미래정보 필요) → 학습가능 신호만 평가, macro-F1↑.
MERGE_C1C2 = False
if MERGE_C1C2:
    NUM_LABELS, LABELS, NAMES = 3, [0, 1, 2], ["C0예측없음", "C1+2 예측함", "C3날짜적중"]
else:
    NUM_LABELS, LABELS, NAMES = 4, [0, 1, 2, 3], ["C0예측없음", "C1실패", "C2방향적중", "C3날짜적중"]

MAX_LEN = 512          # plan 확정. 짧은 댓글은 dynamic padding으로 효율 처리
VAL_SIZE = 0.15        # 학습셋에서 떼는 검증 비율(early stopping 용, 평가셋과 분리)
EPOCHS = 6
BATCH = 32
LR = 1e-5              # 과적합 완화(2e-5→1e-5)
WEIGHT_DECAY = 0.01    # 정규화
WARMUP_RATIO = 0.1
FOCAL_GAMMA = 2.0      # focal loss(0이면 일반 CE). 쉬운 다수클래스 비중↓ → 소수클래스 집중
WEIGHT_TEMPER = 0.5    # class weight 완화 지수(1=원래, 0.5=sqrt) → C1 과예측 억제
SEED = 42
HOLDER_TOKENS = ["[주주]", "[비주주]"]
OUT_DIR = "runs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── 데이터 ───────────────────────────────────────────────────────────────────
def make_inputs(df: pd.DataFrame) -> list:
    """텍스트 앞에 [주주]/[비주주] 토큰 prepend (입력 = 텍스트 + 주주여부)."""
    holder = df["주주_여부"].astype(str).str.lower().eq("true").map(
        {True: "[주주]", False: "[비주주]"})
    return (holder + " " + df["text"].astype(str)).tolist()


def load_fold(fold: str):
    f = FOLDS[fold]
    tr = pd.read_csv(os.path.join(DATA_DIR, f["train"]), encoding="utf-8-sig")
    te = pd.read_csv(os.path.join(DATA_DIR, f["test"]), encoding="utf-8-sig")
    for d in (tr, te):
        d.dropna(subset=["Class", "text"], inplace=True)
        d["Class"] = d["Class"].astype(int)
        if MERGE_C1C2:                       # C1·C2 병합: {0:0, 1:1, 2:1, 3:2}
            d["Class"] = d["Class"].map({0: 0, 1: 1, 2: 1, 3: 2})
    return tr, te, f["note"]


class TextDS(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tok):
        self.enc = tok(texts, truncation=True, max_length=MAX_LEN)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["labels"] = int(self.labels[i])
        return item


# ── 토크나이저(특수토큰 함정 보정) ───────────────────────────────────────────
def build_tokenizer(model_name: str):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": HOLDER_TOKENS})
    # KcELECTRA 등 post_processor 없는 토크나이저: [CLS]/[SEP] 자동추가 안 됨 → 강제
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
        print(f"    [특수토큰 보정] {model_name}: [CLS]/[SEP] TemplateProcessing 적용")
    return tok


class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **k):
        super().__init__(*a, **k)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = self.class_weights.to(out.logits.device)
        ce = F.cross_entropy(out.logits, labels, weight=w, reduction="none")
        if FOCAL_GAMMA > 0:                              # focal: 쉬운 샘플 비중↓
            pt = torch.exp(-F.cross_entropy(out.logits, labels, reduction="none"))
            loss = ((1 - pt) ** FOCAL_GAMMA * ce).mean()
        else:
            loss = ce.mean()
        return (loss, out) if return_outputs else loss


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


# ── 단일 모델 fine-tuning + 교차평가 ─────────────────────────────────────────
def run_model(model_name, hf_id, tr_df, te_df, tag):
    tok = build_tokenizer(hf_id)
    Xtr = make_inputs(tr_df); ytr = tr_df["Class"].to_numpy()
    Xte = make_inputs(te_df); yte = te_df["Class"].to_numpy()
    # 학습셋 내부 train/val(early stopping용) — 평가셋(타 종목)은 손대지 않음
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=VAL_SIZE,
                                      stratify=ytr, random_state=SEED)
    ds_tr, ds_va, ds_te = TextDS(Xt, yt, tok), TextDS(Xv, yv, tok), TextDS(Xte, yte, tok)

    model = AutoModelForSequenceClassification.from_pretrained(hf_id, num_labels=NUM_LABELS)
    model.resize_token_embeddings(len(tok))     # [주주]/[비주주] 추가분 반영

    cw = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=yt)
    cw = torch.tensor(cw, dtype=torch.float) ** WEIGHT_TEMPER   # 가중 완화(C1 과예측 억제)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, f"{tag}_{model_name}"),
        num_train_epochs=EPOCHS, learning_rate=LR,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        eval_strategy="epoch", save_strategy="epoch", logging_steps=50,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        greater_is_better=True, save_total_limit=1, seed=SEED, report_to="none",
        fp16=torch.cuda.is_available())
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics_fn,
        class_weights=cw, callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    trainer.train()
    pred = trainer.predict(ds_te).predictions.argmax(-1)
    mf1 = report(f"[{tag}] {model_name} ({FOLDS[tag[-1]]['note'] if tag[-1] in FOLDS else tag})",
                 yte, pred)
    return mf1


# ── 베이스라인 ───────────────────────────────────────────────────────────────
def baselines(tr_df, te_df, tag):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    Xtr, ytr = make_inputs(tr_df), tr_df["Class"].to_numpy()
    Xte, yte = make_inputs(te_df), te_df["Class"].to_numpy()
    # 다수 클래스
    maj = np.bincount(ytr, minlength=NUM_LABELS).argmax()
    report(f"[{tag}] 베이스라인-다수클래스(={maj})", yte, np.full_like(yte, maj))
    # TF-IDF + LogReg
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    Ztr, Zte = vec.fit_transform(Xtr), vec.transform(Xte)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1)
    lr.fit(Ztr, ytr)
    report(f"[{tag}] 베이스라인-TFIDF+LogReg", yte, lr.predict(Zte))


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    print(f"device={DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)
    trA, teA, _ = load_fold("A")
    print("학습셋 Class 분포(A-train):", dict(trA["Class"].value_counts().sort_index()))

    # [1] 벤치마크: Fold A 단방향(TSLA학습→NVDA평가)에서 4모델 비교 → 최고 선택
    print("\n===== [1] 모델 벤치마크 (Fold A: TSLA학습→NVDA평가) =====")
    scores = {}
    for name, hf_id in MODELS.items():
        try:
            scores[name] = run_model(name, hf_id, trA, teA, "A")
        except Exception as e:
            print(f"  ⚠️ {name} 실패: {e}")
    best = max(scores, key=scores.get)
    print("\n벤치마크 macro-F1:", {k: round(v, 4) for k, v in scores.items()})
    print(f"→ 최고 모델: {best} (macro-F1 {scores[best]:.4f})")

    # [2] 베이스라인 (Fold A)
    print("\n===== [2] 베이스라인 (Fold A) =====")
    baselines(trA, teA, "A")
    print("\n완료. 단방향(TSLA→NVDA) 평가: macro-F1·클래스별 PR·4x4 혼동행렬 + 베이스라인 대비.")
    print("※ 양방향 교차(B: NVDA→TSLA)는 비용상 생략 — 단방향 일반화 결과만 보고.")


if __name__ == "__main__":
    main()
