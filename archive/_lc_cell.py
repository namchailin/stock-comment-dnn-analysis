
# ============================================================================
#  러닝커브 — 학습량 vs F1 (특히 C2 방향적중)
#   "C2-F1이 데이터 늘수록 계속 오르면 → 데이터부족(b), 평평하면 → 신호한계(a)"
#   위 train_colab 기계장치(build_tokenizer·TextDS·WeightedTrainer 등) 재사용.
#   최고 모델(KLUE-RoBERTa) 1개로, 빠르게(3epoch).
# ============================================================================
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

LC_HF = "klue/roberta-base"          # [1]단계 벤치마크 최고 모델
SIZES = [1000, 2500, 5000, 10000, None]   # None = 전체(14,991)
LC_EPOCHS = 3                         # 러닝커브는 여러번 돌려서 짧게

trA, teA, _ = load_fold("A")
Xte = make_inputs(teA); yte = teA["Class"].to_numpy()


def _train_eval(sub_df):
    tok = build_tokenizer(LC_HF)
    Xtr = make_inputs(sub_df); ytr = sub_df["Class"].to_numpy()
    Xt, Xv, yt, yv = train_test_split(Xtr, ytr, test_size=VAL_SIZE,
                                      stratify=ytr, random_state=SEED)
    ds_tr, ds_va, ds_te = TextDS(Xt, yt, tok), TextDS(Xv, yv, tok), TextDS(Xte, yte, tok)
    model = AutoModelForSequenceClassification.from_pretrained(LC_HF, num_labels=NUM_LABELS)
    model.resize_token_embeddings(len(tok))
    cw = torch.tensor(compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=yt),
                      dtype=torch.float) ** WEIGHT_TEMPER
    args = TrainingArguments(
        output_dir="lc_run", num_train_epochs=LC_EPOCHS, learning_rate=LR,
        weight_decay=WEIGHT_DECAY, warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        eval_strategy="no", save_strategy="no", seed=SEED, report_to="none",
        fp16=torch.cuda.is_available())
    tr = WeightedTrainer(model=model, args=args, train_dataset=ds_tr, eval_dataset=ds_va,
                         data_collator=DataCollatorWithPadding(tok),
                         compute_metrics=metrics_fn, class_weights=cw)
    tr.train()
    pred = tr.predict(ds_te).predictions.argmax(-1)
    pc = f1_score(yte, pred, average=None, labels=LABELS, zero_division=0)
    macro = f1_score(yte, pred, average="macro")
    del model, tr
    torch.cuda.empty_cache()
    return macro, pc


rows = []
for sz in SIZES:
    sub = trA if (sz is None or sz >= len(trA)) else train_test_split(
        trA, train_size=sz, stratify=trA["Class"], random_state=SEED)[0]
    n2 = int((sub["Class"] == 2).sum()); n3 = int((sub["Class"] == 3).sum())
    macro, pc = _train_eval(sub)
    rows.append([len(sub), macro] + list(pc) + [n2, n3])
    print(f"size={len(sub):>6} | macro={macro:.3f} | "
          + " ".join(f"{NAMES[i]}={pc[i]:.3f}" for i in range(len(LABELS)))
          + f" | n_C2={n2} n_C3={n3}", flush=True)

lc = pd.DataFrame(rows, columns=["size", "macro"] + NAMES + ["n_C2", "n_C3"])
print("\n", lc.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.plot(lc["size"], lc["macro"], "o-", label="macro-F1")
if "C2방향적중" in lc:
    plt.plot(lc["size"], lc["C2방향적중"], "s-", label="C2 방향적중 F1")
if "C3날짜적중" in lc:
    plt.plot(lc["size"], lc["C3날짜적중"], "^-", label="C3 날짜적중 F1")
plt.xlabel("학습 표본 수"); plt.ylabel("F1 (NVDA 평가셋)")
plt.title("러닝커브 — 계속 ↑면 데이터부족(b) / 평평하면 신호한계(a)")
plt.legend(); plt.grid(alpha=0.3)
plt.savefig("learning_curve.png", dpi=130, bbox_inches="tight"); plt.show()
print("\n[해석] C2-F1이 size 늘수록 계속 오르면 → (b) 데이터부족. "
      "마지막 구간에서 평평/정체면 → (a) 텍스트 신호 한계.")
