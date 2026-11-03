"""
Fine-tune biomedical-ner-all on prescription data with LoRA (v2)
=================================================================
Changes from v1:
  * labels come from model.config.label2id (v1 used a wrong hand-written id table)
  * trained on build_ner_dataset.py output, whose character offsets are exact
  * model selection by VALIDATION entity-level F1 (v1 used training loss)
  * per-epoch metrics are written to training_logs/training_log.json

Run:
    python build_ner_dataset.py     # once
    python train_ner_lora.py        # trains, saves best adapter to ./finetuned_model
    python evaluate_ner.py          # compare base / fine-tuned / regex / hybrid
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ner_common import decode_spans, example_to_token_labels, gold_spans, score

warnings.filterwarnings("ignore")

BASE_MODEL = "d4data/biomedical-ner-all"
OUTPUT_DIR = Path("./finetuned_model")
DATA_DIR = Path("./training_data")
LOG_DIR = Path("./training_logs")
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 6
LEARNING_RATE = 3e-4
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["q_lin", "k_lin", "v_lin", "lin1", "lin2"]


class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label2id):
        self.items = []
        for ex in data:
            enc = tokenizer(ex["text"], max_length=MAX_LENGTH, truncation=True, return_offsets_mapping=True)
            offsets = [tuple(o) for o in enc["offset_mapping"]]
            labels = example_to_token_labels(ex["text"], ex["entities"], offsets, label2id)
            self.items.append({"input_ids": enc["input_ids"], "labels": labels})

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    n = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        k = len(b["input_ids"])
        ids[i, :k] = torch.tensor(b["input_ids"])
        lab[i, :k] = torch.tensor(b["labels"])
        att[i, :k] = 1
    return {"input_ids": ids, "attention_mask": att, "labels": lab}


@torch.no_grad()
def predict_spans(model, tokenizer, data, id2label):
    """Entity-level predictions for a list of examples (same decoder used in evaluate_ner.py)."""
    model.eval()
    out = []
    for ex in data:
        enc = tokenizer(ex["text"], max_length=MAX_LENGTH, truncation=True, return_offsets_mapping=True, return_tensors="pt")
        offsets = [tuple(o) for o in enc.pop("offset_mapping")[0].tolist()]
        logits = model(**enc).logits[0]
        out.append(decode_spans(ex["text"], offsets, logits.argmax(-1).tolist(), id2label))
    return out


def load(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_data, val_data = load("train"), load("val")
    print(f"train={len(train_data)}  val={len(val_data)}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForTokenClassification.from_pretrained(BASE_MODEL)
    label2id, id2label = model.config.label2id, {int(k): v for k, v in model.config.id2label.items()}

    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.TOKEN_CLS, r=LORA_R, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGET_MODULES, bias="none",
    ))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    ds = NERDataset(train_data, tokenizer, label2id)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate(b, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_RATIO * total_steps), total_steps)

    val_gold = [gold_spans(e["entities"]) for e in val_data]
    log, best_f1 = [], -1.0
    t0 = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        tot_loss, correct, count = 0.0, 0, 0
        for batch in loader:
            out = model(**batch)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            tot_loss += out.loss.item()
            mask = batch["labels"] != -100
            correct += (out.logits.argmax(-1)[mask] == batch["labels"][mask]).sum().item()
            count += mask.sum().item()

        val_pred = predict_spans(model, tokenizer, val_data, id2label)
        val_strict = score(val_gold, val_pred)["micro"]
        val_len = score(val_gold, val_pred, lenient=True)["micro"]
        row = {
            "epoch": epoch,
            "train_loss": round(tot_loss / len(loader), 4),
            "train_token_acc": round(correct / count, 4),
            "val_f1_strict": val_strict["f1"], "val_precision_strict": val_strict["precision"],
            "val_recall_strict": val_strict["recall"], "val_f1_lenient": val_len["f1"],
            "elapsed_s": round(time.time() - t0, 1),
        }
        log.append(row)
        print(row)

        if val_strict["f1"] > best_f1:
            best_f1 = val_strict["f1"]
            save(model, tokenizer, epoch, best_f1, trainable, total)
            print(f"  * best val F1 {best_f1:.4f} -> adapter saved")

    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "training_log.json", "w") as f:
        json.dump({"best_val_f1_strict": best_f1, "epochs": log}, f, indent=2)
    print(f"done. best val F1 (strict) = {best_f1:.4f}; log -> {LOG_DIR / 'training_log.json'}")


def save(model, tokenizer, epoch, val_f1, trainable, total):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    with open(OUTPUT_DIR / "training_config.json", "w") as f:
        json.dump({
            "version": 2,
            "base_model": BASE_MODEL,
            "lora_config": {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT, "target_modules": LORA_TARGET_MODULES},
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "epochs_total": NUM_EPOCHS,
            "best_epoch": epoch, "best_val_f1_strict": val_f1, "learning_rate": LEARNING_RATE,
            "trainable_params": trainable, "total_params": total, "seed": SEED,
            "labels": "base-model label space (see ner_common.OURS_TO_BASE)",
        }, f, indent=2)


if __name__ == "__main__":
    train()
