from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.task3.data_loader import DEFAULT_MASTER_PATH, ROOT_DIR, SENTIMENT_TO_ID, load_t3_records


DEFAULT_MODEL_NAME = "indolem/indobertweet-base-uncased"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models" / "task3"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class T3TorchDataset(Dataset):

    def __init__(self, records: Sequence[Dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.records = list(records)
        self.texts = [r["text"] for r in self.records]
        self.sent_labels = [
            SENTIMENT_TO_ID[r["sentiment"]] if r.get("sentiment") is not None else -100
            for r in self.records
        ]
        self.sarc_labels = [
            int(r["is_sarcasm"]) if r.get("is_sarcasm") is not None else -100
            for r in self.records
        ]
        self.encodings = tokenizer(
            self.texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "id": self.records[idx]["id"],
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "sent_label": torch.tensor(self.sent_labels[idx], dtype=torch.long),
            "sarc_label": torch.tensor(self.sarc_labels[idx], dtype=torch.long),
        }


class IndoBERTweetMultiTask(nn.Module):

    def __init__(self, model_name: str, dropout: float = 0.2, from_pretrained: bool = True) -> None:
        super().__init__()
        if from_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
            self.config = self.encoder.config
        else:
            self.config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(self.config)
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.sentiment_head = nn.Linear(hidden_size, 3)
        self.sarcasm_head = nn.Linear(hidden_size, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = self.dropout(outputs.last_hidden_state[:, 0, :])
        sent_logits = self.sentiment_head(cls_repr)
        sarc_logits = self.sarcasm_head(cls_repr)
        return sent_logits, sarc_logits


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    valid = [y for y in labels if y != -100]
    counts = Counter(valid)
    total = max(1, len(valid))
    weights = [
        total / (num_classes * max(1, counts.get(c, 0)))
        for c in range(num_classes)
    ]
    mean_w = sum(weights) / num_classes
    normalized = [w / mean_w for w in weights]
    return torch.tensor(normalized, dtype=torch.float32)


@torch.no_grad()
def predict_loader(
    model: IndoBERTweetMultiTask,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    sent_true: List[int] = []
    sent_pred: List[int] = []
    sarc_true: List[int] = []
    sarc_pred: List[int] = []

    t0 = time.perf_counter()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        sent_logits, sarc_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        sent_probs = torch.softmax(sent_logits, dim=-1).cpu().tolist()
        sarc_probs = torch.softmax(sarc_logits, dim=-1)[:, 1].cpu().tolist()

        for rid, sp, scp, sl, scl in zip(
            batch["id"],
            sent_probs,
            sarc_probs,
            batch["sent_label"].tolist(),
            batch["sarc_label"].tolist(),
        ):
            rows.append(
                {
                    "id": rid,
                    "sent_probs": [round(float(x), 6) for x in sp],
                    "sarc_prob": round(float(scp), 6),
                }
            )
            if sl != -100:
                sent_true.append(sl)
                sent_pred.append(max(range(3), key=lambda i: sp[i]))
            if scl != -100:
                sarc_true.append(scl)
                sarc_pred.append(1 if scp >= 0.5 else 0)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    sent_f1 = (
        float(f1_score(sent_true, sent_pred, labels=[0, 1, 2], average="macro", zero_division=0))
        if sent_true
        else 0.0
    )
    sarc_f1 = (
        float(f1_score(sarc_true, sarc_pred, pos_label=1, average="binary", zero_division=0))
        if sarc_true
        else 0.0
    )
    metrics = {
        "sent_macro_f1": round(sent_f1, 4),
        "sarc_f1": round(sarc_f1, 4),
        "combined_score": round((sent_f1 + sarc_f1) / 2.0, 4),
        "latency_ms_per_sample": round(elapsed_ms / max(1, len(rows)), 4),
    }
    return rows, metrics


def save_predictions_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_single_seed(
    seed: int,
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.fp16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = T3TorchDataset(train_records, tokenizer, args.max_length)
    val_ds = T3TorchDataset(val_records, tokenizer, args.max_length)
    test_ds = T3TorchDataset(test_records, tokenizer, args.max_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False)

    sent_weights = compute_class_weights(train_ds.sent_labels, 3).to(device)
    sarc_weights = compute_class_weights(train_ds.sarc_labels, 2).to(device)
    sent_loss_fn = nn.CrossEntropyLoss(weight=sent_weights, ignore_index=-100)
    sarc_loss_fn = nn.CrossEntropyLoss(weight=sarc_weights, ignore_index=-100)

    model = IndoBERTweetMultiTask(args.model_name, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    seed_dir = args.output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = seed_dir / "model_weights.pt"

    best_score = -1.0
    no_improve = 0
    history: List[Dict[str, Any]] = []
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            sent_labels = batch["sent_label"].to(device)
            sarc_labels = batch["sarc_label"].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                sent_logits, sarc_logits = model(input_ids=input_ids, attention_mask=attention_mask)
                losses: List[torch.Tensor] = []
                if (sent_labels != -100).any():
                    losses.append(sent_loss_fn(sent_logits, sent_labels))
                if (sarc_labels != -100).any():
                    losses.append(sarc_loss_fn(sarc_logits, sarc_labels))
                loss = sum(losses)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += float(loss.item())
            steps += 1

        _, val_metrics = predict_loader(model, val_loader, device)
        avg_loss = round(running_loss / max(1, steps), 4)
        epoch_log = {"epoch": epoch, "train_loss": avg_loss, **val_metrics}
        history.append(epoch_log)
        print(f"[seed={seed} epoch={epoch}/{args.epochs}] {epoch_log}")

        if val_metrics["combined_score"] > best_score:
            best_score = val_metrics["combined_score"]
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[seed={seed}] Early stopping at epoch {epoch}.")
                break

    train_seconds = round(time.perf_counter() - train_start, 2)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    val_rows, final_val_metrics = predict_loader(model, val_loader, device)
    test_rows, final_test_metrics = predict_loader(model, test_loader, device)

    tokenizer.save_pretrained(seed_dir)
    save_predictions_jsonl(seed_dir / "val_predictions.jsonl", val_rows)
    save_predictions_jsonl(seed_dir / "test_predictions.jsonl", test_rows)

    meta = {
        "seed": seed,
        "model_name": args.model_name,
        "max_length": args.max_length,
        "dropout": args.dropout,
        "epochs_requested": args.epochs,
        "train_seconds": train_seconds,
        "device": str(device),
        "fp16": use_amp,
        "best_val_combined_score": best_score,
        "val_metrics": final_val_metrics,
        "test_metrics": final_test_metrics,
        "history": history,
    }
    (seed_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi-task IndoBERTweet for Task 3.")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2025])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--max-train-samples", type=Optional[int], default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_records = load_t3_records(args.data_path, split="train", subset="all")
    val_records = load_t3_records(args.data_path, split="val", subset="all")
    test_records = load_t3_records(args.data_path, split="test", subset="all")

    if args.max_train_samples is not None:
        train_records = train_records[: args.max_train_samples]

    print(
        f"Loaded T3 records -> train={len(train_records):,}, "
        f"val={len(val_records):,}, test={len(test_records):,}"
    )

    all_runs: List[Dict[str, Any]] = []
    for seed in args.seeds:
        meta = train_single_seed(seed, train_records, val_records, test_records, args)
        all_runs.append(meta)

    summary_path = args.output_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(all_runs, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved training summary -> {summary_path}")


if __name__ == "__main__":
    main()
