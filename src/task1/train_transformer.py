"""E2/E3/E4 XLM-R experiments. Run from the repository root with python -m."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from src.task1.data_loader import DEFAULT_DATA, DOMAINS, INTENTS, ROOT, load_records
from src.task1.evaluate import evaluate

MODEL_NAME = "FacebookAI/xlm-roberta-base"
DOM2ID = {label: index for index, label in enumerate(DOMAINS)}
INT2ID = {label: index for index, label in enumerate(INTENTS)}


class TextDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, max_length: int):
        self.rows = rows
        self.tokens = tokenizer([r["text"] for r in rows], truncation=True,
                                padding="max_length", max_length=max_length, return_tensors="pt")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return {"input_ids": self.tokens["input_ids"][index],
                "attention_mask": self.tokens["attention_mask"][index],
                "domain": torch.tensor(DOM2ID.get(row["domain"], -100)),
                "intent": torch.tensor(INT2ID.get(row["intent"], -100))}


class T1Model(nn.Module):
    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(backbone))
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.domain_head = nn.Linear(hidden, len(DOMAINS))
        self.intent_head = nn.Linear(hidden, len(INTENTS))

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = self.dropout(x)
        return self.domain_head(x), self.intent_head(x)


def focal_loss(logits, labels, weights=None, gamma=2.0):
    valid = labels != -100
    if not valid.any():
        return None
    logits, labels = logits[valid], labels[valid]
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, labels[:, None]).squeeze(1)
    pt = (-ce).exp()
    alpha = weights[labels] if weights is not None else 1.0
    return (alpha * (1 - pt).pow(gamma) * ce).mean()


def class_weights(rows, target, mapping, device):
    counts = Counter(r[target] for r in rows if r[target] is not None)
    weights = torch.tensor([1 / max(1, counts[label]) for label in mapping], dtype=torch.float32, device=device)
    return weights / weights.mean()


@torch.no_grad()
def predict_loader(model, loader, device, mode="joint_ce"):
    model.eval()
    predictions = []
    for batch in loader:
        d, i = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        predictions.extend({"domain": DOMAINS[a] if mode != "intent_ce" else None,
                            "intent": INTENTS[b] if mode != "domain_ce" else None}
                           for a, b in zip(d.argmax(-1).tolist(), i.argmax(-1).tolist()))
    return predictions


def train_one(args, experiment: str, seed: int, train, val, test, tokenizer, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    mode = {"E2": "joint_ce", "E3": "joint_focal", "E4_domain": "domain_ce", "E4_intent": "intent_ce"}[experiment]
    # Single-task ablations use exactly the matching labeled training rows.
    training_rows = [r for r in train if (mode != "domain_ce" or r["domain"] is not None)
                     and (mode != "intent_ce" or r["intent"] is not None)]
    train_loader = DataLoader(TextDataset(training_rows, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TextDataset(val, tokenizer, args.max_length), batch_size=args.batch_size)
    test_loader = DataLoader(TextDataset(test, tokenizer, args.max_length), batch_size=args.batch_size)
    model = T1Model(args.model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    d_weights = class_weights(training_rows, "domain", DOMAINS, device) if mode == "joint_focal" else None
    i_weights = class_weights(training_rows, "intent", INTENTS, device) if mode == "joint_focal" else None
    checkpoint = args.model_dir / experiment / f"seed_{seed}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    best_score, best_epoch, stale = -1.0, 0, 0
    history = []
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            d, i = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            yd, yi = batch["domain"].to(device), batch["intent"].to(device)
            ld = focal_loss(d, yd, d_weights, 2.0 if mode == "joint_focal" else 0.0)
            li = focal_loss(i, yi, i_weights, 2.0 if mode == "joint_focal" else 0.0)
            active = [loss for loss, enabled in ((ld, mode != "intent_ce"), (li, mode != "domain_ce"))
                      if enabled and loss is not None]
            if not active:
                continue
            loss = sum(active) / len(active)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_report = evaluate(val, predict_loader(model, val_loader, device, mode))
        vd = val_report["all"]["domain"]["macro_f1"]
        vi = val_report["all"]["intent"]["macro_f1"]
        score = vi if mode == "intent_ce" else vd if mode == "domain_ce" else (vd + vi) / 2
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 5),
                        "val_domain_macro_f1": vd, "val_intent_macro_f1": vi})
        print(experiment, seed, history[-1], flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint / "model_weights.pt")
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(torch.load(checkpoint / "model_weights.pt", map_location=device, weights_only=True))
    model.encoder.config.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    metadata = {"experiment": experiment, "seed": seed, "model_name": args.model_name,
                "max_length": args.max_length, "best_epoch": best_epoch, "mode": mode,
                "selection_metric": "intent_macro_f1" if mode == "intent_ce" else "domain_macro_f1" if mode == "domain_ce" else "mean_macro_f1",
                "history": history, "train_count": len(training_rows)}
    (checkpoint / "train_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = {**metadata, "validation": evaluate(val, predict_loader(model, val_loader, device, mode)),
              "test": evaluate(test, predict_loader(model, test_loader, device, mode)),
              "elapsed_seconds": round(time.perf_counter() - start, 1)}
    report_path = args.output_dir / f"task1_{experiment}_seed_{seed}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/task1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--experiments", nargs="+", choices=("E2", "E3", "E4_domain", "E4_intent"), default=("E2", "E3", "E4_domain", "E4_intent"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size, and patience must be positive")
    train, val, test = (load_records(args.data, split) for split in ("train", "val", "test"))
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}; T1 rows train/val/test: {len(train)}/{len(val)}/{len(test)}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    results = []
    for experiment in args.experiments:
        for seed in args.seeds:
            results.append(train_one(args, experiment, seed, train, val, test, tokenizer, device))
    print("Reports:", [f"{r['experiment']} seed {r['seed']}" for r in results])


if __name__ == "__main__":
    main()
