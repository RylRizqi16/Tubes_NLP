"""
Task 2 Phase 4 — Transformer NER Trainer (`src/task2/train_ner_transformer.py`).

Fine-tunes `indobenchmark/indobert-base-p1` across 3 seeds (`42, 123, 2025`) with:
- `max_length=384` (verified 0% span truncation across all 9,257 NER rows)
- Dynamic per-batch padding via `dynamic_pad_collate`
- Optional pure-PyTorch Linear-Chain CRF head (`--use-crf`, default `False` so E2/E3 share weights)
- `--real-only` ablation flag (`models/task2/ner_full/` vs. `models/task2/ner_real_only/`)
- `--fp16` mixed precision with `GradScaler` `scale_before` guard so `scheduler.step()`
  is never called on a skipped optimizer step
- Early stopping on `val` strict exact-span Micro-F1
- Cached `{val,test}_predictions.jsonl` + `train_meta.json` (with measured neural forward
  latency) + group `training_summary.json` for fast 2x2 evaluation in `evaluate_ner.py`.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from src.task2.data_loader import (
    BIO2_TAGS,
    DEFAULT_MASTER_PATH,
    DEFAULT_MAX_LENGTH,
    IGNORE_INDEX,
    NERTorchDataset,
    ROOT_DIR,
    decode_bio2_to_spans,
    dynamic_pad_collate,
    load_ner_records,
)
from src.task2.evaluate_ner import compute_ner_metrics


DEFAULT_MODEL_NAME = "indobenchmark/indobert-base-p1"
DEFAULT_MODELS_DIR = ROOT_DIR / "models" / "task2"
DEFAULT_SEEDS: Tuple[int, ...] = (42, 123, 2025)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LinearChainCRF(nn.Module):
    """
    Pure-PyTorch Linear-Chain CRF for BIO2 token classification (no `pytorch-crf` required).
    Operates strictly over non-special valid token positions per sequence to prevent
    `[CLS]`/`[SEP]`/`[PAD]` boundary contamination.
    """

    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def _log_partition_single(self, emissions: torch.Tensor) -> torch.Tensor:
        # emissions: (L, C)
        score = self.start_transitions + emissions[0]  # (C,)
        for t in range(1, emissions.size(0)):
            # broadcast_score: (C, 1) + transitions: (C, C) + emissions[t]: (1, C)
            next_score = score.unsqueeze(1) + self.transitions + emissions[t].unsqueeze(0)
            score = torch.logsumexp(next_score, dim=0)
        score = score + self.end_transitions
        return torch.logsumexp(score, dim=0)

    def _gold_score_single(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        # emissions: (L, C), tags: (L,)
        score = self.start_transitions[tags[0]] + emissions[0, tags[0]]
        for t in range(1, emissions.size(0)):
            score = score + self.transitions[tags[t - 1], tags[t]] + emissions[t, tags[t]]
        score = score + self.end_transitions[tags[-1]]
        return score

    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean negative log-likelihood loss over valid token sequences."""
        emissions_f = emissions.float()
        losses: List[torch.Tensor] = []
        batch_size = emissions_f.size(0)
        for i in range(batch_size):
            m = mask[i].bool()
            if not m.any():
                continue
            em_i = emissions_f[i][m]
            tg_i = tags[i][m]
            ll = self._gold_score_single(em_i, tg_i) - self._log_partition_single(em_i)
            losses.append(-ll / em_i.size(0))
        if not losses:
            return emissions_f.sum() * 0.0
        return torch.stack(losses).mean()

    @torch.no_grad()
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi decode the highest-scoring BIO2 tag path for each sequence."""
        emissions_f = emissions.float()
        batch_size, seq_len, _ = emissions_f.shape
        results: List[List[int]] = []

        for i in range(batch_size):
            m = mask[i].bool()
            valid_indices = torch.nonzero(m, as_tuple=False).squeeze(-1).tolist()
            full_tags = [IGNORE_INDEX] * seq_len
            if not valid_indices:
                results.append(full_tags)
                continue

            em_i = emissions_f[i][m]  # (L, C)
            L = em_i.size(0)
            score = self.start_transitions + em_i[0]  # (C,)
            backpointers: List[torch.Tensor] = []

            for t in range(1, L):
                next_score = score.unsqueeze(1) + self.transitions  # (C, C)
                best_prev_score, best_prev_tag = next_score.max(dim=0)
                score = best_prev_score + em_i[t]
                backpointers.append(best_prev_tag)

            score = score + self.end_transitions
            best_last_tag = int(score.argmax(dim=0).item())
            path = [best_last_tag]
            for bp in reversed(backpointers):
                path.append(int(bp[path[-1]].item()))
            path.reverse()

            for pos, tag_id in zip(valid_indices, path):
                full_tags[pos] = tag_id
            results.append(full_tags)

        return results


class IndoBERTNER(nn.Module):
    """IndoBERT Token Classification model with optional pure-PyTorch CRF head."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_labels: int = len(BIO2_TAGS),
        dropout: float = 0.1,
        use_crf: bool = False,
        from_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.use_crf = use_crf

        if from_pretrained:
            self.encoder = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.encoder = AutoModel.from_config(config)

        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = LinearChainCRF(num_labels) if use_crf else None
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(self.dropout(outputs.last_hidden_state))

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if self.crf is not None:
                valid_mask = labels != IGNORE_INDEX
                loss = self.crf(logits, labels.clamp(min=0), valid_mask)
            else:
                loss = self.ce_loss(logits.view(-1, self.num_labels), labels.view(-1))
        return logits, loss

    @torch.no_grad()
    def predict_tag_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mappings: Sequence[Sequence[Tuple[int, int]]],
    ) -> List[List[int]]:
        logits, _ = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        if self.crf is not None:
            batch_size, seq_len, _ = logits.shape
            valid_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=logits.device)
            for b, offsets in enumerate(offset_mappings):
                for t, (s, e) in enumerate(offsets):
                    if t < seq_len and not (s == 0 and e == 0) and s < e:
                        valid_mask[b, t] = True
            return self.crf.decode(logits, valid_mask)

        preds = logits.argmax(dim=-1).cpu().tolist()
        out: List[List[int]] = []
        for row_preds, offsets in zip(preds, offset_mappings):
            trimmed = row_preds[: len(offsets)]
            out.append(
                [
                    IGNORE_INDEX if (s == 0 and e == 0) or s >= e else int(pid)
                    for pid, (s, e) in zip(trimmed, offsets)
                ]
            )
        return out


def _make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_ctx(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


@torch.no_grad()
def run_inference(
    model: IndoBERTNER,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> Tuple[List[Dict[str, Any]], List[List[Dict[str, Any]]], float]:
    """
    Run batched neural inference, measuring exact forward-pass latency (`ms/sample`),
    and return `(cached_prediction_rows, decoded_raw_spans, forward_latency_ms)`.
    """
    model.eval()
    cached_rows: List[Dict[str, Any]] = []
    decoded_spans: List[List[Dict[str, Any]]] = []
    total_samples = 0
    forward_elapsed_ms = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        offsets_batch = batch["offset_mappings"]

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with _autocast_ctx(device, enabled=use_amp):
            tag_ids_batch = model.predict_tag_ids(input_ids, attention_mask, offsets_batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_elapsed_ms += (time.perf_counter() - t_fwd) * 1000.0

        for rid, src, text, offsets, tokens, tag_ids in zip(
            batch["ids"],
            batch["sources"],
            batch["texts"],
            offsets_batch,
            batch["tokens"],
            tag_ids_batch,
        ):
            spans = decode_bio2_to_spans(
                text,
                offsets,
                tag_ids,
                tokens=tokens,
                repair_subwords=False,
                promote_orphan_i=False,
            )
            decoded_spans.append(spans)
            cached_rows.append(
                {
                    "id": rid,
                    "source": src,
                    "offset_mapping": [[int(s), int(e)] for s, e in offsets],
                    "tag_ids": [int(t) for t in tag_ids],
                    "tokens": list(tokens),
                }
            )
            total_samples += 1

    forward_latency_ms = round(forward_elapsed_ms / max(1, total_samples), 4)
    return cached_rows, decoded_spans, forward_latency_ms


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def train_single_seed(
    seed: int,
    train_records: Sequence[Dict[str, Any]],
    val_records: Sequence[Dict[str, Any]],
    test_records: Sequence[Dict[str, Any]],
    model_name: str,
    group_dir: Path,
    max_length: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    patience: int,
    fp16: bool,
    use_crf: bool,
    real_only: bool,
    max_grad_norm: float = 1.0,
) -> Dict[str, Any]:
    set_seed(seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    use_amp = bool(fp16 and device.type == "cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    collate_fn = partial(dynamic_pad_collate, pad_token_id=pad_token_id)

    train_ds = NERTorchDataset(train_records, tokenizer=tokenizer, max_length=max_length)
    val_ds = NERTorchDataset(val_records, tokenizer=tokenizer, max_length=max_length)
    test_ds = NERTorchDataset(test_records, tokenizer=tokenizer, max_length=max_length)

    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)

    model = IndoBERTNER(
        model_name=model_name,
        num_labels=len(BIO2_TAGS),
        dropout=dropout,
        use_crf=use_crf,
        from_pretrained=True,
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias")
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)
    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    scaler = _make_grad_scaler(enabled=use_amp)

    seed_dir = group_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    weights_path = seed_dir / "model_weights.pt"

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_ctx(device, enabled=use_amp):
                _, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            assert loss is not None

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # Guard against skipped optimizer.step() on gradient overflow
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before:
                    scheduler.step()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

            running_loss += float(loss.item())
            num_batches += 1

        avg_train_loss = round(running_loss / max(1, num_batches), 4)
        _, val_decoded, val_latency = run_inference(model, val_loader, device, use_amp=use_amp)
        val_m = compute_ner_metrics(val_records, val_decoded)
        val_f1 = float(val_m["micro_f1"])

        ep_log = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_micro_f1": val_f1,
            "val_macro_f1": float(val_m["macro_f1"]),
            "val_partial_f1": float(val_m["partial_boundary_f1"]),
            "latency_ms_per_sample": val_latency,
        }
        history.append(ep_log)
        print(f"[{group_dir.name} seed={seed} epoch={epoch}/{epochs}] {ep_log}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), weights_path)
            tokenizer.save_pretrained(seed_dir)
            model.encoder.config.save_pretrained(seed_dir)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{group_dir.name} seed={seed}] Early stopping at epoch {epoch}.", flush=True)
                break

    # Restore best checkpoint & write cached val + test predictions
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    val_cached, val_decoded, val_fwd_lat = run_inference(model, val_loader, device, use_amp=use_amp)
    test_cached, test_decoded, test_fwd_lat = run_inference(model, test_loader, device, use_amp=use_amp)

    _write_jsonl(seed_dir / "val_predictions.jsonl", val_cached)
    _write_jsonl(seed_dir / "test_predictions.jsonl", test_cached)

    best_val_metrics = compute_ner_metrics(val_records, val_decoded)
    best_val_metrics["latency_ms_per_sample"] = val_fwd_lat
    best_test_metrics = compute_ner_metrics(test_records, test_decoded)
    best_test_metrics["latency_ms_per_sample"] = test_fwd_lat

    meta = {
        "model_name": model_name,
        "group": group_dir.name,
        "seed": seed,
        "real_only": real_only,
        "use_crf": use_crf,
        "max_length": max_length,
        "dropout": dropout,
        "lr": lr,
        "batch_size": batch_size,
        "epochs_planned": epochs,
        "best_epoch": best_epoch,
        "best_val_micro_f1": best_val_f1,
        "train_rows": len(train_records),
        "val_rows": len(val_records),
        "test_rows": len(test_records),
        "history": history,
        "val_metrics": best_val_metrics,
        "test_metrics": best_test_metrics,
    }
    (seed_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Task 2 IndoBERT NER (E2/E3/E4).")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_MASTER_PATH)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use-crf", action="store_true")
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Train ablation group on real NER sources only (saves to models/task2/ner_real_only/).",
    )
    args = parser.parse_args()

    train_subset = "real" if args.real_only else "all"
    group_name = "ner_real_only" if args.real_only else "ner_full"
    group_dir = args.models_dir / group_name

    train_records = load_ner_records(data_path=args.data_path, split="train", subset=train_subset)
    val_records = load_ner_records(data_path=args.data_path, split="val", subset="all")
    test_records = load_ner_records(data_path=args.data_path, split="test", subset="all")

    print(
        f"Loaded Task 2 NER ({group_name}) -> train={len(train_records):,} "
        f"(subset={train_subset}), val={len(val_records):,}, test={len(test_records):,} | "
        f"model={args.model_name}, max_length={args.max_length}, use_crf={args.use_crf}, fp16={args.fp16}",
        flush=True,
    )

    seed_summaries: List[Dict[str, Any]] = []
    for seed in args.seeds:
        meta = train_single_seed(
            seed=seed,
            train_records=train_records,
            val_records=val_records,
            test_records=test_records,
            model_name=args.model_name,
            group_dir=group_dir,
            max_length=args.max_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            patience=args.patience,
            fp16=args.fp16,
            use_crf=args.use_crf,
            real_only=args.real_only,
        )
        seed_summaries.append(
            {
                "seed": seed,
                "seed_dir": f"seed_{seed}",
                "best_epoch": meta["best_epoch"],
                "best_val_micro_f1": meta["best_val_micro_f1"],
                "val_macro_f1": meta["val_metrics"]["macro_f1"],
                "test_micro_f1": meta["test_metrics"]["micro_f1"],
                "test_macro_f1": meta["test_metrics"]["macro_f1"],
                "test_forward_latency_ms": meta["test_metrics"]["latency_ms_per_sample"],
            }
        )

    best_seed_entry = max(seed_summaries, key=lambda x: x["best_val_micro_f1"])
    summary = {
        "group": group_name,
        "model_name": args.model_name,
        "real_only": args.real_only,
        "use_crf": args.use_crf,
        "max_length": args.max_length,
        "best_seed": best_seed_entry["seed"],
        "best_seed_dir": best_seed_entry["seed_dir"],
        "best_val_micro_f1": best_seed_entry["best_val_micro_f1"],
        "seeds": seed_summaries,
    }
    summary_path = group_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved training summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
