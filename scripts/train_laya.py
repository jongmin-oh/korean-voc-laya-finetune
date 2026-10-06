#!/usr/bin/env python3
"""Fine-tune Laya for eight-way Korean consultation-group classification.

Adapted from Laya's public RLCD notebook and 2nugu/laya-ko training recipe.
Validation data is never used for gradient updates; it selects the checkpoint and
fits the final temperature.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from laya.common import QTYPES, build_sequence, proper_reward, render_options


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_context(device: torch.device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    return nullcontext()


def load_raw(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize_items(tokenizer, raw: list[dict], max_len: int, head_max_len: int) -> tuple[list[dict], int]:
    items, dropped = [], 0
    for item in raw:
        q = item["q"]
        sequence, markers = build_sequence(tokenizer, item["state"], q, max_len, head_max_len)
        option_count = len(render_options(q))
        if len(markers) != option_count:
            dropped += 1
            continue
        target = [0.0] * option_count
        target[item["gold_idx"]] = 1.0
        items.append({
            "ids": sequence,
            "markers": markers,
            "qtype": QTYPES[q["t"]],
            "target": target,
            "label": item["gold_idx"],
        })
    return items, dropped


def collate(items: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    size, length = len(items), max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    ids = torch.full((size, length), pad_id, dtype=torch.long)
    attention = torch.zeros((size, length), dtype=torch.long)
    marker_pos = torch.zeros((size, kmax), dtype=torch.long)
    marker_mask = torch.zeros((size, kmax), dtype=torch.bool)
    target = torch.zeros((size, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        n, k = len(item["ids"]), len(item["markers"])
        ids[i, :n] = torch.tensor(item["ids"])
        attention[i, :n] = 1
        marker_pos[i, :k] = torch.tensor(item["markers"])
        marker_mask[i, :k] = True
        target[i, :k] = torch.tensor(item["target"])
    return {
        "input_ids": ids, "attention_mask": attention, "marker_pos": marker_pos,
        "marker_mask": marker_mask, "target": target,
        "qtype": torch.tensor([item["qtype"] for item in items]),
    }


def batches(items: list[dict], batch_size: int, seed: int, shuffle: bool) -> list[list[dict]]:
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    result = [[items[i] for i in order[start:start + batch_size]] for start in range(0, len(order), batch_size)]
    if shuffle:
        random.Random(seed).shuffle(result)
    return result


@torch.no_grad()
def infer_logits(model, items, tokenizer, device, batch_size):
    model.eval()
    outputs = []
    for chunk in batches(items, batch_size, 0, False):
        batch = collate(chunk, tokenizer.pad_token_id)
        with amp_context(device):
            logits, _ = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                batch["qtype"].to(device),
            )
        for row, item in zip(logits.float().cpu(), chunk):
            outputs.append((row[:len(item["markers"])], item["label"]))
    return outputs


def classification_metrics(pairs) -> dict[str, float]:
    gold = np.array([label for _, label in pairs])
    pred = np.array([int(logits.argmax()) for logits, _ in pairs])
    labels = sorted(set(gold.tolist()))
    f1s = []
    for label in labels:
        tp = int(((pred == label) & (gold == label)).sum())
        fp = int(((pred == label) & (gold != label)).sum())
        fn = int(((pred != label) & (gold == label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return {"accuracy": float((pred == gold).mean()), "macro_f1": float(np.mean(f1s))}


def fit_temperature(pairs) -> float:
    logits = torch.stack([values for values, _ in pairs]).float()
    labels = torch.tensor([label for _, label in pairs])
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    # Laya accepts persisted temperatures only in [0.5, 5.0].
    return float(log_temperature.exp().clamp(0.5, 5.0).item())


def save_checkpoint(output_dir: Path, model, tokenizer, cfg: dict, temperature: float, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {name: value.detach().half().contiguous().cpu() for name, value in model.state_dict().items()}
    save_file(weights, str(output_dir / "model.safetensors"))
    model.encoder.config.save_pretrained(output_dir / "encoder")
    tokenizer.save_pretrained(output_dir / "tokenizer")
    saved_cfg = dict(cfg)
    saved_cfg.update({
        "fine_tuned": True,
        "model_name": "laya-korean-card-groups",
        "temperature": [temperature, 1.0, 1.0],
        "training": meta,
    })
    saved_cfg.pop("temperature_by_options", None)
    with (output_dir / "rl_agent_config.json").open("w", encoding="utf-8") as f:
        json.dump(saved_cfg, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="2nugu/laya-ko")
    parser.add_argument("--train", type=Path, default=Path("data/laya/train.jsonl"))
    parser.add_argument("--validation", type=Path, default=Path("data/laya/val.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/laya-card-groups"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--epoch-offset", type=int, default=0, help="display/save epoch numbers after prior training")
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr-encoder", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.micro_batch < 1 or args.grad_accum < 1:
        parser.error("epochs, micro-batch and grad-accum must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    import laya
    print(f"Loading {args.model} on {device} ...")
    agent = laya.load(args.model, device=str(device))
    model, tokenizer, cfg = agent.model, agent.tok, dict(agent.cfg)
    train_items, train_dropped = tokenize_items(tokenizer, load_raw(args.train), cfg["max_len"], cfg["head_max_len"])
    val_items, val_dropped = tokenize_items(tokenizer, load_raw(args.validation), cfg["max_len"], cfg["head_max_len"])
    print(f"train={len(train_items)} (dropped={train_dropped}), val={len(val_items)} (dropped={val_dropped})")

    if not args.no_gradient_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    if device.type in {"mps", "cpu"}:
        # The upstream Apple-Silicon recipe trains in fp32 for MPS stability.
        model.float()
    model.to(device).train()
    encoder = [p for name, p in model.named_parameters() if name.startswith("encoder.")]
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": args.lr_encoder}, {"params": head, "lr": args.lr_head}],
        weight_decay=0.01,
    )
    updates_per_epoch = math.ceil(len(train_items) / (args.micro_batch * args.grad_accum))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[args.lr_encoder, args.lr_head], total_steps=updates_per_epoch * args.epochs,
        pct_start=0.06, anneal_strategy="cos", div_factor=10.0, final_div_factor=100.0,
    )

    history, best_f1 = [], -1.0
    for epoch in range(args.epochs):
        epoch_number = args.epoch_offset + epoch + 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        epoch_batches = batches(train_items, args.micro_batch, args.seed + epoch, True)
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)
        for step, chunk in enumerate(epoch_batches):
            batch = collate(chunk, tokenizer.pad_token_id)
            with amp_context(device):
                logits, activation = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            target = batch["target"].to(device)
            qtype = batch["qtype"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            noise = torch.randn((4,) + logits.shape, device=device) * sigma * mask
            noise = (noise - noise.sum(-1, keepdim=True) / k) * mask
            noisy = logits.detach().unsqueeze(0) + noise
            probabilities = torch.softmax(noisy.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(probabilities, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0)
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)
            logp = -(((noisy - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (args.rl_weight * loss_rl + args.ce_weight * loss_ce + 0.0 * activation.sum()) / args.grad_accum
            loss.backward()
            running += float(loss.detach()) * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(epoch_batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if (step + 1) % 100 == 0:
                print(
                    f"epoch {epoch_number}/{args.epoch_offset + args.epochs} "
                    f"batch {step + 1}/{len(epoch_batches)} "
                    f"loss={running / (step + 1):.4f}",
                    flush=True,
                )

        pairs = infer_logits(model, val_items, tokenizer, device, args.micro_batch)
        metrics = classification_metrics(pairs)
        temperature = fit_temperature(pairs)
        record = {"epoch": epoch_number, "loss": running / len(epoch_batches), **metrics, "temperature": temperature}
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            save_checkpoint(args.output_dir, model, tokenizer, cfg, temperature, {
                "base_model": args.model, "best_epoch": epoch_number, "validation": metrics,
                "args": vars(args) | {"train": str(args.train), "validation": str(args.validation), "output_dir": str(args.output_dir)},
            })
            print(f"Saved new best checkpoint to {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with (args.output_dir / "history.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
