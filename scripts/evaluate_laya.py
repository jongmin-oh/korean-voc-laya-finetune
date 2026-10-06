#!/usr/bin/env python3
"""Evaluate a Laya checkpoint and save per-row predictions for Jev comparison."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from laya_data import QUESTION_ID, load_schema, public_questions, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/laya-card-groups")
    parser.add_argument("--test", type=Path, default=Path("processed/test.csv"))
    parser.add_argument("--label-map", type=Path, default=Path("processed/label_map.json"))
    parser.add_argument("--input-column", choices=("text", "text_customer"), default="text_customer")
    parser.add_argument("--output-dir", type=Path, default=Path("results/laya"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    import laya
    groups, criteria = load_schema(args.label_map)
    rows = read_csv(args.test)
    usable = [row for row in rows if row[args.input_column].strip()]
    agent = laya.load(args.model, device=args.device)
    started = time.perf_counter()
    results = agent.predict_batch(
        [row[args.input_column].strip() for row in usable], public_questions(criteria),
        batch_size=args.batch_size, sort_by_length=True,
    )
    elapsed = time.perf_counter() - started

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gold, pred, confidences, probability_rows, records = [], [], [], [], []
    for row, result in zip(usable, results):
        answer = result["answers"][QUESTION_ID]
        probabilities = answer["probabilities"]
        gold.append(row["group"])
        pred.append(answer["choice"])
        confidences.append(float(answer["answer_confidence"]))
        probability_rows.append([float(probabilities[group]) for group in groups])
        records.append({
            "source_id": row["source_id"], "gold": row["group"], "prediction": answer["choice"],
            "correct": row["group"] == answer["choice"], "confidence": float(answer["answer_confidence"]),
            "probabilities": probabilities,
        })
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    per_group, f1s = {}, []
    for group in groups:
        tp = sum(g == group and p == group for g, p in zip(gold, pred))
        fp = sum(g != group and p == group for g, p in zip(gold, pred))
        fn = sum(g == group and p != group for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_group[group] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(g == group for g in gold)}
        f1s.append(f1)
    correct = np.array([g == p for g, p in zip(gold, pred)], dtype=float)
    conf = np.array(confidences)
    probability_array = np.array(probability_rows)
    gold_indices = np.array([groups.index(label) for label in gold])
    one_hot = np.eye(len(groups))[gold_indices]
    brier = np.square(probability_array - one_hot).sum(axis=1).mean()
    log_loss = -np.log(probability_array[np.arange(len(gold)), gold_indices].clip(1e-12, 1.0)).mean()
    confusion_matrix = {
        actual: {predicted: sum(g == actual and p == predicted for g, p in zip(gold, pred)) for predicted in groups}
        for actual in groups
    }
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        selected = (conf >= low) & (conf <= high) if low == 0 else (conf > low) & (conf <= high)
        if selected.any():
            ece += selected.mean() * abs(conf[selected].mean() - correct[selected].mean())
    metrics = {
        "model": args.model, "input_column": args.input_column, "n": len(gold),
        "accuracy": float(correct.mean()), "macro_f1": float(np.mean(f1s)), "ece": float(ece),
        "brier_score": float(brier), "log_loss": float(log_loss),
        "elapsed_seconds": elapsed, "milliseconds_per_item": elapsed * 1000 / len(gold),
        "per_group": per_group, "confusion_matrix": confusion_matrix,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
