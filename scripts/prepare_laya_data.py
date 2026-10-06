#!/usr/bin/env python3
"""Convert processed CSV splits into Laya typed-decision JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from laya_data import load_schema, make_item, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=Path("processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/laya"))
    parser.add_argument("--input-column", choices=("text", "text_customer"), default="text_customer")
    parser.add_argument("--skip-empty", action="store_true")
    args = parser.parse_args()

    groups, criteria = load_schema(args.processed_dir / "label_map.json")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"input_column": args.input_column, "groups": groups, "splits": {}}

    for split in ("train", "val", "test"):
        rows = read_csv(args.processed_dir / f"{split}.csv")
        output = args.output_dir / f"{split}.jsonl"
        written = skipped = 0
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                try:
                    item = make_item(row, groups, criteria, args.input_column)
                except ValueError:
                    if not args.skip_empty:
                        raise
                    skipped += 1
                    continue
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                written += 1
        summary["splits"][split] = {"written": written, "skipped": skipped}
        print(f"{split}: {written} written, {skipped} skipped -> {output}")

    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
