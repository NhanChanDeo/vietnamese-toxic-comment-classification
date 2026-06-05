"""Convert the Gemini labels JSON to CSV and merge with the source comments.

Steps:
  1. Read the Gemini predictions JSON (array of ``{"id", "label"}``) and write
     it out as a CSV.
  2. Read the source ``VOZ_comments_sampled.csv`` (columns ``texts``, ``label``),
     slice it from the first comment whose original label is NOT 2 (HATE), and
     merge the Gemini labels onto it by ``id`` (the original 0-based row index).

Output columns of the merged CSV: ``id, texts, label_original, label_gemini``.

Usage:
  python scripts/merge_voz_labels.py \
      --labels-json data/raw/VOZ_labeled/VOZ_comments_sampled_labeled.json \
      --source VOZ_comments_sampled.csv \
      --labels-csv data/raw/VOZ_labeled/VOZ_comments_sampled_labeled.csv \
      --merged-csv data/raw/VOZ_labeled/VOZ_comments_sampled_merged.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

HATE = 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--labels-json",
                   default="data/raw/VOZ_labeled/VOZ_comments_sampled_labeled.json",
                   help="Gemini predictions JSON (array of {id, label}).")
    p.add_argument("--source", default="VOZ_comments_sampled.csv",
                   help="Source comments CSV (columns: texts, label).")
    p.add_argument("--labels-csv",
                   default="data/raw/VOZ_labeled/VOZ_comments_sampled_labeled.csv",
                   help="Where to write the JSON-converted CSV.")
    p.add_argument("--merged-csv",
                   default="data/raw/VOZ_labeled/VOZ_comments_sampled_merged.csv",
                   help="Where to write the merged CSV.")
    p.add_argument("--text-col", default="texts", help="Text column in the source CSV.")
    p.add_argument("--label-col", default="label", help="Label column in the source CSV.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.labels_json, args.source):
        if not os.path.exists(path):
            raise SystemExit(f"File not found: {path}")

    # 1) JSON -> CSV
    labels = pd.read_json(args.labels_json)
    os.makedirs(os.path.dirname(args.labels_csv) or ".", exist_ok=True)
    labels.to_csv(args.labels_csv, index=False)
    print(f"Wrote {len(labels)} rows to {args.labels_csv}", file=sys.stderr)

    # 2) Source sliced from the first non-HATE comment, merged by id
    src = pd.read_csv(args.source).reset_index().rename(columns={"index": "id"})
    non_hate = src.index[src[args.label_col] != HATE]
    if len(non_hate) == 0:
        raise SystemExit("No comments with a non-HATE label in the source.")
    first_id = int(src.loc[non_hate[0], "id"])
    sliced = src[src["id"] >= first_id].copy()
    print(
        f"First non-HATE comment is id={first_id}; "
        f"merging {len(sliced)} source rows from there.",
        file=sys.stderr,
    )

    merged = sliced.merge(
        labels.rename(columns={"label": "label_gemini"}), on="id", how="left"
    )
    merged = merged.rename(columns={args.text_col: "texts", args.label_col: "label_original"})
    merged = merged[["id", "texts", "label_original", "label_gemini"]]

    os.makedirs(os.path.dirname(args.merged_csv) or ".", exist_ok=True)
    merged.to_csv(args.merged_csv, index=False)
    print(f"Wrote {len(merged)} merged rows to {args.merged_csv}", file=sys.stderr)

    missing = merged["label_gemini"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no Gemini label.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
