"""Label Vietnamese comments with Gemini.

Pipeline:
  1. Read an input CSV of comments (default: ``VOZ_comments_sampled.csv``).
  2. Keep only rows whose existing label is 0 (CLEAN) or 1 (OFFENSIVE);
     drop label 2 (HATE).
  3. Send the kept comments to Gemini in configurable batches (default 100).
  4. For every comment, Gemini returns ``{"id": <int>, "label": <int>}`` where
     label is 0 (CLEAN), 1 (OFFENSIVE) or 2 (HATE).
  5. Write the predictions to an output file (CSV or JSON).

Gemini is prompted to role-play a Vietnamese social-network moderator who is
used to slang and informal chat language.

The output format is inferred from the ``--output`` extension (``.json`` or
``.csv``). If ``--output`` is a directory (no recognised extension), a JSON
file named ``<input-stem>_labeled.json`` is written inside it.

Usage:
  export GEMINI_API_KEY=...   # or GOOGLE_API_KEY
  python scripts/gemini_label_comments.py \
      --input VOZ_comments_sampled.csv \
      --output data/raw/VOZ_labeled/ \
      --model gemini-3.1-flash-lite \
      --batch-size 100
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

try:
    from google import genai
    from google.genai import types
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "google-genai is not installed. Run: pip install google-genai"
    ) from exc


# --- Label definitions -------------------------------------------------------

CLEAN, OFFENSIVE, HATE = 0, 1, 2
KEEP_LABELS = {CLEAN, OFFENSIVE}  # label 2 (HATE) is excluded from the input

# Common column names so the script works across the project's CSV variants.
TEXT_COL_CANDIDATES = ["free_text", "texts", "text", "comment", "content", "text_clean"]
LABEL_COL_CANDIDATES = ["label_id", "label", "label_binary"]

SYSTEM_PROMPT = """\
Bạn là một quản trị viên (moderator) kiểm duyệt nội dung của một mạng xã hội \
Việt Nam. Bạn đọc rất nhiều bình luận chứa tiếng lóng (slang), viết tắt, sai \
chính tả và văn phong suồng sã, không trang trọng. Hãy hiểu ngữ cảnh tiếng \
Việt đời thường khi phân loại.

Phân loại mỗi bình luận vào MỘT trong ba nhãn sau:
- CLEAN (0): bình luận không có ngôn từ xúc phạm, lành mạnh, trung lập.
- OFFENSIVE (1): bình luận rất thô lỗ, dùng từ tục tĩu / chửi thề / văng tục, \
NHƯNG không nhắm vào (tấn công) bất kỳ ai cụ thể.
- HATE (2): bình luận xúc phạm và nhắm thẳng vào một người, tổ chức, nhóm \
hoặc thực thể dựa trên một đặc điểm cụ thể (ví dụ: giới tính, chủng tộc, tôn \
giáo, quan điểm chính trị, định kiến, sở thích, tình trạng hôn nhân, v.v.).

Chỉ trả về JSON theo đúng cấu trúc được yêu cầu, không thêm giải thích."""

USER_PROMPT_HEADER = """\
Phân loại từng bình luận dưới đây. Với MỖI bình luận, trả về một đối tượng \
gồm "id" (chính là id được cung cấp) và "label" (0, 1 hoặc 2).
Trả về một mảng JSON, mỗi phần tử ứng với một bình luận.

Danh sách bình luận:
"""


class LabelResult(BaseModel):
    id: int
    label: int


# --- Helpers -----------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, explicit: Optional[str], candidates: List[str], kind: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise SystemExit(
                f"--{kind}-col '{explicit}' not found. Available columns: {list(df.columns)}"
            )
        return explicit
    for cand in candidates:
        if cand in df.columns:
            return cand
    raise SystemExit(
        f"Could not auto-detect the {kind} column. Tried {candidates}. "
        f"Pass --{kind}-col explicitly. Available columns: {list(df.columns)}"
    )


def load_and_filter(args: argparse.Namespace) -> pd.DataFrame:
    if not os.path.exists(args.input):
        raise SystemExit(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    text_col = _resolve_column(df, args.text_col, TEXT_COL_CANDIDATES, "text")
    label_col = _resolve_column(df, args.label_col, LABEL_COL_CANDIDATES, "label")

    # id column: use the provided one, else the 0-based row index of the
    # original file so predictions can be joined back to the source.
    if args.id_col:
        if args.id_col not in df.columns:
            raise SystemExit(
                f"--id-col '{args.id_col}' not found. Available columns: {list(df.columns)}"
            )
        df = df.rename(columns={args.id_col: "id"})
    else:
        df = df.reset_index().rename(columns={"index": "id"})

    df["label_int"] = pd.to_numeric(df[label_col], errors="coerce")
    before = len(df)
    df = df[df["label_int"].isin(KEEP_LABELS)].copy()
    print(
        f"Loaded {before} rows; kept {len(df)} after filtering to labels "
        f"{sorted(KEEP_LABELS)} (excluded HATE=2).",
        file=sys.stderr,
    )

    out = df[["id", text_col]].rename(columns={text_col: "text"})
    out["text"] = out["text"].fillna("").astype(str)
    return out.reset_index(drop=True)


def build_user_prompt(batch: pd.DataFrame) -> str:
    lines = [USER_PROMPT_HEADER]
    for row in batch.itertuples(index=False):
        # Keep each comment on one logical line; collapse newlines.
        text = " ".join(str(row.text).splitlines()).strip()
        lines.append(f"- id={row.id}: {text}")
    return "\n".join(lines)


def label_batch(client: "genai.Client", model: str, batch: pd.DataFrame,
                max_retries: int = 3) -> List[LabelResult]:
    prompt = build_user_prompt(batch)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=list[LabelResult],
        temperature=0.0,
    )
    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            parsed = resp.parsed
            if parsed:
                return parsed
            # Fallback: parse raw text if structured parsing returned nothing.
            return [LabelResult(**item) for item in json.loads(resp.text)]
        except Exception as err:  # noqa: BLE001 - surface + retry on API errors
            last_err = err
            wait = 2 ** attempt
            print(
                f"  batch attempt {attempt}/{max_retries} failed: {err}. "
                f"Retrying in {wait}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise SystemExit(f"Gemini call failed after {max_retries} attempts: {last_err}")


def resolve_output_path(output: str, input_path: str) -> str:
    """Resolve --output to a concrete file path.

    - ``.json`` / ``.csv`` extension -> used as-is.
    - anything else (e.g. a directory or trailing '/') -> treated as a
      directory; a ``<input-stem>_labeled.json`` file is created inside it.
    """
    ext = os.path.splitext(output)[1].lower()
    if ext in (".json", ".csv"):
        return output
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output, f"{stem}_labeled.json")


def write_predictions(df: pd.DataFrame, out_path: str) -> None:
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if out_path.lower().endswith(".json"):
        records = [{"id": int(r.id), "label": (None if pd.isna(r.label) else int(r.label))}
                   for r in df.itertuples(index=False)]
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(records, fh, ensure_ascii=False, indent=2)
    else:
        df.to_csv(out_path, index=False)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="VOZ_comments_sampled.csv",
                   help="Input CSV of comments (default: VOZ_comments_sampled.csv).")
    p.add_argument("--output", default="VOZ_comments_gemini_labeled.json",
                   help="Output file (.json or .csv) or a directory "
                        "(a JSON file is created inside it).")
    p.add_argument("--batch-size", type=int, default=100,
                   help="Number of comments per Gemini request (default: 100).")
    p.add_argument("--model", default="gemini-2.5-flash",
                   help="Gemini model name (default: gemini-2.5-flash).")
    p.add_argument("--text-col", default=None,
                   help="Comment text column (auto-detected if omitted).")
    p.add_argument("--label-col", default=None,
                   help="Existing label column for filtering (auto-detected if omitted).")
    p.add_argument("--id-col", default=None,
                   help="ID column (defaults to the original row index).")
    p.add_argument("--limit", type=int, default=None,
                   help="Optionally cap the number of comments (useful for testing).")
    p.add_argument("--api-key", default=None,
                   help="Gemini API key (else GEMINI_API_KEY / GOOGLE_API_KEY env var).")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "No API key. Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass --api-key."
        )

    df = load_and_filter(args)
    if args.limit is not None:
        df = df.head(args.limit)
    if df.empty:
        raise SystemExit("No comments to label after filtering.")

    client = genai.Client(api_key=api_key)

    results: List[LabelResult] = []
    n = len(df)
    batch_size = max(1, args.batch_size)
    for start in range(0, n, batch_size):
        batch = df.iloc[start:start + batch_size]
        idx = start // batch_size + 1
        total = (n + batch_size - 1) // batch_size
        print(f"Labeling batch {idx}/{total} ({len(batch)} comments)...", file=sys.stderr)
        results.extend(label_batch(client, args.model, batch))

    pred = pd.DataFrame([r.model_dump() for r in results])
    if pred.empty:
        raise SystemExit("Gemini returned no predictions.")

    # Keep id + label as the core output (the requested constraint).
    merged = df.merge(pred, on="id", how="left")[["id", "label"]]

    out_path = resolve_output_path(args.output, args.input)
    write_predictions(merged, out_path)
    print(f"Wrote {len(merged)} predictions to {out_path}", file=sys.stderr)

    missing = merged["label"].isna().sum()
    if missing:
        print(f"WARNING: {missing} comments were not labeled by Gemini.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
