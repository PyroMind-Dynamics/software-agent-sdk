"""Extract complete rows from a Hugging Face rows/first-rows API response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleaning_utils import CleaningStats, iter_huggingface_rows


def build_parser() -> argparse.ArgumentParser:
    """Build the extraction command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Downloaded API response JSON")
    parser.add_argument("--output", required=True, help="Output sample JSONL")
    parser.add_argument("--limit", type=int, default=0, help="Maximum complete rows")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Write complete viewer rows and report rejected/truncated previews."""
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = CleaningStats()

    with output_path.open("w", encoding="utf-8") as output:
        for parsed in iter_huggingface_rows(args.input):
            if args.limit and stats.kept >= args.limit:
                break
            stats.record_input()
            if not parsed.ok:
                stats.record_drop(
                    parsed.error_type or "invalid_format",
                    line_number=parsed.line_number,
                    error=parsed.error,
                    sample=parsed.raw,
                )
                continue
            output.write(json.dumps(parsed.data, ensure_ascii=False) + "\n")
            stats.record_keep()

    print(json.dumps(stats.to_dict(), ensure_ascii=False))
    return 0 if stats.kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
