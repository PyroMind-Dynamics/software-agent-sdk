"""Example cleaner over the bundled fixture data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleaning_utils import (
    DEFAULT_SYSTEM_PROMPT,
    CleaningStats,
    ExactDeduper,
    ensure_system_message,
    first_validation_reason,
    is_too_long,
    iter_records,
    to_messages_record,
    validate_record,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the example cleaner command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=20000)
    return parser


def main() -> int:
    """Clean a sample file into messages JSONL and write stats.json."""
    args = build_parser().parse_args()
    stats = CleaningStats()
    deduper = ExactDeduper("messages")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output:
        for parsed in iter_records(args.input):
            if args.limit and stats.total >= args.limit:
                break
            stats.record_input()
            if not parsed.ok:
                stats.record_drop(
                    parsed.error_type or "parse_error",
                    sample=parsed.raw,
                    line_number=parsed.line_number,
                    error=parsed.error,
                )
                continue
            if not isinstance(parsed.data, dict):
                stats.record_drop(
                    "type_error",
                    sample=parsed.data,
                    line_number=parsed.line_number,
                )
                continue
            preference = "chosen" if "rejected" in parsed.data else None
            cleaned = to_messages_record(
                parsed.data,
                preference=preference,
                keep_metadata=False,
            )
            cleaned["messages"] = ensure_system_message(
                cleaned["messages"], DEFAULT_SYSTEM_PROMPT
            )
            errors = validate_record(
                cleaned,
                "messages",
                line_number=parsed.line_number,
            )
            if errors:
                stats.record_drop(
                    first_validation_reason(errors),
                    sample=parsed.data,
                    line_number=parsed.line_number,
                    error=errors[0].message,
                )
                continue
            if is_too_long(cleaned, max_chars=args.max_chars, fields=["messages"]):
                stats.record_drop(
                    "too_long",
                    sample=parsed.data,
                    line_number=parsed.line_number,
                )
                continue
            if deduper.is_duplicate(cleaned):
                stats.record_drop(
                    "duplicate",
                    sample=parsed.data,
                    line_number=parsed.line_number,
                )
                continue
            output.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
            stats.record_keep()

    stats.write_json(output_path.with_name("stats.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
