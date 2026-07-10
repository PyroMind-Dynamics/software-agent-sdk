# Example Clean Scripts

The runnable example lives at
`scripts/fixtures/example_clean_script.py`. It converts the bundled
mixed-shape fixture into `messages` JSONL and writes `stats.json`.

## System Prompt Injection

A dataset shares one fixed system prompt. When source rows have no system
message, pick the prompt once and apply it to every row with
`ensure_system_message`:

```python
from cleaning_utils import DEFAULT_SYSTEM_PROMPT, ensure_system_message

# No tool usage in the data: use the fixed default.
SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT  # "You are a helpful assistant."

# Tool usage present: identify the tools by reading the sample rows, then
# write one fixed prompt enumerating them, e.g.:
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the following tools:\n"
    "- lookup: search the knowledge base for a query\n"
    "- calc: evaluate an arithmetic expression\n"
    "Call a tool when it is needed, read its result, then answer the user."
)

cleaned["messages"] = ensure_system_message(cleaned["messages"], SYSTEM_PROMPT)
```

`ensure_system_message` keeps an existing non-empty system message untouched,
so it is safe to apply unconditionally.

## Example 1: Multiple-Choice Rows to `messages`

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleaning_utils import (
    DEFAULT_SYSTEM_PROMPT,
    CleaningStats,
    answer_from_choices,
    ensure_system_message,
    iter_records,
    multiple_choice_prompt,
    validate_record,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

stats = CleaningStats()
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as handle:
    for parsed in iter_records(args.input):
        if args.limit and stats.total >= args.limit:
            break
        stats.record_input()
        if not parsed.ok or not isinstance(parsed.data, dict):
            stats.record_drop(parsed.error_type or "type_error", sample=parsed.raw)
            continue
        row = parsed.data
        choices = row.get("choices")
        if not isinstance(choices, list):
            stats.record_drop("missing_field", sample=row)
            continue
        cleaned = {
            "messages": ensure_system_message([
                {
                    "role": "user",
                    "content": multiple_choice_prompt(
                        row.get("question", ""),
                        choices,
                        subject=row.get("subject"),
                    ),
                },
                {
                    "role": "assistant",
                    "content": answer_from_choices(row.get("answer"), choices),
                },
            ], DEFAULT_SYSTEM_PROMPT)
        }
        errors = validate_record(cleaned, "messages")
        if errors:
            stats.record_drop(errors[0].code, sample=row, error=errors[0].message)
            continue
        handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        stats.record_keep()

stats.write_json(out.with_name("stats.json"))
```

## Tool Traces and Training Exclusions

`normalize_messages` parses stringified standard messages and auto-detects the
custom `reasoning/tool_call/tool_output/answer` sequence:

```python
from cleaning_utils import normalize_messages

cleaned = {"messages": normalize_messages(row["messages"])}
```

Filter explicit benchmark metadata before conversion:

```python
from cleaning_utils import training_exclusion_reason

reason = training_exclusion_reason(row)
if reason:
    stats.record_drop("benchmark_exclusion", error=reason, sample=row)
    continue
```

Do not substitute these filters for reading the dataset card; they only catch
explicit row metadata.

## Example 2: DPO-Style Source (chosen/rejected) to SFT `messages`

Use this only after the user confirms they want SFT from `chosen`, not DPO.

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleaning_utils import (
    CleaningStats,
    ExactDeduper,
    first_validation_reason,
    iter_records,
    to_messages_record,
    validate_record,
)


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=0)
args = parser.parse_args()

stats = CleaningStats()
deduper = ExactDeduper("messages")
out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)

with out.open("w", encoding="utf-8") as handle:
    for parsed in iter_records(args.input):
        if args.limit and stats.total >= args.limit:
            break
        stats.record_input()
        if not parsed.ok or not isinstance(parsed.data, dict):
            stats.record_drop(parsed.error_type or "type_error", sample=parsed.raw)
            continue
        cleaned = to_messages_record(
            parsed.data,
            preference="chosen",
            keep_metadata=False,
        )
        errors = validate_record(cleaned, "messages")
        if errors:
            stats.record_drop(
                first_validation_reason(errors),
                sample=parsed.data,
                error=errors[0].message,
            )
            continue
        if deduper.is_duplicate(cleaned):
            stats.record_drop("duplicate", sample=parsed.data)
            continue
        handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
        stats.record_keep()

stats.write_json(out.with_name("stats.json"))
```
