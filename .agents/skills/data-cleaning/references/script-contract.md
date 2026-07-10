# Cleaning Script Contract

Run every generated cleaner as:

```bash
PYTHONPATH=<skill>/scripts python clean_script.py \
  --input <path> --output <path> [--limit N]
```

## Required behavior

- Import `cleaning_utils`; never copy it.
- Use only Python standard-library runtime dependencies.
- Perform deterministic row transformation. Do not call an LLM from the
  script. Embed user-confirmed global constants such as `SYSTEM_PROMPT`.
- Stream with `iter_records()` unless confirmed grouping requires bounded
  materialization.
- Count `--limit N` against input rows or grouped examples, not kept rows.
- Isolate parse, mapping, validation, length, exclusion, and duplicate failures
  per row; never abort the run for one bad row.
- Write UTF-8 JSONL and `stats.json` beside the output before exiting.
- Keep sample and full runs on the same code path.

Do not add dataset-specific CLI flags. Put confirmed thresholds and mappings in
named constants so the exact `--input/--output/--limit` interface remains
portable.

## `stats.json`

```json
{
  "total": 5,
  "kept": 3,
  "dropped": 2,
  "drop_reasons": {
    "truncated_field": 1,
    "duplicate": 1
  },
  "error_samples": [
    {
      "reason": "truncated_field",
      "line_number": 3,
      "error": "Hugging Face preview truncated fields: messages"
    }
  ]
}
```

Maintain `total == kept + dropped`. Keep error samples bounded and truncated;
do not put credentials or full oversized rows in stats.

Stable reasons include:

- `json_decode_error`, `truncated_field`
- `missing_field`, `type_error`, `empty_string`, `invalid_role`,
  `invalid_format`
- `too_long`, `duplicate`
- `benchmark_exclusion`

The validator also reports dataset-level `empty_dataset`; an empty output is
not a valid training artifact.

## Process failures

Missing input, unwritable output, or programmer errors may exit non-zero with
actionable stderr. A deliberate all-row benchmark exclusion may finish the
cleaner successfully, but `validate_format.py` must reject its empty output and
the agent must report that no training artifact should be used.
