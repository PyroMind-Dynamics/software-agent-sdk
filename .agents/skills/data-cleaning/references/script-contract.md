# Cleaning Script Contract

Every generated `clean_script.py` must be runnable with:

```bash
PYTHONPATH=<skill>/scripts python clean_script.py \
  --input <path> \
  --output <path> \
  [--limit N]
```

## Required Behavior

- Import helpers with `from cleaning_utils import ...`.
- Read input with `iter_records()` unless the source needs custom grouping.
- Stream rows where possible; do not load a large file unless grouping requires
  it and the sample/full size is known to be small.
- Treat per-row parse, mapping, validation, duplicate, empty, and too-long
  failures as dropped rows.
- Do not stop the whole run for a single bad row.
- Write UTF-8 JSONL output.
- Write `stats.json` next to the output path before exiting.
- Honor `--limit N` by processing at most N input rows or grouped examples.

## Stats JSON

Write this shape:

```json
{
  "total": 5,
  "kept": 4,
  "dropped": 1,
  "drop_reasons": {
    "json_decode_error": 1
  },
  "error_samples": [
    {
      "reason": "json_decode_error",
      "line_number": 3,
      "error": "could not parse JSON",
      "sample": "{\"bad\":"
    }
  ]
}
```

Use stable reason codes when possible:

- `json_decode_error`
- `missing_field`
- `type_error`
- `empty_string`
- `invalid_role`
- `invalid_format`
- `too_long`
- `duplicate`

## Error Handling

Per-row failures:

- JSON parse failure
- missing required source fields
- target schema validation failure
- empty prompt or answer
- row exceeds configured length
- exact duplicate

These should call `stats.record_drop(...)` and continue.

Process-level failures:

- input file missing
- output directory not writable
- programmer error in the script

These may exit non-zero. Include enough stderr context to fix the script.

