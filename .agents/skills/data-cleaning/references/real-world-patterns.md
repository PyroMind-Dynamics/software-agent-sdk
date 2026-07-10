# Real-world Source Patterns

Use this reference after sampling, before choosing an adapter.

| Source shape | Deterministic mapping | Guardrail |
|---|---|---|
| Hugging Face `rows` envelope | Extract each `.rows[].row` with `extract_hf_rows.py` | Reject `truncated_cells`; fetch full data instead |
| `question` + `choices` + numeric `answer` | Build one user multiple-choice prompt and map the answer index to label plus choice text | Confirm whether few-shot `input_formatted` should replace the compact prompt |
| `conversations` with `human/gpt` or `from/value` | `normalize_messages(conversations)` | Do not treat column names alone as semantic proof |
| Stringified `messages` | Parse strictly enough that truncated JSON is dropped | Do not turn a broken JSON array into one user transcript |
| Agent messages with content blocks and tool calls | Normalize blocks, preserve call `id/index`, and link tool results by call ID/name | Tool result content must end as a string |
| Roles `reasoning/tool_call/tool_output/answer` | Merge reasoning into the following assistant action/answer; convert XML tool payloads | Use `normalize_tool_trace_messages()` or `normalize_messages()` auto-detection |
| `chosen/rejected` | Preserve DPO, or explicitly select the confirmed branch for SFT | `to_messages_record()` refuses an implicit choice |
| Benchmark canary / `do_not_train` | Drop with `benchmark_exclusion` | An all-dropped output is intentionally not a training artifact |

## Sampling pitfalls

- `first-rows` may return fewer rows than requested and may truncate large
  cells. Inspect response metadata, not only the visible value.
- A Viewer `rows` endpoint may fail for a very large or gated dataset while a
  repository file remains available. Prefer another official endpoint or a raw
  file; do not install a new runtime dependency without user approval.
- Sample ordinary rows and structural outliers. At minimum inspect distinct
  role labels, content types, tool-call shapes, empty fields, and row-length
  distribution.

## Content preservation

Text cleanup must not compatibility-fold Unicode (`10⁴` must not become
`104`), delete code indentation, or remove Markdown's two-space line breaks.
Use aggressive whitespace collapsing only for identifiers such as roles and
tool names.
