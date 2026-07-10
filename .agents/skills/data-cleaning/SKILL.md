---
name: data-cleaning
description: Inspect, clean, normalize, and validate local or Hugging Face datasets into training-ready Transformers messages, Alpaca, or ShareGPT JSONL. Use when converting CSV/JSON/JSONL, chat or agent trajectories, tool-use traces, preference data, multiple-choice data, or dirty generated data; when writing a reusable deterministic cleaning script; or when validating a cleaned training dataset.
---

# Data Cleaning

Produce a deterministic Python cleaning script, validate its output, and ask
the user to confirm representative before/after rows. Prefer Transformers
`messages`; use Alpaca or ShareGPT only when the user explicitly needs them.

## Non-negotiable rules

- Do not guess field semantics from names alone. Confirm ambiguous mappings.
- Do not call an LLM once per row. An LLM may produce a small, reusable
  constant such as one system prompt or mapping configuration; embed it in the
  Python script and apply it to every row.
- Use the same script and code path for sample and full data.
- Preserve meaningful Unicode, indentation, Markdown line breaks, code, and
  tool-call linkage. Clean controls and malformed structure, not content style.
- Inspect dataset documentation and explicit metadata such as `canary` or
  `do_not_train`. Exclude benchmark rows that prohibit training.
- Never silently collapse `chosen`/`rejected` preference data into SFT.

## Workflow

1. Fetch real rows and save a sample.
   - For Hugging Face, use Terminal and `curl`; do not rely on a rendered web
     preview. Download a `rows` or `first-rows` API response to JSON.
   - Extract complete row payloads with:

     ```bash
     PYTHONPATH=<skill>/scripts python <skill>/scripts/extract_hf_rows.py \
       --input rows-response.json --output sample.jsonl --limit 5
     ```

   - Treat `truncated_cells` as incomplete transport data, not a repairable
     source row. Fetch full rows or a raw repository file instead.
   - For gated data, use the user's configured token without printing it.
   - Also inspect the dataset card for field meaning, license, preference
     semantics, and training-exclusion canaries.

2. Analyze representative rows.
   - Include ordinary rows plus rare structures: null/empty fields, longest
     rows, stringified JSON, content blocks, tool calls/results, custom roles,
     and malformed rows.
   - Identify whether examples are independent rows or events that must be
     grouped by a session/trace key.
   - Read [references/real-world-patterns.md](references/real-world-patterns.md)
     for proven source-shape decisions.

3. Confirm intent with the user.
   - Explain what the sample actually contains and ask 1-2 focused questions.
   - Confirm the target schema and semantic mapping before writing code.
   - For DPO, ask whether to preserve preference pairs or explicitly select
     `chosen`/`rejected` for SFT.
   - If deterministic transformation cannot create the target labels, stop and
     name the missing information. Do not fabricate it. Recommend a separate
     labeling/generation step only when needed.

4. Decide reusable constants.
   - If rows lack a system message, confirm one fixed prompt and apply it with
     `ensure_system_message`. Keep an existing non-empty system message unless
     the user approves replacement.
   - If the source contains tools, prefer definitions already present in the
     data. Otherwise infer a single prompt from sampled tool calls/results,
     scan the full run for unseen tool names, then update and rerun if needed.

5. Read the relevant contracts.
   - Read [references/script-contract.md](references/script-contract.md) before
     writing the cleaner.
   - Read [references/target-formats.md](references/target-formats.md) for the
     selected target.
   - Use [references/example-scripts.md](references/example-scripts.md) only
     for the matching source family.

6. Write `clean_script.py`.
   - Import helpers with `from cleaning_utils import ...`; do not copy the
     library.
   - Support `--input`, `--output`, and optional `--limit` exactly as specified.
   - Stream rows, isolate per-row failures, perform exact dedupe, enforce the
     confirmed length threshold, and write `stats.json` beside the output.
   - Emit only target fields unless the user asks to retain provenance metadata.

7. Run, repair, and validate the sample.

   ```bash
   PYTHONPATH=<skill>/scripts python clean_script.py \
     --input sample.jsonl --output cleaned.jsonl --limit 5

   PYTHONPATH=<skill>/scripts python <skill>/scripts/validate_format.py \
     --input cleaned.jsonl --format messages
   ```

   Fix the script and rerun until the validator succeeds. An empty output fails
   validation; if every row was correctly excluded, report that no training
   artifact should be produced instead of treating it as success.

8. Show 1-3 compact before/after examples and `stats.json` to the user.
   Include the fixed system prompt and any lossy decision such as selecting the
   DPO `chosen` branch. If approved, run the same script without `--limit`.

## Bundled scripts

- `scripts/cleaning_utils.py`: stdlib-only tolerant readers, normalization,
  tool-trace conversion/linking, filters, dedupe, stats, and schema validation.
- `scripts/extract_hf_rows.py`: extract complete rows from a downloaded Hugging
  Face viewer response and reject server-truncated cells.
- `scripts/validate_format.py`: validate every output row and reject empty data.
- `scripts/selftest_cleaning_utils.py`: dependency-free regression tests.
- `scripts/fixtures/example_clean_script.py`: runnable mixed-shape example.
