---
name: data-cleaning
description: Clean, normalize, validate, and convert user-provided datasets into training-ready JSON formats. Use when the user asks to clean data, convert CSV/JSON/JSONL/logs into alpaca/sharegpt/messages, prepare SFT/DPO/tool-use data, inspect dirty dataset samples, generate a cleaning script, or validate cleaned training data.
---

# Data Cleaning

Turn a raw dataset into training-ready JSON: sample the source, infer its
shape, confirm the user's intent, write a cleaning script on top of the
bundled helpers, run it on sample data, validate the target format, then show
before/after examples for user confirmation.

## Workflow

1. Sample the data source.
   - The source can be anything: a local file, a shared storage path, a URL,
     an API, or rows pasted into the chat. Use whatever access method fits;
     for remote sources fetch a small preview instead of downloading
     everything.
   - Only collect enough rows to understand structure. Save 3-5 representative
     rows to a local JSONL sample file.

2. Infer the shape of the data from the sample.
   - Judge from the rows themselves plus any available context (field names,
     the user's description, accompanying docs). Common shape families:
     - Chat arrays: `messages`/`conversations` of role+content turns, possibly
       stringified JSON, possibly with `tool_calls` and tool-role turns.
     - Flat pairs: `prompt`/`instruction`/`question` alongside
       `response`/`output`/`answer`.
     - Preference pairs: `chosen`/`rejected` (DPO-style).
     - Multiple-choice: `question` + `choices` + an `answer` index or label.
     - Event streams: one event per row that must be grouped by a
       `session_id`/`trace_id`-like key before it forms one conversation.
     - Raw text or documents with no labels.
   - Watch for common dirty patterns: stringified JSON inside fields, role
     aliases (`human`/`gpt`, `from`/`value`), numeric answer indices, empty
     assistant content that is valid because tool calls are present, oversized
     rows, exact duplicates, broken JSON.
   - `cleaning_utils` already normalizes most of these families
     (`messages_from_record`, `normalize_messages`,
     `messages_from_trace_events`, `multiple_choice_prompt`,
     `answer_from_choices`); prefer them over hand-rolled parsing.

3. Explain what the sample appears to contain and ask focused questions.
   - Confirm the semantic task and target format before writing the script.
   - Do not infer meaning only from column names.
   - Ask 1-2 questions at a time, for example: "Should `chosen` be the SFT
     assistant answer, or do you want a DPO pair with `chosen/rejected`?"
   - If the sample lacks fields needed for the target format, or the source is
     non-structured text/events that cannot be converted by deterministic rules,
     stop the cleaning flow instead of fabricating labels. Tell the user which
     fields are missing, suggest the minimum schema they should add, and
     recommend an LLM labeling/generation step before rerunning data cleaning.
     Example: "This file has only raw documents, but SFT needs an `input` and
     `output`; generate or provide answer labels, then rerun this skill."

4. Decide the system prompt (when rows have no system message).
   - A dataset shares one fixed system prompt; it does not vary row by row, so
     a missing system prompt never requires per-row labeling.
   - No tool usage in the data: use the fixed default
     `You are a helpful assistant.` (`DEFAULT_SYSTEM_PROMPT` in
     `cleaning_utils`).
   - Tool usage present (tool calls or tool-role turns): read the sample rows
     and identify each tool yourself — name, what it does (inferred from its
     arguments and tool outputs, or from a `tools` definition field when the
     row has one). Then write one fixed system prompt: state the assistant's
     role and enumerate the available tools with a one-line description each.
     Embed it as a string constant in `clean_script.py`.
   - The sample may not cover every tool in the full file. Have
     `clean_script.py` collect the distinct tool names it encounters (e.g. in
     a set, reported alongside stats); if the full run surfaces tools missing
     from the system prompt, update the prompt and rerun.
   - In both cases apply it with `ensure_system_message(messages, SYSTEM_PROMPT)`
     so every output row carries the system message, and show the chosen
     system prompt to the user during the confirmation step.

5. Read references as needed.
   - For the generated script contract, read
     [references/script-contract.md](references/script-contract.md).
   - For target schemas and examples, read
     [references/target-formats.md](references/target-formats.md).
   - For reusable examples, read
     [references/example-scripts.md](references/example-scripts.md).

6. Write `clean_script.py`.
   - Import helpers with `from cleaning_utils import ...`; never copy
     `cleaning_utils.py` into the working directory.
   - Support exactly:
     `python clean_script.py --input <path> --output <path> [--limit N]`.
   - End every run by writing `stats.json` next to the output file.
   - Treat per-row parse/validation failures as dropped rows, not process
     crashes.

7. Run the cleaner with `PYTHONPATH`.
   - Use:
     `PYTHONPATH=<skill>/scripts python clean_script.py --input sample.jsonl --output cleaned.jsonl --limit 5`
   - If it fails, fix the script and rerun.

8. Validate the cleaned output.
   - Use:
     `PYTHONPATH=<skill>/scripts python <skill>/scripts/validate_format.py --input cleaned.jsonl --format <alpaca|sharegpt|messages>`
   - If validation fails, return to step 6 and fix the script.

9. Show before/after examples and ask for confirmation.
   - Present a compact comparison of 1-3 rows, including the system prompt that
     was added.
   - If the user is not satisfied, update `clean_script.py` and rerun.
   - If the sample is accepted, the same script can be run without `--limit` on
     the local full file.

## Helper Scripts

- `scripts/cleaning_utils.py`: stdlib-only parsing, normalization, filtering,
  dedupe, stats, schema validation, and system-prompt injection
  (`ensure_system_message`, `DEFAULT_SYSTEM_PROMPT`).
- `scripts/validate_format.py`: CLI validator for `alpaca`, `sharegpt`, and
  `messages`.
- `scripts/selftest_cleaning_utils.py`: local self-test over bundled
  sample-shape fixtures.
- `scripts/fixtures/example_clean_script.py`: runnable example cleaner over the
  bundled fixture data.
