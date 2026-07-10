# Target Formats

Prefer `messages`, matching the Transformers
[chat message patterns](https://huggingface.co/docs/transformers/chat_content_patterns).
Validate the finished JSONL with `scripts/validate_format.py`.

## `messages`

Use for chat, agent, tool-use, multimodal, and multi-turn SFT data.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check whether 91 is prime."},
    {
      "role": "assistant",
      "content": "I will factor it.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {"name": "factor", "arguments": {"n": 91}}
        }
      ]
    },
    {
      "role": "tool",
      "name": "factor",
      "tool_call_id": "call_1",
      "content": "91 = 7 * 13"
    },
    {"role": "assistant", "content": "91 is not prime."}
  ]
}
```

Rules enforced by the validator:

- `messages` is a non-empty list; each item has role `system`, `user`,
  `assistant`, or `tool`.
- `content` is a string or explicit content-part list. A text part is
  `{"type":"text","text":"..."}`; image/video/audio parts require `url`
  or `path`.
- An assistant may omit or leave content empty only when it has non-empty,
  structurally valid `tool_calls`.
- Each tool call uses `type: function`, a non-empty function `name`, and an
  object `arguments`. Preserve source `id` and `index` when present.
- A tool result has string `content`. Preserve or infer `name` and
  `tool_call_id` so calls remain linked.
- One fixed system prompt may be generated once and applied to all rows; do not
  generate a prompt per row.

Extra provenance metadata is allowed only when the user requests it. Default
cleaning scripts should emit only `messages`.

## `alpaca`

Use only for explicitly requested single-turn instruction data.

```json
{"system":"You are concise.","input":"A question...","output":"An answer..."}
```

`input` and `output` are required non-empty strings. `system` is optional but
must be a string.

## `sharegpt`

Use only when downstream tooling requires ShareGPT.

```json
{
  "conversations": [
    {"from": "human", "value": "Solve 2x + 3 = 11."},
    {"from": "gpt", "value": "x = 4."}
  ]
}
```

`conversations` is non-empty. Each item has `from` and non-empty `value`;
common roles map to the same four standard message roles.

## Preference data

`chosen`/`rejected` is not equivalent to SFT `messages`. Preserve a custom DPO
schema unless the user explicitly requests an SFT branch. For confirmed SFT,
call `to_messages_record(row, preference="chosen")` or normalize that branch
directly. Record the lossy decision in the before/after confirmation.
