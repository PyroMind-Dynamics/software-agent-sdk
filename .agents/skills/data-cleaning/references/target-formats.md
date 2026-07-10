# Target Formats

Use one of these formats unless the user confirms a custom target. Validate the
output with `scripts/validate_format.py`.

## `messages`

Preferred for chat, agent, tool-use, and multi-turn SFT data.

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Check whether 91 is prime."},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "type": "function",
          "function": {"name": "factor", "arguments": {"n": 91}}
        }
      ]
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "91 = 7 * 13"},
    {"role": "assistant", "content": "91 is not prime."}
  ]
}
```

Rules:

- `messages` must be a non-empty list.
- Each message has `role`; allowed roles are `system`, `user`, `assistant`,
  and `tool`.
- Each message has `content` or assistant `tool_calls`.
- `content` may be a string or a list of content parts such as
  `{"type": "text", "text": "..."}`.

## `alpaca`

Use for simple single-turn instruction data or the project's "system + input"
standard JSON.

```json
{
  "system": "You are concise.",
  "input": "Subject: high_school_physics\nA question...\nA. ...\nB. ...",
  "output": "B. velocity"
}
```

Rules:

- `input` is required and must be a non-empty string.
- `output` is required and must be a non-empty string.
- `system` is optional but must be a string when present.

## `sharegpt`

Use when downstream tooling expects ShareGPT-style rows.

```json
{
  "conversations": [
    {"from": "human", "value": "Solve 2x + 3 = 11."},
    {"from": "gpt", "value": "x = 4."}
  ]
}
```

Rules:

- `conversations` must be a non-empty list.
- `from` maps to roles: `human`/`user` -> user, `gpt`/`assistant` -> assistant,
  `system` -> system, `tool` -> tool.
- `value` must be non-empty unless the assistant turn has tool calls in a
  custom extension.

## DPO Note

DPO-style sources contain `chosen` and `rejected`. Do not silently collapse a
DPO dataset into SFT unless the user confirms it. If the user wants SFT, use
`chosen` as the assistant conversation and keep `rejected` in metadata or drop
it. If the user wants DPO, write a custom schema and state it explicitly before
validation, because `scripts/validate_format.py` only checks `alpaca`,
`sharegpt`, and `messages`.

