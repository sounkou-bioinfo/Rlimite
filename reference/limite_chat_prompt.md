# Format messages with the published Limite template

Adds the Limite end-of-text and ChatML control tokens. A default system
message is inserted when the first message is not a system message.

## Usage

``` r
limite_chat_prompt(messages, add_generation_prompt = TRUE)
```

## Arguments

- messages:

  A list of messages. Each message contains scalar `role` and `content`
  fields. Supported roles are `system`, `user`, and `assistant`.

- add_generation_prompt:

  Append the assistant prefix.

## Value

A scalar prompt string.
