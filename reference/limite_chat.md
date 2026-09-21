# Generate a Limite response

Violetto is intended for direct, independent mathematical problems
rather than general multi-turn assistant use.

## Usage

``` r
limite_chat(model, messages, tokenizer = limite_tokenizer(), ...)
```

## Arguments

- model:

  A pretrained Limite model.

- messages:

  Conversation messages accepted by
  [`limite_chat_prompt()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_chat_prompt.md).

- tokenizer:

  A tokenizer returned by
  [`limite_tokenizer()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_tokenizer.md).

- ...:

  Arguments passed to
  [`limite_generate()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_generate.md).

## Value

A scalar response string with generated token IDs attached as the
`token_ids` attribute.
