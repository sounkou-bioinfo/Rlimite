# Generate tokens with a Limite model

Public token IDs follow the zero-based Hugging Face convention. The
model uses a key/value cache after the prompt prefill.

## Usage

``` r
limite_generate(
  model,
  input_ids,
  max_new_tokens = 64L,
  temperature = 0.8,
  top_k = 50L,
  eos_token_id = model$config$eos_token_id,
  seed = NULL,
  include_prompt = FALSE,
  callback = NULL
)
```

## Arguments

- model:

  A model returned by
  [`limite_from_pretrained()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_from_pretrained.md)
  or
  [`limite()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite.md).

- input_ids:

  Zero-based token IDs for one prompt.

- max_new_tokens:

  Maximum number of tokens to generate.

- temperature:

  Sampling temperature. Use zero for greedy decoding.

- top_k:

  Number of highest-scoring tokens retained for sampling, or `NULL` for
  the complete vocabulary.

- eos_token_id:

  Zero-based end token. Defaults to the model config.

- seed:

  Optional torch random seed.

- include_prompt:

  Include the prompt IDs in the returned vector.

- callback:

  Optional function called with each generated token ID.

## Value

An integer vector of zero-based token IDs.
