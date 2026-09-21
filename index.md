# Rlimite

[![R-CMD-check](https://github.com/sounkou-bioinfo/Rlimite/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/sounkou-bioinfo/Rlimite/actions/workflows/R-CMD-check.yaml)
[![Real model
tests](https://github.com/sounkou-bioinfo/Rlimite/actions/workflows/real-model.yaml/badge.svg)](https://github.com/sounkou-bioinfo/Rlimite/actions/workflows/real-model.yaml)
[![R-universe](https://sounkou-bioinfo.r-universe.dev/badges/Rlimite)](https://sounkou-bioinfo.r-universe.dev/Rlimite)
[![Lifecycle:
experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)

Rlimite implements the
[Limite](https://github.com/paradigma-inc/limite-violetto) decoder
architecture directly in R with [`torch`](https://torch.mlverse.org/).
It loads the official safetensors and tokenizer assets without Python,
vLLM, or a model-serving process.

Limite 1B - Violetto is a math-specialized, lightly instruction-tuned
model for independent single-turn problems. It is not intended to behave
like a general chat assistant. The Base and Base Soup checkpoints expose
earlier members of the same published model family.

## Published checkpoints

The README pins the exact revisions used for its evaluated examples and
for the real-model CI matrix.

``` r

checkpoints <- limite_checkpoints()
knitr::kable(checkpoints)
```

| checkpoint | model_id | revision | profile |
|:---|:---|:---|:---|
| Base | paradigma-inc/limite-1b-base | c55f6dd9741d89b88186235c6d43bff367ae3cb0 | base |
| Base Soup | paradigma-inc/limite-1b-base-soup | 9ade12f28483ec61896427bb983719501325e275 | base-soup |
| Violetto | paradigma-inc/limite-1b-violetto | e47321c08d6820a0b2491bc8a10381999f2d0cde | math-single-turn |

## Architecture

The shared checkpoint contract contains 48 decoder layers, grouped-query
attention, mixed global and sliding-window layers, and a 131,072-token
context limit.

``` r

config <- limite_config()
unlist(config[c(
  "vocab_size",
  "hidden_size",
  "num_hidden_layers",
  "num_attention_heads",
  "num_key_value_heads",
  "max_position_embeddings"
)])
#>              vocab_size             hidden_size       num_hidden_layers
#>                  151680                    1280                      48
#>     num_attention_heads     num_key_value_heads max_position_embeddings
#>                      10                       2                  131072
```

The implementation includes partial interleaved RoPE, NoPE global
layers, value embeddings, per-head attention gates, cross-stream
attention, MUDD history mixing, learned residual coefficients,
projection-scale folding, sigmoid-softcapped logits, and bounded
local-layer key/value caches.

## Installation

Install Rlimite from R-universe with:

``` r

install.packages(
  "Rlimite",
  repos = c(
    "https://sounkou-bioinfo.r-universe.dev",
    "https://cloud.r-project.org"
  )
)
```

Rlimite requires [torch](https://torch.mlverse.org/docs) with its
LibTorch runtime, [safetensors](https://github.com/mlverse/safetensors),
[hfhub](https://mlverse.github.io/hfhub/), and
[tok](https://github.com/mlverse/tok). A CUDA-enabled LibTorch
installation is recommended, but all examples below are evaluated on
CPU. Each published BF16 checkpoint is about 2.08 GB before runtime
activations and caches.

## Verify all three checkpoints

This chunk loads every pinned checkpoint and begins the same
mathematical problem with eight greedy tokens. It reports
runtime-contract checks rather than comparing the quality of differently
trained checkpoints.

``` r

problem <- "Compute 17 * 23. Show your reasoning."
messages <- list(list(role = "user", content = problem))
checks <- data.frame(
  checkpoint = checkpoints$checkpoint,
  generated_tokens = integer(nrow(checkpoints)),
  valid_token_ids = logical(nrow(checkpoints))
)

for (i in seq_len(nrow(checkpoints))) {
  model <- limite_from_pretrained(
    checkpoints$model_id[[i]],
    revision = checkpoints$revision[[i]],
    device = "cpu"
  )
  tokenizer <- limite_tokenizer(
    checkpoints$model_id[[i]],
    revision = checkpoints$revision[[i]]
  )
  continuation <- limite_chat(
    model,
    messages,
    tokenizer = tokenizer,
    max_new_tokens = 8,
    temperature = 0,
    eos_token_id = NULL
  )
  token_ids <- attr(continuation, "token_ids")
  checks$generated_tokens[[i]] <- length(token_ids)
  checks$valid_token_ids[[i]] <- all(
    token_ids >= 0L & token_ids < model$config$vocab_size
  )

  if (i < nrow(checkpoints)) {
    rm(model, tokenizer)
    gc()
  }
}

stopifnot(all(checks$generated_tokens == 8L), all(checks$valid_token_ids))
knitr::kable(checks)
```

| checkpoint | generated_tokens | valid_token_ids |
|:-----------|-----------------:|:----------------|
| Base       |                8 | TRUE            |
| Base Soup  |                8 | TRUE            |
| Violetto   |                8 | TRUE            |

## Solve a problem with Violetto

The final model retained by the preceding evaluated chunk is Violetto.
Its published message template wraps the direct single-turn problem.
Violetto emits a long reasoning trace; the evaluated example extracts
the boxed result from the generated token stream so the landing page
stays concise.

``` r

solution <- limite_chat(
  model,
  messages,
  tokenizer = tokenizer,
  max_new_tokens = 256,
  temperature = 0
)
boxed_answer <- regmatches(
  solution,
  regexpr("\\\\boxed\\{[^}]+\\}", solution, perl = TRUE)
)
stopifnot(length(boxed_answer) == 1L)
boxed_answer
#> [1] "\\boxed{391}"
```

## Token IDs and model assets

Public token IDs are zero-based, matching Hugging Face tokenizers.
Rlimite converts them to R torch’s one-based embedding indices only at
the model boundary.

Rlimite does not redistribute model weights. Weight access and use
remain subject to the terms attached to the selected model repository.
