# Rlimite: a math model on the machine you already have

**One package, one argument: a trained model is a file, and a file is
enough. Solving a mathematical problem should not require a Python
interpreter, a serving process, or a second copy of the weights in
another runtime.**

Rlimite implements the
[Limite](https://github.com/paradigma-inc/limite-violetto) decoder
architecture directly in R with [`torch`](https://torch.mlverse.org/).
The published safetensors and tokenizer are read where they already
live, in the Hugging Face cache, and executed in the calling R session.
There is no `reticulate` bridge, no vLLM, no model server, and no
network round-trip per prompt. After the first download the machine in
front of you is the whole system, on CPU or on GPU.

Limite 1B - Violetto is math-specialized and lightly instruction-tuned
for independent single-turn problems. It is a solver, not a general chat
assistant, and this README does not pretend otherwise. Base and Base
Soup are earlier checkpoints of the same published family and are shown
here doing what they actually do.

## Installation

``` r

install.packages(
  "Rlimite",
  repos = c(
    "https://sounkou-bioinfo.r-universe.dev",
    "https://cloud.r-project.org"
  )
)
```

Rlimite needs [torch](https://torch.mlverse.org/docs) with its LibTorch
runtime, plus [safetensors](https://github.com/mlverse/safetensors),
[hfhub](https://mlverse.github.io/hfhub/) and
[tok](https://github.com/mlverse/tok). Each published BF16 checkpoint is
about 2.08 GB on disk, before activations and caches.

## A file and a tokenizer are enough

Every example below is evaluated when this README is rendered, against
the pinned revisions in
[`limite_checkpoints()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_checkpoints.md).
Nothing on this page is a transcript written by hand.

``` r

library(Rlimite)

violetto <- subset(limite_checkpoints(), checkpoint == "Violetto")
model <- limite_from_pretrained(
  violetto$model_id,
  revision = violetto$revision,
  device = device
)
tokenizer <- limite_tokenizer(violetto$model_id, revision = violetto$revision)

messages <- list(list(
  role = "user",
  content = "Compute 17 * 23. Show your reasoning."
))
solution <- limite_chat(
  model,
  messages,
  tokenizer = tokenizer,
  max_new_tokens = 256,
  temperature = 0
)
```

The model reasons before it commits to an answer, so the trace is long.
Its opening is the interesting part:

``` r

show_trace(solution)
#> <think>We need to compute 17 * 23. The user asks: "Compute 17 * 23. Show your reasoning. Please reason step by step, and put your final answer within \boxed{}."
#>
#> We need to provide a step-by-step multiplication and final answer in a box. This is straightforward: 17*23 = 17*(20+3) = 340 + 51 = 391. Or 23*10=230, 23*7=161, sum=391. So final answer \boxed{391}.
#>
#> We need to ensure we follow instructions: reason step by step, final answer within \boxed{}. No extra text outside? It says "put your final answer within \boxed{}". We can include reasoning before that. That's fine.
#>
#> Thus produce a short reasoning: 17*23 = 17*(20+3) = 17*20 + 17*3 = 340 + 51 = 391. Then final answer \boxed{391}.
#>
#> We must ensure no disallowed content. It's simple arithmetic
```

Weights attach on the meta device and are then bound to the checkpoint
tensors, so loading never materializes a second initialized copy of the
1B parameters. The safetensors mapping stays authoritative.

## The device is an argument, not an architecture

The same call runs on either processor. `device = "cuda"` moves the
parameters onto the GPU, and generation follows the weights rather than
being rewritten around them:

``` r

model <- limite_from_pretrained(
  "paradigma-inc/limite-1b-violetto",
  device = "cuda"
)
```

[`limite_generate()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_generate.md)
reads its device from the embedding weights, so no calling code carries
a device argument at all. This page reports whichever processor rendered
it, and `RLIMITE_README_DEVICE` forces the choice:

``` r

prompt <- limite_chat_prompt(messages)
prompt_ids <- tokenizer$encode(prompt, add_special_tokens = FALSE)$ids

elapsed <- system.time(
  tokens <- limite_generate(
    model,
    prompt_ids,
    max_new_tokens = 32L,
    temperature = 0,
    eos_token_id = NULL
  )
)[["elapsed"]]

data.frame(
  device = device,
  prompt_tokens = length(prompt_ids),
  new_tokens = length(tokens),
  seconds = round(elapsed, 1),
  tokens_per_second = round(length(tokens) / elapsed, 2)
)
#>   device prompt_tokens new_tokens seconds tokens_per_second
#> 1    cpu            33         32      11              2.91
```

This is a single warm greedy decode with a key/value cache, not a kernel
microbenchmark, and it should be read as such.

## Watching it decode

Generation is not a black box that returns after a minute.
[`limite_generate()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_generate.md)
takes a `callback` fired on every sampled token, so a solution can
stream into the console while it is produced:

``` r

invisible(limite_generate(
  model,
  prompt_ids,
  max_new_tokens = 24L,
  temperature = 0,
  eos_token_id = NULL,
  callback = function(id) cat(tokenizer$decode(id, skip_special_tokens = TRUE))
))
#> <think>We need to compute 17 * 23. The user asks: "Compute 17 *
```

## Nothing has to leave the machine

Once a revision is in the Hugging Face cache, `local_files_only = TRUE`
removes the network from the loading path entirely:

``` r

model <- limite_from_pretrained(
  "paradigma-inc/limite-1b-violetto",
  local_files_only = TRUE
)
```

The same argument accepts a checkpoint directory or a single
`.safetensors` file, so an air-gapped machine loads weights copied in by
hand, and a machine with no Python installation at all is not a special
case.

## The family is three checkpoints, not one

All three pinned revisions are loaded and given the same problem, with
twelve greedy tokens each. The comparison is the point: the base
checkpoints continue text, and only Violetto has been tuned to solve.

``` r

checkpoints <- limite_checkpoints()
checkpoints$opening <- NA_character_

for (i in seq_len(nrow(checkpoints))) {
  is_violetto <- identical(checkpoints$checkpoint[[i]], "Violetto")

  if (is_violetto) {
    current_model <- model
    current_tokenizer <- tokenizer
  } else {
    current_model <- limite_from_pretrained(
      checkpoints$model_id[[i]],
      revision = checkpoints$revision[[i]],
      device = device
    )
    current_tokenizer <- limite_tokenizer(
      checkpoints$model_id[[i]],
      revision = checkpoints$revision[[i]]
    )
  }

  ids <- limite_generate(
    current_model,
    prompt_ids,
    max_new_tokens = 12L,
    temperature = 0,
    eos_token_id = NULL
  )
  checkpoints$opening[[i]] <- current_tokenizer$decode(
    ids,
    skip_special_tokens = TRUE
  )

  if (!is_violetto) {
    rm(current_model, current_tokenizer)
    gc()
  }
}
```

| checkpoint | profile | revision | opening |
|:---|:---|:---|:---|
| Base | base | c55f6dd9741d89b88186235c6d43bff367ae3cb0 | To compute 17 \* 23, we can |
| Base Soup | base-soup | 9ade12f28483ec61896427bb983719501325e275 | You are a helpful assistant. 23 \* 1 |
| Violetto | math-single-turn | e47321c08d6820a0b2491bc8a10381999f2d0cde | We need to compute 17 \* 23 |

The revisions are immutable commit hashes rather than `main`, so this
table and the real-model CI matrix describe the same three artifacts.

## What the architecture actually required

The shared checkpoint contract is 48 decoder layers, grouped-query
attention, mixed global and sliding-window layers, and a 131,072-token
context.

``` R
#>              vocab_size             hidden_size       num_hidden_layers
#>                  151680                    1280                      48
#>     num_attention_heads     num_key_value_heads max_position_embeddings
#>                      10                       2                  131072
```

None of the following is optional for numerical agreement with the
published checkpoints, and each is implemented in R rather than deferred
to a kernel library:

- partial interleaved RoPE, with NoPE on the global layers
- value embeddings and per-head attention gates
- cross-stream attention
- MUDD history mixing with learned taps
- learned residual coefficients, and projection-scale folding at load
  time
- sigmoid-softcapped logits
- bounded key/value caches on the sliding-window layers

Public token IDs stay zero-based, matching the Hugging Face tokenizers.
The conversion to R torch’s one-based embedding indices happens only at
the model boundary, so IDs handed to
[`limite_generate()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_generate.md)
and read back from the `token_ids` attribute line up with the Hub
tokenizer exactly.

## What is not measured yet

The throughput table above reports one device: the one that rendered the
page. A CPU-versus-CUDA comparison on the same prompt, the same revision
and the same decode length is the honest next measurement, and it is not
in this README because it has not been run. Attention is a dense
materialized product here rather than a fused kernel, and no claim is
made about how this stack compares with a tuned serving runtime on the
same weights. Those numbers belong here once they exist.

## Model weights

Rlimite does not redistribute model weights. Access to and use of the
checkpoints remain subject to the terms attached to the selected model
repository.
