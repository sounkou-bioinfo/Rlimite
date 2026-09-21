# Construct a Limite language model

Construct a Limite language model

## Usage

``` r
limite(config = limite_config())
```

## Arguments

- config:

  A configuration produced by
  [`limite_config()`](https://sounkou-bioinfo.github.io/Rlimite/reference/limite_config.md)
  or loaded from a checkpoint.

## Value

A
[`torch::nn_module`](https://torch.mlverse.org/docs/reference/nn_module.html)
implementing `LimiteForCausalLM`.
