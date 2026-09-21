# Load a pretrained Limite model

The model is created on the meta device and then attached to the
checkpoint tensors, avoiding a second initialized copy of the 1B
parameters.

## Usage

``` r
limite_from_pretrained(
  identifier = "paradigma-inc/limite-1b-violetto",
  revision = "main",
  local_files_only = FALSE,
  device = NULL
)
```

## Arguments

- identifier:

  Model identifier or local path.

- revision:

  Hugging Face revision.

- local_files_only:

  If `TRUE`, do not download missing Hub files.

- device:

  Optional torch device such as `"cpu"` or `"cuda"`.

## Value

A pretrained Limite model in evaluation mode.
