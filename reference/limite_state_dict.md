# Load a Limite checkpoint state dictionary

Loads safetensors directly as R torch tensors. `identifier` may be a
Hugging Face model identifier, a checkpoint directory, or one
safetensors file.

## Usage

``` r
limite_state_dict(identifier, revision = "main", local_files_only = FALSE)
```

## Arguments

- identifier:

  Model identifier or local path.

- revision:

  Hugging Face revision.

- local_files_only:

  If `TRUE`, do not download missing Hub files.

## Value

A named list of torch tensors.
