# Construct Limite from checkpoint configuration

Construct Limite from checkpoint configuration

## Usage

``` r
limite_from_config(identifier, revision = "main", local_files_only = FALSE)
```

## Arguments

- identifier:

  Model identifier or local path.

- revision:

  Hugging Face revision.

- local_files_only:

  If `TRUE`, do not download missing Hub files.

## Value

An uninitialized Limite model with the requested architecture.
