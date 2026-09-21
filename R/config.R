limite_abort_if <- function(condition, message) {
  if (isTRUE(condition)) {
    cli::cli_abort(message)
  }
}

#' Published Limite checkpoints
#'
#' Returns the model identifiers and immutable revisions covered by Rlimite's
#' real-model tests. Violetto is the math-specialized single-turn solver; Base
#' and Base Soup are published family checkpoints.
#'
#' @return A data frame with checkpoint labels, model identifiers, revisions,
#'   and profiles.
#' @export
limite_checkpoints <- function() {
  data.frame(
    checkpoint = c("Base", "Base Soup", "Violetto"),
    model_id = paste0(
      "paradigma-inc/",
      c("limite-1b-base", "limite-1b-base-soup", "limite-1b-violetto")
    ),
    revision = c(
      "c55f6dd9741d89b88186235c6d43bff367ae3cb0",
      "9ade12f28483ec61896427bb983719501325e275",
      "e47321c08d6820a0b2491bc8a10381999f2d0cde"
    ),
    profile = c("base", "base-soup", "math-single-turn"),
    stringsAsFactors = FALSE
  )
}

#' Limite model configuration
#'
#' Creates the architecture contract used by the published 1B checkpoints.
#' Values can be overridden to construct smaller models for experiments.
#'
#' @param ... Named configuration overrides.
#'
#' @return A validated Limite configuration list.
#' @export
limite_config <- function(...) {
  config <- list(
    architectures = "LimiteForCausalLM",
    model_type = "limite",
    vocab_size = 151680L,
    hidden_size = 1280L,
    intermediate_size = 3328L,
    num_hidden_layers = 48L,
    num_attention_heads = 10L,
    num_key_value_heads = 2L,
    head_dim = 128L,
    max_position_embeddings = 131072L,
    attention_softmax_scale = 0.1,
    sliding_window = 1024L,
    global_window = -1L,
    global_layers = seq.int(3L, 47L, by = 4L),
    global_nope = TRUE,
    rope_n_pairs = 32L,
    rope_base_local = 1024,
    rope_base_global = 1024,
    rope_per_layer = FALSE,
    mlp_type = "swiglu",
    ve_dim = 128L,
    ve_stored_heads = 2L,
    ve_gate_channels = 12L,
    ve_gate_scale = 2,
    ve_layers = seq.int(1L, 46L, by = 3L),
    attn_gate_channels = 128L,
    attn_gate_scale = 2,
    xsa = TRUE,
    xsa_layers = 0:47,
    xsa_normalize_eps = 1e-4,
    mudd = TRUE,
    mudd_layers = c(24L, 47L),
    mudd_tap_idx = list(`24` = c(0L, 12L, 24L), `47` = c(0L, 23L, 47L)),
    mudd_taps = 3L,
    mudd_inter = 32L,
    mudd_mlp = TRUE,
    mudd_r_site = "resid",
    final_softcap = 0,
    softcap_logits = list(kind = "sigmoid", a = 23, b = 5, c = 7.5),
    lm_head_precision_mode = "oracle_exact",
    tie_word_embeddings = TRUE,
    bos_token_id = 151643L,
    eos_token_id = 151645L,
    pad_token_id = 151643L,
    torch_dtype = "bfloat16"
  )

  overrides <- list(...)
  if (length(overrides) > 0L) {
    unnamed <- !nzchar(names(overrides))
    if (is.null(names(overrides)) || any(unnamed)) {
      cli::cli_abort("Every configuration override must be named.")
    }
    config[names(overrides)] <- overrides
  }

  validate_limite_config(config)
}

validate_limite_config <- function(config) {
  required <- c(
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "attention_softmax_scale", "sliding_window",
    "global_window", "global_layers", "global_nope", "rope_n_pairs",
    "rope_base_local", "rope_per_layer", "mlp_type",
    "ve_dim", "ve_stored_heads", "ve_gate_channels", "ve_gate_scale",
    "ve_layers", "attn_gate_channels", "attn_gate_scale", "xsa",
    "xsa_layers", "xsa_normalize_eps", "mudd", "mudd_layers",
    "mudd_tap_idx", "mudd_taps", "mudd_inter", "mudd_mlp", "mudd_r_site",
    "final_softcap", "softcap_logits", "lm_head_precision_mode",
    "tie_word_embeddings"
  )
  missing <- setdiff(required, names(config))
  limite_abort_if(
    length(missing) > 0L,
    c(
      "Invalid Limite configuration.",
      "x" = "Missing fields: {paste(missing, collapse = ', ')}."
    )
  )

  integers <- c(
    "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "head_dim",
    "max_position_embeddings", "rope_n_pairs", "ve_dim", "ve_stored_heads", "ve_gate_channels", "attn_gate_channels",
    "mudd_taps", "mudd_inter"
  )
  valid_integer <- vapply(config[integers], function(x) {
    if (!is.numeric(x) || length(x) != 1L) {
      return(FALSE)
    }
    isTRUE(x > 0) && isTRUE(x == as.integer(x))
  }, logical(1))
  limite_abort_if(
    !all(valid_integer),
    "Positive integer fields are invalid: {paste(integers[!valid_integer], collapse = ', ')}."
  )

  limite_abort_if(
    config$hidden_size != config$num_attention_heads * config$head_dim,
    "`hidden_size` must equal `num_attention_heads * head_dim`."
  )
  limite_abort_if(
    config$num_attention_heads %% config$num_key_value_heads != 0L,
    "`num_attention_heads` must be divisible by `num_key_value_heads`."
  )
  limite_abort_if(
    isTRUE(config$rope_per_layer),
    "Per-layer RoPE frequencies are not supported by the published checkpoint."
  )
  limite_abort_if(
    config$global_window >= 0L,
    "The published checkpoint requires unlimited global attention."
  )
  limite_abort_if(
    config$head_dim %% 2L != 0L || 2L * config$rope_n_pairs > config$head_dim,
    "`head_dim` must be even and at least `2 * rope_n_pairs`."
  )
  limite_abort_if(config$ve_dim != config$head_dim, "`ve_dim` must equal `head_dim`.")
  limite_abort_if(
    config$ve_stored_heads < config$num_key_value_heads,
    "`ve_stored_heads` cannot be smaller than `num_key_value_heads`."
  )
  limite_abort_if(
    config$ve_gate_channels > config$hidden_size ||
      config$attn_gate_channels > config$hidden_size,
    "Gate input channels cannot exceed `hidden_size`."
  )
  limite_abort_if(
    !identical(config$mlp_type, "swiglu"),
    "Only the checkpoint's `swiglu` MLP is supported."
  )
  limite_abort_if(
    !isTRUE(config$tie_word_embeddings),
    "Limite requires tied token embedding and output weights."
  )
  limite_abort_if(
    !identical(config$lm_head_precision_mode, "oracle_exact"),
    "Only `lm_head_precision_mode = 'oracle_exact'` is supported."
  )
  limite_abort_if(
    isTRUE(config$mudd) &&
      (!isTRUE(config$mudd_mlp) || !identical(config$mudd_r_site, "resid")),
    "MUDD requires its residual-way mixer at the `resid` site."
  )
  limite_abort_if(
    !identical(config$softcap_logits$kind, "sigmoid"),
    "Only sigmoid logit softcapping is supported."
  )

  layer_fields <- c("global_layers", "ve_layers", "xsa_layers", "mudd_layers")
  for (field in layer_fields) {
    values <- as.integer(unlist(config[[field]], use.names = FALSE))
    limite_abort_if(
      length(values) > 0L && any(values < 0L | values >= config$num_hidden_layers),
      "`{field}` contains a layer outside the model."
    )
    config[[field]] <- values
  }

  taps <- config$mudd_tap_idx
  if (isTRUE(config$mudd)) {
    limite_abort_if(
      is.null(names(taps)) || !setequal(as.integer(names(taps)), config$mudd_layers),
      "`mudd_tap_idx` must cover exactly `mudd_layers`."
    )
    taps <- lapply(taps, function(x) as.integer(unlist(x, use.names = FALSE)))
    limite_abort_if(
      any(lengths(taps) > config$mudd_taps),
      "A MUDD layer has more history taps than `mudd_taps`."
    )
    limite_abort_if(
      any(unlist(taps, use.names = FALSE) < 0L),
      "MUDD history indices must be non-negative."
    )
    invalid_tap <- vapply(names(taps), function(layer) {
      any(taps[[layer]] > as.integer(layer))
    }, logical(1))
    limite_abort_if(
      any(invalid_tap),
      "A MUDD layer references history that is not yet available."
    )
  } else {
    config$mudd_layers <- integer()
    taps <- list()
  }
  config$mudd_tap_idx <- taps
  config
}

read_limite_config <- function(identifier, revision = "main", local_files_only = FALSE) {
  if (dir.exists(identifier)) {
    path <- file.path(identifier, "config.json")
  } else if (file.exists(identifier)) {
    path <- identifier
  } else {
    path <- hfhub::hub_download(
      repo_id = identifier,
      filename = "config.json",
      revision = revision,
      local_files_only = local_files_only
    )
  }
  if (!file.exists(path)) {
    cli::cli_abort("Could not find a Limite `config.json` at {.path {path}}.")
  }
  config <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  validate_limite_config(config)
}
