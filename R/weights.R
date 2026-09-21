limite_weight_files <- function(identifier, revision, local_files_only) {
  if (dir.exists(identifier)) {
    direct <- file.path(identifier, "model.safetensors")
    if (file.exists(direct)) {
      return(direct)
    }
    index_path <- file.path(identifier, "model.safetensors.index.json")
    if (!file.exists(index_path)) {
      cli::cli_abort("No safetensors checkpoint was found in {.path {identifier}}.")
    }
    index <- jsonlite::fromJSON(index_path, simplifyVector = TRUE)
    return(file.path(identifier, unique(unname(index$weight_map))))
  }
  if (file.exists(identifier)) {
    return(identifier)
  }

  direct <- tryCatch(
    hfhub::hub_download(
      repo_id = identifier,
      filename = "model.safetensors",
      revision = revision,
      local_files_only = local_files_only
    ),
    error = identity
  )
  if (!inherits(direct, "error")) {
    return(direct)
  }

  index_path <- hfhub::hub_download(
    repo_id = identifier,
    filename = "model.safetensors.index.json",
    revision = revision,
    local_files_only = local_files_only
  )
  index <- jsonlite::fromJSON(index_path, simplifyVector = TRUE)
  files <- unique(unname(index$weight_map))
  vapply(files, function(filename) {
    hfhub::hub_download(
      repo_id = identifier,
      filename = filename,
      revision = revision,
      local_files_only = local_files_only
    )
  }, character(1))
}

#' Load a Limite checkpoint state dictionary
#'
#' Loads safetensors directly as R torch tensors. `identifier` may be a Hugging
#' Face model identifier, a checkpoint directory, or one safetensors file.
#'
#' @param identifier Model identifier or local path.
#' @param revision Hugging Face revision.
#' @param local_files_only If `TRUE`, do not download missing Hub files.
#'
#' @return A named list of torch tensors.
#' @export
limite_state_dict <- function(identifier, revision = "main", local_files_only = FALSE) {
  files <- limite_weight_files(identifier, revision, local_files_only)
  states <- lapply(files, safetensors::safe_load_file, framework = "torch")
  names_flat <- unlist(lapply(states, names), use.names = FALSE)
  duplicate <- unique(names_flat[duplicated(names_flat)])
  if (length(duplicate) > 0L) {
    cli::cli_abort("Duplicate checkpoint tensors: {paste(duplicate, collapse = ', ')}.")
  }
  unlist(states, recursive = FALSE, use.names = TRUE)
}

#' Construct Limite from checkpoint configuration
#'
#' @inheritParams limite_state_dict
#'
#' @return An uninitialized Limite model with the requested architecture.
#' @export
limite_from_config <- function(identifier, revision = "main", local_files_only = FALSE) {
  limite(read_limite_config(identifier, revision, local_files_only))
}

#' Load a pretrained Limite model
#'
#' The model is created on the meta device and then attached to the checkpoint
#' tensors, avoiding a second initialized copy of the 1B parameters.
#'
#' @inheritParams limite_state_dict
#' @param device Optional torch device such as `"cpu"` or `"cuda"`.
#'
#' @return A pretrained Limite model in evaluation mode.
#' @export
limite_from_pretrained <- function(identifier = "paradigma-inc/limite-1b-violetto",
                                   revision = "main", local_files_only = FALSE,
                                   device = NULL) {
  config <- read_limite_config(identifier, revision, local_files_only)
  with_device(device = "meta", {
    model <- limite(config)
  })
  state <- limite_state_dict(identifier, revision, local_files_only)
  model$load_state_dict(state, strict = TRUE, .refer_to_state_dict = TRUE)
  model$fold_projection_scales()
  if (!is.null(device)) {
    model$to(device = device)
  }
  model$eval()
  model
}
