limite_sample_token <- function(logits, temperature, top_k) {
  if (temperature == 0) {
    return(as.integer(torch_argmax(logits)$item() - 1L))
  }
  logits <- logits / temperature
  if (!is.null(top_k)) {
    top_k <- min(as.integer(top_k), logits$numel())
    top <- logits$topk(top_k)
    probability <- nnf_softmax(top[[1]], dim = -1)
    selected <- torch_multinomial(probability, num_samples = 1L)
    return(as.integer(top[[2]]$gather(-1, selected)$item() - 1L))
  }
  probability <- nnf_softmax(logits, dim = -1)
  as.integer(torch_multinomial(probability, num_samples = 1L)$item() - 1L)
}

#' Generate tokens with a Limite model
#'
#' Public token IDs follow the zero-based Hugging Face convention. The model
#' uses a key/value cache after the prompt prefill.
#'
#' @param model A model returned by [limite_from_pretrained()] or [limite()].
#' @param input_ids Zero-based token IDs for one prompt.
#' @param max_new_tokens Maximum number of tokens to generate.
#' @param temperature Sampling temperature. Use zero for greedy decoding.
#' @param top_k Number of highest-scoring tokens retained for sampling, or
#'   `NULL` for the complete vocabulary.
#' @param eos_token_id Zero-based end token. Defaults to the model config.
#' @param seed Optional torch random seed.
#' @param include_prompt Include the prompt IDs in the returned vector.
#' @param callback Optional function called with each generated token ID.
#'
#' @return An integer vector of zero-based token IDs.
#' @export
limite_generate <- function(model, input_ids, max_new_tokens = 64L,
                            temperature = 0.8, top_k = 50L,
                            eos_token_id = model$config$eos_token_id,
                            seed = NULL, include_prompt = FALSE,
                            callback = NULL) {
  input_ids <- as.integer(input_ids)
  valid_ids <- all(c(length(input_ids) > 0L, !anyNA(input_ids), input_ids >= 0L))
  limite_abort_if(!valid_ids, "`input_ids` must contain non-negative token IDs.")
  valid_count <- all(c(
    length(max_new_tokens) == 1L,
    !is.na(max_new_tokens),
    max_new_tokens >= 0L
  ))
  limite_abort_if(!valid_count, "`max_new_tokens` must be one non-negative integer.")
  valid_temperature <- all(c(
    length(temperature) == 1L,
    !is.na(temperature),
    temperature >= 0
  ))
  limite_abort_if(!valid_temperature, "`temperature` must be one non-negative number.")
  if (!is.null(seed)) {
    torch_manual_seed(as.integer(seed))
  }

  device <- model$model$embed_tokens$weight$device
  prompt <- torch_tensor(
    matrix(input_ids + 1L, nrow = 1L),
    dtype = torch_int64(),
    device = device
  )
  model$eval()
  generated <- integer()

  with_no_grad({
    result <- model(prompt, use_cache = TRUE, logits_to_keep = 1L)
    for (step in seq_len(as.integer(max_new_tokens))) {
      logits <- result$logits[1, result$logits$size(2L), ]
      token <- limite_sample_token(logits, temperature, top_k)
      generated <- c(generated, token)
      if (!is.null(callback)) {
        callback(token)
      }
      if (!is.null(eos_token_id) && token == as.integer(eos_token_id)) {
        break
      }
      next_input <- torch_tensor(
        matrix(token + 1L, nrow = 1L),
        dtype = torch_int64(),
        device = device
      )
      result <- model(
        next_input,
        cache = result$cache,
        use_cache = TRUE,
        logits_to_keep = 1L
      )
    }
  })

  if (include_prompt) c(input_ids, generated) else generated
}

#' Load the Limite tokenizer
#'
#' @inheritParams limite_state_dict
#'
#' @return A `tok_tokenizer`.
#' @export
limite_tokenizer <- function(identifier = "paradigma-inc/limite-1b-violetto",
                             revision = "main", local_files_only = FALSE) {
  if (dir.exists(identifier)) {
    path <- file.path(identifier, "tokenizer.json")
    if (!file.exists(path)) {
      cli::cli_abort("No `tokenizer.json` was found in {.path {identifier}}.")
    }
    return(tok::tokenizer$from_file(path))
  }
  if (file.exists(identifier)) {
    return(tok::tokenizer$from_file(identifier))
  }
  if (local_files_only) {
    path <- hfhub::hub_download(
      repo_id = identifier,
      filename = "tokenizer.json",
      revision = revision,
      local_files_only = TRUE
    )
    return(tok::tokenizer$from_file(path))
  }
  tok::tokenizer$from_pretrained(identifier, revision = revision)
}

#' Format messages with the published Limite template
#'
#' Adds the Limite end-of-text and ChatML control tokens. A default system
#' message is inserted when the first message is not a system message.
#'
#' @param messages A list of messages. Each message contains scalar `role` and
#'   `content` fields. Supported roles are `system`, `user`, and `assistant`.
#' @param add_generation_prompt Append the assistant prefix.
#'
#' @return A scalar prompt string.
#' @export
limite_chat_prompt <- function(messages, add_generation_prompt = TRUE) {
  if (is.data.frame(messages)) {
    messages <- lapply(seq_len(nrow(messages)), function(i) as.list(messages[i, , drop = FALSE]))
  }
  if (!is.list(messages) || !length(messages)) {
    cli::cli_abort("`messages` must be a non-empty list.")
  }
  validated <- lapply(messages, function(message) {
    role <- message$role
    content <- message$content
    if (length(role) != 1L || !role %in% c("system", "user", "assistant")) {
      cli::cli_abort("Each message role must be `system`, `user`, or `assistant`.")
    }
    if (length(content) != 1L || is.na(content)) {
      cli::cli_abort("Each message must have one non-missing `content` string.")
    }
    list(role = role, content = as.character(content))
  })

  first_is_system <- identical(validated[[1L]]$role, "system")
  system <- if (first_is_system) {
    validated[[1L]]$content
  } else {
    "You are a helpful assistant."
  }
  turns <- if (first_is_system) validated[-1L] else validated
  rendered <- vapply(turns, function(message) {
    paste0(
      "<|im_start|>", message$role, "\n",
      message$content,
      "<|im_end|>\n"
    )
  }, character(1))

  paste0(
    "<|endoftext|><|im_start|>system\n",
    system,
    "<|im_end|>\n",
    paste0(rendered, collapse = ""),
    if (add_generation_prompt) "<|im_start|>assistant\n" else ""
  )
}

#' Generate a Limite response
#'
#' Violetto is intended for direct, independent mathematical problems rather
#' than general multi-turn assistant use.
#'
#' @param model A pretrained Limite model.
#' @param messages Conversation messages accepted by [limite_chat_prompt()].
#' @param tokenizer A tokenizer returned by [limite_tokenizer()].
#' @param ... Arguments passed to [limite_generate()].
#'
#' @return A scalar response string with generated token IDs attached as the
#'   `token_ids` attribute.
#' @export
limite_chat <- function(model, messages, tokenizer = limite_tokenizer(), ...) {
  prompt <- limite_chat_prompt(messages)
  prompt_ids <- tokenizer$encode(prompt, add_special_tokens = FALSE)$ids
  token_ids <- limite_generate(model, prompt_ids, ...)
  output <- tokenizer$decode(token_ids, skip_special_tokens = TRUE)
  structure(output, token_ids = token_ids, prompt = prompt)
}
