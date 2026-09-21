.torch_rms_norm <- getFromNamespace("torch_rms_norm", "torch")

limite_rms_norm <- function(x) {
  .torch_rms_norm(x, normalized_shape = x$size(-1))
}

limite_linear_no_bias <- function(in_features, out_features) {
  nn_linear(in_features, out_features, bias = FALSE)
}

nn_limite_rotary <- nn_module(
  "nn_limite_rotary",
  initialize = function(config) {
    self$head_dim <- as.integer(config$head_dim)
    self$n_pairs <- as.integer(config$rope_n_pairs)
    self$base <- as.numeric(config$rope_base_local)
    pair_freq <- self$base^(-seq(0, 1, length.out = self$n_pairs))
    frequency <- c(
      rep(pair_freq, each = 2L),
      rep(0, self$head_dim - 2L * self$n_pairs)
    )
    with_device(device = "cpu", {
      self$register_buffer(
        "frequency",
        torch_tensor(frequency, dtype = torch_float32()),
        persistent = FALSE
      )
      self$register_buffer(
        "sign",
        torch_tensor(rep(c(1, -1), self$head_dim / 2L), dtype = torch_float32()),
        persistent = FALSE
      )
    })
  },
  forward = function(q, k, positions) {
    pos <- positions$to(dtype = torch_float32())
    theta <- pos$unsqueeze(-1) * self$frequency$unsqueeze(1)
    cos <- theta$cos()$to(dtype = q$dtype)$unsqueeze(1)$unsqueeze(1)
    sin <- theta$sin()$to(dtype = q$dtype)
    sin <- (sin * self$sign$to(dtype = q$dtype))$unsqueeze(1)$unsqueeze(1)

    rotate <- function(x) {
      shape <- x$shape
      paired <- x$reshape(c(shape[1:3], self$head_dim / 2L, 2L))
      torch_flip(paired, dims = -1)$reshape(shape)
    }

    list(q * cos + rotate(q) * sin, k * cos + rotate(k) * sin)
  }
)

nn_limite_mudd <- nn_module(
  "nn_limite_mudd",
  initialize = function(config) {
    self$hidden_size <- as.integer(config$hidden_size)
    self$intermediate_size <- as.integer(config$mudd_inter)
    self$num_layers <- as.integer(config$num_hidden_layers)
    self$num_taps <- as.integer(config$mudd_taps)
    self$uses_mlp <- isTRUE(config$mudd_mlp)

    self$dense1 <- nn_parameter(torch_zeros(c(self$intermediate_size, self$hidden_size)))
    self$dense2 <- nn_parameter(torch_zeros(c(self$num_layers, self$num_taps, self$intermediate_size)))
    self$bias <- nn_parameter(torch_zeros(c(self$num_layers, self$num_taps)))
    if (self$uses_mlp) {
      self$dense2_mlp <- nn_parameter(torch_zeros(c(self$num_layers, self$num_taps, self$intermediate_size)))
      self$bias_mlp <- nn_parameter(torch_zeros(c(self$num_layers, self$num_taps)))
    }
  },
  coefficients = function(x, layer_index, count, residual_way = FALSE) {
    hidden <- nnf_gelu(nnf_linear(
      limite_rms_norm(x),
      self$dense1$to(dtype = x$dtype)
    ))
    rows <- seq_len(count)
    if (residual_way) {
      weight <- self$dense2_mlp[layer_index + 1L, rows, ]
      bias <- self$bias_mlp[layer_index + 1L, rows]
    } else {
      weight <- self$dense2[layer_index + 1L, rows, ]
      bias <- self$bias[layer_index + 1L, rows]
    }
    nnf_linear(
      hidden,
      weight$to(dtype = hidden$dtype),
      bias$to(dtype = hidden$dtype)
    )
  },
  combine = function(values, x, layer_index, residual_way = FALSE) {
    coefficient <- self$coefficients(x, layer_index, length(values), residual_way)
    output <- coefficient[, , 1L]$unsqueeze(-1)$to(dtype = values[[1]]$dtype) * values[[1]]
    if (length(values) > 1L) {
      for (i in 2:length(values)) {
        output <- output +
          coefficient[, , i]$unsqueeze(-1)$to(dtype = values[[i]]$dtype) * values[[i]]
      }
    }
    output
  }
)

nn_limite_attention <- nn_module(
  "nn_limite_attention",
  initialize = function(config, layer_index) {
    self$hidden_size <- as.integer(config$hidden_size)
    self$num_heads <- as.integer(config$num_attention_heads)
    self$num_kv_heads <- as.integer(config$num_key_value_heads)
    self$head_dim <- as.integer(config$head_dim)
    self$groups <- self$num_heads %/% self$num_kv_heads
    self$scale <- as.numeric(config$attention_softmax_scale)
    self$sliding_window <- as.integer(config$sliding_window)
    self$is_global <- layer_index %in% config$global_layers
    self$uses_rope <- !(isTRUE(config$global_nope) && self$is_global)
    self$uses_value_embeddings <- layer_index %in% config$ve_layers
    self$uses_xsa <- isTRUE(config$xsa) && layer_index %in% config$xsa_layers
    self$value_dim <- as.integer(config$ve_dim)
    self$value_stored_heads <- as.integer(config$ve_stored_heads)
    self$value_gate_channels <- as.integer(config$ve_gate_channels)
    self$value_gate_scale <- as.numeric(config$ve_gate_scale)
    self$attention_gate_channels <- as.integer(config$attn_gate_channels)
    self$attention_gate_scale <- as.numeric(config$attn_gate_scale)
    self$xsa_epsilon <- as.numeric(config$xsa_normalize_eps)
    self$scales_folded <- FALSE

    self$q_proj <- limite_linear_no_bias(self$hidden_size, self$num_heads * self$head_dim)
    self$k_proj <- limite_linear_no_bias(self$hidden_size, self$num_kv_heads * self$head_dim)
    self$v_proj <- limite_linear_no_bias(self$hidden_size, self$num_kv_heads * self$head_dim)
    self$o_proj <- limite_linear_no_bias(self$num_heads * self$head_dim, self$hidden_size)
    self$qkv_scale <- nn_parameter(torch_ones(1L)$squeeze())
    self$o_scale <- nn_parameter(torch_ones(1L)$squeeze())
    self$xsa_alpha <- nn_parameter(torch_zeros(self$num_heads))
    if (self$uses_value_embeddings) {
      self$ve_gate <- nn_parameter(torch_zeros(c(self$value_stored_heads, self$value_gate_channels)))
    }
    if (self$attention_gate_channels > 0L) {
      self$attn_gate <- nn_parameter(torch_zeros(c(self$num_heads, self$attention_gate_channels)))
    }
  },
  fold_projection_scales = function() {
    if (self$scales_folded) {
      return(invisible(self))
    }
    with_no_grad({
      qkv_scale <- self$qkv_scale$to(dtype = self$q_proj$weight$dtype)
      self$q_proj$weight$mul_(qkv_scale)
      self$k_proj$weight$mul_(qkv_scale)
      self$v_proj$weight$mul_(qkv_scale)
      self$o_proj$weight$mul_(self$o_scale$to(dtype = self$o_proj$weight$dtype))
    })
    self$scales_folded <- TRUE
    invisible(self)
  },
  expand_kv = function(x) {
    torch_repeat_interleave(x, repeats = self$groups, dim = 2L)
  },
  attention_mask = function(query_start, query_length, key_start, key_length, device) {
    query_position <- torch_arange(
      start = query_start,
      end = query_start + query_length - 1L,
      dtype = torch_int64(),
      device = device
    )
    key_position <- torch_arange(
      start = key_start,
      end = key_start + key_length - 1L,
      dtype = torch_int64(),
      device = device
    )
    mask <- key_position$unsqueeze(1) <= query_position$unsqueeze(2)
    if (!self$is_global) {
      mask <- mask & (key_position$unsqueeze(1) >= query_position$unsqueeze(2) - self$sliding_window)
    }
    mask$unsqueeze(1)$unsqueeze(1)
  },
  forward = function(input_ids, positions, position_start, attn_in, value_embeddings, rotary, past = NULL) {
    shape <- attn_in$shape
    batch_size <- shape[[1]]
    query_length <- shape[[2]]

    projection_scale <- if (self$scales_folded) 1 else self$qkv_scale$to(dtype = attn_in$dtype)
    q <- nnf_linear(attn_in, self$q_proj$weight) * projection_scale
    k <- nnf_linear(attn_in, self$k_proj$weight) * projection_scale
    v <- nnf_linear(attn_in, self$v_proj$weight) * projection_scale
    q <- q$reshape(c(batch_size, query_length, self$num_heads, self$head_dim))
    k <- k$reshape(c(batch_size, query_length, self$num_kv_heads, self$head_dim))
    v <- v$reshape(c(batch_size, query_length, self$num_kv_heads, self$head_dim))

    if (self$uses_value_embeddings) {
      value <- value_embeddings(input_ids)$reshape(c(
        batch_size, query_length, self$value_stored_heads, self$value_dim
      ))
      value <- value[, , seq_len(self$num_kv_heads), ]
      gate_input <- attn_in[, , seq_len(self$value_gate_channels)]
      gate <- self$value_gate_scale * torch_sigmoid(nnf_linear(
        gate_input,
        self$ve_gate$to(dtype = attn_in$dtype)
      ))
      v <- v + gate$to(dtype = v$dtype)$unsqueeze(-1) * value
    }

    q <- limite_rms_norm(q)$transpose(2L, 3L)
    k <- limite_rms_norm(k)$transpose(2L, 3L)
    v <- v$transpose(2L, 3L)
    if (self$uses_rope) {
      rotated <- rotary(q, k, positions)
      q <- rotated[[1]]
      k <- rotated[[2]]
    }

    current_v <- v
    key_start <- 0L
    if (!is.null(past)) {
      key_start <- as.integer(past$start)
      k <- torch_cat(list(past$k, k), dim = 3L)
      v <- torch_cat(list(past$v, v), dim = 3L)
    }
    mask <- self$attention_mask(
      query_start = position_start,
      query_length = query_length,
      key_start = key_start,
      key_length = k$size(3L),
      device = q$device
    )

    y <- torch_scaled_dot_product_attention(
      q,
      self$expand_kv(k),
      self$expand_kv(v),
      attn_mask = mask,
      dropout_p = 0,
      is_causal = FALSE,
      scale = self$scale
    )
    y <- y$transpose(2L, 3L)

    if (self$uses_xsa) {
      normalized_v <- nnf_normalize(
        self$expand_kv(current_v)$transpose(2L, 3L)$to(dtype = torch_float32()),
        p = 2,
        dim = -1,
        eps = self$xsa_epsilon
      )
      projection <- (y$to(dtype = torch_float32()) * normalized_v)$sum(dim = -1, keepdim = TRUE)
      alpha <- torch_tanh(self$xsa_alpha$to(dtype = torch_float32()))$reshape(c(1L, 1L, self$num_heads, 1L))
      y <- y - (alpha * projection * normalized_v)$to(dtype = y$dtype)
    }

    if (self$attention_gate_channels > 0L) {
      gate_input <- attn_in[, , seq_len(self$attention_gate_channels)]
      gate <- self$attention_gate_scale * torch_sigmoid(nnf_linear(
        gate_input,
        self$attn_gate$to(dtype = attn_in$dtype)
      ))
      y <- y * gate$to(dtype = y$dtype)$unsqueeze(-1)
    }

    y <- y$reshape(c(batch_size, query_length, self$hidden_size))
    output_scale <- if (self$scales_folded) 1 else self$o_scale$to(dtype = y$dtype)
    output <- nnf_linear(y, self$o_proj$weight) * output_scale

    cache_length <- k$size(3L)
    cache_start <- key_start
    if (!self$is_global && cache_length > self$sliding_window + 1L) {
      retained <- self$sliding_window + 1L
      first <- cache_length - retained + 1L
      k <- k$narrow(dim = 3L, start = first, length = retained)
      v <- v$narrow(dim = 3L, start = first, length = retained)
      cache_start <- key_start + cache_length - retained
    }
    list(output = output, cache = list(k = k, v = v, start = cache_start))
  }
)

nn_limite_mlp <- nn_module(
  "nn_limite_mlp",
  initialize = function(config) {
    self$gate_proj <- limite_linear_no_bias(config$hidden_size, config$intermediate_size)
    self$up_proj <- limite_linear_no_bias(config$hidden_size, config$intermediate_size)
    self$down_proj <- limite_linear_no_bias(config$intermediate_size, config$hidden_size)
  },
  forward = function(x) {
    self$down_proj(nnf_silu(self$gate_proj(x)) * self$up_proj(x))
  }
)

nn_limite_layer <- nn_module(
  "nn_limite_layer",
  initialize = function(config, layer_index) {
    self$self_attn <- nn_limite_attention(config, layer_index)
    self$mlp <- nn_limite_mlp(config)
    self$resid_lambda_attn <- nn_parameter(torch_ones(1L)$squeeze())
    self$post_lambda_attn <- nn_parameter(torch_ones(1L)$squeeze())
    self$resid_lambda_mlp <- nn_parameter(torch_ones(1L)$squeeze())
    self$post_lambda_mlp <- nn_parameter(torch_ones(1L)$squeeze())
  },
  fold_projection_scales = function() {
    self$self_attn$fold_projection_scales()
    invisible(self)
  },
  forward = function(input_ids, positions, position_start, x, attn_in,
                     value_embeddings, rotary, residual_base = x, past = NULL) {
    attention <- self$self_attn(
      input_ids,
      positions,
      position_start,
      attn_in,
      value_embeddings,
      rotary,
      past
    )
    x <- self$resid_lambda_attn$to(dtype = x$dtype) * residual_base +
      self$post_lambda_attn$to(dtype = x$dtype) * attention$output
    mlp_output <- self$mlp(limite_rms_norm(x))
    output <- self$resid_lambda_mlp$to(dtype = x$dtype) * x +
      self$post_lambda_mlp$to(dtype = x$dtype) * mlp_output
    list(output = output, cache = attention$cache)
  }
)

nn_limite_body <- nn_module(
  "nn_limite_body",
  initialize = function(config) {
    self$config <- config
    self$embed_tokens <- nn_embedding(config$vocab_size, config$hidden_size)
    self$value_embeds <- nn_embedding(
      config$vocab_size,
      config$ve_stored_heads * config$ve_dim
    )
    self$rotary <- nn_limite_rotary(config)
    if (isTRUE(config$mudd)) {
      self$mudd <- nn_limite_mudd(config)
    }
    self$layers <- nn_module_list(lapply(
      0:(config$num_hidden_layers - 1L),
      function(index) nn_limite_layer(config, index)
    ))
    self$mudd_tap_idx <- config$mudd_tap_idx
    self$retained_history <- unique(unlist(config$mudd_tap_idx, use.names = FALSE))
    self$max_positions <- as.integer(config$max_position_embeddings)
    self$final_softcap <- as.numeric(config$final_softcap)
  },
  fold_projection_scales = function() {
    for (layer in as.list(self$layers)) {
      layer$fold_projection_scales()
    }
    invisible(self)
  },
  forward = function(input_ids, cache = NULL) {
    sequence_length <- input_ids$size(2L)
    position_start <- if (is.null(cache)) 0L else as.integer(cache$position)
    limite_abort_if(
      position_start + sequence_length > self$max_positions,
      "The input and cache exceed `max_position_embeddings`."
    )
    positions <- torch_arange(
      start = position_start,
      end = position_start + sequence_length - 1L,
      dtype = torch_int64(),
      device = input_ids$device
    )

    x <- limite_rms_norm(self$embed_tokens(input_ids))
    history <- list()
    if (length(self$mudd_tap_idx) > 0L) {
      history[["0"]] <- x
    }
    layer_cache <- vector("list", length(self$layers))

    for (index in 0:(length(self$layers) - 1L)) {
      key <- as.character(index)
      if (key %in% names(self$mudd_tap_idx)) {
        values <- lapply(as.character(self$mudd_tap_idx[[key]]), function(tap) history[[tap]])
        attn_in <- limite_rms_norm(self$mudd$combine(values, x, index))
        residual_base <- if (isTRUE(self$config$mudd_mlp)) {
          self$mudd$combine(values, x, index, residual_way = TRUE)
        } else {
          x
        }
      } else {
        attn_in <- limite_rms_norm(x)
        residual_base <- x
      }
      past <- if (is.null(cache)) NULL else cache$layers[[index + 1L]]
      result <- self$layers[[index + 1L]](
        input_ids,
        positions,
        position_start,
        x,
        attn_in,
        self$value_embeds,
        self$rotary,
        residual_base,
        past
      )
      x <- result$output
      layer_cache[[index + 1L]] <- result$cache
      history_index <- index + 1L
      if (history_index %in% self$retained_history) {
        history[[as.character(history_index)]] <- x
      }
    }

    if (self$final_softcap > 0) {
      x <- self$final_softcap * torch_tanh(x / self$final_softcap)
    }
    list(
      hidden = limite_rms_norm(x),
      cache = list(position = position_start + sequence_length, layers = layer_cache)
    )
  }
)

nn_limite_for_causal_lm <- nn_module(
  "nn_limite_for_causal_lm",
  initialize = function(config) {
    self$config <- config
    self$model <- nn_limite_body(config)
    self$softcap_a <- as.numeric(config$softcap_logits$a)
    self$softcap_b <- as.numeric(config$softcap_logits$b)
    self$softcap_c <- as.numeric(config$softcap_logits$c)
  },
  fold_projection_scales = function() {
    self$model$fold_projection_scales()
    invisible(self)
  },
  forward = function(input_ids, cache = NULL, use_cache = FALSE, logits_to_keep = NULL) {
    result <- self$model(input_ids, cache)
    hidden <- result$hidden
    if (!is.null(logits_to_keep)) {
      keep <- min(as.integer(logits_to_keep), hidden$size(2L))
      hidden <- hidden$narrow(
        dim = 2L,
        start = hidden$size(2L) - keep + 1L,
        length = keep
      )
    }
    raw <- nnf_linear(hidden, self$model$embed_tokens$weight$to(dtype = hidden$dtype))
    logits <- self$softcap_a * torch_sigmoid(
      (raw$to(dtype = torch_float32()) + self$softcap_b) / self$softcap_c
    )
    if (use_cache) {
      list(logits = logits, cache = result$cache)
    } else {
      logits
    }
  }
)

#' Construct a Limite language model
#'
#' @param config A configuration produced by [limite_config()] or loaded from a
#'   checkpoint.
#'
#' @return A `torch::nn_module` implementing `LimiteForCausalLM`.
#' @export
limite <- function(config = limite_config()) {
  nn_limite_for_causal_lm(validate_limite_config(config))
}
