checkpoints <- limite_checkpoints()
expect_equal(checkpoints$checkpoint, c("Base", "Base Soup", "Violetto"))
expect_equal(checkpoints$profile[[3L]], "math-single-turn")
expect_true(all(nchar(checkpoints$revision) == 40L))

tiny_config <- function() {
  limite_config(
    vocab_size = 32L,
    hidden_size = 16L,
    intermediate_size = 32L,
    num_hidden_layers = 4L,
    num_attention_heads = 4L,
    num_key_value_heads = 2L,
    head_dim = 4L,
    max_position_embeddings = 128L,
    sliding_window = 2L,
    global_layers = 3L,
    rope_n_pairs = 1L,
    ve_dim = 4L,
    ve_stored_heads = 2L,
    ve_gate_channels = 4L,
    ve_layers = 1L,
    attn_gate_channels = 4L,
    xsa_layers = 0:3,
    mudd_layers = c(2L, 3L),
    mudd_tap_idx = list(`2` = c(0L, 2L), `3` = c(0L, 3L)),
    mudd_taps = 2L,
    mudd_inter = 4L,
    bos_token_id = 0L,
    eos_token_id = 31L,
    pad_token_id = 0L
  )
}

expect_error(
  limite_config(hidden_size = 1279L),
  pattern = "hidden_size.*num_attention_heads"
)
expect_error(
  limite_config(rope_per_layer = TRUE),
  pattern = "Per-layer RoPE"
)

config <- tiny_config()
torch::torch_manual_seed(1L)
model <- limite(config)
state_names <- names(model$state_dict())
expect_true("model.embed_tokens.weight" %in% state_names)
expect_true("model.value_embeds.weight" %in% state_names)
expect_true("model.layers.1.self_attn.ve_gate" %in% state_names)
expect_false("model.layers.0.self_attn.ve_gate" %in% state_names)

torch::with_device(device = "meta", {
  production_model <- limite()
})
expect_equal(length(production_model$state_dict()), 743L)
expect_equal(
  production_model$state_dict()[["model.embed_tokens.weight"]]$shape,
  c(151680, 1280)
)

input <- torch::torch_tensor(matrix(1:5, nrow = 1L), dtype = torch::torch_int64())
torch::with_no_grad({
  full <- model(input)
  expect_equal(full$shape, c(1, 5, 32))

  prefix <- model(input[, 1:3], use_cache = TRUE)
  cached <- model(input[, 4:5], cache = prefix$cache, use_cache = TRUE)
  expected <- full$narrow(dim = 2L, start = 4L, length = 2L)
  expect_true(torch::torch_allclose(cached$logits, expected, rtol = 1e-5, atol = 1e-5))
  expect_equal(cached$cache$layers[[1]]$k$size(3L), 3)
  expect_equal(cached$cache$layers[[1]]$start, 2L)
  expect_equal(cached$cache$layers[[4]]$k$size(3L), 5)
})

generated <- limite_generate(
  model,
  input_ids = 0:2,
  max_new_tokens = 3L,
  temperature = 0,
  eos_token_id = NULL
)
expect_equal(length(generated), 3L)
expect_true(all(generated >= 0L & generated < config$vocab_size))

sampled <- limite_generate(
  model,
  input_ids = 0:2,
  max_new_tokens = 2L,
  temperature = 1,
  top_k = 5L,
  eos_token_id = NULL,
  seed = 1L
)
expect_equal(length(sampled), 2L)
expect_true(all(sampled >= 0L & sampled < config$vocab_size))

prompt <- limite_chat_prompt(list(
  list(role = "system", content = "Be concise."),
  list(role = "user", content = "Hello")
))
expect_identical(
  prompt,
  paste0(
    "<|endoftext|><|im_start|>system\nBe concise.<|im_end|>\n",
    "<|im_start|>user\nHello<|im_end|>\n",
    "<|im_start|>assistant\n"
  )
)

expect_identical(
  limite_chat_prompt(list(list(role = "user", content = "Solve 2 + 2."))),
  paste0(
    "<|endoftext|><|im_start|>system\n",
    "You are a helpful assistant.<|im_end|>\n",
    "<|im_start|>user\nSolve 2 + 2.<|im_end|>\n",
    "<|im_start|>assistant\n"
  )
)
