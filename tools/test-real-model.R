library(Rlimite)
library(torch)

model_id <- Sys.getenv("RLIMITE_MODEL_ID")
revision <- Sys.getenv("RLIMITE_MODEL_REVISION")
if (!nzchar(model_id) || !nzchar(revision)) {
  stop("RLIMITE_MODEL_ID and RLIMITE_MODEL_REVISION must be set", call. = FALSE)
}
checkpoint <- limite_checkpoints()
row <- match(model_id, checkpoint$model_id)
stopifnot(!is.na(row), identical(revision, checkpoint$revision[[row]]))

model <- limite_from_pretrained(model_id, revision = revision, device = "cpu")
tokenizer <- limite_tokenizer(model_id, revision = revision)
problem <- "Compute 17 * 23. Show your reasoning."
prompt <- limite_chat_prompt(list(list(role = "user", content = problem)))
input_ids <- tokenizer$encode(prompt, add_special_tokens = FALSE)$ids
logits <- with_no_grad({
  input <- torch_tensor(
    matrix(as.integer(input_ids) + 1L, nrow = 1L),
    dtype = torch_int64()
  )
  model(input, logits_to_keep = 1L)
})
stopifnot(
  identical(as.integer(logits$shape), c(1L, 1L, model$config$vocab_size)),
  isTRUE(torch_isfinite(logits)$all()$item())
)

generated <- limite_generate(
  model,
  input_ids,
  max_new_tokens = 8L,
  temperature = 0,
  eos_token_id = NULL
)
stopifnot(
  length(generated) == 8L,
  all(generated >= 0L),
  all(generated < model$config$vocab_size)
)
cat(model_id, tokenizer$decode(generated), sep = ": ")
cat("\n")
