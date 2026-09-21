# Rlimite 0.1.0

- Implements the Limite decoder architecture directly with R torch.
- Loads the published Base, Base Soup, and Violetto safetensors checkpoints and tokenizers from Hugging Face.
- Provides mixed global and sliding-window grouped-query attention, partial RoPE, value embeddings, XSA, MUDD history mixing, learned residual coefficients, and tied sigmoid-softcapped logits.
- Provides cached token generation and the official single-turn message format used by Violetto for mathematical reasoning.
