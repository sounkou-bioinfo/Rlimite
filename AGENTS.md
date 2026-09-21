# Rlimite development

Rlimite implements Limite directly with R torch. Runtime code must not invoke Python, vLLM, reticulate, or a model-serving process.

The serialized contracts in `paradigma-inc/limite-1b-base`, `paradigma-inc/limite-1b-base-soup`, and `paradigma-inc/limite-1b-violetto`, together with the executable architecture in `paradigma-inc/limite-violetto`, define model behavior. Violetto examples must use direct single-turn mathematical problems and the published message template. Preserve zero-based Hugging Face token IDs at the public API and convert to R torch's one-based embedding indices only at the model boundary.

Use tinytest for package tests. Keep unit tests independent of production checkpoints. The real-model workflow covers every pinned checkpoint, and `README.Rmd` must remain fully evaluated against those revisions.
