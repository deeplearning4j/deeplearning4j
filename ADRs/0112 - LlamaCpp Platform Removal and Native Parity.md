# ADR: LlamaCpp Platform Removal and Native Parity

## Status
Accepted

## Date
2026-07-13

## Context

`ops/declarable/platform/llamacpp/` (59 files, ~17.3k LOC) was an optional
`ENGINE_CPU` platform helper that delegated ~119 LLM/GGML op names to
`ggml_*` calls behind the `HAVE_LLAMACPP` build gate and a `llamacpp` jar
classifier (ADR-0091). It carried real cost and risk:

- It was never built by any CI workflow — the `ggml` test suite ran against a
  no-helper binary, so the directory was effectively untested.
- Roughly a third of its op names had **no** generic fallback: they resolved
  only when `HAVE_LLAMACPP` was compiled in. `moe_gate` was one of these and
  was failing in production (`Could not find descriptor for op: moe_gate`).
- Several of its implementations were nonstandard or buggy (e.g.
  `load_balance_loss` reduced over the wrong axis and dropped the `numExperts`
  factor; `sinusoidal_position_encoding` used RoPE; `moe_expert_ffn` was a
  placeholder matmul), so treating it as a faithful reference was unsound.

## Decision

Remove `platform/llamacpp` entirely, but only after every op it was the sole
provider of gained a native implementation on both CPU and CUDA.

The 36 candidate op names resolved to: 5 already-native false positives;
10 pure aliases (`DECLARE_SYN` / thin adapters to existing native ops such as
`gqa_attention`→`grouped_query_attention`, `get_rows`→`gather`,
`paged_attention`→`paged_attention_forward`); 7 compositions of existing
primitives (`swiglu`/`geglu`/`reglu`, `win_part`/`win_unpart`,
`timestep_embedding`, `sinusoidal_position_encoding`); and 14 genuinely new
native implementations — `moe_gate`, `group_norm`, `l2_normalize`,
`load_balance_loss`, `sparse_mul_mat`, `embedding_lookup_bp`,
`kv_cache_attention`, `ssm_conv`, `ssm_scan`, `quantize_q4_0`, `quantize_q8_0`,
and the recurrence kernels `rwkv_wkv6`, `rwkv_wkv7`, `gated_linear_attn`.

Where the llamacpp op was nonstandard, the native op follows the correct/
standard definition (documented per-op) rather than replicating the bug.

## Consequences

- Op-name coverage is preserved: all former llamacpp op names resolve
  natively, verified by 38 CPU + 38 CUDA regression tests.
- The `llamacpp` jar classifier, `HELPERS_llamacpp`/`HAVE_LLAMACPP` build
  gating, `setup_llamacpp()`, `ENGINE_LLAMACPP`, and the Java
  `KernelManager.LLAMA_CPP` / `KernelSelectionConfig.LLAMACPP` enum members are
  removed. ADR-0091's llamacpp scope is superseded; its OneDNN/cuDNN classifier
  scheme is unaffected.
- The new quantize ops are the exact inverse of `ggml_dequantize` (round-trip
  verified). The RWKV/GLA recurrence kernels implement the standard published
  recurrences and are validated against reference implementations, not against
  a llamacpp golden (which no longer exists).
- Generated JavaCPP bindings (`Nd4jCpu`/`Nd4jCuda`/`Nd4jVulkan`) regenerate
  from `Engine.h` without `ENGINE_LLAMACPP`.

## Notes

`GgmlQMatMul` / `ggml_dequantize` (the runtime-quantized inference path in
`nd4j-ggml`) are native ops unrelated to the removed platform helper and are
unaffected.
