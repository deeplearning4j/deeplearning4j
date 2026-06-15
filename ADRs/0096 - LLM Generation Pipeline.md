# ADR 0096 - LLM Generation Pipeline

## Status
Implemented

Proposed by: Adam Gibson (March 2026)

## Context

ADR 0064 covers the VLM-specific inference pipeline (multi-model loading, vision encoding, decode loop). However, general-purpose LLM generation infrastructure — chunked prefill for long contexts, speculative decoding for throughput, and a hierarchy of KV cache strategies — is a broader concern that applies to all autoregressive models, not just VLMs.

The `samediff-llm` module implements this infrastructure.

## Decision

### Chunked Prefill

`ChunkedPrefillEngine` splits long prompts into fixed-size windows to bound the O(n²) memory cost of full-context attention. Each chunk is processed sequentially, accumulating KV cache entries across chunks. Decode begins only after all chunks complete.

**Location**: `nd4j/samediff-llm/.../generation/ChunkedPrefillEngine.java`

### Speculative Decoding

`SpeculativeDecodeLoop` uses an n-gram pattern matcher (no separate draft model) to predict K tokens ahead, then verifies all predictions in a single forward pass:

1. `NgramSpeculator` scans generated token history for repeating patterns and predicts K tokens
2. Decoder runs one forward pass over K+1 positions (current + K speculated)
3. `TreeAttentionVerifier` compares predicted vs. actual, accepts the longest matching prefix
4. `SpeculativeKVCacheManager` handles rollback on misprediction

A probe mechanism detects models that cannot handle multi-token input (e.g., SmolDocling) and auto-disables speculation with cooldown/retry logic.

Throughput improvement: 2-5× for structured/repetitive outputs.

**Key files**: `SpeculativeDecodeLoop.java`, `NgramSpeculator.java`, `TreeAttentionVerifier.java`, `SpeculativeKVCacheManager.java`

### KV Cache Hierarchy

| Strategy | Mechanism |
|---|---|
| **Paged** | Block-allocated non-contiguous storage, avoids fragmentation |
| **Evictable** | LRU eviction of oldest entries under memory pressure |
| **Quantized** | INT8/FP8 compressed storage, dequantize on read |
| **MLA** | Multi-head Latent Attention compressed KV projections (DeepSeek) |
| **Radix Prefix** | Prefix-tree sharing of common prompt prefixes across requests |

### Serving Infrastructure

- **Continuous Batch Scheduler**: dynamic batching of concurrent generation requests
- **Beam Search KV Manager**: KV cache forking/merging for beam search
- **Tiered KV Management**: disk/host offload for entries exceeding GPU memory
- **Attention Sink Detection**: preserves sink tokens during cache eviction
- **Composite Samplers**: chainable strategies (temperature → top-k → top-p → repetition penalty)

### Native C++ Ops

The pipeline is backed by ~45 custom C++ ops in `libnd4j/include/ops/declarable/headers/llm.h`:

- **KV cache**: `kv_cache_update`, `kv_cache_quantize`, `kv_cache_dequantize`
- **SSM/Mamba**: `selective_scan`, `causal_conv1d`, `mamba2_ssm`, `gated_delta_rule`, `gated_delta_net_block` — enabling non-transformer architectures
- **Alternative attention**: `lightning_attention`, `linear_attention_decode`, `cascade_attention`, `sliding_window_attention`, `shared_kv_attention`
- **Position encoding**: `rope`, `fused_rope`, `fused_mrope`, `dual_rope`
- **Core pipeline**: `autoregressive_decode`, `vision_embedding_merge`
- **Tensor parallelism**: `column_parallel_linear`, `row_parallel_linear`

### Evaluation Benchmarks

Built-in harness for MMLU, HellaSwag, Winogrande, GSM8K, TruthfulQA, and ARC.

## Consequences

- Long prompts work within GPU memory limits via chunked prefill
- Speculative decoding improves throughput without requiring a separate draft model
- KV cache strategy can be selected per deployment (memory-constrained vs. throughput-optimized)
- Serving infrastructure supports production batch inference

## Related ADRs

- [0064](0064%20-%20VLM%20Inference%20Pipeline.md) — VLM pipeline (uses this generation infrastructure for its decode loop)
- [0061](0061%20-%20DynamicShapePlan%20Execution.md) — DSP execution engine powering the decode loop
