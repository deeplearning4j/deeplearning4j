# ADR: Scaled Dot-Product Attention Optimization

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Discussed with: Development Team

## Context

Scaled Dot-Product Attention (SDPA) is the core computation in transformer models:

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

This operation is executed for every attention head, in every layer, for every token during both encoding and decoding. For a model with 32 layers and 32 heads, SDPA accounts for 60-80% of total compute time.

LibND4J's original SDPA implementation used three separate operations: (1) matrix multiplication Q@K^T, (2) softmax, and (3) matrix multiplication attn@V. This approach has several inefficiencies:

**Memory Bandwidth**: The full attention score matrix `Q@K^T` (shape `[batch, heads, seq_q, seq_kv]`) is materialized in GPU memory, written out from GEMM, then read back for softmax, then written again, then read back for the second GEMM. This 4x read/write overhead dominates execution time for long sequences.

**No Kernel Fusion**: Each operation launches a separate CUDA kernel or MKL call, paying kernel launch overhead (~5μs each) and preventing the GPU/CPU from keeping data in registers/cache across operations.

**No Compiled Partition Caching**: OneDNN's graph API supports compiling fused operation graphs into optimized partitions, but the original implementation recompiled on every call even when shapes matched.

**Limited Backend Coverage**: SDPA was only available as unfused ops on CPU. No optimized paths existed for OneDNN (CPU), cuDNN (CUDA), or llama.cpp backends.

## Decision

We implement fused SDPA and Flash Attention operations across three backends (OneDNN, cuDNN, llama.cpp) with compiled partition caching and multi-dimensional layout support.

### Multi-Backend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    sd::ops::SDPA                             │
│                    sd::ops::flash_attention                   │
│                                                              │
│  ┌────────────────┐ ┌────────────────┐ ┌──────────────────┐ │
│  │ OneDNN (CPU)   │ │ cuDNN (GPU)    │ │ llama.cpp (CPU)  │ │
│  │                │ │                │ │                   │ │
│  │ Graph API      │ │ Flash Attention│ │ Reference impl    │ │
│  │ Fused SDPA     │ │ Kernel         │ │ for validation    │ │
│  │                │ │                │ │                   │ │
│  │ Compiled       │ │ Compiled       │ │ Direct execution  │ │
│  │ Partition Cache│ │ Plan Cache     │ │                   │ │
│  └────────────────┘ └────────────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### OneDNN Fused SDPA (CPU Backend)

The OneDNN implementation uses the Graph API to fuse Q@K^T → scale → mask → softmax → attn@V into a single compiled partition:

```cpp
// Build OneDNN graph
dnnl::graph::graph g(dnnl::engine::kind::cpu);
auto q_lt = g.create_logical_tensor(Q_ID, dt, q_dims, layout::strided);
auto k_lt = g.create_logical_tensor(K_ID, dt, k_dims, layout::strided);
auto v_lt = g.create_logical_tensor(V_ID, dt, v_dims, layout::strided);

auto matmul_qk = g.create_op(op_kind::MatMul, {q_lt, k_lt}, {qk_lt});
auto scale_op  = g.create_op(op_kind::Divide, {qk_lt, scale_lt}, {scaled_lt});
auto softmax   = g.create_op(op_kind::SoftMax, {scaled_lt}, {attn_lt});
auto matmul_av = g.create_op(op_kind::MatMul, {attn_lt, v_lt}, {out_lt});

// Compile and cache
auto partitions = g.get_partitions();
auto compiled = partitions[0].compile(inputs, outputs, engine);
```

**Compiled Partition Cache**: Shape-keyed cache avoids recompilation:

```cpp
struct SDPACache {
    struct Key {
        int batch, seqQ, seqKV, numHeads, headDim;
        int dtype;
        bool is4D;
    };
    std::unordered_map<Key, dnnl::compiled_partition> cache;
    std::mutex mtx;
};
```

Cache key includes batch size, sequence lengths, head dimensions, data type, and layout (3D vs 4D). In autoregressive decoding, the seqQ=1 case has a stable cache entry, and seqKV changes are handled by recompilation only when the key misses.

**Thread-Local Stream**: Each thread maintains its own OneDNN stream to avoid contention:

```cpp
static thread_local std::unique_ptr<dnnl::stream> tls_stream;
```

### Layout Support

Both 3D and 4D tensor layouts are supported:

- **3D**: `[batch, seq, dim]` — heads folded into dim. Used by some model exports.
- **4D**: `[batch, heads, seq, head_dim]` — standard multi-head layout.

The implementation detects layout from input rank and adjusts tensor descriptors accordingly. The `is4D` flag is part of the cache key to prevent cross-layout cache hits.

### Flash Attention

Flash Attention extends SDPA with block-wise computation to reduce memory I/O:

```
Standard SDPA:  O(n²) memory for attention matrix
Flash Attention: O(n) memory — processes in blocks of 256 tokens
```

**Causal Masking**: Built into the kernel — no separate mask tensor needed:

```cpp
struct FlashAttentionCache {
    struct Key {
        int batch, seqQ, seqKV, numHeads, headDim;
        int dtype;
        bool isCausal;  // Causal masking for autoregressive generation
    };
};
```

**Data Type Support**: FP32, FP16, BF16 supported across backends. FP16/BF16 significantly reduce memory bandwidth requirements.

### Direct Buffer Access

All backends operate directly on pre-allocated buffers — no intermediate copies:

```cpp
// OneDNN: Direct pointer binding
auto q_mem = dnnl::memory(q_md, engine, q->buffer());
auto k_mem = dnnl::memory(k_md, engine, k->buffer());
auto v_mem = dnnl::memory(v_md, engine, v->buffer());
compiled_partition.execute(stream, {q_mem, k_mem, v_mem}, {out_mem});
```

This eliminates the allocation and copy overhead that would otherwise negate the fusion benefit.

### Batch GEMM Integration

For MKL-based execution, batch GEMM is used for the multi-head parallelism:

```cpp
// All heads processed in a single batched GEMM call
cblas_sgemm_batch_strided(
    CblasRowMajor, CblasNoTrans, CblasTrans,
    seqQ, seqKV, headDim,
    1.0f / sqrt(headDim),
    Q, headDim, seqQ * headDim,  // stride between heads
    K, headDim, seqKV * headDim,
    0.0f,
    QK, seqKV, seqQ * seqKV,
    batch * numHeads  // number of batches
);
```

This eliminates the per-head loop and lets MKL optimize thread scheduling across all heads simultaneously.

## Consequences

### Advantages

**2-3x Attention Speedup**: Kernel fusion eliminates intermediate materialization and reduces kernel launch overhead. OneDNN graph compilation produces highly optimized code.

**O(n) Memory**: Flash Attention reduces attention memory from O(n²) to O(n), enabling longer sequence lengths without OOM.

**Compiled Partition Reuse**: Shape-keyed caching eliminates 50%+ of compilation overhead. In autoregressive decode (seqQ=1), the compiled partition is reused for every step.

**Multi-Backend**: Same API across OneDNN (CPU), cuDNN (GPU), and llama.cpp (reference), with automatic backend selection based on available hardware and helpers.

**Masking Stability on Padded Inputs**: Attention masking paths use `-FLT_MAX`-equivalent suppression instead of `-1e9`, and CPU/CUDA windowed-attention helpers apply the same thresholding strategy to prevent masked-score leakage through softmax under extreme padding.

### Disadvantages

**Compilation Latency**: First execution with a new shape triggers OneDNN graph compilation (~10-50ms). This is amortized over subsequent cache hits but affects first-token latency.

**OneDNN Dependency**: CPU SDPA optimization requires OneDNN (Intel MKL). ARM and other non-Intel CPUs fall back to the unfused implementation.

**Cache Memory**: Compiled partitions consume host memory proportional to the number of unique shape combinations. For autoregressive decode with growing KV cache, this is bounded by the maximum sequence length.

## References

- libnd4j/include/ops/declarable/platform/mkldnn/sdpa.cpp
- libnd4j/include/ops/declarable/platform/mkldnn/flash_attention.cpp
- libnd4j/include/ops/declarable/platform/cudnn/flash_attention.cu
- libnd4j/include/ops/declarable/platform/llamacpp/flash_attention.cpp
