# ADR: VLM Inference Pipeline

## Status

Implemented

Proposed by: Adam Gibson (January 2025)

Updated by: Runtime maintainers (March 31, 2026)

Discussed with: Development Team

## Context

Vision-Language Models (VLMs) like SmolDocling combine a vision encoder (processing images) with a language decoder (generating text). These models present unique challenges for inference frameworks:

**Multi-Model Architecture**: VLMs consist of 3+ separate models (vision encoder, token embedder, language decoder) that must be loaded, managed, and executed independently but coordinated in a pipeline.

**Memory Pressure**: A typical VLM requires 5-8GB for model constants alone. The vision encoder is only needed during image processing but consumes significant GPU memory. Keeping it loaded during text generation wastes memory that the decoder needs for KV cache growth.

**Multi-Page Documents**: Document understanding tasks (OCR, layout analysis) require processing multiple pages. Each page must be vision-encoded independently, but the decoder generates text for all pages sequentially.

**Autoregressive Generation**: The decoder generates one token at a time in a loop, with the KV cache growing each step. This requires dynamic shape handling and efficient memory management across thousands of steps.

**Multi-GPU Opportunity**: Encoding and decoding use different models with different compute profiles. A smaller GPU can handle vision encoding while a larger GPU handles the memory-intensive decoder — but coordinating cross-device execution and data transfer adds complexity.

## Decision

We implement a pipelined VLM inference system with multi-GPU support, deferred model release, and integration with DynamicShapePlan for autoregressive generation.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    VLMPipelineExecutor                           │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ ImageTiler    │  │ VLMImage     │  │ MultiPartModelLoader  │ │
│  │ - Splits docs │  │ Preprocessor │  │ - vision_encoder.sdz  │ │
│  │   into tiles  │  │ - Resize     │  │ - embed_tokens.sdz    │ │
│  │ - Parallel    │  │ - Normalize  │  │ - decoder.sdz         │ │
│  │   encoding    │  │ - Patch      │  │ - Separate load/free  │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Multi-GPU Pipeline                           │  │
│  │                                                           │  │
│  │  GPU 0 (Large, e.g. RTX 4090 24GB):                      │  │
│  │    - Decoder model constants                              │  │
│  │    - Token embedding                                      │  │
│  │    - Autoregressive generation (KV cache)                 │  │
│  │                                                           │  │
│  │  GPU 1 (Smaller, e.g. RTX 3070 Ti 8GB):                  │  │
│  │    - Vision encoder model constants                       │  │
│  │    - Image tile encoding                                  │  │
│  │    - Released after all pages encoded                     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Decode Loop                                  │  │
│  │  for each token:                                          │  │
│  │    1. Embed current token (embed_tokens model)            │  │
│  │    2. Merge with vision features (first step only)        │  │
│  │    3. Execute decoder with KV cache (DynamicShapePlan)    │  │
│  │    4. ArgMax on logits → next token ID                    │  │
│  │    5. Check for EOS token                                 │  │
│  │    6. Release intermediates (liveness schedule)           │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### Multi-Part Model Loading

VLMs are loaded as separate SameDiff graphs from individual `.sdz` files:

```java
VisionLanguageModel vlm = MultiPartModelLoader.load(modelDirectory);
// Loads:
//   vision_encoder.sdz → SameDiff (assigned to encoder GPU)
//   embed_tokens.sdz   → SameDiff (assigned to decoder GPU)
//   decoder.sdz        → SameDiff (assigned to decoder GPU)
```

Each model is loaded onto its assigned GPU using `selectBestGpu()` for the decoder (largest GPU) and the next-best GPU for the encoder.

### Device-Affinity Execution

A single-thread executor ensures all encoder work stays on the encoder GPU:

```java
ExecutorService encoderExecutor = Executors.newSingleThreadExecutor(r -> {
    Thread t = new Thread(() -> {
        DeviceMemoryManager.switchDevice(encoderDeviceId);
        r.run();
    });
    t.setDaemon(true);
    return t;
});
```

This prevents device context switching overhead and ensures the encoder's CUDA streams, memory pools, and context buffers are isolated from the decoder's.

### Pipelined Batch Processing

For multi-page documents, encoding and preprocessing are pipelined:

```java
Future<INDArray> currentEncoding = encodeAsync(currentPage);
INDArray nextPagePreprocessed = preprocess(nextPage); // CPU work overlaps GPU encoding
INDArray encodedFeatures = currentEncoding.get();
// Transfer to decoder device
INDArray onDecoderGpu = CudaAffinityManager.replicateToDevice(decoderDeviceId, encodedFeatures);
```

Page N+1 preprocessing (CPU-bound: image loading, resizing, normalization) overlaps with page N encoding (GPU-bound), reducing total pipeline latency.

### Deferred Vision Encoder Release

After all pages are encoded, the vision encoder model is freed:

```java
void freeVisionEncoder() {
    visionEncoder.close();         // Free SameDiff graph
    SameDiffMemoryUtils.safeClose(visionEncoderArrays); // Free constant arrays
    // GPU memory freed: 5-8GB now available for decoder KV cache growth
}
```

This is critical for single-GPU setups where the encoder and decoder share memory. Even on multi-GPU setups, freeing the encoder's GPU makes it available for memory spillover via `allocateFailover`.

### Autoregressive Decode Loop

The token generation loop integrates with DynamicShapePlan:

1. **Token Embedding**: Current token ID → embedding vector via `embed_tokens` model
2. **Feature Merging**: On first step, concatenate vision features with token embeddings
3. **Decoder Execution**: Run decoder with DynamicShapePlan (handles growing KV cache). Each segment tracks its own `ExecutionPhase` (WARMUP -> COMPILING -> COMPILED -> REPLAYING). The selected `GraphExecutionMode` is a complete, non-cascading execution path — failure is a hard error, not a fallback to another mode.
4. **Token Selection**: ArgMax on output logits to select next token
5. **EOS Check**: Stop if end-of-sequence token generated
6. **Memory Release**: One persistent array per slot — arrays are reused across executions without close/reopen cycles

### Memory Management Integration

The VLM pipeline integrates with several memory management systems:

- **CudaMemoryPool**: All GPU allocations go through the async pool for fast reuse
- **ArrayCacheMemoryMgr**: Intermediate arrays cached for reuse across decode steps (growth factor must be 1.0 for standard op-by-op path)
- **DynamicShapePlan**: One persistent array per slot. Arrays are reused across executions without pendingClose/deferredClose cycles. The memory model is simple: allocate once, reuse forever (or until shape change invalidates the slot cache).
- **Workspace**: Native workspace for C++ temporary allocations (prevents heap corruption)
- **Explicit Close**: All intermediate arrays are explicitly closed — GC-based cleanup is broken (PhantomRef strong reference cycle)

## Consequences

### Advantages

**Multi-GPU Utilization**: Encoder and decoder run on separate GPUs simultaneously. The smaller GPU handles the one-time encoding while the larger GPU is dedicated to the memory-intensive decode loop.

**Memory Efficiency**: Deferred encoder release frees 5-8GB after encoding completes. DynamicShapePlan's one-array-per-slot memory model with persistent arrays keeps decode memory growth to ~1MB/step.

**Pipeline Parallelism**: Preprocessing and encoding overlap reduces multi-page document processing time. GPU utilization stays high during batch processing.

**End-to-End Pipeline**: Single API call (`vlm.generate(image, prompt)`) handles the entire flow from image preprocessing through token generation, abstracting the multi-model, multi-GPU complexity.

### Disadvantages

**Complexity**: The pipeline coordinates 3 models across 2 GPUs with async execution, cross-device transfers, deferred release, and workspace management. Debugging failures requires understanding all these subsystems.

**Non-P2P Overhead**: Cross-device transfers for non-P2P GPU pairs require host-staged copies (D2H + H2D), adding latency proportional to feature tensor size.

**Single-GPU Bottleneck**: On single-GPU systems, the encoder must complete and be freed before decode can use its memory. This serialization eliminates the pipeline parallelism benefit.

**Model Format Dependency**: VLMs must be pre-exported as separate `.sdz` files (vision_encoder, embed_tokens, decoder). No support for loading monolithic ONNX VLM exports directly.

## Performance Characteristics

- SmolDocling on RTX 4090 (24GB): ~87-92 tok/s steady-state decode (~11ms/step) with CUDA graph replay + Triton fusion + static KV cache
- Model constants baseline: ~5.3GB poolUsed after vision encoder freed
- Memory growth: ~1MB/step with all fixes applied
- 1000 tokens: ~1GB total decode memory
- Vision encoder: 1962 DSP ops per frame, ~150ms with native executor
- GPU execution is memory-bandwidth-bound (~8ms to load 5.3GB weights at ~650 GB/s on RTX 4090)

## References

- VLMPipelineExecutor.java, VisionLanguageModel.java in samediff-vlm module
- MultiPartModelLoader.java
- VLMImagePreprocessor.java, ImageTiler.java
- ADR 0061 - DynamicShapePlan Execution
- ADR 0060 - CUDA Async Memory Pool
- ADR 0065 - Multi-GPU Memory Management
