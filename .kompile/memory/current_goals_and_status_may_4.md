---
name: Current goals and status May 4
description: "Two goals: VLM 100+ tok/s with mythic heroes output AND Qwen outputting France on CUDA"
type: project
---

## Active Goals (May 4 2026)

### Goal 1: VLM CUDA Performance
- **Metric**: run-benchmark.sh --tokens 250 on CUDA
- **Current**: ~60 tok/s with correct "mythic heroes" output (pre-CudaMemoryPool fix)
- **Target**: 100+ tok/s with correct output maintained
- **Root cause identified**: CUDA graph node count overhead (2742 nodes × 2-4µs = 5.5-11ms/replay)
- **Path forward**: Reduce graph node count by eliminating concat/reshape_no_copy kernels

### Goal 2: Qwen CUDA Correctness
- **Metric**: TestQwen35Pipeline outputs "France" (token 271)
- **Current**: First token correct, second token garbage (76828) with GraphOptimizer
- **GraphOptimizer**: Must remain ENABLED
- **Root cause**: rmsNormLinearFusedKernel assumes C-contiguous weight, gets F-order permuted LM head
- **Fix applied**: Contiguity check + dup in rms_norm.cu

### Latest Build (May 4 2026 10:20 JST)
- CudaMemoryPool conversion: ALL raw cudaMalloc/cudaFree → pool API
- Files converted: batchgemm, batchzero, cublas, cuda, gated_delta_rule, ExtraArguments, kv_scatter, qr, triangular_solve, CutlassGemmHelper, ggml_dequantize
- Also: cross-device failover (non-peer allowed), async stream copies with events, Java routing fix
- Build: SUCCESS in 6:31 (mostly ccache hits)

### Constraints
- Do NOT run builds and tests concurrently (RAM limit)
- Do NOT disable features as workarounds — fix root causes
- Always use -Dlibnd4j.triton=ON
- Benchmark with --tokens 250
- Commit and record milestones after each correctness+performance increase

### Run Commands
```bash
# VLM benchmark
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250

# Qwen test
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=TestQwen35Pipeline -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/qwen-cuda.log
```

**Why:** These are the two deliverables the user is tracking.
**How to apply:** Always validate both after any change. Performance without correctness is worthless.
