---
name: Qwen CUDA decode garbage status
description: Qwen first token correct (France/271) but second token produces garbage (76828) with GraphOptimizer enabled
type: project
---

## Qwen CUDA Decode Correctness Issue (May 4 2026)

**Status**: First token correct (token 271 = "France"), second token produces garbage (token 76828)
**Test**: TestQwen35Pipeline in platform-tests
**GraphOptimizer**: ENABLED (confirmed — NOT disabled as a workaround)

### Symptoms

- First decode step: correctly outputs token 271 ("France") 
- Second decode step: outputs token 76828 (garbage, expected continuation)
- The graph optimizer fuses RMSNorm+Linear patterns (confirmed in logs: "Fused RMSNorm+Linear pattern")
- The optimizer correctly refuses to remove matmul_186 whose output 'lm_logits' is a registered graph output

### Hypotheses Under Investigation

1. **GDN (Gated Delta Rule) .dup() device memory issue** — Qwen has 18 recurrent state layers (shape [1,16,128,128]). The `.dup()` operation may leave device memory unpopulated on second step, causing garbage inputs to the next layer.

2. **RMSNorm+Linear fusion accuracy** — the optimizer fuses decomposed RMSNorm into single op. If the fused op has different numerical behavior on second decode step (when KV cache is populated), it could produce wrong logits.

3. **KV cache state corruption** — between first and second decode step, KV scatter writes new K/V values. If the scatter writes to wrong positions or the attention mask is incorrect for step 2, logits are garbage.

### Key Files

- Test: `platform-tests/.../llm/TestQwen35Pipeline.java`
- Graph optimizer: `nd4j/.../optimize/GraphOptimizer.java`
- Attention fusion: `nd4j/.../optimize/optimizations/AttentionFusionOptimizations.java`
- OptimizationUtils: `nd4j/.../optimize/optimizations/OptimizationUtils.java`
- RMSNorm fusion log: "Fused RMSNorm+Linear pattern: x=layer_out_23, gamma=model.norm.weight, W=permute_186"

### Run Command

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  mvn test -Dtest=TestQwen35Pipeline -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/qwen-cuda.log
```

**Why:** Qwen correctness is a hard requirement alongside VLM performance.
**How to apply:** Fix must maintain GraphOptimizer enabled. NEVER disable optimizer as a workaround.
