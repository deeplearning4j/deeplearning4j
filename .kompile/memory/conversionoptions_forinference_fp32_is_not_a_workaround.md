---
name: ConversionOptions forInference FP32 is NOT a workaround
description: forInference() using FLOAT32 for GGML/Qwen is a legitimate precision choice, not a workaround for the FP16 autocast bug
type: project
---

## ConversionOptions.forInference() — NOT a Workaround

`ConversionOptions.forInference()` dequantizes GGML weights to FLOAT32 (not FLOAT16).

**Why this is correct (not a workaround):**
- GGML weights are stored in various quantization formats (Q4_0, Q5_1, etc.)
- Dequantizing to FLOAT32 preserves maximum precision from the quantized format
- The GraphOptimizer (used for ONNX/VLM models) separately handles FP16 pre-casting
- Qwen correctness is confirmed with FLOAT32 weights → no need to change
- FP16 pre-casting for GGML could be added as a PERFORMANCE optimization later

**The VLM path is separate:**
- VLM uses ONNX import → weights stored as FLOAT32 in the graph
- GraphOptimizer `--fp16` flag pre-casts large weights to HALF
- The mmulMxV GEMV bug was the issue (HALF weight × FLOAT32 activation), now fixed

**File:** `nd4j/nd4j-ggml/src/main/java/org/nd4j/ggml/convert/ConversionOptions.java:128-133`
