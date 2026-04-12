# Multi-Task Results: Fix frozen decode path issues

**Subtasks:** 2

---

## fix-frozen-kv-scatter-strides (qwen)

**Timed out** after 10 minutes.

---

## fix-frozen-causal-mask (qwen)

Subagent 'qwen' completed in 122.8s

## Summary
Now I have a complete picture. Here's my analysis:

---

## Investigation Findings: Causal Mask Not Updated Per Step in Frozen Decode Path

### Bug Identified: Stale Causal Mask in `reusableInputs` Cache

**Location:** `DecoderUtils.java:737-746` (causal mask path in `buildDecoderInputMap`)

```java
} else if (ioConfig.isCausalMask(inputName)) {
    if (usePadded) {
        // Padded mode: causal mask matches concat shape (maxKvLen + currentSeqLen)
        if (canReuse && reusableInputs.containsKey(inputName)) {
            decoderInputMap.put(inputName, reusableInputs.get(inputName));  // ← BUG: mask content NEVER updated
        } else {
            long totalSeqLen = maxKvLen + currentSeqLen;
            INDArray causalMask = buildCausalMask(currentSeqLen, totalSeqLen);
            decoderInputMap.put(inputName, causalMask);

... (6372 chars total) ...

```java
if (canReuse && reusableInputs.containsKey(inputName)) {
    INDArray causalMask = reusableInputs.get(inputName);
    causalMask.assign(0.0f);  // Unmask all
    for (int k = (int) cachePos; k < (int) maxKvLen; k++) {
        causalMask.putScalar(new long[]{0, 0, 0, k}, MASK_FILL);
    }
    decoderInputMap.put(inputName, causalMask);
}
```

**Full output (6372 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260412-203328.md`
Use the `read` tool to access the full result if needed.

---

**Summary:** 1/2 subtasks completed successfully.