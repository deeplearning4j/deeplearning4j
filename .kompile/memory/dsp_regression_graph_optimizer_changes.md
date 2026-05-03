---
name: dsp-regression-graph-optimizer-changes
description: GraphOptimizer and fusion changes since 9bb2680e2b — DCE pass, NormalizationFusion, AttentionFusion, stripTrivialOps
type: project
---

## GraphOptimizer & Fusion Changes Since 9bb2680e2b (May 2 2026)

### GraphOptimizer.java — BOTH

**New DCE pass (COMMITTED — HIGH RISK):**
- File: `nd4j/.../optimize/GraphOptimizer.java`
- Dead code elimination removes ops not reachable from requiredOutputs
- RISK: KV cache update ops (scatter, concat) may be pruned if not in requiredOutputs
- If KV cache ops pruned → model loses context between tokens → catastrophic accuracy failure
- Also: output restoration after dup() — ensures outputs survive round-trip
- SKIP_OPTIMIZERS mechanism added for debugging

**Double-optimization risk (COMMITTED):**
- File: `GenerationPipeline.java`
- GraphOptimizer runs during pipeline construction
- If model was already optimized before pipeline, optimizations run twice
- DCE on already-DCE'd graph is idempotent, but fusion passes may double-fuse

### NormalizationFusionOptimizations.java — BOTH

**stripTrivialOps fix (COMMITTED — POSITIVE):**
- Was stripping through reshape ops → caused wrong-shaped fusion inputs
- Fixed: restricted to cast/identity only
- This was a REAL bug causing fusion to produce wrong shapes

**Output variable rename with output-guard (COMMITTED):**
- Prevents renaming variables that are graph outputs
- Positive defensive check

### AttentionFusionOptimizations.java — BOTH

**Permute absorption for rank-4 Q/K/V (COMMITTED — HIGH RISK):**
- Absorbs permute ops into attention fusion pattern
- If permute detection is wrong (wrong axes, wrong rank), attention input layout is wrong
- Wrong layout → Q@K^T computes wrong dot products → completely wrong attention

**K transpose absorption extended to Permute ops (COMMITTED):**
- Previously only absorbed Transpose ops for K
- Now also absorbs Permute ops
- If permute doesn't actually represent a transpose, K layout is wrong

### GenerationPipeline.java — BOTH

**DSP enable/disable conditional (COMMITTED):**
- DSP enabled/disabled based on `config.isDspEnabled()`
- If config defaults changed, DSP may be unexpectedly disabled

**Default logits name change (COMMITTED):**
- Changed "logits" → "lm_logits"
- If model doesn't have "lm_logits" output variable, logits lookup fails
- Different model architectures may use different names

**cachePosExtIdx resolution (COMMITTED):**
- Resolved and passed to AutoregressiveDecode
- If resolution is wrong, cache position tracking is wrong → KV cache writes to wrong position

**Why:** The optimizer transforms the graph before execution. Wrong transformations produce structurally different graphs that compute wrong results even if all ops individually work correctly.
**How to apply:** If accuracy regression appears ONLY after optimization, test with SKIP_OPTIMIZERS to isolate. The DCE pass and permute absorption are the highest-risk items.
