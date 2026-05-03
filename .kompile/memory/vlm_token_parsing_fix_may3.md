---
name: vlm-token-parsing-fix-may3
description: "Fixed VLM native token output parsing: tid==0 zero-padding hack removed, buildStopTokenIds eosId<100 guard removed"
type: project
---

## VLM Token Parsing Fix (May 3 2026, 09:05 JST)

### WHAT WAS FIXED
1. **GenerationPipeline.java line 1826-1829**: VLM path used `nativeTokenIds.length()` (full maxNewTokens buffer) and broke on first `tid == 0`. Fixed to use `results[1]` (actual token count from native op), matching the correct GGUF path at lines 1052-1066.

2. **GenerationPipeline.java line 1147-1162**: `buildStopTokenIds()` discarded any eosTokenId in [0,99] including SmolDocling's valid `eosTokenId=2`. Removed the arbitrary guard entirely.

### WHY
- Token ID 0 is a valid vocabulary token (`<unk>`, `<pad>`, or real word piece in many tokenizers)
- The GGUF path already correctly uses `results[1]` for token count — VLM path was inconsistent
- The `< 100` guard was known-buggy: `generateNative` path at line 1373-1386 already bypassed it with a comment explaining it was wrong

### STILL UNKNOWN
- Whether the native C++ argmax genuinely produces token 0 on step 3 (possible HALF×FLOAT matmul garbage → argmax=0)
- Or whether there were more valid tokens after the first 0 that were being truncated by the Java hack

### NEXT STEP
- Rebuild Java (DONE) and run VLM benchmark to see if output changes
- If still producing token 0, the bug is in the native plan execution (HALF×FLOAT or stale logits buffer)
