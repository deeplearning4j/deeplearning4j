---
name: cpu-iargs-stop-token-parsing-fix
description: "CPU iArgs parsing bug: stopTokenCount=62 instead of 2 — kvOutputIndices count mismatch between Java (0 for GGUF) and C++ (assumed 2*numKvPairs)"
type: project
---

## Root Cause

In `autoregressive_decode.cpp` (op definition), line 244: `int nextIdx = kvStart + 4 * numKvPairs` assumes both kvInputExtIndices (2*numKvPairs) AND kvOutputIndices (2*numKvPairs) are packed. But the Java GGUF constructor (`AutoregressiveDecode.java` line 956) sets `kvOutputIndices = new int[0]` — zero entries packed.

This causes `stopTokenStartIdx` to be 12 positions too high (2*6 KV pairs = 12 missing kvOutput entries). The C++ parser then reads GDN/conv state indices as kvOutput, and interprets the actual stop token region as part of GDN/conv. The remaining 60+ iArgs all become stop tokens, including token 13 which matches the first generated token.

## Fix Applied

Changed `autoregressive_decode.cpp` to skip kvOutputIndices reading when `hasInGraphKvCache=true` (bit 4):
- Only read kvInput (2*numKvPairs) for GGUF path
- Read both kvInput + kvOutput (4*numKvPairs) for non-GGUF path
- `nextIdx = kvStart + 2*numKvPairs + (hasInGraphKvCache ? 0 : 2*numKvPairs)`

## Debug Evidence

CPU decode trace: `CPU_DECODE_STEP[0/18]: stopTokenCount=62` and `STOP matched token 13 == stopId 13` — loop generates only 1 token.

**Why:** Java GGUF path sends 0 kvOutputIndices (attention writes KV in-place), but C++ assumed 2*numKvPairs entries.
**How to apply:** This fix only affects CPU Qwen (GGUF path). SmolDocling uses non-GGUF (inGraphKv=false) with full kvOutputIndices — not affected by this bug.
