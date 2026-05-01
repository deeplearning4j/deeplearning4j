---
name: dsp-ssm-op-miscompilation
description: DSP silently skips GDN/SSM ops (gated_delta_rule, causal_conv1d) during Triton IR emission — trait fallback makes them appear compilable but buildOpTable has no entry
type: project
---

# DSP SSM Op Miscompilation (fixed 2026-04-27)

## Root Cause
Three-tier Triton lookup asymmetry:
- `isTritonMappable()` uses trait fallback from OpTraitTable.cpp → returns true for gated_delta_rule (UNARY_ACT) and causal_conv1d (UNARY_EW)
- `analyzeSegment()` returns canCompile=true 
- `buildModule()` only checks buildOpTable() → op not found → SILENTLY SKIPPED
- Output buffers retain zeros/garbage → NaN propagates through network

## Fix
Added gated_delta_rule, gated_delta_net_block, causal_conv1d, selective_scan, mamba2_ssm to OpCategoryTable.h as UNSUPPORTED. getOpCategory() now returns UNSUPPORTED before the trait fallback is reached, so segments with these ops fall back to slot-by-slot native execution.

## Key Files
- `libnd4j/include/graph/gpu/OpCategoryTable.h` — added UNSUPPORTED entries
- `libnd4j/include/ops/impl/OpTraitTable.cpp` — traits are misleading (UNARY_ACT for a stateful SSM op) but harmless with OpCategoryTable fix
- `libnd4j/include/graph/gpu/TritonIRBuilder.cpp` — buildOpTable() is tier-1 lookup, buildModule() line 1276 silently skips missing ops
- `libnd4j/include/graph/gpu/TritonIRBuilder_module.cpp` — actual silent skip at buildModule

## Rule
Any new complex/stateful op MUST be added to OpCategoryTable.h as UNSUPPORTED unless a real Triton IR emitter exists. The trait fallback in OpTraitTable.cpp is too coarse to distinguish simple unary ops from complex stateful ones.
