---
name: vlm-second-call-segdispatch-weightprotect
description: segDispatchWarmup demotion + weight protection RULED OUT for second-call zero logits
type: project
---

## segDispatchWarmup + Weight Protection — RULED OUT (2026-05-02)

### segDispatchWarmup (NativeDynamicShapePlan_gpubackend.cpp:265-380)
INVERTED from promote-to-FROZEN to demote-to-WARMUP. Demotion at lines 337-350 clears `state_` but NOT `cachedOutputShapes`. SAFE because:
- phaseWarmup handles initial warmup and bypasses segDispatchWarmup
- segDispatchWarmup only fires for segments already in REPLAYING state that detect shape changes
- On fresh plan, no segments are in REPLAYING state

### Weight DataBuffer Protection (SameDiff.java destroySession)
Three protection passes correctly identify and skip weight/constant buffers:
1. Pass 1: protectedWeightBuffers from variableMap (non-placeholder, non-constant)
2. Pass 2: protectedConstantBuffers from variableMap (constants)
3. Pass 3: force-close loop skips all protected buffers
Ordering: destroySession reads protectedConstantBuffers BEFORE DynamicShapePlanExecutor.close() nulls them (close() is called from session.closePooledResources() AFTER destroySession).

**Why:** Early suspects eliminated by parallel investigation agents.
**How to apply:** Don't re-investigate these paths. They are architecturally sound.
