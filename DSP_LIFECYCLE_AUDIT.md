# DSP Plan Lifecycle — Factual Timeline and Root Cause

## SESSION VERDICT v2 (2026-09-23, post proc-068) — read this first

CAPTURE WALL ELIMINATED (proc-068): capability-stable placement sort
(DynamicShapePlan.assignDevices: resident device first, then by TOTAL memory,
never remaining-free budget) keeps assignments identical across generates.
All three placement traces: device0=all ops. No capture-memory failure.

REMAINING BLOCKER (precisely defined, one fix away):
Double-compile. execute() resolves requested mode from sd.getGraphExecutionMode()
(=AUTO), while -Dnd4j.dsp.graphExecutionMode=CUDA_GRAPHS forces recompile at the
mode check. Consequences at doc2:
  1. parked-prefill restore hits parkedMode(CUDA_GRAPHS) != requested(AUTO)
     -> honest rejection (correct behavior), full recompile.
  2. fresh compile's DataBuffer::migrate hits dev0's collapsed REMAINING budget
     (3479MB; dev0 holds ~16GB of live plans) -> "actual target exceeds device
     or DEVICE-group memory limits". Test failure.
FIX: initial auto-compile must resolve the EFFECTIVE mode (same SameDiff/property
path as execute()'s check), not raw AUTO. Plans then born under CUDA_GRAPHS:
doc2 restore matches bytes + mode -> adopts without recompile -> no migrations.
Secondary: even with the fix, restored plans may still want re-prefill migrations;
if migrate rejections persist, the budget used by DataBuffer::migrate must count
pinned/parked plans' live allocations (dev0 holds both plans by design).

CONFIRMED WORKING (proc-068 evidence):
- Capability-stable sort (this session) — placement identical across generates.
- parkedMode captured at park; restore validates vs requested mode, never relabels.
- Park/restore machinery; prefill-only parking.
- Bounded placement trace (assignDevices: budgets=... ) — keep permanently.

FALSIFIED / CORRECTED THIS SESSION (do not retry):
- CAPACITY_SHIFT_CAPTURE: CUDA 700 (warmup outputs on secondary). Reverted.
- Budget-descending placement sort: flips between generates (proc-064 trace).
- Java capture gate from estimateSlotOutputBytes(): inert pre-warmup (proc-053).
- Native rebuild without -Dlibnd4j.triton=ON: wrong binary (tritonAvailable=false).
- "Disk cache freezes assignments": FALSE — byte-identity gate rejects changed
  assignment bytes; the cache never overrode fresh placement.
- "Restore validated end-to-end": NOT YET — blocked by double-compile.

COMPILE ERRORS CAUGHT BY INSTALL-BEFORE-RUN (reminder 8 validated):
proc-062 break-outside-if; proc-065 nativeOps scope; proc-066 getDeviceTotalMemory
signature (int form: totalOps.getDeviceTotalMemory(deviceId)).


CONFIRMED FIX (in tree, code-correct, validated only to the point of "no regression"):
- `DynamicShapePlanExecutor` restore branch now re-derives mode/JIT/cudaGraphs/timing/trace
  + rebuilds constant protection on park-restore (kills the spurious "mode change
  detected → recompiling" that was destroying every restored plan since proc-041).
  NOT yet observed working end-to-end because doc1 decode dies first (below).

FALSIFIED EXPERIMENTS (evidence-backed, do not retry):
1. CAPACITY_SHIFT_CAPTURE (proc-055/057, REVERTED in proc-058): re-homing a capture
   segment to the plan primary + slot-by-slot re-exec produces CUDA 700 illegal
   memory access — warmup-era output arrays remain on the secondary device. A
   correct re-home requires migrating existing allocations first = the parked
   multi-GPU defect, not a local patch. proc-059 (reverted) returned to the clean
   capture-check failure, proving the 700s were caused by the experiment.
2. Proc-054 native build lacked -Dlibnd4j.triton=ON (tritonAvailable=false, proc-055
   initially misattributed the 700 to it). Build contract: iterative native builds
   MUST pass -Dlibnd4j.triton=ON (blasbuild preserves the Triton toolchain).
3. Java capture-capacity gate in assignDevices() (in tree, harmless): provably never
   fires — estimateSlotOutputBytes()=0 at placement time (shapes unknown pre-warmup).

THE REMAINING WALL (proc-051 == 053 == 059, byte-identical):
  CUDA graph capture memory check failed for seg[73-1872]:
  requiredFree=818MB, gpuFree=194MB, workingSet=4092MB, gpuTotal=7851MB (dev1)
Mechanism: the nested decode plan's assignDevices() byte-aware band split places a
slot band on the 8GB card (viable: ~7.3GB budget > 10% rule). Warmup allocates the
band; at capture (execCount>=2) free = budget - band - staging - weights-share =
194MB < 20% margin (818MB). Placement models band fit, NOT capture-time free memory.
Fix direction (NOT yet implemented): placement must reserve capture margin per
device: budget - bandBytes - staging/weights - 0.2*bandBytes >= 0, else the device
keeps no band. Pure Java change; no migration machinery needed.
Known separate inefficiency (pre-existing, all runs): fresh compiles run twice —
auto-compile calls compileNativePlan(AUTO), then execute()'s mode check recompiles
with the property-forced CUDA_GRAPHS.

## Original root-cause finding (unchanged, code-proven)


Session: proc-044 .. proc-049, 2026-09-23. Machine: RTX 4090 (24GB, dev used) + RTX 3070 Ti.
Model: Qwen3.5-2B runtime-quantized (`model-rq.sdz`, GGML Q4_K_M → RUNTIME_QUANTIZED_MATMUL + HALF).

## Observed events (per-run, from logs only)

| Run | Change tested | Observed |
|---|---|---|
| 044 | Park only prefill plans (`isPrefillPlan` gate) | doc1 43.6s OK. doc2: decode teardown called `freeNativePlanHandle("PLAN_CHANGED")` → tail `retainedFrozenPlans.clear()` erased the parked prefill (`retained=0`). doc2 prefill: `dot_pro` `allocateSpecial` 37,748,736B rejected, limit 20,461,457,408. |
| 045 | Independent-lease entries survive teardown | doc1 42.6s OK. doc2: parked prefill **survived** and `restored parked frozen plan 0x7f7e9e894b80 … independentLease=true` logged. doc2 prefill then FAILED same 37MB-over allocation. `releaseGpuIntermediates: freed 1611 arrays` logged AFTER the failure (cleanup). |
| 046 | Release intermediates at non-park switch (before unpin) | doc1 42.9s OK. doc2: release fired (1611 arrays) at switch, restore fired, prefill ran (12.7s), re-park fired. doc2 decode then FAILED: `DSP staging failed: stage=rebound_transfer ext=474 'past_key_values.23.value' … cudaError=1 (invalid argument)`. 55-array release logged AFTER failure (close/cleanup). |
| 047 | Gate release on incoming==prefill | Same shape as 046: boundary release fired once (1611), doc2 prefill OK, doc2 decode FAILED same `rebound_transfer`. |
| 048 | Revert boundary release entirely | doc1 43.4s OK. doc2: restore fired, prefill FAILED same 37MB-over allocation. NO staging errors anywhere. |
| 049 | 87% cap (20,944,379,904), release still reverted | doc1 42.8s OK. doc2: restore fired, "mode change detected (AUTO -> CUDA_GRAPHS), recompiling native plan" IMMEDIATELY after restore, prefill ran (12.7s), decode completed 1022 tok **no EOS, 117.8s**. doc3 prefill: capture-time check `requiredFree=7574MB gpuFree=860MB workingSet=37870MB gpuTotal=24084MB` → KERNEL_FAILURE. |

## Separating observation from claim

OBSERVED:
- The parked-plan restore fires and the Java handle identity/serialization matches every time since 041.
- "mode change detected … recompiling native plan" logs immediately after every restore (041, 045, 049).
- Any intermediate release at a generate boundary precedes the NEXT decode's `rebound_transfer` failure (046, 047).
- Without boundary releases, the next prefill hits an `allocateSpecial` rejection of 36MB (045, 048) at 85% and capture-time KERNEL_FAILURE at 87% (049).
- doc2 in 049 ran 1022 tokens with no EOS at 117.8s (only full-budget doc observed end-to-end).

CLAIMS that were NOT established (do not build on these):
- "37MB over cap" — 37,748,736B is the size of ONE rejected allocation, not the deficit.
- "decode staging was fixed in 048" — 048 never reached the later decode.
- "shared staging makes release infeasible" — the 046/047 release targeted the OUTGOING (decode) plan's intermediates; whether the failure is caused by sharing or by releasing a plan whose LEASE was still live has not been traced.
- "capture workingSet=37870MB is live allocation" — it is `estimatedCaptureBytes` = sum of slot array sizes for the segment (cudagraph.cu:930-953), i.e. a per-segment working-set estimate, not a live-RSS reading.
- "restore fully preserves the frozen plan" — contradicted by the immediate recompile log.

## ROOT CAUSE (code-proven, 2026-09-23)

`DynamicShapePlanExecutor.freeNativePlanHandle(...)` tail unconditionally resets:

    configuredGraphExecutionMode = GraphExecutionMode.AUTO;   // line ~6170
    cachedEffectiveGraphModeCode = -1;                        // line ~6176

`compileNativePlan()` restore branch returns EARLY (line ~2483) on restore success —
BEFORE the mode-application block (lines ~2664-2683) that would set
`configuredGraphExecutionMode = effectiveMode` and re-populate
`cachedEffectiveGraphModeCode` / `configuredHandleAddresses`.

Consequence after every restore:
- executor says mode=AUTO (stale from teardown)
- native handle is actually in CUDA_GRAPHS (mode is part of the cache key)
- `execute()` mode check (line ~3731) sees AUTO != CUDA_GRAPHS → `compileNativePlan(plan, AUTO, …)` → full recompile → destroys restored frozen state, re-warms, re-allocates → doc2 runs half-warm (12.7s prefill instead of ~2-3s), doc3 sees doubled working set at capture time.

The doc2 "no EOS, 117.8s" and doc3 "requiredFree 7574MB > gpuFree 860MB" are the
downstream symptoms of this recompile-on-restore: the recompile's slot-by-slot warmup
and fresh capture compete with the parked plan's retained working set for the same
24GB device.

## Secondary root cause (code-proven, same mechanism)

The restore branch restores `nativePlanHandle`, `cachedSortedOutputs`, `cachedPhKeys`,
`pinnedPlanHandles`, `configuredHandleAddresses` — but NOT:
- `cachedEffectiveGraphModeCode` (dispatch cache-key component, line ~3070!)
- `cachedJitModeInt`, `cachedCudaGraphsEnabled`, `cachedExecTiming`, `cachedTraceEnabled`
- `maxAllocationConfigured` (reset false by recompile; restore leaves it from before teardown=false — actually OK because teardown set it false, but the flag must be re-set true before execute, verify path)
- `protectedConstantBuffers` (weights protection! restore leaves null → next session cleanup could close constants the restored plan still references)

`cachedEffectiveGraphModeCode` is the WORST omission: it is used as the dispatch
cache-key mode component at line 3070. With -1 → `0` (AUTO code) while the parked
handle was created under CUDA_GRAPHS code → dispatch asks the cache for the WRONG
mode-identity → cache MISS/mismatch behavior undefined per cache policy.

## Fix (smallest coherent)

In the restore branch (before `return` at line ~2483), restore the cached settings
values that the parked plan was compiled under — they are per-plan constants for a
fixed build, so re-derive them the same way the compile path does (same code, or a
small helper both paths share):

    cachedEffectiveGraphModeCode = <effective mode for CUDA_GRAPHS>.getNativeCode();
    configuredGraphExecutionMode = <same>;
    configuredHandleAddresses.clear();   // force applySettingsIfNewHandle() to re-apply per-handle settings on next redispatch
    cachedJitModeInt / cachedCudaGraphsEnabled / cachedExecTiming / cachedTraceEnabled: re-derive (same as lines 2609-2629)
    protectedConstantBuffers: rebuild identically to lines 2647-2658

Do NOT change: park gate, lease acquisition/release, eviction, teardown tail reset
(the tail reset is correct for a REAL teardown; the bug is the restore not restoring).

## Validation gates (before corpus)

1. Unit: restore branch restores mode fields (direct executor test, no GPU).
2. Real-model 2-doc run (proc-049 config, 85% cap): expect log shows restore WITHOUT
   the following "mode change detected" line; doc2 prefill << 12.7s; both docs EOS.
3. 3+ cycles at 85%: no capture KERNEL_FAILURE, no rebound_transfer error.
4. Only then corpus + accuracy.

## Model-name coupling to remove (same session, low risk)

`isPrefillPlan()` (executor) hardcodes "lm_logits_last" — replace the generic-executor
gate with a policy hook (e.g. `PlanParkingPolicy` callback with a default that parks
any frozen plan, and the pipeline/model layer supplies the prefill-preferring
predicate) OR move the gate to GenerationPipeline where the decision belongs.
Prefer the latter (pipeline already knows which request is prefill).
