# ADR 0118: Virtual-Memory-Reserved Growable KV Cache (CUDA VMM)

## Status

Proposed

Proposed by: Adam Gibson (session analysis of ovg-project/kvcached allocation patterns vs our KV cache manager, 2026)

## Context

### Current state (landed, working)

Our production decode path is STATIC: per-layer dense `BSHD [batch, maxKvLen, kvHeads, headDim]`
buffers allocated once at init (`GenerationPipeline` in-graph path), `setCloseable(false)`,
position advanced through the capture-safe `decodeCachePosition` INT64 device pointer
(`KvScatterDynEntry.kvPosPtr`). This satisfies the frozen-DSP-plan / CUDA-graph
pointer-stability contract exactly — but total KV capacity is claimed at init and
never returned. `KvContinuationMode.GROWABLE` throws
`UnsupportedOperationException` because growing "requires re-establishing frozen-plan
pointer stability and CUDA-graph capture/replay".

`KvCacheStrategy.PAGED` exists (`PagedKVCache`), with a fully preallocated block pool
sized `maxBatch × maxSeqLen/64 × poolSizeFactor`. The recently landed improvements
(native batched `paged_kv_append` routing, refcounted zero-copy prefix sharing,
slack-band free list, correct sliding-window eviction) make the pool semantics sound,
but the pool is still physically claimed at init in its entirety.

### The physical/virtual coupling problem

Every strategy we have couples **virtual address** (what kernels see, what graphs
bake in) to **physical residency** (what VRAM is committed). That coupling is the
root cause of three product limitations:

1. **Session memory ceiling**: a session configured for the model's full context
   ceiling holds that VRAM even when the conversation is 3% full.
2. **No multi-tenant coexistence**: a large KV cache cannot yield memory to a
   second process/model on the same GPU (the mechanism behind kvcached's
   reported 2–28× TTFT improvements on intermittent-peak serving).
3. **GROWABLE is unimplementable under reallocation**: growing a dense buffer by
   `cudaMalloc`-and-copy invalidates every captured pointer — precisely the
   contract our DSP replay infrastructure is built on.

### Prior art: kvcached (ovg-project/KVCacheD, Prism balloon driver)

kvcached decouples the two with CUDA virtual memory management APIs:

- `cuMemAddressReserve` the full VA span at init — zero physical bytes claimed.
- Whole range initially mapped to a shared **zero page**: kernels see a valid
  fully-zeroed tensor that costs nothing physically.
- `map(offset)`: lazily back a 2MB-aligned range with `cuMemCreate` +
  `cuMemMap` + `cuMemSetAccess`. `unmap(offset)`: `cuMemUnmap` + `cuMemRelease`
  returns physical memory to the driver immediately.
- Slack/reserved page list (min/max bounds, background refill) keeps the VMM
  calls off the allocation critical path.
- Capacity invariant: `free = min(virtual_free_pages, physical_free_VRAM)`.
- Fail-closed: VMM support is mandatory; no cudaMalloc fallback.
- Zero-page integrity: unmapped slots read as zeros, so a stale kernel touching
  a freed slot cannot read another request's data.

Production evidence: deployed on 10K+ GPUs, 2–28× TTFT reduction on
intermittent-peak workloads.

### Why this is uniquely compatible with our capture contract

The decisive observation: **kvcached never moves virtual addresses.** Graphs bake
in VA pointers; VMM mapping changes only the physical backing behind those
addresses. Under our contract:

- pointers in captured graphs stay valid for the lifetime of the session (same VA);
- physical pages may appear/disappear between replays without re-capture;
- `compute-sanitizer`-visible memory does not regress, because `cuMemUnmap`
  makes the range *inaccessible* rather than freed-and-reused.

The one hard constraint: **no VMM calls inside an active capture.** Mapping and
unmapping are host-side operations performed between replays (map during
append bookkeeping, unmap on sequence-free), never inside the captured region.
This matches our existing rule that allocation bookkeeping happens in Java-side
cache managers, not inside graph nodes.

### Design space considered

| Option | Growth | Pointer stability | Memory returned | Complexity |
|---|---|---|---|---|
| Status quo STATIC ceiling | no | yes | no | — |
| Realloc-and-copy dense | yes | **no** (breaks capture) | on free only | low |
| Host/disk offload tiers (TieredKVCacheManager) | effective | yes (stable GPU buffers) | no (moves, doesn't shrink VA) | medium; exists, unwired |
| **VMM reserve + lazy physical pages** | **yes** | **yes by construction** | **yes, to driver** | high |

The offloader tiers remain complementary: VMM handles *within-process* elasticity;
host/disk tiers handle *overflow beyond physical VRAM*. They compose.

## Decision

Adopt a CUDA-VMM-backed reserve-then-commit KV cache as a new `KvContinuationMode`
backing `GROWABLE`, implemented as a new strategy `KvCacheStrategy.VMM_PAGED`
alongside (not replacing) STATIC and PAGED:

1. **Reserve once, map lazily.** At session init, `cuMemAddressReserve` the
   full context-ceiling VA span per K/V per layer (or one fused arena), mapped
   to the zero page. Physical 2MB pages are committed on first append into a
   block and released when the last referencing sequence frees it (reusing the
   landed refcount machinery in `PagedKVCache`).

2. **Existing Java cache structure is preserved.** `PagedKVCache`'s page tables,
   refcounts, slack band, and native `paged_kv_append` routing are unchanged;
   only the *physical backing* of `keyBlockPool`/`valueBlockPool` becomes
   VMM-backed instead of a single eager allocation. `INDArray` views over the
   reserved span keep kernels and graphs operating on stable addresses.

3. **Capacity invariant.** Admission control uses
   `min(free_virtual_blocks, free_physical_blocks)` where physical availability
   comes from `cudaMemGetInfo` minus a configured utilization headroom. Pool
   exhaustion errors distinguish "VA exhausted" (misconfiguration) from
   "physical exhausted" (admit fewer sequences / trigger offload).

4. **Slack band maps to the slack band.** The landed
   `reservedBlocks` deque becomes the VMM slack pool: recently freed *physical*
   pages stay mapped (bounded by `reservedBlockLimit`, default 32) for instant
   re-commit, with a background (or opportunistic, between-replay) drain
   releasing overflow to the driver.

5. **Zero-page integrity for safety.** Unmapped-but-reserved slots read as
   zeros. This is a safety property (stale kernels read zeros, not other
   requests' data) and matches our existing "zeros beyond context length"
   attention-mask semantics.

6. **Fail-closed capability gate.** At session init, probe VMM support
   (`cuMemCreate` on the target device, driver ≥ 11.2). If unavailable, refuse
   `VMM_PAGED` with a clear error; do NOT silently fall back to eager
   allocation. STATIC/PAGED remain available for such systems.

7. **Capture discipline.** `cuMemMap`/`cuMemUnmap` calls are permitted only on
   the host between graph replays (same place `PagedKVCache` already does
   bookkeeping). Any proposal to map inside a captured region is rejected at
   review. The DSP plan cache sees no shape or pointer changes, so no plan
   invalidation occurs.

8. **Explicit spec.** Ship as an ADR-governed, feature-flagged path
   (`nd4j.kv.vmm.enabled=true` default-off initially) with validation against
   the DSP regression gate before default-on.

## Consequences

### Positive

- `GROWABLE` becomes real: sessions grow toward the context ceiling on demand;
  idle capacity returns to the driver.
- Multi-engine coexistence on one GPU becomes possible without process surgery
  (the mechanism behind kvcached's production wins).
- Pointer stability is preserved *by construction* rather than by discipline —
  the strongest possible answer to the frozen-plan contract.
- The landed PagedKVCache improvements (native append, refcount sharing, slack
  band, sliding-window eviction) are the exact substrate this rides on.

### Negative / risks

- First `cuMem*` usage in libnd4j (currently zero) — new memory-management
  layer, new failure modes (mapping churn, access-window bugs). Mitigation:
  fail-closed gate, feature flag, kvcached's design as proven reference.
- 2MB granularity: small caches waste up to ~2MB per K/V per layer tail.
  Acceptable at LLM cache scales; page size configurable via
  `KVCACHED_PAGE_SIZE_MB`-style env.
- Windows/ROCm parity: VMM APIs are CUDA-first. ZLUDA exposes a subset; ROCm
  has an equivalent (HMM/SVM) but it is out of scope for v1 — strategy simply
  refuses on those platforms (consistent with the fail-closed gate).
- Fragmentation across many map/unmap cycles: bounded by the slack band and by
  mapping whole compound blocks (page-aligned), mirroring kvcached's compound
  pages (2MB × layers × K/V in one mapping).
- Validation cost: the full DSP regression gate (concurrent plan sharing,
  lifecycle, replay) must run against a VMM-backed session before flag-on.

### Neutral

- `TieredKVCacheManager` integration becomes strictly more valuable: VMM
  handles elastic residency; tiers handle overflow. Wiring the offloaders into
  `GenerationPipeline` is a separate, unchanged decision.
- The `min(virtual, physical)` invariant plus an external memory-limit channel
  (kvcached's shm `MemInfoStruct` + `kvctl` pattern) enables future
  multi-tenant budget control; deferred until a consumer exists.

## Validation plan (before flag-on)

1. Unit: VMM arena reserve/commit/release on CUDA; zero-page read semantics;
   refcount-driven unmap; slack-band drain.
2. `DspConcurrentPlanSharingTest` + full DSP core batch against a VMM-backed
   KV session (the required regression gate).
3. Benchmark sweep per AGENTS.md: `run-benchmark.sh --tokens 250` across
   SLOT_BY_SLOT / OPTIMAL / TRITON / CUDA_GRAPHS, comparing lateSteady tok/s
   against the STATIC baseline; VMM overhead must be inside noise on the
   steady-state path (map/unmap only at block boundaries).
4. Multi-session coexistence smoke: two `GenerationPipeline` sessions sharing a
   GPU, confirming aggregate residency tracks live tokens, not configured
   ceilings.
