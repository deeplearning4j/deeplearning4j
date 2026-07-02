# ADR 0103: DSP Execution-Mode Auction and Per-Unit Performance Ledger

## Status

Proposed

Proposed by: Adam Gibson (2 Jul 2026)

Scopes roadmap items **G3** (per-unit performance ledger — "do first, unblocks everything"),
**G1/G2** (replay-unit mode auction + verify-then-race gating). Ties in **H1** (Triton
tile-config PGO, ADR-to-come) and the split-KV/split-K flash-decoding kernel work as future
auction candidates.

## Context

DSP compiles a SameDiff segment into a `ReplaySchedule` of `ReplayScheduleUnit`s
(`NativeDynamicShapePlan.h:811-830`). Today a unit has exactly **two** kinds:

```
enum ReplayUnitKind { REPLAY_UNIT_TRITON_ISLAND, REPLAY_UNIT_GAP };
```

The choice between them is **static and binary**, made once at schedule-build time
(`NativeDynamicShapePlan_gpubackend.cu:942-963`, `emplace_back(REPLAY_UNIT_TRITON_ISLAND …)`
vs `emplace_back(REPLAY_UNIT_GAP …)`) purely from `OpTraitTable` Triton-mappability. Several
other selection decisions are likewise static guards, not measurements:

- Default Triton compile scope is `ELEMENTWISE + IDENTITY` only unless `tritonCompileAll`
  (`SectionTypeConfig.h:64-82`).
- Hand GEMV is routed by a hardcoded `M==1 && K<=8192` guard (`rms_norm.cu:619`), not by
  measuring GEMV-vs-cuBLAS for the actual shape.
- Island merging disables batched-GEMM groups with `<2` members (`batchgemm.cu:413-435`).
- cuBLAS is a fixed fallback in `MmulHelper`, never raced against a Triton matmul.

**Consequences of static selection:** a "gap" op that would be faster as a hand-kernel stays
slow; a Triton island that is slower than cuBLAS for a given shape still wins; a merge that
hurts is taken anyway. Nothing measures whether a chosen mode is actually winning, so
regressions in a lowering are invisible and good lowerings are never discovered.

The Part II thesis is *"the selection is the product"*: make execution policy **empirical**.
Measure each replay-unit's cost per candidate mode, select the winner per unit, and persist
the history so the decision improves across runs.

Field norms DSP diverges from (see the perf-gap audit field-mapping): TF-XLA auto-clustering
has a `min_cluster_size` cost gate; TensorRT keys plans by shape **range** (optimization
profiles) and picks tactics by measured latency; vLLM/PyTorch cudagraph-trees choose capture
granularity deliberately. DSP has none of this measurement-driven selection.

## Decision

Introduce a **per-unit performance ledger** (G3) as the measurement substrate, then a
**mode auction** (G1) gated by **verify-then-race** (G2) that consults the ledger to assign a
mode per unit. Build incrementally: first make the *existing binary* island-vs-gap decision
measured; then widen the candidate set.

### 1. Candidate execution modes (per unit)

Generalize `ReplayUnitKind` from a 2-value kind into a `(kind, mode)` where mode is drawn from
an extensible set:

- `SLOT_BY_SLOT` — interpreter (today's GAP); always the correctness reference.
- `TRITON_ISLAND` — captured Triton graph replay (today's island).
- `HAND_KERNEL` — cuBLAS / hand-CUDA for matmul-shaped units (GEMV, GEMM).
- `MERGED_CAPTURE` — grouped with capture-safe neighbors (today's `mergedGroupId`).
- *(future)* `TRITON_TILE_variant` (H1 tile-config PGO — a sub-auction over tile configs
  inside `TRITON_ISLAND`), `SPLIT_KV` flash-decode, CPU codegen (triton-cpu, WS-L).

`SLOT_BY_SLOT` is always a candidate and is the golden reference for verification.

### 2. Unit signature — the ledger key (G3)

A unit is keyed by a **stable signature** that survives process restarts and generalizes
across shapes that behave the same:

```
UnitSignature = hash(
    planKey / model id,          // which plan
    segmentId,                   // which segment
    opStructureHash,             // ops in [startSlot,endSlot] + dtypes + fusion structure
    shapeBucket )                // bucketed shape class, NOT exact dims (à la TRT profiles)
```

`opStructureHash` makes the key robust to slot renumbering; `shapeBucket` (e.g. power-of-two
seq-length buckets, batch=1 vs batch>1) lets one ledger entry cover many prompt lengths so the
auction does not re-explore every token. Mode is the secondary key.

### 3. The ledger (G3 — do this first; it unblocks everything)

```
Ledger : map<UnitSignature, map<Mode, CostStats>>
CostStats { runs, meanDeviceUs, minUs, p50Us, p95Us,
            verified: {UNVERIFIED, VERIFIED, FAILED}, lastUpdated, kernelVersion }
```

- **Measurement source:** extend the existing timing hooks (`executionTimingEnabled_`,
  `tIslandLaunchUs` / `tArgRefreshUs` / `tIslandDirtyUs`, `nIslandLaunches`,
  `gpubackend.cu:1518-1525`) from per-*category* accumulators to **per-unit device timing**:
  bracket each unit's launch with CUDA events (device time, not host wall-clock — host time is
  dominated by launch overhead and is noisy). Record into the ledger keyed by the unit
  signature + mode.
- **Persistence:** serialize the ledger to a file keyed by plan/model signature; load at plan
  compile, save at plan destroy / periodically. Cross-run history means the auction converges
  once, not every process start. Version each entry by `kernelVersion` so a changed kernel
  invalidates its stale costs.
- Everything below reads/writes this structure.

### 4. The auction (G1)

At schedule-build (and during a bounded warmup auction phase), for each unit consult the
ledger:

- **Exploit:** if a mode is `VERIFIED` and clearly cheapest (min beyond a noise margin) → select it.
- **Explore:** if a candidate mode is `UNVERIFIED` and within the warmup budget → try it,
  measure, record. `FAILED` modes are never retried (until `kernelVersion` changes).
- Explore/exploit policy: epsilon-greedy or UCB over the candidate modes, bounded by a per-plan
  warmup budget (reuse the existing `≥2 slot-by-slot warmups/segment` window as the initial
  exploration budget).

Output: a per-unit mode assignment that drives `ReplaySchedule` construction (which ranges
become islands, gaps, hand-kernels, or merge).

### 5. Verify-then-race gating (G2) — correctness before speed

This is the guardrail that makes the auction safe and is non-negotiable (the repo's hardest
rule is "no fast-but-wrong"):

- A mode is only **eligible to win** once it is `VERIFIED`: its output matches the
  `SLOT_BY_SLOT` golden for that unit within tolerance (reuse the `DSP_DIAG_VERIFY` /
  fingerprint infrastructure and the `run-validation.sh outputAccuracy` tolerances).
- **Race** = among *verified* modes, the measured-fastest wins.
- A mode that fails verification is recorded `FAILED` and excluded — it is never selected, so a
  faster-but-incorrect lowering can never regress output.
- Runtime deopt: if a selected mode fails at replay (handle not ready, kernel error), fall back
  to `SLOT_BY_SLOT` (the always-verified baseline) and record the failure. (This also finally
  gives the silent island-not-ready fallback — see F2, `DSP_DIAG(FALLBACK)` — a principled home:
  a fallback is a ledger event, not a hidden cliff.)

### 6. Integration points (where the code changes)

| Concern | Site today | Change |
|---|---|---|
| Schedule build (static split) | `gpubackend.cu:942-963` | Replace OpTraitTable-only binary split with `auction.assign(unit)` reading the ledger |
| Replay dispatch | `gpubackend.cu:2020-2114` (ISLAND vs GAP) | Extend to dispatch `HAND_KERNEL` / `MERGED` / future modes |
| Timing | `gpubackend.cu:1518-1525` counters | Per-unit CUDA-event device timing → ledger writes |
| Verification | `DSP_DIAG_VERIFY`, `fingerprintArray`, `run-validation.sh` | Per-unit golden compare feeding `verified` state |
| Persistence | *(new)* | Ledger serialize/deserialize at plan compile/destroy, keyed by plan signature |
| Plan cache | `computeShapeKey` / plan lifecycle | Ledger keyed by `UnitSignature`; invalidate on kernelVersion / shape-bucket change |

### 7. Rollout / safety

- Behind `nd4j.dsp.auction.enabled` (default **off**), like every other DSP feature flag.
- **Phase 1 (make the existing decision measured):** only two candidates —
  `TRITON_ISLAND` vs `SLOT_BY_SLOT` — so the auction reproduces today's islands unless
  measurement says a gap is faster. Zero new kernels; lowest risk; proves the ledger + verify
  + race loop end-to-end.
- **Phase 2:** add `HAND_KERNEL` (race Triton matmul vs cuBLAS for real shapes — kills the
  static `M==1 && K<=8192` guard) and `MERGED_CAPTURE` as an auctioned choice.
- **Phase 3:** tile-config sub-auction (H1) and split-KV as candidate modes.
- Verify-then-race guarantees no correctness regression at every phase.

## Consequences

**Positive**

- Execution policy becomes empirical: modes win by measured device time, not a static guess.
  Kernel regressions become visible (a mode's cost moving in the ledger); good lowerings are
  discovered instead of guarded out.
- Cross-run persistence → converge once, not per process start.
- Verify-then-race makes "no fast-but-wrong" a structural invariant, not a hope.
- A unifying home for the currently-scattered static guards (`SectionTypeConfig` scope, GEMV
  guard, merge threshold, cuBLAS fallback) and for future work (H1 tiles, split-KV, CPU codegen)
  — each becomes "just another candidate mode."

**Negative / costs**

- Warmup exploration cost (bounded by the warmup budget; amortized by persistence).
- Ledger keying + persistence + versioning complexity; risk of staleness (mitigated by
  `kernelVersion` invalidation + shape-bucketing).
- The verification pass adds warmup overhead (one golden `SLOT_BY_SLOT` run per unit per new
  mode) — acceptable because it is warmup-only and gated behind the flag.

**Dependencies / ordering**

- G3 (ledger) is a hard prerequisite for G1 (auction). G2 (verify-then-race) lands with G1.
- H1 (Triton tile PGO) is a sub-auction inside `TRITON_ISLAND` once the ledger exists.
- Split-KV / split-K flash-decoding becomes a candidate mode for attention units.

## Open questions

- **Signature stability under shape drift / recompile:** the shape-bucketing function is the
  crux — too coarse mis-selects, too fine never generalizes. Start with power-of-two seq buckets
  + batch=1 special-case and measure bucket hit-rate.
- **Warmup budget:** how many explore iterations before committing, relative to today's
  `≥2 warmups/segment`, without visibly slowing first-token latency.
- **Persistence format/location/versioning:** where the ledger lives, how a code change bumps
  `kernelVersion`, invalidation on model/plan change.
- **Interaction with plan-cache eviction and the frozen-shapes lifecycle** (a pinned/frozen
  plan's ledger entries must survive eviction of transient plans).
- **Multi-device:** the ledger is per-device (a mode's cost differs by GPU); key by device
  class or keep per-device sub-ledgers.
