# ADR 0104: Triton Tile-Config Profile-Guided Optimization (PGO)

## Status

Proposed

Proposed by: Adam Gibson (2 Jul 2026)

Refines **ADR 0103** (DSP execution-mode auction) for the *intra-kernel* tile-config dimension.
Scopes roadmap item **H1** ("self-contained, highest-ROI Triton win").

## Context

Every Triton kernel DSP emits is parameterized by a **tile config**:

- Matmul / GEMM: `blockM, blockN, blockK, numWarps, numStages`
  (`TritonIRBuilder_kernels.cpp:61-119`, `emitMatmulKernel`).
- Elementwise / normalization: `blockSize, numWarps, numStages`, chosen by
  `TritonIRBuilder::selectTileConfig` (`TritonIRBuilder_analysis.cpp:856-950`), which derives
  them **heuristically** from `getLaunchDims(...)` + op categories.
- Fused attention: `chooseFusedAttentionTileConfig(batchSize, numHeads, ...)`
  (`TritonIRBuilder_internal.h:242`, `TritonIRBuilder_sections.cpp:1071,2312`) — a rule table.

The optimal tile config is **hardware- and shape-dependent** — it trades occupancy vs
shared-memory vs register pressure vs tensor-core utilization, which no static heuristic can
match across GPUs and shapes. The build already exposes `--kernel-autotuning` (default **OFF**,
`buildnativeoperations.sh:1426`) and `--kernel-strategy fastest`, but **no PGO loop drives
them** — the flag toggles nothing measured. This is exactly the problem Triton's `@autotune`
solves: search the tile space, measure real latency, cache the winner per `(kernel, shape)`.

## Decision

Add a tile-config PGO loop as a **sub-auction inside the `TRITON_ISLAND` mode** (ADR 0103). For
each Triton kernel signature × shape-bucket, the candidate "modes" are **tile-config variants of
the same kernel**; ADR 0103's ledger + verify-then-race select the fastest *verified* config.

Because every candidate here is still `TRITON_ISLAND` (no cross-mode dispatch, no new replay
plumbing), this is the **lowest-risk first consumer of the 0103 ledger** — it proves the
ledger + verify + race machinery end-to-end before the full multi-mode auction (HAND_KERNEL,
MERGED, …) is wired.

### 1. Tile-config search space (per kernel class, occupancy-pruned)

- **GEMM/matmul:** `blockM ∈ {32,64,128,256}`, `blockN ∈ {32,64,128,256}`,
  `blockK ∈ {32,64,128}`, `numWarps ∈ {2,4,8}`, `numStages ∈ {2,3,4,5}`.
- **Gated / two-layer MLP:** analogous M/K/N tiles (share the matmul emitter).
- **Fused attention:** the `chooseFusedAttentionTileConfig` axes (seq-block M/N, numWarps);
  split-KV / split-K variants (ADR 0103 `SPLIT_KV`) enter as additional configs here.
- **Normalization:** `blockSize` (currently `min(paddedRowLen, 4096)`); the wide-row chunk-loop
  lowering (see R2 / `DSP_DIAG(COMPILE)` truncation warning) is itself a config variant for
  rows > 4096.

Configs are **pruned before compile** by feasibility (shared-mem > device limit, register
pressure, `blockK` not evenly dividing K) so the candidate set stays small.

### 2. Signature + ledger reuse (ADR 0103 G3)

- Key: `(kernelClass, shapeBucket, dtype, deviceClass)`; the **tile config is the "mode."**
- Reuse 0103's `map<UnitSignature, map<Mode, CostStats>>`, its persistence, and
  `kernelVersion` invalidation (a changed emitter invalidates its cached configs).
- Measurement: per-kernel **device timing** via CUDA events. Each config is compiled once
  (NVRTC/Triton, `--kernel-caching ON`) and cached, so re-selection is free.

### 3. PGO loop (search strategy)

- Warmup-time, bounded by a **compile budget** (K configs per kernel×bucket).
- **Seed = the current heuristic** (`selectTileConfig` / `chooseFusedAttentionTileConfig`
  output), so the worst case is today's behavior.
- Explore a pruned neighborhood via coordinate descent (or a small grid) around the seed; keep
  the fastest **verified** config.
- Cache compiled variants; persist the winner so subsequent runs skip the search.

### 4. Verify-then-race (ADR 0103 G2)

Each tile-config variant must match the `SLOT_BY_SLOT` golden within tolerance before it is
eligible — a config with a boundary bug (e.g. a missing load/store mask at `M < blockM`, the
exact R1 class) is recorded `FAILED` and **excluded**, never selected. Among verified configs,
fastest wins.

### 5. Integration points

| Site | Today | Change |
|---|---|---|
| `TritonIRBuilder::selectTileConfig` (`analysis.cpp:856`) | heuristic blockSize/warps/stages | consult ledger → verified winner, else return heuristic seed + mark for exploration |
| `chooseFusedAttentionTileConfig` (`internal.h:242`) | rule table | same ledger-first pattern |
| JIT / compile path | compile one config | compile + cache the pruned candidate set during warmup |
| Build flag `--kernel-autotuning` | toggles nothing measured | wire to enable the PGO loop; runtime flag `nd4j.dsp.triton.pgo.enabled` |

### 6. Rollout / safety

- Behind `nd4j.dsp.triton.pgo.enabled` (default **off**), gated by the 0103 auction flag.
- Seeded by the current heuristic → worst case equals today.
- Verify-then-race prevents a fast-but-wrong tile.
- Bounded compile budget so first-token latency is not blown; persistence → converge once.

## Consequences

**Positive**

- Tile configs become **measured-optimal** per shape/GPU — the occupancy/tensor-core wins a
  static heuristic leaves on the table — which is the highest-ROI self-contained Triton lever.
- Ships as the first, lowest-risk consumer of the ADR 0103 ledger (all-`TRITON_ISLAND`, no new
  dispatch), de-risking the larger auction.
- Turns the dormant `--kernel-autotuning` flag into a real, measured loop.

**Negative / costs**

- Warmup compile cost for the candidate set — bounded by the budget, mitigated by kernel
  caching + persistence, but the pruning model must be conservative or compile cost explodes.
- Adds a compile-and-time inner loop to warmup; only acceptable behind the flag and warmup-only.

**Dependencies**

- Reuses ADR 0103's ledger (G3) and verify-then-race (G2). Recommended as the **first**
  end-to-end exercise of that machinery.

## Open questions

- Search strategy (coordinate descent vs pruned grid vs a learned/regressed prior) and the
  per-kernel compile budget.
- Occupancy/feasibility pruning: static shared-mem/register model vs trial-compile-and-reject.
- Shared shape-bucketing with ADR 0103 (the tile optimum and the mode optimum want the same
  buckets — keep one bucketing function).
- Persisted-config portability across driver/toolkit versions (bump `kernelVersion` on toolkit
  change).
