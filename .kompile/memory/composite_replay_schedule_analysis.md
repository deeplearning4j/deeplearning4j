---
name: composite-replay-schedule-analysis
description: Detailed analysis of composite replay schedule structure — 305 units, 93 merged groups, 93 unmerged gaps, gap splitting opportunity blocked by view ops
type: project
---

## Composite Replay Schedule Analysis (04-27)

**Model**: SmolDocling-256M decode segment [0-2742]

### Current Schedule Structure
- **305 total replay units** (islands + gaps)
- **93 merged groups** — capture-safe gaps successfully merged with adjacent islands into CUDA graphs
- **93 unmerged gaps** — contain view/identity/frozen-constant ops, dispatched slot-by-slot
- **~152 islands** — Triton-compiled, replayed via cudaGraphLaunch
- Per decode step: 93 cudaGraphLaunch calls (merged groups) + 93 gap dispatches

### Gap Classification Stats
- **272 capture-safe gaps** (isCaptureSafe=1) — ALL slots launch CUDA kernels, can be merged
- **238 non-capture-safe gaps** (isCaptureSafe=0) — contain at least 1 view/identity/frozen-constant op

### Key Pattern: View Ops Poison Entire Gaps
Example gap [1205-1209]: `gather, multiply, cast, reduce_sum, expand_dims`
- Slots 1205-1208 are compute ops (gather, multiply, cast, reduce_sum) → capture-safe
- Slot 1209 is expand_dims (isViewCapableOp=true) → poisons entire gap as non-capture-safe
- Result: 4 compute ops that COULD be merged are forced into slot-by-slot dispatch

This pattern repeats ~90 times across the decode segment. Each non-capture-safe gap typically has 5-15 slots with only 1-2 view ops blocking the rest.

### Active Gap Slot Cache
- **456 active gap slots** out of 460 total in gaps (0.9% skip rate)
- Earlier claim of "82 active / 97% skip" was for a different segment configuration
- Virtually no gap slots are skipped — the cache provides minimal benefit at current config

### Previous Related Work (Commit History)
1. `3c6ed79824` — Composite replay + frozen fast path (6.8→23.8 tok/s). Created the island/gap schedule.
2. `bc8695b7cc` — Skip frozen constants/identity in gap loop. CPU-side overhead only.
3. `158f30a383` — Active gap slot cache (34.5→34.7, noise). Confirmed loop overhead not bottleneck.
4. `056cb35b34` — Reclassify gap isCaptureSafe at capture time (47→51 tok/s, +8%). Fixed stale flags.
5. `3075ec44ac` — Batched GEMM + plan cache (10→17.6 tok/s). Grouped matmul gaps for batch dispatch.
6. **mergedCaptureThroughViews** (04-26) — Reverted, -5.4%. Including view ops in CUDA graph capture adds overhead (empty graph nodes for zero-kernel ops).
7. **reconcileSlotDispatchAfterMerge** — Existing function in _batchgemm.cu. When merged capture swallows matmul slots, removes them from batched GEMM groups to avoid double dispatch.

### Proposed: Gap Splitting at View Boundaries
Instead of one gap `[gather, multiply, cast, reduce_sum, expand_dims]` classified as non-capture-safe:
- Sub-gap A: `[gather, multiply, cast, reduce_sum]` → capture-safe → merge with adjacent island
- Sub-gap B: `[expand_dims]` → 1-slot view → trivial tickWriteDevice

**Key difference from mergedCaptureThroughViews**: That approach included view ops IN the CUDA graph (adding overhead). Gap splitting EXCLUDES view ops from graphs while FREEING the surrounding compute ops to be captured.

### Risks
- More units means more loop iterations (but the loop is not the bottleneck per bc8695b7cc)
- reconcileSlotDispatchAfterMerge needs to handle the new split structure
- batched GEMM groups may lose members if their matmul slots get absorbed into merged captures
- Must verify accuracy after changing what gets captured vs dispatched live

### Why/How to Apply
The 93 unmerged gaps account for most of the gap dispatch overhead. If gap splitting moves 60-80% of compute ops from unmerged gaps into merged CUDA graphs, we reduce individual cuBLAS and kernel dispatch calls significantly. Each kernel call eliminated saves ~5-20µs of driver overhead. 200+ ops moved to merged capture ≈ 1-4ms/step savings.

**However**: The reshape_no_copy bypass trial showed that changing what executes as live ops vs captured graphs can cause downstream perf regressions (non-contiguous views killing cuBLAS). Must benchmark carefully.
