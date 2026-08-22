# DSP Execution Performance Roadmap — Implementation Handoff

Date: 2026-07-02 · Branch context: `ag_new_release_updates_2` · Author: perf audit session (Claude)
Status: HANDOFF — not an ADR. An ADR pass happens after we know what works.

This document is self-contained: it briefs an implementing agent that has no prior
context. It distills (a) a full file:line audit of DSP execution on CPU and GPU and
(b) a literature review of peer frameworks (torch.compile/cudagraph-trees, JAX/XLA,
LazyTensor, TF-XLA auto-clustering, TensorRT(-LLM), vLLM V1 piecewise CUDA graphs,
llama.cpp cudaGraphExecUpdate, ONNX Runtime) into concrete, implementable workstreams.

---

## 0. CRITICAL RULES — READ BEFORE DOING ANYTHING

1. NEVER run: `git checkout <file>`, `git stash`, `git reset --hard`, `git clean`
2. NEVER run: `ccache -C` or `ccache --clear` (destroys 2+ hours of compiled cache)
3. NEVER run `make` directly — always the full `mvn` build with libnd4j + bindings module
4. NEVER pipe build or test output through `tail` — always `tee` and read the log file
5. NEVER use `LD_PRELOAD=libjemalloc.so`
6. NEVER run `mvn test` from the repo root — ALL tests run from `platform-tests/`
7. NEVER `export VAR=val` before `mvn test` — Surefire forks; wire env via `-D` props / pom
8. Fix root causes — NO workarounds, NO disabling features. Specifically:
   - NEVER force `GraphExecutionMode.SLOT_BY_SLOT` to dodge a bug
   - NEVER call `setDspAutoCompileEnabled(false)` or clear the plan cache mid-inference
   - NEVER route decode through `StaticKvCacheDecodeLoop`
9. Avoid editing headers when a `.cpp`/`.cu` change suffices (header edits cascade
   ccache invalidation). New config options go in BOTH `libnd4j/include/system/Environment.h/.cpp`
   AND `org/nd4j/linalg/factory/Environment.java` + `ND4JSystemProperties.java` + preset wiring.
10. If you undo your own edits, edit the lines back — never via git.

Build (CUDA):
```
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda \
  -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log \
  -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee cuda-build-output.log
```
Build (CPU):
```
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcpu -Dlibnd4j.triton=ON -Dlibnd4j.buildthreads=12 \
  -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cpu-backend-common,:nd4j-native \
  clean install -DskipTests 2>&1 | tee cpu-build-output.log
```
(Add `:nd4j-api` to `-pl` when touching nd4j-api Java.)

---

## 1. Context and terminology

DSP (Dynamic Shape Plan) compiles a SameDiff graph into a native plan of **segments**.
Within a segment, contiguous Triton-mappable slot runs are **islands** (each captured
as a CUDA graph, `REPLAY_UNIT_TRITON_ISLAND`); the slots between them are **gaps**
(`REPLAY_UNIT_GAP`, executed live every step). **Island merging** records islands AND
their intervening gaps into a single merged CUDA graph (`mergedGroupId >= 0`) — the
"megagraph" path. Lifecycle: slot-by-slot warmup → freezeShapes → pointer stability →
capture → REPLAYING. Plans live in a shape-keyed native cache (`NativePlanCache`).

Background reading in-repo: ADR 0061 (DynamicShapePlan Execution), ADR 0089 (CUDA
Graph Capture/Replay), ADR 0097 (Decode Path Perf), ADR 0094 (Buffer Coloring/
Passivation), ADR 0085 (MLIR JIT backend), ADR 0098 (OpenVINO CPU backend).

**Already being patched — do NOT duplicate, but VERIFY current state before building
on these code regions** (they were audit findings; fixes are in flight on this branch):
- Empty-schedule → monolithic-with-gaps → permanent slot-by-slot (~8 tok/s) cliff
  (`NativeDynamicShapePlan_gpubackend.cu:3283-3307`)
- OOM pre-instantiate gate → `markFailed` → permanent slot-by-slot, no recovery
  (`gpubackend.cu:484-503`)
- Island-handle-not-ready silent per-island slot-by-slot fallback (`gpubackend.cu:2046-2066`)
- `cudaStreamSynchronize` on capture-failure retry path (`gpubackend.cu:4141-4157`)

Design principle for everything below (from the literature review): **make policy
explicit where it is currently emergent** — declared shape buckets, chosen gap
placement, init-time warmup, bounded+re-promotable fallback — and **prefer patching
captured graphs over recapturing them**.

---

## 2. Workstreams

Effort scale: S (≤1 day), M (2-5 days), L (1-3 weeks). Every item ends with the
validation gate it must pass (Section 4 defines the full protocol).

### WS-A · GPU capture stability (highest GPU leverage)

**A1. Patch instantiated CUDA graphs with `cudaGraphExecUpdate` instead of recapturing.**
- Problem: pointer/param drift today triggers invalidate → recapture → re-instantiate →
  validation replay, and post-capture `bumpArgGeneration()` (`gpubackend.cu:4023, 4938`)
  ejects the next step from the frozen fast path (`_cuda.cu:458-464` returns MAYBE).
  Instantiation is the expensive phase; validation replay adds a full graph launch
  (`gpubackend.cu:519-539`).
- Change: on drift where topology is unchanged, re-capture into a fresh `cudaGraph_t`
  (cheap) and call `cudaGraphExecUpdate(existingExec, newGraph, ...)` to patch the
  existing executable in place; fall back to full re-instantiate only when update
  reports a topology change. Skip validation replay on the update path (it validated
  at first instantiate). Add the update path to `CudaGraphReplayHandle`
  (`libnd4j/include/graph/cuda/CudaGraphReplayHandle.cu/.h`) and route the
  invalidation sites in `gpubackend.cu` / `_cuda.cu:1330-1670` through it.
- Precedent: llama.cpp CUDA graphs (ggerganov/llama.cpp PR #6763, NVIDIA); NVIDIA blogs
  "Constructing CUDA Graphs with Dynamic Parameters" / "Employing CUDA Graphs in a
  Dynamic Environment".
- Risk: update semantics differ across CUDA versions for memset/memcpy nodes — gate on
  runtime CUDA >= 12.x check and keep the full-recapture path as fallback. Effort: L.
- Validate: DSP regression gate + 4-config benchmark sweep; expect recapture-heavy
  scenarios (shape drift, multi-page VLM) to stop paying instantiate+validation.

**A2. Static input placement: captured graphs reference only plan-owned addresses.**
- Problem: the consolidated arg-table is rebuilt on host and re-uploaded H2D whenever
  any gap output pointer changes (`gpubackend.cu:1993, 2027-2044, 1563-1576`), and
  addr-hash rechecks scan O(slots) (`gpubackend.cu:1253-1344`). Root cause: captured
  work references addresses that can move.
- Change: formalize the existing staging-buffer mechanism (`_prereplay.cu:397-430`)
  into an invariant — every VARIABLE external input is D2D-copied into a plan-owned,
  never-moving staging buffer, and captures/arg-tables reference ONLY plan-owned or
  frozen-weight addresses. Then: pointer drift for external inputs becomes impossible,
  `needsArgRefresh` ejection (`_cuda.cu:458-464`) is removable, and A1's update path
  is needed only for gap-output churn.
- Precedent: PyTorch cudagraph-trees (all graph inputs copied into static placeholder
  buffers in a graph-private pool — pointer churn eliminated by construction); ONNX
  Runtime IOBinding-at-fixed-address requirement for its CUDA graph support.
- Risk: one extra D2D per variable input per step (already paid today in most paths);
  audit which ext inputs are truly variable vs weights. Effort: M-L. Order: do A2
  BEFORE A1 — it shrinks the class of drift A1 must handle.
- Validate: regression gate + sweep; grep `COMPOSITE_REPLAY_TIMING` before/after —
  `argRefresh=...(n)` count should go to ~0 in steady state.

**A3. `launchBatchMemset`: stop re-uploading stable `dstPtrs`/`sizes` every step.** (Do first — small and isolated.)
- Problem: `NativeDynamicShapePlan_batchzero.cu:282-291` re-uploads both device arrays
  per step; the sibling `launchBatchD2D` (line ~248) already documents and implements
  the "upload once, only srcPtrs per step" optimization. Sizes are immutable after
  freeze; dstPtrs stable once `planLifecycle_.pointersStable()`.
- Change: mirror the D2D dirty-flag scheme; re-upload only when the gap-prezero set or
  pointers change. Effort: S.
- Validate: regression gate batch + one OPTIMAL benchmark run (`prezero=` timing line).

**A4 (stretch). CUDA 12.3+ conditional graph nodes to absorb tiny control-flow gaps.**
- Small IF/WHILE gaps between islands could live inside the merged graph as device-side
  conditional nodes, removing island boundaries entirely. Only attempt after A1/A2;
  requires driver-version gating. Effort: L, speculative.

### WS-B · Island/gap policy

**B1. Deliberate gap placement: capture-all-except-blacklist mode + minimum island size.**
- Problem: islands emerge from `SectionTypeConfig` coverage; the DEFAULT compile scope
  is ELEMENTWISE+IDENTITY only (`libnd4j/include/graph/gpu/SectionTypeConfig.h:62-83`,
  ADR 0061:651) — matmul/attention/norm/gather/concat all default to gaps, so
  out-of-box islands degenerate to 1-4-op elementwise chains. The OPTIMAL benchmark
  config already proves broad coverage correct (`BenchmarkConfig.java:206-211` sets
  `tritonCompileAll(true)` + include types + fusion). Separately, ops with
  `OP_TRAIT_EXTERNAL_WORKSPACE` are permanently-live gaps
  (`gpubackend.cu:861-894`, `dspGapCaptureBlockExternalWorkspace=true` default).
- Change (two knobs, both wired through Environment per rule 9):
  a) `dspCaptureAllExceptBlacklist` (decode-oriented default ON for LLM plans): treat
     everything as capturable EXCEPT an explicit blacklist (attention/external-workspace/
     value-dependent ops) — the inverse of today's whitelist. vLLM V1 ships exactly
     this shape: piecewise CUDA graphs split ONLY at attention; everything else is
     captured in large pieces.
  b) `dspMinIslandSlots` (default ~4-8): in `buildCompositeReplaySchedule`
     (`gpubackend.cu:920-1005`), demote sub-threshold islands into the adjacent gap
     instead of paying per-graph launch + dirty-mark + handle overhead for 1-4 ops.
     Precedent: TF-XLA auto-clustering `min_cluster_size` exists for exactly this.
  The existing ≤8-slot glue-gap merging heuristic (`gpubackend.cu:845-860`) is the
  dual of (b); keep both, make both configurable.
- Risk: (a) widens Triton kernel coverage on non-benchmark models — gate rollout with
  the accuracy validation suite. Effort: M for (b), M-L for (a).
- Validate: full sweep + `run-validation.sh --test outputAccuracy` + dsp-matrix; expect
  fewer, larger islands (`COMPOSITE_REPLAY_ENTRY: ... islands=N gaps=M` diag line).

**B2. Batched-GEMM groups must survive island merging.**
- Problem: bgemm groups are built BEFORE island merging; merging can swallow a trigger
  slot and any group shrinking below 2 members is disabled
  (`NativeDynamicShapePlan_batchgemm.cu:379-435`). Pass-3 cross-segment recovery only
  re-groups identical (M,N,K,trans,type) signatures (449-541) — heterogeneous Q/K/V vs
  FFN projections fall back to individual `cublasGemmEx` calls.
- Change: either build/reconcile bgemm groups AFTER the merge decision, or make the
  merge planner cost-aware: don't absorb a slot whose absorption disables a ≥2-member
  bgemm group unless the merged graph covers ALL members. Effort: M.
- Validate: regression gate + sweep; add a diag counter for `disabledGroups` before/after.

### WS-C · Plan cache and shape policy

**C1. Bucketed / symbolic cache keys (kill plan-per-prompt-length).**
- Problem: the cache key content-hashes every placeholder shape-info exactly
  (`NativePlanCache.cpp:51-71`), so each distinct prompt length = full plan compile +
  ≥2 slot-by-slot warmups per segment (`NativeDynamicShapePlan.cpp:286-290`) + Triton
  JIT + capture. `SymbolicShapeRanges.cpp` exists but never reaches the cache key.
- Change: canonicalize shapes through a bucket table before hashing. Two policies,
  both worth having:
  a) Declared manifest (TensorRT optimization-profile style): config lists bucket
     boundaries per dynamic dim (e.g. seq ∈ {128, 256, 512, 1024}); runtime pads/keys
     to the bucket ceiling.
  b) Automatic promotion (TorchDynamo automatic-dynamic / TF `reduce_retracing`
     style): track misses per (outputSetHash, rank-signature); on the k-th miss that
     differs only in one dim, mark that dim symbolic/bucketed and re-key.
- Touch: `NativePlanCache::Key` + `hashShapeInfoContents`, key computation in
  `NativeOps_dsp.cpp:316-333`, Java `computePlaceholderShapeHash` must apply the same
  canonicalization (`DynamicShapePlanExecutor.java:1395-1407, 2520`). Effort: L.
- Validate: regression gate; a varying-prompt-length test (20 distinct lengths) should
  produce ≤ #buckets plans instead of 20 — assert via `PLAN_CACHE NEW_PLAN` diag count.

**C2. Init-time prewarm/capture sweep API (move warmup off the request path).**
- Problem: DSP warms, freezes, captures, and validation-replays on live traffic — the
  first requests pay everything. Field norm is the opposite: vLLM captures per declared
  batch-size bucket at engine init with dummy inputs; TensorRT builds engines offline;
  JAX offers AOT lower/compile + persistent cache.
- Change: `DynamicShapePlanExecutor.prewarm(List<ShapeSignature>)` (and a
  GenerationPipeline hook): for each declared bucket, synthesize dummy placeholder
  inputs, run the warmup executions to REPLAYING, then return. Pairs naturally with C1
  buckets. Effort: M (mechanism exists; this is orchestration + API).
- Validate: TTFT measurement on the VLM benchmark — first-token latency after prewarm
  should approach steady-state token latency.

**C3. Cache concurrency + eviction quality.**
- Problem: `getOrInsert` takes one exclusive mutex for hits AND misses, runs the whole
  `factory()` (deserialize + FusionPass + buildSegments, including the O(N²)
  matmul-boundary scan `NativeDynamicShapePlan.cpp:6300-6348`) under that lock, plus an
  O(N) donor scan (`NativePlanCache.cpp:133-199`). Eviction/passivation destroys
  captured graphs of possibly-hot plans; `reactivate()` restores nothing
  (`NativePlanCache.cpp:338-355`).
- Change: (i) per-key singleflight — build the plan OUTSIDE the global lock, insert
  under it; (ii) `std::shared_mutex` so hits are read-locked; (iii) secondary index for
  donor lookup; (iv) frequency-weighted eviction (protect plans with high
  executions-per-minute), and make reactivation schedule re-capture instead of
  silently re-warming on the request path. Effort: M.
- Validate: `DspConcurrentPlanSharingTest` (full class) + regression gate.

### WS-D · Steady-state hot-path trims (small, high-certainty wins)

**D1. Java per-token tax** (`DynamicShapePlanExecutor.java`, all verified):
- Hash only true placeholders, not all external inputs incl. weights (1395-1407 —
  currently ~1332 iterations + map lookups per token on VLM; invoked unconditionally
  at 2520).
- Collapse the four full O(numExtInputs) scans per token (generic identity catch-all
  3244-3261; frozen snapshot validation 3501-3533; final closed-buffer scan 3540-3552)
  into ONE pass gated by an epoch/dirty counter; keep the safety semantics, run the
  full paranoid version only when the epoch changed or every Nth step.
- Stop allocating `numOutputs` dummy scalars + OpaqueNDArray + JNI call per token
  (3431-3435; comment says slot indices may reorder across executions — so either add
  a native `setGraphContextOutputPlaceholders(ctx, n)` set-once API, or cache the
  dummy OpaqueNDArrays and re-set only the indices).
- Demote hot-path `log.info` to `log.debug`/gated: `DSP_EXEC_PRE/POST` (3561-3570),
  `SHAPE_RESET_CHECK` (3470-3473).
- Effort: S-M total. Validate: regression gate + sweep; profile-visible Java CPU per
  token should drop; correctness covered by existing DSP core batch.

**D2. C++ composite-replay trims** (`gpubackend.cu`):
- Precompute per-island/per-merged-group output-slot index arrays at schedule build;
  replace the per-step wiring-walk dirty-mark loops (2087-2099, 1692-1704) with a flat
  iteration over the precomputed list.
- Cache the `TritonGraphBackend*` once per replay call — no `dynamic_cast` per refresh
  site (2030-2034, 1564-1568).
- Hoist the cuBLAS-Lt TLS enable/disable toggle from per-gap-unit (1751-1753,
  2001-2003) to schedule scope.
- Effort: S-M. Validate: regression gate + one OPTIMAL run comparing
  `COMPOSITE_REPLAY_TIMING` breakdown (islandDirty/mergedDirty should shrink).

**D3. After A2 lands: remove or widen the addr-hash recheck** (`gpubackend.cu:1253-1344`,
  `kAddrStableSkipThreshold=3`, `kAddrRecheckInterval=64`) — with plan-owned captured
  addresses the O(slots) hash adds no safety. Keep behind a verify-mode flag.

### WS-E · CPU tier (largest structural gap: CPU has no real replay)

**E1. CPU frozen-replay micro-schedule (tier-1.5).**
- Problem: `FunctionalReplayHandle::replay()` is a no-op counter
  (`FunctionalReplayHandle.cpp:69-79`); even the CPU_FROZEN_REPLAY fast path re-enters
  `executeSlot` per slot with SlotSyncGuard input/output loops, TLS deferred-delete
  guards, shape-key checks, and diag hooks (`_segments.cpp:995-1021`,
  `_slotexec.cpp:56-103`).
- Change: at freeze, bake a flat array of dispatch records — {DeclarableOp*, prebound
  Context*, resolved input/output NDArray*s} — and make CPU steady state a straight
  loop over records calling `op->execute(ctx)`, with none of the per-slot machinery.
  All shape/lifecycle checks were already done at freeze; guard the schedule with the
  same invalidation triggers that unfreeze GPU plans.
- Precedent: this is baseline-tier JIT design (V8/HotSpot: profile in interpreter,
  then run a specialized artifact); ONNX Runtime similarly derives fixed memory
  patterns + kernel sequences from first runs. Effort: L.
- Validate: CPU regression tests + `run-bge-test.sh`; measure per-pass wall time on
  the [32x512] embedding workload (historical reference: 68-82s/pass pre-fixes).

**E2. Wavefront inter-op parallelism.**
- Problem: the segment loop is strictly serial (`_segments.cpp:1006-1021`); independent
  slots (Q/K/V projections, parallel branches) never run concurrently even though
  `PlanTopology` computes the DAG.
- Change: schedule E1's dispatch records in topological wavefronts; execute each
  wavefront's records on a small thread pool (`dspCpuInterOpThreads`, default 1 =
  today's behavior; opt-in initially). Cap intra-op OMP threads when wavefront width
  > 1 to avoid oversubscription (coordinate with E3).
- Risk: op-internal thread-safety assumptions; start with a whitelist of known-safe
  ops. Effort: L.

**E3. BLAS threading defaults** (`libnd4j/include/helpers/impl/BlasHelper.cpp`, verified current):
- `SD_BLAS_SERIALIZE` defaults to TRUE (616-619) — a global mutex around every GEMM.
  Correct default for multi-threaded OpenBLAS safety, pure overhead for the dominant
  single-caller DSP case. Change: default serialization OFF when the process has a
  single DSP execution thread (or when MKL — already handled at 107-114), keep ON when
  E2's inter-op pool is active with OpenBLAS.
- MKL builds still pin `mkl_set_num_threads(1)` at library load when no env var is set
  (52-56) — a hard single-core GEMM cliff. Change: default to physical-core count,
  mirroring the fix already made for OpenBLAS (72-79 is now a documented no-op).
- Effort: S. Validate: CPU embedding benchmark before/after; watch for the historical
  deadlock class the "BlasHelper deadlock fix" addressed (d52307d35e) — regression-test
  multi-threaded CPU inference.

**E4. CPU attention decode parallelism** (`ops/declarable/helpers/cpu/`):
- `paged_attention.cpp:59-64`: outer OMP loop parallelizes over BATCH only — at
  batch=1 decode all heads run on one thread. Change to collapse(2) over (batch, head)
  for the forward; remove the tiny inner `PRAGMA_OMP_PARALLEL_FOR_SIMD` regions over
  headDim (64-128 elems) in KV append (168-176) — fork/join exceeds the work; plain
  SIMD loop instead.
- `cascade_attention.cpp:44-57`: hoist per-(b,h,q)/per-chunk `std::vector` allocations
  to per-thread scratch buffers.
- Effort: S-M. Validate: CPU LLM decode benchmark (`run-llm-benchmarks.sh --backend
  cpu`), correctness via existing attention op tests.

**E5 (stretch). Promote compiled CPU tiers.**
- MLIR `CpuIRBuilder` (ADR 0085) is dead code without `HAVE_MLIR`; OpenVINO rejects
  any segment containing one unsupported op (all-or-nothing) while only oneDNN gets
  the island-style partial fusion via `NativeSlotExecutor` (`_segments.cpp:420-430,
  666-786`). Change: give OpenVINO the same island treatment; longer-term make a
  compiled CPU tier the default build (field norm: Inductor C++/ORT fused CPU EP/XLA
  CPU are all default-on). Also: stop rebuilding the `nativeSlotCallback`
  `std::function` on every `executeSegmentWithSpecificBackend` call in steady state
  (700-786). Effort: L (the std::function fix alone: S).

### WS-F · Fallback discipline (coordinate with in-flight cliff patches)

**F1. Re-promotable deoptimization.** Wherever the in-flight patches leave a
`markFailed`/degraded terminal state, replace permanence with cooldown + bounded
re-attempt (exponential backoff, capped attempts per plan lifetime). Precedent: JVM/V8
deopt discipline — a deoptimized method is never pinned to the interpreter forever.
Effort: S-M on top of the patches.

**F2. Degradation must be loud.** One WARN-level line (not DSP_DIAG-only) when a
segment has run > N consecutive steps in a degraded mode (gap-fallback slot-by-slot,
monolithic invalidation loop, bgemm disabled), including the reason and the config
knob that would surface more diagnostics. The audit found every cliff was silent at
default log levels. Effort: S.

---

## 3. Phasing and dependencies

Phase 1 — independent quick wins, each mergeable alone (order within phase is free):
  A3, D1, D2, E3, E4, F2
Phase 2 — structural, sequenced:
  A2 → A1 → D3   (static placement first shrinks what update must handle)
  B1 (b: min-island-size first, then a: blacklist inversion)
  C3, E1
Phase 3 — larger bets:
  C1 → C2 (buckets before prewarm API), B2, E2 (needs E1), E5, A4, F1
Expected headline impact (directional, verify by measurement): Phase 1 trims fixed
per-token host overhead (Java scans/allocs/logging + replay-loop trims); Phase 2
removes the recapture/arg-refresh tax and interpreter-heavy CPU steady state; Phase 3
removes plan-explosion and first-request warmup from the serving path.

## 4. Validation protocol (mandatory per AGENTS.md — run from `platform-tests/`, always `tee`)

For ANY change in this doc:
1. Rebuild the relevant backend (Section 0 commands).
2. DSP regression gate:
```
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspHandleDataModelTest,DspBufferAliasAccuracyTest,DspHandleTest,DspLifecycleExhaustiveTest,DspLifecycleValidationTest,DspFrozenConstantInvariantTest,DspExtInputStalenessTest,DspSlotLifecycleAuditTest,DspConcurrentPlanSharingTest,DspCompositeReplayTest,TestDspShapePrePass \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  -Dnd4j.dsp.diagnostics.file=/tmp/dsp-core-batch.json 2>&1 | tee /tmp/dsp-core-batch.log
```
   Reference milestone: 1590 tests, 0 failures/errors/skipped.
3. GPU perf changes — 4-config benchmark sweep, 250 tokens each
   (SLOT_BY_SLOT, OPTIMAL, TRITON, CUDA_GRAPHS via `./run-benchmark.sh --backend cuda
   --tokens 250 --config <C> --op-timing [--diag-replay --diag-stream]`), recording
   commit, config, `lateSteady tok/s`, `steady tok/s`, `decode tok/s`, replay status.
   Use `COMPOSITE_REPLAY_TIMING` (enable execution timing + TIMING diag category) for
   the per-step breakdown: `total/prezero/mergedLaunch/gapExec/islandLaunch/argRefresh`
   — capture BEFORE and AFTER per item; that line is the primary evidence of effect.
4. Accuracy-affecting changes (B1a especially): `./run-validation.sh --test
   outputAccuracy` + `./run-dsp-matrix.sh`.
5. CPU changes: `run-bge-test.sh` timing + `./run-llm-benchmarks.sh --backend cpu
   --test baseline --models qwen --tokens 20`.
6. Never report throughput from <250-token runs.

## 5. Anti-goals

- No execution-mode forcing, no DSP disabling, no cache-clearing workarounds (Section 0 rule 8).
- Do not regress the multi-mode flexibility that is DSP's actual differentiator: any
  change must keep per-island graphs, merged groups, live gaps, and bgemm groups all
  functional and freely mixable (that flexibility is ahead of vLLM/TRT/torch.compile,
  which each commit to a single capture topology — keep it).
- `-Dlibnd4j.compute` stays untouched; header edits only when unavoidable.

---
---

# PART II — Building on DSP's unique advantages (strategy workstreams)

Part I fixes overheads. Part II is the offensive play: DSP was built chip-agnostic
with substitutable execution components so that whatever is *genuinely faster* can be
subbed in per region — slot-by-slot hand-optimized kernels, Triton islands, merged
captures, cuBLAS/bgemm, oneDNN/OpenVINO/MLIR on CPU, and per-chip replay handles
(CUDA/HIP-ZLUDA/LevelZero-ZLUDA/Vulkan/Metal/TPU/Hexagon via `GraphReplayFactory.cpp:19-55`).
No peer framework has this: vLLM/TRT/torch.compile each commit to one topology.

**The thesis: the selection is the product.** Today the choice of what runs where is
made by static tables (`OpTraitTable.cpp`, `SectionTypeConfig.h`), config flags
(`BenchmarkConfig` hand-tuning per model), and priority order (CPU backend cascade).
The architecture supports substitution; the *policy* is still configured by hand.
Everything needed to make the policy EMPIRICAL already exists in-house:
- A mandatory warmup phase per plan = a free measurement window (no peer has this as
  a first-class lifecycle stage).
- A shipped per-op kernel auction: ADR 0055/0058 `KernelDispatchHelper` /
  `SD_KERNEL_AUTOTUNE=1` benchmarks all usable platform helpers and caches winners in
  a performance registry.
- Triton PGO scaffolding: `TritonGraphBackend.h:140-150` `AutoTuneEntry` +
  `autoTuneCache_` (shapeKey → {firstExecTimeMs, configIndex, numAttempts, settled,
  MAX_ATTEMPTS=3}) with a comment describing threshold-triggered recompilation with
  alternative tile configs — declared but not yet driving compilation.
- Plan persistence to make decisions durable across processes: ADR 0093 +
  `TritonCacheBundle.cpp` (already serializes numWarps/config metadata to disk).

Precedents for measured selection: FFTW's planner (a "plan" IS a measured winner),
cuDNN benchmark mode / `torch.backends.cudnn.benchmark`, PyTorch Inductor
`max-autotune` (races ATen vs Triton matmul templates), TVM/Ansor, and — closest —
**Collage (PACT 2022)**: per-region *backend placement* mixing TensorRT/cuDNN/TVM by
measurement, shown to beat any single backend end-to-end. DSP's replay-unit schedule
is the natural substrate Collage had to bolt onto other systems.

### WS-G · The replay-unit auction (flagship)

**G1. Per-unit mode auction during warmup.** For each schedule unit, enumerate
candidates: {slot-by-slot native (incl. hand-fused kernels), Triton island (captured),
absorb-into-merged-group, cuBLAS-gap vs bgemm-group variant}; on CPU: {interpreter
micro-schedule, oneDNN, OpenVINO, MLIR-JIT}. Extend warmup from "2 execs to observe
shapes" to K auction rounds: round-robin candidates across rounds, time with CUDA
events / steady clocks (`OpTimingTracker` + `COMPOSITE_REPLAY_TIMING` machinery),
pick per-unit winners, bake into the schedule. Lift the ADR-0058 dispatch pattern one
level: same philosophy, unit granularity instead of op granularity.
- Touch: `buildCompositeReplaySchedule` (`gpubackend.cu:920-1005`) gains a
  candidate-set builder; `SegmentLifecycle`/warmup driver gains auction rounds;
  reuse ADR 0055's performance-registry storage.
- Effort: L. Gate behind `dspUnitAuction=true` initially.

**G2. Verify-then-race.** A candidate is eligible only after its warmup output matches
the slot-by-slot reference (infrastructure exists: ADR 0080/0081 accuracy validation,
`DspPlanAssertions`, `run-validation.sh` tolerances). Correctness gates the auction;
measurement decides among the verified. This turns the accuracy-validation infra from
a test tool into part of the execution engine.

**G3. Performance ledger + persistence (do FIRST — cheap, enables everything).**
Per-unit decision record: {unitId, candidateSet, measured µs each, winner, environment
fingerprint = (device, arch, driver, dtype, shape bucket)}. Persist with the plan
(ADR 0093) and in the Triton cache bundle. Re-auction triggers: fingerprint change,
sustained steady-state drift beyond threshold (the AutoTuneEntry comment's design),
or explicit flag. Side benefit: the ledger IS a perf regression detector — CI can
diff ledgers across commits.

**G4. Hand-optimized kernels as first-class schedule citizens.** Today a hand-fused
kernel (`skip_rms_norm`, `rms_norm_linear`, fused warp-shuffle attention — ADR 0097)
runs as an anonymous gap: live-dispatched every step. Add an explicit unit kind
(e.g. `REPLAY_UNIT_NATIVE_FUSED`) so "this region runs a hand kernel because it won"
is encoded in the schedule, and **capture-wrap hand kernels when capture-safe** so a
winning hand kernel still gets CUDA-graph launch amortization inside merged groups.
FusionPass rewrite candidates *compete* in the auction with Triton sections instead
of being mutually exclusive. This is the "slot-by-slot/islands for hand kernels" idea
made explicit, measured, and captured.

### WS-H · Triton as the portable codegen substrate

**H1. Close the tile-config PGO loop the code already sketches.**
`selectTileConfig` (`TritonIRBuilder_analysis.cpp:858-952`) picks ONE static config
per section (decode: warps=4/stages=1 "empirically optimal"; size buckets 2/2, 4/3,
8/4; clamp ≤16) and the only runtime override is a single GLOBAL env pair
(`TritonGraphBackend_binary.cpp:330-346`). Implement what `AutoTuneEntry` describes:
2-4 candidate configs per section (warps × stages × block), compiled in the existing
parallel worker pool, raced during warmup rounds (G1), settled at MAX_ATTEMPTS, winner
persisted. The disk cache ALREADY keys binaries by numWarps/numStages
(`TritonGraphBackend_cache.cpp:135-136`), so variants coexist on disk today.
Effort: M — this is the highest-ROI standalone Triton item.

**H2. Shape-bucket-specialized kernels per section.** `autoTuneCache_` is keyed by
shapeKey — drive it: decode (q_len=1) vs prefill vs bucket-N get separately tuned
binaries of the same section. Pairs with Part I C1 buckets (auction keying = bucket).

**H3. Widen what Triton may BID on, not what it OWNS.** Flip the meaning of
`SectionTypeConfig` compiled flags from "wins by default" to "eligible to bid":
MATMUL DotOps, NORMALIZATION, GATHER, FUSED_ATTENTION sections get compiled as
*candidates* and race cuBLAS/bgemm/hand kernels per unit (G1). Resolves Part I B1's
narrow-default finding in the architecture's own spirit — no static default to argue
about; measurement decides per model, per shape bucket, per chip.

**H4. One section analysis, N lowerings (the chip-agnostic payoff).**
`TritonIRBuilder_sections` (section discovery, fusion legality, epilogue absorption)
is target-agnostic analysis; lowering is where targets diverge: Triton→PTX (CUDA),
Triton via ZLUDA (HIP/LevelZero paths already routed in `GraphReplayFactory.cpp:24-39`),
MLIR-CPU (`CpuIRBuilder` linalg→LLVM), HLO (TPU backend). Refactor so section
building is shared and lowering is a target plug-in; then every chip inherits the
same fusion analysis, and the G1 auction compares "lowered section" vs "native ops"
uniformly per chip. This is the concrete form of "chip agnostic": the *analysis* is
the asset, targets are interchangeable.

**H5. Launch strategy as a candidate, not a flag.** Cooperative launch (global
spin-barrier, known occupancy risk on large grids) vs threadfence-barrier multi-kernel
vs plain sections: make these auction candidates per section rather than a global
`tritonCooperativeLaunch` config.

### WS-I · Harden the chip-agnostic contract

**I1. Capability descriptors on `GraphReplayHandle`.** The base class
(`GraphReplayFactory.cpp:60-80`+) already gives every chip addr-snapshot/match. Add a
static capability descriptor per handle: {canCapture, canUpdateInPlace (Part I A1),
supportsConditionalNodes, launchOverheadHint, prefersLargeUnits, memoryModel}. The
scheduler and auction then reason portably ("this target has expensive launches →
bias toward bigger merged units") instead of via #ifdefs.

**I2. Make CPU replay real (Part I E1) so the auction is apples-to-apples.** Once
`FunctionalReplayHandle` executes a baked micro-schedule instead of counting, "replay"
means the same thing on every chip, and CPU candidates (interpreter vs oneDNN vs
OpenVINO vs MLIR) race under identical rules.

**I3 (long-term). Placement auction.** The same ledger extends to per-unit *device*
placement (embedding gather on CPU while attention runs on GPU, Hexagon offload for
quantized units). `DeviceExecutionContext`/multi-device infra exists; Collage is the
literature precedent that measured placement wins. Do not attempt before G1-G3 are
stable.

### WS-K · Kernel-level audit findings (hand-written Triton emitters + CUDA kernels)

Audited 2026-07-02: the hand-authored Triton kernel EMITTERS (TritonIRBuilder_kernels/
_module/_sections/_emitters — these define the algorithm of every generated kernel)
and the hand-written CUDA kernels on the decode path (FlashAttentionHelper.cu,
rms_norm.cu fused ops, kv_scatter.cu, batchzero/batchgemm kernels).

**Architectural fact:** FUSED_ATTENTION sections emit a full FlashAttention-2-style
MLIR kernel (`TritonIRBuilder_kernels.cpp:1227-1841`, online softmax, GQA mapping,
dual past/current KV) — they do NOT dispatch to FlashAttentionHelper.cu. The hand
CUDA kernels serve the helper/gap path. There are therefore TWO attention
implementations to maintain — prime auction candidates (G1), and both share gap K1.

**K1. No split-KV / split-K anywhere — the decode long-context ceiling (CRITICAL).**
Three independent confirmations of the same algorithmic gap:
- Hand CUDA decode attention: grid = (numQHeads, batch), each block serially scans
  ALL of seqKV in 64-wide tiles (`FlashAttentionHelper.cu:999, 1149`). B=1, H=32,
  KV=32k → 32 blocks / 512 serial tiles on a 100+-SM GPU.
- Triton attention decode: blockM=1, grid=(B×numQHeads, 1)
  (`TritonIRBuilder_internal.h:255-257`) — same shape, same ceiling.
- Triton matmul: grid has no K dimension (`TritonIRBuilder_kernels.cpp:102-107`), so
  decode GEMV [1,K]×[K,N] gets zero K-parallelism; the hand rms_norm_linear GEMV
  (`rms_norm.cu:494-500`) likewise runs a serial K loop.
Fix shape: flash-decoding style — partition KV (or K) across blocks, each computes
partial (max, sumexp, acc), then a small reduce kernel combines. Applies to all three
sites. This is THE kernel-level item for long-KV decode throughput.

**K2. Correctness triage before any tuning (verify, then fix — these force slow or
wrong paths):**
- GatedMLP / FusedTwoLayerMLP emitters load x and store output with NO mask
  (`TritonIRBuilder_kernels.cpp:820-822, 914, 1217`) → OOB reads/writes whenever
  M < blockM (decode M=1!). Other emitters (matmul :428, rmsNormLinear :502-514)
  do mask — these two were missed.
- NORMALIZATION blockSize is capped at 4096 (`TritonIRBuilder_analysis.cpp:924`,
  `_module.cpp:947-955`); a softmax row wider than 4096 (32k vocab logits) would
  reduce only a prefix. Determine which path the vocab softmax actually takes before
  assuming impact — but this is a silent-wrong-answer class if hit.
- `kvInPlaceWrite` dtype-mismatch path calls `NDArray::cast()` (allocates + launches)
  (`kv_scatter.cu:395-399`) — illegal under CUDA graph capture; same allocation-
  during-capture class the rms_norm_linear comment (rms_norm.cu:615-618) explicitly
  engineered around.

**K3. Kernel micro-fix bundle (each S-M effort, independent):**
- batchMemsetKernel / batchD2DKernel: grid = one 256-thread block PER BUFFER
  regardless of size (`batchzero.cu:188-209, 253, 306`) — a 10MB KV buffer is zeroed
  by one block; tiny buffers waste whole blocks. Size-proportional grid partitioning.
- Attention PV dot runs f32×f32: P (from exp) and V are cast to f32 before tt.dot
  (`TritonIRBuilder_kernels.cpp:1744, 1789-1805`); FA2 practice casts P to fp16 for
  the PV product → ~4× tensor-core throughput on that dot. Accuracy-gate the change.
- Mixed-section matmul falls to `emitPerElementMatmul` — scalar K-loop, step=1, no
  tt.dot (`TritonIRBuilder_sections.cpp:1900-1997`) — orders of magnitude below
  tensor cores; prefer section-splitting so matmul lands in a PURE_MATMUL section.
- Sectioned REDUCTION unrolls `reductionSize` loads into straight-line IR
  (`_module.cpp:5654-5698`; ~6 IR ops × axis length) instead of tt.reduce —
  compile-time blowup + icache thrash. Use the tt.reduce path normalization uses.
- N-D broadcast emits a div/mod chain per dim per element per input
  (`_module.cpp:5339-5354, 1162-1191`) — hoist/strength-reduce or cache
  delinearization across inputs sharing the output shape.
- FusedTwoLayerMLP re-loads x from global for every H-tile
  (`TritonIRBuilder_kernels.cpp:1005-1059`): x read (H/blockH)× more than necessary;
  stage x in shared memory across H-tiles.
- skipRmsNormKernel recomputes input+skip in pass 2 instead of reusing hiddenRow
  (`rms_norm.cu:280-306`) — 2× reads of two arrays per row.
- batchedGemmCastFloat2Half: scalar 1-elem/thread + one launch PER group member in a
  loop (`batchgemm.cu:60-67, 835-844`) — vectorize (half2/float4) + single batched
  launch.
- rmsNormLinear SMEM request always max(required, 33792) (`rms_norm.cu:529-532`,
  `LaunchDims.h:944`) — caps occupancy at 1 block/SM even when K needs 8KB.
- fusedGQADecode threadCount fixed 256 while headDim=64 models leave 75% of threads
  idle in score+accumulate phases (`LaunchDims.cu:1287-1289`).
- kvScatterBatched re-uploads entry metadata H2D every step with no change-detect
  (`kv_scatter.cu:175-178`) — copy the batchzero dirty-flag pattern.
- Elementwise/gather loads are emitted width-1 with i32 offsets (`_module.cpp:5199,
  1007, 5275`) — Triton's coalescer may or may not vectorize these; INSPECT GENERATED
  PTX before acting (suspected-needs-measurement, unlike the rest of this list).

**K4. Auction tie-ins (Part II):** the hand GEMV wins decode by a static guard
(`M == 1 && K <= 8192`, rms_norm.cu:619), not measurement — exactly what G1 should
decide per shape bucket vs cuBLAS GEMV vs a Triton split-K kernel. Baked shape
constants (M/N/K/seqQ/seqK as ConstantIntOp, `TritonIRBuilder_kernels.cpp:109-110,
172-194, 1304-1307`) mean one kernel per shape tuple — pairs with H2 bucket-
specialized binaries and C1 bucketed keys (bake the BUCKET ceiling, mask to the real
length, instead of baking the exact length). Two-pass softmax/variance in standalone
NORMALIZATION emitters (`TritonIRBuilder_emitters.cpp:1083-1091, 1120-1138`; Welford
single-pass is the alternative) are also auction-able variants rather than must-fix.
Historical note: the "attention hardcodes float32" concern is REFUTED in current code
— `FlashAccType` (`FlashAttentionHelper.cu:55-71`) gives fp16/bf16→f32 accumulation
and double→double, which is standard FA2 practice.

### WS-L · Triton-CPU: built, linked, defined — and unreachable (audit 2026-07-02)

> FULL DESIGN: see `DSP_CPU_CODEGEN_DESIGN.md` (repo root) — expands this workstream
> into components, ABI/build decisions, phases P0-P5, validation, risks. Implement
> from the design doc; this section remains the audit summary.

**User-visible symptom:** `-Dlibnd4j.triton=ON` is mandatory on CPU builds, yet no
Triton executes on CPU. Audit confirms both halves.

**What `triton=ON` actually does on a CPU build (all verified):**
- Auto-enables the oneDNN helper on x86 (`buildnativeoperations.sh:2688-2705`) and
  gates OpenVINO (`cmake/Options.cmake:87`, `Dependencies.cmake:2602-2605` — "OpenVINO
  is triggered by SD_TRITON=ON, not helpers"). On CPU, "triton" is a de-facto
  meta-flag for the compiled-CPU-backend stack.
- Downloads and builds the REAL thing: `triton-lang/triton-cpu` at pinned commit
  c4ccb98 (`Dependencies.cmake:1573, 1603-1604`), with its own LLVM built from source
  at pinned commit 20902f0, host-target-only (`:1650-1657, 1781-1787`), codegen
  backend "cpu" (`:1616-1618`), a patch that strips the Python dependency
  (`:2005-2008`), installed to `triton_cpu_install`/`triton_cpu_llvm_install`
  (`:1438-1443`, reuse-if-present at `:1446-1453`).
- Defines `HAVE_TRITON=1` AND `HAVE_TRITON_CPU=1` (`blasbuild/cpu/include/config.h:11-12`,
  CMakeCache `HAVE_TRITON_CPU:BOOL=ON`), and links `triton_interface` (libtriton.a +
  globbed MLIR/LLVM static libs, libMLIRIR/libMLIRSupport in whole-archive mode,
  `Dependencies.cmake:1464-1523`) into the object lib (`MainBuildFlow.cmake:871-872`).

**Why nothing uses it:**
- `MainBuildFlow.cmake:189-198`: on `NOT SD_CUDA`, ALL `graph/gpu/*.cpp` sources are
  removed from the build — every TritonIRBuilder_*, TritonGraphBackend_*,
  TritonTargetDispatch, TritonCacheBundle file. The comment calls them "CUDA-only
  infrastructure". So the section analysis + emitters + dispatch simply do not exist
  in the CPU binary.
- `HAVE_TRITON_CPU` has ZERO references in any .cpp/.cu/.h — only config plumbing
  (config.h.in:12, PostBuild/TypeRegistryGenerator cmake) and one unused generated
  Java constant (`Nd4jCpu.java:239`). No code path anywhere invokes triton-cpu's
  pass pipeline.
- The CPU fallback JIT is ALSO off: `HAVE_MLIR=0` in the current CPU config.h, and
  `MainBuildFlow.cmake:213-224` excludes `CpuIRBuilder.cpp`/`MlirCpu*` when MLIR is
  off. Runtime CPU backend chain is therefore OpenVINO → oneDNN → interpreter
  (`graph/cpu/NativeDynamicShapePlan_cuda_stubs.cpp:566-578`), no codegen tier.
- Net: the CPU build pays an LLVM-from-source + triton-cpu toolchain build (hours
  cold, GBs of disk; amortized by install-dir reuse) and ships whole-archive MLIR
  core in the .so, for zero runtime use today. Two halves of a bridge that don't
  meet: the toolchain exists with no consumer; the consumer code exists but is
  excluded from CPU compilation.

**Upstream grounding (fetched 2026-07-02):**
- `triton-lang/triton-cpu` README: official experimental CPU backend, "long-lived
  development branch", WIP; activated in Python via `TRITON_CPU_BACKEND=1` /
  `set_active_to_cpu()`. No tagged releases — the repo's pinned-commit approach is
  correct. (Lowering path: Triton IR → TritonCPU dialect → vector dialect → LLVM;
  grid "programs" map to parallel loop iterations on host threads.)
- `microsoft/triton-shared` README: **"This repository is no longer maintained"** —
  the linalg middle-layer alternative is dead upstream. triton-cpu is the live path;
  do NOT plan around triton-shared.

**Integration plan (this is H4 made concrete — the missing 20%):**
- L1. Make the graph/gpu exclusion file-granular: compile the target-agnostic .cpp
  set (TritonIRBuilder_*, TritonGraphBackend_{cache,lru,binary}, TritonTargetDispatch,
  SectionTypeConfig users) on CPU builds under `HAVE_TRITON`; keep genuinely
  CUDA-only files (.cu: kernel launch, cudagraph, driver dispatch) excluded.
  (`MainBuildFlow.cmake:192-198`)
- L2. Add a CPU target to `TritonTargetDispatch`: drive triton-cpu's pass pipeline
  (TritonToTritonCPU → vector → LLVM) from the ALREADY-LINKED libtriton.a, JIT via
  LLVM ORC instead of PTX/cuModule, cache objects in the existing disk cache
  (TritonCacheBundle already serializes per-config binaries). Launcher: iterate the
  grid, dispatching block-programs across a host thread pool; respect E2/E3 thread
  budgets (cap BLAS/OMP threads under it).
- L3. CPU tile profile in `selectTileConfig`: numWarps/numStages are GPU concepts —
  CPU profile = vector-width-multiple block sizes, single stage; let H1's PGO race
  pick per section.
- L4. Register a `TritonCpuGraphBackend` in `getCpuGraphBackendChain()` alongside
  OpenVINO/oneDNN and let the G1 auction choose per unit (Triton-CPU section vs
  oneDNN fusion vs OpenVINO vs interpreter). Cheaper interim alternative: flip
  `HAVE_MLIR` on so the existing CpuIRBuilder (ADR 0085) provides a codegen tier —
  but check LLVM-pin compatibility with the triton-cpu toolchain (20902f0) before
  assuming the installs can be shared; the triton-cpu route is strictly more
  valuable because it reuses the SAME section analysis + emitters as GPU.
- Validation: CPU regression suites + `run-bge-test.sh` timing + `run-llm-benchmarks
  --backend cpu`; auction gate per G2 (verify vs interpreter reference).
- Until L1-L4 land, be honest about the cost: triton=ON on CPU buys oneDNN+OpenVINO
  enablement plus an unused toolchain build. Do NOT gate it off (repo anti-goal:
  no disabling features) — wire it up; it is the chip-agnostic thesis's own test case.

**L5. Alternatives survey (checked 2026-07-02; search down, READMEs fetched directly):**
Three slots, not one:
- Section-codegen slot (drop-in for the TritonTargetDispatch CPU target): (a)
  triton-cpu — PRIMARY, already built, reuses the GPU section emitters verbatim;
  experimental/no releases, so keep the lowering behind the H4 plug-in seam; (b)
  in-tree CpuIRBuilder + upstream MLIR (ADR 0085) — written, currently excluded
  (HAVE_MLIR=0); linalg-based; needs LLVM-pin reconciliation with the triton-cpu
  toolchain; (c) IREE llvm-cpu — VERIFIED ALIVE (LF AI&Data project, stable+nightly
  releases, AMD MLPerf 2025 submission; Apache-2+LLVM-exception); consumes
  linalg/StableHLO so it pairs with the CpuIRBuilder emitter, not the Triton-dialect
  one; heavier, graph-backend-shaped integration.
- Graph-backend slot (peer of oneDNN/OpenVINO in getCpuGraphBackendChain): XLA:CPU
  via PJRT — CHEAPEST EXPERIMENT: graph/tpu/PjrtClientManager plus the shared
  KernelSpec/KernelExpr StableHLO target already exist in-tree for TPU; the same
  PJRT client API loads the CPU plugin. Also: Apache
  TVM (Relax/MetaSchedule — autotuning philosophy matches WS-G, but Python-centric
  tuning). ONNX Runtime: circular for this repo, skip.
- Microkernel slot (no codegen — G4 auction hand-kernel candidates; directly hits
  the CPU decode-GEMV / batched-GEMM findings): libxsmm — VERIFIED ALIVE (BSD-3,
  active CI) — JIT'd small/batched GEMM microkernels with AVX-512/AMX, C API;
  oneDNN brgemm/ukernel API (already linked!); XNNPACK (elementwise/conv
  microkernels); KleidiAI for ARM (pairs with the in-tree ArmHybridGraphBackend.cpp,
  also currently excluded).
- Dead/avoid: triton-shared (README: "no longer maintained" — verified), Glow
  (archived), PlaidML/TensorComprehensions (dead). BladeDISC: README fetched but
  activity unverified (search down) — check commit history before depending on it.
Recommended shape: triton-cpu primary + lowering kept pluggable (H4) with IREE as
the hedge; XLA:CPU-via-PJRT as a near-free graph-backend experiment; libxsmm +
oneDNN brgemm registered as auction candidates for decode GEMV/bgemm on CPU. All
compete under G1 measurement, none win by fiat.

### WS-M · DSP-based TRAINING performance (audit 2026-07-02)

Training runs through DSP (TrainingSession extends InferenceSession; forward+backward
+loss are ONE plan — gradients are requested outputs, no separate backward plan).
The audit verdict: **training is architecturally pinned to DSP's slowest tier, by
explicit design, and then pays three more layers of per-step tax on top.**

**M0 — Headline: training never freezes, never captures, never replays.**
- `TrainingSession.java:1024-1031` (verified verbatim): "Training MUST NOT freeze
  shapes. Frozen shapes trigger CUDA graph capture, which records ... specific
  weight values → zero effective learning." So every iteration of every epoch runs
  slot-by-slot.
- Native side confirms the gates that make freezing hostile to training today:
  (a) any weight DataBuffer REBIND (Adam lazily allocating moment buffers, clipping
  that reallocates) → full unseal + ALL-segment capture invalidation + cast-cache
  purge (`NativeDynamicShapePlan.cpp:2354-2406`, fired from the per-exec
  refreshProtectedWeightBuffers at :2428); (b) weights are NOT marked variable by
  default (`:1900-1905`) so detectFrozenConstants would freeze weight-only-input
  ops (stale outputs after updates — latent correctness trap,
  `_slotexec.cpp:1453-1489`); (c) marking a weight variable AFTER freeze unseals +
  invalidates everything (`:3753-3769`).
- The fix is precedented, not speculative: PyTorch `make_graphed_callables` captures
  forward+backward+optimizer with static placement — weights updated IN PLACE at
  stable addresses, so replay reads current values (graphs bake POINTERS, not
  values; what bakes values in DSP is frozen-constant detection and the fp16
  pre-cast caches — those need trainable-input awareness, not a blanket no-freeze).
  Requirements mapped from the audit: pre-allocate ALL optimizer state before first
  exec; declare trainable inputs up front (a "stable-address mutable" input class =
  Part I A2 static placement applied to weights); exempt trainable inputs from
  frozen-constant detection and value-baking caches (invalidate-on-write); then
  freeze → capture forward+backward (+in-graph updaters, M2) → replay the whole
  training step. This is the single biggest training-perf lever (slot-by-slot vs
  replay was measured at ~8 vs ~68 tok/s on decode; training pays the same class of
  gap every iteration).

**M1 — Per-step forced syncs (quick win).** `commitAndTrimAfterDspStep`
(`TrainingSession.java:452-463`): because `frozen` is always false, `commit()`
(stream sync) + `trimMemoryPoolOnStream` run EVERY step (inference trims every 10th).
Plus loss scalar D2H per loss var per step (`:354-364`). Fix: use the same
TRIM_INTERVAL semantics regardless of frozen-state, make loss extraction async/
batched.

**M2 — In-plan updater fusion is a STALE DISABLE (quick win, verified).**
`TrainingSession.java:272-276` disables fused updater ops citing "misclassified as
BINARY_EW in OpTraitTable.cpp" — but the table is ALREADY FIXED:
`OpTraitTable.cpp:787-812` documents the old bug and now classifies `sgd_updater`
UNARY_EW, `adam_updater`/`nesterovs_updater`/etc. FULLY_WRITING with correct arity.
Re-validate and re-enable. Payoff stack: removes ~4×numParams Java-side kernel
launches per step (`applyUpdaterForGradientPreUnscaled`, :916-994), removes the
per-param `paramArr.syncToDevice()` (:987-993), and — once updaters are in-plan —
gradient READBACK itself becomes unnecessary except for listeners: today every
gradient is copied TWICE per step (native copyBuffer D2D into a fresh INDArray,
`DynamicShapePlanExecutor.java:3895-3906`, then `.dup()` again at
`TrainingSession.java:539`).

**M3 — Listener-forced whole-graph readback.** Any non-empty listener list expands
the requested outputs to EVERY op output in the DAG
(`TrainingSession.java:332-346`, verified; same expansion in InferenceSession
:915-927), and fit() always installs a HistoryListener — so training plans compute
AND return every intermediate activation every step. Fix: expansion only for
listeners that declare activation requirements (ListenerVariables already exists);
HistoryListener needs loss only.

**M4 — Native per-exec training taxes.** refreshProtectedWeightBuffers rebuilds an
unordered_set over all weights EVERY exec (`NativeDynamicShapePlan.cpp:2307-2331`,
called at :2428) — O(numWeights) hashing per step, swap-stored even when unchanged;
`dirtySlotGenerations_` std::fill O(totalOutputSlots) per exec (:2192); JNI
setOutputArray per gradient output per step (`NativeOps_dsp.cpp:242-246`); addr-hash
recompute ramp (3 execs × all segments) after every invalidation event
(`gpubackend.cu:1253-1305` + `_segments.cpp:343-368`); per-mark O(N²)-class
detectFrozenConstants re-runs when weights are marked variable one-by-one
(`_slotexec.cpp:1690-1724`, `:3808`). Fix direction: epoch/dirty-flag gating (same
family as Part I D1), batch the variable-marking API.

**M5 — Plan-cache split: forward vs training never share warmup.** Cache key
includes outputSetHash (`NativeOps_dsp.cpp:297-315`), so `{logits}` and
`{loss, grad_*...}` are disjoint plans — each pays full compile+warmup+capture;
donor-plan reuse requires identical output sets (`NativePlanCache.cpp:162-168`).
Eval-during-training alternates two cold-ish plans. Fix: donor across output
subsets (forward plan donates structure to training plan), or one plan with
optional outputs.

**M6 — Backward kernels fragment islands worse than forward.**
- `gather_bp` has NO OpTraitTable entry at all → unconditional gap every step
  (embedding backward; table has `embedding_lookup` forward only,
  `OpTraitTable.cpp:1031`); CUDA impl is unconditional atomicAdd + full-tensor
  memset (`gather_bp.cu:121,148` — hot-token contention, no deterministic/sorted
  path); CPU impl is a SERIAL loop (`gather_bp.cpp:69`).
- `clipbynorm_bp`/`clipbyavgnorm_bp` carry only the BP trait (no category → cannot
  anchor islands, `OpTraitTable.cpp:677-708`); `clipByGlobalNorm` CUDA does a
  PER-PARAMETER synchronous D2H readback in a loop (`clip.cu:336-342`) — N stream
  stalls per step. Fix: single fused device-side global-norm (one readback or fully
  in-graph).
- Decomposed backwards where forward is fused: `rms_norm_bp` = 9-13 temp NDArrays +
  dispatches (`llm_ops.cpp:152-186`); `rms_norm_linear_bp` RECOMPUTES the forward
  norm (`llm_ops.cpp:1181-1196`, confirms ADR 0097 note);
  `dot_product_attention_bp` re-materializes the full [B,H,Tq,Tk] score matrix —
  no flash backward on the default path (`dot_product_attention.cpp:190-258`;
  `flash_attention_bp` exists in the trait table :391 — check wiring);
  softmax-CE grad = 6 dispatches (`softmaxCrossEntropyWithLogits.cpp:144-160`).
- No multi-tensor updater kernel (one launch per param,
  `updaterAdam.cu:38-48`; PyTorch/apex multi_tensor_apply is the precedent);
  `matmul_bp` runs dX/dY GEMMs sequentially on one stream (`matmul.cpp:335-336`);
  dropout re-inits RNG on host + H2D per call (`dropout.cu:100-112` — also VERIFY
  the fixed `(3019L, seed)` pattern doesn't repeat masks across steps).

**M7 — Training memory: coloring cannot help, checkpointing absent.** Buffer
coloring shares by disjoint liveness intervals (`DspBufferColorMap.cpp:98-210`);
forward activations live until their backward consumer → near-zero reuse for
activation-class buffers in training plans. No rematerialization/gradient-
checkpointing support at plan level. Long-term item; pairs with M0 capture work.

**M8 — Correctness gates for training-perf work:** the allocateSpecial-on-frozen
guard (`DataBuffer.cpp:478-492`) kills gradcheck perturbation re-execution on any
frozen plan (known failure class) — M0's trainable-input class must carry a
perturbation story; batchZero/captured memsets fire on gradient outputs that ops
fully overwrite (`batchzero.cu:40-66`) — audit `needsPrezero` flags for _bp ops.

**M9 — No training perf harness exists** (DspTrainingE2ETest / SameDiffTrainingTest
are correctness-only). Add a step-time benchmark (fwd+bwd+updater breakdown,
steps/sec, per-phase timers wired into tryDspTrainingIteration) BEFORE fixing
anything above, so M-items get before/after numbers.

Suggested order: M9 (harness) → M1+M2+M3 (quick wins, Java-only) → M4 → M6 kernel
items (gather_bp trait + global-norm + multi-tensor updater first) → M0 (capture
training steps; the big lift) → M5, M7.

### WS-N · CUDA stream-sync over-synchronization (audit 2026-07-02)

Premise (user-stated, audit-confirmed): syncs are correct but frequently STRONGER
than needed. Good news first: the steady-state replay path is clean —
`CudaGraphReplayHandle::replay()` is a bare async `cudaGraphLaunch`
(`CudaGraphReplayHandle.cu:103-111`), no post-launch sync. Everything OUTSIDE
captured graphs over-syncs.

**N1 — KEYSTONE: `DebugHelper::checkErrorCode` = unconditional `cudaStreamSynchronize`
in production** (`helpers/DebugHelper.h:103-125, 151-170` — only skipped during
capture; no debug gating; `checkGlobalErrorCode` :129-149 is the cheap
sticky-error-only variant). Census: ~309 stream-drain call sites — ~215 in
`ops/declarable/helpers/cuda/`, ~94 in `loops/cuda/` (EVERY primitive transform/
reduce/broadcast/scalar/random dispatch drains). Consequence: every eager op, every
training slot-by-slot op, and every DSP live-gap op is fully synchronous; async
pipelining exists ONLY inside captured graphs (where the sync is skipped — which is
part of why replay wins so big).
Fix (PyTorch model — no post-launch syncs in prod; `CUDA_LAUNCH_BLOCKING=1` for
debug): make checkErrorCode sync only when `Environment::isDebug()`/new
`SD_SYNC_AFTER_KERNEL` env is set; production path = `cudaGetLastError` sticky check
(catches launch-config errors immediately; execution errors surface at the next
natural sync with less precise attribution — acceptable, documented). ROLLOUT:
env-flag first (`auto|always|never`), full DSP regression gate + 4-config benchmark
sweep + an nsys trace to confirm pipelining actually materializes, THEN flip
default. NOTE: removing these syncs may expose latent races the drains were hiding —
that is the point of the gate, not a reason to keep the drains.

**N2 — Multi-drain helpers: one tail sync (or none), not per-kernel.** Worst
offenders (drains per file): `lup.cu` 18; `sg_cb.cu` 12 (+ a synchronous
`cudaMemcpy` D2H mid-loop at :487); `loops/cuda/random.cu` 12 (dropout = one drain
per layer per step); `segment_{sum,prod,min,mean,max}.cu` 7-8 each;
`triangular_solve.cu` 8 (incl. two drains inside one branch for independent
kernels :411-440); `broadcastableFused.cu` 6 (residual adds!);
`BarnesHutTsne.cu` 4-5. Same-stream sequential kernels need NO intermediate drains
(in-stream ordering); collapse to one tail checkErrorCode — which N1 then gates.
Load-bearing exceptions catalogued (keep): D2H reads driving host branches
(`segment.cu:93`, `sg_cb.cu:487`, `image_suppression.cu:301` — NMS per-box sync
loop, fix is architectural GPU-side NMS), drain-before-free (`histogram.cu:131`,
though redundant with the drain above it), `autoregressive_decode.cu:914` (token-id
readback — correctly placed step boundary).

**N3 — Core runtime (buffer/allocator/JNI):**
- `syncToPrimary` does D2H on LEGACY STREAM 0 then drains it
  (`array/cuda/DataBuffer.cu:1046-1105`) — stream 0 serializes with every
  non-blocking stream, so every host read (`getDouble`, loss fetch, `e<T>()`) is a
  device-wide fence. Fix: copy on the producing LC stream (or a dedicated D2H
  stream ordered via the existing `_writeEvent`) and sync only that.
- `cudaPointerGetAttributes` on EVERY sync/set call — 5 hot sites
  (`DataBuffer.cu:1022, 1254, 1688, 1711, 1740`): thousands of synchronous driver
  round-trips per step to re-validate what `_specialDeviceId` already tracks. Fix:
  trust the tracked id; attribute-check only under debug or on failover paths.
- `ConstantHelper::replicatePointer` (`ConstantHelper.cu:286-294`) and
  `PointersManager::replicatePointer` no-context path (`PointersManager.cu:191-194`)
  drain after every shape/TAD/dimension H2D — same-stream consumers need no drain;
  warmup does this hundreds of times.
- `PointersManager::synchronize()` per op (sort/shuffle/gather/concat idiom;
  `PointersManager.cu:217-219`, e.g. `NDArray_core.cu:240`) — justification is
  temp-free safety, but frees are `cudaFreeAsync` stream-ordered already. Remove;
  N1 covers the error-check role.
- JNI surface is fully synchronous: `memcpySync/memsetSync` AND the cross-device
  `memcpyAsync` branch all drain `cudaStreamPerThread` before returning
  (`legacy/cuda/NativeOps.cu:900-903, 981, 1025`); Java-side async is structurally
  impossible. Fix: honor the Async contract; `commit()` (:1087) remains the explicit
  Java-side fence.
- `DataBuffer::migrate` drains the whole source stream + uses synchronous
  `cudaMemcpyPeer` (`DataBuffer.cu:2130-2179`) — event-ordered `cudaMemcpyPeerAsync`
  on a P2P stream for multi-GPU paths.
- Per-H2D driver chatter: `cudaGetDevice` + conditional event-create in
  `recordSpecialWriteEvent` (`DataBuffer.cu:886-920`); `cudaStreamIsCapturing` per
  `syncToSpecial` (:1322-1328) duplicating the TLS authority (`DebugHelper`
  consolidation exists — use it).
- Allocator: `cudaMemGetInfo` per allocation when soft-limit is set
  (`CudaMemoryPool.cu:363-396` — cache with a TTL/alloc-count refresh); trimPool
  drains ALL dirty free-streams incl. the exec stream on alloc failure
  (:1221-1235); `allocateDirect` + capture-arena growth drain per allocation
  (:1518-1520, 1593, 1688 — event-bridge instead); `DataBuffer::expand` drains
  where a stream-ordered `cudaFreeAsync` suffices (:344-354).

**N4 — DSP steady-state driver-call trims (small, mechanical, high certainty):**
per-sub-kernel `cudaGetDevice` at `TritonGraphBackend_kernel.cu:111` BYPASSES the
TL cache defined at :45-49 (use it); per-sub-kernel `cudaStreamIsCapturing`
(:118-125) re-queries what `executeSegment` already computed at
`_execute.cu:296` (pass the bool down); per-sub-kernel unconditional
`cudaGetLastError` (:101-107 — gate on a prior-failure flag); per-segment
`cudaGetDevice` (`_execute.cu:287` — use plan `deviceId_`); permanently-signaled
`preallocReadyEvent` waited every step (`_execute.cu:345-358` — consume-once flag);
`dspSyncDefaultStream` full LC drain on the MmulHelper non-Triton fallback
(`DspCudaDispatch.cu:265-270`, caller `MmulHelper.cpp:1218` — event instead);
`drainFingerprintRingPublic` uses `cudaDeviceSynchronize` + sync memcpy for a
single-stream producer (`_cudagraph.cu:176-182` — stream-scoped). Combined
sub-kernel driver chatter alone ≈ 100-250µs/step at 14 sub-kernels.

**N5 — Scalar-to-host long tail:** `batched_gemm.cu:181-206` reads alpha/beta via
`e<T>(0)` (D2H fence) per call — pass as host scalars/iArgs;
`attention/feature_distillation_loss.cu` reduceNumber+`e<double>(0)` per training
step — keep loss on device.

**N6 — UNDERSYNC found (debug-mode correctness):** `TRITON_VERIFY_KERNELS` compares
`memcmp` results BEFORE the async D2H completes (`_execute.cu:1093-1097`, same at
:1220-1228) — verification silently always passes. Add the missing stream sync in
verify mode.

**Order:** N4 (mechanical, zero-risk) → N1 behind env flag + validation → N3
syncToPrimary/pointer-attributes/replicatePointer → N2 mechanical collapse (can be
scripted file-by-file) → JNI async surface + allocator items → N5. Every step: DSP
regression gate + benchmark sweep; add one nsys capture per phase to verify
overlap actually appears (the win is pipelining, not the µs of the syscalls).

### WS-O · CUDA multi-device failover (audit 2026-07-02)

**Compliance verdict on the repo law (non-peer devices must remain failover
candidates): PASS.** `CudaMemoryPool::allocateFailover` (`memory/cuda/
CudaMemoryPool.cu:541-850`) builds ONE candidate list — peers via the async pool,
non-peers via `cudaMallocManaged` (:759-789) — sorted peers-first then by free
memory (:695-699); the only exclusions are the explicit `excludedFailoverDevices_`
set and physical device count. Final fallback = pinned host with a limit check
(:795-846). Capture-time OOM throws cleanly rather than recording wrong-device work
(`DataBuffer.cu:787-794` + pool capture-workspace throw — though the "fall through
to pool allocation" comment is misleading; the code is safe). Single-device and
zero-device degrade correctly.

Gaps found (correct-but-incomplete, plus test holes):

**O1 — Concurrent-OOM thrash storm.** N threads OOMing simultaneously each
independently run the full ladder: per-device `cudaSetDevice` + `trimPool` (which
drains that device's dirty free-streams) + `cudaMemGetInfo` (:664-692) — no
single-flight, no shared pressure snapshot. 8 threads × D devices × (drain +
meminfo) = latency cliff exactly when the system is already stressed. Fix:
single-flight the candidate scan (one thread scans, others wait on the result) +
a cached free-mem snapshot with ~ms TTL. (Related: soft-limit mode's per-alloc
`cudaMemGetInfo` is already filed in WS-N.)

**O2 — No return-home/hysteresis.** A buffer failed over under transient pressure
stays remote for process lifetime: managed path sets `_specialDeviceId` to the
requesting device (:769-789 → `DataBuffer.cu:826-828`) so `migrate()`'s
"already on target" check (:1968-1971) never fires, and nothing watches for
pressure clearing. Hot buffers can be permanently PCIe-demand-paged after one OOM
burst. Fix: pressure-cleared event → re-allocate home + `cudaMemPrefetchAsync`
back, rate-limited.

**O3 — Managed prefetch is mis-ordered.** `cudaMemPrefetchAsync(ptr, size, dev,
nullptr)` (:781) runs on the legacy null stream; the consumer computes on
`tl_dspExecutionStream` (non-blocking) — no ordering, so first touch can still
demand-fault mid-kernel, defeating the prefetch. One-liner class fix: prefetch on
the consumer stream (or event-bridge).

**O4 — Java/C++ split-brain.** ADR 0090 is "Proposed": Java
`DeviceMemoryManager.selectFailoverDevice` (`DeviceMemoryManager.java:835-864`) and
its memory caps are NEVER consulted by the C++ ladder — `setMemoryCap(dev1, 4GB)`
does not stop the pool from filling dev1 on failover. Also observability:
native ground truth is `TransferMetrics` (recorded at DataBuffer D2H/H2D/migrate —
`DataBuffer.cu:1118, 1365, 2212-2220`) but pool failover allocations are NOT
recorded there, and `DspReplayTransferAnalytics`/`ReplayProfileManager` are
Java-side only (zero native references) — they see stub topology, not real
failover events. Fix: JNI cap/exclusion bridge (the exclusion-list mechanism
already exists in the pool) + record failover allocs in TransferMetrics.

**O5 — Test-fidelity hole: the C++ ladder is never integration-tested under real
pressure.** `configureStubTopology` swaps Java-side providers only
(`DeviceMemoryManager.java:1084-1126`); `DeviceRoutingTest` failover tests call
Java `selectFailoverDevice` without allocating a single CUDA byte;
`CudaMemoryAllocationFailoverTest` uses Java-layer memory simulation. NOT covered
anywhere: multi-threaded concurrent OOM failover; free-after-failover device
correctness; managed-buffer kernel-access perf; failover-mid-capture throw;
CUDA_VISIBLE_DEVICES subsets; Java-cap enforcement. These tests come FIRST —
before any of the fixes above.

**O6 — Free-path device-identity edge.** `free(ptr, -1, stream)` defaults
`trackDevice` to 0 and skips the device switch (`CudaMemoryPool.cu:1073-1083` +
:963-970) — wrong-device `cudaFreeAsync` → silent leak for failover buffers freed
via that path; `deleteSpecial` falls back to `_deviceId` when `_specialDeviceId`
is -1 (legacy setSpecial paths, `DataBuffer.cu:1416-1420`). Fix: assert/resolve
device from pointer attributes on the -1 paths (rare, so the query is affordable
there).

**O7 — DSP inter-segment migration is a peer-only, uncached primitive (dormant
today).** The function is `platformMigrateSegmentInputs`
(`NativeDynamicShapePlan_cuda.cu:944-1050`; earlier draft mislabeled it by its diag
tag). It serves the slot-level heterogeneous-placement mechanism
(`slots_[i].targetDeviceId` + `platformBindSegmentDevice` → `bindSegmentCudaDevice`
:936-938) and migrates inter-segment intermediate outputs to a segment's explicit
target device. DORMANT by default: `targetDeviceId = -1` (auto) early-returns
(:951) — nothing in the default pipeline sets explicit targets. When activated it
needs hardening: (a) `cudaMemcpyPeerAsync` only, and on copy failure it logs and
`continue`s — the input silently stays remote (correct only for managed/UVA
buffers; plain non-peer allocations then fault at kernel time). Add host-staged
fallback. (b) NO caching: `migratedInputs_.clear()` + fresh `new NDArray` per call
(:953, :1019) → per-exec re-migration + addr-key churn if ever used per-step;
copy ownership/deletion needs an audit. (c) O(numSlots) producer back-walk per
needed input (:987-996) — precompute a producer index. This machinery is the
natural substrate for WS-G I3 (placement auction) — harden it there, not before.
DSP interplay already safe per prior audits: peer-pool failover's NEW pointer is
caught by addr-key drift → args refresh; weight rebind → full unseal (WS-M);
managed failover keeps the SAME pointer → captured graphs stay valid via UVA
(perf risk = O3).

**Open items — RESOLVED (2026-07-02 follow-up):**
- Weight migration (`_cuda.cu:2703-2789`): TEARDOWN/PASSIVATION-only, not
  per-step — inside the releaseGpuIntermediates flow it moves protected weights
  from pool memory into `allocateDirect` persistent storage so the pool can trim,
  correctly deferring while another frozen plan holds refs (:2713-2720). Minor
  trims: per-buffer `cudaPointerGetAttributes` (:2727, off hot path — fine),
  duplicate `trimPool` at :2780 AND :2787. The `:2832` site is
  `copyStagingToBuffer` — warmup-only JIT staging sync with a per-call
  cudaEventCreate/Destroy (:2845-2849; WS-N event-pool theme, low frequency).
- `DeviceExecutionContext`: dormant scaffolding — a thread-local snapshot struct
  (`fromThreadLocals`/`applyToThreadLocals`) referenced ONLY by its own impl
  files; no scheduler passes it. Multi-device segment scheduling exists instead
  via the slot `targetDeviceId` path above (wired, dormant).
- AffinityManager vs DeviceMemoryManager: execution-affecting paths use the NEW
  mechanism (`MultiGpuWorkspaceSessionMemMgr` pins via
  `DeviceMemoryManager.selectBestGpu()`/`switchDevice()` incl. re-pin per scopeIn,
  `MultiGpuWorkspaceSessionMemMgr.java:73-98`; TrainingSession trim uses
  `DeviceMemoryManager.getCurrentDeviceId()` :461). The OLD
  `AffinityManager.getDeviceForCurrentThread()` survives ONLY in
  diagnostics/pool-stats calls (`InferenceSession.java:2038, 2115, 2182`) —
  worst case is stats read from the wrong device, not wrong execution. Residual
  sliver to verify when convenient: does `DeviceMemoryManager.switchDevice()`
  update AffinityManager's thread mapping (two "current device" sources could
  disagree in logs). Also noted: the periodic trim loops trim ALL devices every
  interval (`InferenceSession.java:642-645, 963-966`) — on multi-GPU boxes that
  drains dirty free-streams on every device each interval (ties to O1/WS-N).

**Order:** O5 tests first (they gate everything) → O3 prefetch stream (one-liner)
→ O6 free-path asserts → O1 single-flight scan → O7a migration fallback + O7b
verification → O2 return-home → O4 cap bridge + metrics.

### Sequencing for Part II

1. **G3** (ledger + persistence — days, unblocks everything, immediate CI value)
2. **H1** (tile-config PGO — self-contained, pure Triton win)
3. **G1 minimal** (auction over {slot-by-slot, Triton island} only) + **G2** gating
4. **G4** (hand-kernel units + capture-wrapping) and **H3** (bid-widening)
5. **H2, H4, I1, H5**, then **I3** last.
Dependencies on Part I: A1/A2 make candidate swapping cheap; C1 buckets define
auction keys; C2 prewarm is *where* auctions run so production requests never pay
for measurement. Validation: everything in Part II is still bound by Section 4; the
auction adds its own gate — a chosen winner must never be slower than the slot-by-slot
reference measured in the same warmup (if it is, the ledger entry is marked suspect
and the reference wins).
