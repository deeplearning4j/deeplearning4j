# DSP Validation Playbook

A change-type → tests reference for DSP work. Pairs with `AGENTS.md` (authoritative rules) and
the required regression gate. Goal: for any change, know the *minimum* teed-up validation before
you claim it works, and escalate by blast radius.

All tests run from `platform-tests/`, always `tee`'d to a file, read from the tee log. Full mvn
path: `/home/agibsonccc/dev-apps/mvn/bin/mvn`. One build/benchmark at a time.

---

## Test tiers (cheapest → most thorough)

| Tier | What | When | Cost |
|---|---|---|---|
| T0 Compile | `mvn install -DskipTests -pl :nd4j-api` (Java) or the native build | every change | 30 s (java) / 3-10 min (native ccache-warm) |
| T1 Isolation | one class/method, `-Dtest=Class#method*` | reproduce a bug, first correctness check | 30 s - 2 min |
| T2 Plan-cache keying | `DspHandleTest,DspHandleDataModelTest,DspConcurrentPlanSharingTest,TestDspShapePrePass` | any plan-cache / keying / executor change | ~2 min (104 tests) |
| T3 DSP core gate | the 11-class batch (below), target **1590 / 0 / 0** | any DSP-internal / replay / DataBuffer change | 10-30 min |
| T4 Config matrix | `run-dsp-matrix.sh` (8 configs vs SLOT_BY_SLOT golden) | Triton / capture / mode changes | model-dependent |
| T5 Accuracy | `run-validation.sh --test outputAccuracy` | token-level correctness across modes | model-dependent |
| T6 Benchmark | `run-benchmark.sh` (the real gate for perf claims) | any perf change; final validation | ~5 min CUDA / slow CPU |

**T3 DSP core gate** (mandatory for DSP-internal changes — catches cross-test contamination +
multi-threaded plan sharing that a narrow reproducer misses):

```bash
cd platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspHandleDataModelTest,DspBufferAliasAccuracyTest,DspHandleTest,DspLifecycleExhaustiveTest,DspLifecycleValidationTest,DspFrozenConstantInvariantTest,DspExtInputStalenessTest,DspSlotLifecycleAuditTest,DspConcurrentPlanSharingTest,DspCompositeReplayTest,TestDspShapePrePass \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full -Dnd4j.dsp.diagnostics.file=/tmp/dsp-core-batch.json \
  2>&1 | tee /tmp/dsp-core-batch.log
```

---

## Change-type → required tests

| You changed… | Files (typical) | Minimum | Escalate to |
|---|---|---|---|
| **Java per-token hot path** (D1 family, executor scan/hash) | `DynamicShapePlanExecutor.java` | T0(:nd4j-api) → T2 → T6 | T3 if keying touched |
| **Plan-cache keying** (D1a) | `computePlaceholderShapeHash`, `computeShapeKey` | T2 (esp. `DspConcurrentPlanSharingTest`, `TestDspShapePrePass`) + a **varying-shape** dispatch check (#plans == #distinct dynamic shapes) → T6 | T3 |
| **CUDA composite-replay loop** (D2, arg-table, dirty-mark, addr-hash) | `NativeDynamicShapePlan_gpubackend.cu` replay | **T3 (full gate)** + T6 + T4 | — |
| **Triton kernel / emitter** (R1 masks, tile configs) | `TritonIRBuilder_*.cpp` | T4 (TRITON configs) + T5 (verify correctness) + T6 | T3 if replay dispatch touched |
| **CUDA graph capture / replay handle** | `CudaGraphReplayHandle`, capture path | T3 + T4 (`CUDA_GRAPHS_frozen`) + T6 | — |
| **DataBuffer / device transfer / sync** (WS-N) | `DataBuffer.cu`, `DebugHelper`, sync helpers | **T3** (esp. `DspBufferAliasAccuracyTest`, `DspExtInputStalenessTest`) + T6 | broad op suites |
| **KV-cache / scatter** (R3) | `kv_scatter.cu`, KV write | T5 (decode accuracy) + T6 (coherent decode) | T3 |
| **CPU BLAS / helpers** (C1, E4) | `BlasHelper`, cpu attention | CPU build → CPU DSP tests (`-Dbackend.artifactId=nd4j-native`) + CPU benchmark **with a CPU config** (see gotcha) + relevant op tests | — |
| **Multi-device / failover** (WS-O) | `CudaMemoryPool.cu`, `DeviceMemoryManager` | `DeviceRoutingTest` + real-pressure failover tests | T3 |
| **Training path** (WS-M) | `TrainingSession.java`, `_bp` kernels | training tests + a training step-time harness (before/after) + gradcheck for touched `_bp` | — |
| **Diagnostics only** (F2, R2) | `DSP_DIAG(...)` adds/recategorization | T0 (compile) + one T6 `[PASS]` (proves no behavior change) | — |
| **Frozen-constant / snapshot** | `detectFrozenConstants`, slotexec freeze | T3 (`DspFrozenConstantInvariantTest`) + T5 (decode must not degenerate — RoPE-gather class) | — |

Backends: `-Dbackend.artifactId=nd4j-cuda-12.9` (GPU) / `nd4j-native` (CPU). Changes that touch
`nd4j-api` or backend-agnostic C++ (`DeclarableOp.cpp`) affect **both** — validate on both.
Changes in `.cu` / CUDA-Triton files do **not** compile into the CPU `.so` and cannot affect CPU.

---

## Gotchas this session paid for

- **Parameterized tests need `#method*`.** JUnit5 `@ParameterizedTest`/`@MethodSource` display as
  `method(Nd4jBackend)[1]`; a bare `#method` runs **0 tests**. Also, some all-parameterized classes
  intermittently run 0 tests under bare-class / `#*` / `%regex` — target `#specificMethod*`.
- **CPU benchmark needs a CPU config.** `run-benchmark.sh --backend cpu` with the default
  `SMOLDOC_IDEAL` fails CONFIG_COMPILE ("resolved to EMULATED_REPLAY instead of TRITON") — that
  config asserts Triton, which CPU can't provide. Use `--config SLOT_BY_SLOT`. Its exact-match
  `FINAL_VALIDATE` also diverges from the CUDA golden on CPU/CUDA FP differences — CPU should be
  validated on a **tolerance/match-rate**, not exact CUDA-match (open item under investigation).
- **`--tokens 250` for perf claims; fewer only for debugging** and never reported as throughput.
- **CUDA build must keep `-Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda`.** `-Pcuda` alone repackages a
  stale `.so`. Incremental (no `clean`) only if `blasbuild/cuda` exists — killing a build mid-`clean`
  wipes it, forcing a full rebuild.
- **`level=full` diagnostics = per-op `syncToHost`** — can mask freeze / force slot-by-slot. Fine
  for the gate (its milestone is measured with full), but for perf tracing use `detailed` +
  explicit categories, not `ALL`/`full`.
- **Don't run a GPU benchmark concurrently with GPU tests / another GPU job** — one-GPU contention
  SIGSEGVs. Read `plan: {N replay, M host}` in the `[PASS]` line: `host>0` in steady state = a
  slot-by-slot cliff (also surfaces as a high `FALLBACK` diag count — see F2).
- **`--debug` value dumps** (`Nd4j.getEnvironment().setDebug(true)+setVerbose(true)`) print op I/O
  without instrumenting code; view-safe after the crash-A fix.

---

## The perf-claim bar

A DSP perf change is not "done" until: correctness first (T1-T5 as mapped), **then** T6 with
`tee`, recording commit / backend / config / tokens / `lateSteady tok/s` / replay status / any op
timing. `lateSteady` is the stable number. Run-to-run variance is real (kompile services share the
GPU) — a change that does strictly less work and causes no plan-cache thrashing (`1/1 cap, N
replay, 0 host`) is non-regressing even if a single run dips.
