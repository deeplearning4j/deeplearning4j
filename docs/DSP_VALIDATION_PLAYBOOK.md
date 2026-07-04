# DSP Validation Playbook

All commands run from `platform-tests/`. All output must be piped through `tee`.
Never use `export VAR=val`; pass env via `-D` Maven properties.
Never read surefire reports — read the tee log.

---

## 1. Test Tiers: Fast → Slow Ladder

### Tier 0 — Single Method (< 1 min)

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspSlotLifecycleAuditTest#testReplayAccuracy* \
  2>&1 | tee /tmp/dsp-single-method.log
```

**Parameterized-wildcard rule (CRITICAL):** Bare `#methodName` matches 0 tests.
Always append `*` for any `@ParameterizedTest` or `@MethodSource` method:
- WRONG: `-Dtest=DspSlotLifecycleAuditTest#testReplayAccuracy`
- CORRECT: `-Dtest=DspSlotLifecycleAuditTest#testReplayAccuracy*`

**DspSlotLifecycleAuditTest narrowing** — the fixture×mode matrix is large.
Use `-Daudit.fixture=A,B` and `-Daudit.mode=M1,M2` to restrict to specific rows:
```bash
-Daudit.fixture=add_scalar -Daudit.mode=CUDA_GRAPHS
```
Each method logs `PARAM_BANNER testXxx fixture=... mode=...` — use these line numbers
to extract windows from the tee log for a specific fixture×mode pair.

### Tier 1 — Single Class (1–10 min)

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspSlotLifecycleAuditTest \
  2>&1 | tee /tmp/dsp-class.log
```

### Tier 2 — DSP Core Gate (3 steps, sequential, ~20–30 min)

**Step 1: Focused multi-threaded output isolation**
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspConcurrentPlanSharingTest#testOutputBufferIsolation \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  -Dnd4j.dsp.diagnostics.file=/tmp/dsp-output-buffer-isolation.json \
  2>&1 | tee /tmp/dsp-output-buffer-isolation.log
```

**Step 2: Full concurrent plan sharing class**
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspConcurrentPlanSharingTest \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  -Dnd4j.dsp.diagnostics.file=/tmp/dsp-concurrent-plan-sharing.json \
  2>&1 | tee /tmp/dsp-concurrent-plan-sharing.log
```

**Step 3: Full DSP core batch** (last known good: 1590 tests, 0 failures, 0 errors, 0 skipped)
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=DspHandleDataModelTest,DspBufferAliasAccuracyTest,DspHandleTest,DspLifecycleExhaustiveTest,DspLifecycleValidationTest,DspFrozenConstantInvariantTest,DspExtInputStalenessTest,DspSlotLifecycleAuditTest,DspConcurrentPlanSharingTest,DspCompositeReplayTest,TestDspShapePrePass \
  -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
  -Dnd4j.dsp.diagnostics.file=/tmp/dsp-core-batch.json \
  2>&1 | tee /tmp/dsp-core-batch.log
```

The tee log doubles as a diagnostic trail. Extract fixture windows by searching
`PARAM_BANNER` line numbers.

### Tier 3 — Benchmark Sweep (~60 min, requires --tokens 250 for perf claims)

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
./run-benchmark.sh --backend cuda --tokens 250 --config SLOT_BY_SLOT --op-timing \
  2>&1 | tee /tmp/dsp-bench-slot-by-slot.log

./run-benchmark.sh --backend cuda --tokens 250 --config OPTIMAL --op-timing \
  --diag-replay --diag-stream --diag-json /tmp/dsp-bench-optimal.json \
  2>&1 | tee /tmp/dsp-bench-optimal.log

./run-benchmark.sh --backend cuda --tokens 250 --config TRITON --op-timing \
  --diag-replay --diag-stream --diag-json /tmp/dsp-bench-triton.json \
  2>&1 | tee /tmp/dsp-bench-triton.log

./run-benchmark.sh --backend cuda --tokens 250 --config CUDA_GRAPHS --op-timing \
  --diag-replay --diag-stream --diag-json /tmp/dsp-bench-cuda-graphs.json \
  2>&1 | tee /tmp/dsp-bench-cuda-graphs.log
```

Key metrics to record: `lateSteady tok/s`, `steady tok/s`, `decode tok/s`.
Fewer than 250 tokens is acceptable only for crash/correctness debugging; it MUST NOT
be cited as a throughput number.

### Tier 4 — LLM Matrix

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
./run-llm-benchmarks.sh --backend cuda --test matrix --models qwen,gemma,lfm2-extract \
  --tokens 20 --op-timing --debug \
  2>&1 | tee /tmp/dsp-llm-matrix.log
```

---

## 2. Change-Type → Required-Tests Table

| Change type | Minimum gate | Notes |
|---|---|---|
| **(a) Replay/capture internals** (`NativeDynamicShapePlan_*.cu`, `CudaGraphHandle.cu`, `CudaGraphReplayHandle.cu`, `gpubackend.cu`) | Core gate (Tier 2 all 3 steps) + Tier 3 `CUDA_GRAPHS` config | `DspCompositeReplayTest`, `DspConcurrentPlanSharingTest`, `DspLifecycleExhaustiveTest` are the primary tripwires |
| **(b) DataBuffer / sync** (`DataBuffer.*`, `CudaMemoryPool.*`, stream management) | Core gate (all 3 steps) + `DspBufferAliasAccuracyTest` isolated run (Tier 1) | `DspBufferAliasAccuracyTest` is nondeterministic — apply 3x-run rule |
| **(c) Triton / JIT dispatch** (`OpTraitTable.cpp`, `NvrtcKernelCache.*`, `TritonGraphBackend*`) | Core gate step 3 + `TRITON` benchmark + `run-dsp-matrix.sh --config TRITON_sectionFusion` + `run-dsp-matrix.sh --config TRITON_compileAll` | Verify `--config TRITON_frozen_batchedGemm` separately |
| **(d) Plan cache / lifecycle** (`DynamicShapePlanCompiler`, `DynamicShapePlanExecutor`, `NativePlanCache`) | Core gate all 3 steps + `DspLifecycleValidationTest` isolated + `DspExtInputStalenessTest` | `DspFrozenConstantInvariantTest` + `DspHandleDataModelTest` are lifecycle sentinels |
| **(e) cuBLAS / math-mode / precision** (`MmulHelper.*`, cuBLAS dispatch, `specials_single.hpp`) | `DspBufferAliasAccuracyTest` (3x) + `DspSlotLifecycleAuditTest#testReplayAccuracy*` (3x) + Core gate step 3 | `testReplayAccuracy` is nondeterministic — always run 3 times; any single failure is a real regression |
| **(f) Perf-only claims** | Tier 3 full 4-config sweep (`--tokens 250`) + Tier 4 LLM matrix | Must pass Core gate (Tier 2) first — perf work is invalid without correctness |

**3x rule for nondeterministic tests:** `DspBufferAliasAccuracyTest` and
`DspSlotLifecycleAuditTest#testReplayAccuracy*` expose GPU contention and plan-sharing
races that are probabilistic. A single green run does NOT clear them. Run 3 times;
require 3/3 green.

---

## 3. Flaky / Batch-Only Failure Protocol

### Step 1 — Is it real?
Run the failing method 3 times in isolation:
```bash
-Dtest=FailingClass#failingMethod* 2>&1 | tee /tmp/repro-1.log  # repeat x3
```
If it fails ≥2/3 times, it is a real regression. If 0/3, it is likely a batch
contamination issue — proceed to Step 2.

### Step 2 — Isolated vs batch bisection
Run the suspected class alone; if it passes, the fault is cross-test contamination.
Bisect by halving the `-Dtest=` class list until the minimal pair that triggers the
failure is found.

### Step 3 — Fixture×mode order-dependence bisect (DspSlotLifecycleAuditTest)
Use `-Daudit.fixture` and `-Daudit.mode` to run fixture pairs in different orders:
```bash
-Daudit.fixture=add_scalar,matmul -Daudit.mode=CUDA_GRAPHS,SLOT_BY_SLOT
```
Compare `PARAM_BANNER` sequence in tee log against failure timing to identify the
leaking fixture or mode transition.

### compute-sanitizer initcheck recipe
For uninitialized memory reads (silent garbage values, not crashes):
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=FailingClass#failingMethod* \
  -Dtest.prefix=/usr/local/cuda/bin/compute-sanitizer \
  2>&1 | tee /tmp/sanitizer-initcheck.log
```
The `bin/java` wrapper routes the forked JVM through compute-sanitizer.

### DSP_DIAG level selection
| Level | When to use | Side-effect |
|---|---|---|
| `detailed` | Default investigation; per-step info goes to ring buffer | None; zero execution overhead |
| `full` | Real-time echo of every event; required to see events as they happen | **Distorts timing**; forces some synchronous paths — do NOT use for throughput measurements |

The Core gate (Tier 2) always uses `full` so the tee log doubles as a complete
diagnostic trail. Extract fixture windows by line number pairs around `PARAM_BANNER`.

Prefer explicit category lists over `ALL` when timing matters:
```bash
-Dnd4j.dsp.diagnostics=EXECUTE,GRAPH_REPLAY,STREAM_SYNC \
-Dnd4j.dsp.diagnostics.level=detailed
```

---

## 4. Env / Property Rules

| Rule | Detail |
|---|---|
| Never `export VAR=val` before `mvn test` | Surefire forks a new JVM; shell env does NOT propagate |
| Use `-D` Maven properties | Properties reach the forked JVM via Surefire `<systemPropertyVariables>` |
| Wire new env vars via `platform-tests/pom.xml` | Add to `<environmentVariables>` under surefire `<configuration>` |
| Always `tee` everything | `2>&1 | tee /tmp/descriptive-name.log` — never omit |
| Never read surefire reports | `target/surefire-reports/*` splits output and drops C++ diagnostics |
| All tests from `platform-tests/` only | Never `mvn test` from project root |
| Use full Maven path | `/home/agibsonccc/dev-apps/mvn/bin/mvn` — never bare `mvn` |

**DSP diagnostic Maven→env mapping** (wired in `platform-tests/pom.xml`):
```
-Dnd4j.dsp.diagnostics=ALL          →  ND4J_DSP_DIAGNOSTICS
-Dnd4j.dsp.diagnostics.level=full   →  ND4J_DSP_DIAGNOSTICS_LEVEL
-Dnd4j.dsp.diagnostics.file=FILE    →  ND4J_DSP_DIAGNOSTICS_FILE
```

**Key DSP system properties for test narrowing:**
```
-Dnd4j.dsp.graphExecutionMode=CUDA_GRAPHS   # force a specific mode
-Dnd4j.dsp.noFreeze=true                    # disable shape freezing
-Dnd4j.dsp.cudaGraphs.enabled=false         # disable graph capture
-Daudit.fixture=add_scalar,matmul           # DspSlotLifecycleAuditTest fixture filter
-Daudit.mode=CUDA_GRAPHS                    # DspSlotLifecycleAuditTest mode filter
```

**NEVER** call `setGraphExecutionMode()` or `setDspAutoCompileEnabled(false)` in test
setup to work around a failure — fix the root cause. The only legitimate forced-mode
call is `BenchmarkConfigApplier` resetting to `AUTO`.
