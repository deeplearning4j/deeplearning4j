# Kompile And DL4J Regression Prompts

## Prompt 1: Kompile Bug Report For Detached Agent Tracking And Rule Propagation

You are working in the Kompile codebase. Investigate and fix two related orchestration bugs in Kompile's agent-dispatch tooling.

Problem:

Kompile `task` and `multi_task` dispatches can exceed the MCP tool-call timeout while the spawned CLI agent keeps running. When this happens, the tool call returns a timeout/error instead of a durable task id. Later, `poll action=status` can report no active background tasks even though `ps` shows a still-running spawned agent process. The caller has no reliable handle for polling, cancellation, stdout/stderr streaming, or deterministic result collection. This causes duplicate work, incorrect status reporting, and orphaned agent processes.

There is also a related prompt/context propagation issue: spawned agents do not reliably inherit project `AGENTS.md` rules or mandatory guardrails. The parent agent currently has to paste critical rules into every subagent prompt manually. In a DL4J session, this matters because commands such as `tail`, `git reset --hard`, `git checkout <file>`, direct `make`, and `ccache --clear` are explicitly banned. A process snapshot from the same machine showed an external Claude/Kompile-side test command using `tail -30`, which is exactly the class of rule violation the dispatch layer should help prevent.

Concrete repro observed in DL4J work:

1. Dispatch a long-running `mcp__kompile__.multi_task` with several agents.
2. Let one or more child agents run longer than the tool timeout.
3. The MCP call times out, but the child process continues.
4. Result files may appear later under `.kompile/task-results/`, but the caller never received a durable task id.
5. `mcp__kompile__.poll action=status` can report no active background tasks while `ps` still shows spawned agent processes.
6. A separate `mcp__kompile__.task` investigation also timed out after about 120 seconds while leaving a CLI agent process running.

Expected behavior:

1. Dispatch returns a durable parent task id immediately, before child work can exceed any tool timeout.
2. `multi_task` returns both the parent id and child ids.
3. Tool timeout must not sever tracking. If the request times out, Kompile should return or persist a resumable task handle and mark the task `RUNNING`, `DETACHED`, or `TIMED_OUT_BUT_ACTIVE`.
4. `poll action=status` must list all running/detached agent tasks, including pid, command, agent name, role, start time, elapsed time, prompt summary, output path, and last activity.
5. A caller must be able to poll by id, stream recent output, collect final result, and cancel/cleanup a detached task.
6. Result files must include enough metadata to tie the file back to the parent task id, child id, pid, agent, role, and original prompt.
7. Project instructions must be automatically discovered and injected into spawned agents. At minimum, Kompile should detect `AGENTS.md` in the working tree and include it or a configured required excerpt in every spawned prompt.
8. Kompile should support command guardrails for spawned agents. For DL4J, the dispatch layer should be able to deny or warn on banned commands such as `git reset --hard`, `git checkout <file>`, `git stash`, `git clean`, `ccache -C`, `ccache --clear`, direct `make`, `tail` on build/test logs, `LD_PRELOAD=libjemalloc.so`, root `mvn test`, and shell `export VAR=...` before Maven tests.

Actual behavior:

1. Long-running dispatch calls can return only a timeout/error to the caller.
2. The child agent continues outside the caller's control.
3. `poll action=status` may not show the active child process.
4. Result discovery becomes filesystem scraping instead of a tracked API flow.
5. Guardrails depend on the parent prompt being manually written perfectly.

Acceptance criteria:

1. Add or update tests that simulate a child agent exceeding the MCP request timeout and verify the task remains visible and pollable.
2. Add or update tests for `multi_task` parent and child id persistence.
3. Add or update tests that verify `poll action=status` reports detached/running child agents after the original call times out.
4. Add or update tests for final result collection after a detached child completes.
5. Add or update tests for cancellation/cleanup of detached children.
6. Add tests or integration coverage for automatic project instruction injection from `AGENTS.md`.
7. Add command-guardrail tests for configured denylisted commands.
8. Do not rely on terminal output scraping as the source of truth; persist task metadata in a structured registry.
9. Preserve backwards compatibility for existing result files where practical.

Suggested implementation direction:

1. Create task records before launching any child process.
2. Persist task records atomically to the Kompile coordination directory or task registry.
3. Register process pid and output file path as soon as the process starts.
4. Separate "request timed out" from "task failed"; a tool-call timeout should not imply child failure.
5. Make `poll` read the durable registry and reconcile it with live processes.
6. Add a cleanup path for stale records whose pids are gone.
7. Make instruction injection and command guardrails explicit configuration, with DL4J-compatible defaults when `AGENTS.md` is present.

## Prompt 2: DL4J Regression Orchestrator

You are an orchestration agent working in `/home/agibsonccc/Documents/GitHub/deeplearning4j`. Your job is to coordinate focused subagents to fix the current DL4J regression set without starting new optimization work. Do not benchmark or commit until the full regression gate passes.

CRITICAL RULES - READ BEFORE DOING ANYTHING:
1. NEVER run: git checkout <file>, git stash, git reset --hard, git clean
2. NEVER run: ccache -C or ccache --clear (destroys 2+ hours of compiled cache)
3. NEVER run: make directly (always use full mvn with libnd4j + bindings module)
4. NEVER pipe build or test output through tail (use tee, read the tee log)
5. NEVER use LD_PRELOAD=libjemalloc.so
6. NEVER run mvn test from the project root
7. NEVER use export VAR=val before mvn test (use -D Maven properties)
8. ALL tests run from platform-tests/ directory only
9. Fix root causes - NO workarounds, NO fallbacks, NO disabling features
10. If you need to undo your own changes, edit the specific lines - NEVER use git commands

Additional hard constraints:

1. No fully qualified class names in Java code. Use imports.
2. Do not apply more than one optimization change at a time.
3. Do not make benchmark claims with fewer than 250 tokens.
4. Do not run any benchmark until all required platform tests pass.
5. Do not commit unless the change is atomic, fully tested, and improves or preserves the measured target.
6. Every dispatched subagent must receive the critical rules above verbatim, the current `git status --short` modified-file list, exact scope boundaries, exact build/test commands, and success criteria.
7. If a subagent needs to undo its own edit, it must edit only the exact lines it changed.
8. If any test fails, fix the root cause. Do not dismiss it as known or unrelated.

Current state to verify first:

1. Check for active Maven or Java test processes before starting any build, test, or benchmark.
2. An old full platform test run may still be active:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dsp-full-platform-tests-after-training-replay.log
```

3. That old run began before some candidate patches were applied, so it does not validate the current worktree. Use it only for failure discovery.
4. Do not start another Maven build/test while another DL4J Maven build/test is active unless explicitly instructed.
5. Before dispatching subagents, run `git status --short` and include the current list in every subagent prompt.

Known baseline benchmark:

```text
Command: cd platform-tests && ./run-benchmark.sh --tokens 250
Config: OPTIMAL
Tokens: 250
overall tok/s: 23.12
steady tok/s: 56.23
lateSteady tok/s: 63.78
Native decode: 249 tokens in 4428 ms, 56.2 tok/s
Target: 100 steady tok/s
```

Do not run a new benchmark until the full test gate is green.

Current candidate patches that need validation:

1. DSP frozen output verification and `dspValidateOutputs` changes.
2. Training DSP mutable external input handling and one-plan-per-shape semantics.
3. `meanSqErr` replay-friendly denominator changes.
4. VLM training pipeline test geometry/config fixes.
5. Adam8bit convergence label-mapping fix.
6. CUDA non-peer relocation/failover changes in `CudaZeroHandler` and `SynchronousFlowController`.
7. Random generator layout and random op copy fixes.
8. CUDA shuffle JNI reachability fences.
9. TAD trie key/stride fix.
10. NPY validation last-error clearing.
11. Multi-backend workspace spill metric/accounting fix.

Known failures to assign to subagents:

1. `TestVlmTrainingPipeline`: `imageResolution (384) must be divisible by patchSize (14)`.
2. `MmulMixedPrecisionRegressionTest#testTop5TokenAgreement`: CUDA failover path reached unsupported relocation.
3. `MmulMixedPrecisionRegressionTest#testVocabProjectionArgmaxStabilityOverSteps`: same relocation/failover class.
4. `TestAdam8bit#testAdam8bitConvergence`: label mapping mismatch after labels were marked unused.
5. Random/native crashes:
   - `TestRandomOpValidation#testRandomExponential2`
   - `RngTests#testRandomBinomial`
   - `DataSetTest#testSplitTestAndTrainRng`
   - `SpecialTests#testScalarShuffle2`
   - `RandomTests#testGaussianDistribution3`
   - `ShufflesTests#testSymmetricShuffle1`
6. `CrossDeviceTransferTest`: crash around `cublasSetStream_v2`.
7. `TestTensorAlongDimension#testTadShapes2d`: offset mismatch.
8. `OpExecutionerTests#testMeanSumSimple`: expected `[256]`, actual `[16]`; previous read-only investigation suggested the actual sum may be correct and expected scalar suspicious. Verify, do not assume.
9. `ValidationUtilTests#testNpyValidation`: expected true but false.
10. `MultiBackendWorkspaceIntegrationTest#testSpillMetrics`: spill metric assertion and later double-free/corruption.
11. `TestGenerationPipelineBenchmarkAccuracy` or related generation pipeline test: native abort with `dspPublishThreadCompletionEvent: cudaEventRecord failed: an illegal memory access was encountered (700)`.
12. `TestQwen35Pipeline`: `corrupted size vs. prev_size`.
13. DL4J core assertion failures:
   - `TestLrChanges#testChangeLrCompGraphSchedule`
   - `DenseTest#testMLPMultiLayerPretrain`

Required build commands:

For native/CUDA changes, use exactly:

```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn -Pcuda -Dlibnd4j.triton=ON -Dlibnd4j.chip=cuda -Dlibnd4j.buildthreads=12 -Dlibnd4j.log=libnd4j-build.log -pl libnd4j,:nd4j-cuda-12.9 clean install -DskipTests 2>&1 | tee /tmp/dl4j-regression-cuda-build.log
```

For Java-only CUDA backend changes when no native code changed, use:

```bash
/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl :nd4j-cuda-12.9 2>&1 | tee /tmp/dl4j-regression-nd4j-cuda-install.log
```

For Java-only API changes when no native code changed, build the specific module with full Maven path and `install -DskipTests`, with output captured through `tee`.

Targeted test commands:

Run from `platform-tests` only. Every command must capture output through `tee`.

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestVlmTrainingPipeline 2>&1 | tee /tmp/test-vlm-training-pipeline.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestAdam8bit#testAdam8bitConvergence 2>&1 | tee /tmp/test-adam8bit-convergence.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MmulMixedPrecisionRegressionTest#testTop5TokenAgreement 2>&1 | tee /tmp/test-mmul-top5-token-agreement.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MmulMixedPrecisionRegressionTest#testVocabProjectionArgmaxStabilityOverSteps 2>&1 | tee /tmp/test-mmul-vocab-projection.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=CrossDeviceTransferTest 2>&1 | tee /tmp/test-cross-device-transfer.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestRandomOpValidation#testRandomExponential2,RngTests#testRandomBinomial,RandomTests#testGaussianDistribution3,SpecialTests#testScalarShuffle2,ShufflesTests#testSymmetricShuffle1,DataSetTest#testSplitTestAndTrainRng 2>&1 | tee /tmp/test-random-native-crashes.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestTensorAlongDimension#testTadShapes2d 2>&1 | tee /tmp/test-tad-shapes-2d.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=OpExecutionerTests#testMeanSumSimple 2>&1 | tee /tmp/test-op-executioner-mean-sum-simple.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=ValidationUtilTests#testNpyValidation 2>&1 | tee /tmp/test-npy-validation.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MultiBackendWorkspaceIntegrationTest#testSpillMetrics 2>&1 | tee /tmp/test-workspace-spill-metrics.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestGenerationPipelineBenchmarkAccuracy,TestGenerationPipelineBenchmarkInputPipeline,TestQwen35Pipeline 2>&1 | tee /tmp/test-generation-and-qwen35.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=TestLrChanges#testChangeLrCompGraphSchedule,DenseTest#testMLPMultiLayerPretrain 2>&1 | tee /tmp/test-dl4j-core-regressions.log
```

Full regression gate:

After targeted tests pass, run the full platform test gate:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 2>&1 | tee /tmp/dl4j-regression-full-platform-tests.log
```

Only after the full gate passes, run the required benchmark:

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && ./run-benchmark.sh --tokens 250 2>&1 | tee /tmp/dl4j-regression-benchmark-250.log
```

Orchestration workflow:

1. Check active processes and current `git status --short`.
2. Read the old full-test log only as failure discovery, not as validation of the current worktree.
3. Divide failures by subsystem and dispatch subagents with non-overlapping file scopes.
4. Require each subagent to report root cause, exact files changed, exact command run, log path, and pass/fail result.
5. For native changes, run the required CUDA native build before targeted tests.
6. For Java-only changes, run the narrow module install before targeted tests.
7. Do not combine unrelated fixes into one commit candidate.
8. If a candidate patch fails, fix the root cause or undo only the exact lines from that candidate by editing.
9. Record each successful milestone using Kompile test milestones/memory.
10. When targeted tests and full platform tests pass, run the 250-token benchmark and compare `steady tok/s` and `lateSteady tok/s` against the baseline.
11. Commit only atomic, fully tested improvements with the tokens-per-second data in the commit message.

Success criteria:

1. No active untracked/orphaned Maven test is mistaken for validation.
2. All targeted failures above pass.
3. Full `platform-tests` with CUDA backend passes.
4. No Java FQCN style violations are introduced.
5. No banned commands are used by the orchestrator or any subagent.
6. No workaround, fallback, test disablement, or benchmark shortcut is accepted.
7. Benchmark is run with exactly 250 tokens only after tests pass.
