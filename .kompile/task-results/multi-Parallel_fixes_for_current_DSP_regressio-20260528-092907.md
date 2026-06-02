# Multi-Task Results: Parallel fixes for current DSP regression gate blockers

**Subtasks:** 4

---

## fix-vlm-training-config-validation (qwen)

Subagent 'qwen' exited with code 1 after 14.7s

## Summary
Qwen OAuth free tier was discontinued on 2026-04-15. Run /auth to switch to Coding Plan, OpenRouter, Fireworks AI, or another provider.

**Full output (136 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/qwen-20260528-091144.md`
Use the `read` tool to access the full result if needed.

---

## fix-cuda-non-peer-relocation (codex)

Subagent 'codex' completed in 457.5s

## Summary
Reading additional input from stdin...
OpenAI Codex v0.134.0
--------
workdir: /home/agibsonccc/Documents/GitHub/deeplearning4j
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: none
reasoning summaries: none
session id: 019e6bec-09b8-70f0-8fd0-3fbea61d3410
--------
user
Fix the non-peer failover relocation failure in CUDA memory handling.

CRITICAL RULES — READ BEFORE DOING ANYTHING:
1. NEVER run: git checkout <file>, git stash, git reset --hard, git clean
2. NEVER run: ccache -C or ccache --clear (destroys 2+ hours of compiled cache)
3. NEVER run: make directly (always use full mvn with libnd4j + bindings module)
4. NEVER pipe build or test output through tail (use tee, read the tee log)

... (579831 chars total) ...

/home/agibsonccc/dev-apps/mvn/bin/mvn install -DskipTests -pl :nd4j-cuda-12.9 2>&1 | tee /tmp/fix-cuda-relocation-build.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MmulMixedPrecisionRegressionTest#testTop5TokenAgreement 2>&1 | tee /tmp/fix-cuda-relocation-top5.log
```

```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 -Dtest=MmulMixedPrecisionRegressionTest#testVocabProjectionArgmaxStabilityOverSteps 2>&1 | tee /tmp/fix-cuda-relocation-vocab.log
```

**Full output (579831 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/codex-20260528-091907.md`
Use the `read` tool to access the full result if needed.

---

## fix-adam8bit-label-mapping (claude)

Subagent 'claude' completed in 209.1s

## Summary
There is an active Maven/ForkedBooter process running the full platform-tests suite. I cannot run the targeted test now.

**Fix applied:** Changed `TestAdam8bit.java:163` from `new DataSet(dummyInput, dummyInput)` to `new DataSet(dummyInput, null)`.

**Root cause:** The test uses `markLabelsUnused()` in its TrainingConfig (line 157), indicating this is unsupervised optimization (minimizing x^2, no label data needed). However, it was constructing a `DataSet(dummyInput, dummyInput)` — passing the same dummy scalar as both features and labels. This creates a DataSet with 1 label array.

At `SameDiff.java:2396`, `fitHelper` validates that `labelMapping.size() == ds.numLabelsArrays()`. Since `markLabelsUnused()` leaves labelMapping null/empty (size 0) but the DataSet has 1 label array, the check fails with: `"number of dataset label mapping variables in training configuration (0) must match number of dataset label arrays (1)"`.

Note: `markLabelsUnused()` only suppresses the builder-time validation at `TrainingConfig.java:630`, not the runtime check in `fitHelper`. This is correct behavior — the runtime should enforce consistency between the config and the actual data. The bug is in the test passing labels when it declared them unused.

**Pending verification:** The active full platform-tests run is in progress. Once it finishes, run:
```bash
cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
  /home/agibsonccc/dev-apps/mvn/bin/mvn test -Dbackend.artifactId=nd4j-cuda-12.9 \
  -Dtest=TestAdam8bit#testAdam8bitConvergence \
  2>&1 | tee /tmp/fix-adam8bit-label-mapping.log
```

**Full output (1616 chars) written to:** `/home/agibsonccc/Documents/GitHub/deeplearning4j/.kompile/task-results/claude-20260528-091458.md`
Use the `read` tool to access the full result if needed.

---

## fix-random-exponential-native-crash (gemini)

**Timed out** after 10 minutes.

---

**Summary:** 3/4 subtasks completed successfully.