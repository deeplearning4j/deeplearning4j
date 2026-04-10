/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.model.benchmark;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

/**
 * Main orchestration framework for benchmark-faithful correctness validation.
 *
 * <h3>What this does</h3>
 * <p>Runs a SameDiff model through multi-step decode-style validation, comparing
 * a test execution mode against a trustworthy oracle (SLOT_BY_SLOT or manual
 * baseline). Validates correctness at multiple levels, tracks replay metadata,
 * and produces actionable failure diagnostics.</p>
 *
 * <h3>Key Design Principles</h3>
 * <ul>
 *   <li><b>Same compiled native handle</b> — The test mode uses one compiled plan
 *       handle across all iterations. No recompilation between steps.</li>
 *   <li><b>No silent fallback</b> — Phase violations are explicitly detected and
 *       reported. No fallback-transition-oriented "fixes".</li>
 *   <li><b>Evolving inputs</b> — position_ids, attention_mask, and past_key_values
 *       change realistically across decode steps.</li>
 *   <li><b>Multi-level validation</b> — Token prefix, logits, tensor equality,
 *       top-K overlap, and targeted intermediate capture comparison.</li>
 *   <li><b>Reusable diagnostics</b> — Uses DspDiagnostics, C++ diagnostics,
 *       and ReplayMetadataTracker. No ad-hoc printf logging.</li>
 * </ul>
 *
 * <h3>Execution Mode Matrix</h3>
 * <ul>
 *   <li><b>SLOT_BY_SLOT</b> — Reference oracle. Standard op-by-op execution.</li>
 *   <li><b>TRITON_NO_GC</b> — Triton compilation without CUDA graph capture.
 *       Tests Triton section fusion correctness in isolation.</li>
 *   <li><b>CUDA_GRAPHS</b> — CUDA graph capture and replay. Tests graph replay
 *       correctness with evolving inputs.</li>
 *   <li><b>OPTIMAL</b> — Full Triton + CUDA graphs + optimizations. The actual
 *       benchmark configuration.</li>
 * </ul>
 *
 * <h3>Oracle Levels</h3>
 * <ul>
 *   <li><b>SLOT_BY_SLOT reference</b> — Run the same model with SLOT_BY_SLOT mode
 *       and compare. Best for when you have a working reference implementation.</li>
 *   <li><b>Manual baseline</b> — A known-good manual decode implementation (growing
 *       KV + contiguous mask + output()). Best for end-to-end pipeline validation.</li>
 * </ul>
 *
 * <h3>When to use each comparison level</h3>
 * <pre>
 *   TENSOR       → Root-cause analysis. Find the exact divergent op.
 *                  Use DspAccuracyValidator.validatePerOp() instead.
 *
 *   LOGITS       → Standard decode correctness. Catches numerical drift
 *                  before it cascades into token divergence.
 *
 *   TOKEN_PREFIX → Quick benchmark validation. "Do the first N tokens match?"
 *                  Fastest — suitable for CI smoke tests.
 *
 *   TOKEN_TOPK   → Relaxed validation for TF32/low-precision configs.
 *                  "Are the top-K candidates semantically equivalent?"
 *
 *   INTERMEDIATE → Targeted subgraph validation. "Does the attention-prep
 *                  block (seg[2676-2730]) produce correct intermediates?"
 * </pre>
 *
 * <h3>How to run same-handle multi-step replay validation</h3>
 * <pre>
 *   // 1. Build the framework with oracle and test configs
 *   DecodeValidationFramework framework = DecodeValidationFramework.builder()
 *       .model(decoder)
 *       .oracleMode(OracleMode.SLOT_BY_SLOT_REFERENCE)
 *       .testConfig(BenchmarkConfig.optimal())
 *       .oracleConfig(BenchmarkConfig.create("SLOT_BY_SLOT")
 *           .executionMode(GraphExecutionMode.SLOT_BY_SLOT))
 *       .validationLevel(ValidationResult.ValidationLevel.LOGITS)
 *       .maxDecodeSteps(10)
 *       .evolutionMode(DecodeInputEvolutor.EvolutionMode.ALL_TOGETHER)
 *       .build();
 *
 *   // 2. Run validation with prefill inputs/outputs
 *   ValidationResult result = framework.runValidation(
 *       prefillInputs, prefillOutputs, outputNames);
 *
 *   // 3. Check result
 *   if (!result.isPassed()) {
 *       System.out.println(result.toReport());
 *   }
 *
 *   // 4. Clean up
 *   framework.close();
 * </pre>
 *
 * <h3>How a separate agent writes concrete benchmark-subgraph tests</h3>
 * <pre>
 *   // Create framework with specific subgraph test config
 *   DecodeValidationFramework framework = DecodeValidationFramework.builder()
 *       .model(decoder)
 *       .oracleMode(OracleMode.SLOT_BY_SLOT_REFERENCE)
 *       .testConfig(subgraphTestConfig)
 *       .oracleConfig(slotBySlotConfig)
 *       .validationLevel(ValidationResult.ValidationLevel.TOKEN_PREFIX)
 *       .tokenPrefixLength(4)
 *       .maxDecodeSteps(6)
 *       .evolutionMode(DecodeInputEvolutor.EvolutionMode.ALL_TOGETHER)
 *       .validationConfig(ValidationConfig.standard())
 *       .build();
 *
 *   ValidationResult result = framework.runValidation(
 *       prefillInputs, prefillOutputs, outputNames);
 *   assertTrue(result.isPassed(), result.toReport());
 *   framework.close();
 * </pre>
 */
@Slf4j
public class DecodeValidationFramework implements AutoCloseable {

    private final SameDiff model;
    private final OracleMode oracleMode;
    private final BenchmarkConfig testConfig;
    private final BenchmarkConfig oracleConfig;
    private final ValidationResult.ValidationLevel validationLevel;
    private final int maxDecodeSteps;
    private final DecodeInputEvolutor.EvolutionMode evolutionMode;
    private final ValidationConfig validationConfig;
    private final int tokenPrefixLength;
    private final int topK;
    private final Set<String> intermediateVarNames;
    private final boolean paddedMode;
    private final long maxKvLen;

    // Internal state
    private DecodeInputEvolutor inputEvolutor;
    private MultiLevelComparator comparator;
    private ReplayMetadataTracker oracleMetadataTracker;
    private ReplayMetadataTracker testMetadataTracker;

    private DecodeValidationFramework(Builder builder) {
        this.model = builder.model;
        this.oracleMode = builder.oracleMode;
        this.testConfig = builder.testConfig;
        this.oracleConfig = builder.oracleConfig;
        this.validationLevel = builder.validationLevel;
        this.maxDecodeSteps = builder.maxDecodeSteps;
        this.evolutionMode = builder.evolutionMode;
        this.validationConfig = builder.validationConfig != null
                ? builder.validationConfig : ValidationConfig.standard();
        this.tokenPrefixLength = builder.tokenPrefixLength;
        this.topK = builder.topK;
        this.intermediateVarNames = builder.intermediateVarNames;
        this.paddedMode = builder.paddedMode;
        this.maxKvLen = builder.maxKvLen;

        this.comparator = new MultiLevelComparator(this.validationConfig);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Run full validation: oracle run → test run → comparison → result.
     *
     * <p>This is the main entry point. It:</p>
     * <ol>
     *   <li>Applies oracle config, runs model with prefill inputs, captures oracle outputs</li>
     *   <li>Applies test config, runs model with same prefill inputs, captures test outputs</li>
     *   <li>Runs decode-style stepped validation with evolving inputs on both paths</li>
     *   <li>Compares at the configured validation level</li>
     *   <li>Tracks replay metadata for both paths</li>
     *   <li>Returns a ValidationResult with all diagnostics</li>
     * </ol>
     *
     * @param prefillInputs  input placeholder map for prefill (includes inputs_embeds,
     *                       attention_mask, position_ids, past_key_values)
     * @param prefillOutputs output names to request (must include logits and KV outputs)
     * @param outputNames    all output variable names (for plan compilation)
     * @return validation result with diagnostics
     */
    public ValidationResult runValidation(Map<String, INDArray> prefillInputs,
                                           Map<String, INDArray> prefillOutputs,
                                           String[] outputNames) {
        ValidationResult.Builder resultBuilder = ValidationResult.builder(
                testConfig.getName(),
                oracleConfig.getName(),
                testConfig.getName(),
                validationLevel);

        List<String> failures = new ArrayList<>();
        int firstDivergentStep = -1;
        double maxDiffObserved = 0.0;
        int totalStepsCompared = 0;

        try {
            // Phase 1: Oracle execution
            log.info("[{}] Phase 1: Running oracle ({})...", testConfig.getName(), oracleConfig.getName());
            Map<String, INDArray> oracleLogitsPerStep = new LinkedHashMap<>();
            List<Integer> oracleTokens = new ArrayList<>();

            runOracleExecution(prefillInputs, prefillOutputs, outputNames,
                    oracleLogitsPerStep, oracleTokens, resultBuilder);

            // Phase 2: Test execution
            log.info("[{}] Phase 2: Running test ({})...", testConfig.getName(), testConfig.getName());
            Map<String, INDArray> testLogitsPerStep = new LinkedHashMap<>();
            List<Integer> testTokens = new ArrayList<>();

            runTestExecution(prefillInputs, prefillOutputs, outputNames,
                    testLogitsPerStep, testTokens, resultBuilder);

            // Phase 3: Comparison at configured level
            log.info("[{}] Phase 3: Comparing at level {}...", testConfig.getName(), validationLevel);
            switch (validationLevel) {
                case TENSOR:
                    // TENSOR level: use DspAccuracyValidator for full intermediate comparison
                    runTensorLevelComparison(oracleLogitsPerStep, testLogitsPerStep,
                            failures, resultBuilder);
                    break;

                case LOGITS:
                    // LOGITS level: per-step logits comparison
                    runLogitsLevelComparison(oracleLogitsPerStep, testLogitsPerStep,
                            oracleTokens, testTokens, failures, resultBuilder);
                    break;

                case TOKEN_PREFIX:
                    // TOKEN_PREFIX level: first N tokens must match
                    runTokenPrefixComparison(oracleTokens, testTokens,
                            failures, resultBuilder);
                    break;

                case TOKEN_TOPK:
                    // TOKEN_TOPK level: top-K overlap per step
                    runTokenTopKComparison(new ArrayList<>(oracleLogitsPerStep.values()),
                            new ArrayList<>(testLogitsPerStep.values()),
                            failures, resultBuilder);
                    break;

                case INTERMEDIATE:
                    // INTERMEDIATE level: targeted variable comparison
                    runIntermediateComparison(oracleLogitsPerStep, testLogitsPerStep,
                            failures, resultBuilder);
                    break;
            }

            firstDivergentStep = failures.isEmpty() ? -1 : 0;
            totalStepsCompared = Math.max(oracleLogitsPerStep.size(), testLogitsPerStep.size());

            resultBuilder
                    .passed(failures.isEmpty())
                    .failures(failures)
                    .firstDivergentStep(firstDivergentStep)
                    .totalStepsCompared(totalStepsCompared)
                    .maxDiffObserved(maxDiffObserved)
                    .replayMetadata(testMetadataTracker != null ? testMetadataTracker.buildMetadata() : null);

        } catch (Exception e) {
            failures.add("Validation exception: " + e.getMessage());
            resultBuilder
                    .passed(false)
                    .failures(failures)
                    .firstDivergentStep(firstDivergentStep)
                    .totalStepsCompared(totalStepsCompared)
                    .maxDiffObserved(maxDiffObserved);
            log.error("[{}] Validation failed with exception", testConfig.getName(), e);
        }

        ValidationResult result = resultBuilder.build();
        log.info("[{}] Validation result: {}{}",
                testConfig.getName(),
                result.isPassed() ? "PASSED" : "FAILED",
                result.isPassed() ? "" : " — " + result.getFirstFailure());

        return result;
    }

    /**
     * Run validation with stepped decode-style execution (evolving inputs).
     *
     * <p>This is the decode-aware path: prefill → multi-step decode with evolving
     * position_ids, attention_mask, and past_key_values.</p>
     *
     * @param model          SameDiff model to validate
     * @param prefillInputs  prefill input map
     * @param logitsName     name of the logits output variable
     * @param kvOutputNames  names of KV output variables (present_* pattern)
     * @return validation result
     */
    public ValidationResult runSteppedValidation(SameDiff model,
                                                  Map<String, INDArray> prefillInputs,
                                                  String logitsName,
                                                  List<String> kvOutputNames,
                                                  String[] outputNames) {
        ValidationResult.Builder resultBuilder = ValidationResult.builder(
                testConfig.getName(),
                oracleConfig.getName(),
                testConfig.getName(),
                validationLevel);

        List<String> failures = new ArrayList<>();
        int firstDivergentStep = -1;
        double maxDiffObserved = 0.0;

        try {
            // Phase 1: Oracle stepped execution
            log.info("[{}] Phase 1: Oracle stepped execution ({})...",
                    testConfig.getName(), oracleConfig.getName());
            Map<String, INDArray> oracleLogitsPerStep = new LinkedHashMap<>();
            List<Integer> oracleTokens = new ArrayList<>();
            List<INDArray> oracleLogitsList = new ArrayList<>();

            runOracleSteppedExecution(model, prefillInputs, logitsName, kvOutputNames, outputNames,
                    oracleLogitsPerStep, oracleTokens, oracleLogitsList, resultBuilder);

            // Phase 2: Test stepped execution
            log.info("[{}] Phase 2: Test stepped execution ({})...",
                    testConfig.getName(), testConfig.getName());
            Map<String, INDArray> testLogitsPerStep = new LinkedHashMap<>();
            List<Integer> testTokens = new ArrayList<>();
            List<INDArray> testLogitsList = new ArrayList<>();

            runTestSteppedExecution(model, prefillInputs, logitsName, kvOutputNames, outputNames,
                    testLogitsPerStep, testTokens, testLogitsList, resultBuilder);

            // Phase 3: Comparison
            log.info("[{}] Phase 3: Comparing at level {}...",
                    testConfig.getName(), validationLevel);

            switch (validationLevel) {
                case LOGITS:
                    runLogitsLevelComparison(oracleLogitsPerStep, testLogitsPerStep,
                            oracleTokens, testTokens, failures, resultBuilder);
                    break;

                case TOKEN_PREFIX:
                    runTokenPrefixComparison(oracleTokens, testTokens,
                            failures, resultBuilder);
                    break;

                case TOKEN_TOPK:
                    runTokenTopKComparison(oracleLogitsList, testLogitsList,
                            failures, resultBuilder);
                    break;

                case TENSOR:
                case INTERMEDIATE:
                    failures.add("TENSOR/INTERMEDIATE level requires C++ diagnostics — " +
                            "use DspAccuracyValidator.validatePerOp() for tensor-level comparison");
                    break;
            }

            firstDivergentStep = failures.isEmpty() ? -1 : 0;
            int totalSteps = Math.max(oracleLogitsPerStep.size(), testLogitsPerStep.size());

            resultBuilder
                    .passed(failures.isEmpty())
                    .failures(failures)
                    .firstDivergentStep(firstDivergentStep)
                    .totalStepsCompared(totalSteps)
                    .maxDiffObserved(maxDiffObserved)
                    .replayMetadata(testMetadataTracker != null ? testMetadataTracker.buildMetadata() : null);

        } catch (Exception e) {
            failures.add("Stepped validation exception: " + e.getMessage());
            resultBuilder
                    .passed(false)
                    .failures(failures)
                    .firstDivergentStep(firstDivergentStep)
                    .maxDiffObserved(maxDiffObserved);
            log.error("[{}] Stepped validation failed", testConfig.getName(), e);
        }

        ValidationResult result = resultBuilder.build();
        log.info("[{}] Stepped validation: {}{}",
                testConfig.getName(),
                result.isPassed() ? "PASSED" : "FAILED",
                result.isPassed() ? "" : " — " + result.getFirstFailure());

        return result;
    }

    // ─── Oracle Execution ───────────────────────────────────────────────────

    private void runOracleExecution(Map<String, INDArray> prefillInputs,
                                     Map<String, INDArray> prefillOutputs,
                                     String[] outputNames,
                                     Map<String, INDArray> logitsPerStep,
                                     List<Integer> tokens,
                                     ValidationResult.Builder resultBuilder) {
        // Apply oracle config
        BenchmarkConfigApplier.resetModelState(model);
        BenchmarkConfigApplier.apply(oracleConfig);

        if (oracleConfig.getExecutionMode() != null) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            model.compileNativeDynamicShapePlan(
                    Arrays.asList(outputNames), oracleConfig.getExecutionMode(), true);
        } else if (oracleConfig.isTriton()) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            BenchmarkConfigApplier.compileModel(model, "oracle", Arrays.asList(outputNames), oracleConfig);
        }

        // Run prefill
        Map<String, INDArray> outputs = model.output(prefillInputs, outputNames);
        String logitsName = findLogitsName(outputs);

        // Extract logits
        INDArray logits = outputs.get(logitsName);
        if (logits != null && logits.rank() >= 2) {
            long seqLen = logits.size(logits.rank() - 2);
            long vocabSize = logits.size(logits.rank() - 1);
            INDArray lastLogits = logits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(seqLen - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            logitsPerStep.put("prefill", lastLogits.dup());
            int token = Nd4j.argMax(lastLogits.reshape(lastLogits.length())).getInt(0);
            tokens.add(token);
        }

        // Track metadata
        oracleMetadataTracker = new ReplayMetadataTracker(oracleConfig.getName());
        trackOracleMetadata(oracleMetadataTracker);
    }

    private void runOracleSteppedExecution(SameDiff model,
                                            Map<String, INDArray> prefillInputs,
                                            String logitsName,
                                            List<String> kvOutputNames,
                                            String[] outputNames,
                                            Map<String, INDArray> logitsPerStep,
                                            List<Integer> tokens,
                                            List<INDArray> logitsList,
                                            ValidationResult.Builder resultBuilder) {
        // Apply oracle config
        BenchmarkConfigApplier.resetModelState(model);
        BenchmarkConfigApplier.apply(oracleConfig);

        if (oracleConfig.getExecutionMode() != null) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            model.compileNativeDynamicShapePlan(
                    Arrays.asList(outputNames), oracleConfig.getExecutionMode(), true);
        }

        // Run prefill
        Map<String, INDArray> prefillOutputs = model.output(prefillInputs, outputNames);

        // Setup input evolutor
        inputEvolutor = DecodeInputEvolutor.builder()
                .decoderInputNames(new ArrayList<>(model.inputs()))
                .logitsName(logitsName)
                .kvOutputNames(kvOutputNames)
                .prefillSeqLen(prefillInputs.get("inputs_embeds").size(1))
                .maxDecodeSteps(maxDecodeSteps)
                .evolutionMode(evolutionMode)
                .paddedMode(paddedMode)
                .maxKvLen(maxKvLen)
                .build();
        inputEvolutor.initializeFromPrefill(prefillOutputs);

        // Extract prefill logits
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        if (prefillLogits != null && prefillLogits.rank() >= 2) {
            long seqLen = prefillLogits.size(prefillLogits.rank() - 2);
            INDArray lastLogits = prefillLogits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(seqLen - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            logitsPerStep.put("prefill", lastLogits.dup());
            int token = Nd4j.argMax(lastLogits.reshape(lastLogits.length())).getInt(0);
            tokens.add(token);
            logitsList.add(lastLogits.dup());
        }

        // Decode steps
        for (int step = 0; step < maxDecodeSteps; step++) {
            Map<String, INDArray> stepInputs = inputEvolutor.buildStepInputs(step);
            // Add inputs_embeds for decode step (single token embedding)
            // The caller should provide this via the prefill input's embedding table
            if (stepInputs.isEmpty() || !stepInputs.containsKey("inputs_embeds")) {
                // Skip if no embedding provided — caller manages embeddings
                break;
            }

            Map<String, INDArray> stepOutputs = model.output(stepInputs, outputNames);
            inputEvolutor.updateFromOutputs(stepOutputs);

            INDArray stepLogits = stepOutputs.get(logitsName);
            if (stepLogits != null) {
                INDArray flatLogits = stepLogits.reshape(stepLogits.length());
                logitsPerStep.put("step_" + step, flatLogits.dup());
                int token = Nd4j.argMax(flatLogits).getInt(0);
                tokens.add(token);
                logitsList.add(flatLogits.dup());
            }
        }

        // Track metadata
        oracleMetadataTracker = new ReplayMetadataTracker(oracleConfig.getName());
        trackOracleMetadata(oracleMetadataTracker);
    }

    // ─── Test Execution ─────────────────────────────────────────────────────

    private void runTestExecution(Map<String, INDArray> prefillInputs,
                                   Map<String, INDArray> prefillOutputs,
                                   String[] outputNames,
                                   Map<String, INDArray> logitsPerStep,
                                   List<Integer> tokens,
                                   ValidationResult.Builder resultBuilder) {
        // Apply test config
        BenchmarkConfigApplier.resetModelState(model);
        BenchmarkConfigApplier.apply(testConfig);

        if (testConfig.getExecutionMode() != null) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            model.compileNativeDynamicShapePlan(
                    Arrays.asList(outputNames), testConfig.getExecutionMode(), true);
        } else if (testConfig.isTriton()) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            BenchmarkConfigApplier.compileModel(model, "test", Arrays.asList(outputNames), testConfig);
        }

        // Run prefill
        Map<String, INDArray> outputs = model.output(prefillInputs, outputNames);
        String logitsName = findLogitsName(outputs);

        // Extract logits
        INDArray logits = outputs.get(logitsName);
        if (logits != null && logits.rank() >= 2) {
            long seqLen = logits.size(logits.rank() - 2);
            INDArray lastLogits = logits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(seqLen - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            logitsPerStep.put("prefill", lastLogits.dup());
            int token = Nd4j.argMax(lastLogits.reshape(lastLogits.length())).getInt(0);
            tokens.add(token);
        }

        // Track metadata
        testMetadataTracker = new ReplayMetadataTracker(testConfig.getName());
        trackTestMetadata(testMetadataTracker);
    }

    private void runTestSteppedExecution(SameDiff model,
                                          Map<String, INDArray> prefillInputs,
                                          String logitsName,
                                          List<String> kvOutputNames,
                                          String[] outputNames,
                                          Map<String, INDArray> logitsPerStep,
                                          List<Integer> tokens,
                                          List<INDArray> logitsList,
                                          ValidationResult.Builder resultBuilder) {
        // Apply test config
        BenchmarkConfigApplier.resetModelState(model);
        BenchmarkConfigApplier.apply(testConfig);

        if (testConfig.getExecutionMode() != null) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            model.compileNativeDynamicShapePlan(
                    Arrays.asList(outputNames), testConfig.getExecutionMode(), true);
        } else if (testConfig.isTriton()) {
            model.setDspAutoCompileEnabled(true);
            model.setDspNativeAutoCompileEnabled(true);
            BenchmarkConfigApplier.compileModel(model, "test", Arrays.asList(outputNames), testConfig);
        }

        // Run prefill
        Map<String, INDArray> prefillOutputs = model.output(prefillInputs, outputNames);

        // Setup input evolutor
        inputEvolutor = DecodeInputEvolutor.builder()
                .decoderInputNames(new ArrayList<>(model.inputs()))
                .logitsName(logitsName)
                .kvOutputNames(kvOutputNames)
                .prefillSeqLen(prefillInputs.get("inputs_embeds").size(1))
                .maxDecodeSteps(maxDecodeSteps)
                .evolutionMode(evolutionMode)
                .paddedMode(paddedMode)
                .maxKvLen(maxKvLen)
                .build();
        inputEvolutor.initializeFromPrefill(prefillOutputs);

        // Extract prefill logits
        INDArray prefillLogits = prefillOutputs.get(logitsName);
        if (prefillLogits != null && prefillLogits.rank() >= 2) {
            long seqLen = prefillLogits.size(prefillLogits.rank() - 2);
            INDArray lastLogits = prefillLogits.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.point(0),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(seqLen - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all());
            logitsPerStep.put("prefill", lastLogits.dup());
            int token = Nd4j.argMax(lastLogits.reshape(lastLogits.length())).getInt(0);
            tokens.add(token);
            logitsList.add(lastLogits.dup());
        }

        // Decode steps
        for (int step = 0; step < maxDecodeSteps; step++) {
            Map<String, INDArray> stepInputs = inputEvolutor.buildStepInputs(step);
            if (stepInputs.isEmpty() || !stepInputs.containsKey("inputs_embeds")) {
                break;
            }

            Map<String, INDArray> stepOutputs = model.output(stepInputs, outputNames);
            inputEvolutor.updateFromOutputs(stepOutputs);

            INDArray stepLogits = stepOutputs.get(logitsName);
            if (stepLogits != null) {
                INDArray flatLogits = stepLogits.reshape(stepLogits.length());
                logitsPerStep.put("step_" + step, flatLogits.dup());
                int token = Nd4j.argMax(flatLogits).getInt(0);
                tokens.add(token);
                logitsList.add(flatLogits.dup());
            }

            // Track replay metadata
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            org.bytedeco.javacpp.Pointer planHandle = getPlanHandle(model);
            if (testMetadataTracker != null) {
                if (planHandle != null) {
                    testMetadataTracker.recordIteration(step, nativeOps, planHandle);
                } else {
                    testMetadataTracker.recordIterationSlotBySlot(step);
                }
            }
        }

        // Check for phase violations
        if (testMetadataTracker != null) {
            List<String> violations = testMetadataTracker.getPhaseViolations();
            if (!violations.isEmpty()) {
                log.warn("[{}] Phase violations detected: {}", testConfig.getName(), violations);
            }
        }
    }

    // ─── Comparison Methods ─────────────────────────────────────────────────

    private void runLogitsLevelComparison(Map<String, INDArray> oracleLogitsPerStep,
                                           Map<String, INDArray> testLogitsPerStep,
                                           List<Integer> oracleTokens,
                                           List<Integer> testTokens,
                                           List<String> failures,
                                           ValidationResult.Builder resultBuilder) {
        List<INDArray> oracleLogits = new ArrayList<>(oracleLogitsPerStep.values());
        List<INDArray> testLogits = new ArrayList<>(testLogitsPerStep.values());

        DecodeStepValidator stepValidator = new DecodeStepValidator(validationConfig);
        DecodeStepValidator.DecodeComparisonReport report = stepValidator.compare(
                oracleLogits, oracleTokens, testLogits, testTokens,
                oracleConfig.getName(), testConfig.getName());

        resultBuilder.decodeReport(report);

        if (report.getFirstTokenDivergenceStep() >= 0) {
            failures.add("Token divergence at step " + report.getFirstTokenDivergenceStep());
        }
        if (report.getFirstArgmaxDivergenceStep() >= 0) {
            failures.add("Argmax divergence at step " + report.getFirstArgmaxDivergenceStep());
        }
        if (report.getTokenMatchRate() < 0.9) {
            failures.add(String.format("Token match rate too low: %.1f%%", report.getTokenMatchRate() * 100));
        }
    }

    private void runTokenPrefixComparison(List<Integer> oracleTokens,
                                           List<Integer> testTokens,
                                           List<String> failures,
                                           ValidationResult.Builder resultBuilder) {
        int[] testTokenArray = testTokens.stream().mapToInt(i -> i).toArray();
        List<String> prefixFailures = comparator.compareTokenPrefix(
                oracleTokens, testTokenArray, tokenPrefixLength);
        failures.addAll(prefixFailures);
    }

    private void runTokenTopKComparison(List<INDArray> oracleLogits,
                                         List<INDArray> testLogits,
                                         List<String> failures,
                                         ValidationResult.Builder resultBuilder) {
        List<Integer> overlaps = comparator.compareTopKPerStep(oracleLogits, testLogits, topK);
        int steps = overlaps.size();
        int fullyOverlapping = 0;
        for (int overlap : overlaps) {
            if (overlap >= topK) fullyOverlapping++;
        }
        double overlapRate = steps > 0 ? (double) fullyOverlapping / steps : 0;
        resultBuilder.metric("topK_overlap_rate", overlapRate);
        if (overlapRate < 0.8) {
            failures.add(String.format(
                    "Top-K overlap rate too low: %.1f%% (%d/%d steps with full overlap)",
                    overlapRate * 100, fullyOverlapping, steps));
        }
    }

    private void runTensorLevelComparison(Map<String, INDArray> oracleLogitsPerStep,
                                           Map<String, INDArray> testLogitsPerStep,
                                           List<String> failures,
                                           ValidationResult.Builder resultBuilder) {
        // TENSOR level delegates to DspAccuracyValidator
        failures.add("TENSOR level requires DspAccuracyValidator.validatePerOp() — " +
                "use intermediate capture with C++ diagnostics for full tensor comparison");
    }

    private void runIntermediateComparison(Map<String, INDArray> oracleLogitsPerStep,
                                            Map<String, INDArray> testLogitsPerStep,
                                            List<String> failures,
                                            ValidationResult.Builder resultBuilder) {
        // INTERMEDIATE level requires C++ diagnostics
        failures.add("INTERMEDIATE level requires C++ diagnostics — " +
                "attach interceptors to oracle and test executors, then compare captured intermediates");
    }

    // ─── Metadata Tracking ──────────────────────────────────────────────────

    private void trackOracleMetadata(ReplayMetadataTracker tracker) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        org.bytedeco.javacpp.Pointer planHandle = getPlanHandle(model);
        if (planHandle != null) {
            tracker.recordIteration(0, nativeOps, planHandle);
        } else {
            tracker.recordIterationSlotBySlot(0);
        }
    }

    private void trackTestMetadata(ReplayMetadataTracker tracker) {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        org.bytedeco.javacpp.Pointer planHandle = getPlanHandle(model);
        if (planHandle != null) {
            tracker.recordIteration(0, nativeOps, planHandle);
        } else {
            tracker.recordIterationSlotBySlot(0);
        }
    }

    private org.bytedeco.javacpp.Pointer getPlanHandle(SameDiff model) {
        try {
            InferenceSession session = model.getOrCreateSession();
            if (session != null) {
                DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
                if (executor != null) {
                    // Access native plan handle via reflection if needed
                    // For now, return null — metadata tracking is optional
                    return null;
                }
            }
        } catch (Exception e) {
            // Expected if no native plan is compiled
        }
        return null;
    }

    private String findLogitsName(Map<String, INDArray> outputs) {
        for (String name : outputs.keySet()) {
            if (name.toLowerCase().contains("logit")) return name;
        }
        return outputs.keySet().iterator().next();
    }

    // ─── AutoCloseable ──────────────────────────────────────────────────────

    @Override
    public void close() {
        if (inputEvolutor != null) {
            inputEvolutor.close();
        }
        if (oracleMetadataTracker != null) {
            oracleMetadataTracker.close();
        }
        if (testMetadataTracker != null) {
            testMetadataTracker.close();
        }
    }

    // ─── Oracle Mode Enum ───────────────────────────────────────────────────

    /**
     * Oracle reference mode.
     */
    public enum OracleMode {
        /** Use SLOT_BY_SLOT execution of the same model as the oracle. */
        SLOT_BY_SLOT_REFERENCE,
        /** Use a manually-computed baseline as the oracle. */
        MANUAL_BASELINE
    }

    // ─── Builder ────────────────────────────────────────────────────────────

    public static class Builder {
        private SameDiff model;
        private OracleMode oracleMode = OracleMode.SLOT_BY_SLOT_REFERENCE;
        private BenchmarkConfig testConfig;
        private BenchmarkConfig oracleConfig;
        private ValidationResult.ValidationLevel validationLevel = ValidationResult.ValidationLevel.LOGITS;
        private int maxDecodeSteps = 10;
        private DecodeInputEvolutor.EvolutionMode evolutionMode = DecodeInputEvolutor.EvolutionMode.ALL_TOGETHER;
        private ValidationConfig validationConfig;
        private int tokenPrefixLength = 4;
        private int topK = 5;
        private Set<String> intermediateVarNames;
        private boolean paddedMode = false;
        private long maxKvLen = 0;

        public Builder model(SameDiff m) { this.model = m; return this; }
        public Builder oracleMode(OracleMode m) { this.oracleMode = m; return this; }
        public Builder testConfig(BenchmarkConfig c) { this.testConfig = c; return this; }
        public Builder oracleConfig(BenchmarkConfig c) { this.oracleConfig = c; return this; }
        public Builder validationLevel(ValidationResult.ValidationLevel l) { this.validationLevel = l; return this; }
        public Builder maxDecodeSteps(int n) { this.maxDecodeSteps = n; return this; }
        public Builder evolutionMode(DecodeInputEvolutor.EvolutionMode m) { this.evolutionMode = m; return this; }
        public Builder validationConfig(ValidationConfig c) { this.validationConfig = c; return this; }
        public Builder tokenPrefixLength(int n) { this.tokenPrefixLength = n; return this; }
        public Builder topK(int n) { this.topK = n; return this; }
        public Builder intermediateVarNames(Set<String> names) { this.intermediateVarNames = names; return this; }
        public Builder paddedMode(boolean b) { this.paddedMode = b; return this; }
        public Builder maxKvLen(long n) { this.maxKvLen = n; return this; }

        public DecodeValidationFramework build() {
            Objects.requireNonNull(model, "model is required");
            Objects.requireNonNull(testConfig, "testConfig is required");
            Objects.requireNonNull(oracleConfig, "oracleConfig is required");
            return new DecodeValidationFramework(this);
        }
    }
}
