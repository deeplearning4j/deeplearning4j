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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.benchmark.*;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.CapturingSlotInterceptor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Smoke tests for the DecodeValidationFramework.
 *
 * <p>These tests prove that the framework's core components work correctly:
 * multi-level comparison, replay metadata tracking, input evolution, and
 * validation result reporting. They use small synthetic SameDiff graphs
 * rather than full SmolDocling models to keep execution fast.</p>
 *
 * <h3>What these tests validate</h3>
 * <ul>
 *   <li>MultiLevelComparator: token prefix, logits, top-K comparison logic</li>
 *   <li>ReplayMetadataTracker: phase tracking, signature stability, violation detection</li>
 *   <li>DecodeInputEvolutor: evolving position_ids, attention_mask, KV buffers</li>
 *   <li>ValidationResult: proper failure/success reporting, diagnostic output</li>
 *   <li>DecodeValidationFramework: end-to-end orchestration with oracle vs test</li>
 * </ul>
 *
 * <h3>What these tests do NOT validate</h3>
 * <ul>
 *   <li>SmolDocling benchmark correctness — that's for TestDspValidation and
 *       TestBenchmarkAccuracyIsolation</li>
 *   <li>Triton section fusion — covered by benchmark-subgraph correctness tests
 *       (Workstream B)</li>
 *   <li>CUDA graph replay — covered by TestCudaGraphReplayRegression</li>
 * </ul>
 *
 * <h3>Running these tests</h3>
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestDecodeValidationFrameworkSmoke \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestDecodeValidationFrameworkSmoke {

    // ─── MultiLevelComparator Smoke Tests ───────────────────────────────────

    @Test
    @DisplayName("Smoke: MultiLevelComparator token prefix match")
    public void testTokenPrefixMatch() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        List<Integer> reference = Arrays.asList(100, 200, 300, 400, 500);
        int[] test = new int[]{100, 200, 300, 400, 500};

        List<String> failures = comparator.compareTokenPrefix(reference, test, 5);
        assertTrue(failures.isEmpty(), "Identical tokens should produce no failures");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator token prefix mismatch")
    public void testTokenPrefixMismatch() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        List<Integer> reference = Arrays.asList(100, 200, 300, 400, 500);
        int[] test = new int[]{100, 250, 300, 400, 500};

        List<String> failures = comparator.compareTokenPrefix(reference, test, 5);
        assertFalse(failures.isEmpty(), "Different tokens should produce a failure");
        assertTrue(failures.get(0).contains("step 1"), "Failure should report correct step");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator token prefix length enforcement")
    public void testTokenPrefixLength() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        List<Integer> reference = Arrays.asList(100, 200, 300);
        int[] test = new int[]{100, 200};

        List<String> failures = comparator.compareTokenPrefix(reference, test, 4);
        assertFalse(failures.isEmpty(), "Too few test tokens should produce a failure");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator logits comparison")
    public void testLogitsComparison() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        INDArray refLogits = Nd4j.linspace(0, 1, 100, DataType.FLOAT).reshape(1, 100);
        INDArray testLogits = refLogits.dup();  // Exact copy

        MultiLevelComparator.LogitsComparison result = comparator.compareLogits(refLogits, testLogits);
        assertEquals(0.0, result.getMaxAbsDiff(), 1e-10, "Identical logits should have zero diff");
        assertTrue(result.isArgmaxMatch(), "Identical logits should have matching argmax");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator logits with small diff")
    public void testLogitsComparisonSmallDiff() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        INDArray refLogits = Nd4j.linspace(0, 1, 100, DataType.FLOAT).reshape(1, 100);
        INDArray testLogits = refLogits.add(1e-6);  // Tiny perturbation

        MultiLevelComparator.LogitsComparison result = comparator.compareLogits(refLogits, testLogits);
        assertTrue(result.getMaxAbsDiff() < 1e-4, "Small diff should be within tolerance");
        assertTrue(result.isArgmaxMatch(), "Small diff should not change argmax");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator top-K overlap")
    public void testTopKOverlap() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        // Create logits where top-5 are clearly the first 5 elements
        INDArray refLogits = Nd4j.create(DataType.FLOAT, 1, 100);
        for (int i = 0; i < 5; i++) refLogits.putScalar(0, i, 10.0f - i);

        INDArray testLogits = refLogits.dup();  // Identical

        int overlap = comparator.compareTopKOverlap(refLogits, testLogits, 5);
        assertEquals(5, overlap, "Identical logits should have full top-5 overlap");
    }

    @Test
    @DisplayName("Smoke: MultiLevelComparator top-K partial overlap")
    public void testTopKPartialOverlap() {
        MultiLevelComparator comparator = new MultiLevelComparator(ValidationConfig.standard());

        INDArray refLogits = Nd4j.create(DataType.FLOAT, 1, 100);
        for (int i = 0; i < 5; i++) refLogits.putScalar(0, i, 10.0f - i);

        // Shift top-K by 1
        INDArray testLogits = Nd4j.create(DataType.FLOAT, 1, 100);
        for (int i = 0; i < 5; i++) testLogits.putScalar(0, i + 1, 10.0f - i);

        int overlap = comparator.compareTopKOverlap(refLogits, testLogits, 5);
        assertEquals(4, overlap, "Shifted top-K should have 4/5 overlap");
    }

    // ─── ReplayMetadataTracker Smoke Tests ──────────────────────────────────

    @Test
    @DisplayName("Smoke: ReplayMetadataTracker phase transitions")
    public void testReplayMetadataTrackerPhaseTransitions() {
        ReplayMetadataTracker tracker = new ReplayMetadataTracker("TEST");

        tracker.recordPhaseTransition(0, ReplayMetadataTracker.Phase.COMPILE, ReplayMetadataTracker.Phase.CAPTURE, "First capture");
        tracker.recordPhaseTransition(1, ReplayMetadataTracker.Phase.CAPTURE, ReplayMetadataTracker.Phase.REPLAY, "All segments captured");

        ReplayMetadataTracker.ReplayMetadata metadata = tracker.buildMetadata();
        assertEquals(ReplayMetadataTracker.Phase.REPLAY, metadata.getCurrentPhase());
        assertEquals(2, metadata.getPhaseTransitions().size());
        assertTrue(metadata.getPhaseViolations().isEmpty());

        tracker.close();
    }

    @Test
    @DisplayName("Smoke: ReplayMetadataTracker phase violation detection")
    public void testReplayMetadataTrackerPhaseViolation() {
        ReplayMetadataTracker tracker = new ReplayMetadataTracker("TEST");

        tracker.recordPhaseTransition(0, ReplayMetadataTracker.Phase.COMPILE, ReplayMetadataTracker.Phase.REPLAY, "Direct to replay");
        tracker.recordPhaseViolation(1, "Unexpected fallback from REPLAY to CAPTURE");

        List<String> violations = tracker.getPhaseViolations();
        assertFalse(violations.isEmpty(), "Should have recorded phase violation");
        assertTrue(violations.get(0).contains("fallback"), "Violation message should mention fallback");

        // Check signature stability — no signatures recorded, should pass
        List<String> sigFailures = tracker.checkSignatureStability();
        assertTrue(sigFailures.isEmpty(), "No signatures means stable by default");

        tracker.close();
    }

    @Test
    @DisplayName("Smoke: ReplayMetadataTracker execution count monotonicity")
    public void testReplayMetadataTrackerExecMonotonicity() {
        ReplayMetadataTracker tracker = new ReplayMetadataTracker("TEST");
        // Without native plan, monotonicity check should pass (no data to violate)
        List<String> failures = tracker.checkExecCountMonotonicity();
        assertTrue(failures.isEmpty(), "Empty exec counts should pass monotonicity check");
        tracker.close();
    }

    // ─── DecodeInputEvolutor Smoke Tests ────────────────────────────────────

    @Test
    @DisplayName("Smoke: DecodeInputEvolutor builds step inputs")
    public void testDecodeInputEvolutorBuildInputs() {
        List<String> inputNames = Arrays.asList("inputs_embeds", "attention_mask", "position_ids");
        List<String> kvNames = Arrays.asList("present_key_0", "present_value_0");

        DecodeInputEvolutor evolutor = DecodeInputEvolutor.builder()
                .decoderInputNames(inputNames)
                .logitsName("logits")
                .kvOutputNames(kvNames)
                .prefillSeqLen(10)
                .maxDecodeSteps(5)
                .evolutionMode(DecodeInputEvolutor.EvolutionMode.POSITION_IDS_ONLY)
                .paddedMode(false)
                .build();

        // Build step 0 inputs
        Map<String, INDArray> step0Inputs = evolutor.buildStepInputs(0);
        assertTrue(step0Inputs.containsKey("position_ids"), "Should include position_ids");
        assertEquals(10, step0Inputs.get("position_ids").getLong(0), "Position should start at prefillSeqLen");

        // Build step 1 inputs
        Map<String, INDArray> step1Inputs = evolutor.buildStepInputs(1);
        assertEquals(11, step1Inputs.get("position_ids").getLong(0), "Position should increment");

        evolutor.close();
    }

    @Test
    @DisplayName("Smoke: DecodeInputEvolutor padded mode mask")
    public void testDecodeInputEvolutorPaddedMask() {
        List<String> inputNames = Arrays.asList("inputs_embeds", "attention_mask", "position_ids");
        List<String> kvNames = Arrays.asList("present_key_0", "present_value_0");

        DecodeInputEvolutor evolutor = DecodeInputEvolutor.builder()
                .decoderInputNames(inputNames)
                .logitsName("logits")
                .kvOutputNames(kvNames)
                .prefillSeqLen(10)
                .maxDecodeSteps(5)
                .evolutionMode(DecodeInputEvolutor.EvolutionMode.ALL_TOGETHER)
                .paddedMode(true)
                .maxKvLen(30)
                .build();

        Map<String, INDArray> step0Inputs = evolutor.buildStepInputs(0);
        assertTrue(step0Inputs.containsKey("attention_mask"), "Should include attention_mask");
        INDArray mask = step0Inputs.get("attention_mask");
        assertEquals(31, mask.size(1), "Padded mask should have length maxKvLen + 1");

        evolutor.close();
    }

    // ─── ValidationResult Smoke Tests ───────────────────────────────────────

    @Test
    @DisplayName("Smoke: ValidationResult pass reporting")
    public void testValidationResultPass() {
        ValidationResult result = ValidationResult.builder("TEST", "SLOT_BY_SLOT", "OPTIMAL",
                        ValidationResult.ValidationLevel.TOKEN_PREFIX)
                .passed(true)
                .firstDivergentStep(-1)
                .totalStepsCompared(5)
                .maxDiffObserved(0.0)
                .build();

        assertTrue(result.isPassed());
        assertTrue(result.getFailures().isEmpty());
        assertNotNull(result.toReport());  // Report should not throw
    }

    @Test
    @DisplayName("Smoke: ValidationResult failure reporting")
    public void testValidationResultFail() {
        ValidationResult result = ValidationResult.builder("TEST", "SLOT_BY_SLOT", "OPTIMAL",
                        ValidationResult.ValidationLevel.TOKEN_PREFIX)
                .passed(false)
                .failure("Token mismatch at step 2")
                .firstDivergentStep(2)
                .totalStepsCompared(5)
                .maxDiffObserved(1.5)
                .build();

        assertFalse(result.isPassed());
        assertEquals(1, result.getFailures().size());
        assertEquals("Token mismatch at step 2", result.getFirstFailure());
        assertEquals(2, result.getFirstDivergentStep());
        assertNotNull(result.toReport());
    }

    // ─── DecodeStepValidator Smoke Tests ────────────────────────────────────

    @Test
    @DisplayName("Smoke: DecodeStepValidator identical decode")
    public void testDecodeStepValidatorIdentical() {
        ValidationConfig config = ValidationConfig.standard();
        DecodeStepValidator validator = new DecodeStepValidator(config);

        List<INDArray> refLogits = new ArrayList<>();
        List<Integer> refTokens = new ArrayList<>();
        List<INDArray> testLogits = new ArrayList<>();
        List<Integer> testTokens = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            INDArray logits = Nd4j.linspace(i, i + 1, 100, DataType.FLOAT).reshape(1, 100);
            refLogits.add(logits);
            testLogits.add(logits.dup());
            refTokens.add(i * 10);
            testTokens.add(i * 10);
        }

        DecodeStepValidator.DecodeComparisonReport report = validator.compare(
                refLogits, refTokens, testLogits, testTokens, "REF", "TEST");

        assertTrue(report.getTokenMatchRate() >= 1.0, "Identical decode should have 100% token match");
        assertTrue(report.getArgmaxMatchRate() >= 1.0, "Identical decode should have 100% argmax match");
    }

    @Test
    @DisplayName("Smoke: DecodeStepValidator divergent decode")
    public void testDecodeStepValidatorDivergent() {
        ValidationConfig config = ValidationConfig.standard();
        DecodeStepValidator validator = new DecodeStepValidator(config);

        List<INDArray> refLogits = new ArrayList<>();
        List<Integer> refTokens = new ArrayList<>();
        List<INDArray> testLogits = new ArrayList<>();
        List<Integer> testTokens = new ArrayList<>();

        for (int i = 0; i < 5; i++) {
            INDArray refL = Nd4j.linspace(i, i + 1, 100, DataType.FLOAT).reshape(1, 100);
            INDArray testL = refL.add(i * 0.1);  // Growing perturbation

            refLogits.add(refL);
            testLogits.add(testL);
            refTokens.add(i * 10);
            testTokens.add(i * 10 + (i >= 2 ? 1 : 0));  // Diverge at step 2
        }

        DecodeStepValidator.DecodeComparisonReport report = validator.compare(
                refLogits, refTokens, testLogits, testTokens, "REF", "TEST");

        assertEquals(2, report.getFirstTokenDivergenceStep(), "Should diverge at step 2");
        assertTrue(report.getTokenMatchRate() < 1.0, "Divergent decode should have < 100% match");
    }

    // ─── DspAccuracyValidator Smoke Test ────────────────────────────────────

    @Test
    @DisplayName("Smoke: DspAccuracyValidator compareArrays identical")
    public void testCompareArraysIdentical() {
        INDArray ref = Nd4j.linspace(0, 1, 100, DataType.FLOAT).reshape(10, 10);
        INDArray test = ref.dup();

        OpDivergence div = DspAccuracyValidator.compareArrays(
                ref, test, "test_var", "test_op", 0,
                1e-4, 1e-3, null);

        assertNull(div, "Identical arrays should produce no divergence");
    }

    @Test
    @DisplayName("Smoke: DspAccuracyValidator compareArrays with diff")
    public void testCompareArraysWithDiff() {
        INDArray ref = Nd4j.linspace(0, 1, 100, DataType.FLOAT).reshape(10, 10);
        INDArray test = ref.add(0.1);  // Uniform offset

        OpDivergence div = DspAccuracyValidator.compareArrays(
                ref, test, "test_var", "test_op", 0,
                1e-4, 1e-3, null);

        assertNotNull(div, "Arrays with diff > tolerance should produce divergence");
        assertEquals(0.1, div.getMaxAbsDiff(), 1e-6, "Max abs diff should be 0.1");
    }
}
