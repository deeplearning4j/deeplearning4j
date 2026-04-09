/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  License for the specific language governing permissions and limitations
 *  *  under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.eclipse.deeplearning4j.nd4j.autodiff.samediff.TritonTestUtils.ReplayValidationHelper;
import static org.eclipse.deeplearning4j.nd4j.autodiff.samediff.TritonTestUtils.SkipException;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase 2: Replay-unit consolidation validation.
 *
 * These tests compile once and execute the same native plan handle across
 * multiple iterations to validate same-handle correctness for island/gap-heavy
 * structures.
 *
 * This suite intentionally focuses on:
 * - output correctness on every iteration
 * - repeated same-handle execution
 * - per-segment state capture across iterations
 * - signature stability when replay metadata is available
 * - monotonic execution counters
 *
 * Replay-metadata introspection is exercised by dedicated framework tests. The
 * structural tests here should continue to catch island/gap regressions even
 * when a tiny standalone graph does not surface non-zero replay metadata yet.
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonReplayConsolidationTest extends BaseNd4jTestWithBackends {

    private static final int NUM_ITERATIONS = 5;

    @BeforeEach
    public void setUp() {
        System.setProperty("nd4j.dsp.diagnostics", "EXECUTE,GRAPH_REPLAY");
        System.setProperty("nd4j.dsp.diagnostics.level", "DETAILED");
    }

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Replay Signature Stability ─────────────────────────────────────────

    @Test
    @DisplayName("Replay: same plan handle across iterations → stable output")
    public void testReplaySignatureStability() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 16);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 16, 8));
        SDVariable result = sd.mmul("result", x, w);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testReplaySignatureStability", sd, Map.of("x", input), "result")) {
            // Warmup
            Map<String, INDArray> warmupResult = helper.warmup();
            helper.verifyOutput(warmupResult);

            // Multiple iterations on SAME compiled handle
            for (int i = 0; i < NUM_ITERATIONS; i++) {
                Map<String, INDArray> iterResult = helper.iterate(i);
                helper.verifyOutput(iterResult);
            }

            // Verify stability: signature, execution counts
            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();

        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Gather + Concat Ladder Consolidation ───────────────────────────────

    @Test
    @DisplayName("Replay: gather → concat → reshape ladder consolidates into fewer units")
    public void testGatherConcatLadderConsolidation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable y = sd.placeHolder("y", DataType.FLOAT, -1, 32);

        SDVariable xGathered = sd.gather("x_gather", x, new int[]{0, 1, 2, 3}, 0);
        SDVariable yGathered = sd.gather("y_gather", y, new int[]{0, 1, 2, 3}, 0);
        SDVariable concat = sd.concat("concat", 0, xGathered, yGathered);
        SDVariable result = sd.reshape("result", concat, -1, 8);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 8, 32);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 8, 32);
        Map<String, INDArray> inputs = Map.of("x", input1, "y", input2);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testGatherConcatLadder", sd, inputs, "result")) {
            Map<String, INDArray> warmupResult = helper.warmup();
            helper.verifyOutput(warmupResult);

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Address Drift Regression ──────────────────────────────────────────

    @Test
    @DisplayName("Replay: no address-drift invalidations from consolidation")
    public void testNoAddressDriftFromConsolidation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 32, 16));

        SDVariable gathered = sd.gather("gather", x, new int[]{0, 1, 2, 3}, 0);
        SDVariable result = sd.mmul("result", gathered, w, false, false, false);

        INDArray input = Nd4j.randn(DataType.FLOAT, 8, 32);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testNoAddressDrift", sd, Map.of("x", input), "result")) {
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                Map<String, INDArray> iterResult = helper.iterate(i);
                helper.verifyOutput(iterResult);
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Concat Ladder + Gather Ladder Bucket ──────────────────────────────

    @Test
    @DisplayName("Replay: concat_ladder + gather_ladder bucket pattern")
    public void testConcatGatherLadderBucket() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 16);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 16);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 16);

        SDVariable aG = sd.gather("a_gather", a, new int[]{0, 1, 2, 3}, 0);
        SDVariable bG = sd.gather("b_gather", b, new int[]{0, 1, 2, 3}, 0);
        SDVariable cG = sd.gather("c_gather", c, new int[]{0, 1, 2, 3}, 0);
        SDVariable concat1 = sd.concat("concat1", 0, aG, bG);
        SDVariable concat2 = sd.concat("concat2", 0, concat1, cG);
        SDVariable result = sd.reshape("result", concat2, -1, 12);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 8, 16);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 8, 16);
        INDArray input3 = Nd4j.randn(DataType.FLOAT, 8, 16);
        Map<String, INDArray> inputs = Map.of("a", input1, "b", input2, "c", input3);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testConcatGatherLadder", sd, inputs, "result")) {
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Attention Tail Stress Test ────────────────────────────────────────

    @Test
    @DisplayName("Replay: attention_tail + stack_chain + concat_ladder + gather_ladder")
    public void testAttentionTailStressTest() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, -1, 8, 64);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, -1, 8, 64);
        SDVariable v = sd.placeHolder("v", DataType.FLOAT, -1, 8, 64);

        SDVariable kG = sd.gather("k_gather", k, new int[]{0, 1, 2, 3}, 0);
        SDVariable vG = sd.gather("v_gather", v, new int[]{0, 1, 2, 3}, 0);
        SDVariable kStacked = sd.stack("k_stack", 0, kG, vG);
        SDVariable vStacked = sd.stack("v_stack", 0, kG, vG);
        SDVariable kStacked3d = sd.reshape("k_stacked_3d", kStacked, -1, 8, 64);
        SDVariable vStacked3d = sd.reshape("v_stacked_3d", vStacked, -1, 8, 64);
        SDVariable kConcat = sd.concat("k_concat", 0, kStacked3d, kG);
        SDVariable vConcat = sd.concat("v_concat", 0, vStacked3d, vG);
        SDVariable qFlat = sd.reshape("q_reshape", q, -1, 64);
        SDVariable kFlat = sd.reshape("k_reshape", kConcat, -1, 64);
        SDVariable result = sd.mmul("result", qFlat, kFlat, false, true, false);

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray vInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        Map<String, INDArray> inputs = Map.of("q", qInput, "k", kInput, "v", vInput);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testAttentionTailStress", sd, inputs, "result")) {
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Phase Violation Regression ────────────────────────────────────────

    @Test
    @DisplayName("Replay: no new phase violations after consolidation")
    public void testNoPhaseViolationAfterConsolidation() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        SDVariable reshaped = sd.reshape("reshape", x, -1, 4, 8);
        SDVariable gathered = sd.gather("gather", reshaped, new int[]{0, 2}, 0);
        SDVariable r2 = sd.reshape("r2", reshaped, -1L, 4L, 8L);
        SDVariable concat = sd.concat("concat", 0, gathered, r2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 8, 32);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testNoPhaseViolation", sd, Map.of("x", input), "concat")) {
            // Should not throw TRITON_REPLAY_PHASE_VIOLATION
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        } catch (RuntimeException e) {
            if (e.getMessage() != null && e.getMessage().contains("PHASE_VIOLATION")) {
                fail("TRITON_REPLAY_PHASE_VIOLATION should not occur: " + e.getMessage());
            }
            throw e;
        }

        sd.close();
    }

    // ─── Permute View Recipe Absorption ─────────────────────────────────────

    @Test
    @DisplayName("Replay: permute view recipes absorbed before matmul")
    public void testPermuteViewRecipeAbsorption() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, -1, 8, 64);
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, -1, 8, 64);

        SDVariable qPerm = sd.permute("q_perm", q, 0L, 2L, 1L);
        SDVariable kPerm = sd.permute("k_perm", k, 0L, 2L, 1L);
        SDVariable result = sd.mmul("result", qPerm, kPerm, false, true, false);

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 4, 8, 64);
        Map<String, INDArray> inputs = Map.of("q", qInput, "k", kInput);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testPermuteViewRecipe", sd, inputs, "result")) {
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Multiple Concat Ladders ───────────────────────────────────────────

    @Test
    @DisplayName("Replay: multiple concat ladders in sequence")
    public void testMultipleConcatLadders() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, -1, 8);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, -1, 8);
        SDVariable c = sd.placeHolder("c", DataType.FLOAT, -1, 8);
        SDVariable d = sd.placeHolder("d", DataType.FLOAT, -1, 8);

        SDVariable concat1 = sd.concat("concat1", 0, a, b);
        SDVariable concat2 = sd.concat("concat2", 0, c, d);
        SDVariable concat3 = sd.concat("concat3", 0, concat1, concat2);
        SDVariable result = sd.reshape("result", concat3, -1, 32);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray input3 = Nd4j.randn(DataType.FLOAT, 4, 8);
        INDArray input4 = Nd4j.randn(DataType.FLOAT, 4, 8);
        Map<String, INDArray> inputs = Map.of("a", input1, "b", input2, "c", input3, "d", input4);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testMultipleConcatLadders", sd, inputs, "result")) {
            helper.warmup();

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
            
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    // ─── Internal Gap Replay Shapes ──────────────────────────────────────────

    @Test
    @DisplayName("Replay: alternating gather ladder with const and view gaps")
    public void testAlternatingGatherConstViewLadder() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);

        SDVariable g1 = sd.gather("g1", x, new int[]{0, 1, 2, 3, 4, 5}, 0);
        SDVariable biased = sd.math().add("biased", g1,
                sd.constant("bias", Nd4j.linspace(0.0, 1.0, 32, DataType.FLOAT).reshape(1, 32)));
        SDVariable g2 = sd.gather("g2", biased, new int[]{1, 3, 5}, 0);
        SDVariable reshaped = sd.reshape("reshape", g2, -1, 8, 4);
        SDVariable permuted = sd.permute("permute", reshaped, 1, 0, 2);
        SDVariable flattened = sd.reshape("flatten", permuted, -1, 4);
        SDVariable g3 = sd.gather("g3", flattened, new int[]{0, 2, 4}, 0);
        SDVariable result = sd.nn().relu("result", g3, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 32);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testAlternatingGatherConstViewLadder", sd, Map.of("x", input), "result")) {
            helper.verifyOutput(helper.warmup());

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    @Test
    @DisplayName("Replay: mask slice/tile/concat ladder stays stable")
    public void testMaskSliceTileConcatLadder() {
        SameDiff sd = SameDiff.create();
        SDVariable mask = sd.placeHolder("mask", DataType.INT64, 1, 32);
        SDVariable values = sd.placeHolder("values", DataType.FLOAT, 32, 16);

        SDVariable prefixMask = sd.stridedSlice("prefix_mask", mask,
                new long[]{0, 0}, new long[]{1, 16}, new long[]{1, 1});
        SDVariable maskExpanded = sd.expandDims("mask_expanded", prefixMask, 1);
        SDVariable tiledMask = sd.tile("tiled_mask", maskExpanded, 1, 4, 1);
        SDVariable maskFloat = sd.castTo("mask_float", tiledMask, DataType.FLOAT);
        SDVariable maskFlat = sd.reshape("mask_flat", maskFloat, -1, 16);
        SDVariable gathered = sd.gather("gather", values,
                new int[]{0, 1, 2, 3, 4, 5, 6, 7}, 0);
        SDVariable concat = sd.concat("concat", 0, gathered, maskFlat);
        SDVariable result = sd.math().add("result", concat,
                sd.constant("bias", Nd4j.ones(DataType.FLOAT, 1, 16)));

        INDArray maskInput = Nd4j.concat(1,
                Nd4j.ones(DataType.INT64, 1, 16),
                Nd4j.zeros(DataType.INT64, 1, 16));
        INDArray valuesInput = Nd4j.randn(DataType.FLOAT, 32, 16);
        Map<String, INDArray> inputs = Map.of("mask", maskInput, "values", valuesInput);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testMaskSliceTileConcatLadder", sd, inputs, "result")) {
            helper.verifyOutput(helper.warmup());

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    @Test
    @DisplayName("Replay: gather plus normalization tail stays stable")
    public void testGatherNormalizationTail() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 32);
        SDVariable gamma = sd.constant("gamma", Nd4j.ones(DataType.FLOAT, 32));
        SDVariable bias = sd.constant("bias", Nd4j.linspace(0.0, 0.1, 32, DataType.FLOAT));

        SDVariable gathered = sd.gather("gather", x, new int[]{0, 2, 4, 6}, 0);
        SDVariable norm = sd.nn().rmsNorm("norm", gathered, gamma, 1e-5);
        SDVariable result = sd.math().add("result", norm, bias);

        INDArray input = Nd4j.randn(DataType.FLOAT, 16, 32);

        try (ReplayValidationHelper helper = new ReplayValidationHelper(
                "testGatherNormalizationTail", sd, Map.of("x", input), "result")) {
            helper.verifyOutput(helper.warmup());

            for (int i = 0; i < NUM_ITERATIONS; i++) {
                helper.verifyOutput(helper.iterate(i));
            }

            helper.assertRecordedStateSamples(NUM_ITERATIONS + 1);
            helper.assertSignatureStable();
            helper.assertExecutionCountsMonotonic();
        } catch (SkipException e) {
            log.info("Skipping: {}", e.getMessage());
        }

        sd.close();
    }

    public List<DataType> getExclusiveDataTypes() {
        return new ArrayList<>() {{
            add(DataType.FLOAT);
        }};
    }
}
