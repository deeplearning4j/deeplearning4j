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

package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.vlm.data.VLMModelDownloader;
import org.eclipse.deeplearning4j.vlm.model.OnnxModelCache;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation test: CUDA graph replay correctness for SmolDocling vision encoder.
 *
 * Determines whether frozen-path NaN is a CUDA graph replay bug or model behavior
 * by using IDENTICAL deterministic inputs across warmup and replay calls.
 * If outputs match on warmup but diverge/NaN on replay, it's a replay data corruption bug.
 */
@Slf4j
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class TestDSPFrozenReplayCorrectness {

    private static SameDiff visionEncoder;
    private static boolean modelsLoaded = false;
    private static final int TARGET_SIZE = 512;

    @BeforeAll
    public static void setup() throws Exception {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        try {
            log.info("Loading SmolDocling vision encoder...");
            File visionDir = VLMModelDownloader.download(
                    VLMModelDownloader.VLMModel.SMOLDOCLING_VISION_ENCODER).getModelFile();
            visionEncoder = OnnxModelCache.importWithCache(visionDir.getAbsolutePath());
            visionEncoder.setDspAutoCompileEnabled(true);
            visionEncoder.setDspNativeAutoCompileEnabled(true);
            log.info("Vision encoder loaded: {} ops", visionEncoder.ops().length);
            modelsLoaded = true;
        } catch (Exception e) {
            log.error("Failed to load model: {}", e.getMessage());
        }
    }

    @AfterAll
    public static void teardown() {
        if (visionEncoder != null) visionEncoder.close();
    }

    /**
     * Test 1: Deterministic constant input — same on every call.
     *
     * Uses Nd4j.valueArrayOf(0.5) so every call gets bit-identical input.
     * If warmup (Call 2) passes but replay (Call 3) NaN, it's a replay bug.
     * If both NaN, it's model behavior.
     */
    @Test
    @Order(1)
    @DisplayName("Frozen replay with identical constant input must produce consistent output")
    public void testFrozenReplayConstantInput() {
        Assumptions.assumeTrue(modelsLoaded, "Model not loaded — skipping");

        List<String> outputNames = List.of("image_features");

        // Deterministic input: all 0.5
        INDArray pv = Nd4j.valueArrayOf(
                new long[]{1, 1, 3, TARGET_SIZE, TARGET_SIZE}, 0.5, DataType.FLOAT);
        INDArray mask = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);

        // Call 1: compile (unfrozen)
        log.info("Call 1: compile (unfrozen)");
        Map<String, INDArray> out1 = visionEncoder.output(
                Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
        INDArray feat1 = out1.get("image_features");
        assertNotNull(feat1, "Call 1 failed");
        double sum1 = feat1.sumNumber().doubleValue();
        log.info("Call 1 (compile): sum={}, NaN={}", sum1, Double.isNaN(sum1));

        if (Double.isNaN(sum1)) {
            // Model produces NaN for all-0.5 input even without freezing.
            // This is model behavior (normalization of constant input → 0/0).
            // Try varied input.
            log.warn("All-0.5 input produces NaN without freezing. Trying linspace...");
            pv.close();

            // Create varied input: values from 0.0 to 1.0
            long totalElems = 1L * 1 * 3 * TARGET_SIZE * TARGET_SIZE;
            pv = Nd4j.linspace(0.01, 0.99, totalElems, DataType.FLOAT)
                    .reshape(1, 1, 3, TARGET_SIZE, TARGET_SIZE);

            visionEncoder.resetSession();
            InferenceSession.setDynamicShapePlanEnabled(true);
            visionEncoder.setDspAutoCompileEnabled(true);
            visionEncoder.setDspNativeAutoCompileEnabled(true);

            out1 = visionEncoder.output(
                    Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            feat1 = out1.get("image_features");
            sum1 = feat1.sumNumber().doubleValue();
            log.info("Call 1 (linspace): sum={}, NaN={}", sum1, Double.isNaN(sum1));
            assertFalse(Double.isNaN(sum1), "Model produces NaN even for linspace input — model issue");
        }

        // Freeze
        freezeDspShapes(visionEncoder);

        // Call 2: frozen warmup with SAME input
        log.info("Call 2: frozen warmup with same input");
        Map<String, INDArray> out2 = visionEncoder.output(
                Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
        INDArray feat2 = out2.get("image_features");
        double sum2 = feat2.sumNumber().doubleValue();
        log.info("Call 2 (frozen warmup): sum={}, NaN={}", sum2, Double.isNaN(sum2));
        assertFalse(Double.isNaN(sum2), "Call 2 (warmup): NaN with deterministic input");

        // Call 3: frozen REPLAY with SAME input
        log.info("Call 3: frozen REPLAY with same input");
        Map<String, INDArray> out3 = visionEncoder.output(
                Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
        INDArray feat3 = out3.get("image_features");
        double sum3 = feat3.sumNumber().doubleValue();
        log.info("Call 3 (frozen replay): sum={}, NaN={}", sum3, Double.isNaN(sum3));
        assertFalse(Double.isNaN(sum3),
                String.format("Call 3 REPLAY NaN with SAME input that was OK on warmup (sum2=%.6f)" +
                        " — CUDA graph replay data corruption!", sum2));

        // Consistency check
        if (!Double.isNaN(sum2) && !Double.isNaN(sum3)) {
            double relDiff = Math.abs(sum2 - sum3) / (Math.abs(sum2) + 1e-10);
            log.info("sum2={}, sum3={}, relDiff={}", sum2, sum3, relDiff);
            assertTrue(relDiff < 0.01,
                    String.format("Replay output differs from warmup: sum2=%.6f sum3=%.6f relDiff=%.6f",
                            sum2, sum3, relDiff));
        }

        // Call 4: second replay
        log.info("Call 4: second replay with same input");
        Map<String, INDArray> out4 = visionEncoder.output(
                Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
        double sum4 = out4.get("image_features").sumNumber().doubleValue();
        log.info("Call 4 (second replay): sum={}, NaN={}", sum4, Double.isNaN(sum4));
        assertFalse(Double.isNaN(sum4), "Call 4 (second replay): NaN");

        unfreezeDspShapes(visionEncoder);
        pv.close();
        mask.close();
        log.info("Test PASSED: deterministic frozen replay is correct");
    }

    /**
     * Test 2: Multiple unfrozen calls with random input — baseline NaN check.
     *
     * If this test produces NaN on any call, the model inherently produces NaN
     * for some random inputs (model behavior, not DSP bug).
     */
    @Test
    @Order(2)
    @DisplayName("Multiple unfrozen calls: check if model inherently produces NaN for random inputs")
    public void testUnfrozenMultipleCallsNaN() {
        Assumptions.assumeTrue(modelsLoaded, "Model not loaded — skipping");

        visionEncoder.resetSession();
        InferenceSession.setDynamicShapePlanEnabled(true);
        visionEncoder.setDspAutoCompileEnabled(true);
        visionEncoder.setDspNativeAutoCompileEnabled(true);

        List<String> outputNames = List.of("image_features");
        int nanCount = 0;

        for (int i = 1; i <= 5; i++) {
            INDArray pv = Nd4j.rand(DataType.FLOAT, 1, 1, 3, TARGET_SIZE, TARGET_SIZE);
            INDArray mask = Nd4j.ones(DataType.BOOL, 1, TARGET_SIZE, TARGET_SIZE);

            Map<String, INDArray> out = visionEncoder.output(
                    Map.of("pixel_values", pv, "pixel_attention_mask", mask), outputNames);
            double sum = out.get("image_features").sumNumber().doubleValue();
            boolean nan = Double.isNaN(sum);
            log.info("Unfrozen call {}: sum={}, NaN={}", i, sum, nan);
            if (nan) nanCount++;

            pv.close();
            mask.close();
        }

        log.info("Unfrozen calls: {}/5 produced NaN", nanCount);
        if (nanCount > 0) {
            log.warn("Model inherently produces NaN for some random inputs ({}/5 calls)",
                    nanCount);
        } else {
            log.info("No NaN in 5 unfrozen calls — model is stable with random inputs");
        }
        // Don't assert — this is a diagnostic test
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    private void freezeDspShapes(SameDiff sd) {
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(true);
                    log.info("Froze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not freeze DSP shapes: {}", e.getMessage());
        }
    }

    private void unfreezeDspShapes(SameDiff sd) {
        try {
            var sessionsField = SameDiff.class.getDeclaredField("sessions");
            sessionsField.setAccessible(true);
            @SuppressWarnings("unchecked")
            Map<Long, InferenceSession> sessions = (Map<Long, InferenceSession>) sessionsField.get(sd);
            InferenceSession session = sessions.get(Thread.currentThread().getId());
            if (session != null) {
                DynamicShapePlanExecutor dsp = session.getDynamicShapePlanExecutor();
                if (dsp != null) {
                    dsp.setShapesFrozen(false);
                    log.info("Unfroze DSP shapes");
                }
            }
        } catch (Exception e) {
            log.warn("Could not unfreeze DSP shapes: {}", e.getMessage());
        }
    }
}
