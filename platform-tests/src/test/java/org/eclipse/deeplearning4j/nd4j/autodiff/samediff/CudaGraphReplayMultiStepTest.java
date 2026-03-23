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

package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests CUDA graph capture and replay across 50+ steps to detect stale replay.
 *
 * Builds a graph with matmul + activation (mimicking a decoder layer), compiles DSP,
 * freezes shapes, and runs 50 steps with outputDirect(). Each step changes one input
 * (simulating different token embeddings).
 *
 * Key checks:
 * - Verify outputs change when inputs change (not stale replay)
 * - After step 20, check if outputs start repeating
 * - Compare outputDirect() results with output() results at checkpoints
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class CudaGraphReplayMultiStepTest extends BaseNd4jTestWithBackends {

    private static final int HIDDEN_SIZE = 64;
    private static final int NUM_LAYERS = 3;
    private static final int TOTAL_STEPS = 50;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    /**
     * Build a multi-layer MLP: input -> (matmul + bias + tanh) * N -> output
     * Fixed input shape [1, HIDDEN_SIZE] for frozen shape execution.
     */
    private SameDiff buildDecoderLikeGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, HIDDEN_SIZE);
        SDVariable current = input;

        for (int i = 0; i < NUM_LAYERS; i++) {
            SDVariable w = sd.var("w_" + i, Nd4j.randn(DataType.FLOAT, HIDDEN_SIZE, HIDDEN_SIZE).muli(0.02));
            SDVariable b = sd.var("b_" + i, Nd4j.zeros(DataType.FLOAT, HIDDEN_SIZE));
            current = sd.mmul("matmul_" + i, current, w);
            current = current.add("add_" + i, b);
            if (i < NUM_LAYERS - 1) {
                current = sd.math.tanh("tanh_" + i, current);
            }
        }
        current = sd.identity("output", current);
        sd.setOutputs("output");
        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testCudaGraphReplayOutputsChangeWithInputs(Nd4jBackend backend) {
        SameDiff sd = buildDecoderLikeGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Fixed-address input buffer (simulates reusable decode buffer)
        INDArray reusableInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN_SIZE);

        // Step 0: warm up with output()
        Map<String, INDArray> warmupResult = sd.output(
                Collections.singletonMap("input", reusableInput), "output");
        assertNotNull(warmupResult.get("output"));

        // Step 1: second output() call (DSP auto-compile should trigger)
        reusableInput.assign(Nd4j.randn(DataType.FLOAT, 1, HIDDEN_SIZE));
        sd.output(Collections.singletonMap("input", reusableInput), "output");

        // Freeze shapes for CUDA graph replay
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("Shapes frozen for CUDA graph replay");
        }

        // Run TOTAL_STEPS with outputDirect, tracking outputs
        float[][] directOutputs = new float[TOTAL_STEPS][];
        Set<Integer> uniqueArgmax = new HashSet<>();
        int firstStaleStep = -1;

        // Also compute reference outputs at checkpoint steps using a separate SameDiff instance
        SameDiff refSd = buildDecoderLikeGraph();
        // Copy weights from original
        for (String varName : sd.variableMap().keySet()) {
            if (sd.getVariable(varName).getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                INDArray arr = sd.getVariable(varName).getArr();
                if (arr != null && refSd.hasVariable(varName)) {
                    refSd.getVariable(varName).getArr().assign(arr);
                }
            }
        }

        int[] checkpointSteps = {9, 19, 29, 39, 49};
        Map<Integer, float[]> referenceOutputs = new HashMap<>();

        for (int step = 0; step < TOTAL_STEPS; step++) {
            // Generate deterministic but varying input
            Nd4j.getRandom().setSeed(step * 1000L + 42);
            INDArray stepInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN_SIZE).muli(0.1f);
            reusableInput.assign(stepInput);
            Nd4j.getExecutioner().commit();

            // outputDirect (frozen path / CUDA graph replay)
            Map<String, INDArray> result = sd.outputDirect(
                    Collections.singletonMap("input", reusableInput), "output");
            Nd4j.getExecutioner().commit();
            INDArray output = result.get("output");
            assertNotNull(output, "Output null at step " + step);

            float[] data = output.dup().data().asFloat();
            directOutputs[step] = data;

            // Track argmax
            int argmax = 0;
            for (int i = 1; i < data.length; i++) {
                if (data[i] > data[argmax]) argmax = i;
            }
            uniqueArgmax.add(argmax);

            // Check for stale output (identical to previous)
            if (step > 0 && firstStaleStep < 0) {
                boolean identical = Arrays.equals(directOutputs[step], directOutputs[step - 1]);
                if (identical) {
                    firstStaleStep = step;
                    log.warn("STALE REPLAY DETECTED at step {}: output identical to step {}", step, step - 1);
                }
            }

            // At checkpoint steps, also compute reference via output()
            boolean isCheckpoint = false;
            for (int cp : checkpointSteps) {
                if (step == cp) { isCheckpoint = true; break; }
            }
            if (isCheckpoint) {
                Nd4j.getRandom().setSeed(step * 1000L + 42);
                INDArray refInput = Nd4j.randn(DataType.FLOAT, 1, HIDDEN_SIZE).muli(0.1f);
                Map<String, INDArray> refResult = refSd.output(
                        Collections.singletonMap("input", refInput), "output");
                Nd4j.getExecutioner().commit();
                referenceOutputs.put(step, refResult.get("output").dup().data().asFloat());
            }

            if (step < 5 || step % 10 == 0) {
                log.info("Step {}: argmax={} first5=[{},{},{},{},{}]",
                        step, argmax,
                        String.format("%.4f", data[0]),
                        String.format("%.4f", data[1]),
                        String.format("%.4f", data[2]),
                        String.format("%.4f", data[3]),
                        String.format("%.4f", data[4]));
            }
        }

        // Compare direct vs reference at checkpoints
        int divergenceCount = 0;
        for (int cp : checkpointSteps) {
            float[] direct = directOutputs[cp];
            float[] ref = referenceOutputs.get(cp);
            if (ref == null) continue;

            double maxDiff = 0;
            for (int i = 0; i < Math.min(direct.length, ref.length); i++) {
                maxDiff = Math.max(maxDiff, Math.abs(direct[i] - ref[i]));
            }

            log.info("Checkpoint step {}: maxDiff={}", cp, String.format("%.6f", maxDiff));
            if (maxDiff > 0.01) {
                divergenceCount++;
                log.warn("DIVERGENCE at step {}: maxDiff={} exceeds tolerance",
                        cp, String.format("%.6f", maxDiff));
            }
        }

        // Summary
        double diversity = (double) uniqueArgmax.size() / TOTAL_STEPS;
        log.info("=== CUDA GRAPH REPLAY MULTI-STEP RESULTS ===");
        log.info("  Total steps: {}", TOTAL_STEPS);
        log.info("  Unique argmax values: {} / {} ({}%)", uniqueArgmax.size(), TOTAL_STEPS, String.format("%.1f", diversity * 100));
        log.info("  First stale step: {}", firstStaleStep);
        log.info("  Divergence count at checkpoints: {} / {}", divergenceCount, checkpointSteps.length);

        // Assertions
        if (firstStaleStep >= 0) {
            assertTrue(firstStaleStep > 20,
                    "Stale replay should not occur before step 20 -- started at step " + firstStaleStep);
        }
        assertTrue(uniqueArgmax.size() > 5,
                "Must have >5 unique argmax values across 50 steps, got " + uniqueArgmax.size());

        sd.close();
        refSd.close();
    }
}
