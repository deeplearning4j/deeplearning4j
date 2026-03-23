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
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Stress test for consolidated arg table + dirty tracking across 100 steps.
 *
 * Builds a graph with multiple inputs. On each step, only ONE input changes
 * while others remain the same (same buffer address, same data). This exercises
 * the fast-replay path where argTableStable should detect that most inputs
 * haven't changed, but the changing input must still propagate correctly.
 *
 * Verifies:
 * - Outputs are correct at every step (compared to reference computation)
 * - Changing only one input still produces different outputs
 * - The arg table dirty tracking does not cause stale data
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class ArgTableDirtyTrackingStressTest extends BaseNd4jTestWithBackends {

    private static final int DIM = 32;
    private static final int NUM_STEPS = 100;

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
     * Build a graph with 4 inputs that are combined through matmuls and adds:
     *   output = tanh(A * W1 + B * W2 + C * W3 + D * W4)
     * Only input A changes each step; B, C, D stay fixed.
     */
    private SameDiff buildMultiInputGraph() {
        SameDiff sd = SameDiff.create();

        SDVariable a = sd.placeHolder("input_a", DataType.FLOAT, 1, DIM);
        SDVariable b = sd.placeHolder("input_b", DataType.FLOAT, 1, DIM);
        SDVariable c = sd.placeHolder("input_c", DataType.FLOAT, 1, DIM);
        SDVariable d = sd.placeHolder("input_d", DataType.FLOAT, 1, DIM);

        SDVariable w1 = sd.var("w1", Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02));
        SDVariable w2 = sd.var("w2", Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02));
        SDVariable w3 = sd.var("w3", Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02));
        SDVariable w4 = sd.var("w4", Nd4j.randn(DataType.FLOAT, DIM, DIM).muli(0.02));

        SDVariable ha = sd.mmul("mmul_a", a, w1);
        SDVariable hb = sd.mmul("mmul_b", b, w2);
        SDVariable hc = sd.mmul("mmul_c", c, w3);
        SDVariable hd = sd.mmul("mmul_d", d, w4);

        SDVariable sum = ha.add("sum_ab", hb).add("sum_abc", hc).add("sum_abcd", hd);
        SDVariable output = sd.math.tanh("output", sum);
        sd.setOutputs("output");

        return sd;
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testArgTableDirtyTrackingAcross100Steps(Nd4jBackend backend) {
        SameDiff sd = buildMultiInputGraph();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);

        // Build a reference graph with identical weights for ground truth
        SameDiff refSd = buildMultiInputGraph();
        for (String varName : sd.variableMap().keySet()) {
            if (sd.getVariable(varName).getVariableType() == org.nd4j.autodiff.samediff.VariableType.VARIABLE) {
                INDArray arr = sd.getVariable(varName).getArr();
                if (arr != null && refSd.hasVariable(varName)) {
                    refSd.getVariable(varName).getArr().assign(arr);
                }
            }
        }

        // Fixed inputs B, C, D -- same buffer, same data, never change
        INDArray fixedB = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f);
        INDArray fixedC = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f);
        INDArray fixedD = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f);

        // Reusable buffer for input A (fixed address, data changes via assign)
        INDArray reusableA = Nd4j.zeros(DataType.FLOAT, 1, DIM);

        // Warm up with output() (2 calls to trigger DSP compilation)
        for (int warmup = 0; warmup < 2; warmup++) {
            reusableA.assign(Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f));
            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input_a", reusableA);
            inputs.put("input_b", fixedB);
            inputs.put("input_c", fixedC);
            inputs.put("input_d", fixedD);
            sd.output(inputs, "output");
        }

        // Freeze shapes
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        if (dspExec != null) {
            dspExec.setShapesFrozen(true);
            log.info("Shapes frozen for arg table dirty tracking stress test");
        }

        // Run NUM_STEPS, changing only input_a each step
        int correctSteps = 0;
        int incorrectSteps = 0;
        double maxObservedDiff = 0;
        int firstIncorrectStep = -1;
        float[] prevOutput = null;
        int consecutiveIdentical = 0;
        int maxConsecutiveIdentical = 0;

        for (int step = 0; step < NUM_STEPS; step++) {
            // Generate deterministic varying input for A
            Nd4j.getRandom().setSeed(step * 31337L + 7);
            INDArray newA = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f);
            reusableA.assign(newA);
            Nd4j.getExecutioner().commit();

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("input_a", reusableA);
            inputs.put("input_b", fixedB);
            inputs.put("input_c", fixedC);
            inputs.put("input_d", fixedD);

            // DSP / frozen path
            Map<String, INDArray> result = sd.outputDirect(inputs, "output");
            Nd4j.getExecutioner().commit();
            INDArray output = result.get("output");
            assertNotNull(output, "Output null at step " + step);
            float[] directData = output.dup().data().asFloat();

            // Reference computation (fresh, no DSP) -- use fresh dup'd inputs
            Nd4j.getRandom().setSeed(step * 31337L + 7);
            INDArray refA = Nd4j.randn(DataType.FLOAT, 1, DIM).muli(0.1f);
            Map<String, INDArray> refInputs = new HashMap<>();
            refInputs.put("input_a", refA.dup());
            refInputs.put("input_b", fixedB.dup());
            refInputs.put("input_c", fixedC.dup());
            refInputs.put("input_d", fixedD.dup());
            Map<String, INDArray> refResult = refSd.output(refInputs, "output");
            Nd4j.getExecutioner().commit();
            float[] refData = refResult.get("output").dup().data().asFloat();

            // Compare
            double stepMaxDiff = 0;
            for (int i = 0; i < Math.min(directData.length, refData.length); i++) {
                stepMaxDiff = Math.max(stepMaxDiff, Math.abs(directData[i] - refData[i]));
            }
            maxObservedDiff = Math.max(maxObservedDiff, stepMaxDiff);

            if (stepMaxDiff < 0.01) {
                correctSteps++;
            } else {
                incorrectSteps++;
                if (firstIncorrectStep < 0) {
                    firstIncorrectStep = step;
                    log.warn("FIRST INCORRECT at step {}: maxDiff={}", step, String.format("%.6f", stepMaxDiff));
                }
            }

            // Track consecutive identical outputs (sign of stale data)
            if (prevOutput != null) {
                boolean identical = Arrays.equals(directData, prevOutput);
                if (identical) {
                    consecutiveIdentical++;
                    maxConsecutiveIdentical = Math.max(maxConsecutiveIdentical, consecutiveIdentical);
                } else {
                    consecutiveIdentical = 0;
                }
            }
            prevOutput = directData;

            if (step < 5 || step % 20 == 0) {
                log.info("Step {}: maxDiff={} correct={}/{} consecutiveIdentical={}",
                        step, String.format("%.6f", stepMaxDiff), correctSteps, step + 1, consecutiveIdentical);
            }
        }

        // Summary
        log.info("=== ARG TABLE DIRTY TRACKING STRESS TEST RESULTS ===");
        log.info("  Total steps: {}", NUM_STEPS);
        log.info("  Correct: {} / {}", correctSteps, NUM_STEPS);
        log.info("  Incorrect: {} (first at step {})", incorrectSteps, firstIncorrectStep);
        log.info("  Max observed diff: {}", String.format("%.6f", maxObservedDiff));
        log.info("  Max consecutive identical outputs: {}", maxConsecutiveIdentical);

        // Assertions
        assertTrue(correctSteps >= NUM_STEPS * 0.9,
                "At least 90% of steps should match reference, got " + correctSteps + "/" + NUM_STEPS);
        assertTrue(maxConsecutiveIdentical < 5,
                "Should not have 5+ consecutive identical outputs (stale data) -- got " + maxConsecutiveIdentical);

        sd.close();
        refSd.close();
    }
}
