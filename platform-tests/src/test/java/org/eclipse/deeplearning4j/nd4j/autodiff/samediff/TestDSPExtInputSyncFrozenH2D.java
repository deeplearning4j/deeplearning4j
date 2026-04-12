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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that external input sync (H2D copies) works correctly when shapes are
 * frozen and the plan is replaying via CUDA graph.
 *
 * <p>Background: When DSP shapes are frozen and replay occurs, the input data
 * must be copied from host to device (H2D) each step via the {@code tl_dspExecutionStream}
 * path. If H2D sync is broken, the GPU will read stale data from a prior step
 * and produce identical outputs regardless of input changes.</p>
 *
 * <p>Key assertion: different inputs must produce different outputs. If outputs
 * are identical across steps despite input changes, H2D sync is broken.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPExtInputSyncFrozenH2D extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    @AfterEach
    void cleanup() {
        Nd4j.getExecutioner().commit();
    }

    /**
     * Test 1: Basic matmul + add graph with frozen shape replay.
     * Changes input data each step and verifies outputs change accordingly.
     */
    @Test
    @DisplayName("H2D sync: matmul+add produces different outputs for different inputs under frozen replay")
    public void testH2DSyncMatmulAdd() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8).muli(0.1));
        SDVariable b = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 1, 8));
        SDVariable mm = sd.mmul("mm", x, w);
        SDVariable out = mm.add("out", b);

        enableDsp(sd);

        // Warmup DSP
        INDArray input0 = Nd4j.createFromArray(new float[][]{{0.5f, 1.5f, 2.5f, 3.5f}});
        Map<String, INDArray> warmupResult = sd.outputDirect(Collections.singletonMap("x", input0), "out");
        INDArray warmupActual = warmupResult.get("out").dup();

        // Standard reference for warmup input
        Map<String, INDArray> warmupStdResult = sd.output(Collections.singletonMap("x", input0), "out");
        INDArray expected0 = warmupStdResult.get("out").dup();

        double warmupDiff = expected0.sub(warmupActual).amaxNumber().doubleValue();
        log.info("Warmup: max diff = {}", warmupDiff);
        assertTrue(warmupDiff < TOL, "Warmup: max diff " + warmupDiff + " exceeds tolerance");

        // Freeze shapes
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec, "DSP executor should exist after warmup");
        dspExec.setShapesFrozen(true);
        log.info("Shapes frozen");

        // Run 5 steps with different inputs (starting from step=1, all different from warmup)
        INDArray[] allResults = new INDArray[6]; // warmup + 5 steps
        allResults[0] = warmupActual;

        for (int step = 1; step <= 5; step++) {
            // Different input each step (all distinct from warmup input)
            INDArray input = Nd4j.createFromArray(new float[][]{{
                    (step + 10) * 1.0f, (step + 10) * 2.0f, (step + 10) * 3.0f, (step + 10) * 4.0f
            }});

            // Standard reference for this input
            Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
            INDArray expected = stdResult.get("out").dup();

            // DSP frozen replay
            Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
            INDArray actual = dspResult.get("out").dup();
            allResults[step] = actual;

            // Must match standard execution
            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Step {}: max diff from reference = {}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Step " + step + ": max diff " + maxDiff + " exceeds tolerance " + TOL);

            // Must NOT be identical to step 0's result (detects stale H2D)
            double diffFromStep0 = actual.sub(allResults[0]).amaxNumber().doubleValue();
            assertTrue(diffFromStep0 > 1e-5,
                    "Step " + step + " output is identical to step 0 despite different input — "
                            + "H2D sync likely broken (stale GPU data). diffFromStep0=" + diffFromStep0);
        }

        sd.close();
    }

    /**
     * Test 2: Multi-input graph (two placeholders) with frozen replay.
     * Verifies H2D sync works for multiple external inputs.
     */
    @Test
    @DisplayName("H2D sync: multi-input graph under frozen replay")
    public void testH2DSyncMultiInput() {
        SameDiff sd = SameDiff.create();
        SDVariable a = sd.placeHolder("a", DataType.FLOAT, 1, 4);
        SDVariable b = sd.placeHolder("b", DataType.FLOAT, 1, 4);
        SDVariable sum = a.add("sum", b);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 2).muli(0.1));
        SDVariable out = sd.mmul("out", sum, w);

        enableDsp(sd);

        // Warmup
        INDArray a0 = Nd4j.createFromArray(new float[][]{{1, 1, 1, 1}});
        INDArray b0 = Nd4j.createFromArray(new float[][]{{1, 1, 1, 1}});
        Map<String, INDArray> inputs0 = Map.of("a", a0, "b", b0);

        Map<String, INDArray> warmupResult = sd.outputDirect(inputs0, "out");
        INDArray warmupActual = warmupResult.get("out").dup();

        // Freeze
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec);
        dspExec.setShapesFrozen(true);

        // Run with different inputs
        INDArray firstResult = warmupActual;
        for (int step = 0; step < 5; step++) {
            INDArray aStep = Nd4j.createFromArray(new float[][]{{
                    (step + 1) * 1.0f, (step + 1) * 2.0f, (step + 1) * 3.0f, (step + 1) * 4.0f
            }});
            INDArray bStep = Nd4j.createFromArray(new float[][]{{
                    (step + 1) * 0.5f, (step + 1) * 0.5f, (step + 1) * 0.5f, (step + 1) * 0.5f
            }});
            Map<String, INDArray> inputs = Map.of("a", aStep, "b", bStep);

            // Standard reference
            Map<String, INDArray> stdResult = sd.output(inputs, "out");
            INDArray expected = stdResult.get("out").dup();

            // DSP replay
            Map<String, INDArray> dspResult = sd.outputDirect(inputs, "out");
            INDArray actual = dspResult.get("out").dup();

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Multi-input step {}: max diff = {}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Multi-input step " + step + ": max diff " + maxDiff + " exceeds tolerance");

            // Must differ from first result
            double diffFromFirst = actual.sub(firstResult).amaxNumber().doubleValue();
            assertTrue(diffFromFirst > 1e-5,
                    "Multi-input step " + step + " identical to first result — H2D sync broken");

            firstResult = actual;
        }

        sd.close();
    }

    /**
     * Test 3: Repeated replay with identical inputs — must produce identical outputs.
     * This verifies that H2D sync does NOT corrupt data when input is unchanged.
     */
    @Test
    @DisplayName("H2D sync: identical inputs produce identical outputs across replay")
    public void testH2DSyncIdenticalInputs() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8);

        // Warmup
        Map<String, INDArray> warmup = sd.outputDirect(Collections.singletonMap("x", input), "out");
        INDArray warmupActual = warmup.get("out").dup();

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // Freeze
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec);
        dspExec.setShapesFrozen(true);

        // Replay 5 times — all should match expected
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
            INDArray actual = dspResult.get("out").dup();

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Identical-input step {}: max diff = {}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Identical-input step " + step + ": max diff " + maxDiff + " exceeds tolerance");

            // All steps must produce identical results
            double diffFromWarmup = actual.sub(warmupActual).amaxNumber().doubleValue();
            assertTrue(diffFromWarmup < TOL,
                    "Identical-input step " + step + " differs from warmup — diff=" + diffFromWarmup);
        }

        sd.close();
    }
}
