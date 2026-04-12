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
 * Tests the cuBLAS fallback path for matmul ops within DSP segments.
 *
 * <p>Background: DSP segments use Triton for large batched matmuls but fall back
 * to cuBLAS for small/decode-style matmuls (M=1). This test verifies that the
 * cuBLAS fallback path produces correct results within a DSP segment.</p>
 *
 * <p>The decode pattern is typical of autoregressive LLM inference: single-token
 * matmul (batch=1, seq=1) producing [1, 1, hidden] output.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPCuBlasSegmentFallbackIsolation extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-3;

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
     * Test 1: Single M=1 matmul (decode-style) within DSP.
     * This is the minimal cuBLAS fallback case.
     */
    @Test
    @DisplayName("cuBLAS fallback: M=1 matmul (decode-style) correctness")
    public void testM1MatmulFallback() {
        SameDiff sd = SameDiff.create();

        // Decode-style: [1, 1, hidden] × [hidden, vocab] = [1, 1, vocab]
        int hidden = 64;
        int vocab = 128;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, hidden);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, hidden, vocab).muli(0.02));

        // Reshape to 2D for mmul: [1, hidden] × [hidden, vocab] = [1, vocab]
        SDVariable xFlat = sd.reshape("x_flat", x, 1, hidden);
        SDVariable mm = sd.mmul("mm", xFlat, w);
        SDVariable out = sd.reshape("out", mm, 1, 1, vocab);

        enableDsp(sd);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, hidden);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("M=1 matmul: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL, "M=1 matmul: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 2: M=1 matmul with frozen replay — verify correctness across steps.
     */
    @Test
    @DisplayName("cuBLAS fallback: M=1 matmul with frozen replay correctness")
    public void testM1MatmulFrozenReplay() {
        SameDiff sd = SameDiff.create();

        int hidden = 32;
        int vocab = 64;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hidden);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, hidden, vocab).muli(0.02));
        SDVariable out = sd.mmul("out", x, w);

        enableDsp(sd);

        // Warmup
        INDArray input0 = Nd4j.randn(DataType.FLOAT, 1, hidden);
        Map<String, INDArray> warmupResult = sd.outputDirect(Collections.singletonMap("x", input0), "out");
        INDArray warmupActual = warmupResult.get("out").dup();

        // Standard reference for warmup
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input0), "out");
        INDArray expected0 = stdResult.get("out").dup();

        double warmupDiff = expected0.sub(warmupActual).amaxNumber().doubleValue();
        log.info("Warmup: max diff = {}", warmupDiff);
        assertTrue(warmupDiff < TOL, "Warmup: max diff " + warmupDiff + " exceeds tolerance");

        // Freeze
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec);
        dspExec.setShapesFrozen(true);

        // Frozen replay with different inputs
        INDArray[] allResults = new INDArray[6];
        allResults[0] = warmupActual;

        for (int step = 1; step <= 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, hidden).muli(step + 1);

            // Standard reference
            Map<String, INDArray> stdR = sd.output(Collections.singletonMap("x", input), "out");
            INDArray expected = stdR.get("out").dup();

            // DSP replay
            Map<String, INDArray> dspR = sd.outputDirect(Collections.singletonMap("x", input), "out");
            INDArray actual = dspR.get("out").dup();
            allResults[step] = actual;

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Frozen step {}: max diff = {}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Frozen step " + step + ": max diff " + maxDiff + " exceeds tolerance");

            // Must differ from step 0
            double diffFrom0 = actual.sub(allResults[0]).amaxNumber().doubleValue();
            assertTrue(diffFrom0 > 1e-5,
                    "Step " + step + " identical to step 0 — cuBLAS fallback may be reading stale data");
        }

        sd.close();
    }

    /**
     * Test 3: Mixed M=1 and M>1 matmuls in same graph.
     * Verifies cuBLAS fallback coexists with batched/batched Triton matmuls.
     */
    @Test
    @DisplayName("cuBLAS fallback: mixed M=1 and M>1 matmuls in same DSP graph")
    public void testMixedMatmulSizes() {
        SameDiff sd = SameDiff.create();

        int hidden = 32;
        int vocab = 64;

        // M=1 decode path
        SDVariable xDecode = sd.placeHolder("x_decode", DataType.FLOAT, 1, hidden);
        SDVariable wDecode = sd.constant("w_decode", Nd4j.randn(DataType.FLOAT, hidden, vocab).muli(0.02));
        SDVariable mmDecode = sd.mmul("mm_decode", xDecode, wDecode);

        // M=4 batch path
        SDVariable xBatch = sd.placeHolder("x_batch", DataType.FLOAT, 4, hidden);
        SDVariable wBatch = sd.constant("w_batch", Nd4j.randn(DataType.FLOAT, hidden, vocab).muli(0.02));
        SDVariable mmBatch = sd.mmul("mm_batch", xBatch, wBatch);

        // Combine: sum of decode output (broadcast) + batch output
        SDVariable decodeSum = mmDecode.sum("decode_sum", 1); // [1, vocab] -> [vocab]
        SDVariable decodeBroadcast = sd.reshape("decode_bc", decodeSum, 1, vocab);
        SDVariable out = mmBatch.add("out", decodeBroadcast);

        enableDsp(sd);

        INDArray xDecodeArr = Nd4j.randn(DataType.FLOAT, 1, hidden);
        INDArray xBatchArr = Nd4j.randn(DataType.FLOAT, 4, hidden);
        Map<String, INDArray> inputs = Map.of("x_decode", xDecodeArr, "x_batch", xBatchArr);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(inputs, "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP
        Map<String, INDArray> dspResult = sd.outputDirect(inputs, "out");
        INDArray actual = dspResult.get("out").dup();

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Mixed matmul sizes: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Mixed matmul: max diff " + maxDiff + " exceeds tolerance " + TOL);

        // Freeze and replay
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec);
        dspExec.setShapesFrozen(true);

        for (int step = 0; step < 3; step++) {
            INDArray xDecodeStep = Nd4j.randn(DataType.FLOAT, 1, hidden).muli(step + 1);
            INDArray xBatchStep = Nd4j.randn(DataType.FLOAT, 4, hidden).muli(step + 1);
            Map<String, INDArray> stepInputs = Map.of("x_decode", xDecodeStep, "x_batch", xBatchStep);

            Map<String, INDArray> stdR = sd.output(stepInputs, "out");
            INDArray expectedStep = stdR.get("out").dup();

            Map<String, INDArray> dspR = sd.outputDirect(stepInputs, "out");
            INDArray actualStep = dspR.get("out").dup();

            double stepDiff = expectedStep.sub(actualStep).amaxNumber().doubleValue();
            log.info("Mixed frozen step {}: max diff = {}", step, stepDiff);
            assertTrue(stepDiff < TOL,
                    "Mixed frozen step " + step + ": max diff " + stepDiff + " exceeds tolerance");
        }

        sd.close();
    }
}
