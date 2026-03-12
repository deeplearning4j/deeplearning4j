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

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for batched GEMM optimization in DSP execution.
 *
 * The batched GEMM optimization groups matmul ops with identical (M,N,K,transA,transB,dtype)
 * into single cublasGemmBatchedEx calls to reduce CUDA graph node count.
 *
 * These tests verify correctness of non-consecutive batching where view/reshape ops
 * exist between matmul slots.
 */
public class BatchedGemmTest extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(BatchedGemmTest.class);
    private static final double TOL = 1e-3;

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Helper: run a graph with standard execution, then with frozen DSP + batchedGemm,
     * compare results over multiple executions to exercise shape freezing and detection.
     */
    private void runBatchedGemmComparison(SameDiff sd, String outputName, INDArray input, int numRuns) {
        // Standard execution (baseline)
        Map<String, INDArray> standardResult = sd.output(Map.of("x", input), outputName);
        INDArray expected = standardResult.get(outputName).dup();
        log.info("Standard result shape: {} first 4 values: [{}, {}, {}, {}]",
                Arrays.toString(expected.shape()),
                expected.getDouble(0), expected.getDouble(1),
                expected.getDouble(2), expected.getDouble(3));

        // Compile DSP plan
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        Nd4j.getEnvironment().setDspBatchedGemm(true);

        GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                List.of(outputName), GraphExecutionMode.SLOT_BY_SLOT, true);
        log.info("Compiled DSP plan, effective mode: {}", mode);

        // Run 0: warmup (populates shape caches)
        Map<String, INDArray> warmupResult = sd.outputDirect(Map.of("x", input), outputName);
        INDArray warmupActual = warmupResult.get(outputName).dup();
        double warmupDiff = expected.sub(warmupActual).amaxNumber().doubleValue();
        log.info("Warmup: max diff = {}", warmupDiff);
        assertTrue(warmupDiff < TOL, "Warmup: max diff " + warmupDiff + " exceeds tolerance");

        // Freeze shapes after warmup — this enables batched GEMM detection on next exec
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec, "DSP executor should be non-null after first execution");
        dspExec.setShapesFrozen(true);
        log.info("Shapes frozen, running {} more executions to trigger detection + batching", numRuns);

        // Run 1+ : exec 0 after freeze populates caches, exec 1 triggers detection, exec 2+ uses batching
        for (int run = 0; run < numRuns; run++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Map.of("x", input), outputName);
            INDArray actual = dspResult.get(outputName).dup();

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Frozen run {}: max diff = {}, first 4 values: [{}, {}, {}, {}]",
                    run, maxDiff,
                    actual.getDouble(0), actual.getDouble(1),
                    actual.getDouble(2), actual.getDouble(3));
            assertTrue(maxDiff < TOL, "Run " + run + ": max diff " + maxDiff + " exceeds tolerance " + TOL);
        }

        Nd4j.getEnvironment().setDspBatchedGemm(false);
    }

    /**
     * Test 1: Three parallel matmuls (Q/K/V pattern) with reshapes between them.
     *
     * All three matmuls share the same input x and have identical (M=1, K=576, N=576).
     * They should be batchable even with reshapes/relus between them.
     */
    @Test
    @DisplayName("Batched GEMM: Q/K/V parallel matmuls with reshapes")
    public void testParallelMatmulsWithReshapes() {
        SameDiff sd = SameDiff.create();

        int M = 1, K = 576, N = 576;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, M, K);
        SDVariable wq = sd.constant("wq", Nd4j.randn(DataType.FLOAT, K, N).mul(0.02));
        SDVariable wk = sd.constant("wk", Nd4j.randn(DataType.FLOAT, K, N).mul(0.02));
        SDVariable wv = sd.constant("wv", Nd4j.randn(DataType.FLOAT, K, N).mul(0.02));

        // Q projection + reshape + relu
        SDVariable q = sd.mmul("q_proj", x, wq);
        SDVariable qReshaped = sd.reshape("q_reshape", q, 1, 1, N);
        SDVariable qAct = sd.nn.relu("q_act", qReshaped, 0);
        SDVariable qFlat = sd.reshape("q_flat", qAct, M, N);

        // K projection + reshape + relu
        SDVariable k = sd.mmul("k_proj", x, wk);
        SDVariable kReshaped = sd.reshape("k_reshape", k, 1, 1, N);
        SDVariable kAct = sd.nn.relu("k_act", kReshaped, 0);
        SDVariable kFlat = sd.reshape("k_flat", kAct, M, N);

        // V projection + reshape + relu
        SDVariable v = sd.mmul("v_proj", x, wv);
        SDVariable vReshaped = sd.reshape("v_reshape", v, 1, 1, N);
        SDVariable vAct = sd.nn.relu("v_act", vReshaped, 0);
        SDVariable vFlat = sd.reshape("v_flat", vAct, M, N);

        // Combine
        SDVariable sum = qFlat.add("qk_sum", kFlat);
        SDVariable out = sum.add("out", vFlat);

        INDArray input = Nd4j.randn(DataType.FLOAT, M, K);

        try {
            runBatchedGemmComparison(sd, "out", input, 5);
        } finally {
            sd.close();
        }
    }

    /**
     * Test 2: Two-layer FFN with interleaved ops.
     *
     * Layer 1 and Layer 2 have matmuls with the SAME shapes but data dependencies
     * between layers (l2_up depends on l1_out which depends on l1_down).
     * The batched GEMM grouping must respect these data dependencies.
     */
    @Test
    @DisplayName("Batched GEMM: two-layer FFN with interleaved ops")
    public void testTwoLayerFFNInterleaved() {
        SameDiff sd = SameDiff.create();

        int hidden = 576, ffn = 1536;

        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, hidden);

        SDVariable w1_up = sd.constant("w1_up", Nd4j.randn(DataType.FLOAT, hidden, ffn).mul(0.02));
        SDVariable w1_down = sd.constant("w1_down", Nd4j.randn(DataType.FLOAT, ffn, hidden).mul(0.02));
        SDVariable w2_up = sd.constant("w2_up", Nd4j.randn(DataType.FLOAT, hidden, ffn).mul(0.02));
        SDVariable w2_down = sd.constant("w2_down", Nd4j.randn(DataType.FLOAT, ffn, hidden).mul(0.02));

        // Layer 1
        SDVariable l1_up = sd.mmul("l1_up", x, w1_up);
        SDVariable l1_act = sd.nn.relu("l1_act", l1_up, 0);
        SDVariable l1_down = sd.mmul("l1_down", l1_act, w1_down);
        SDVariable l1_out = x.add("l1_out", l1_down);

        // Layer 2
        SDVariable l2_up = sd.mmul("l2_up", l1_out, w2_up);
        SDVariable l2_act = sd.nn.relu("l2_act", l2_up, 0);
        SDVariable l2_down = sd.mmul("l2_down", l2_act, w2_down);
        SDVariable out = l1_out.add("out", l2_down);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, hidden);

        try {
            runBatchedGemmComparison(sd, "out", input, 5);
        } finally {
            sd.close();
        }
    }

    /**
     * Test 3: Four independent matmuls feeding into a final sum.
     * No data dependencies between the matmuls themselves -- pure batching opportunity.
     */
    @Test
    @DisplayName("Batched GEMM: four independent same-shape matmuls")
    public void testFourIndependentMatmuls() {
        SameDiff sd = SameDiff.create();

        int M = 1, K = 128, N = 128;
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, M, K);

        SDVariable[] weights = new SDVariable[4];
        SDVariable[] projs = new SDVariable[4];
        for (int i = 0; i < 4; i++) {
            weights[i] = sd.constant("w" + i, Nd4j.randn(DataType.FLOAT, K, N).mul(0.02));
            projs[i] = sd.mmul("proj" + i, x, weights[i]);
        }

        SDVariable sum01 = projs[0].add("sum01", projs[1]);
        SDVariable sum23 = projs[2].add("sum23", projs[3]);
        SDVariable out = sum01.add("out", sum23);

        INDArray input = Nd4j.randn(DataType.FLOAT, M, K);

        try {
            runBatchedGemmComparison(sd, "out", input, 5);
        } finally {
            sd.close();
        }
    }
}
