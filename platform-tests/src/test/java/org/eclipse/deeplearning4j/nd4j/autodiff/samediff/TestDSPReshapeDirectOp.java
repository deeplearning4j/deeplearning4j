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
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that the standard reshape op works correctly through DSP execution
 * WITHOUT the reshape→reshape_no_copy remapping that was previously in
 * DynamicShapePlanCompiler.
 *
 * These tests verify that DSP can compile and execute graphs containing
 * reshape ops using their native iArgs format [-order, dim0, dim1, ...].
 */
@Slf4j
@DisplayName("DSP Reshape Direct Op Tests")
public class TestDSPReshapeDirectOp {

    /**
     * Basic 2D→2D reshape through DSP. Verifies DSP result matches standard execution.
     */
    @Test
    @DisplayName("testReshape2Dto2DDirectOp")
    public void testReshape2Dto2DDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable out = reshaped.mul("out", sd.constant("scale", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 12, 12, DataType.FLOAT).reshape(2, 6);

        // Standard execution
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false); // Java DSP only for isolation
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        log.info("2D→2D reshape: expected shape={}, actual shape={}", expected.shape(), actual.shape());
        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match");
        assertEquals(expected, actual, "DSP reshape result must match standard execution");

        sd.close();
    }

    /**
     * 3D→2D reshape through DSP — common in transformer decode paths.
     */
    @Test
    @DisplayName("testReshape3Dto2DDirectOp")
    public void testReshape3Dto2DDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 6, 4);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 6, 4));
        SDVariable out = reshaped.add("out", bias);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        // Standard
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual, "3D→2D DSP reshape must match standard");

        sd.close();
    }

    /**
     * Reshape with -1 dimension inference through DSP.
     */
    @Test
    @DisplayName("testReshapeWithNeg1DirectOp")
    public void testReshapeWithNeg1DirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 2, -1);
        SDVariable out = reshaped.mul("out", sd.constant("scale", Nd4j.scalar(3.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(new long[]{2, 12}, actual.shape(), "Inferred shape should be [2,12]");
        assertEquals(expected, actual, "Reshape with -1 must match standard");

        sd.close();
    }

    /**
     * Critical transformer decode pattern: reshape → matmul chain.
     * [1,3,4] → reshape [3,4] → mmul [4,8] → [3,8]
     */
    @Test
    @DisplayName("testReshapeMatmulChainDirectOp")
    public void testReshapeMatmulChainDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable out = sd.mmul("out", reshaped, weights);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 4).mul(0.1);

        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        log.info("Reshape→matmul chain: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "Reshape→matmul DSP result diverges by " + maxDiff);

        sd.close();
    }

    /**
     * Multiple reshapes in sequence — tests DSP handles view chains correctly.
     */
    @Test
    @DisplayName("testMultipleReshapesDirectOp")
    public void testMultipleReshapesDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 3, 4);
        SDVariable r1 = sd.reshape("r1", x, 6, 4);
        SDVariable bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 6, 4).mul(0.5));
        SDVariable added = r1.add("added", bias);
        SDVariable r2 = sd.reshape("r2", added, 2, 12);
        SDVariable out = r2.mul("out", sd.constant("scale", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(2, 3, 4);

        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(expected.shape(), actual.shape());
        assertEquals(expected, actual, "Multiple reshapes in sequence must match standard");

        sd.close();
    }

    /**
     * Attention-style reshape: [1,12,64] → [1,3,4,64]
     * This is the critical Q/K/V split pattern in multi-head attention.
     */
    @Test
    @DisplayName("testAttentionReshapeDirectOp")
    public void testAttentionReshapeDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 12, 64);
        // [1,12,64] → [1,3,4,64] (split 12 into 3 heads × 4 dim)
        SDVariable reshaped = sd.reshape("reshaped", x, 1, 3, 4, 64);
        SDVariable out = reshaped.mul("out", sd.constant("scale", Nd4j.scalar(0.5f)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 12, 64).mul(0.1);

        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(new long[]{1, 3, 4, 64}, actual.shape());
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        log.info("Attention reshape: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-5, "Attention reshape DSP diverges by " + maxDiff);

        sd.close();
    }

    /**
     * Multi-step execution with changing input values — verifies no stale view data.
     */
    @Test
    @DisplayName("testReshapeMultiStepDirectOp")
    public void testReshapeMultiStepDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable out = sd.mmul("out", reshaped, weights);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 4).mul(0.1);

            // Standard
            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
            INDArray expected = stdResult.get("out").dup();

            // DSP
            sd.setDspAutoCompileEnabled(true);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
            INDArray actual = dspResult.get("out");

            double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
            log.info("Step {}: maxDiff={}", step, maxDiff);
            assertTrue(maxDiff < 1e-4, "Step " + step + " diverges by " + maxDiff);
        }

        sd.close();
    }

    /**
     * Reshape + permute chain — simulates Q/K/V head splitting in attention.
     * [1,1,192] → reshape [1,1,3,64] → permute [1,3,1,64] → matmul
     */
    @Test
    @DisplayName("testReshapePermuteMatmulDirectOp")
    public void testReshapePermuteMatmulDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, 192);
        SDVariable reshaped = sd.reshape("reshaped", x, 1, 1, 3, 64);
        SDVariable permuted = sd.permute("permuted", reshaped, 0, 2, 1, 3); // [1,3,1,64]
        // Simulate attention: Q*K^T where K is [1,3,64,5] (5 past tokens)
        SDVariable k = sd.constant("k", Nd4j.randn(DataType.FLOAT, 1, 3, 64, 5).mul(0.1));
        SDVariable scores = sd.mmul("scores", permuted, k); // [1,3,1,5]
        SDVariable out = scores.mul("out", sd.constant("scale", Nd4j.scalar(1.0f / Math.sqrt(64))));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 192).mul(0.1);

        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();

        Map<String, INDArray> dspResult = sd.output(Map.of("x", input), "out");
        INDArray actual = dspResult.get("out");

        assertArrayEquals(expected.shape(), actual.shape(), "Shapes must match");
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        log.info("Reshape+permute+matmul: maxDiff={}", maxDiff);
        assertTrue(maxDiff < 1e-4, "Reshape+permute+matmul diverges by " + maxDiff);

        sd.close();
    }

    /**
     * Also test native DSP execution if available (both Java and native paths).
     */
    @Test
    @DisplayName("testReshapeNativeDspDirectOp")
    public void testReshapeNativeDspDirectOp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4);
        SDVariable reshaped = sd.reshape("reshaped", x, 3, 4);
        SDVariable weights = sd.constant("weights", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable out = sd.mmul("out", reshaped, weights);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 4).mul(0.1);

        // Standard baseline
        Map<String, INDArray> stdResult = sd.output(Map.of("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // Java DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        Map<String, INDArray> javaDspResult = sd.output(Map.of("x", input), "out");
        INDArray javaDsp = javaDspResult.get("out");

        double javaDiff = javaDsp.sub(expected).amaxNumber().doubleValue();
        log.info("Java DSP maxDiff={}", javaDiff);
        assertTrue(javaDiff < 1e-4, "Java DSP diverges by " + javaDiff);

        // Native DSP
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        Map<String, INDArray> nativeDspResult = sd.output(Map.of("x", input), "out");
        INDArray nativeDsp = nativeDspResult.get("out");

        double nativeDiff = nativeDsp.sub(expected).amaxNumber().doubleValue();
        log.info("Native DSP maxDiff={}", nativeDiff);
        assertTrue(nativeDiff < 1e-4, "Native DSP diverges by " + nativeDiff);

        sd.close();
    }
}
