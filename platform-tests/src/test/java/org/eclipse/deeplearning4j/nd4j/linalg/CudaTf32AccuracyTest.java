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
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Regression tests for TF32 precision in cuBLAS matmul operations.
 *
 * TF32 uses 10-bit mantissa (vs 23-bit for FP32) but accumulates in FP32.
 * For typical LLM inference tensors (values ~[-2, 2]), the relative error
 * should be < 0.1% which is well within acceptable bounds.
 *
 * These tests verify that enabling TF32 does NOT break accuracy beyond
 * acceptable thresholds for common LLM operations.
 */
public class CudaTf32AccuracyTest extends BaseNd4jTestWithBackends {

    private boolean originalTf32Setting;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeEach
    public void saveTf32Setting() {
        originalTf32Setting = Nd4j.getEnvironment().cublasTf32Enabled();
    }

    @AfterEach
    public void restoreTf32Setting() {
        Nd4j.getEnvironment().setCublasTf32Enabled(originalTf32Setting);
    }

    /**
     * Test that TF32 matmul for SmolDocling-sized logits projection
     * matches IEEE FP32 within acceptable tolerance.
     * Shape: [1, 576] x [576, 49280] -> [1, 49280]
     */
    @Test
    public void testTf32LogitsProjectionAccuracy() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // SmolDocling dimensions: hidden=576, vocab=49280
        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02);

        // Reference: IEEE FP32
        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        INDArray refOutput = Nd4j.create(DataType.FLOAT, 1, 49280);
        runMatmul(activations, weights, refOutput);

        // TF32
        Nd4j.getEnvironment().setCublasTf32Enabled(true);
        INDArray tf32Output = Nd4j.create(DataType.FLOAT, 1, 49280);
        runMatmul(activations, weights, tf32Output);

        assertArrayEquals(new long[]{1, 49280}, tf32Output.shape());

        // TF32 relative error should be < 0.5% for typical LLM values
        double maxRelErr = maxRelativeDiff(refOutput, tf32Output);
        assertTrue(maxRelErr < 0.005,
                "TF32 logits projection relative error too high: " + maxRelErr);
    }

    /**
     * Test TF32 accuracy for attention-sized matmul: QK^T computation.
     * Shape: [1, 64] x [64, 2048] -> [1, 2048]
     * (simulates single-head Q @ K^T for seqK=2048)
     */
    @Test
    public void testTf32AttentionQKAccuracy() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray q = Nd4j.randn(DataType.FLOAT, 1, 64).mul(0.5);
        INDArray kT = Nd4j.randn(DataType.FLOAT, 64, 2048).mul(0.5);

        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        INDArray refScores = Nd4j.create(DataType.FLOAT, 1, 2048);
        runMatmul(q, kT, refScores);

        Nd4j.getEnvironment().setCublasTf32Enabled(true);
        INDArray tf32Scores = Nd4j.create(DataType.FLOAT, 1, 2048);
        runMatmul(q, kT, tf32Scores);

        double maxRelErr = maxRelativeDiff(refScores, tf32Scores);
        assertTrue(maxRelErr < 0.005,
                "TF32 attention QK relative error too high: " + maxRelErr);
    }

    /**
     * Test TF32 accuracy for FFN matmul: x @ W_gate.
     * Shape: [1, 576] x [576, 1536] -> [1, 1536]
     */
    @Test
    public void testTf32FFNMatmulAccuracy() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray x = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray wGate = Nd4j.randn(DataType.FLOAT, 576, 1536).mul(0.02);

        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        INDArray refOut = Nd4j.create(DataType.FLOAT, 1, 1536);
        runMatmul(x, wGate, refOut);

        Nd4j.getEnvironment().setCublasTf32Enabled(true);
        INDArray tf32Out = Nd4j.create(DataType.FLOAT, 1, 1536);
        runMatmul(x, wGate, tf32Out);

        double maxRelErr = maxRelativeDiff(refOut, tf32Out);
        assertTrue(maxRelErr < 0.005,
                "TF32 FFN matmul relative error too high: " + maxRelErr);
    }

    /**
     * Test TF32 with mixed-precision (FP16 weights, FP32 activations).
     * This is the typical VLM decode path with FP16 weight pre-casting.
     * Shape: [1, 576] x [576, 49280] -> [1, 49280]
     */
    @Test
    public void testTf32MixedPrecisionLogitsAccuracy() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray weightsF32 = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02);
        INDArray weightsF16 = weightsF32.castTo(DataType.HALF);

        // Reference: FP32 weights, no TF32
        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        INDArray refOutput = Nd4j.create(DataType.FLOAT, 1, 49280);
        runMatmul(activations, weightsF32, refOutput);

        // Test: FP16 weights + TF32 (typical decode config)
        Nd4j.getEnvironment().setCublasTf32Enabled(true);
        INDArray tf32Output = Nd4j.create(DataType.FLOAT, 1, 49280);
        runMatmul(activations, weightsF16, tf32Output);

        // Mixed precision + TF32: allow slightly more tolerance
        double maxRelErr = maxRelativeDiff(refOutput, tf32Output);
        assertTrue(maxRelErr < 0.01,
                "TF32 mixed-precision logits relative error too high: " + maxRelErr);
    }

    /**
     * Test that toggling TF32 on/off doesn't corrupt state.
     */
    @Test
    public void testTf32ToggleDoesNotCorruptState() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray a = Nd4j.randn(DataType.FLOAT, 1, 128).mul(0.1);
        INDArray b = Nd4j.randn(DataType.FLOAT, 128, 256).mul(0.02);
        INDArray ref = Nd4j.create(DataType.FLOAT, 1, 256);

        // Baseline
        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        runMatmul(a, b, ref);

        // Toggle on, off, on, off
        for (int i = 0; i < 4; i++) {
            boolean tf32 = (i % 2 == 0);
            Nd4j.getEnvironment().setCublasTf32Enabled(tf32);
            INDArray out = Nd4j.create(DataType.FLOAT, 1, 256);
            runMatmul(a, b, out);

            double maxRelErr = maxRelativeDiff(ref, out);
            assertTrue(maxRelErr < 0.005,
                    "Toggle iteration " + i + " (tf32=" + tf32 + ") error too high: " + maxRelErr);
        }

        // Final: back to IEEE should be exact
        Nd4j.getEnvironment().setCublasTf32Enabled(false);
        INDArray finalOut = Nd4j.create(DataType.FLOAT, 1, 256);
        runMatmul(a, b, finalOut);
        double finalErr = maxRelativeDiff(ref, finalOut);
        assertTrue(finalErr < 1e-6,
                "IEEE FP32 after toggle should be exact, got error: " + finalErr);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void runMatmul(INDArray a, INDArray b, INDArray c) {
        DynamicCustomOp op = DynamicCustomOp.builder("matmul")
                .addInputs(a, b)
                .addOutputs(c)
                .addIntegerArguments(0, 0, 0)  // transA=false, transB=false, transC=false
                .build();
        Nd4j.getExecutioner().exec(op);
    }

    private double maxRelativeDiff(INDArray ref, INDArray test) {
        float[] refData = ref.data().asFloat();
        float[] testData = test.data().asFloat();
        double maxRel = 0;
        for (int i = 0; i < refData.length; i++) {
            float refVal = refData[i];
            float testVal = testData[i];
            if (Math.abs(refVal) < 1e-6) continue;  // skip near-zero
            double rel = Math.abs(refVal - testVal) / Math.abs(refVal);
            maxRel = Math.max(maxRel, rel);
        }
        return maxRel;
    }

    private boolean isCudaBackend() {
        try {
            return Nd4j.getBackend().getClass().getSimpleName().toLowerCase().contains("cuda") ||
                    Nd4j.getBackend().getClass().getSimpleName().toLowerCase().contains("jcublas");
        } catch (Exception e) {
            return false;
        }
    }
}
