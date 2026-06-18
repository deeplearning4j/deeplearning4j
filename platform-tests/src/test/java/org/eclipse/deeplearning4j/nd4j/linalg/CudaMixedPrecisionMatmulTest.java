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

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CudaMixedPrecisionMatmulTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void testRowVectorFloatByHalfCWeightsMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 128).mul(0.1);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 128, 256).mul(0.02).castTo(DataType.HALF);
        INDArray actual = Nd4j.create(DataType.FLOAT, 1, 256);

        runMatmul(activations, weights, actual);
        float[] expected = manualRowVectorMatmul(activations, weights.castTo(DataType.FLOAT));

        assertArrayEquals(new long[]{1, 256}, actual.shape(), "Unexpected output shape");
        assertEquals(DataType.FLOAT, actual.dataType(), "Output should stay FLOAT");
        assertTrue(maxAbsDiff(actual, expected) < 1e-3, "C-order mixed matmul drifted");
    }

    @Test
    public void testRowVectorFloatByHalfFWeightsMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 128).mul(0.1);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 128, 256).dup('f').mul(0.02).castTo(DataType.HALF);
        INDArray actual = Nd4j.create(DataType.FLOAT, 1, 256);

        runMatmul(activations, weights, actual);
        float[] expected = manualRowVectorMatmul(activations, weights.castTo(DataType.FLOAT));

        assertArrayEquals(new long[]{1, 256}, actual.shape(), "Unexpected output shape");
        assertEquals(DataType.FLOAT, actual.dataType(), "Output should stay FLOAT");
        assertTrue(maxAbsDiff(actual, expected) < 1e-3, "F-order mixed matmul drifted");
    }

    @Test
    public void testRank3RowVectorFloatByHalfMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 1, 128).mul(0.1);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 128, 256).mul(0.02).castTo(DataType.HALF);
        INDArray actual = Nd4j.create(DataType.FLOAT, 1, 1, 256);

        runMatmul(activations, weights, actual);

        INDArray flatActivations = activations.reshape('c', 1, 128);
        float[] expected = manualRowVectorMatmul(flatActivations, weights.castTo(DataType.FLOAT));

        assertArrayEquals(new long[]{1, 1, 256}, actual.shape(), "Unexpected rank-3 output shape");
        assertEquals(DataType.FLOAT, actual.dataType(), "Output should stay FLOAT");
        assertTrue(maxAbsDiff(actual.reshape('c', 1, 256), expected) < 1e-3, "Rank-3 mixed matmul drifted");
    }

    @Test
    public void testRepeatedRowVectorFloatByHalfMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 128).mul(0.1);
        INDArray weightsA = Nd4j.randn(DataType.FLOAT, 128, 192).mul(0.02).castTo(DataType.HALF);
        INDArray weightsB = Nd4j.randn(DataType.FLOAT, 128, 96).mul(0.02).castTo(DataType.HALF);
        INDArray outA = Nd4j.create(DataType.FLOAT, 1, 192);
        INDArray outB = Nd4j.create(DataType.FLOAT, 1, 96);

        runMatmul(activations, weightsA, outA);
        runMatmul(activations, weightsB, outB);

        float[] expectedA = manualRowVectorMatmul(activations, weightsA.castTo(DataType.FLOAT));
        float[] expectedB = manualRowVectorMatmul(activations, weightsB.castTo(DataType.FLOAT));

        assertTrue(maxAbsDiff(outA, expectedA) < 1e-3, "First repeated row-vector matmul drifted");
        assertTrue(maxAbsDiff(outB, expectedB) < 1e-3, "Second repeated row-vector matmul drifted");
    }

    @Test
    public void testRepeatedTripleRowVectorFloatByHalfMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 128).mul(0.1);
        INDArray weightsQ = Nd4j.randn(DataType.FLOAT, 128, 192).mul(0.02).castTo(DataType.HALF);
        INDArray weightsK = Nd4j.randn(DataType.FLOAT, 128, 192).mul(0.02).castTo(DataType.HALF);
        INDArray weightsV = Nd4j.randn(DataType.FLOAT, 128, 192).mul(0.02).castTo(DataType.HALF);
        INDArray outQ = Nd4j.create(DataType.FLOAT, 1, 192);
        INDArray outK = Nd4j.create(DataType.FLOAT, 1, 192);
        INDArray outV = Nd4j.create(DataType.FLOAT, 1, 192);

        runMatmul(activations, weightsQ, outQ);
        runMatmul(activations, weightsK, outK);
        runMatmul(activations, weightsV, outV);

        float[] expectedQ = manualRowVectorMatmul(activations, weightsQ.castTo(DataType.FLOAT));
        float[] expectedK = manualRowVectorMatmul(activations, weightsK.castTo(DataType.FLOAT));
        float[] expectedV = manualRowVectorMatmul(activations, weightsV.castTo(DataType.FLOAT));

        assertTrue(maxAbsDiff(outQ, expectedQ) < 1e-3, "First triple row-vector matmul drifted");
        assertTrue(maxAbsDiff(outK, expectedK) < 1e-3, "Second triple row-vector matmul drifted");
        assertTrue(maxAbsDiff(outV, expectedV) < 1e-3, "Third triple row-vector matmul drifted");
    }

    @Test
    public void testRepeatedRank3RowVectorFloatByHalfMatchesReference() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 1, 128).mul(0.1);
        INDArray weightsA = Nd4j.randn(DataType.FLOAT, 128, 192).mul(0.02).castTo(DataType.HALF);
        INDArray weightsB = Nd4j.randn(DataType.FLOAT, 128, 96).mul(0.02).castTo(DataType.HALF);
        INDArray outA = Nd4j.create(DataType.FLOAT, 1, 1, 192);
        INDArray outB = Nd4j.create(DataType.FLOAT, 1, 1, 96);

        runMatmul(activations, weightsA, outA);
        runMatmul(activations, weightsB, outB);

        INDArray flatActivations = activations.reshape('c', 1, 128);
        float[] expectedA = manualRowVectorMatmul(flatActivations, weightsA.castTo(DataType.FLOAT));
        float[] expectedB = manualRowVectorMatmul(flatActivations, weightsB.castTo(DataType.FLOAT));

        assertTrue(maxAbsDiff(outA.reshape('c', 1, 192), expectedA) < 1e-3, "First repeated rank-3 matmul drifted");
        assertTrue(maxAbsDiff(outB.reshape('c', 1, 96), expectedB) < 1e-3, "Second repeated rank-3 matmul drifted");
    }

    @Test
    public void testDecoderLogitsProjectionFloatHalfLargeVocab() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // Decoder logits projection: [1, 576] x [576, 49280] -> [1, 49280]
        // This targets the cuBLAS Lt fast path for large vocab output projection
        Nd4j.getRandom().setSeed(12345);
        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray weights = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02).castTo(DataType.HALF);
        INDArray actual = Nd4j.create(DataType.FLOAT, 1, 49280);

        runMatmul(activations, weights, actual);

        float[] expected = manualRowVectorMatmul(activations, weights.castTo(DataType.FLOAT));

        assertArrayEquals(new long[]{1, 49280}, actual.shape(), "Unexpected logits output shape");
        assertEquals(DataType.FLOAT, actual.dataType(), "Logits output should stay FLOAT");
        assertTrue(maxAbsDiff(actual, expected) < 1e-2, "Large vocab logits matmul drifted");

        // Verify no NaN values in output (regression check)
        assertFalse(actual.isNaN().any(), "Logits output should not contain NaN");
    }

    @Test
    public void testDecoderLogitsProjectionRepeatedLargeVocab() {
        Assumptions.assumeTrue(isCudaBackend(), "Test requires CUDA backend");

        // Test repeated logits projections (q/k/v or gate/up pattern)
        Nd4j.getRandom().setSeed(12345);
        INDArray activations = Nd4j.randn(DataType.FLOAT, 1, 576).mul(0.1);
        INDArray weightsA = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02).castTo(DataType.HALF);
        INDArray weightsB = Nd4j.randn(DataType.FLOAT, 576, 49280).mul(0.02).castTo(DataType.HALF);
        INDArray outA = Nd4j.create(DataType.FLOAT, 1, 49280);
        INDArray outB = Nd4j.create(DataType.FLOAT, 1, 49280);

        runMatmul(activations, weightsA, outA);
        runMatmul(activations, weightsB, outB);

        float[] expectedA = manualRowVectorMatmul(activations, weightsA.castTo(DataType.FLOAT));
        float[] expectedB = manualRowVectorMatmul(activations, weightsB.castTo(DataType.FLOAT));

        assertTrue(maxAbsDiff(outA, expectedA) < 1e-2, "First repeated logits matmul drifted");
        assertTrue(maxAbsDiff(outB, expectedB) < 1e-2, "Second repeated logits matmul drifted");
        assertFalse(outA.isNaN().any(), "First logits output should not contain NaN");
        assertFalse(outB.isNaN().any(), "Second logits output should not contain NaN");
    }

    private static boolean isCudaBackend() {
        return Nd4j.getExecutioner().getClass().getSimpleName().toLowerCase().contains("cuda");
    }

    private static void runMatmul(INDArray a, INDArray b, INDArray out) {
        DynamicCustomOp matmul = DynamicCustomOp.builder("matmul")
                .addInputs(a, b)
                .addOutputs(out)
                .build();
        Nd4j.getExecutioner().exec(matmul);
    }

    private static double maxAbsDiff(INDArray actual, float[] expected) {
        float[] actualValues = actual.dup('c').reshape(actual.length()).data().asFloat();
        assertEquals(expected.length, actualValues.length, "Length mismatch");

        double maxDiff = 0.0;
        for (int i = 0; i < expected.length; i++) {
            maxDiff = Math.max(maxDiff, Math.abs(actualValues[i] - expected[i]));
        }
        return maxDiff;
    }

    private static float[] manualRowVectorMatmul(INDArray activations, INDArray weights) {
        float[] activationValues = activations.dup('c').reshape(activations.length()).data().asFloat();
        float[] weightValues = weights.dup('c').reshape(weights.length()).data().asFloat();
        int k = (int) activations.size(1);
        int n = (int) weights.size(1);
        float[] out = new float[n];

        for (int col = 0; col < n; col++) {
            double sum = 0.0;
            for (int i = 0; i < k; i++) {
                sum += activationValues[i] * weightValues[i * n + col];
            }
            out[col] = (float) sum;
        }
        return out;
    }
}
