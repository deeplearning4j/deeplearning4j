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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNormLinear;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.BaseNd4jTestWithBackends;

import java.util.Collections;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the fused RMSNorm + Linear op.
 * Verifies that helpers::rmsNormLinear produces identical results to
 * the decomposed path: matmul(rmsNorm(x, gamma, eps), W).
 */
public class RmsNormLinearTest extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    /**
     * Compute invRms = 1 / sqrt(mean(x^2, axis=-1) + eps), shape [M, 1]
     */
    private static INDArray computeInvRms(INDArray x, float eps) {
        INDArray xSq = x.mul(x);
        INDArray meanSq = xSq.mean(true, -1);
        INDArray rms = Transforms.sqrt(meanSq.add(eps));
        return rms.rdiv(1.0);
    }

    static Stream<Arguments> shapes() {
        return Stream.of(
            // M=1 decode path (fused kernel)
            Arguments.of(1, 64, 64),
            Arguments.of(1, 128, 256),
            Arguments.of(1, 1142, 1142),    // SmolDocling hidden dim
            Arguments.of(1, 1142, 3072),    // SmolDocling MLP up-proj
            // M>1 prefill path (rmsNorm + cuBLAS)
            Arguments.of(4, 64, 64),
            Arguments.of(16, 128, 256),
            Arguments.of(32, 512, 1024)
        );
    }

    /**
     * Compare fused rms_norm_linear against manual decomposition:
     *   normalized = x / sqrt(mean(x^2) + eps) * gamma
     *   output = normalized @ W
     */
    @ParameterizedTest
    @MethodSource("shapes")
    public void testRmsNormLinearMatchesDecomposed(int M, int K, int N) {
        float eps = 1e-5f;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.02);

        // Reference: manual decomposition
        INDArray invRms = computeInvRms(x, eps);
        INDArray normalized = x.mul(invRms).mul(gamma);
        INDArray expected = normalized.mmul(W);

        // Fused op
        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        // Tolerance: fused path uses float32 accumulation throughout
        double rtol = 1e-3;
        double atol = 1e-4;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float exp = expected.getFloat(i, j);
                float act = output.getFloat(i, j);
                float diff = Math.abs(exp - act);
                float tol = (float)(atol + rtol * Math.abs(exp));
                assertTrue(diff < tol,
                    String.format("Mismatch at [%d,%d]: expected=%f actual=%f diff=%f tol=%f (M=%d K=%d N=%d)",
                        i, j, exp, act, diff, tol, M, K, N));
            }
        }
    }

    @Test
    public void testDecodePathM1() {
        float eps = 1e-6f;
        int K = 256;
        int N = 512;

        INDArray x = Nd4j.randn(DataType.FLOAT, 1, K);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, K);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.01);

        // Reference
        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mul(gamma).mmul(W);

        // Fused
        INDArray output = Nd4j.create(DataType.FLOAT, 1, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
            "M=1 decode path max diff too large: " + maxDiff);
    }

    @Test
    public void testGammaOnes() {
        float eps = 1e-5f;
        int M = 2, K = 64, N = 32;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.ones(DataType.FLOAT, K);
        INDArray W = Nd4j.randn(DataType.FLOAT, K, N).muli(0.02);

        // With gamma=1, rmsNorm just normalizes
        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mmul(W);

        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, W, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3, "Gamma=ones max diff: " + maxDiff);
    }

    @Test
    public void testNonContiguousProjectionWeight() {
        float eps = 1e-6f;
        int M = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, M, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray invRms = computeInvRms(x, eps);
        INDArray expected = x.mul(invRms).mul(gamma).mmul(weightTransposed);

        INDArray output = Nd4j.create(DataType.FLOAT, M, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, weightTransposed, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3, "Non-contiguous projection weight max diff: " + maxDiff);
    }

    @Test
    public void testRank3InputWithNonContiguousProjectionWeight() {
        float eps = 1e-6f;
        int batch = 1, seqLen = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seqLen, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray x2d = x.reshape('c', batch * seqLen, K);
        INDArray invRms = computeInvRms(x2d, eps);
        INDArray expected = x2d.mul(invRms).mul(gamma)
                .mmul(weightTransposed)
                .reshape('c', batch, seqLen, N);

        INDArray output = Nd4j.create(DataType.FLOAT, batch, seqLen, N);
        Nd4j.exec(new RmsNormLinear(x, gamma, weightTransposed, output, eps));

        double maxDiff = expected.sub(output).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Rank-3 non-contiguous projection weight max diff: " + maxDiff);
    }

    @Test
    public void testSameDiffRank3InputWithPermuteProjectionWeight() {
        assertSameDiffRank3InputWithPermuteProjectionWeight(true);
    }

    @Test
    public void testSameDiffRank3InputWithPermuteProjectionWeightDspDisabled() {
        assertSameDiffRank3InputWithPermuteProjectionWeight(false);
    }

    private static void assertSameDiffRank3InputWithPermuteProjectionWeight(boolean dspEnabled) {
        float eps = 1e-6f;
        int batch = 1, seqLen = 7, K = 64, N = 128;

        INDArray x = Nd4j.randn(DataType.FLOAT, batch, seqLen, K);
        INDArray gamma = Nd4j.rand(DataType.FLOAT, K).addi(0.5);
        INDArray weightRaw = Nd4j.randn(DataType.FLOAT, N, K).muli(0.02);
        INDArray weightTransposed = weightRaw.transpose();

        INDArray x2d = x.reshape('c', batch * seqLen, K);
        INDArray invRms = computeInvRms(x2d, eps);
        INDArray expected = x2d.mul(invRms).mul(gamma)
                .mmul(weightTransposed)
                .reshape('c', batch, seqLen, N);

        SameDiff sd = SameDiff.create();
        SDVariable xVar = sd.placeHolder("x", DataType.FLOAT, batch, seqLen, K);
        SDVariable gammaVar = sd.var("gamma", gamma);
        SDVariable weightRawVar = sd.var("lm_head", weightRaw);
        SDVariable weightTransposedVar = sd.permute("lm_head_t", weightRawVar, 1, 0);
        sd.nn().rmsNormLinear("lm_logits", xVar, gammaVar, weightTransposedVar, eps);
        sd.setOutputs("lm_logits");
        sd.setDspAutoCompileEnabled(dspEnabled);
        sd.setDspNativeAutoCompileEnabled(dspEnabled);
        sd.resetSession();

        Map<String, INDArray> ph = Collections.singletonMap("x", x);
        INDArray actual = sd.outputSingle(ph, "lm_logits");

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "SameDiff rank-3 permuted projection weight max diff: " + maxDiff);
    }
}
