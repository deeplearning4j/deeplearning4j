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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.MoeWeightedSum;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for moe_weighted_sum.
 *
 * Reference computation:
 *   out[t, d] = Σ_k  weights[t, k] * expertOutputs[t, k, d]
 *
 * Run command (CPU):
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dtest=TestMoeWeightedSum#* 2>&1 | tee /tmp/test-moe-weighted-sum.log
 *
 * Run command (CUDA):
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests && \
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dbackend.artifactId=nd4j-cuda-12.9 \
 *     -Dtest=TestMoeWeightedSum#* 2>&1 | tee /tmp/test-moe-weighted-sum-cuda.log
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("MoeWeightedSum Op Tests")
public class TestMoeWeightedSum extends BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helper: compute reference output via plain loops (fp32 accumulation)
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Reference dense: expertOutputs [T, topK, D], weights [T, topK] → out [T, D]
     * Uses fp32 accumulation, stores in fp32.
     */
    private static float[] referenceDense(float[] eo, float[] w, int T, int topK, int D) {
        float[] out = new float[T * D];
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < topK; k++) {
                float wk = w[t * topK + k];
                for (int d = 0; d < D; d++) {
                    out[t * D + d] += wk * eo[t * topK * D + k * D + d];
                }
            }
        }
        return out;
    }

    /**
     * Reference flat: eo [totalRows, D], w [T, topK], rowIndex [totalRows, 2] → out [T, D]
     */
    private static float[] referenceFlat(float[] eo, float[] w, int[] rowIndex,
                                          int totalRows, int T, int topK, int D) {
        float[] out = new float[T * D];
        for (int r = 0; r < totalRows; r++) {
            int tokenIdx = rowIndex[r * 2];
            int kIdx     = rowIndex[r * 2 + 1];
            float wk = w[tokenIdx * topK + kIdx];
            for (int d = 0; d < D; d++) {
                out[tokenIdx * D + d] += wk * eo[r * D + d];
            }
        }
        return out;
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Dense mode (mode=0) tests
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Parameterized over (T, topK, D) combinations.
     *
     * Run: -Dtest=TestMoeWeightedSum#testDenseCorrectness*
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - correctness sweep T x topK x D")
    public void testDenseCorrectness(Nd4jBackend backend) {
        int[]  Tvals    = {1, 7, 128};
        int[]  topKvals = {1, 2, 4, 6, 8};
        int[]  Dvals    = {16, 512};

        for (int T : Tvals) {
            for (int topK : topKvals) {
                for (int D : Dvals) {
                    Nd4j.getRandom().setSeed(12345L + T + topK * 100 + D);

                    INDArray eo = Nd4j.rand(DataType.FLOAT, T, topK, D);
                    INDArray w  = Nd4j.rand(DataType.FLOAT, T, topK);

                    // Normalise weights so each token sums to 1
                    INDArray wSum = w.sum(true, 1);
                    w = w.div(wSum);

                    // Run op
                    MoeWeightedSum op = new MoeWeightedSum(eo, w);
                    INDArray[] outs = Nd4j.exec(op);
                    INDArray result = outs[0];

                    // Verify shape
                    assertArrayEquals(new long[]{T, D}, result.shape(),
                        "Shape mismatch for T=" + T + " topK=" + topK + " D=" + D);

                    // Reference
                    float[] eoF = eo.toFloatVector();
                    float[] wF  = w.toFloatVector();
                    float[] ref = referenceDense(eoF, wF, T, topK, D);
                    float[] got = result.toFloatVector();

                    for (int i = 0; i < T * D; i++) {
                        assertEquals(ref[i], got[i], 1e-4f,
                            String.format("Mismatch at T=%d topK=%d D=%d idx=%d ref=%f got=%f",
                                T, topK, D, i, ref[i], got[i]));
                    }
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Dense mode — fp16 output
    // ──────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - fp16 input/output")
    public void testDenseFp16(Nd4jBackend backend) {
        final int T = 32, topK = 4, D = 64;
        Nd4j.getRandom().setSeed(9999L);

        INDArray eoFp32 = Nd4j.rand(DataType.FLOAT, T, topK, D);
        INDArray wFp32  = Nd4j.rand(DataType.FLOAT, T, topK);
        // normalise
        wFp32 = wFp32.div(wFp32.sum(true, 1));

        // fp32 reference
        MoeWeightedSum opFp32 = new MoeWeightedSum(eoFp32, wFp32);
        INDArray refOut = Nd4j.exec(opFp32)[0];

        // fp16
        INDArray eoFp16 = eoFp32.castTo(DataType.HALF);
        INDArray wFp16  = wFp32.castTo(DataType.HALF);
        MoeWeightedSum opFp16 = new MoeWeightedSum(eoFp16, wFp16);
        INDArray fp16Out = Nd4j.exec(opFp16)[0];

        assertEquals(DataType.HALF, fp16Out.dataType(), "Output dataType must be HALF for HALF input");
        assertArrayEquals(new long[]{T, D}, fp16Out.shape());

        INDArray fp16OutFp32 = fp16Out.castTo(DataType.FLOAT);
        float[] ref = refOut.toFloatVector();
        float[] got = fp16OutFp32.toFloatVector();

        for (int i = 0; i < T * D; i++) {
            // fp16 accumulation error: allow up to 2e-2 relative
            float absErr = Math.abs(ref[i] - got[i]);
            float relTol = Math.max(2e-2f * Math.abs(ref[i]), 5e-3f);
            assertTrue(absErr <= relTol,
                String.format("fp16 mismatch at idx=%d ref=%f got=%f absErr=%f relTol=%f",
                    i, ref[i], got[i], absErr, relTol));
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Flat mode (mode=1) tests
    // ──────────────────────────────────────────────────────────────────────────

    /**
     * Verify flat mode matches dense mode when all tokens have the same topK.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Flat - correctness vs dense reference")
    public void testFlatCorrectness(Nd4jBackend backend) {
        int[] Tvals    = {1, 7, 128};
        int[] topKvals = {1, 2, 4, 8};
        int[] Dvals    = {16, 512};

        for (int T : Tvals) {
            for (int topK : topKvals) {
                for (int D : Dvals) {
                    Nd4j.getRandom().setSeed(55555L + T + topK * 100 + D);

                    // Build dense expertOutputs and weights
                    INDArray eoDense = Nd4j.rand(DataType.FLOAT, T, topK, D);
                    INDArray w       = Nd4j.rand(DataType.FLOAT, T, topK);
                    w = w.div(w.sum(true, 1));

                    // Build flat [totalRows, D] = reshape(eoDense, [T*topK, D])
                    int totalRows = T * topK;
                    INDArray eoFlat = eoDense.reshape(totalRows, D);

                    // Build rowIndex [totalRows, 2]: row r maps to (r/topK, r%topK)
                    int[] riData = new int[totalRows * 2];
                    for (int r = 0; r < totalRows; r++) {
                        riData[r * 2]     = r / topK;   // token_idx
                        riData[r * 2 + 1] = r % topK;   // k_idx
                    }
                    INDArray rowIndex = Nd4j.createFromArray(riData).reshape(totalRows, 2);

                    // Flat op
                    MoeWeightedSum flatOp = new MoeWeightedSum(eoFlat, w, rowIndex);
                    INDArray flatResult   = Nd4j.exec(flatOp)[0];

                    // Dense op (reference)
                    MoeWeightedSum denseOp = new MoeWeightedSum(eoDense, w);
                    INDArray denseResult   = Nd4j.exec(denseOp)[0];

                    assertArrayEquals(new long[]{T, D}, flatResult.shape(),
                        "Shape mismatch flat T=" + T + " topK=" + topK + " D=" + D);

                    float[] flatF  = flatResult.toFloatVector();
                    float[] denseF = denseResult.toFloatVector();

                    for (int i = 0; i < T * D; i++) {
                        assertEquals(denseF[i], flatF[i], 1e-4f,
                            String.format("Flat vs Dense mismatch T=%d topK=%d D=%d idx=%d dense=%f flat=%f",
                                T, topK, D, i, denseF[i], flatF[i]));
                    }
                }
            }
        }
    }

    /**
     * Flat mode with INT64 rowIndex.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Flat - INT64 rowIndex")
    public void testFlatInt64RowIndex(Nd4jBackend backend) {
        final int T = 8, topK = 2, D = 32;
        Nd4j.getRandom().setSeed(77777L);

        INDArray eoDense = Nd4j.rand(DataType.FLOAT, T, topK, D);
        INDArray w       = Nd4j.rand(DataType.FLOAT, T, topK);
        w = w.div(w.sum(true, 1));

        int totalRows = T * topK;
        INDArray eoFlat = eoDense.reshape(totalRows, D);

        // INT64 rowIndex
        long[] riData = new long[totalRows * 2];
        for (int r = 0; r < totalRows; r++) {
            riData[r * 2]     = r / topK;
            riData[r * 2 + 1] = r % topK;
        }
        INDArray rowIndex64 = Nd4j.createFromArray(riData).reshape(totalRows, 2);
        assertEquals(DataType.INT64, rowIndex64.dataType());

        MoeWeightedSum flatOp  = new MoeWeightedSum(eoFlat, w, rowIndex64);
        INDArray flatResult    = Nd4j.exec(flatOp)[0];

        // Dense reference
        MoeWeightedSum denseOp = new MoeWeightedSum(eoDense, w);
        INDArray denseResult   = Nd4j.exec(denseOp)[0];

        assertArrayEquals(new long[]{T, D}, flatResult.shape());
        float[] flatF  = flatResult.toFloatVector();
        float[] denseF = denseResult.toFloatVector();
        for (int i = 0; i < T * D; i++) {
            assertEquals(denseF[i], flatF[i], 1e-4f,
                "INT64 rowIndex mismatch at idx=" + i);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CPU / CUDA parity: compare outputs for same inputs
    // (relies on running both backends in the parameterised suite)
    // ──────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum - matches pure SameDiff mul+sum reference")
    public void testMatchesPureComposition(Nd4jBackend backend) {
        // Verify against a SameDiff mul + sum construction (the "obvious" formulation)
        final int T = 16, topK = 4, D = 64;
        Nd4j.getRandom().setSeed(42L);

        INDArray eoArr = Nd4j.rand(DataType.FLOAT, T, topK, D);
        INDArray wArr  = Nd4j.rand(DataType.FLOAT, T, topK);
        wArr = wArr.div(wArr.sum(true, 1));

        // Reference: wArr expanded [T, topK, 1] * eoArr [T, topK, D] → sum over dim 1 → [T, D]
        INDArray wExpanded = wArr.reshape(T, topK, 1);          // [T, topK, 1] broadcast
        INDArray refOut    = eoArr.mul(wExpanded).sum(1);        // [T, D]

        // Op
        MoeWeightedSum op = new MoeWeightedSum(eoArr, wArr);
        INDArray opOut = Nd4j.exec(op)[0];

        assertArrayEquals(new long[]{T, D}, opOut.shape());
        float[] ref = refOut.toFloatVector();
        float[] got = opOut.toFloatVector();
        for (int i = 0; i < T * D; i++) {
            assertEquals(ref[i], got[i], 2e-5f,
                "mul+sum mismatch at idx=" + i + " ref=" + ref[i] + " got=" + got[i]);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Validation error cases
    // ──────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - rank validation errors")
    public void testDenseValidationErrors(Nd4jBackend backend) {
        // rank-2 expertOutputs should fail in dense mode
        INDArray badEo = Nd4j.rand(DataType.FLOAT, 4, 8);  // should be [T, topK, D]
        INDArray w     = Nd4j.rand(DataType.FLOAT, 4, 2);
        MoeWeightedSum op = new MoeWeightedSum(badEo, w);
        assertThrows(Exception.class, () -> Nd4j.exec(op),
            "Dense mode with rank-2 expertOutputs must throw");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Flat - rank validation errors")
    public void testFlatValidationErrors(Nd4jBackend backend) {
        // rank-3 expertOutputs should fail in flat mode (requires rank 2)
        INDArray badEo    = Nd4j.rand(DataType.FLOAT, 4, 2, 8);
        INDArray w        = Nd4j.rand(DataType.FLOAT, 4, 2);
        INDArray rowIndex = Nd4j.createFromArray(new int[]{0,0, 0,1, 1,0, 1,1}).reshape(4, 2);
        MoeWeightedSum op = new MoeWeightedSum(badEo, w, rowIndex);
        assertThrows(Exception.class, () -> Nd4j.exec(op),
            "Flat mode with rank-3 expertOutputs must throw");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - T mismatch validation error")
    public void testDenseTMismatch(Nd4jBackend backend) {
        // expertOutputs T=4, weights T=5 — must fail
        INDArray eo = Nd4j.rand(DataType.FLOAT, 4, 2, 8);
        INDArray w  = Nd4j.rand(DataType.FLOAT, 5, 2);  // T mismatch
        MoeWeightedSum op = new MoeWeightedSum(eo, w);
        assertThrows(Exception.class, () -> Nd4j.exec(op),
            "T mismatch must throw");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Edge case: topK = 1 (single expert, no summation needed)
    // ──────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - topK=1")
    public void testTopK1(Nd4jBackend backend) {
        final int T = 32, D = 128;
        Nd4j.getRandom().setSeed(1111L);

        INDArray eo = Nd4j.rand(DataType.FLOAT, T, 1, D);
        INDArray w  = Nd4j.ones(DataType.FLOAT, T, 1);  // all weights = 1.0

        MoeWeightedSum op = new MoeWeightedSum(eo, w);
        INDArray result   = Nd4j.exec(op)[0];

        assertArrayEquals(new long[]{T, D}, result.shape());

        // With topK=1 and weight=1.0, output must equal expertOutputs squeezed on dim 1
        INDArray expected = eo.reshape(T, D);
        float[] expF = expected.toFloatVector();
        float[] gotF = result.toFloatVector();
        for (int i = 0; i < T * D; i++) {
            assertEquals(expF[i], gotF[i], 1e-5f,
                "topK=1 mismatch at idx=" + i);
        }
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Large-batch smoke test (T=128, topK=8, D=512)
    // ──────────────────────────────────────────────────────────────────────────

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("MoeWeightedSum Dense - large batch smoke test T=128 topK=8 D=512")
    public void testLargeBatchSmoke(Nd4jBackend backend) {
        final int T = 128, topK = 8, D = 512;
        Nd4j.getRandom().setSeed(2024L);

        INDArray eo = Nd4j.rand(DataType.FLOAT, T, topK, D);
        INDArray w  = Nd4j.rand(DataType.FLOAT, T, topK);
        w = w.div(w.sum(true, 1));

        MoeWeightedSum op = new MoeWeightedSum(eo, w);
        INDArray result   = Nd4j.exec(op)[0];

        assertArrayEquals(new long[]{T, D}, result.shape());

        // Spot-check a handful of elements
        float[] eoF  = eo.toFloatVector();
        float[] wF   = w.toFloatVector();
        float[] ref  = referenceDense(eoF, wF, T, topK, D);
        float[] got  = result.toFloatVector();

        // Check every 100th element for speed
        for (int i = 0; i < T * D; i += 100) {
            assertEquals(ref[i], got[i], 1e-3f,
                "Large smoke mismatch at idx=" + i);
        }
    }
}
