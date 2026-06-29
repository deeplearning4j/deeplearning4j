/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.eclipse.deeplearning4j.nd4j.linalg.sparse;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.GradCheckUtil;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.SparseEmbedding;
import org.nd4j.linalg.api.ndarray.SparseFormat;
import org.nd4j.linalg.api.ndarray.SparseNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for {@link SparseEmbedding}: embedding lookup, sparse gradient,
 * embedding-bag pooling, and SameDiff gradient checks.
 *
 * <p>Design notes:
 * <ul>
 *   <li>All floating-point arrays use DOUBLE for numerical-differentiation
 *       accuracy in grad-check tests.</li>
 *   <li>Index arrays use INT32 as required by the underlying scatter/segment ops.</li>
 *   <li>Each test runs on both the CPU and CUDA backends via
 *       {@code @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")}.</li>
 *   <li>The constant-handler cache is purged before every test to avoid stale
 *       released-buffer crashes in multi-test runs (same fix as SparseGradCheckTest).</li>
 * </ul>
 */
public class SparseEmbeddingTest extends BaseNd4jTestWithBackends {

    // ------------------------------------------------------------------
    // Shared test fixtures
    // ------------------------------------------------------------------

    /**
     * Embedding table for lookup tests: 5 rows x 3 dims (DOUBLE).
     *
     * <pre>
     * row 0: [1, 2, 3]
     * row 1: [4, 5, 6]
     * row 2: [7, 8, 9]
     * row 3: [10, 11, 12]
     * row 4: [13, 14, 15]
     * </pre>
     */
    private static INDArray makeTable5x3() {
        return Nd4j.create(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9,
                10, 11, 12,
                13, 14, 15
        }, new long[]{5, 3}, DataType.DOUBLE);
    }

    /**
     * Smaller 4x2 table for embedding-bag tests (DOUBLE).
     *
     * <pre>
     * row 0: [1, 2]
     * row 1: [3, 4]
     * row 2: [5, 6]
     * row 3: [7, 8]
     * </pre>
     */
    private static INDArray makeTable4x2() {
        return Nd4j.create(new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8
        }, new long[]{4, 2}, DataType.DOUBLE);
    }

    /**
     * Purge the constant-handler cache so each test starts with a clean state.
     * Without this, {@code DeallocatorService.forceFlushAll()} frees the backing
     * native buffers of cached constants, and subsequent tests see stale Java
     * wrappers pointing at freed memory.
     */
    @BeforeEach
    public void purgeConstantHandlerCache() {
        Nd4j.getConstantHandler().purgeConstants();
    }

    // ------------------------------------------------------------------
    // 1. Forward lookup
    // ------------------------------------------------------------------

    /**
     * {@code lookup(table, indices)} must return exactly the rows of the
     * embedding table selected by {@code indices}, in order.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLookupForward(Nd4jBackend backend) {
        INDArray table   = makeTable5x3();
        INDArray indices = Nd4j.createFromArray(2, 0, 4);   // INT32

        INDArray result = SparseEmbedding.lookup(table, indices);

        // Expected: row 2, row 0, row 4
        INDArray expected = Nd4j.create(new double[]{
                7, 8, 9,
                1, 2, 3,
                13, 14, 15
        }, new long[]{3, 3}, DataType.DOUBLE);

        assertEquals(expected.shape()[0], result.shape()[0], "output rows");
        assertEquals(expected.shape()[1], result.shape()[1], "output cols");
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(expected.getDouble(r, c), result.getDouble(r, c), 1e-6,
                        "Mismatch at [" + r + ", " + c + "]");
            }
        }
    }

    // ------------------------------------------------------------------
    // 2. Sparse gradient — toSparseGradient
    // ------------------------------------------------------------------

    /**
     * The COO sparse gradient must:
     * <ul>
     *   <li>Have the correct logical shape [vocabSize, dim].</li>
     *   <li>Contain exactly N*dim non-zero entries (one per element of the
     *       upstream gradient matrix).</li>
     *   <li>Materialise to the correct dense gradient when converted via
     *       {@link SparseNDArray#toDense()}, with unused rows zeroed and
     *       repeated indices accumulated (summed).</li>
     * </ul>
     *
     * Setup: indices = [0, 2, 0, 3]; row 0 is looked up twice.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLookupSparseGrad(Nd4jBackend backend) {
        // upstream gradient for 4 lookups, dim=3
        INDArray upstreamGrad = Nd4j.create(new double[]{
                1,  2,  3,
                4,  5,  6,
                7,  8,  9,
                10, 11, 12
        }, new long[]{4, 3}, DataType.DOUBLE);

        // indices: row 0 appears at positions 0 and 2 → should be summed
        INDArray indices = Nd4j.createFromArray(0, 2, 0, 3);

        long vocabSize = 5;
        SparseNDArray sparseGrad = SparseEmbedding.toSparseGradient(upstreamGrad, indices, vocabSize);

        // Verify logical shape and format
        assertArrayEquals(new long[]{vocabSize, 3}, sparseGrad.getShape(),
                "Sparse gradient logical shape");
        assertEquals(SparseFormat.COO, sparseGrad.getFormat(), "Should be COO format");

        // nnz = N * dim = 4 * 3 = 12
        assertEquals(12L, sparseGrad.getValues().length(),
                "COO must have N*dim non-zero entries");

        // Materialise to dense and verify per-row correctness
        INDArray dense = sparseGrad.toDense();
        assertArrayEquals(new long[]{vocabSize, 3}, dense.shape(), "Dense shape");

        // row 0: grad[0] + grad[2] = [1+7, 2+8, 3+9] = [8, 10, 12]
        assertRow(dense, 0, new double[]{8, 10, 12});
        // row 1: not used → zero
        assertRow(dense, 1, new double[]{0, 0, 0});
        // row 2: grad[1] = [4, 5, 6]
        assertRow(dense, 2, new double[]{4, 5, 6});
        // row 3: grad[3] = [10, 11, 12]
        assertRow(dense, 3, new double[]{10, 11, 12});
        // row 4: not used → zero
        assertRow(dense, 4, new double[]{0, 0, 0});
    }

    // ------------------------------------------------------------------
    // 3. Dense gradient — lookupDenseGrad
    // ------------------------------------------------------------------

    /**
     * {@code lookupDenseGrad} must produce the same dense result as materialising
     * the COO sparse gradient, with the same accumulation semantics for repeated
     * indices.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testLookupDenseGrad(Nd4jBackend backend) {
        INDArray upstreamGrad = Nd4j.create(new double[]{
                1,  2,  3,
                4,  5,  6,
                7,  8,  9,
                10, 11, 12
        }, new long[]{4, 3});
        INDArray indices = Nd4j.createFromArray(0, 2, 0, 3);
        long vocabSize = 5;

        INDArray denseGrad = SparseEmbedding.lookupDenseGrad(upstreamGrad, indices, vocabSize);

        assertArrayEquals(new long[]{vocabSize, 3}, denseGrad.shape(), "Dense grad shape");

        // row 0: [1+7, 2+8, 3+9] = [8, 10, 12]  (index 0 looked up twice)
        assertRow(denseGrad, 0, new double[]{8, 10, 12});
        // row 1: unused → zero
        assertRow(denseGrad, 1, new double[]{0, 0, 0});
        // row 2: [4, 5, 6]
        assertRow(denseGrad, 2, new double[]{4, 5, 6});
        // row 3: [10, 11, 12]
        assertRow(denseGrad, 3, new double[]{10, 11, 12});
        // row 4: unused → zero
        assertRow(denseGrad, 4, new double[]{0, 0, 0});
    }

    // ------------------------------------------------------------------
    // 4. EmbeddingBag — SUM
    // ------------------------------------------------------------------

    /**
     * EmbeddingBag SUM: sum the gathered rows within each bag.
     *
     * <pre>
     * indices    = [0, 1, 2, 3, 1]
     * segmentIds = [0, 0, 1, 2, 1]  (bags 0,1,2)
     *
     * gathered = [[1,2],[3,4],[5,6],[7,8],[3,4]]
     * bag 0: [1,2] + [3,4] = [4, 6]
     * bag 1: [5,6] + [3,4] = [8, 10]
     * bag 2: [7,8]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmbeddingBagSum(Nd4jBackend backend) {
        INDArray table      = makeTable4x2();
        INDArray indices    = Nd4j.createFromArray(0, 1, 2, 3, 1);
        INDArray segmentIds = Nd4j.createFromArray(0, 0, 1, 2, 1);
        int numBags = 3;

        INDArray result = SparseEmbedding.embeddingBag(
                table, indices, segmentIds, numBags, SparseEmbedding.MODE_SUM);

        assertArrayEquals(new long[]{3, 2}, result.shape(), "Shape");
        assertRow(result, 0, new double[]{4,  6});
        assertRow(result, 1, new double[]{8, 10});
        assertRow(result, 2, new double[]{7,  8});
    }

    // ------------------------------------------------------------------
    // 5. EmbeddingBag — MEAN
    // ------------------------------------------------------------------

    /**
     * EmbeddingBag MEAN: average the gathered rows within each bag.
     *
     * <pre>
     * bag 0: mean([1,2],[3,4]) = [2, 3]
     * bag 1: mean([5,6],[3,4]) = [4, 5]
     * bag 2: [7, 8]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmbeddingBagMean(Nd4jBackend backend) {
        INDArray table      = makeTable4x2();
        INDArray indices    = Nd4j.createFromArray(0, 1, 2, 3, 1);
        INDArray segmentIds = Nd4j.createFromArray(0, 0, 1, 2, 1);
        int numBags = 3;

        INDArray result = SparseEmbedding.embeddingBag(
                table, indices, segmentIds, numBags, SparseEmbedding.MODE_MEAN);

        assertArrayEquals(new long[]{3, 2}, result.shape(), "Shape");
        assertRow(result, 0, new double[]{2.0, 3.0});
        assertRow(result, 1, new double[]{4.0, 5.0});
        assertRow(result, 2, new double[]{7.0, 8.0});
    }

    // ------------------------------------------------------------------
    // 6. EmbeddingBag — MAX
    // ------------------------------------------------------------------

    /**
     * EmbeddingBag MAX: element-wise max within each bag.
     *
     * <pre>
     * bag 0: max([1,3],[2,4]) = [3, 4]
     * bag 1: max([5,3],[6,4]) = [5, 6]
     * bag 2: [7, 8]
     * </pre>
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testEmbeddingBagMax(Nd4jBackend backend) {
        INDArray table      = makeTable4x2();
        INDArray indices    = Nd4j.createFromArray(0, 1, 2, 3, 1);
        INDArray segmentIds = Nd4j.createFromArray(0, 0, 1, 2, 1);
        int numBags = 3;

        INDArray result = SparseEmbedding.embeddingBag(
                table, indices, segmentIds, numBags, SparseEmbedding.MODE_MAX);

        assertArrayEquals(new long[]{3, 2}, result.shape(), "Shape");
        assertRow(result, 0, new double[]{3, 4});
        assertRow(result, 1, new double[]{5, 6});
        assertRow(result, 2, new double[]{7, 8});
    }

    // ------------------------------------------------------------------
    // 7. SameDiff gradient check — lookup
    // ------------------------------------------------------------------

    /**
     * Numerical gradient check for {@link SparseEmbedding#sdLookup}.
     *
     * <p>The embedding table is a {@code sd.var} (DOUBLE) so GradCheckUtil
     * numerically differentiates w.r.t. it. Indices are {@code sd.constant}
     * (INT32) and are skipped by the gradient checker.
     *
     * <p>Verifies that Gather's analytical gradient (a ScatterAdd) matches the
     * finite-difference gradient, confirming the sparse-gradient path.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradCheckLookup(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42L);

        SameDiff sd = SameDiff.create();
        try {
            // Embedding table — the variable we differentiate w.r.t.
            SDVariable table = sd.var("table",
                    Nd4j.rand(DataType.DOUBLE, 5, 4).addi(0.1));

            // Indices constant — not differentiated
            SDVariable indices = sd.constant("indices",
                    Nd4j.createFromArray(new int[]{2, 0, 4, 1, 2}));

            // Build the lookup sub-graph
            SDVariable out = SparseEmbedding.sdLookup(sd, table, indices);

            // Scalar loss so GradCheckUtil has a single output to back-prop
            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for sdLookup");
        } finally {
            sd.close();
        }
    }

    // ------------------------------------------------------------------
    // 8. SameDiff gradient check — embeddingBag SUM
    // ------------------------------------------------------------------

    /**
     * Numerical gradient check for {@link SparseEmbedding#sdEmbeddingBag} (SUM mode).
     *
     * <p>Both the embedding table and the gradient flow through Gather →
     * UnsortedSegmentSum, both of which have registered doDiff implementations.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradCheckEmbeddingBagSum(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(100L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable table = sd.var("table",
                    Nd4j.rand(DataType.DOUBLE, 4, 3).addi(0.1));

            SDVariable indices = sd.constant("indices",
                    Nd4j.createFromArray(new int[]{0, 1, 2, 3, 1}));
            SDVariable segmentIds = sd.constant("segmentIds",
                    Nd4j.createFromArray(new int[]{0, 0, 1, 2, 1}));

            SDVariable out = SparseEmbedding.sdEmbeddingBag(
                    sd, table, indices, segmentIds, 3, SparseEmbedding.MODE_SUM);

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for sdEmbeddingBag (SUM)");
        } finally {
            sd.close();
        }
    }

    // ------------------------------------------------------------------
    // 9. SameDiff gradient check — embeddingBag MEAN
    // ------------------------------------------------------------------

    /**
     * Numerical gradient check for {@link SparseEmbedding#sdEmbeddingBag} (MEAN mode).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradCheckEmbeddingBagMean(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(200L);

        SameDiff sd = SameDiff.create();
        try {
            SDVariable table = sd.var("table",
                    Nd4j.rand(DataType.DOUBLE, 4, 3).addi(0.1));

            SDVariable indices = sd.constant("indices",
                    Nd4j.createFromArray(new int[]{0, 1, 2, 3, 1}));
            SDVariable segmentIds = sd.constant("segmentIds",
                    Nd4j.createFromArray(new int[]{0, 0, 1, 2, 1}));

            SDVariable out = SparseEmbedding.sdEmbeddingBag(
                    sd, table, indices, segmentIds, 3, SparseEmbedding.MODE_MEAN);

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for sdEmbeddingBag (MEAN)");
        } finally {
            sd.close();
        }
    }

    // ------------------------------------------------------------------
    // 10. SameDiff gradient check — embeddingBag MAX
    // ------------------------------------------------------------------

    /**
     * Numerical gradient check for {@link SparseEmbedding#sdEmbeddingBag} (MAX mode).
     *
     * <p>UnsortedSegmentMax is not differentiable at tie-breaking points; the
     * test fixture is set up so all bags have a clear max to avoid those
     * degenerate cases.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testGradCheckEmbeddingBagMax(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(300L);

        SameDiff sd = SameDiff.create();
        try {
            // Use a table where bag-max is unambiguous: rows differ by large margins
            double[] tableData = {
                    0.1, 0.2, 0.3,   // row 0 — small values
                    5.0, 6.0, 7.0,   // row 1 — clearly largest
                    2.0, 3.0, 4.0,   // row 2 — medium
                    8.0, 9.0, 10.0   // row 3 — largest in its bag
            };
            SDVariable table = sd.var("table",
                    Nd4j.create(tableData, new long[]{4, 3}, DataType.DOUBLE));

            // Each bag has a single clear winner
            SDVariable indices    = sd.constant("indices",    Nd4j.createFromArray(new int[]{0, 1, 2, 3}));
            SDVariable segmentIds = sd.constant("segmentIds", Nd4j.createFromArray(new int[]{0, 0, 1, 1}));

            SDVariable out = SparseEmbedding.sdEmbeddingBag(
                    sd, table, indices, segmentIds, 2, SparseEmbedding.MODE_MAX);

            sd.mean("loss", out);

            assertTrue(
                    GradCheckUtil.checkGradients(sd, null),
                    "Gradient check failed for sdEmbeddingBag (MAX)");
        } finally {
            sd.close();
        }
    }

    // ------------------------------------------------------------------
    // Helper
    // ------------------------------------------------------------------

    /** Assert that row {@code r} of a 2-D INDArray equals {@code expected[]}. */
    private static void assertRow(INDArray arr, int r, double[] expected) {
        assertEquals(expected.length, (int) arr.size(1),
                "Row width mismatch at row " + r);
        for (int c = 0; c < expected.length; c++) {
            assertEquals(expected[c], arr.getDouble(r, c), 1e-6,
                    "arr[" + r + ", " + c + "]");
        }
    }
}
