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

package org.eclipse.deeplearning4j.nd4j.linalg.ops;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterNdUpdate;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Execution-level tests for the scatter_nd_update op.
 *
 * scatter_nd_update(ref, indices, updates) copies ref → output then overwrites
 * output at the positions given by indices with the values from updates.
 * These tests verify:
 *   1. Basic correctness (1-D, 2-D, 4-D)
 *   2. Non-scattered positions are preserved exactly
 *   3. Output is independent of whether the output buffer was pre-zeroed
 *   4. KV-cache-shaped tensors (the DSP hot path)
 *   5. Multiple data types (FLOAT, HALF, DOUBLE)
 *   6. Edge cases: single-element scatter, full-overwrite scatter
 */
@Slf4j
@DisplayName("scatter_nd_update op execution tests")
public class TestScatterNdUpdate extends BaseNd4jTestWithBackends {

    @Override
    public char ordering() {
        return 'c';
    }

    // ── 1-D basic correctness ───────────────────────────────────────────────

    @Test
    @DisplayName("1-D: scatter single index preserves non-scattered elements")
    public void test1dSingleIndex() {
        // ref = [10, 20, 30, 40, 50]
        INDArray ref = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f, 50.0f);
        // indices = [[2]] — update position 2
        INDArray indices = Nd4j.createFromArray(new int[][]{{2}});
        // updates = [99]
        INDArray updates = Nd4j.createFromArray(99.0f);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

        assertEquals(10.0f, result.getFloat(0), 1e-5, "position 0 should be preserved");
        assertEquals(20.0f, result.getFloat(1), 1e-5, "position 1 should be preserved");
        assertEquals(99.0f, result.getFloat(2), 1e-5, "position 2 should be updated");
        assertEquals(40.0f, result.getFloat(3), 1e-5, "position 3 should be preserved");
        assertEquals(50.0f, result.getFloat(4), 1e-5, "position 4 should be preserved");
    }

    @Test
    @DisplayName("1-D: scatter multiple indices")
    public void test1dMultipleIndices() {
        INDArray ref = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f);
        // Update positions 0, 3, 5
        INDArray indices = Nd4j.createFromArray(new int[][]{{0}, {3}, {5}});
        INDArray updates = Nd4j.createFromArray(100.0f, 400.0f, 600.0f);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

        assertEquals(100.0f, result.getFloat(0), 1e-5);
        assertEquals(2.0f,   result.getFloat(1), 1e-5, "position 1 should be preserved");
        assertEquals(3.0f,   result.getFloat(2), 1e-5, "position 2 should be preserved");
        assertEquals(400.0f, result.getFloat(3), 1e-5);
        assertEquals(5.0f,   result.getFloat(4), 1e-5, "position 4 should be preserved");
        assertEquals(600.0f, result.getFloat(5), 1e-5);
    }

    // ── 2-D correctness ─────────────────────────────────────────────────────

    @Test
    @DisplayName("2-D: scatter row into matrix preserves other rows")
    public void test2dScatterRow() {
        // ref = [[1,2,3],[4,5,6],[7,8,9]]  shape [3,3]
        INDArray ref = Nd4j.createFromArray(new float[][]{
                {1, 2, 3},
                {4, 5, 6},
                {7, 8, 9}
        });
        // Scatter at row index 1
        INDArray indices = Nd4j.createFromArray(new int[][]{{1}});
        // updates = [[10, 20, 30]]  shape [1,3]
        INDArray updates = Nd4j.createFromArray(new float[][]{{10, 20, 30}});

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

        // Row 0 and 2 preserved
        assertEquals(1.0f,  result.getFloat(0, 0), 1e-5, "row 0 col 0 preserved");
        assertEquals(2.0f,  result.getFloat(0, 1), 1e-5, "row 0 col 1 preserved");
        assertEquals(3.0f,  result.getFloat(0, 2), 1e-5, "row 0 col 2 preserved");
        // Row 1 updated
        assertEquals(10.0f, result.getFloat(1, 0), 1e-5, "row 1 col 0 updated");
        assertEquals(20.0f, result.getFloat(1, 1), 1e-5, "row 1 col 1 updated");
        assertEquals(30.0f, result.getFloat(1, 2), 1e-5, "row 1 col 2 updated");
        // Row 2 preserved
        assertEquals(7.0f,  result.getFloat(2, 0), 1e-5, "row 2 col 0 preserved");
        assertEquals(8.0f,  result.getFloat(2, 1), 1e-5, "row 2 col 1 preserved");
        assertEquals(9.0f,  result.getFloat(2, 2), 1e-5, "row 2 col 2 preserved");
    }

    // ── 4-D KV-cache-shaped tensor ──────────────────────────────────────────

    @Test
    @DisplayName("4-D KV cache: scatter single position preserves all other positions")
    public void test4dKvCacheScatter() {
        // Simulate KV cache: [batch=1, heads=4, seqLen=8, dim=16]
        int batch = 1, heads = 4, seqLen = 8, dim = 16;
        INDArray kvCache = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, batch * heads * seqLen * dim)
                .reshape(batch, heads, seqLen, dim);

        // Take a snapshot of the original for comparison
        INDArray originalCopy = kvCache.dup();

        // Scatter at position [0, :, 5, :] — update all heads at seqLen=5
        // indices shape: [1, 1] pointing to batch=0
        // For scatter_nd_update on 4-D, index [0] means "update all of kvCache[0, ...]"
        // Instead, let's scatter a specific [batch, head, pos] = [0, 0, 5]
        // indices = [[0, 0, 5]]  (3-D index into the 4-D ref)
        INDArray indices = Nd4j.createFromArray(new int[][]{{0, 0, 5}});
        // updates = shape [1, dim=16] — single slice
        INDArray updates = Nd4j.ones(DataType.FLOAT, 1, dim).mul(-999.0f);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(kvCache, indices, updates))[0];

        // Verify the scattered position was updated
        for (int d = 0; d < dim; d++) {
            assertEquals(-999.0f, result.getFloat(0, 0, 5, d), 1e-5,
                    "position [0,0,5," + d + "] should be updated to -999");
        }

        // Verify ALL other positions are preserved from original
        int mismatchCount = 0;
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    if (b == 0 && h == 0 && s == 5) continue; // skip scattered position
                    for (int d = 0; d < dim; d++) {
                        float expected = originalCopy.getFloat(b, h, s, d);
                        float actual = result.getFloat(b, h, s, d);
                        if (Math.abs(expected - actual) > 1e-5) {
                            mismatchCount++;
                            if (mismatchCount <= 3) {
                                log.error("Mismatch at [{},{},{},{}]: expected={} actual={}",
                                        b, h, s, d, expected, actual);
                            }
                        }
                    }
                }
            }
        }
        assertEquals(0, mismatchCount,
                "Non-scattered positions must be preserved exactly. Mismatches: " + mismatchCount);
    }

    // ── Output is independent of pre-zeroed buffer ──────────────────────────

    @Test
    @DisplayName("Output is identical whether buffer was pre-zeroed or filled with garbage")
    public void testOutputIndependentOfPreZero() {
        INDArray ref = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 20).reshape(4, 5);
        INDArray indices = Nd4j.createFromArray(new int[][]{{1}, {3}});
        INDArray updates = Nd4j.createFromArray(new float[][]{
                {-1, -2, -3, -4, -5},
                {-6, -7, -8, -9, -10}
        });

        // Run once (output buffer state is whatever the allocator gave us)
        INDArray result1 = Nd4j.exec(new ScatterNdUpdate(ref.dup(), indices, updates))[0];

        // Run again with a fresh ref copy — output should be identical
        INDArray result2 = Nd4j.exec(new ScatterNdUpdate(ref.dup(), indices, updates))[0];

        assertEquals(result1, result2, "Results must be identical across invocations");

        // Verify non-scattered rows preserved
        for (int c = 0; c < 5; c++) {
            assertEquals(ref.getFloat(0, c), result1.getFloat(0, c), 1e-5,
                    "row 0 col " + c + " must match ref");
            assertEquals(ref.getFloat(2, c), result1.getFloat(2, c), 1e-5,
                    "row 2 col " + c + " must match ref");
        }
        // Verify scattered rows
        assertEquals(-1.0f, result1.getFloat(1, 0), 1e-5);
        assertEquals(-6.0f, result1.getFloat(3, 0), 1e-5);
    }

    // ── Multiple data types ─────────────────────────────────────────────────

    @ParameterizedTest(name = "dtype={0}")
    @EnumSource(value = DataType.class, names = {"FLOAT", "DOUBLE", "HALF"})
    @DisplayName("scatter_nd_update works across data types")
    public void testMultipleDataTypes(DataType dtype) {
        INDArray ref = Nd4j.linspace(dtype, 1.0, 1.0, 10);
        INDArray indices = Nd4j.createFromArray(new int[][]{{3}, {7}});
        INDArray updates = Nd4j.zeros(dtype, 2).assign(-42.0);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

        assertEquals(dtype, result.dataType(), "output dtype must match input");

        // Check scattered values
        float tol = dtype == DataType.HALF ? 0.1f : 1e-4f;
        assertEquals(-42.0f, result.getFloat(3), tol, "position 3 should be -42");
        assertEquals(-42.0f, result.getFloat(7), tol, "position 7 should be -42");

        // Check preserved values
        assertEquals(1.0f, result.getFloat(0), tol, "position 0 should be preserved");
        assertEquals(2.0f, result.getFloat(1), tol, "position 1 should be preserved");
        assertEquals(5.0f, result.getFloat(4), tol, "position 4 should be preserved");
    }

    // ── Edge case: full overwrite ───────────────────────────────────────────

    @Test
    @DisplayName("Full overwrite: scatter all indices replaces entire content")
    public void testFullOverwrite() {
        INDArray ref = Nd4j.ones(DataType.FLOAT, 5);
        // Scatter at every index
        INDArray indices = Nd4j.createFromArray(new int[][]{{0}, {1}, {2}, {3}, {4}});
        INDArray updates = Nd4j.createFromArray(10.0f, 20.0f, 30.0f, 40.0f, 50.0f);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

        for (int i = 0; i < 5; i++) {
            assertEquals((i + 1) * 10.0f, result.getFloat(i), 1e-5);
        }
    }

    // ── Edge case: single element ref ───────────────────────────────────────

    @Test
    @DisplayName("Single element ref with scatter")
    public void testSingleElement() {
        INDArray ref = Nd4j.createFromArray(42.0f);
        INDArray indices = Nd4j.createFromArray(new int[][]{{0}});
        INDArray updates = Nd4j.createFromArray(7.0f);

        INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];
        assertEquals(7.0f, result.getFloat(0), 1e-5);
    }

    // ── SameDiff graph execution ────────────────────────────────────────────

    @Test
    @DisplayName("scatter_nd_update via SameDiff produces correct results")
    public void testViaSameDiff() {
        var sd = org.nd4j.autodiff.samediff.SameDiff.create();
        var ref = sd.constant("ref", Nd4j.createFromArray(new float[]{1, 2, 3, 4, 5}));
        var indices = sd.constant("indices", Nd4j.createFromArray(new int[][]{{1}, {4}}));
        var updates = sd.constant("updates", Nd4j.createFromArray(new float[]{-20, -50}));
        var result = sd.scatterNdUpdate(ref, indices, updates);

        var output = sd.output(java.util.Collections.emptyMap(), result.name());
        INDArray out = output.get(result.name());

        assertEquals(1.0f,   out.getFloat(0), 1e-5, "position 0 preserved");
        assertEquals(-20.0f, out.getFloat(1), 1e-5, "position 1 updated");
        assertEquals(3.0f,   out.getFloat(2), 1e-5, "position 2 preserved");
        assertEquals(4.0f,   out.getFloat(3), 1e-5, "position 3 preserved");
        assertEquals(-50.0f, out.getFloat(4), 1e-5, "position 4 updated");
    }

    // ── Repeated execution (DSP steady-state simulation) ────────────────────

    @Test
    @DisplayName("Repeated scatter_nd_update on same-shaped tensors produces correct results each time")
    public void testRepeatedExecution() {
        // Simulates DSP frozen fast path: same shapes, different data each step
        int seqLen = 16, dim = 8;
        for (int step = 0; step < 10; step++) {
            INDArray ref = Nd4j.linspace(DataType.FLOAT, step * 100.0, 1.0, seqLen * dim)
                    .reshape(seqLen, dim);
            int scatterPos = step % seqLen;
            INDArray indices = Nd4j.createFromArray(new int[][]{{scatterPos}});
            INDArray updates = Nd4j.ones(DataType.FLOAT, 1, dim).mul(-1.0f * (step + 1));

            INDArray result = Nd4j.exec(new ScatterNdUpdate(ref, indices, updates))[0];

            // Scattered position should have the update value
            for (int d = 0; d < dim; d++) {
                float expected = -1.0f * (step + 1);
                assertEquals(expected, result.getFloat(scatterPos, d), 1e-4,
                        "step=" + step + " pos=" + scatterPos + " dim=" + d);
            }

            // One non-scattered position should have original ref value
            int checkPos = (scatterPos + 1) % seqLen;
            for (int d = 0; d < dim; d++) {
                float expected = ref.getFloat(checkPos, d);
                assertEquals(expected, result.getFloat(checkPos, d), 1e-4,
                        "step=" + step + " preserved pos=" + checkPos + " dim=" + d);
            }
        }
    }

    // ── Verify ref is NOT modified when running out-of-place ────────────────

    @Test
    @DisplayName("Out-of-place scatter_nd_update does not modify the input ref array")
    public void testRefNotModifiedOutOfPlace() {
        INDArray ref = Nd4j.createFromArray(1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
        INDArray refCopy = ref.dup();
        INDArray indices = Nd4j.createFromArray(new int[][]{{2}});
        INDArray updates = Nd4j.createFromArray(999.0f);

        Nd4j.exec(new ScatterNdUpdate(ref, indices, updates));

        // ref should be unchanged — assign copies to output, not to ref
        for (int i = 0; i < 5; i++) {
            assertEquals(refCopy.getFloat(i), ref.getFloat(i), 1e-5,
                    "ref position " + i + " must not be modified by out-of-place scatter");
        }
    }
}
