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

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.DisplayName;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) execution correctness for reshape_no_copy operations.
 *
 * Reproduces the stale shape cache bug where:
 * - reshape_no_copy was missing from VALUE_DEPENDENT_SHAPE_OPS in NativePlanCompiler.cpp
 * - frozenConstantSlot was not guarded by shapesFrozen_ check
 * - This caused "Cannot reshape array of size X into shape with known dimensions product Y"
 *   on Frame 2+ when input sizes change between DSP plan executions.
 */
public class TestDSPReshapeNoCopy extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPReshapeNoCopy.class);
    private static final double TOL = 1e-5;

    @Override
    public char ordering() {
        return 'c';
    }

    @AfterEach
    public void cleanup() {
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
    }

    /**
     * Helper: run graph with standard execution, then with native DSP, and compare.
     */
    private void compareStandardVsDsp(SameDiff sd, Map<String, INDArray> placeholders, String outputName) {
        // 1. Standard execution (reference)
        Map<String, INDArray> refResult = sd.output(placeholders, outputName);
        INDArray expected = refResult.get(outputName).dup();
        log.info("Standard output shape: {}", expected.shape());

        // 2. Reset and compile native DSP
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // 3. DSP execution
        Map<String, INDArray> dspResult = sd.output(placeholders, outputName);
        INDArray actual = dspResult.get(outputName);

        // 4. Compare
        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch: expected " + java.util.Arrays.toString(expected.shape())
                        + " but got " + java.util.Arrays.toString(actual.shape()));
        assertEquals(expected.dataType(), actual.dataType(), "DataType mismatch");

        double maxDiff = expected.sub(actual.castTo(expected.dataType())).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Value mismatch: max diff = " + maxDiff + " (tolerance = " + TOL + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Basic reshape_no_copy with static target shape
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy: 3D → 2D with static shape")
    public void testReshapeNoCopy3Dto2D() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable reshaped = sd.reshape("output", input, -1, 32);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 2, 4, 8);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    @Test
    @DisplayName("reshape_no_copy: 2D → 3D with static shape")
    public void testReshapeNoCopy2Dto3D() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable reshaped = sd.reshape("output", input, -1, 4, 8);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 2, 32);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    @Test
    @DisplayName("reshape_no_copy: flatten to 1D")
    public void testReshapeNoCopyFlatten() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable reshaped = sd.reshape("output", input, -1);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 3, 4, 8);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-execution with SAME shapes (should cache correctly)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy: multiple executions with same shape")
    public void testReshapeNoCopyMultiExecSameShape() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 32);
        SDVariable output = sd.math.add("output", reshaped, 1.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        for (int i = 0; i < 5; i++) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, 2, 4, 8);
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Output null on iteration " + i);
            assertArrayEquals(new long[]{2, 32}, out.shape(),
                    "Shape wrong on iteration " + i);
            log.info("Iteration {}: shape={}", i, java.util.Arrays.toString(out.shape()));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-execution with CHANGING shapes (stale cache bug reproducer)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy: changing batch size across executions (main bug)")
    public void testReshapeNoCopyChangingBatchSize() {
        // This is the primary bug reproducer.
        // Frame 1: input [2, 4, 8] → reshape to [-1, 32] → [2, 32]
        // Frame 2: input [3, 4, 8] → reshape to [-1, 32] → [3, 32]
        // Bug: Frame 2 uses cached output shape [2, 32] from Frame 1
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 32);
        SDVariable output = sd.math.add("output", reshaped, 1.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[] batchSizes = {2, 3, 1, 5, 2};
        for (int batch : batchSizes) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, batch, 4, 8);
            System.out.println("=== Executing with batch=" + batch + ", input shape=" + java.util.Arrays.toString(arr.shape()) + " ===");
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Output null for batch=" + batch);
            System.out.println("batch=" + batch + ": output shape=" + java.util.Arrays.toString(out.shape()) + " length=" + out.length());
            assertArrayEquals(new long[]{batch, 32}, out.shape(),
                    "Shape mismatch for batch=" + batch + ": expected [" + batch + ", 32] but got "
                            + java.util.Arrays.toString(out.shape()));
        }
    }

    @Test
    @DisplayName("reshape_no_copy: changing spatial dimensions across executions")
    public void testReshapeNoCopyChangingSpatialDims() {
        // Simulates vision encoder where patch count varies
        // Frame 1: [1, 1024, 768] → reshape → [1024, 768]
        // Frame 2: [1, 256, 768]  → reshape → [256, 768]
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, -1, 768);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 768);
        SDVariable output = sd.math.add("output", reshaped, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[] seqLens = {1024, 256, 512, 1};
        for (int seqLen : seqLens) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, 1, seqLen, 768);
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Output null for seqLen=" + seqLen);
            assertArrayEquals(new long[]{seqLen, 768}, out.shape(),
                    "Shape mismatch for seqLen=" + seqLen + ": expected [" + seqLen + ", 768] but got "
                            + java.util.Arrays.toString(out.shape()));
            log.info("seqLen={}: shape={}", seqLen, java.util.Arrays.toString(out.shape()));
        }
    }

    @Test
    @DisplayName("reshape_no_copy: batch=1 then larger batch (Frame 2+ fail case)")
    public void testReshapeNoCopySmallToLarge() {
        // Exact scenario: small input then large input
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 768);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 768);
        SDVariable output = sd.math.mul("output", reshaped, 1.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Frame 1: small
        INDArray small = Nd4j.rand(DataType.FLOAT, 1, 1, 768);
        Map<String, INDArray> r1 = sd.output(Map.of("input", small), "output");
        assertArrayEquals(new long[]{1, 768}, r1.get("output").shape());

        // Frame 2: large (this is where the stale cache bug manifests)
        INDArray large = Nd4j.rand(DataType.FLOAT, 1, 1024, 768);
        Map<String, INDArray> r2 = sd.output(Map.of("input", large), "output");
        assertArrayEquals(new long[]{1024, 768}, r2.get("output").shape(),
                "Frame 2 should produce [1024, 768] but got "
                        + java.util.Arrays.toString(r2.get("output").shape()));
    }

    @Test
    @DisplayName("reshape_no_copy: large then small batch (reverse direction)")
    public void testReshapeNoCopyLargeToSmall() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 768);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 768);
        SDVariable output = sd.math.mul("output", reshaped, 1.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Frame 1: large
        INDArray large = Nd4j.rand(DataType.FLOAT, 1, 1024, 768);
        Map<String, INDArray> r1 = sd.output(Map.of("input", large), "output");
        assertArrayEquals(new long[]{1024, 768}, r1.get("output").shape());

        // Frame 2: small
        INDArray small = Nd4j.rand(DataType.FLOAT, 1, 1, 768);
        Map<String, INDArray> r2 = sd.output(Map.of("input", small), "output");
        assertArrayEquals(new long[]{1, 768}, r2.get("output").shape(),
                "Frame 2 should produce [1, 768] but got "
                        + java.util.Arrays.toString(r2.get("output").shape()));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Dynamic shape tensor (input[1] provides target shape)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy: dynamic shape from shape_of op")
    public void testReshapeNoCopyDynamicShapeFromShapeOf() {
        // Reshape target computed dynamically from input shape
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, -1);
        SDVariable flat = sd.reshape("flat", input, -1);  // flatten
        SDVariable output = sd.math.add("output", flat, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Multiple different shapes
        long[][] shapes = {{2, 3, 4}, {1, 5, 6}, {3, 2, 2}, {1, 1, 768}};
        for (long[] shape : shapes) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, shape);
            long expectedLen = shape[0] * shape[1] * shape[2];
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out);
            assertEquals(1, out.rank(), "Should be rank-1");
            assertEquals(expectedLen, out.length(),
                    "Length mismatch for shape " + java.util.Arrays.toString(shape));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Reshape in multi-op graphs (closer to real vision encoder pattern)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy in matmul → reshape → relu pipeline")
    public void testReshapeInPipeline() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable weights = sd.constant("w", Nd4j.rand(DataType.FLOAT, 32, 64));
        SDVariable mm = sd.mmul("mm", input, weights);
        SDVariable reshaped = sd.reshape("reshaped", mm, -1, 8, 8);
        SDVariable output = sd.nn.relu("output", reshaped, 0.0);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 4, 32);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    @Test
    @DisplayName("reshape_no_copy in pipeline with changing batch size")
    public void testReshapeInPipelineChangingBatch() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 32);
        SDVariable weights = sd.constant("w", Nd4j.rand(DataType.FLOAT, 32, 64));
        SDVariable mm = sd.mmul("mm", input, weights);
        SDVariable reshaped = sd.reshape("reshaped", mm, -1, 8, 8);
        SDVariable output = sd.nn.relu("output", reshaped, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[] batches = {4, 2, 8, 1, 4};
        for (int batch : batches) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, batch, 32);
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null output for batch=" + batch);
            assertArrayEquals(new long[]{batch, 8, 8}, out.shape(),
                    "Shape mismatch for batch=" + batch);
            log.info("Pipeline batch={}: shape={}", batch, java.util.Arrays.toString(out.shape()));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Reshape followed by reduce (compound dynamic shape)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy → reduce_sum with changing shapes")
    public void testReshapeThenReduceChangingShapes() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 8);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 8);
        SDVariable output = sd.sum("output", reshaped, 1);  // reduce along dim 1

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] configs = {{2, 4}, {3, 6}, {1, 2}, {4, 8}};
        for (int[] cfg : configs) {
            int batch = cfg[0], seq = cfg[1];
            INDArray arr = Nd4j.rand(DataType.FLOAT, batch, seq, 8);
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null for batch=" + batch + " seq=" + seq);
            assertEquals(batch * seq, out.length(),
                    "Length mismatch for batch=" + batch + " seq=" + seq);
            log.info("batch={} seq={}: output shape={}", batch, seq,
                    java.util.Arrays.toString(out.shape()));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multiple reshape ops in same graph (tests cache independence)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("multiple reshape_no_copy ops with changing shapes")
    public void testMultipleReshapeOps() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4, 8);
        SDVariable flat = sd.reshape("flat", input, -1, 32);
        SDVariable expanded = sd.reshape("expanded", flat, -1, 4, 8);
        SDVariable output = sd.math.add("output", expanded, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[] batches = {2, 5, 1, 3};
        for (int batch : batches) {
            INDArray arr = Nd4j.rand(DataType.FLOAT, batch, 4, 8);
            Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
            INDArray out = result.get("output");
            assertNotNull(out);
            assertArrayEquals(new long[]{batch, 4, 8}, out.shape(),
                    "Shape mismatch for batch=" + batch);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Correctness check: values preserved through reshape
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy preserves values correctly")
    public void testReshapeNoCopyValues() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 6);
        SDVariable reshaped = sd.reshape("output", input, -1, 2, 3);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray arr = Nd4j.createFromArray(new float[][]{
                {1, 2, 3, 4, 5, 6},
                {7, 8, 9, 10, 11, 12}
        });

        Map<String, INDArray> result = sd.output(Map.of("input", arr), "output");
        INDArray out = result.get("output");
        assertArrayEquals(new long[]{2, 2, 3}, out.shape());

        // Check values: [[[1,2,3],[4,5,6]], [[7,8,9],[10,11,12]]]
        assertEquals(1.0f, out.getFloat(0, 0, 0), TOL);
        assertEquals(6.0f, out.getFloat(0, 1, 2), TOL);
        assertEquals(7.0f, out.getFloat(1, 0, 0), TOL);
        assertEquals(12.0f, out.getFloat(1, 1, 2), TOL);
    }

    @Test
    @DisplayName("reshape_no_copy: values correct across multiple dynamic shapes")
    public void testReshapeNoCopyValuesMultiExec() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 4);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 2, 2);
        SDVariable output = sd.math.add("output", reshaped, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Exec 1: batch=1
        INDArray arr1 = Nd4j.createFromArray(new float[][]{{1, 2, 3, 4}});
        Map<String, INDArray> r1 = sd.output(Map.of("input", arr1), "output");
        assertEquals(1.0f, r1.get("output").getFloat(0, 0, 0), TOL);
        assertEquals(4.0f, r1.get("output").getFloat(0, 1, 1), TOL);

        // Exec 2: batch=2 (different shape, must invalidate cache)
        INDArray arr2 = Nd4j.createFromArray(new float[][]{
                {10, 20, 30, 40},
                {50, 60, 70, 80}
        });
        Map<String, INDArray> r2 = sd.output(Map.of("input", arr2), "output");
        assertArrayEquals(new long[]{2, 2, 2}, r2.get("output").shape());
        assertEquals(10.0f, r2.get("output").getFloat(0, 0, 0), TOL);
        assertEquals(80.0f, r2.get("output").getFloat(1, 1, 1), TOL);

        // Exec 3: back to batch=1
        INDArray arr3 = Nd4j.createFromArray(new float[][]{{100, 200, 300, 400}});
        Map<String, INDArray> r3 = sd.output(Map.of("input", arr3), "output");
        assertArrayEquals(new long[]{1, 2, 2}, r3.get("output").shape());
        assertEquals(100.0f, r3.get("output").getFloat(0, 0, 0), TOL);
        assertEquals(400.0f, r3.get("output").getFloat(0, 1, 1), TOL);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Standard vs DSP comparison with dynamic shapes
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reshape_no_copy: standard vs DSP comparison")
    public void testReshapeNoCopyStandardVsDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, 12);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 3, 4);
        SDVariable output = sd.math.add("output", reshaped, 1.0);

        INDArray arr = Nd4j.rand(DataType.FLOAT, 5, 12);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    @Test
    @DisplayName("reshape_no_copy: standard vs DSP with double precision")
    public void testReshapeNoCopyDoublePrecision() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.DOUBLE, -1, 6);
        SDVariable reshaped = sd.reshape("output", input, -1, 2, 3);

        INDArray arr = Nd4j.rand(DataType.DOUBLE, 3, 6);
        compareStandardVsDsp(sd, Map.of("input", arr), "output");
    }

    @Test
    @DisplayName("reshape_no_copy: standard vs DSP with half precision")
    public void testReshapeNoCopyHalfPrecision() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.HALF, -1, 8);
        SDVariable reshaped = sd.reshape("output", input, -1, 2, 4);

        Map<String, INDArray> placeholders = Map.of("input", Nd4j.rand(DataType.HALF, 4, 8));

        Map<String, INDArray> ref = sd.output(placeholders, "output");
        INDArray expected = ref.get("output").dup();

        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        Map<String, INDArray> dspResult = sd.output(placeholders, "output");
        INDArray actual = dspResult.get("output");

        assertArrayEquals(expected.shape(), actual.shape());
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < 0.01, "Half precision max diff: " + maxDiff);
    }
}
