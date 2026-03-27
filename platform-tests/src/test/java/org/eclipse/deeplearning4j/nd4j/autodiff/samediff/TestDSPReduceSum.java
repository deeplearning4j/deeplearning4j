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
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) execution correctness for reduce_sum operations.
 *
 * Reproduces the shape mismatch bug where DECLARE_SHAPE_FN(reduce_sum) and
 * CUSTOM_OP_IMPL(reduce_sum) disagree on the output shape, specifically when:
 * - Empty dimensions (reduce-all) case
 * - Various keepDims / axis combinations
 * - Negative axis values
 * - Multi-layer graphs with reduce_sum embedded in larger computation
 */
public class TestDSPReduceSum extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPReduceSum.class);
    private static final double TOL = 1e-4;

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
        log.info("Standard output shape: {}, values: {}", expected.shape(), expected);

        // 2. Reset and compile native DSP
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // 3. Execute via DSP
        Map<String, INDArray> dspResult = sd.output(placeholders, outputName);
        INDArray actual = dspResult.get(outputName).dup();
        log.info("DSP output shape: {}, values: {}", actual.shape(), actual);

        // 4. Compare shapes
        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch: expected " + java.util.Arrays.toString(expected.shape()) +
                        " but got " + java.util.Arrays.toString(actual.shape()));

        // 5. Compare values
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL, "Value mismatch: max diff " + maxDiff + " exceeds tolerance " + TOL);
    }

    // ── Single-axis reduce_sum tests ──────────────────────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: keepDims=false, axis=1 on 2D input")
    public void testReduceSumAxis1KeepDimsFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable out = sd.sum("out", x, false, 1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=true, axis=1 on 2D input")
    public void testReduceSumAxis1KeepDimsTrue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable out = sd.sum("out", x, true, 1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=false, axis=0 on 2D input")
    public void testReduceSumAxis0KeepDimsFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable out = sd.sum("out", x, false, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=true, axis=0 on 2D input")
    public void testReduceSumAxis0KeepDimsTrue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable out = sd.sum("out", x, true, 0);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── Negative axis tests ──────────────────────────────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: keepDims=false, axis=-1 on 3D input")
    public void testReduceSumNegativeAxis() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, false, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=true, axis=-1 on 3D input")
    public void testReduceSumNegativeAxisKeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, true, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=false, axis=-2 on 3D input")
    public void testReduceSumNegativeAxis2() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, false, -2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── Multi-axis tests ─────────────────────────────────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: keepDims=false, axes=[0,2] on 3D input")
    public void testReduceSumMultiAxes() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, false, 0, 2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: keepDims=true, axes=[0,2] on 3D input")
    public void testReduceSumMultiAxesKeepDims() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, true, 0, 2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── Reduce-all tests (MAIN BUG REPRODUCER) ──────────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: reduce ALL (no axes), keepDims=false")
    public void testReduceSumReduceAllKeepDimsFalse() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // No axes = reduce all dimensions
        SDVariable out = sd.sum("out", x, false);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: reduce ALL (no axes), keepDims=true")
    public void testReduceSumReduceAllKeepDimsTrue() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // No axes = reduce all dimensions
        SDVariable out = sd.sum("out", x, true);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: reduce ALL on 3D input, keepDims=false")
    public void testReduceSumReduceAll3D() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable out = sd.sum("out", x, false);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 4, 8);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── 4D input tests (vision encoder-like shapes) ──────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: 4D input [1,3,224,224], axis=-1, keepDims=true")
    public void testReduceSum4DLastAxis() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 14, 14);
        SDVariable out = sd.sum("out", x, true, -1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 3, 14, 14);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: 4D input, axes=[2,3] (spatial reduce), keepDims=false")
    public void testReduceSum4DSpatialReduce() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64, 7, 7);
        SDVariable out = sd.sum("out", x, false, 2, 3);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 64, 7, 7);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── Multi-layer graph tests (vision encoder-like) ────────────────────────

    @Test
    @DisplayName("DSP reduce_sum: in multi-layer graph (matmul -> relu -> reduce_sum)")
    public void testReduceSumInMultiLayerGraph() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 16);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 16, 32));
        SDVariable b1 = sd.constant("b1", Nd4j.randn(DataType.FLOAT, 32));

        // Layer 1: matmul + bias + relu
        SDVariable mm = sd.mmul("mm", x, w1);
        SDVariable add = mm.add("add", b1);
        SDVariable relu = sd.nn.relu("relu", add, 0);

        // reduce_sum along axis 1 (feature dimension)
        SDVariable out = sd.sum("out", relu, false, 1);

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 16);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: layer norm pattern (reduce -> subtract -> square -> reduce)")
    public void testReduceSumLayerNormPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8, 64);

        // Mean: sum / N
        SDVariable sum1 = sd.sum("sum1", x, true, -1);
        SDVariable n = sd.constant("n", Nd4j.scalar(DataType.FLOAT, 64.0f));
        SDVariable mean = sum1.div("mean", n);

        // Variance: sum((x - mean)^2) / N
        SDVariable centered = x.sub("centered", mean);
        SDVariable squared = sd.math.square("squared", centered);
        SDVariable varSum = sd.sum("varSum", squared, true, -1);
        SDVariable variance = varSum.div("out", n);

        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 8, 64);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP reduce_sum: attention score pattern (matmul -> reduce_sum -> softmax)")
    public void testReduceSumAttentionPattern() {
        SameDiff sd = SameDiff.create();
        SDVariable q = sd.placeHolder("q", DataType.FLOAT, 1, 4, 8, 16);  // [batch, heads, seqLen, dim]
        SDVariable k = sd.placeHolder("k", DataType.FLOAT, 1, 4, 8, 16);

        // Simplified attention: element-wise multiply + reduce over last dim
        SDVariable qk = q.mul("qk", k);
        SDVariable scores = sd.sum("scores", qk, false, -1);  // [1, 4, 8]

        INDArray qInput = Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16);
        INDArray kInput = Nd4j.randn(DataType.FLOAT, 1, 4, 8, 16);
        try {
            compareStandardVsDsp(sd, Map.of("q", qInput, "k", kInput), "scores");
        } finally {
            sd.close();
        }
    }

    // ── Integer input reduce_sum (ONNX import edge case) ─────────────────────

    @Test
    @DisplayName("DSP reduce_sum: integer input (INT64), axis=0")
    public void testReduceSumIntegerInput() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.INT64, 4, 3);
        SDVariable out = sd.sum("out", x, false, 0);

        INDArray input = Nd4j.createFromArray(new long[][]{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}});
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }

    // ── Parameterized test covering many configurations ──────────────────────

    static Stream<Arguments> reduceSumConfigs() {
        return Stream.of(
                // shape, keepDims, axes
                Arguments.of(new long[]{4, 8}, false, new long[]{0}),
                Arguments.of(new long[]{4, 8}, false, new long[]{1}),
                Arguments.of(new long[]{4, 8}, true, new long[]{0}),
                Arguments.of(new long[]{4, 8}, true, new long[]{1}),
                Arguments.of(new long[]{4, 8}, false, new long[]{-1}),
                Arguments.of(new long[]{4, 8}, true, new long[]{-1}),
                Arguments.of(new long[]{2, 4, 8}, false, new long[]{0}),
                Arguments.of(new long[]{2, 4, 8}, false, new long[]{1}),
                Arguments.of(new long[]{2, 4, 8}, false, new long[]{2}),
                Arguments.of(new long[]{2, 4, 8}, false, new long[]{-1}),
                Arguments.of(new long[]{2, 4, 8}, true, new long[]{-1}),
                Arguments.of(new long[]{2, 4, 8}, false, new long[]{0, 2}),
                Arguments.of(new long[]{2, 4, 8}, true, new long[]{0, 2}),
                Arguments.of(new long[]{1, 4, 8, 16}, false, new long[]{-1}),
                Arguments.of(new long[]{1, 4, 8, 16}, true, new long[]{-1}),
                Arguments.of(new long[]{1, 4, 8, 16}, false, new long[]{2, 3}),
                Arguments.of(new long[]{1, 4, 8, 16}, true, new long[]{2, 3})
        );
    }

    @ParameterizedTest(name = "shape={0}, keepDims={1}, axes={2}")
    @MethodSource("reduceSumConfigs")
    @DisplayName("DSP reduce_sum: parameterized configurations")
    public void testReduceSumParameterized(long[] shape, boolean keepDims, long[] axes) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, shape);
        SDVariable out = sd.sum("out", x, keepDims, axes);

        INDArray input = Nd4j.randn(DataType.FLOAT, shape);
        try {
            compareStandardVsDsp(sd, Map.of("x", input), "out");
        } finally {
            sd.close();
        }
    }
}
