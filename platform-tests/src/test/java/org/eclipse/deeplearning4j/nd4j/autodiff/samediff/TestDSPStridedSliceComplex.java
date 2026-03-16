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
 * Complex DSP isolation tests for strided_slice.
 *
 * These tests target the known bug: native DSP view creation for strided_slice
 * uses offset 0 into the input buffer, which is WRONG for slices that start
 * at non-zero positions (e.g., x[1:3,:]).
 *
 * Tests cover:
 * - Non-zero offset slices (the core bug)
 * - strided_slice with inputs from other ops' outputs (SOURCE_OP_OUTPUT wiring)
 * - Multi-op chains: slice → matmul, slice → add → relu, etc.
 * - Both Java DSP and native DSP paths
 * - Multi-step execution (stale view data detection)
 * - Complex graph patterns that mirror VLM model structure
 */
@Slf4j
@DisplayName("DSP StridedSlice Complex Isolation Tests")
public class TestDSPStridedSliceComplex {

    private static final double TOL = 1e-4;

    // ── Helper: compare standard vs DSP execution ────────────────────────

    private void assertDspMatchesStandard(SameDiff sd, Map<String, INDArray> placeholders,
                                           String outputName, boolean nativeDsp) {
        // Standard execution
        sd.setDspAutoCompileEnabled(false);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray expected = sd.output(placeholders, outputName).get(outputName).dup();

        // DSP execution
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(nativeDsp);
        sd.clearDynamicShapePlanCache();
        sd.resetSession();
        INDArray actual = sd.output(placeholders, outputName).get(outputName);

        String pathName = nativeDsp ? "Native DSP" : "Java DSP";
        assertArrayEquals(expected.shape(), actual.shape(),
                pathName + ": shape mismatch — expected " + java.util.Arrays.toString(expected.shape())
                        + " got " + java.util.Arrays.toString(actual.shape()));
        double maxDiff = actual.sub(expected).amaxNumber().doubleValue();
        log.info("{}: maxDiff={}", pathName, maxDiff);
        assertTrue(maxDiff < TOL,
                pathName + ": result diverges by " + maxDiff);
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group A: Non-zero offset slices (the core view offset bug)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("A1: Slice at non-zero row offset — Java DSP")
    void testNonZeroRowOffsetSliceJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // x[2:4, :] → [2, 8] — starts at row 2, NOT row 0
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 8}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("A2: Slice at non-zero row offset — Native DSP")
    void testNonZeroRowOffsetSliceNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 8}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    @Test
    @DisplayName("A3: Slice at non-zero column offset — Java DSP")
    void testNonZeroColOffsetSliceJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // x[:, 3:7] → [4, 4] — columns 3-6
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 3}, new long[]{4, 7}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("A4: Slice in middle of 3D tensor — Java DSP")
    void testMiddle3DSliceJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6, 8);
        // x[0:1, 2:5, :] → [1, 3, 8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 2, 0}, new long[]{1, 5, 8}, new long[]{1, 1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 96, 96, DataType.FLOAT).reshape(2, 6, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("A5: Slice in middle of 3D tensor — Native DSP")
    void testMiddle3DSliceNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 6, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 2, 0}, new long[]{1, 5, 8}, new long[]{1, 1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 96, 96, DataType.FLOAT).reshape(2, 6, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group B: strided_slice with SOURCE_OP_OUTPUT inputs
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("B1: Slice input produced by matmul — Java DSP")
    void testSliceAfterMatmulJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable matmulOut = sd.mmul("mm", x, w); // [1, 8]

        // Reshape to [2, 4], then slice [1:2, :] → [1, 4]
        SDVariable reshaped = sd.reshape("reshaped", matmulOut, 2, 4);
        SDVariable sliced = sd.stridedSlice("sliced", reshaped,
                new long[]{1, 0}, new long[]{2, 4}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("B2: Slice input produced by matmul — Native DSP")
    void testSliceAfterMatmulNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 8).mul(0.1));
        SDVariable matmulOut = sd.mmul("mm", x, w);
        SDVariable reshaped = sd.reshape("reshaped", matmulOut, 2, 4);
        SDVariable sliced = sd.stridedSlice("sliced", reshaped,
                new long[]{1, 0}, new long[]{2, 4}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 4).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    @Test
    @DisplayName("B3: Slice input produced by add+relu chain — Java DSP")
    void testSliceAfterAddReluJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable bias = sd.constant("bias", Nd4j.rand(DataType.FLOAT, 4, 8));
        SDVariable added = x.add("added", bias);
        SDVariable relu = sd.nn.relu("relu", added, 0);

        // Slice relu output [1:3, 2:6] → [2, 4]
        SDVariable sliced = sd.stridedSlice("sliced", relu,
                new long[]{1, 2}, new long[]{3, 6}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(3.0f)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("B4: Slice input produced by concat — Java DSP")
    void testSliceAfterConcatJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x1 = sd.placeHolder("x1", DataType.FLOAT, 2, 4);
        SDVariable x2 = sd.placeHolder("x2", DataType.FLOAT, 3, 4);
        SDVariable concat = sd.concat("concat", 0, x1, x2); // [5, 4]

        // Slice [2:4, :] → [2, 4] — crosses the boundary between x1 and x2
        SDVariable sliced = sd.stridedSlice("sliced", concat,
                new long[]{2, 0}, new long[]{4, 4}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray in1 = Nd4j.linspace(1, 8, 8, DataType.FLOAT).reshape(2, 4);
        INDArray in2 = Nd4j.linspace(9, 20, 12, DataType.FLOAT).reshape(3, 4);
        assertDspMatchesStandard(sd, Map.of("x1", in1, "x2", in2), "out", false);
        sd.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group C: Slice → downstream op chains (data correctness through chain)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("C1: Slice → matmul chain with non-zero offset — Java DSP")
    void testSliceMatmulNonZeroOffsetJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // x[2:4, :] → [2, 8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 8}, new long[]{1, 1});
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 3).mul(0.1));
        SDVariable out = sd.mmul("out", sliced, w);

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("C2: Slice → matmul chain with non-zero offset — Native DSP")
    void testSliceMatmulNonZeroOffsetNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 8}, new long[]{1, 1});
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 3).mul(0.1));
        SDVariable out = sd.mmul("out", sliced, w);

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    @Test
    @DisplayName("C3: Slice → reshape → matmul — Java DSP")
    void testSliceReshapeMatmulJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        // x[0:1, 1:3, :] → [1, 2, 8]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 1, 0}, new long[]{1, 3, 8}, new long[]{1, 1, 1});
        SDVariable reshaped = sd.reshape("reshaped", sliced, 2, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).mul(0.1));
        SDVariable out = sd.mmul("out", reshaped, w);

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(1, 4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("C4: Multiple slices from same input — Java DSP")
    void testMultipleSlicesSameInputJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 6, 4);
        // First two rows
        SDVariable s1 = sd.stridedSlice("s1", x,
                new long[]{0, 0}, new long[]{2, 4}, new long[]{1, 1});
        // Middle two rows
        SDVariable s2 = sd.stridedSlice("s2", x,
                new long[]{2, 0}, new long[]{4, 4}, new long[]{1, 1});
        // Last two rows
        SDVariable s3 = sd.stridedSlice("s3", x,
                new long[]{4, 0}, new long[]{6, 4}, new long[]{1, 1});
        // Sum all three slices
        SDVariable sum12 = s1.add("sum12", s2);
        SDVariable out = sum12.add("out", s3);

        INDArray input = Nd4j.linspace(1, 24, 24, DataType.FLOAT).reshape(6, 4);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group D: Multi-step execution (stale data detection)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("D1: Non-zero offset slice multi-step — Java DSP")
    void testNonZeroOffsetMultiStepJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        // x[2:4, :] → [2, 4]
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 4}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray[] results = new INDArray[5];
        INDArray[] expecteds = new INDArray[5];

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 4).mul(step + 1);

            // Standard
            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            expecteds[step] = sd.output(Map.of("x", input), "out").get("out").dup();

            // Java DSP
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            results[step] = sd.output(Map.of("x", input), "out").get("out").dup();
        }

        for (int step = 0; step < 5; step++) {
            assertArrayEquals(expecteds[step].shape(), results[step].shape(),
                    "Step " + step + " shape mismatch");
            double maxDiff = results[step].sub(expecteds[step]).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL, "Step " + step + " Java DSP diverges by " + maxDiff);
        }

        // Steps must produce different values
        for (int step = 1; step < 5; step++) {
            double diff = results[step].sub(results[0]).amaxNumber().doubleValue();
            assertTrue(diff > TOL, "Step " + step + " should differ from step 0");
        }
        sd.close();
    }

    @Test
    @DisplayName("D2: Non-zero offset slice multi-step — Native DSP")
    void testNonZeroOffsetMultiStepNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 4);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{2, 0}, new long[]{4, 4}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(0.0f)));

        INDArray[] results = new INDArray[5];
        INDArray[] expecteds = new INDArray[5];

        for (int step = 0; step < 5; step++) {
            INDArray input = Nd4j.rand(DataType.FLOAT, 4, 4).mul(step + 1);

            sd.setDspAutoCompileEnabled(false);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            expecteds[step] = sd.output(Map.of("x", input), "out").get("out").dup();

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            sd.clearDynamicShapePlanCache();
            sd.resetSession();
            results[step] = sd.output(Map.of("x", input), "out").get("out").dup();
        }

        for (int step = 0; step < 5; step++) {
            double maxDiff = results[step].sub(expecteds[step]).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL, "Step " + step + " Native DSP diverges by " + maxDiff);
        }
        sd.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group E: VLM-like patterns (complex multi-op graphs)
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("E1: Transformer-style slice pattern — shape op → slice → matmul — Java DSP")
    void testTransformerSlicePatternJavaDsp() {
        // Simulates: linear projection → reshape (split heads) → slice (select head range)
        //            → matmul (attention scores)
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable projW = sd.constant("projW", Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1));

        // Project: [1,1,16] → reshape [1,16] → matmul → [1,32]
        SDVariable flat = sd.reshape("flat", x, 1, 16);
        SDVariable projected = sd.mmul("proj", flat, projW); // [1, 32]

        // Split into 4 heads: reshape [1, 4, 8]
        SDVariable heads = sd.reshape("heads", projected, 1, 4, 8);

        // Select heads 1-2: [0:1, 1:3, :] → [1, 2, 8]
        SDVariable selectedHeads = sd.stridedSlice("selected", heads,
                new long[]{0, 1, 0}, new long[]{1, 3, 8}, new long[]{1, 1, 1});

        // Attention-like matmul: [1, 2, 8] × [8, 4] → [1, 2, 4]
        SDVariable attnW = sd.constant("attnW", Nd4j.randn(DataType.FLOAT, 8, 4).mul(0.1));
        SDVariable reshaped = sd.reshape("reshaped", selectedHeads, 2, 8);
        SDVariable scores = sd.mmul("scores", reshaped, attnW);
        SDVariable out = scores.add("out", sd.constant("b", Nd4j.zeros(DataType.FLOAT, 2, 4)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("E2: Transformer-style slice pattern — Native DSP")
    void testTransformerSlicePatternNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 1, 16);
        SDVariable projW = sd.constant("projW", Nd4j.randn(DataType.FLOAT, 16, 32).mul(0.1));

        SDVariable flat = sd.reshape("flat", x, 1, 16);
        SDVariable projected = sd.mmul("proj", flat, projW);
        SDVariable heads = sd.reshape("heads", projected, 1, 4, 8);
        SDVariable selectedHeads = sd.stridedSlice("selected", heads,
                new long[]{0, 1, 0}, new long[]{1, 3, 8}, new long[]{1, 1, 1});
        SDVariable attnW = sd.constant("attnW", Nd4j.randn(DataType.FLOAT, 8, 4).mul(0.1));
        SDVariable reshaped = sd.reshape("reshaped", selectedHeads, 2, 8);
        SDVariable scores = sd.mmul("scores", reshaped, attnW);
        SDVariable out = scores.add("out", sd.constant("b", Nd4j.zeros(DataType.FLOAT, 2, 4)));

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 1, 16).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    @Test
    @DisplayName("E3: KV cache slice pattern — slice past positions — Java DSP")
    void testKvCacheSlicePatternJavaDsp() {
        // Simulates: static KV buffer → slice [0:cachePos] → attention
        SameDiff sd = SameDiff.create();
        int heads = 4, dim = 8, maxLen = 16, cachePos = 10;
        SDVariable kvBuf = sd.placeHolder("kv", DataType.FLOAT, 1, heads, maxLen, dim);
        // Slice to [1, 4, 10, 8] — first cachePos positions
        SDVariable past = sd.stridedSlice("past", kvBuf,
                new long[]{0, 0, 0, 0}, new long[]{1, heads, cachePos, dim}, new long[]{1, 1, 1, 1});
        SDVariable query = sd.placeHolder("q", DataType.FLOAT, 1, heads, 1, dim);
        // Transpose past keys: [1, 4, 10, 8] → permute [1, 4, 8, 10]
        SDVariable pastT = sd.permute("pastT", past, 0, 1, 3, 2);
        // Attention scores: [1, 4, 1, 8] × [1, 4, 8, 10] → [1, 4, 1, 10]
        SDVariable scores = sd.mmul("scores", query, pastT);
        SDVariable out = scores.mul("out", sd.constant("scale", Nd4j.scalar((float)(1.0 / Math.sqrt(dim)))));

        INDArray kvInput = Nd4j.randn(DataType.FLOAT, 1, heads, maxLen, dim).mul(0.1);
        INDArray qInput = Nd4j.randn(DataType.FLOAT, 1, heads, 1, dim).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("kv", kvInput, "q", qInput), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("E4: KV cache slice pattern — Native DSP")
    void testKvCacheSlicePatternNativeDsp() {
        SameDiff sd = SameDiff.create();
        int heads = 4, dim = 8, maxLen = 16, cachePos = 10;
        SDVariable kvBuf = sd.placeHolder("kv", DataType.FLOAT, 1, heads, maxLen, dim);
        SDVariable past = sd.stridedSlice("past", kvBuf,
                new long[]{0, 0, 0, 0}, new long[]{1, heads, cachePos, dim}, new long[]{1, 1, 1, 1});
        SDVariable query = sd.placeHolder("q", DataType.FLOAT, 1, heads, 1, dim);
        SDVariable pastT = sd.permute("pastT", past, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", query, pastT);
        SDVariable out = scores.mul("out", sd.constant("scale", Nd4j.scalar((float)(1.0 / Math.sqrt(dim)))));

        INDArray kvInput = Nd4j.randn(DataType.FLOAT, 1, heads, maxLen, dim).mul(0.1);
        INDArray qInput = Nd4j.randn(DataType.FLOAT, 1, heads, 1, dim).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("kv", kvInput, "q", qInput), "out", true);
        sd.close();
    }

    @Test
    @DisplayName("E5: Long chain: matmul → relu → reshape → slice → matmul → add — Java DSP")
    void testLongChainWithSliceJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16).mul(0.1));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));

        // Layer 1: matmul → add → relu
        SDVariable h1 = sd.mmul("h1_mm", x, w1);
        SDVariable h1b = h1.add("h1_add", b1);
        SDVariable h1a = sd.nn.relu("h1_relu", h1b, 0);

        // Reshape [1, 16] → [4, 4]
        SDVariable reshaped = sd.reshape("reshaped", h1a, 4, 4);

        // Slice [1:3, :] → [2, 4]
        SDVariable sliced = sd.stridedSlice("sliced", reshaped,
                new long[]{1, 0}, new long[]{3, 4}, new long[]{1, 1});

        // Layer 2: matmul → add
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 4, 2).mul(0.1));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 2, 2));
        SDVariable h2 = sd.mmul("h2_mm", sliced, w2);
        SDVariable out = h2.add("out", b2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("E6: Long chain with slice — Native DSP")
    void testLongChainWithSliceNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 8);
        SDVariable w1 = sd.constant("w1", Nd4j.randn(DataType.FLOAT, 8, 16).mul(0.1));
        SDVariable b1 = sd.constant("b1", Nd4j.zeros(DataType.FLOAT, 1, 16));

        SDVariable h1 = sd.mmul("h1_mm", x, w1);
        SDVariable h1b = h1.add("h1_add", b1);
        SDVariable h1a = sd.nn.relu("h1_relu", h1b, 0);
        SDVariable reshaped = sd.reshape("reshaped", h1a, 4, 4);
        SDVariable sliced = sd.stridedSlice("sliced", reshaped,
                new long[]{1, 0}, new long[]{3, 4}, new long[]{1, 1});
        SDVariable w2 = sd.constant("w2", Nd4j.randn(DataType.FLOAT, 4, 2).mul(0.1));
        SDVariable b2 = sd.constant("b2", Nd4j.zeros(DataType.FLOAT, 2, 2));
        SDVariable h2 = sd.mmul("h2_mm", sliced, w2);
        SDVariable out = h2.add("out", b2);

        INDArray input = Nd4j.randn(DataType.FLOAT, 1, 8).mul(0.1);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }

    // ══════════════════════════════════════════════════════════════════════
    // Group F: Edge cases
    // ══════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("F1: Slice at offset 0 (should work even with view bug) — Java DSP")
    void testSliceAtOffset0JavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // x[0:2, :] → [2, 8] — starts at offset 0
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 0}, new long[]{2, 8}, new long[]{1, 1});
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("F2: Slice entire tensor (noop slice) — Java DSP")
    void testNoopSliceJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // x[0:4, 0:8] → [4, 8] — entire tensor
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{0, 0}, new long[]{4, 8}, new long[]{1, 1});
        SDVariable out = sliced.add("out", sd.constant("b", Nd4j.scalar(1.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("F3: Slice with shrink_axis at non-zero index — Java DSP")
    void testShrinkAxisNonZeroIndexJavaDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        // Select row 3 → [8] (shrink dim 0 at index 3)
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{3, 0}, new long[]{4, 8}, new long[]{1, 1},
                0, 0, 0, 0, 1);
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", false);
        sd.close();
    }

    @Test
    @DisplayName("F4: Slice with shrink_axis at non-zero index — Native DSP")
    void testShrinkAxisNonZeroIndexNativeDsp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 4, 8);
        SDVariable sliced = sd.stridedSlice("sliced", x,
                new long[]{3, 0}, new long[]{4, 8}, new long[]{1, 1},
                0, 0, 0, 0, 1);
        SDVariable out = sliced.mul("out", sd.constant("s", Nd4j.scalar(2.0f)));

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(4, 8);
        assertDspMatchesStandard(sd, Map.of("x", input), "out", true);
        sd.close();
    }
}
