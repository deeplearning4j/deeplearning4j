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
import org.nd4j.linalg.api.ops.impl.layers.convolution.config.Conv2DConfig;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests DSP (Dynamic Shape Plan) execution for vision-encoder-like graphs.
 *
 * Reproduces three failures seen in SmolDocling-256M vision encoder with native DSP:
 *
 * 1. reshape_no_copy stale shape cache across frames (Slot 1310):
 *    "Cannot reshape array of size 768 into shape with known dimensions product 786432"
 *    Root cause: frozen-shapes fast path blindly trusts cache when input spatial dims change.
 *
 * 2. conv2d KERNEL_FAILURE (Slot 281):
 *    status 50 on Frame 1 because output array was allocated with wrong spatial dimensions
 *    from stale shape cache.
 *
 * 3. reduce_sum wrong target shape (Slot 1210):
 *    "wrong target shape!" from reduceAlongDimension because pre-allocated output has
 *    shape from prior execution with different spatial dims.
 *
 * Common root cause: shape caching retains dimensions from prior execution when
 * shapesFrozen_ is true or capturable segments skip cache clearing.
 */
public class TestDSPVisionEncoderShapes extends BaseNd4jTestWithBackends {

    private static final Logger log = LoggerFactory.getLogger(TestDSPVisionEncoderShapes.class);
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
        log.info("Standard output shape: {}", Arrays.toString(expected.shape()));

        // 2. Reset and compile native DSP
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // 3. DSP execution
        Map<String, INDArray> dspResult = sd.output(placeholders, outputName);
        INDArray actual = dspResult.get(outputName).dup();

        // 4. Compare
        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch: expected " + Arrays.toString(expected.shape())
                        + " but got " + Arrays.toString(actual.shape()));
        assertEquals(expected.dataType(), actual.dataType(), "DataType mismatch");

        double maxDiff = expected.sub(actual.castTo(expected.dataType())).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOL,
                "Value mismatch: max diff = " + maxDiff + " (tolerance = " + TOL + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. reshape_no_copy stale shape cache: vision encoder pattern
    //    Reproduces Slot 1310: 768 elements vs 786432 expected
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Vision encoder reshape: [1, patches, hidden] → [-1, hidden] with changing patch count")
    public void testReshapeStaleCache_ChangingPatchCount() {
        // Mimics vision encoder: input [1, numPatches, hiddenDim] → flatten to [numPatches, hiddenDim]
        // SmolDocling: frame 1 has 1024 patches, frame 2 has 1 patch
        // Bug: frame 2 tries to reshape 768 elements into shape requiring 786432 (1024*768)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, -1, 768);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 768);
        SDVariable output = sd.math.add("output", reshaped, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Frame 1: large patch count (like full image)
        INDArray frame1 = Nd4j.rand(DataType.FLOAT, 1, 1024, 768);
        Map<String, INDArray> r1 = sd.output(Map.of("input", frame1), "output");
        assertArrayEquals(new long[]{1024, 768}, r1.get("output").shape(),
                "Frame 1 should produce [1024, 768]");
        log.info("Frame 1: output shape = {}", Arrays.toString(r1.get("output").shape()));

        // Frame 2: single patch (dramatically smaller)
        // This is where the stale cache bug manifests: cache has [1024, 768]
        INDArray frame2 = Nd4j.rand(DataType.FLOAT, 1, 1, 768);
        Map<String, INDArray> r2 = sd.output(Map.of("input", frame2), "output");
        assertArrayEquals(new long[]{1, 768}, r2.get("output").shape(),
                "Frame 2 should produce [1, 768] but stale cache gives [1024, 768]");
        log.info("Frame 2: output shape = {}", Arrays.toString(r2.get("output").shape()));

        // Frame 3: medium (different again)
        INDArray frame3 = Nd4j.rand(DataType.FLOAT, 1, 256, 768);
        Map<String, INDArray> r3 = sd.output(Map.of("input", frame3), "output");
        assertArrayEquals(new long[]{256, 768}, r3.get("output").shape(),
                "Frame 3 should produce [256, 768]");
    }

    @Test
    @DisplayName("Vision encoder reshape: exact SmolDocling 786432 vs 768 mismatch")
    public void testReshapeStaleCache_ExactSmolDoclingMismatch() {
        // Exact reproduction of "Cannot reshape array of size 768 into shape with
        // known dimensions product 786432" (786432 = 1024 * 768)
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 768);
        // This reshape merges batch and seq dims
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 768);
        // Add downstream op to prevent graph from being trivial
        SDVariable weights = sd.constant("w", Nd4j.rand(DataType.FLOAT, 768, 64));
        SDVariable mm = sd.mmul("mm", reshaped, weights);
        SDVariable output = sd.math.add("output", mm, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Frame 1: [1, 1024, 768] → reshape → [1024, 768] → mm → [1024, 64]
        INDArray f1 = Nd4j.rand(DataType.FLOAT, 1, 1024, 768);
        Map<String, INDArray> r1 = sd.output(Map.of("input", f1), "output");
        assertArrayEquals(new long[]{1024, 64}, r1.get("output").shape());

        // Frame 2: [1, 1, 768] → reshape → [1, 768] → mm → [1, 64]
        // Bug: cache still thinks output is [1024, 768], reshape fails
        INDArray f2 = Nd4j.rand(DataType.FLOAT, 1, 1, 768);
        Map<String, INDArray> r2 = sd.output(Map.of("input", f2), "output");
        assertArrayEquals(new long[]{1, 64}, r2.get("output").shape(),
                "Frame 2: expected [1, 64] but stale cache causes mismatch");

        // Frame 3: back to large
        INDArray f3 = Nd4j.rand(DataType.FLOAT, 1, 512, 768);
        Map<String, INDArray> r3 = sd.output(Map.of("input", f3), "output");
        assertArrayEquals(new long[]{512, 64}, r3.get("output").shape());
    }

    @Test
    @DisplayName("Vision encoder: reshape in conv → reshape → linear pipeline")
    public void testReshapeInConvPipeline() {
        // Mimics: conv2d output [1, C, H, W] → reshape to [1, H*W, C] → linear
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);

        // Conv2d: [1, 3, H, W] → [1, 16, H, W] (same padding)
        // Weights: [kH, kW, inC, outC]
        SDVariable convW = sd.constant("convW", Nd4j.rand(DataType.FLOAT, 3, 3, 3, 16));
        SDVariable convB = sd.constant("convB", Nd4j.zeros(DataType.FLOAT, 16));
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();
        SDVariable conv = sd.cnn.conv2d("conv", input, convW, convB, config);

        // Reshape: [1, 16, H, W] → [H*W, 16]
        SDVariable reshaped = sd.reshape("reshaped", conv, -1, 16);

        // reduce_sum along last axis: [H*W, 16] → [H*W]
        SDVariable output = sd.sum("output", reshaped, false, -1);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Frame 1: 14x14 spatial
        INDArray f1 = Nd4j.rand(DataType.FLOAT, 1, 3, 14, 14);
        Map<String, INDArray> r1 = sd.output(Map.of("input", f1), "output");
        assertEquals(14 * 14, r1.get("output").length(),
                "Frame 1: expected 196 elements, got " + r1.get("output").length());

        // Frame 2: 7x7 spatial (different size)
        INDArray f2 = Nd4j.rand(DataType.FLOAT, 1, 3, 7, 7);
        Map<String, INDArray> r2 = sd.output(Map.of("input", f2), "output");
        assertEquals(7 * 7, r2.get("output").length(),
                "Frame 2: expected 49 elements, got " + r2.get("output").length());

        // Frame 3: 28x28 spatial (larger)
        INDArray f3 = Nd4j.rand(DataType.FLOAT, 1, 3, 28, 28);
        Map<String, INDArray> r3 = sd.output(Map.of("input", f3), "output");
        assertEquals(28 * 28, r3.get("output").length(),
                "Frame 3: expected 784 elements, got " + r3.get("output").length());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. conv2d with changing spatial dimensions
    //    Reproduces Slot 281: KERNEL_FAILURE from wrong output allocation
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Conv2d: changing spatial dims across frames")
    public void testConv2dChangingSpatialDims() {
        // Conv2d output shape depends on input spatial dims
        // Stale shape cache can allocate wrong-size output buffer → KERNEL_FAILURE
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        SDVariable weights = sd.constant("w", Nd4j.rand(DataType.FLOAT, 3, 3, 3, 8));
        SDVariable bias = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 8));
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();
        SDVariable conv = sd.cnn.conv2d("conv", input, weights, bias, config);
        SDVariable output = sd.nn.relu("output", conv, 0.0);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] spatialSizes = {{14, 14}, {7, 7}, {28, 28}, {14, 14}};
        for (int[] hw : spatialSizes) {
            int h = hw[0], w = hw[1];
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, h, w);
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null output for " + h + "x" + w);
            // Same padding: output spatial = input spatial
            assertArrayEquals(new long[]{1, 8, h, w}, out.shape(),
                    "Shape wrong for " + h + "x" + w + ": got " + Arrays.toString(out.shape()));
            log.info("conv2d {}x{}: output shape = {}", h, w, Arrays.toString(out.shape()));
        }
    }

    @Test
    @DisplayName("Conv2d: standard vs DSP comparison using separate graphs")
    public void testConv2dStandardVsDsp() {
        // Use separate SameDiff instances to avoid constant corruption between paths.
        // Standard execution can modify constants in-place; sharing constants between
        // standard and DSP paths causes value mismatches unrelated to shape caching.
        INDArray w = Nd4j.rand(DataType.FLOAT, 3, 3, 3, 16);
        INDArray b = Nd4j.zeros(DataType.FLOAT, 16);
        INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, 14, 14);
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();

        // Standard execution on fresh graph
        SameDiff sdRef = SameDiff.create();
        sdRef.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        sdRef.constant("w", w.dup());
        sdRef.constant("b", b.dup());
        sdRef.cnn.conv2d("output", sdRef.getVariable("input"), sdRef.getVariable("w"),
                sdRef.getVariable("b"), config);
        Map<String, INDArray> refResult = sdRef.output(Map.of("input", in), "output");
        INDArray expected = refResult.get("output").dup();

        // DSP execution on separate fresh graph
        SameDiff sdDsp = SameDiff.create();
        sdDsp.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        sdDsp.constant("w", w.dup());
        sdDsp.constant("b", b.dup());
        sdDsp.cnn.conv2d("output", sdDsp.getVariable("input"), sdDsp.getVariable("w"),
                sdDsp.getVariable("b"), config);
        sdDsp.setDspAutoCompileEnabled(true);
        sdDsp.setDspNativeAutoCompileEnabled(true);

        Map<String, INDArray> dspResult = sdDsp.output(Map.of("input", in), "output");
        INDArray actual = dspResult.get("output").dup();

        // Shape must match
        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch: expected " + Arrays.toString(expected.shape())
                        + " but got " + Arrays.toString(actual.shape()));
        assertEquals(expected.dataType(), actual.dataType(), "DataType mismatch");

        // Per-channel sums must match
        int channels = (int) expected.shape()[1];
        for (int c = 0; c < channels; c++) {
            double expChanSum = expected.slice(0).slice(c).sumNumber().doubleValue();
            double actChanSum = actual.slice(0).slice(c).sumNumber().doubleValue();
            double chanDiff = Math.abs(expChanSum - actChanSum);
            System.out.println("Channel " + c + ": expected=" + expChanSum
                    + " actual=" + actChanSum + " diff=" + chanDiff);
            assertTrue(chanDiff < TOL,
                    "Channel " + c + " sum mismatch: expected=" + expChanSum
                            + " actual=" + actChanSum + " diff=" + chanDiff);
        }
    }

    @Test
    @DisplayName("Conv2d: stride=2 with changing spatial dims")
    public void testConv2dStride2ChangingSpatial() {
        // Stride=2 halves spatial dims — different formula, more cache-sensitive
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        SDVariable weights = sd.constant("w", Nd4j.rand(DataType.FLOAT, 3, 3, 3, 8));
        SDVariable bias = sd.constant("b", Nd4j.zeros(DataType.FLOAT, 8));
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(2).sW(2).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();
        SDVariable conv = sd.cnn.conv2d("conv", input, weights, bias, config);
        SDVariable output = sd.math.add("output", conv, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // With stride=2, padding=1, kernel=3: outH = (inH + 2*1 - 3)/2 + 1 = (inH-1)/2 + 1
        int[][] spatials = {{14, 14}, {28, 28}, {7, 7}, {16, 16}};
        for (int[] hw : spatials) {
            int h = hw[0], w = hw[1];
            int outH = (h + 2 - 3) / 2 + 1;
            int outW = (w + 2 - 3) / 2 + 1;
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, h, w);
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null for " + h + "x" + w);
            assertArrayEquals(new long[]{1, 8, outH, outW}, out.shape(),
                    "Shape wrong for input " + h + "x" + w + ": expected [1,8," + outH + "," + outW
                            + "] got " + Arrays.toString(out.shape()));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. reduce_sum wrong target shape with dynamic inputs
    //    Reproduces Slot 1210: "wrong target shape!" from reduceAlongDimension
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("reduce_sum: wrong target shape with changing input dims across frames")
    public void testReduceSumWrongTargetShape_ChangingDims() {
        // reduce_sum output depends on input shape
        // Stale cache gives wrong output shape → "wrong target shape!" assertion
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 64);
        SDVariable sum = sd.sum("sum", input, false, -1);  // [batch, seq, 64] → [batch, seq]
        SDVariable output = sd.math.add("output", sum, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] configs = {{1, 1024}, {1, 256}, {1, 512}, {2, 128}};
        for (int[] cfg : configs) {
            int batch = cfg[0], seq = cfg[1];
            INDArray in = Nd4j.rand(DataType.FLOAT, batch, seq, 64);
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null for batch=" + batch + " seq=" + seq);
            assertArrayEquals(new long[]{batch, seq}, out.shape(),
                    "Shape wrong for [" + batch + "," + seq + ",64] → reduce_sum axis=-1: expected ["
                            + batch + "," + seq + "] got " + Arrays.toString(out.shape()));
            log.info("reduce_sum [{}x{}x64] → {}", batch, seq, Arrays.toString(out.shape()));
        }
    }

    @Test
    @DisplayName("reduce_sum: keepDims=true with changing spatial dims")
    public void testReduceSumKeepDims_ChangingSpatial() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, -1, -1, 32);
        // Reduce over last axis, keepDims: [1, H, W, 32] → [1, H, W, 1]
        SDVariable sum = sd.sum("sum", input, true, -1);
        SDVariable output = sd.math.add("output", sum, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] spatials = {{14, 14}, {7, 7}, {28, 28}, {3, 3}};
        for (int[] hw : spatials) {
            int h = hw[0], w = hw[1];
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, h, w, 32);
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertArrayEquals(new long[]{1, h, w, 1}, out.shape(),
                    "Shape wrong for [1," + h + "," + w + ",32] → reduce keepDims: expected [1," + h
                            + "," + w + ",1] got " + Arrays.toString(out.shape()));
        }
    }

    @Test
    @DisplayName("reduce_sum: spatial reduce [1,C,H,W] axes=[2,3] with changing H,W")
    public void testReduceSumSpatialReduce_ChangingHW() {
        // Spatial pooling: [1, channels, H, W] → reduce over H,W → [1, channels]
        // Common in vision encoders before classification head
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 32, -1, -1);
        SDVariable sum = sd.sum("sum", input, false, 2, 3);
        SDVariable output = sd.math.add("output", sum, 0.0f);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] spatials = {{14, 14}, {7, 7}, {28, 28}, {1, 1}};
        for (int[] hw : spatials) {
            int h = hw[0], w = hw[1];
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, 32, h, w);
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertArrayEquals(new long[]{1, 32}, out.shape(),
                    "Spatial reduce should always be [1, 32] regardless of H,W. Got "
                            + Arrays.toString(out.shape()) + " for H=" + h + " W=" + w);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Combined pipeline: conv → reshape → reduce (all three failures together)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Full vision encoder pipeline: conv → reshape → reduce with changing spatial dims")
    public void testFullVisionEncoderPipeline() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);

        // Conv2d: [1, 3, H, W] → [1, 16, H, W]
        SDVariable convW = sd.constant("convW", Nd4j.rand(DataType.FLOAT, 3, 3, 3, 16));
        SDVariable convB = sd.constant("convB", Nd4j.zeros(DataType.FLOAT, 16));
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();
        SDVariable conv = sd.cnn.conv2d("conv", input, convW, convB, config);

        // Reshape: [1, 16, H, W] → [H*W, 16]
        SDVariable reshaped = sd.reshape("reshaped", conv, -1, 16);

        // Linear: [H*W, 16] → [H*W, 8]
        SDVariable linearW = sd.constant("linearW", Nd4j.rand(DataType.FLOAT, 16, 8));
        SDVariable mm = sd.mmul("mm", reshaped, linearW);

        // reduce_sum: [H*W, 8] → [H*W]
        SDVariable output = sd.sum("output", mm, false, -1);

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        int[][] spatials = {{14, 14}, {7, 7}, {28, 28}, {14, 14}};
        for (int[] hw : spatials) {
            int h = hw[0], w = hw[1];
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, h, w);
            System.out.println("=== Pipeline frame: " + h + "x" + w + " ===");
            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");
            assertNotNull(out, "Null output for " + h + "x" + w);
            assertEquals(h * w, out.length(),
                    "Pipeline output should have " + (h * w) + " elements for " + h + "x" + w
                            + ", got " + out.length());
            log.info("Pipeline {}x{}: output shape = {}, length = {}",
                    h, w, Arrays.toString(out.shape()), out.length());
        }
    }

    @Test
    @DisplayName("Full vision encoder pipeline: standard vs DSP comparison")
    public void testFullVisionEncoderPipeline_StandardVsDsp() {
        // Use separate SameDiff instances to avoid constant corruption between paths.
        // Standard execution can modify constants in-place; sharing constants between
        // standard and DSP paths causes value mismatches unrelated to shape caching.
        INDArray convWArr = Nd4j.rand(DataType.FLOAT, 3, 3, 3, 16);
        INDArray convBArr = Nd4j.zeros(DataType.FLOAT, 16);
        INDArray linearWArr = Nd4j.rand(DataType.FLOAT, 16, 8);
        INDArray in = Nd4j.rand(DataType.FLOAT, 1, 3, 14, 14);
        Conv2DConfig config = Conv2DConfig.builder()
                .kH(3).kW(3).sH(1).sW(1).pH(1).pW(1)
                .dataFormat("NCHW")
                .build();

        // Standard execution on fresh graph
        SameDiff sdRef = SameDiff.create();
        sdRef.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        sdRef.constant("convW", convWArr.dup());
        sdRef.constant("convB", convBArr.dup());
        sdRef.cnn.conv2d("conv", sdRef.getVariable("input"), sdRef.getVariable("convW"),
                sdRef.getVariable("convB"), config);
        sdRef.reshape("reshaped", sdRef.getVariable("conv"), -1, 16);
        sdRef.constant("linearW", linearWArr.dup());
        sdRef.mmul("mm", sdRef.getVariable("reshaped"), sdRef.getVariable("linearW"));
        sdRef.sum("output", sdRef.getVariable("mm"), false, -1);
        Map<String, INDArray> refResult = sdRef.output(Map.of("input", in), "conv", "reshaped", "mm", "output");
        INDArray expectedConv = refResult.get("conv").dup();
        INDArray expectedReshaped = refResult.get("reshaped").dup();
        INDArray expectedMm = refResult.get("mm").dup();
        INDArray expected = refResult.get("output").dup();
        log.info("Standard output shape: {}", Arrays.toString(expected.shape()));

        // DSP execution on separate fresh graph
        SameDiff sdDsp = SameDiff.create();
        sdDsp.placeHolder("input", DataType.FLOAT, 1, 3, -1, -1);
        sdDsp.constant("convW", convWArr.dup());
        sdDsp.constant("convB", convBArr.dup());
        sdDsp.cnn.conv2d("conv", sdDsp.getVariable("input"), sdDsp.getVariable("convW"),
                sdDsp.getVariable("convB"), config);
        sdDsp.reshape("reshaped", sdDsp.getVariable("conv"), -1, 16);
        sdDsp.constant("linearW", linearWArr.dup());
        sdDsp.mmul("mm", sdDsp.getVariable("reshaped"), sdDsp.getVariable("linearW"));
        sdDsp.sum("output", sdDsp.getVariable("mm"), false, -1);
        sdDsp.setDspAutoCompileEnabled(true);
        sdDsp.setDspNativeAutoCompileEnabled(true);
        Map<String, INDArray> dspResult = sdDsp.output(Map.of("input", in), "conv", "reshaped", "mm", "output");
        INDArray actualConv = dspResult.get("conv").dup();
        INDArray actualReshaped = dspResult.get("reshaped").dup();
        INDArray actualMm = dspResult.get("mm").dup();
        INDArray actual = dspResult.get("output").dup();

        // Compare each intermediate to find where mismatch starts
        double convDiff = expectedConv.sub(actualConv).amaxNumber().doubleValue();
        double reshapedDiff = expectedReshaped.sub(actualReshaped).amaxNumber().doubleValue();
        double mmDiff = expectedMm.sub(actualMm).amaxNumber().doubleValue();
        double outputDiff = expected.sub(actual).amaxNumber().doubleValue();
        System.out.println("=== Intermediate comparison ===");
        System.out.println("conv: shape=" + Arrays.toString(actualConv.shape()) + " maxDiff=" + convDiff);
        System.out.println("reshaped: shape=" + Arrays.toString(actualReshaped.shape()) + " maxDiff=" + reshapedDiff);
        System.out.println("mm: shape=" + Arrays.toString(actualMm.shape()) + " maxDiff=" + mmDiff);
        System.out.println("output: shape=" + Arrays.toString(actual.shape()) + " maxDiff=" + outputDiff);

        // Print strides for conv output to check contiguity
        System.out.println("Standard conv strides: " + Arrays.toString(expectedConv.stride()));
        System.out.println("DSP conv strides: " + Arrays.toString(actualConv.stride()));

        assertArrayEquals(expected.shape(), actual.shape(),
                "Shape mismatch: expected " + Arrays.toString(expected.shape())
                        + " but got " + Arrays.toString(actual.shape()));
        assertEquals(expected.dataType(), actual.dataType(), "DataType mismatch");
        assertTrue(outputDiff < TOL,
                "Value mismatch: max diff = " + outputDiff + " (tolerance = " + TOL + ")");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-execution correctness: verify values across frame switches
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("Vision encoder: values correct across frame size changes")
    public void testVisionEncoderValuesAcrossFrames() {
        SameDiff sd = SameDiff.create();
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, 1, -1, 32);
        SDVariable reshaped = sd.reshape("reshaped", input, -1, 32);
        SDVariable sum = sd.sum("output", reshaped, true, -1); // → [-1, 1]

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // Execute with different sizes and verify values match reference
        int[] seqLens = {16, 4, 64, 1};
        for (int seq : seqLens) {
            INDArray in = Nd4j.rand(DataType.FLOAT, 1, seq, 32);
            INDArray expectedSum = in.reshape(seq, 32).sum(true, -1);

            Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
            INDArray out = result.get("output");

            assertArrayEquals(new long[]{seq, 1}, out.shape(),
                    "Shape wrong for seq=" + seq + ": " + Arrays.toString(out.shape()));

            double maxDiff = expectedSum.sub(out).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOL,
                    "Value mismatch for seq=" + seq + ": max diff=" + maxDiff);
            log.info("seq={}: shape={}, maxDiff={}", seq, Arrays.toString(out.shape()), maxDiff);
        }
    }

    @Test
    @DisplayName("Reduce ops: reduce_max, reduce_min, reduce_prod with changing dims")
    public void testOtherReduceOps_ChangingDims() {
        // Verify all reduce ops handle dynamic shapes, not just reduce_sum
        for (String reduceType : new String[]{"max", "min", "prod"}) {
            SameDiff sd = SameDiff.create();
            SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, 8);
            SDVariable reduced;
            switch (reduceType) {
                case "max":
                    reduced = sd.max("reduced", input, false, -1);
                    break;
                case "min":
                    reduced = sd.min("reduced", input, false, -1);
                    break;
                case "prod":
                    reduced = sd.prod("reduced", input, false, -1);
                    break;
                default:
                    throw new IllegalStateException();
            }
            SDVariable output = sd.math.add("output", reduced, 0.0f);

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            int[][] configs = {{2, 4}, {1, 8}, {3, 2}, {1, 1}};
            for (int[] cfg : configs) {
                int batch = cfg[0], seq = cfg[1];
                INDArray in = Nd4j.rand(DataType.FLOAT, batch, seq, 8).addi(0.1f); // avoid zeros for prod
                Map<String, INDArray> result = sd.output(Map.of("input", in), "output");
                INDArray out = result.get("output");
                assertArrayEquals(new long[]{batch, seq}, out.shape(),
                        reduceType + " shape wrong for [" + batch + "," + seq + ",8]: got "
                                + Arrays.toString(out.shape()));
            }
            log.info("reduce_{}: all dynamic shape configs passed", reduceType);
        }
    }
}
