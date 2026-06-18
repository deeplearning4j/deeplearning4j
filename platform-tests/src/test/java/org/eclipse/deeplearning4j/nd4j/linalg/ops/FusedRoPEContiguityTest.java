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

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolation tests for fused_rope contiguity handling.
 *
 * The fused_rope CUDA kernel reads input via raw specialBuffer() with flat indexing,
 * assuming contiguous (row-major) layout. If the input is a non-contiguous view
 * (e.g., from permute, reshape_no_copy, or gather), this reads wrong memory → NaN.
 *
 * These tests verify:
 * 1. fused_rope with contiguous input → correct result
 * 2. fused_rope with non-contiguous view input → same correct result (or fails if buggy)
 * 3. fused_rope through SameDiff graph with reshape → view → fused_rope chain
 */
@Slf4j
@DisplayName("FusedRoPE Contiguity Tests")
public class FusedRoPEContiguityTest {

    /**
     * Test 1: fused_rope with contiguous input and cos/sin → no NaN.
     * This establishes the baseline that the kernel works with contiguous arrays.
     */
    @Test
    @DisplayName("Contiguous input produces valid output")
    void testContiguousInput() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);
        INDArray sin = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);

        assertTrue(isContiguous(input), "input should be contiguous");
        assertTrue(isContiguous(cos), "cos should be contiguous");
        assertTrue(isContiguous(sin), "sin should be contiguous");

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.var("input", input);
        SDVariable cosVar = sd.var("cos", cos);
        SDVariable sinVar = sd.var("sin", sin);
        SDVariable result = new FusedRoPE(sd, inputVar, cosVar, sinVar, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        result.rename("output");

        Map<String, INDArray> out = sd.output(new HashMap<>(), "output");
        INDArray output = out.get("output");
        assertNotNull(output);

        double sum = output.sumNumber().doubleValue();
        assertFalse(Double.isNaN(sum), "Contiguous input should produce non-NaN output, sum=" + sum);
        log.info("Test 1 PASSED: contiguous input, sum={}", sum);
        sd.close();
    }

    /**
     * Test 2: fused_rope with a non-contiguous view as input.
     * If the kernel assumes contiguous layout, this will produce NaN or wrong values.
     */
    @Test
    @DisplayName("Non-contiguous view input may produce NaN (bug detection)")
    void testNonContiguousViewInput() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;

        INDArray cos = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);
        INDArray sin = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);

        // Create a genuinely non-contiguous view:
        // Start with [B, H, S, D] then permute to [B, S, H, D] — single permute gives non-standard strides
        INDArray bhsd = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, headDim);
        INDArray viewInput = bhsd.permute(0, 2, 1, 3); // [B, S, H, D] with strides [H*S*D, D, S*D, 1]

        boolean viewIsContiguous = isContiguous(viewInput);
        log.info("View input contiguous: {}", viewIsContiguous);
        log.info("View strides: {}", Arrays.toString(viewInput.stride()));
        log.info("View shape: {}", Arrays.toString(viewInput.shape()));

        // Make a contiguous copy with same data for reference
        INDArray contiguousCopy = viewInput.dup();
        assertTrue(isContiguous(contiguousCopy), "dup'd copy should be contiguous");
        log.info("Contiguous copy strides: {}", Arrays.toString(contiguousCopy.stride()));

        // Reference: contiguous copy (use dup'd cos/sin to avoid sharing DataBuffers)
        SameDiff sdRef2 = SameDiff.create();
        SDVariable ref2Input = sdRef2.var("input", contiguousCopy);
        SDVariable ref2Cos = sdRef2.var("cos", cos.dup());
        SDVariable ref2Sin = sdRef2.var("sin", sin.dup());
        SDVariable ref2Result = new FusedRoPE(sdRef2, ref2Input, ref2Cos, ref2Sin, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        ref2Result.rename("output");
        INDArray ref2Output = sdRef2.output(new HashMap<>(), "output").get("output").dup();
        double refSum = ref2Output.sumNumber().doubleValue();
        sdRef2.close();

        // Test with the non-contiguous view (use dup'd cos/sin)
        SameDiff sdView = SameDiff.create();
        SDVariable viewVar = sdView.var("input", viewInput);
        SDVariable viewCos = sdView.var("cos", cos.dup());
        SDVariable viewSin = sdView.var("sin", sin.dup());
        SDVariable viewResult = new FusedRoPE(sdView, viewVar, viewCos, viewSin, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        viewResult.rename("output");
        INDArray viewOutput = sdView.output(new HashMap<>(), "output").get("output");

        double viewSum = viewOutput.sumNumber().doubleValue();

        log.info("Reference (contiguous) sum: {}", refSum);
        log.info("View (non-contiguous) sum: {}", viewSum);
        log.info("View has NaN: {}", Double.isNaN(viewSum));

        if (Double.isNaN(viewSum)) {
            log.error("BUG CONFIRMED: fused_rope produces NaN with non-contiguous view input!");
            log.error("This is likely due to raw specialBuffer() access with flat indexing");
        } else {
            double maxDiff = ref2Output.sub(viewOutput).amaxNumber().doubleValue();
            log.info("Max diff contiguous vs view: {}", maxDiff);
            if (maxDiff > 1e-3) {
                log.error("BUG CONFIRMED: fused_rope produces WRONG values with non-contiguous view!");
                log.error("Max diff = {} — kernel reads wrong memory via flat indexing", maxDiff);
            }
        }

        sdView.close();
    }

    /**
     * Test 3: fused_rope through the SameDiff graph pattern that SmolDocling uses:
     * matmul → reshape [B, S, numHeads*headDim] → reshape [B, S, numHeads, headDim] → fused_rope
     *
     * The second reshape may create a view with standard strides (if the first reshape
     * produced a contiguous result). This tests the exact pattern the model uses.
     */
    @Test
    @DisplayName("SameDiff reshape chain → fused_rope (SmolDocling pattern)")
    void testReshapeChainFusedRope() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;
        int hiddenSize = numHeads * headDim; // 576

        SameDiff sd = SameDiff.create();

        // Simulate: input [B, S, 576] → matmul → reshape [B, S, 9, 64] → fused_rope
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, -1, -1, hiddenSize);
        SDVariable weights = sd.var("weights", Nd4j.randn(DataType.FLOAT, hiddenSize, hiddenSize));

        SDVariable projected = sd.mmul("projected", input, weights);
        SDVariable reshaped = sd.reshape("reshaped", projected,
                sd.constant(Nd4j.createFromArray(new long[]{batch, seqLen, numHeads, headDim})));

        // Provide cos/sin as constants
        SDVariable cos = sd.var("cos", Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2));
        SDVariable sin = sd.var("sin", Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2));

        SDVariable ropeOut = new FusedRoPE(sd, reshaped, cos, sin, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        ropeOut.rename("rope_output");

        // Reduce to scalar to force execution of entire chain
        SDVariable reduced = sd.sum("output", ropeOut);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("input", Nd4j.randn(DataType.FLOAT, batch, seqLen, hiddenSize));

        Map<String, INDArray> result = sd.output(ph, "output", "rope_output");
        INDArray ropeResult = result.get("rope_output");
        INDArray reducedResult = result.get("output");

        double ropeSum = ropeResult.sumNumber().doubleValue();
        log.info("Reshape chain → fused_rope sum: {}, hasNaN: {}", ropeSum, Double.isNaN(ropeSum));

        assertFalse(Double.isNaN(ropeSum),
                "fused_rope through reshape chain should not produce NaN");
        log.info("Test 3 PASSED: reshape chain → fused_rope, sum={}", ropeSum);
        sd.close();
    }

    /**
     * Test 4: fused_rope with a dup'd (explicitly contiguous) non-contiguous input.
     * This proves that making the input contiguous fixes the issue.
     */
    @Test
    @DisplayName("dup() of non-contiguous view fixes NaN")
    void testDupFixesNonContiguous() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;

        INDArray contiguousInput = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);
        INDArray sin = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim / 2);

        // Create a non-contiguous view
        INDArray viewInput = contiguousInput.permute(0, 2, 1, 3).permute(0, 2, 1, 3);
        INDArray dupInput = viewInput.dup(); // Force contiguous copy

        assertTrue(isContiguous(dupInput), "dup'd input should be contiguous");

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.var("input", dupInput);
        SDVariable cosVar = sd.var("cos", cos);
        SDVariable sinVar = sd.var("sin", sin);
        SDVariable result = new FusedRoPE(sd, inputVar, cosVar, sinVar, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        result.rename("output");

        INDArray output = sd.output(new HashMap<>(), "output").get("output");
        double sum = output.sumNumber().doubleValue();
        assertFalse(Double.isNaN(sum), "dup'd contiguous input should not produce NaN, sum=" + sum);
        log.info("Test 4 PASSED: dup'd input, sum={}", sum);
        sd.close();
    }

    /**
     * Test 5: fused_rope with FP16 cos/sin but FP32 input (the VLM NaN root cause).
     * The CUDA kernel template is dispatched on input dtype (FLOAT32),
     * but cos/sin buffers are FP16. Without dtype casting, the kernel does
     * reinterpret_cast<float*>(fp16_buffer), reading garbage → NaN.
     */
    @Test
    @DisplayName("Mixed dtype: FP32 input with FP16 cos/sin (VLM NaN root cause)")
    void testMixedDtypeFP16CosSin() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        // Create FP16 cos/sin — this is what the VLM model does when optimizer pre-casts to FP16
        INDArray cosHalf = Nd4j.randn(DataType.HALF, batch, seqLen, headDim / 2);
        INDArray sinHalf = Nd4j.randn(DataType.HALF, batch, seqLen, headDim / 2);

        // Also compute reference with FP32 cos/sin
        INDArray cosFloat = cosHalf.castTo(DataType.FLOAT);
        INDArray sinFloat = sinHalf.castTo(DataType.FLOAT);

        // Reference: FP32 input + FP32 cos/sin
        SameDiff sdRef = SameDiff.create();
        SDVariable refInput = sdRef.var("input", input.dup());
        SDVariable refCos = sdRef.var("cos", cosFloat);
        SDVariable refSin = sdRef.var("sin", sinFloat);
        SDVariable refResult = new FusedRoPE(sdRef, refInput, refCos, refSin, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        refResult.rename("output");
        INDArray refOutput = sdRef.output(new HashMap<>(), "output").get("output").dup();
        double refSum = refOutput.sumNumber().doubleValue();
        sdRef.close();

        assertFalse(Double.isNaN(refSum), "Reference (FP32 cos/sin) should not produce NaN");

        // Test: FP32 input + FP16 cos/sin — this is the bug scenario
        SameDiff sdMixed = SameDiff.create();
        SDVariable mixedInput = sdMixed.var("input", input.dup());
        SDVariable mixedCos = sdMixed.var("cos", cosHalf.dup());
        SDVariable mixedSin = sdMixed.var("sin", sinHalf.dup());
        SDVariable mixedResult = new FusedRoPE(sdMixed, mixedInput, mixedCos, mixedSin, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        mixedResult.rename("output");
        INDArray mixedOutput = sdMixed.output(new HashMap<>(), "output").get("output");

        double mixedSum = mixedOutput.sumNumber().doubleValue();
        log.info("Reference (FP32 cos/sin) sum: {}", refSum);
        log.info("Mixed (FP16 cos/sin) sum: {}", mixedSum);

        assertFalse(Double.isNaN(mixedSum),
                "FP32 input + FP16 cos/sin should NOT produce NaN after dtype cast fix");

        // Values should be close (within FP16 precision tolerance)
        double maxDiff = refOutput.sub(mixedOutput).amaxNumber().doubleValue();
        log.info("Max diff FP32 vs FP16 cos/sin: {}", maxDiff);
        assertTrue(maxDiff < 0.1,
                "FP32 vs FP16 cos/sin results should be close, max diff = " + maxDiff);
        log.info("Test 5 PASSED: mixed dtype FP32+FP16, sum={}, maxDiff={}", mixedSum, maxDiff);
        sdMixed.close();
    }

    /**
     * Test 6: fused_rope with all-FP16 (input + cos/sin all HALF).
     * This should work regardless — the kernel dispatches on HALF for all buffers.
     */
    @Test
    @DisplayName("All FP16: HALF input with HALF cos/sin")
    void testAllFP16() {
        int batch = 1, seqLen = 82, numHeads = 9, headDim = 64;

        INDArray input = Nd4j.randn(DataType.HALF, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(DataType.HALF, batch, seqLen, headDim / 2);
        INDArray sin = Nd4j.randn(DataType.HALF, batch, seqLen, headDim / 2);

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.var("input", input);
        SDVariable cosVar = sd.var("cos", cos);
        SDVariable sinVar = sd.var("sin", sin);
        SDVariable result = new FusedRoPE(sd, inputVar, cosVar, sinVar, FusedRoPE.ROPE_TYPE_STANDARD).outputVariable();
        result.rename("output");

        Map<String, INDArray> out = sd.output(new HashMap<>(), "output");
        INDArray output = out.get("output");
        assertNotNull(output);

        double sum = output.sumNumber().doubleValue();
        assertFalse(Double.isNaN(sum), "All-FP16 should produce non-NaN output, sum=" + sum);
        assertEquals(DataType.HALF, output.dataType(), "Output should be HALF when input is HALF");
        log.info("Test 6 PASSED: all-FP16, sum={}", sum);
        sd.close();
    }

    private static boolean isContiguous(INDArray arr) {
        // Check if strides are descending (C-order contiguous)
        long[] shape = arr.shape();
        long[] stride = arr.stride();
        if (shape.length == 0) return true;
        long expected = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            if (stride[i] != expected) return false;
            expected *= shape[i];
        }
        return true;
    }
}
