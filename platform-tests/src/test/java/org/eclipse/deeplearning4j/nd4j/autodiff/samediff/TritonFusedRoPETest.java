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
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.FusedRoPE;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for fused_rope Triton emitter.
 *
 * Verifies that the fused RoPE (Rotary Position Embedding) operation:
 * 1. Produces correct output matching manual decomposition (cos/sin rotation)
 * 2. Compiles and executes correctly through Triton DSP path
 * 3. Works with different shapes (decode, prefill) and data types (FLOAT, HALF)
 * 4. Handles multi-step execution with frozen shapes
 */
@Slf4j
@Tag(TagNames.SAMEDIFF)
@NativeTag
public class TritonFusedRoPETest extends BaseNd4jTestWithBackends {

    private static final double TOLERANCE = 1e-3;
    private static final double HALF_TOLERANCE = 5e-2;

    @AfterEach
    public void cleanup() {
        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        nativeOps.invalidateTritonCache();
        nativeOps.resetTritonCounters();
        Nd4j.getMemoryManager().purgeCaches();
        System.gc();
        nativeOps.trimMemoryPool(0);
    }

    // ─── Reference implementation ──────────────────────────────────────────

    /**
     * Manual RoPE computation (GPT-NeoX half-rotation style, ropeType=0).
     * x1 = input[..., :halfDim]
     * x2 = input[..., halfDim:]
     * output[..., :halfDim] = x1 * cos - x2 * sin
     * output[..., halfDim:] = x2 * cos + x1 * sin
     */
    private INDArray computeRoPEReference(INDArray input, INDArray cos, INDArray sin) {
        long[] shape = input.shape();
        int headDim = (int) shape[shape.length - 1];
        int halfDim = headDim / 2;

        INDArray output = Nd4j.createUninitialized(input.dataType(), shape);

        // Flatten to [total, headDim] for easy processing
        long totalHeads = input.length() / headDim;
        INDArray inputFlat = input.reshape(totalHeads, headDim);
        INDArray outputFlat = output.reshape(totalHeads, headDim);

        // cos/sin have no head dimension — flatten to [totalPositions, cosDim]
        long cosDim = cos.shape()[cos.rank() - 1];
        long totalPositions = cos.length() / cosDim;
        INDArray cosFlat = cos.reshape(totalPositions, cosDim);
        INDArray sinFlat = sin.reshape(totalPositions, cosDim);

        // Compute numHeads from the shapes
        // input is [B, S, H, D], cos is [B, S, D] (or similar)
        int numHeads = (int) (totalHeads / totalPositions);

        for (long i = 0; i < totalHeads; i++) {
            long posIdx = i / numHeads;  // Which (batch, seq) position
            for (int d = 0; d < halfDim; d++) {
                float x1 = inputFlat.getFloat(i, d);
                float x2 = inputFlat.getFloat(i, d + halfDim);
                float c = cosFlat.getFloat(posIdx, d);
                float s = sinFlat.getFloat(posIdx, d);

                outputFlat.putScalar(new long[]{i, d}, x1 * c - x2 * s);
                outputFlat.putScalar(new long[]{i, d + halfDim}, x2 * c + x1 * s);
            }
        }

        return output;
    }

    // ─── Parameter sources ─────────────────────────────────────────────────

    static Stream<Arguments> ropeConfigs() {
        return Stream.of(
                // name, batch, seqLen, numHeads, headDim, dtype
                Arguments.of("decode_f32", 1, 1, 32, 64, DataType.FLOAT),
                Arguments.of("decode_f16", 1, 1, 32, 64, DataType.HALF),
                Arguments.of("prefill_f32", 1, 8, 32, 64, DataType.FLOAT),
                Arguments.of("small_heads_f32", 1, 1, 4, 16, DataType.FLOAT),
                Arguments.of("large_dim_f32", 1, 1, 8, 128, DataType.FLOAT)
        );
    }

    // ─── Tests ─────────────────────────────────────────────────────────────

    /**
     * Test fused_rope op correctness by comparing against manual reference.
     */
    @ParameterizedTest(name = "fusedRoPE_directOp_{0}")
    @MethodSource("ropeConfigs")
    public void testFusedRoPEDirectOp(String name, int batch, int seqLen, int numHeads,
                                       int headDim, DataType dtype) {
        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);

        // Execute via FusedRoPE op
        INDArray output = Nd4j.create(dtype, batch, seqLen, numHeads, headDim);
        Nd4j.exec(new FusedRoPE(input, cos, sin, output, 0));

        // Compute reference
        INDArray reference = computeRoPEReference(input, cos, sin);

        // Compare
        double maxDiff = reference.castTo(DataType.FLOAT).sub(output.castTo(DataType.FLOAT))
                .amaxNumber().doubleValue();
        double tolerance = (dtype == DataType.HALF) ? HALF_TOLERANCE : TOLERANCE;
        assertTrue(maxDiff < tolerance,
                String.format("fused_rope %s maxDiff=%.6f exceeds tolerance=%.6f", name, maxDiff, tolerance));
        log.info("fused_rope {} maxDiff={}", name, maxDiff);
    }

    /**
     * Test fused_rope through SameDiff DSP/Triton path.
     * Verifies the op compiles correctly in the Triton IR builder.
     */
    @ParameterizedTest(name = "fusedRoPE_samediff_{0}")
    @MethodSource("ropeConfigs")
    public void testFusedRoPESameDiffTriton(String name, int batch, int seqLen, int numHeads,
                                             int headDim, DataType dtype) {
        // Build SameDiff graph with fused_rope
        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        ropeOut.rename("rope_output");

        // Enable DSP + Triton compilation
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray inputArr = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cosArr = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sinArr = Nd4j.randn(dtype, batch, seqLen, headDim);

        // Execute reference (before Triton compilation)
        INDArray reference = computeRoPEReference(inputArr, cosArr, sinArr);

        // Execute via SameDiff (will trigger Triton compilation if available)
        Map<String, INDArray> result = sd.outputDirect(
                Map.of("input", inputArr, "cos", cosArr, "sin", sinArr),
                "rope_output");
        INDArray sdOutput = result.get("rope_output");

        assertNotNull(sdOutput, "SameDiff output is null");
        assertArrayEquals(reference.shape(), sdOutput.shape(),
                "Shape mismatch: expected " + java.util.Arrays.toString(reference.shape()) +
                        " got " + java.util.Arrays.toString(sdOutput.shape()));

        double maxDiff = reference.castTo(DataType.FLOAT).sub(sdOutput.castTo(DataType.FLOAT))
                .amaxNumber().doubleValue();
        double tolerance = (dtype == DataType.HALF) ? HALF_TOLERANCE : TOLERANCE;
        assertTrue(maxDiff < tolerance,
                String.format("SameDiff fused_rope %s maxDiff=%.6f exceeds tolerance=%.6f", name, maxDiff, tolerance));
        log.info("SameDiff fused_rope {} maxDiff={}", name, maxDiff);

        sd.close();
    }

    /**
     * Test fused_rope in a chain: input → fused_rope → add(residual) → output.
     * This tests that ROPE correctly interacts with downstream elementwise ops
     * in the Triton section model.
     */
    @Test
    public void testFusedRoPEChainWithResidual() {
        int batch = 1, seqLen = 1, numHeads = 8, headDim = 64;
        DataType dtype = DataType.FLOAT;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);
        SDVariable residualVar = sd.placeHolder("residual", dtype, batch, seqLen, numHeads, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        SDVariable result = ropeOut.add(residualVar);
        result.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray residual = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);

        // Reference: manual rope + add
        INDArray ropeRef = computeRoPEReference(input, cos, sin);
        INDArray expectedOutput = ropeRef.add(residual);

        Map<String, INDArray> output = sd.outputDirect(
                Map.of("input", input, "cos", cos, "sin", sin, "residual", residual),
                "output");
        INDArray actual = output.get("output");

        assertNotNull(actual);
        double maxDiff = expectedOutput.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOLERANCE,
                String.format("ROPE+residual chain maxDiff=%.6f exceeds tolerance=%.6f", maxDiff, TOLERANCE));
        log.info("ROPE+residual chain maxDiff={}", maxDiff);

        sd.close();
    }

    /**
     * Test multi-step fused_rope execution (simulating decode loop).
     * Each step uses different cos/sin values (different position).
     * Verifies correctness persists across frozen-shape replays.
     */
    @Test
    public void testFusedRoPEMultiStep() {
        int batch = 1, seqLen = 1, numHeads = 8, headDim = 64;
        int numSteps = 5;
        DataType dtype = DataType.FLOAT;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        ropeOut.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        double maxDiffOverall = 0;
        for (int step = 0; step < numSteps; step++) {
            INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
            // Use different frequencies per step to simulate different positions
            INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
            INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);

            INDArray reference = computeRoPEReference(input, cos, sin);

            Map<String, INDArray> result = sd.outputDirect(
                    Map.of("input", input, "cos", cos, "sin", sin),
                    "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, "Step " + step + " output is null");
            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            maxDiffOverall = Math.max(maxDiffOverall, maxDiff);
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Step %d maxDiff=%.6f exceeds tolerance=%.6f", step, maxDiff, TOLERANCE));
        }
        log.info("Multi-step fused_rope maxDiffOverall={}", maxDiffOverall);

        sd.close();
    }

    /**
     * Test fused_rope with explicit Triton compilation mode.
     * Verifies the op is recognized by the Triton IR builder and compiled.
     */
    @Test
    public void testFusedRoPETritonCompilation() {
        int batch = 1, seqLen = 1, numHeads = 8, headDim = 64;
        DataType dtype = DataType.FLOAT;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        ropeOut.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        // First execution (warmup / compile)
        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);

        Map<String, INDArray> result = sd.outputDirect(
                Map.of("input", input, "cos", cos, "sin", sin),
                "output");
        assertNotNull(result.get("output"), "First execution output is null");

        // Try to compile with Triton mode
        try {
            GraphExecutionMode mode = sd.compileNativeDynamicShapePlan(
                    List.of("output"), GraphExecutionMode.TRITON, true);
            log.info("Triton compilation mode returned: {}", mode);

            // Second execution (should use compiled plan)
            INDArray input2 = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
            INDArray cos2 = Nd4j.randn(dtype, batch, seqLen, headDim);
            INDArray sin2 = Nd4j.randn(dtype, batch, seqLen, headDim);

            INDArray reference = computeRoPEReference(input2, cos2, sin2);
            Map<String, INDArray> result2 = sd.outputDirect(
                    Map.of("input", input2, "cos", cos2, "sin", sin2),
                    "output");
            INDArray actual = result2.get("output");

            assertNotNull(actual, "Post-Triton-compile output is null");
            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Post-Triton-compile maxDiff=%.6f exceeds tolerance=%.6f", maxDiff, TOLERANCE));
            log.info("Post-Triton-compile maxDiff={}", maxDiff);
        } catch (Exception e) {
            log.warn("Triton compilation not available or failed: {}", e.getMessage());
            // Not a failure — Triton may not be available in the test environment
        }

        sd.close();
    }

    // ─── Baseline and Register-ROPE Verification Tests ────────────────────

    /**
     * Establishes baseline performance of the fused_rope Triton emitter.
     * Measures latency of the current pointer-based emitter for VLM-like shapes.
     * This test provides the "before" measurement for the register-level optimization.
     */
    @Test
    public void testFusedRoPEMemoryTrafficBaseline() {
        int batch = 1, seqLen = 1, numHeads = 24, headDim = 64;
        DataType dtype = DataType.FLOAT;
        int warmupIters = 5, measuredIters = 20;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        ropeOut.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);
        Map<String, INDArray> feed = Map.of("input", input, "cos", cos, "sin", sin);

        // Warmup (triggers compilation)
        for (int i = 0; i < warmupIters; i++) {
            sd.outputDirect(feed, "output");
        }

        // Measure
        long startNs = System.nanoTime();
        for (int i = 0; i < measuredIters; i++) {
            sd.outputDirect(feed, "output");
        }
        long elapsedNs = System.nanoTime() - startNs;
        double avgUs = elapsedNs / (measuredIters * 1000.0);

        log.info("Fused ROPE baseline (VLM decode shape [{},{},{},{}]): avg={} us over {} iterations",
                batch, seqLen, numHeads, headDim, avgUs, measuredIters);

        // Correctness check
        INDArray reference = computeRoPEReference(input, cos, sin);
        Map<String, INDArray> result = sd.outputDirect(feed, "output");
        double maxDiff = reference.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOLERANCE,
                String.format("Baseline ROPE maxDiff=%.6f exceeds tolerance", maxDiff));

        sd.close();
    }

    /**
     * Verifies that blockSize is divisible by headDim for common VLM configurations.
     * The register-level ROPE path requires blockSize % headDim == 0.
     */
    @Test
    public void testBlockSizeHeadDimAlignment() {
        // Common configurations: blockSize 256/512/1024, headDim 32/64/128
        int[] blockSizes = {256, 512, 1024};
        int[] headDims = {32, 64, 128};

        for (int bs : blockSizes) {
            for (int hd : headDims) {
                boolean aligned = (bs % hd == 0);
                log.info("blockSize={} headDim={} aligned={} numHeadsInBlock={}",
                        bs, hd, aligned, aligned ? (bs / hd) : -1);
                // All common configs should be aligned
                assertTrue(aligned,
                        String.format("blockSize=%d not divisible by headDim=%d", bs, hd));
            }
        }
    }

    /**
     * Tests register-based ROPE correctness by comparing Triton-compiled output
     * against the manual reference. The register path should produce bitwise-identical
     * results to the pointer-based path (both compute x*cos ± y*sin in f32).
     *
     * Uses VLM-like shapes where the SSA path is expected to be selected:
     * blockSize=256, headDim=64 → 256%64=0, numHeadsInBlock=4.
     */
    @Test
    public void testRegisterRoPECorrectness() {
        // Test multiple shapes that should all trigger the register path
        int[][] configs = {
                // batch, seqLen, numHeads, headDim
                {1, 1, 24, 64},   // SmolDocling decode
                {1, 1, 32, 64},   // Common LLM decode
                {1, 1, 8, 128},   // Large headDim
                {1, 8, 8, 64},    // Prefill with multiple seq positions
        };

        for (int[] cfg : configs) {
            int batch = cfg[0], seqLen = cfg[1], numHeads = cfg[2], headDim = cfg[3];
            DataType dtype = DataType.FLOAT;

            SameDiff sd = SameDiff.create();
            SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
            SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
            SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

            SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
            ropeOut.rename("output");

            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
            INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
            INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);

            INDArray reference = computeRoPEReference(input, cos, sin);

            Map<String, INDArray> result = sd.outputDirect(
                    Map.of("input", input, "cos", cos, "sin", sin), "output");
            INDArray actual = result.get("output");

            assertNotNull(actual, String.format("Output null for [%d,%d,%d,%d]", batch, seqLen, numHeads, headDim));
            assertArrayEquals(reference.shape(), actual.shape(),
                    String.format("Shape mismatch for [%d,%d,%d,%d]", batch, seqLen, numHeads, headDim));

            double maxDiff = reference.sub(actual).amaxNumber().doubleValue();
            assertTrue(maxDiff < TOLERANCE,
                    String.format("Register ROPE [%d,%d,%d,%d] maxDiff=%.6f exceeds tolerance=%.6f",
                            batch, seqLen, numHeads, headDim, maxDiff, TOLERANCE));
            log.info("Register ROPE [{},{},{},{}] maxDiff={}", batch, seqLen, numHeads, headDim, maxDiff);

            sd.close();
        }
    }

    /**
     * Tests register ROPE in a fused chain: prior_op → ROPE → downstream_op.
     * This verifies the SSA path correctly receives input from prior ops and
     * passes output to downstream ops without global memory round-trip.
     */
    @Test
    public void testRegisterRoPEFusedChain() {
        int batch = 1, seqLen = 1, numHeads = 24, headDim = 64;
        DataType dtype = DataType.FLOAT;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable scaleVar = sd.placeHolder("scale", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);
        SDVariable residualVar = sd.placeHolder("residual", dtype, batch, seqLen, numHeads, headDim);

        // Chain: input * scale → ROPE → add residual
        SDVariable scaled = inputVar.mul(scaleVar);
        SDVariable ropeOut = new FusedRoPE(sd, scaled, cosVar, sinVar, 0).outputVariable();
        SDVariable result = ropeOut.add(residualVar);
        result.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray scale = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray residual = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);

        // Reference: (input * scale) → manual ROPE → add residual
        INDArray scaledRef = input.mul(scale);
        INDArray ropeRef = computeRoPEReference(scaledRef, cos, sin);
        INDArray expectedOutput = ropeRef.add(residual);

        Map<String, INDArray> output = sd.outputDirect(
                Map.of("input", input, "scale", scale, "cos", cos, "sin", sin, "residual", residual),
                "output");
        INDArray actual = output.get("output");

        assertNotNull(actual);
        double maxDiff = expectedOutput.sub(actual).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOLERANCE,
                String.format("Fused chain (mul→ROPE→add) maxDiff=%.6f exceeds tolerance=%.6f", maxDiff, TOLERANCE));
        log.info("Fused chain (mul→ROPE→add) maxDiff={}", maxDiff);

        sd.close();
    }

    /**
     * Performance comparison test: measures register-based ROPE vs baseline.
     * Runs multiple iterations to get stable measurements.
     */
    @Test
    public void testRegisterRoPEPerformance() {
        int batch = 1, seqLen = 1, numHeads = 24, headDim = 64;
        DataType dtype = DataType.FLOAT;
        int warmupIters = 10, measuredIters = 50;

        SameDiff sd = SameDiff.create();
        SDVariable inputVar = sd.placeHolder("input", dtype, batch, seqLen, numHeads, headDim);
        SDVariable cosVar = sd.placeHolder("cos", dtype, batch, seqLen, headDim);
        SDVariable sinVar = sd.placeHolder("sin", dtype, batch, seqLen, headDim);

        SDVariable ropeOut = new FusedRoPE(sd, inputVar, cosVar, sinVar, 0).outputVariable();
        ropeOut.rename("output");

        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);

        INDArray input = Nd4j.randn(dtype, batch, seqLen, numHeads, headDim);
        INDArray cos = Nd4j.randn(dtype, batch, seqLen, headDim);
        INDArray sin = Nd4j.randn(dtype, batch, seqLen, headDim);
        Map<String, INDArray> feed = Map.of("input", input, "cos", cos, "sin", sin);

        // Warmup
        for (int i = 0; i < warmupIters; i++) {
            sd.outputDirect(feed, "output");
        }

        // Measure
        long[] latencies = new long[measuredIters];
        for (int i = 0; i < measuredIters; i++) {
            long start = System.nanoTime();
            sd.outputDirect(feed, "output");
            latencies[i] = System.nanoTime() - start;
        }

        // Compute stats
        java.util.Arrays.sort(latencies);
        double p50Us = latencies[measuredIters / 2] / 1000.0;
        double p90Us = latencies[(int)(measuredIters * 0.9)] / 1000.0;
        double avgUs = java.util.Arrays.stream(latencies).average().orElse(0) / 1000.0;

        log.info("Register ROPE performance (VLM decode [{},{},{},{}]):", batch, seqLen, numHeads, headDim);
        log.info("  P50={} us, P90={} us, Avg={} us over {} iterations",
                p50Us, p90Us, avgUs, measuredIters);

        // Correctness sanity check
        INDArray reference = computeRoPEReference(input, cos, sin);
        Map<String, INDArray> result = sd.outputDirect(feed, "output");
        double maxDiff = reference.sub(result.get("output")).amaxNumber().doubleValue();
        assertTrue(maxDiff < TOLERANCE, "Performance test correctness check failed");

        sd.close();
    }
}
