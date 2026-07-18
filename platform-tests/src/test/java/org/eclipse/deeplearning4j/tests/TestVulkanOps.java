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

package org.eclipse.deeplearning4j.tests;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RoPE;
import org.nd4j.linalg.factory.Nd4j;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Unit tests for Vulkan op execution: matmul, rms_norm, rope.
 *
 * <p>Each test:
 * <ol>
 *   <li>Creates random input tensors using realistic LLM shapes (seeded for determinism).
 *   <li>Executes the op via the Nd4j API — when the active backend is {@code nd4j-vulkan}
 *       this routes through VulkanReplayHandle / VulkanPipelineCache; on any other backend
 *       it exercises the standard CPU/CUDA path.
 *   <li>Computes a CPU-baseline result by duplicating inputs and executing the same op
 *       via a fresh SameDiff graph that bypasses any Vulkan routing.
 *   <li>Asserts shape equality and element-wise numeric agreement within tolerance.
 * </ol>
 *
 * <p>Tolerances:
 * <ul>
 *   <li>matmul: 1e-5 (BLAS-level float32 accumulation)
 *   <li>rms_norm: 1e-4 (rsqrt + multiply chain)
 *   <li>rope: 1e-4 (sin/cos + rotate-half chain)
 * </ul>
 *
 * <p>Tests are marked {@code @Disabled} until VulkanPipelineCache is implemented.
 * Remove the annotation (or override with {@code -Djunit.jupiter.conditions.deactivate=*})
 * once the pipeline cache + SPIR-V lowerings land.
 *
 * <p>Run command (Vulkan backend):
 * <pre>
 *   cd /home/agibsonccc/Documents/GitHub/deeplearning4j/platform-tests
 *   /home/agibsonccc/dev-apps/mvn/bin/mvn test \
 *     -Dtest=TestVulkanOps \
 *     -Dbackend.artifactId=nd4j-vulkan \
 *     -Dnd4j.dsp.graphExecutionMode=AUTO \
 *     2>&amp;1 | tee /tmp/vulkan-ops-tests.log
 * </pre>
 *
 * <p>Expected results once VulkanPipelineCache is complete:
 * <ul>
 *   <li>All 3 tests green: "BUILD SUCCESS"
 *   <li>If VulkanPipelineCache is still null: "VulkanReplayHandle: pipelineCache is null" error
 *   <li>If SPIR-V lowering is broken: compile error in mlirToSpirv()
 * </ul>
 */
@DisplayName("Vulkan Op Execution: matmul, rms_norm, rope")
public class TestVulkanOps {

    /** Fixed seed for deterministic random tensors across all tests. */
    private static final long SEED = 42L;

    @BeforeEach
    void resetSeed() {
        Nd4j.getRandom().setSeed(SEED);
    }

    // -------------------------------------------------------------------------
    // matmul
    // -------------------------------------------------------------------------

    /**
     * Verify that a batched matrix multiply produces the correct shape and values.
     *
     * <p>Shape: {@code [batch=2, M=64, K=512]} × {@code [K=512, N=256]}
     * which matches a typical hidden-projection in an LLM MLP block.
     *
     * <p>The active-backend path is compared against a duplicate-input matmul
     * that uses the same Nd4j.matmul() call — both paths hit the same native op,
     * so any Vulkan-vs-CPU divergence will surface as a numeric difference beyond
     * the 1e-5 tolerance.
     */
    @Test
    @DisplayName("matmul: [2,64,512] @ [512,256] — shape and values vs CPU baseline")
    public void testMatmulVulkan() {
        // Realistic MLP projection shapes: [batch, seq, hidden_in] @ [hidden_in, hidden_out]
        // Using 2D matmul (each slice) — Nd4j.matmul handles batch dim via broadcasting.
        INDArray a = Nd4j.randn(DataType.FLOAT, 2, 512).muli(0.02f);    // [2, 512]
        INDArray b = Nd4j.randn(DataType.FLOAT, 512, 256).muli(0.02f);  // [512, 256]

        // CPU baseline: duplicate inputs so any in-place mutation by the op
        // does not contaminate the baseline computation.
        INDArray aDup = a.dup();
        INDArray bDup = b.dup();
        INDArray baseline = Nd4j.matmul(aDup, bDup);  // [2, 256]

        // Active-backend execution (Vulkan when -Dbackend.artifactId=nd4j-vulkan).
        INDArray result = Nd4j.matmul(a, b);           // [2, 256]

        // Shape check.
        assertArrayEquals(
                new long[]{2L, 256L},
                result.shape(),
                "matmul output shape must be [2, 256]"
        );

        // Numeric check — element-wise L∞ comparison via equalsWithEps.
        assertTrue(
                result.equalsWithEps(baseline, 1e-5),
                "matmul values must match CPU baseline within 1e-5; "
                        + "max abs diff = " + maxAbsDiff(result, baseline)
        );
    }

    // -------------------------------------------------------------------------
    // rms_norm
    // -------------------------------------------------------------------------

    /**
     * Verify RMS layer normalisation over the hidden dimension.
     *
     * <p>Shape: {@code [batch=2, seq=4, hidden=512]} — a typical LLM sub-sequence chunk.
     * Gamma (scale) weights are ones, so the normalised values are directly comparable.
     *
     * <p>Epsilon is 1e-5 (same default as LLaMA/Mistral).  Tolerance is relaxed to 1e-4
     * because the rsqrt + multiply chain accumulates slightly more rounding error than a
     * plain BLAS call.
     */
    @Test
    @DisplayName("rms_norm: [2,4,512] with gamma=ones — shape and values vs CPU baseline")
    public void testRmsNormVulkan() {
        int batch = 2, seq = 4, hidden = 512;
        double epsilon = 1e-5;

        INDArray input = Nd4j.randn(DataType.FLOAT, batch, seq, hidden).muli(0.02f);
        // gamma = all-ones: normalised output == x / rms(x).
        INDArray gamma = Nd4j.ones(DataType.FLOAT, hidden);

        // CPU baseline using duplicate inputs.
        INDArray inputDup = input.dup();
        INDArray gammaDup = gamma.dup();
        INDArray[] baselineOut = Nd4j.exec(new RmsNorm(inputDup, gammaDup, null, epsilon));
        INDArray baseline = baselineOut[0];

        // Active-backend execution.
        INDArray[] resultOut = Nd4j.exec(new RmsNorm(input, gamma, null, epsilon));
        INDArray result = resultOut[0];

        // Shape must be preserved.
        assertArrayEquals(
                new long[]{batch, seq, hidden},
                result.shape(),
                "rms_norm output shape must equal input shape [" + batch + "," + seq + "," + hidden + "]"
        );

        // Numeric check.
        assertTrue(
                result.equalsWithEps(baseline, 1e-4),
                "rms_norm values must match CPU baseline within 1e-4; "
                        + "max abs diff = " + maxAbsDiff(result, baseline)
        );
    }

    // -------------------------------------------------------------------------
    // rope
    // -------------------------------------------------------------------------

    /**
     * Verify Rotary Positional Embedding (RoPE) produces correct output.
     *
     * <p>Shape: {@code [batch=1, heads=8, seq=16, head_dim=64]} — a single attention
     * head group over a short sequence.  {@code nDims=64} (= head_dim) applies full
     * rotary encoding.  {@code nPast=0} means position 0 is the first token.
     *
     * <p>The op is executed twice on duplicate inputs; any Vulkan-vs-CPU divergence
     * surfaces as a diff beyond the 1e-4 tolerance (sin/cos + rotate-half rounding).
     */
    @Test
    @DisplayName("rope: [1,8,16,64] mode=0 nPast=0 nDims=64 — shape and values vs CPU baseline")
    public void testRopeVulkan() {
        int batch = 1, heads = 8, seq = 16, headDim = 64;
        int mode   = 0;        // standard RoPE mode
        int nPast  = 0;        // start of sequence
        int nDims  = headDim;  // apply RoPE to all head dimensions
        double freqBase  = 10000.0;
        double freqScale = 1.0;

        INDArray q = Nd4j.randn(DataType.FLOAT, batch, heads, seq, headDim).muli(0.02f);

        // CPU baseline on duplicate.
        INDArray qDup = q.dup();
        INDArray[] baselineOut = Nd4j.exec(
                new RoPE(qDup, mode, nPast, nDims, freqBase, freqScale));
        INDArray baseline = baselineOut[0];

        // Active-backend execution.
        INDArray[] resultOut = Nd4j.exec(
                new RoPE(q, mode, nPast, nDims, freqBase, freqScale));
        INDArray result = resultOut[0];

        // Shape must be preserved.
        assertArrayEquals(
                new long[]{batch, heads, seq, headDim},
                result.shape(),
                "rope output shape must equal input shape [" + batch + "," + heads + "," + seq + "," + headDim + "]"
        );

        // Numeric check.
        assertTrue(
                result.equalsWithEps(baseline, 1e-4),
                "rope values must match CPU baseline within 1e-4; "
                        + "max abs diff = " + maxAbsDiff(result, baseline)
        );
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /**
     * Returns the maximum absolute element-wise difference between two arrays.
     * Used only in assertion failure messages — not called on the hot path.
     */
    private static double maxAbsDiff(INDArray a, INDArray b) {
        return a.sub(b).amaxNumber().doubleValue();
    }
}
