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
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.BaseNd4jTestWithBackends;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Collections;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that view-producer ops (reshape, permute, etc.) that skip output zeroing
 * produce correct results in DSP slot execution.
 *
 * <p>Background: View-producing ops (reshape, permute, squeeze, expandDims) do not
 * allocate new memory — they share the underlying buffer with their input. In DSP
 * slot execution, these ops must correctly produce their output without zeroing
 * (which would destroy the view relationship). If zeroing logic is incorrectly
 * applied to view-producer ops, it will corrupt the shared buffer and produce
 * wrong results in downstream consumers.</p>
 *
 * <p>Key assertion: reshape→matmul→permute chains (like attention) must produce
 * identical output to standard SameDiff execution.</p>
 */
@Slf4j
@Tag("samediff")
public class TestDSPSlotExecViewProducerOrdering extends BaseNd4jTestWithBackends {

    private static final double TOL = 1e-4;

    @Override
    public char ordering() {
        return 'c';
    }

    @BeforeAll
    static void enableDspGlobally() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);
    }

    private void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
    }

    @AfterEach
    void cleanup() {
        Nd4j.getExecutioner().commit();
    }

    /**
     * Test 1: reshape → matmul chain.
     * The reshape is a view-producer; the matmul consumes the view.
     * Verifies the matmul reads correct data from the reshaped view.
     */
    @Test
    @DisplayName("View-producer ordering: reshape → matmul")
    public void testReshapeThenMatmul() {
        SameDiff sd = SameDiff.create();

        // Input: [1, 64] → reshape to [8, 8] → matmul [8,8]×[8,4] = [8,4]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64);
        SDVariable reshaped = sd.reshape("reshaped", x, 8, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", reshaped, w);

        enableDsp(sd);

        INDArray input = Nd4j.linspace(1, 64, 64, DataType.FLOAT).reshape(1, 64);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
        INDArray actual = dspResult.get("out").dup();

        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Reshape→matmul: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Reshape→matmul: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 2: matmul → permute chain.
     * The permute is a view-producer; verify its output is correct.
     */
    @Test
    @DisplayName("View-producer ordering: matmul → permute (transpose)")
    public void testMatmulThenPermute() {
        SameDiff sd = SameDiff.create();

        // [2, 4, 8] × [2, 8, 4] = [2, 4, 4] → permute to [2, 4, 4] (swap last 2 dims)
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 4, 8);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 2, 8, 4).muli(0.1));
        SDVariable mm = sd.linalg().matmul("mm", x, w);
        SDVariable out = sd.permute("out", mm, 0, 2, 1); // [2, 4, 4] → [2, 4, 4]

        enableDsp(sd);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 4, 8);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", xArr), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", xArr), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Matmul→permute: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Matmul→permute: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 3: Attention-like chain: reshape → matmul → permute → matmul.
     * This is the critical attention pattern: Q/K/V projections, head reshape,
     * attention scores, output projection.
     */
    @Test
    @DisplayName("View-producer ordering: attention-like chain (reshape→matmul→permute→matmul)")
    public void testAttentionLikeChain() {
        SameDiff sd = SameDiff.create();

        int batch = 1, seqLen = 4, heads = 2, headDim = 8;
        int hidden = heads * headDim; // 16

        // Input: [batch, seqLen, hidden]
        SDVariable input = sd.placeHolder("input", DataType.FLOAT, batch, seqLen, hidden);

        // Simulated Q, K, V projections (using elementwise for simplicity)
        SDVariable wQ = sd.constant("wQ", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
        SDVariable wK = sd.constant("wK", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));
        SDVariable wV = sd.constant("wV", Nd4j.randn(DataType.FLOAT, hidden, hidden).muli(0.02));

        // Project: [1, 4, 16] × [16, 16] → [1, 4, 16] (via reshape→matmul→reshape)
        SDVariable inFlat = sd.reshape("in_flat", input, batch * seqLen, hidden);
        SDVariable qFlat = sd.mmul("q_flat", inFlat, wQ);
        SDVariable kFlat = sd.mmul("k_flat", inFlat, wK);
        SDVariable vFlat = sd.mmul("v_flat", inFlat, wV);

        // Reshape to multi-head: [4, 16] → [4, 2, 8] → permute to [2, 4, 8]
        SDVariable qReshaped = sd.reshape("q_reshape", qFlat, batch * seqLen, heads, headDim);
        SDVariable qMH = sd.permute("q_mh", qReshaped, 1, 0, 2); // [heads, seqLen, headDim]

        SDVariable kReshaped = sd.reshape("k_reshape", kFlat, batch * seqLen, heads, headDim);
        SDVariable kMH = sd.permute("k_mh", kReshaped, 1, 2, 0); // [heads, headDim, seqLen]

        SDVariable vReshaped = sd.reshape("v_reshape", vFlat, batch * seqLen, heads, headDim);
        SDVariable vMH = sd.permute("v_mh", vReshaped, 1, 0, 2); // [heads, seqLen, headDim]

        // Attention scores: Q×K^T for each head (batched matmul)
        SDVariable scores = sd.linalg().matmul("scores", qMH, kMH); // [heads, seqLen, seqLen]

        // Scale
        float scale = (float) (1.0 / Math.sqrt(headDim));
        SDVariable scaled = scores.mul("scaled", scale);

        // Softmax over last dim
        SDVariable attn = sd.nn.softmax("attn", scaled, -1);

        // Output: attn × V
        SDVariable headOut = sd.linalg().matmul("head_out", attn, vMH); // [heads, seqLen, headDim]

        // Reshape back: [heads, seqLen, headDim] → [seqLen, heads, headDim] → [seqLen, hidden]
        SDVariable headOutPerm = sd.permute("head_out_perm", headOut, 1, 0, 2); // [seqLen, heads, headDim]
        SDVariable outFlat = sd.reshape("out_flat", headOutPerm, batch * seqLen, hidden);
        SDVariable out = sd.reshape("out", outFlat, batch, seqLen, hidden);

        enableDsp(sd);

        INDArray inputArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("input", inputArr), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("input", inputArr), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Attention chain: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Attention chain: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 4: expandDims → matmul chain.
     * expandDims is a view-producer that adds a dimension.
     */
    @Test
    @DisplayName("View-producer ordering: expandDims → matmul")
    public void testExpandDimsThenMatmul() {
        SameDiff sd = SameDiff.create();

        // [2, 8] → expandDims at axis 1 → [2, 1, 8] → matmul with [2, 8, 4] = [2, 1, 4]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 8);
        SDVariable expanded = sd.expandDims("expanded", x, 1); // [2, 1, 8]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 2, 8, 4).muli(0.1));
        SDVariable out = sd.linalg().matmul("out", expanded, w); // [2, 1, 4]

        enableDsp(sd);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 8);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", xArr), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", xArr), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("ExpandDims→matmul: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "ExpandDims→matmul: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 5: squeeze → matmul chain.
     * squeeze removes a dimension — verify downstream matmul reads correctly.
     */
    @Test
    @DisplayName("View-producer ordering: squeeze → matmul")
    public void testSqueezeThenMatmul() {
        SameDiff sd = SameDiff.create();

        // [2, 1, 8] → squeeze axis 1 → [2, 8] → matmul [2,8]×[8,4] = [2,4]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 2, 1, 8);
        SDVariable squeezed = sd.squeeze("squeezed", x, 1);
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1));
        SDVariable out = sd.mmul("out", squeezed, w);

        enableDsp(sd);

        INDArray xArr = Nd4j.randn(DataType.FLOAT, 2, 1, 8);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", xArr), "out");
        INDArray expected = stdResult.get("out").dup();

        // DSP execution
        Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", xArr), "out");
        INDArray actual = dspResult.get("out").dup();

        assertArrayEquals(expected.shape(), actual.shape(), "Shape mismatch");
        double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
        log.info("Squeeze→matmul: max diff = {}", maxDiff);
        assertTrue(maxDiff < TOL,
                "Squeeze→matmul: max diff " + maxDiff + " exceeds tolerance " + TOL);

        sd.close();
    }

    /**
     * Test 6: Frozen replay with view-producer chain.
     * Verifies view-producer ordering is correct across multiple frozen replay steps.
     */
    @Test
    @DisplayName("View-producer ordering: frozen replay with reshape→permute→matmul")
    public void testViewProducerFrozenReplay() {
        SameDiff sd = SameDiff.create();

        // [1, 32] → reshape [4, 8] → permute [8, 4] → matmul [8,4]×[4,2] = [8,2]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 32);
        SDVariable reshaped = sd.reshape("reshaped", x, 4, 8);
        SDVariable permuted = sd.permute("permuted", reshaped, 1, 0); // [8, 4]
        SDVariable w = sd.constant("w", Nd4j.randn(DataType.FLOAT, 4, 2).muli(0.1));
        SDVariable out = sd.mmul("out", permuted, w);

        enableDsp(sd);

        INDArray input = Nd4j.linspace(1, 32, 32, DataType.FLOAT).reshape(1, 32);

        // Standard reference
        Map<String, INDArray> stdResult = sd.output(Collections.singletonMap("x", input), "out");
        INDArray expected = stdResult.get("out").dup();

        // Warmup
        Map<String, INDArray> warmupResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
        INDArray warmupActual = warmupResult.get("out").dup();

        double warmupDiff = expected.sub(warmupActual).amaxNumber().doubleValue();
        log.info("Warmup: max diff = {}", warmupDiff);
        assertTrue(warmupDiff < TOL, "Warmup: max diff " + warmupDiff + " exceeds tolerance");

        // Freeze
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor dspExec = session.getDynamicShapePlanExecutor();
        assertNotNull(dspExec);
        dspExec.setShapesFrozen(true);

        // Frozen replay
        for (int step = 0; step < 5; step++) {
            Map<String, INDArray> dspResult = sd.outputDirect(Collections.singletonMap("x", input), "out");
            INDArray actual = dspResult.get("out").dup();

            double maxDiff = expected.sub(actual).amaxNumber().doubleValue();
            log.info("Frozen replay step {}: max diff = {}", step, maxDiff);
            assertTrue(maxDiff < TOL,
                    "Frozen replay step " + step + ": max diff " + maxDiff + " exceeds tolerance");
        }

        sd.close();
    }
}
