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
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for Triton dual-buffer KV cache integration.
 *
 * The Triton fused attention kernel supports dual-buffer KV:
 * - Past buffer: 4D BHSD [B, H, pastSeq, D] — the static KV cache
 * - Current buffer: 3D BSHD [B, seqKVCur, H*D] — current step's K/V projections
 *
 * The kernel loads from both buffers with separate masks, splitting at position pastSeq.
 * emitPresentKvWrite() writes current K/V into the output buffer at offset pastSeq.
 */
@Slf4j
@DisplayName("DualBufferKvCacheTest")
public class DualBufferKvCacheTest {

    /**
     * Test that OnnxMultiHeadAttention with 6 inputs (Q, K, V, bias, pastKey, pastValue)
     * correctly computes attention and produces present_key/value outputs with past+current
     * KV concatenated.
     */
    @Test
    @DisplayName("testDualBufferKvActivation")
    public void testDualBufferKvActivation() {
        int batch = 1;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;
        int pastSeq = 3;
        int curSeq = 1; // decode step: 1 new token

        // Q: 3D BSHD [batch, curSeq, hidden] — query for current step
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, curSeq, hidden);

        // K, V: 3D BSHD [batch, curSeq, hidden] — current step's K/V projections
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, curSeq, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, curSeq, hidden);

        // Bias: attention bias [1, 1, curSeq, pastSeq + curSeq]
        int totalSeq = pastSeq + curSeq;
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeq);

        // Past key/value: 4D BHSD [batch, numHeads, pastSeq, headDim]
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Output buffers
        INDArray attnOutput = Nd4j.create(DataType.FLOAT, batch, curSeq, hidden);
        INDArray presentKey = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
        INDArray presentValue = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);

        // Execute with 6 inputs: Q, K, V, bias, pastKey, pastValue
        DynamicCustomOp op = DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, bias, pastKey, pastValue)
                .addOutputs(attnOutput, presentKey, presentValue)
                .addIntegerArguments(numHeads, 0)  // numHeads, useCausalMask=false
                .addFloatingPointArguments(0.0)    // scale=0 (auto)
                .build();

        Nd4j.exec(op);

        // Verify attention output shape
        assertArrayEquals(new long[]{batch, curSeq, hidden}, attnOutput.shape(),
                "Attention output should be [batch, curSeq, hidden]");

        // Verify present key/value shape: [batch, numHeads, totalSeq, headDim]
        assertArrayEquals(new long[]{batch, numHeads, totalSeq, headDim}, presentKey.shape(),
                "Present key should contain past + current KV");
        assertArrayEquals(new long[]{batch, numHeads, totalSeq, headDim}, presentValue.shape(),
                "Present value should contain past + current KV");

        // Verify attention output is not all zeros
        double absSum = attnOutput.ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "Attention output should not be all zeros");

        // Verify present key contains past KV data in [0:pastSeq] positions
        INDArray presentKeyPast = presentKey.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all());
        double pastKeyDiff = presentKeyPast.sub(pastKey).amaxNumber().doubleValue();
        log.info("Past key diff in present output: {}", pastKeyDiff);
        assertTrue(pastKeyDiff < 1e-5,
                "Present key [0:pastSeq] should match past key input. Diff=" + pastKeyDiff);

        // Verify present key has new entry at position pastSeq
        INDArray presentKeyNew = presentKey.get(
                NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(pastSeq), NDArrayIndex.all());
        double newKeyAbsSum = presentKeyNew.ameanNumber().doubleValue();
        assertTrue(newKeyAbsSum > 1e-6,
                "Present key at position pastSeq should have data from current key");
    }

    /**
     * Test multi-step decode simulation with static KV buffer.
     * Each step feeds the static buffer as pastKey/pastValue and the current K/V
     * as 3D BSHD. Verifies that the present_key output accumulates entries correctly.
     */
    @Test
    @DisplayName("testDualBufferKvWithStaticCache")
    public void testDualBufferKvWithStaticCache() {
        int batch = 1;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;
        int maxKvLen = 10;
        int numSteps = 5;

        // Pre-allocate static KV buffers with random data at position 0 (simulating prefill)
        INDArray staticKeyBuf = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);
        INDArray staticValBuf = Nd4j.zeros(DataType.FLOAT, batch, numHeads, maxKvLen, headDim);

        // Initialize position 0 from a "prefill step"
        INDArray initKey = Nd4j.randn(DataType.FLOAT, batch, numHeads, 1, headDim);
        INDArray initVal = Nd4j.randn(DataType.FLOAT, batch, numHeads, 1, headDim);
        staticKeyBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(0), NDArrayIndex.all()).assign(initKey.reshape(batch, numHeads, headDim));
        staticValBuf.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.point(0), NDArrayIndex.all()).assign(initVal.reshape(batch, numHeads, headDim));

        int cachePos = 1; // Start at 1 since position 0 was filled by "prefill"

        for (int step = 0; step < numSteps; step++) {
            // Current step's Q, K, V
            INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray key = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);
            INDArray value = Nd4j.randn(DataType.FLOAT, batch, 1, hidden);

            int totalSeq = cachePos + 1;

            // Output buffers
            INDArray attnOutput = Nd4j.create(DataType.FLOAT, batch, 1, hidden);
            INDArray presentKey = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);
            INDArray presentValue = Nd4j.create(DataType.FLOAT, batch, numHeads, totalSeq, headDim);

            // Build and execute op
            DynamicCustomOp.DynamicCustomOpsBuilder builder = DynamicCustomOp.builder("onnx_multi_head_attention")
                    .addOutputs(attnOutput, presentKey, presentValue)
                    .addIntegerArguments(numHeads, 0)
                    .addFloatingPointArguments(0.0);

            if (cachePos > 0) {
                // Past KV: view of static buffer [0:cachePos]
                INDArray pastKey = staticKeyBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                INDArray pastValue = staticValBuf.get(
                        NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(0, cachePos), NDArrayIndex.all());
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeq);
                builder.addInputs(query, key, value, bias, pastKey, pastValue);
            } else {
                // First step: no past KV — basic 3-input form
                builder.addInputs(query, key, value);
            }

            DynamicCustomOp op = builder.build();
            Nd4j.exec(op);

            // Verify attention output
            assertArrayEquals(new long[]{batch, 1, hidden}, attnOutput.shape(),
                    "Step " + step + ": attention output shape");
            assertTrue(attnOutput.ameanNumber().doubleValue() > 1e-6,
                    "Step " + step + ": attention output should not be all zeros");

            // Scatter new KV entry into static buffer (simulating Java scatter path).
            // presentKey has shape [batch, numHeads, totalSeq, headDim].
            // The new entry is at the last position (totalSeq - 1).
            INDArray newKeyEntry = presentKey.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            INDArray targetKeySlice = staticKeyBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all());
            targetKeySlice.assign(newKeyEntry);

            INDArray newValEntry = presentValue.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            INDArray targetValSlice = staticValBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all());
            targetValSlice.assign(newValEntry);

            cachePos++;
            log.info("Step {}: cachePos={}, presentKey shape={}", step, cachePos,
                    Arrays.toString(presentKey.shape()));
        }

        // After all steps, static buffer should have data in [0:cachePos)
        // Position 0 was prefilled, positions 1..numSteps were filled by decode steps
        int expectedFilledPositions = 1 + numSteps; // prefill + decode steps
        for (int pos = 0; pos < expectedFilledPositions; pos++) {
            INDArray keySlice = staticKeyBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all());
            double absSum = keySlice.ameanNumber().doubleValue();
            assertTrue(absSum > 1e-6,
                    "Static key buffer at position " + pos + " should have data");
        }

        // Unused positions should remain zero
        for (int pos = expectedFilledPositions; pos < maxKvLen; pos++) {
            INDArray keySlice = staticKeyBuf.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(pos), NDArrayIndex.all());
            double absSum = keySlice.ameanNumber().doubleValue();
            assertEquals(0.0, absSum, 1e-8,
                    "Static key buffer at unused position " + pos + " should be zero");
        }

        log.info("Multi-step decode simulation passed: {} steps, {} cached KV entries",
                numSteps, cachePos);
    }

    /**
     * Test that SameDiff graph with OnnxMultiHeadAttention (6 inputs) can be compiled
     * and executed through the standard SameDiff execution path.
     */
    @Test
    @DisplayName("testDualBufferKvInSameDiffGraph")
    public void testDualBufferKvInSameDiffGraph() {
        int batch = 1;
        int numHeads = 2;
        int headDim = 8;
        int hidden = numHeads * headDim;
        int pastSeq = 3;

        SameDiff sd = SameDiff.create();

        // Inputs
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, batch, 1, hidden);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, batch, 1, hidden);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, batch, 1, hidden);
        SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, 1, 1, 1, pastSeq + 1);
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Create MHA op via SameDiff
        OnnxMultiHeadAttention attnOp = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                numHeads, 0.0, false);
        SDVariable[] mhaOutputs = attnOp.outputVariables();
        SDVariable attnOut = mhaOutputs[0];
        SDVariable presentKey = mhaOutputs[1];
        SDVariable presentValue = mhaOutputs[2];

        // Feed data
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", Nd4j.randn(DataType.FLOAT, batch, 1, hidden));
        ph.put("key", Nd4j.randn(DataType.FLOAT, batch, 1, hidden));
        ph.put("value", Nd4j.randn(DataType.FLOAT, batch, 1, hidden));
        ph.put("bias", Nd4j.zeros(DataType.FLOAT, 1, 1, 1, pastSeq + 1));
        ph.put("past_key", Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim));
        ph.put("past_value", Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim));

        // Execute
        Map<String, INDArray> result = sd.output(ph,
                attnOut.name(), presentKey.name(), presentValue.name());

        // Verify outputs
        INDArray attnResult = result.get(attnOut.name());
        INDArray pkResult = result.get(presentKey.name());
        INDArray pvResult = result.get(presentValue.name());

        assertNotNull(attnResult, "Attention output should not be null");
        assertNotNull(pkResult, "Present key should not be null");
        assertNotNull(pvResult, "Present value should not be null");

        assertArrayEquals(new long[]{batch, 1, hidden}, attnResult.shape());
        // Present K/V should be [batch, numHeads, pastSeq+1, headDim]
        assertArrayEquals(new long[]{batch, numHeads, pastSeq + 1, headDim}, pkResult.shape(),
                "Present key should have pastSeq+1 entries");

        assertTrue(attnResult.ameanNumber().doubleValue() > 1e-6,
                "Attention output should not be all zeros");

        sd.close();
    }

    /**
     * Test that bias fed via placeholder override produces correct output.
     * Simulates the VLM decode pattern where attention bias is overridden
     * to bypass the attn_mask_reformat subgraph.
     */
    @Test
    @DisplayName("testDualBufferKvWithBiasOverride")
    public void testDualBufferKvWithBiasOverride() {
        int batch = 1, numHeads = 4, headDim = 8, hidden = numHeads * headDim;
        int pastSeq = 5;

        SameDiff sd = SameDiff.create();

        // MHA inputs
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, batch, 1, hidden);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, batch, 1, hidden);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, batch, 1, hidden);
        SDVariable bias = sd.placeHolder("bias", DataType.FLOAT, batch, 1, 1, pastSeq + 1);
        SDVariable pastKey = sd.placeHolder("pastKey", DataType.FLOAT, batch, numHeads, pastSeq, headDim);
        SDVariable pastValue = sd.placeHolder("pastValue", DataType.FLOAT, batch, numHeads, pastSeq, headDim);

        // Build MHA op with 6 inputs (including bias and past KV)
        OnnxMultiHeadAttention attnOp = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                numHeads, 0.0, false);
        SDVariable[] outputs = attnOp.outputVariables();
        SDVariable attnOutput = outputs[0];
        SDVariable presentKey = outputs[1];
        SDVariable presentValue = outputs[2];

        // Create arrays
        INDArray qArr = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        INDArray kArr = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        INDArray vArr = Nd4j.randn(DataType.FLOAT, batch, 1, hidden).mul(0.1);
        INDArray pkArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim).mul(0.1);
        INDArray pvArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, pastSeq, headDim).mul(0.1);

        // Create attention bias — all zeros (allow all positions)
        INDArray biasArr = Nd4j.zeros(DataType.FLOAT, batch, 1, 1, pastSeq + 1);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", qArr);
        ph.put("key", kArr);
        ph.put("value", vArr);
        ph.put("bias", biasArr);
        ph.put("pastKey", pkArr);
        ph.put("pastValue", pvArr);

        Map<String, INDArray> result = sd.output(ph, attnOutput.name(), presentKey.name(), presentValue.name());

        INDArray attnResult = result.get(attnOutput.name());
        assertNotNull(attnResult, "Attention output should not be null");
        assertArrayEquals(new long[]{batch, 1, hidden}, attnResult.shape(),
                "Attention output shape should be [B, 1, H*D]");
        assertTrue(attnResult.ameanNumber().doubleValue() > 1e-6,
                "Attention output should not be all zeros (bias allows all positions)");

        // Now test with a large negative bias that masks ALL past positions
        float maskFill = -3.4028235e+38f;
        INDArray biasAllMasked = Nd4j.valueArrayOf(new long[]{batch, 1, 1, pastSeq + 1}, maskFill, DataType.FLOAT);
        // Only allow the new token position (last position)
        biasAllMasked.putScalar(new long[]{0, 0, 0, pastSeq}, 0.0f);
        ph.put("bias", biasAllMasked);

        Map<String, INDArray> result2 = sd.output(ph, attnOutput.name());
        INDArray attnResult2 = result2.get(attnOutput.name());
        assertNotNull(attnResult2, "Attention output with masked bias should not be null");
        // Output should still be valid (not NaN or Inf)
        assertFalse(Double.isNaN(attnResult2.sumNumber().doubleValue()),
                "Attention output with masked bias should not be NaN");
        assertFalse(Double.isInfinite(attnResult2.sumNumber().doubleValue()),
                "Attention output with masked bias should not be Inf");

        sd.close();
    }
}
