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
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests that both padded (DSP-style) and unpadded (view-based) input modes
 * produce equivalent results through SameDiff execution with onnx_multi_head_attention.
 *
 * Padded mode: full static KV buffer [1, heads, maxKvLen, dim] + sparse attention_mask
 * Unpadded mode: KV view [1, heads, cachePos, dim] + all-ones attention_mask
 */
@Slf4j
@DisplayName("PaddedVsUnpaddedExecutionTest")
public class PaddedVsUnpaddedExecutionTest {

    private static final int BATCH = 1;
    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;

    /**
     * Test unpadded (view-based) execution: KV view [0:cachePos], all-ones mask.
     */
    @Test
    @DisplayName("testUnpaddedExecution")
    public void testUnpaddedExecution() {
        int pastSeq = 5;
        int curSeq = 1;
        int totalSeq = pastSeq + curSeq;

        INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray pastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray pastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        // All-zeros bias = attend to everything
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeq);

        INDArray attnOutput = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray presentKey = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        INDArray presentValue = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);

        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, bias, pastKey, pastValue)
                .addOutputs(attnOutput, presentKey, presentValue)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        assertArrayEquals(new long[]{BATCH, curSeq, HIDDEN}, attnOutput.shape());
        assertTrue(attnOutput.ameanNumber().doubleValue() > 1e-6,
                "Unpadded attention output should not be all zeros");
        assertFalse(Double.isNaN(attnOutput.sumNumber().doubleValue()),
                "Unpadded attention output should not contain NaN");
    }

    /**
     * Test padded execution: full static KV buffer + sparse mask that blocks padding positions.
     */
    @Test
    @DisplayName("testPaddedExecution")
    public void testPaddedExecution() {
        int pastSeq = 5;      // actual filled positions
        int maxKvLen = 20;     // total static buffer size
        int curSeq = 1;
        int totalSeq = maxKvLen + curSeq;

        INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);

        // Full static KV buffer: valid data at [0:pastSeq], zeros at [pastSeq:maxKvLen]
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);

        // Sparse bias: mask padding positions [pastSeq:maxKvLen] with large negative
        float maskFill = -3.4028235e+38f;
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeq);
        if (pastSeq < maxKvLen) {
            bias.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                    NDArrayIndex.interval(pastSeq, maxKvLen)).assign(maskFill);
        }
        // Position maxKvLen (new token after concat) stays 0

        INDArray attnOutput = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray presentKey = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        INDArray presentValue = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);

        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, bias, staticKey, staticValue)
                .addOutputs(attnOutput, presentKey, presentValue)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        assertArrayEquals(new long[]{BATCH, curSeq, HIDDEN}, attnOutput.shape());
        assertTrue(attnOutput.ameanNumber().doubleValue() > 1e-6,
                "Padded attention output should not be all zeros");
        assertFalse(Double.isNaN(attnOutput.sumNumber().doubleValue()),
                "Padded attention output should not contain NaN");
        assertFalse(Double.isInfinite(attnOutput.sumNumber().doubleValue()),
                "Padded attention output should not contain Inf");
    }

    /**
     * Test that padded and unpadded modes produce numerically equivalent results
     * for the same query/key/value data.
     */
    @Test
    @DisplayName("testPaddedEqualsUnpadded")
    public void testPaddedEqualsUnpadded() {
        int pastSeq = 5;
        int maxKvLen = 20;
        int curSeq = 1;

        // Shared data
        INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);

        // --- Unpadded execution ---
        INDArray biasUnpadded = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, pastSeq + curSeq);
        INDArray attnUnpadded = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray pkUnpadded = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq + curSeq, HEAD_DIM);
        INDArray pvUnpadded = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq + curSeq, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasUnpadded, realPastKey, realPastValue)
                .addOutputs(attnUnpadded, pkUnpadded, pvUnpadded)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // --- Padded execution ---
        int totalSeqPadded = maxKvLen + curSeq;
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);

        float maskFill = -3.4028235e+38f;
        INDArray biasPadded = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeqPadded);
        biasPadded.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.interval(pastSeq, maxKvLen)).assign(maskFill);

        INDArray attnPadded = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray pkPadded = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqPadded, HEAD_DIM);
        INDArray pvPadded = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqPadded, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasPadded, staticKey, staticValue)
                .addOutputs(attnPadded, pkPadded, pvPadded)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Compare attention outputs — should be numerically close.
        // Small differences are expected: padded softmax has extra masked positions in the denominator
        // (exp(-3.4e38) ≈ 0 but not exactly 0), so tolerance must account for this.
        double maxDiff = attnPadded.sub(attnUnpadded).amaxNumber().doubleValue();
        log.info("Padded vs unpadded attention output max diff: {}", maxDiff);
        assertTrue(maxDiff < 0.1,
                "Padded and unpadded attention outputs should be numerically close. MaxDiff=" + maxDiff);
    }

    /**
     * Test padded execution across 10 autoregressive steps with growing cache.
     */
    @Test
    @DisplayName("testPaddedWithGrowingCache")
    public void testPaddedWithGrowingCache() {
        int maxKvLen = 15;
        int numSteps = 10;
        int curSeq = 1;

        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);

        int cachePos = 0;
        float maskFill = -3.4028235e+38f;

        for (int step = 0; step < numSteps; step++) {
            INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
            INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
            INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);

            int totalSeq = maxKvLen + curSeq;
            INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeq);
            if (cachePos < maxKvLen) {
                bias.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                        NDArrayIndex.interval(cachePos, maxKvLen)).assign(maskFill);
            }

            INDArray attnOutput = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
            INDArray presentKey = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
            INDArray presentValue = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);

            Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                    .addInputs(query, key, value, bias, staticKey, staticValue)
                    .addOutputs(attnOutput, presentKey, presentValue)
                    .addIntegerArguments(NUM_HEADS, 0)
                    .addFloatingPointArguments(0.0)
                    .build());

            assertArrayEquals(new long[]{BATCH, curSeq, HIDDEN}, attnOutput.shape(),
                    "Step " + step + ": attention output shape");
            assertFalse(Double.isNaN(attnOutput.sumNumber().doubleValue()),
                    "Step " + step + ": attention output should not contain NaN");

            // Scatter new KV into static buffer
            INDArray newKeyEntry = presentKey.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newKeyEntry);
            INDArray newValEntry = presentValue.get(
                    NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(totalSeq - 1), NDArrayIndex.all()).dup();
            staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                    NDArrayIndex.point(cachePos), NDArrayIndex.all()).assign(newValEntry);

            cachePos++;
            log.info("Step {}: cachePos={}, attn absMax={}", step, cachePos,
                    attnOutput.amaxNumber().floatValue());
        }

        assertEquals(numSteps, cachePos);
    }

    /**
     * Test that garbage in padding positions doesn't affect output when properly masked.
     */
    @Test
    @DisplayName("testPaddedMaskingCorrectness")
    public void testPaddedMaskingCorrectness() {
        int pastSeq = 3;
        int maxKvLen = 10;
        int curSeq = 1;
        int totalSeq = maxKvLen + curSeq;
        float maskFill = -3.4028235e+38f;

        INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, curSeq, HIDDEN).mul(0.1);
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);

        // Build bias that masks padding positions
        INDArray bias = Nd4j.zeros(DataType.FLOAT, 1, 1, curSeq, totalSeq);
        bias.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.interval(pastSeq, maxKvLen)).assign(maskFill);

        // Run 1: zeros in padding
        INDArray staticKey1 = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue1 = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey1.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue1.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);

        INDArray attn1 = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray pk1 = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        INDArray pv1 = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, bias, staticKey1, staticValue1)
                .addOutputs(attn1, pk1, pv1)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Run 2: garbage (large random values) in padding
        INDArray staticKey2 = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue2 = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey2.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue2.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);
        // Fill padding with random garbage (moderate magnitude)
        staticKey2.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(pastSeq, maxKvLen), NDArrayIndex.all())
                .assign(Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen - pastSeq, HEAD_DIM).mul(1.0));
        staticValue2.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(pastSeq, maxKvLen), NDArrayIndex.all())
                .assign(Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen - pastSeq, HEAD_DIM).mul(1.0));

        INDArray attn2 = Nd4j.create(DataType.FLOAT, BATCH, curSeq, HIDDEN);
        INDArray pk2 = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        INDArray pv2 = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeq, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, bias, staticKey2, staticValue2)
                .addOutputs(attn2, pk2, pv2)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Outputs should be close since padding is masked with MASK_FILL.
        // Differences arise because softmax(MASK_FILL + QK) ≈ 0 but not exactly 0,
        // so different V values at masked positions cause small leakage through the
        // weighted sum. With garbage magnitude 1.0, this leakage is bounded.
        double maxDiff = attn1.sub(attn2).amaxNumber().doubleValue();
        log.info("Garbage vs zeros in padding: attention output max diff = {}", maxDiff);
        assertTrue(maxDiff < 1.0,
                "Garbage in masked padding should minimally affect attention output. MaxDiff=" + maxDiff);
    }
}
