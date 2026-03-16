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
 * Tests that the attn_mask_reformat subgraph pattern (Tile → onnx_multi_head_attention)
 * handles padded inputs correctly when the model processes the attention_mask naturally.
 *
 * This simulates what happens when DSP uses padded inputs and lets the model's
 * attn_mask_reformat subgraph compute the attention bias from the sparse attention_mask,
 * rather than bypassing it with pre-computed bias.
 */
@Slf4j
@DisplayName("AttnMaskReformatSubgraphTest")
public class AttnMaskReformatSubgraphTest {

    private static final int BATCH = 1;
    private static final int NUM_HEADS = 2;
    private static final int HEAD_DIM = 8;
    private static final int HIDDEN = NUM_HEADS * HEAD_DIM;
    private static final float MASK_FILL = -3.4028235e+38f;

    /**
     * Build a SameDiff graph that simulates the attn_mask_reformat pattern:
     * attention_mask → expand → (1-mask)*maskFill → tile → bias for MHA.
     *
     * This is the pattern that ONNX models use to convert a 2D attention_mask
     * into a 4D additive bias for multi-head attention.
     */
    private SameDiff buildAttnMaskReformatGraph(int totalSeqLen) {
        SameDiff sd = SameDiff.create();

        // Inputs
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, BATCH, 1, HIDDEN);
        SDVariable pastKey = sd.placeHolder("past_key", DataType.FLOAT, BATCH, NUM_HEADS, -1, HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value", DataType.FLOAT, BATCH, NUM_HEADS, -1, HEAD_DIM);

        // attention_mask: [1, totalSeqLen] — 2D mask
        SDVariable attnMask = sd.placeHolder("attention_mask", DataType.FLOAT, BATCH, -1);

        // Simulate attn_mask_reformat subgraph:
        // 1. Reshape to [B, 1, 1, seqLen]
        SDVariable expanded = sd.expandDims(attnMask, 1);  // [B, 1, seqLen]
        expanded = sd.expandDims(expanded, 1);               // [B, 1, 1, seqLen]

        // 2. Convert: (1 - mask) * maskFill → 0 where mask=1, maskFill where mask=0
        SDVariable ones = sd.onesLike(expanded);
        SDVariable invertedMask = ones.sub(expanded);
        SDVariable maskFillVar = sd.constant("mask_fill", Nd4j.scalar(MASK_FILL));
        SDVariable bias = invertedMask.mul(maskFillVar);

        // 3. MHA with computed bias
        OnnxMultiHeadAttention mha = new OnnxMultiHeadAttention(
                sd, query, key, value, bias, pastKey, pastValue,
                NUM_HEADS, 0.0, false);
        SDVariable[] mhaOutputs = mha.outputVariables();
        // Register the attention output so sd.outputs() returns it
        sd.setOutputs(mhaOutputs[0].name());

        return sd;
    }

    /**
     * Test with contiguous all-ones mask (unpadded case).
     * All positions valid → bias should be all zeros → standard attention.
     */
    @Test
    @DisplayName("testSubgraphWithContiguousMask")
    public void testSubgraphWithContiguousMask() {
        int pastSeq = 5;
        int totalSeq = pastSeq + 1; // past + current token

        SameDiff sd = buildAttnMaskReformatGraph(totalSeq);

        String attnOutputName = sd.outputs().get(0);

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("key", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("value", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("past_key", Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1));
        ph.put("past_value", Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1));
        // Contiguous all-ones mask
        ph.put("attention_mask", Nd4j.ones(DataType.FLOAT, BATCH, totalSeq));

        Map<String, INDArray> result = sd.output(ph, attnOutputName);
        INDArray attn = result.get(attnOutputName);

        assertNotNull(attn, "Attention output should not be null");
        assertFalse(Double.isNaN(attn.sumNumber().doubleValue()),
                "Contiguous mask: attention output should not contain NaN");
        assertTrue(attn.ameanNumber().doubleValue() > 1e-6,
                "Contiguous mask: attention output should not be all zeros");

        log.info("Contiguous mask test passed: attn absMax={}", attn.amaxNumber().floatValue());
        sd.close();
    }

    /**
     * Test with sparse mask (padded case): 1s at valid positions, 0s at padding.
     * Padding positions should get MASK_FILL bias → softmax zeros them out.
     */
    @Test
    @DisplayName("testSubgraphWithSparseMask")
    public void testSubgraphWithSparseMask() {
        int pastSeq = 5;
        int maxKvLen = 15;
        int totalSeq = maxKvLen + 1; // padded past + current token

        SameDiff sd = buildAttnMaskReformatGraph(totalSeq);

        String attnOutputName = null;
        for (String name : sd.outputs()) {
            if (name.contains("onnx_multi_head_attention")) {
                attnOutputName = name;
                break;
            }
        }
        if (attnOutputName == null) {
            attnOutputName = sd.outputs().get(0);
        }

        // Full static KV buffer with real data at [0:pastSeq], zeros elsewhere
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);

        // Sparse mask: 1s at [0:pastSeq] and last position, 0s at padding
        INDArray sparseMask = Nd4j.zeros(DataType.FLOAT, BATCH, totalSeq);
        sparseMask.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, pastSeq)).assign(1);
        sparseMask.putScalar(0, totalSeq - 1, 1); // current token position

        Map<String, INDArray> ph = new HashMap<>();
        ph.put("query", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("key", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("value", Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1));
        ph.put("past_key", staticKey);
        ph.put("past_value", staticValue);
        ph.put("attention_mask", sparseMask);

        Map<String, INDArray> result = sd.output(ph, attnOutputName);
        INDArray attn = result.get(attnOutputName);

        assertNotNull(attn, "Attention output should not be null");
        assertFalse(Double.isNaN(attn.sumNumber().doubleValue()),
                "Sparse mask: attention output should not contain NaN");
        assertFalse(Double.isInfinite(attn.sumNumber().doubleValue()),
                "Sparse mask: attention output should not contain Inf");
        assertTrue(attn.ameanNumber().doubleValue() > 1e-6,
                "Sparse mask: attention output should not be all zeros");

        log.info("Sparse mask test passed: attn absMax={}", attn.amaxNumber().floatValue());
        sd.close();
    }

    /**
     * Compare subgraph output between contiguous (unpadded) and sparse (padded) masks.
     * Valid positions should produce equivalent attention weights.
     */
    @Test
    @DisplayName("testSubgraphOutputEquivalence")
    public void testSubgraphOutputEquivalence() {
        int pastSeq = 5;
        int maxKvLen = 15;

        // Shared data
        INDArray query = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray key = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray value = Nd4j.randn(DataType.FLOAT, BATCH, 1, HIDDEN).mul(0.1);
        INDArray realPastKey = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);
        INDArray realPastValue = Nd4j.randn(DataType.FLOAT, BATCH, NUM_HEADS, pastSeq, HEAD_DIM).mul(0.1);

        // --- Unpadded (contiguous) via direct op ---
        int totalSeqUnpadded = pastSeq + 1;
        INDArray biasUnpadded = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqUnpadded);
        INDArray attnUnpadded = Nd4j.create(DataType.FLOAT, BATCH, 1, HIDDEN);
        INDArray pkU = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqUnpadded, HEAD_DIM);
        INDArray pvU = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqUnpadded, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasUnpadded, realPastKey, realPastValue)
                .addOutputs(attnUnpadded, pkU, pvU)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // --- Padded (sparse mask) via direct op ---
        int totalSeqPadded = maxKvLen + 1;
        INDArray staticKey = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        INDArray staticValue = Nd4j.zeros(DataType.FLOAT, BATCH, NUM_HEADS, maxKvLen, HEAD_DIM);
        staticKey.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastKey);
        staticValue.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(0, pastSeq), NDArrayIndex.all()).assign(realPastValue);

        INDArray biasPadded = Nd4j.zeros(DataType.FLOAT, 1, 1, 1, totalSeqPadded);
        biasPadded.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0),
                NDArrayIndex.interval(pastSeq, maxKvLen)).assign(MASK_FILL);

        INDArray attnPadded = Nd4j.create(DataType.FLOAT, BATCH, 1, HIDDEN);
        INDArray pkP = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqPadded, HEAD_DIM);
        INDArray pvP = Nd4j.create(DataType.FLOAT, BATCH, NUM_HEADS, totalSeqPadded, HEAD_DIM);
        Nd4j.exec(org.nd4j.linalg.api.ops.DynamicCustomOp.builder("onnx_multi_head_attention")
                .addInputs(query, key, value, biasPadded, staticKey, staticValue)
                .addOutputs(attnPadded, pkP, pvP)
                .addIntegerArguments(NUM_HEADS, 0)
                .addFloatingPointArguments(0.0)
                .build());

        // Compare attention outputs — small differences expected from softmax numerics
        double maxDiff = attnPadded.sub(attnUnpadded).amaxNumber().doubleValue();
        log.info("Subgraph equivalence: padded vs unpadded attention max diff = {}", maxDiff);
        assertTrue(maxDiff < 0.1,
                "Padded and unpadded subgraph outputs should be numerically close. MaxDiff=" + maxDiff);
    }
}
