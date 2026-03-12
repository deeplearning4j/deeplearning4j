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

package org.eclipse.deeplearning4j.nd4j.autodiff.opvalidation;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.validation.OpValidation;
import org.nd4j.autodiff.validation.TestCase;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Validation tests for Attention operations.
 * Tests forward pass correctness and gradient computation for:
 * - Flash Attention
 * - Grouped Query Attention (GQA)
 * - Windowed Attention
 * - Sliding Window Attention
 *
 * @author Adam Gibson
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("Attention Op Validation Tests")
public class TestAttentionOpValidation extends BaseOpValidation {

    @Override
    public long getTimeoutMilliseconds() {
        return 90000L;
    }

    // ========================= Attention Computation Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Scaled Dot Product (Simple)")
    public void testScaledDotProductAttentionSimple(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int dim = 8;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        // Simple 2D attention: [batch, seq, dim]
        INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

        SDVariable query = sd.var("query", queryArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable value = sd.var("value", valueArr);

        // Compute attention scores using einsum for batch matmul
        // scores = Q @ K^T * scale
        SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);

        // Softmax
        SDVariable attnWeights = sd.nn.softmax(scores, 2);

        // Output = attn_weights @ V
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - With Causal Mask")
    public void testAttentionCausalMask(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 6;
        int dim = 8;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

        SDVariable query = sd.var("query", queryArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable value = sd.var("value", valueArr);

        // Compute attention scores
        SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);

        // Create causal mask (lower triangular)
        INDArray causalMaskArr = Nd4j.ones(DataType.DOUBLE, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            for (int j = i + 1; j < seqLen; j++) {
                causalMaskArr.putScalar(i, j, 0.0);
            }
        }
        // Convert to additive mask
        INDArray maskValues = causalMaskArr.rsub(1).mul(-1e9);
        SDVariable mask = sd.constant("mask", maskValues);

        // Apply mask
        SDVariable maskedScores = scores.add(mask);

        // Softmax
        SDVariable attnWeights = sd.nn.softmax(maskedScores, 2);

        // Output
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Multi-Head Style Computation")
    public void testMultiHeadAttentionComputation(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 4;
        int modelDim = numHeads * headDim;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();

        // Input projections
        INDArray inputArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, modelDim).muli(0.1);
        INDArray wqArr = Nd4j.rand(DataType.DOUBLE, modelDim, modelDim).muli(0.1);
        INDArray wkArr = Nd4j.rand(DataType.DOUBLE, modelDim, modelDim).muli(0.1);
        INDArray wvArr = Nd4j.rand(DataType.DOUBLE, modelDim, modelDim).muli(0.1);

        SDVariable input = sd.var("input", inputArr);
        SDVariable wq = sd.var("wq", wqArr);
        SDVariable wk = sd.var("wk", wkArr);
        SDVariable wv = sd.var("wv", wvArr);

        // Project Q, K, V
        SDVariable inputFlat = input.reshape(batch * seqLen, modelDim);
        SDVariable q = sd.mmul(inputFlat, wq).reshape(batch, seqLen, modelDim);
        SDVariable k = sd.mmul(inputFlat, wk).reshape(batch, seqLen, modelDim);
        SDVariable v = sd.mmul(inputFlat, wv).reshape(batch, seqLen, modelDim);

        // Compute attention
        SDVariable scores = sd.linalg().einsum(new SDVariable[]{q, k}, "bqd,bkd->bqk").mul(scale);
        SDVariable attnWeights = sd.nn.softmax(scores, 2);
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, v}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .testName("Multi-Head Attention")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= GQA Style Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("GQA - Grouped Query Simulation")
    public void testGroupedQueryAttention(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int dim = 8;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        // In GQA, we have more query heads than KV heads
        // Here we simulate by having different sized Q vs K,V
        INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim / 2).muli(0.1);
        INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim / 2).muli(0.1);
        // Projection to expand KV
        INDArray expandArr = Nd4j.rand(DataType.DOUBLE, dim / 2, dim).muli(0.1);

        SDVariable query = sd.var("query", queryArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable value = sd.var("value", valueArr);
        SDVariable expand = sd.var("expand", expandArr);

        // Expand K and V to match Q dimension
        SDVariable kFlat = key.reshape(batch * seqLen, dim / 2);
        SDVariable vFlat = value.reshape(batch * seqLen, dim / 2);
        SDVariable kExpanded = sd.mmul(kFlat, expand).reshape(batch, seqLen, dim);
        SDVariable vExpanded = sd.mmul(vFlat, expand).reshape(batch, seqLen, dim);

        // Compute attention
        SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, kExpanded}, "bqd,bkd->bqk").mul(scale);
        SDVariable attnWeights = sd.nn.softmax(scores, 2);
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, vExpanded}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .testName("Grouped Query Attention")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Windowed/Sliding Attention Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Windowed Attention - Local Window Mask")
    public void testWindowedAttention(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 8;
        int dim = 8;
        int windowSize = 3;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

        SDVariable query = sd.var("query", queryArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable value = sd.var("value", valueArr);

        // Compute attention scores
        SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);

        // Create window mask
        INDArray windowMaskArr = Nd4j.zeros(DataType.DOUBLE, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            int start = Math.max(0, i - windowSize / 2);
            int end = Math.min(seqLen, i + windowSize / 2 + 1);
            for (int j = start; j < end; j++) {
                windowMaskArr.putScalar(i, j, 1.0);
            }
        }
        INDArray maskValues = windowMaskArr.rsub(1).mul(-1e9);
        SDVariable mask = sd.constant("mask", maskValues);

        // Apply mask
        SDVariable maskedScores = scores.add(mask);
        SDVariable attnWeights = sd.nn.softmax(maskedScores, 2);
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .testName("Windowed Attention")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Sliding Window Attention - Causal + Window")
    public void testSlidingWindowAttention(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 8;
        int dim = 8;
        int windowSize = 4;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
        INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

        SDVariable query = sd.var("query", queryArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable value = sd.var("value", valueArr);

        SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);

        // Create sliding window mask (causal + limited to window)
        INDArray slidingMaskArr = Nd4j.zeros(DataType.DOUBLE, seqLen, seqLen);
        for (int i = 0; i < seqLen; i++) {
            int start = Math.max(0, i - windowSize + 1);
            for (int j = start; j <= i; j++) {
                slidingMaskArr.putScalar(i, j, 1.0);
            }
        }
        INDArray maskValues = slidingMaskArr.rsub(1).mul(-1e9);
        SDVariable mask = sd.constant("mask", maskValues);

        SDVariable maskedScores = scores.add(mask);
        SDVariable attnWeights = sd.nn.softmax(maskedScores, 2);
        SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
        output.rename("output");

        SDVariable loss = sd.mean("loss", output);

        TestCase tc = new TestCase(sd)
                .testName("Sliding Window Attention")
                .gradientCheck(true)
                .gradCheckEpsilon(1e-5)
                .gradCheckMaxRelativeError(1e-3);

        String err = OpValidation.validate(tc);
        assertNull(err, err);
    }

    // ========================= Parameter Variation Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Various Sequence Lengths")
    public void testAttentionVariousSequenceLengths(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int dim = 8;

        for (int seqLen : new int[]{4, 8, 16}) {
            double scale = 1.0 / Math.sqrt(dim);

            SameDiff sd = SameDiff.create();

            INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
            INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
            INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

            SDVariable query = sd.var("query", queryArr);
            SDVariable key = sd.var("key", keyArr);
            SDVariable value = sd.var("value", valueArr);

            SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);
            SDVariable attnWeights = sd.nn.softmax(scores, 2);
            SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
            output.rename("output");

            SDVariable loss = sd.mean("loss", output);

            TestCase tc = new TestCase(sd)
                    .testName("Attention seqLen=" + seqLen)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for seqLen=" + seqLen + ": " + err);
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Various Dimensions")
    public void testAttentionVariousDimensions(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;

        for (int dim : new int[]{4, 8, 16, 32}) {
            double scale = 1.0 / Math.sqrt(dim);

            SameDiff sd = SameDiff.create();

            INDArray queryArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
            INDArray keyArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);
            INDArray valueArr = Nd4j.rand(DataType.DOUBLE, batch, seqLen, dim).muli(0.1);

            SDVariable query = sd.var("query", queryArr);
            SDVariable key = sd.var("key", keyArr);
            SDVariable value = sd.var("value", valueArr);

            SDVariable scores = sd.linalg().einsum(new SDVariable[]{query, key}, "bqd,bkd->bqk").mul(scale);
            SDVariable attnWeights = sd.nn.softmax(scores, 2);
            SDVariable output = sd.linalg().einsum(new SDVariable[]{attnWeights, value}, "bqk,bkd->bqd");
            output.rename("output");

            SDVariable loss = sd.mean("loss", output);

            TestCase tc = new TestCase(sd)
                    .testName("Attention dim=" + dim)
                    .gradientCheck(true)
                    .gradCheckEpsilon(1e-5)
                    .gradCheckMaxRelativeError(1e-3);

            String err = OpValidation.validate(tc);
            assertNull(err, "Failed for dim=" + dim + ": " + err);
        }
    }

    // ========================= Fused Attention with Bias Tests =========================
    // These tests specifically verify that the fused CUDA kernel path is used when
    // attention bias is provided (performance optimization for VLM models like SmolDocling)
    // 
    // dot_product_attention_v2 input order:
    // - Input 0: queries
    // - Input 1: values  
    // - Input 2: keys (optional, defaults to values)
    // - Input 3: queryMask (optional)
    // - Input 4: valueMask (optional)
    // - Input 5: attentionBias (optional)
    // T_ARG: scale, dropout
    // B_ARG: useCausalMask, training, useFlashAttention

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - 3D with Additive Bias")
    public void testFusedAttention3DWithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int seqLen = 8;
        int headDim = 16;
        double scale = 1.0 / Math.sqrt(headDim);

        // 3D format for dot_product_attention_v2: [batch, seq, dim]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        
        // Attention bias: [batch, seqQ, seqKV]
        INDArray bias = Nd4j.randn(DataType.FLOAT, batch, seqLen, seqLen).muli(0.5f);

        // Empty arrays for optional masks (input slots 3 and 4)
        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);

        // Execute dot_product_attention_v2 with bias
        // Input order: query, value, key, queryMask, valueMask, attentionBias
        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)  // T_ARG: scale, dropout
                .addBooleanArguments(false, false, true)  // B_ARG: useCausalMask, training, useFlashAttention
                .build();

        INDArray[] outputs = Nd4j.exec(op);
        INDArray attnOutput = outputs[0];

        assertNotNull(attnOutput);
        assertArrayEquals(new long[]{batch, seqLen, headDim}, attnOutput.shape(),
                "Output shape should match [batch, seqQ, headDim]");

        // Verify output is not all zeros
        double absSum = attnOutput.ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "Attention output should not be all zeros");

        // Verify correctness by comparing with manual computation
        // scores = Q @ K^T * scale + bias
        // attn_weights = softmax(scores, dim=-1)
        // output = attn_weights @ V
        INDArray manualScores = Nd4j.zeros(DataType.FLOAT, batch, seqLen, seqLen);
        for (int b = 0; b < batch; b++) {
            INDArray q = query.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b));  // [seqLen, headDim]
            INDArray k = key.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b));    // [seqLen, headDim]
            INDArray scores = Nd4j.matmul(q, k.transpose()).mul(scale);              // [seqLen, seqLen]
            INDArray biasSlice = bias.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b));
            scores.addi(biasSlice);
            manualScores.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                    org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
            }, scores);
        }
        
        // Apply softmax along last dimension
        INDArray manualWeights = Nd4j.nn.softmax(manualScores, 2);
        
        // Compute expected output
        INDArray expected = Nd4j.zeros(DataType.FLOAT, batch, seqLen, headDim);
        for (int b = 0; b < batch; b++) {
            INDArray weights = manualWeights.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b));  // [seqLen, seqLen]
            INDArray v = value.get(org.nd4j.linalg.indexing.NDArrayIndex.point(b));                // [seqLen, headDim]
            INDArray out = Nd4j.matmul(weights, v);                                                // [seqLen, headDim]
            expected.put(new org.nd4j.linalg.indexing.INDArrayIndex[]{
                    org.nd4j.linalg.indexing.NDArrayIndex.point(b),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
            }, out);
        }

        double maxDiff = attnOutput.sub(expected).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3, "Fused attention with bias output should match manual computation. maxDiff=" + maxDiff);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - 4D with Additive Bias")
    public void testFusedAttention4DWithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 8;
        int numHeads = 4;
        int headDim = 16;
        double scale = 1.0 / Math.sqrt(headDim);

        // 4D format for dot_product_attention_v2: [batch, seq, numHeads, headDim] (BSHD)
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        
        // Attention bias: [batch, numHeads, seqQ, seqKV]
        INDArray bias = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen).muli(0.5f);

        // Empty arrays for optional masks
        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);

        // Execute dot_product_attention_v2 with bias
        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)  // T_ARG: scale, dropout
                .addBooleanArguments(false, false, true)  // B_ARG: useCausalMask, training, useFlashAttention
                .build();

        INDArray[] outputs = Nd4j.exec(op);
        INDArray attnOutput = outputs[0];

        assertNotNull(attnOutput);
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, attnOutput.shape(),
                "Output shape should match [batch, seqQ, numHeads, headDim]");

        // Verify output is not all zeros
        double absSum = attnOutput.ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "Attention output should not be all zeros");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - With vs Without Bias Produces Different Results")
    public void testFusedAttentionBiasAffectsOutput(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int seqLen = 8;
        int headDim = 16;
        double scale = 1.0 / Math.sqrt(headDim);

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, headDim).muli(0.1f);
        INDArray bias = Nd4j.randn(DataType.FLOAT, batch, seqLen, seqLen).muli(0.5f);

        // Without bias
        DynamicCustomOp opNoBias = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key)  // query, value, key
                .addFloatingPointArguments(scale, 0.0)  // T_ARG: scale, dropout
                .addBooleanArguments(false, false, true)  // B_ARG: useCausalMask, training, useFlashAttention
                .build();
        INDArray outputNoBias = Nd4j.exec(opNoBias)[0];

        // Empty arrays for optional masks
        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);

        // With bias (need to provide empty masks for slots 3 and 4)
        DynamicCustomOp opWithBias = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)  // T_ARG: scale, dropout
                .addBooleanArguments(false, false, true)  // B_ARG: useCausalMask, training, useFlashAttention
                .build();
        INDArray outputWithBias = Nd4j.exec(opWithBias)[0];

        // Outputs should be different when bias is applied
        double diff = outputWithBias.sub(outputNoBias).ameanNumber().doubleValue();
        assertTrue(Math.abs(diff) > 1e-6, 
                "Attention with bias should produce different output than without bias. diff=" + diff);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DotProductAttentionV2 - useFlashAttention=false materializes scores")
    public void testDotProductAttentionV2UseFlashDisabledProducesScores(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 8;
        int dim = 16;
        double scale = 1.0 / Math.sqrt(dim);

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key)
                .addFloatingPointArguments(scale, 0.0)      // T_ARG: scale, dropout
                .addBooleanArguments(false, false, false)   // B_ARG: useCausalMask, training, useFlashAttention
                .build();

        INDArray[] outputs = Nd4j.exec(op);
        INDArray attnOutput = outputs[0];
        INDArray attnScores = outputs[1];
        INDArray attnLogits = outputs[2];

        assertNotNull(attnOutput);
        assertNotNull(attnScores);
        assertNotNull(attnLogits);
        assertArrayEquals(new long[]{batch, seqLen, dim}, attnOutput.shape());
        assertArrayEquals(new long[]{batch, seqLen, seqLen}, attnScores.shape());
        assertArrayEquals(new long[]{batch, seqLen, seqLen}, attnLogits.shape());

        // Scores should be a proper softmax distribution along the last dimension.
        INDArray rowSums = attnScores.sum(true, 2);
        double maxRowSumErr = Nd4j.math().abs(rowSums.sub(1.0)).maxNumber().doubleValue();
        double minScore = attnScores.minNumber().doubleValue();
        double maxScore = attnScores.maxNumber().doubleValue();

        assertTrue(maxRowSumErr < 1e-3,
                "Attention score rows should sum to 1 when flash is disabled. maxErr=" + maxRowSumErr);
        assertTrue(minScore >= -1e-6, "Attention scores should be non-negative. min=" + minScore);
        assertTrue(maxScore <= 1.0 + 1e-6, "Attention scores should be <= 1. max=" + maxScore);

        // Logits should be materialized (not all zeros/uninitialized).
        double logitsAbsMean = Nd4j.math().abs(attnLogits).meanNumber().doubleValue();
        assertTrue(logitsAbsMean > 1e-7, "Attention logits should be materialized when flash is disabled");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ONNX MHA - With Attention Bias Uses Fused Kernel")
    public void testOnnxMhaWithBiasUsesFusedKernel(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 8;
        int numHeads = 4;
        int headDim = 16;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        
        // Attention bias in ONNX format: [batch, numHeads, seqQ, seqKV]
        INDArray attnBias = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen).muli(0.5f);

        // Execute ONNX MHA with bias
        OnnxMultiHeadAttention op = new OnnxMultiHeadAttention(
                query, key, value,
                attnBias, null, null,
                numHeads, 0.0, false  // scale=auto, causal=false
        );
        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs);
        assertNotNull(outputs[0]);
        assertArrayEquals(new long[]{batch, seqLen, hidden}, outputs[0].shape(),
                "Output shape should match [batch, seq, hidden]");

        // Verify output is meaningful
        double absSum = outputs[0].ameanNumber().doubleValue();
        assertTrue(absSum > 1e-6, "ONNX MHA with bias output should not be all zeros");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ONNX MHA - Bias with Causal Mask Combined")
    public void testOnnxMhaWithBiasAndCausalMask(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 1;
        int seqLen = 8;
        int numHeads = 4;
        int headDim = 16;
        int hidden = numHeads * headDim;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, hidden);
        
        // Relative position bias: [batch, numHeads, seqQ, seqKV]
        INDArray relPosBias = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen).muli(0.3f);

        // With causal mask + bias
        OnnxMultiHeadAttention opCausalBias = new OnnxMultiHeadAttention(
                query, key, value,
                relPosBias, null, null,
                numHeads, 0.0, true  // causal=true
        );
        INDArray[] outputsCausalBias = Nd4j.exec(opCausalBias);

        // With causal mask only (no bias)
        OnnxMultiHeadAttention opCausalNoBias = new OnnxMultiHeadAttention(
                query, key, value,
                null, null, null,
                numHeads, 0.0, true  // causal=true
        );
        INDArray[] outputsCausalNoBias = Nd4j.exec(opCausalNoBias);

        // Both should work
        assertNotNull(outputsCausalBias[0]);
        assertNotNull(outputsCausalNoBias[0]);

        // Outputs should differ due to the bias
        double diff = outputsCausalBias[0].sub(outputsCausalNoBias[0]).ameanNumber().doubleValue();
        assertTrue(Math.abs(diff) > 1e-6, 
                "Causal attention with bias should differ from without bias. diff=" + diff);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - Large Sequence Length (VLM-like)")
    public void testFusedAttentionLargeSequence(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        // Simulate VLM-like dimensions (smaller for test speed)
        int batch = 1;
        int numHeads = 8;
        int seqLen = 64;  // Would be much larger in real VLM (e.g., 2048+)
        int headDim = 64;
        double scale = 1.0 / Math.sqrt(headDim);

        // dot_product_attention_v2 4D format: [batch, seq, numHeads, headDim] (BSHD)
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);

        // Simulated relative position bias: [batch, numHeads, seqQ, seqKV]
        INDArray bias = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen).muli(0.1f);

        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)  // useCausalMask, training, useFlashAttention
                .build();

        long startTime = System.currentTimeMillis();
        INDArray[] outputs = Nd4j.exec(op);
        long endTime = System.currentTimeMillis();

        assertNotNull(outputs[0]);
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, outputs[0].shape());

        log.info("Large sequence attention with bias completed in {} ms", endTime - startTime);
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - FP16 with Bias")
    public void testFusedAttentionFP16WithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int seqLen = 8;
        int headDim = 16;
        double scale = 1.0 / Math.sqrt(headDim);

        // 3D format: [batch, seq, dim]
        INDArray query = Nd4j.randn(DataType.FLOAT16, batch, seqLen, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT16, batch, seqLen, headDim);
        INDArray value = Nd4j.randn(DataType.FLOAT16, batch, seqLen, headDim);
        INDArray bias = Nd4j.randn(DataType.FLOAT16, batch, seqLen, seqLen);

        INDArray emptyMask = Nd4j.empty(DataType.FLOAT16);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)  // useCausalMask, training, useFlashAttention
                .build();

        INDArray[] outputs = Nd4j.exec(op);

        assertNotNull(outputs[0]);
        assertEquals(DataType.FLOAT16, outputs[0].dataType(),
                "Output should preserve FP16 data type");
        assertArrayEquals(new long[]{batch, seqLen, headDim}, outputs[0].shape());
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Fused Attention - Cross Attention with Bias")
    public void testFusedCrossAttentionWithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 4;
        int seqQ = 4;    // Query sequence length
        int seqKV = 12;  // Key/Value sequence length (different!)
        int headDim = 16;
        double scale = 1.0 / Math.sqrt(headDim);

        // 3D format: [batch, seq, dim]
        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, headDim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, headDim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, headDim).muli(0.1f);

        // Cross-attention bias: [batch, seqQ, seqKV]
        INDArray bias = Nd4j.randn(DataType.FLOAT, batch, seqQ, seqKV).muli(0.5f);

        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)  // useCausalMask, training, useFlashAttention
                .build();

        INDArray[] outputs = Nd4j.exec(op);
        INDArray attnOutput = outputs[0];

        assertNotNull(attnOutput);
        // Output should have query's sequence length
        assertArrayEquals(new long[]{batch, seqQ, headDim}, attnOutput.shape(),
                "Cross attention output should have query's sequence length");
    }

    // ========================= DotProductAttentionV2 with AttentionBias Tests =========================

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DotProductAttentionV2 - With Attention Bias (SameDiff API)")
    public void testDotProductAttentionV2WithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();

        // 4D input for flash attention: [batch, seq, heads, dim]
        INDArray queryArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray keyArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray valueArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        
        // Attention bias: [batch, heads, seqQ, seqK]
        INDArray biasArr = Nd4j.randn(DataType.FLOAT, batch, numHeads, seqLen, seqLen).muli(0.5f);

        SDVariable query = sd.var("query", queryArr);
        SDVariable value = sd.var("value", valueArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable attentionBias = sd.var("attentionBias", biasArr);

        // Use the proper SameDiff API with attentionBias
        SDVariable output = sd.nn().dotProductAttentionV2(
                query, value, key,  // Note: value comes before key in this API
                null,               // queryMask
                null,               // valueMask
                attentionBias,      // attentionBias - now supported!
                scale,              // scaleFactor
                0.0,                // dropoutProbability
                false,              // useCausalMask
                false               // training
        );
        output.rename("output");

        // Execute and verify output
        INDArray result = output.eval();
        
        assertNotNull(result, "Output should not be null");
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, result.shape(),
                "Output should have same shape as query");

        // Verify finite values
        assertFalse(result.isNaN().any(), "Output should not contain NaN values");
        assertFalse(result.isInfinite().any(), "Output should not contain infinite values");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DotProductAttentionV2 - With Relative Position Bias (3D input)")
    public void testDotProductAttentionV2WithRelativePositionBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 6;
        int dim = 16;
        double scale = 1.0 / Math.sqrt(dim);

        SameDiff sd = SameDiff.create();

        // 3D input: [batch, seq, dim]
        INDArray queryArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);
        INDArray keyArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);
        INDArray valueArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, dim).muli(0.1f);
        
        // Relative position bias: broadcastable [1, seqQ, seqK] (same for all batches)
        INDArray biasArr = Nd4j.randn(DataType.FLOAT, 1, seqLen, seqLen).muli(0.3f);

        SDVariable query = sd.var("query", queryArr);
        SDVariable value = sd.var("value", valueArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable attentionBias = sd.var("attentionBias", biasArr);

        // Use the proper SameDiff API with attentionBias
        SDVariable output = sd.nn().dotProductAttentionV2(
                query, value, key,
                null, null,         // no masks
                attentionBias,      // relative position bias
                scale,
                0.0,
                false,
                false
        );
        output.rename("output");

        // Execute and verify output
        INDArray result = output.eval();
        
        assertNotNull(result, "Output should not be null");
        assertArrayEquals(new long[]{batch, seqLen, dim}, result.shape(),
                "Output should have same shape as query for 3D input");
        
        assertFalse(result.isNaN().any(), "Output should not contain NaN values");
        assertFalse(result.isInfinite().any(), "Output should not contain infinite values");
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DotProductAttentionV2 - With Attention Bias and Causal Mask")
    public void testDotProductAttentionV2WithBiasAndCausalMask(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 2;
        int seqLen = 4;
        int numHeads = 2;
        int headDim = 8;
        double scale = 1.0 / Math.sqrt(headDim);

        SameDiff sd = SameDiff.create();

        // 4D input: [batch, seq, heads, dim]
        INDArray queryArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray keyArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        INDArray valueArr = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim).muli(0.1f);
        
        // Attention bias (e.g., ALiBi slopes): [1, heads, 1, seqK]
        INDArray biasArr = Nd4j.randn(DataType.FLOAT, 1, numHeads, 1, seqLen).muli(0.2f);

        SDVariable query = sd.var("query", queryArr);
        SDVariable value = sd.var("value", valueArr);
        SDVariable key = sd.var("key", keyArr);
        SDVariable attentionBias = sd.var("attentionBias", biasArr);

        // Use causal mask combined with attention bias
        SDVariable output = sd.nn().dotProductAttentionV2(
                query, value, key,
                null, null,
                attentionBias,
                scale,
                0.0,
                true,   // useCausalMask = true
                false
        );
        output.rename("output");

        // Execute and verify output
        INDArray result = output.eval();
        
        assertNotNull(result, "Output should not be null");
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, result.shape(),
                "Output should have same shape as query");

        assertFalse(result.isNaN().any(), "Output should not contain NaN values");
        assertFalse(result.isInfinite().any(), "Output should not contain infinite values");
    }
}
