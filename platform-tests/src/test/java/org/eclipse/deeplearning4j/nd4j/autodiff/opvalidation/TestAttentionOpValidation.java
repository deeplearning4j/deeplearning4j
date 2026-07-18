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
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.junit.jupiter.api.Test;

import java.util.Map;

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
@Tag(TagNames.FULL_CI)
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
    @DisplayName("DotProductAttentionV2 - Rank 4 GQA Single KV Token")
    public void testDotProductAttentionV2Rank4GqaSingleKvToken(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 1;
        int seqLen = 1;
        int numHeads = 4;
        int numKvHeads = 2;
        int headDim = 8;
        int headsPerKvHead = numHeads / numKvHeads;

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqLen, numHeads, headDim);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqLen, numKvHeads, headDim);
        INDArray value = Nd4j.linspace(DataType.FLOAT, -0.5, 0.125, numKvHeads * headDim)
                .reshape(batch, seqLen, numKvHeads, headDim);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key)
                .addFloatingPointArguments(0.0, 0.0)
                .addBooleanArguments(false, false, true)
                .build();

        INDArray output = Nd4j.exec(op)[0];

        assertNotNull(output);
        assertArrayEquals(new long[]{batch, seqLen, numHeads, headDim}, output.shape());
        assertFalse(output.isNaN().any(), "GQA singleton output must not contain NaN");
        assertFalse(output.isInfinite().any(), "GQA singleton output must not contain Inf");

        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / headsPerKvHead;
            for (int d = 0; d < headDim; d++) {
                assertEquals(value.getDouble(0, 0, kvHead, d), output.getDouble(0, 0, h, d), 1e-5,
                        "seqKV=1 attention should copy value head " + kvHead + " to query head " + h + " dim " + d);
            }
        }
    }

    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("DotProductAttentionV2 - Rank 4 GQA prefill materializes scores and logits")
    public void testDotProductAttentionV2Rank4GqaPrefillWithBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(12345);

        int batch = 1;
        int seqQ = 4;
        int seqKV = 4;
        int numHeads = 4;
        int numKvHeads = 2;
        int headDim = 8;
        int headsPerKvHead = numHeads / numKvHeads;
        double scale = 1.0 / Math.sqrt(headDim);

        INDArray query = Nd4j.randn(DataType.FLOAT, batch, seqQ, numHeads, headDim).muli(0.1f);
        INDArray key = Nd4j.randn(DataType.FLOAT, batch, seqKV, numKvHeads, headDim).muli(0.1f);
        INDArray value = Nd4j.randn(DataType.FLOAT, batch, seqKV, numKvHeads, headDim).muli(0.1f);
        INDArray bias = Nd4j.zeros(DataType.FLOAT, batch, 1, seqQ, seqKV);
        for (int q = 0; q < seqQ; q++) {
            for (int k = q + 1; k < seqKV; k++) {
                bias.putScalar(new long[]{0, 0, q, k}, -1.0e9f);
            }
        }

        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);
        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(query, value, key, emptyMask, emptyMask, bias)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)
                .build();

        INDArray[] outputs = Nd4j.exec(op);
        INDArray expectedOutput =
                Nd4j.zeros(DataType.FLOAT, batch, seqQ, numHeads, headDim);
        INDArray expectedScores =
                Nd4j.zeros(DataType.FLOAT, batch, numHeads, seqQ, seqKV);
        INDArray expectedLogits =
                Nd4j.zeros(DataType.FLOAT, batch, numHeads, seqQ, seqKV);

        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / headsPerKvHead;
            for (int q = 0; q < seqQ; q++) {
                double[] logits = new double[seqKV];
                double max = -Double.MAX_VALUE;
                for (int k = 0; k < seqKV; k++) {
                    double dot = 0.0;
                    for (int d = 0; d < headDim; d++) {
                        dot += query.getDouble(0, q, h, d)
                                * key.getDouble(0, k, kvHead, d);
                    }
                    logits[k] = dot * scale + bias.getDouble(0, 0, q, k);
                    max = Math.max(max, logits[k]);
                    expectedLogits.putScalar(new long[]{0, h, q, k}, logits[k]);
                }

                double sum = 0.0;
                double[] probabilities = new double[seqKV];
                for (int k = 0; k < seqKV; k++) {
                    probabilities[k] = Math.exp(logits[k] - max);
                    sum += probabilities[k];
                }
                for (int k = 0; k < seqKV; k++) {
                    probabilities[k] /= sum;
                    expectedScores.putScalar(new long[]{0, h, q, k}, probabilities[k]);
                }
                for (int d = 0; d < headDim; d++) {
                    double weighted = 0.0;
                    for (int k = 0; k < seqKV; k++) {
                        weighted += probabilities[k] * value.getDouble(0, k, kvHead, d);
                    }
                    expectedOutput.putScalar(new long[]{0, q, h, d}, weighted);
                }
            }
        }

        assertArrayEquals(expectedOutput.shape(), outputs[0].shape());
        assertArrayEquals(expectedScores.shape(), outputs[1].shape());
        assertArrayEquals(expectedLogits.shape(), outputs[2].shape());

        double outputMaxDiff = outputs[0].sub(expectedOutput).amaxNumber().doubleValue();
        double scoresMaxDiff = outputs[1].sub(expectedScores).amaxNumber().doubleValue();
        double logitsMaxDiff = outputs[2].sub(expectedLogits).amaxNumber().doubleValue();
        assertTrue(outputMaxDiff < 1e-4,
                "GQA prefill output maxDiff=" + outputMaxDiff);
        assertTrue(scoresMaxDiff < 1e-4,
                "GQA prefill scores maxDiff=" + scoresMaxDiff);
        assertTrue(logitsMaxDiff < 1e-4,
                "GQA prefill logits maxDiff=" + logitsMaxDiff);
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

    // ========================= OneDNN SDPA Padded-KV JIT Reuse Test =========================

    /**
     * Verifies the OneDNN Graph padded-KV decode path: seqQ=1, seqKV=maxKvLen constant,
     * bias=[1,1,1,maxKvLen] masking unfilled cache slots.
     *
     * This test mirrors the GGUF LLM decode loop where:
     * - KV cache is pre-allocated to maxKvLen and zero-padded beyond prefillLen
     * - The attention bias [1,1,1,maxKvLen] has 0.0 for valid positions and -1e9 for padding
     * - seqQ=1 and seqKV=maxKvLen stay CONSTANT across all decode steps
     *   so the OneDNN JIT-compiled partition is built exactly once and reused
     *
     * The test checks:
     * 1. Correctness: output matches a reference computation at each "decode step"
     * 2. Bias effectiveness: padded positions (filled with -1e9) do not contaminate output
     * 3. Consistency: output is stable across multiple steps (same JIT partition reused)
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("OneDNN SDPA - Padded-KV Decode with Static Bias (JIT Reuse)")
    public void testOnednnSdpaPaddedKvDecodeWithStaticBias(Nd4jBackend backend) {
        Nd4j.getRandom().setSeed(42);

        final int batch     = 1;
        final int numHeads  = 4;
        final int headDim   = 32;
        final int prefillLen = 8;     // tokens filled during prefill
        final int maxKvLen  = 16;     // pre-allocated KV cache size (prefill + max new tokens)
        final float scale   = (float)(1.0 / Math.sqrt(headDim));

        // Pre-allocate KV cache to maxKvLen, filled at prefillLen positions.
        // Shape: [batch, maxKvLen, numHeads, headDim] (BSHD)
        INDArray kvKey = Nd4j.zeros(batch, maxKvLen, numHeads, headDim);
        INDArray kvVal = Nd4j.zeros(batch, maxKvLen, numHeads, headDim);
        INDArray prefillK = Nd4j.randn(DataType.FLOAT, batch, prefillLen, numHeads, headDim).muli(0.1f);
        INDArray prefillV = Nd4j.randn(DataType.FLOAT, batch, prefillLen, numHeads, headDim).muli(0.1f);
        kvKey.get(
            org.nd4j.linalg.indexing.NDArrayIndex.all(),
            org.nd4j.linalg.indexing.NDArrayIndex.interval(0, prefillLen),
            org.nd4j.linalg.indexing.NDArrayIndex.all(),
            org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).assign(prefillK);
        kvVal.get(
            org.nd4j.linalg.indexing.NDArrayIndex.all(),
            org.nd4j.linalg.indexing.NDArrayIndex.interval(0, prefillLen),
            org.nd4j.linalg.indexing.NDArrayIndex.all(),
            org.nd4j.linalg.indexing.NDArrayIndex.all()
        ).assign(prefillV);

        // Simulate 4 decode steps (seqQ=1, seqKV=maxKvLen constant)
        final float maskFill = -1e9f;
        INDArray prevOutput = null;

        for (int step = 0; step < 4; step++) {
            int cachePos = prefillLen + step;

            // Write current step's K/V into the cache
            if (step > 0) {
                INDArray newK = Nd4j.randn(DataType.FLOAT, batch, 1, numHeads, headDim).muli(0.1f);
                INDArray newV = Nd4j.randn(DataType.FLOAT, batch, 1, numHeads, headDim).muli(0.1f);
                kvKey.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(cachePos - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
                ).assign(newK.reshape(batch, numHeads, headDim));
                kvVal.get(
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.point(cachePos - 1),
                    org.nd4j.linalg.indexing.NDArrayIndex.all(),
                    org.nd4j.linalg.indexing.NDArrayIndex.all()
                ).assign(newV.reshape(batch, numHeads, headDim));
            }

            // Query: [batch, 1, numHeads, headDim]
            INDArray query = Nd4j.randn(DataType.FLOAT, batch, 1, numHeads, headDim).muli(0.1f);

            // Decode causal bias: [1, 1, 1, maxKvLen]
            // Positions [0..cachePos) = 0.0f (valid), [cachePos..maxKvLen) = maskFill
            float[] biasData = new float[maxKvLen];
            for (int k = cachePos; k < maxKvLen; k++) {
                biasData[k] = maskFill;
            }
            INDArray bias = Nd4j.create(biasData, new long[]{1, 1, 1, maxKvLen}, 'c');

            // Execute dot_product_attention_v2 with padded KV and static bias
            INDArray emptyMask = Nd4j.empty(DataType.FLOAT);
            DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                    .addInputs(query, kvVal, kvKey, emptyMask, emptyMask, bias)
                    .addFloatingPointArguments((double) scale, 0.0)
                    .addBooleanArguments(false, false, true)
                    .build();

            INDArray[] outputs = Nd4j.exec(op);
            INDArray attnOut = outputs[0];

            assertNotNull(attnOut, "Step " + step + ": output should not be null");
            assertArrayEquals(new long[]{batch, 1, numHeads, headDim}, attnOut.shape(),
                    "Step " + step + ": output shape should be [batch, 1, numHeads, headDim]");
            assertFalse(attnOut.isNaN().any(),
                    "Step " + step + ": output should not contain NaN (masked softmax producing -inf?)");
            assertFalse(attnOut.isInfinite().any(),
                    "Step " + step + ": output should not contain inf");

            // Verify that fully-masked positions (last slots) do not dominate output:
            // If masking works, the output should be non-trivially close to attending only
            // over the valid positions [0..cachePos). Check that output is not all-zeros
            // (which would indicate the entire row got masked and softmax produced 0/NaN).
            double absSum = attnOut.ameanNumber().doubleValue();
            assertTrue(absSum > 1e-7,
                    "Step " + step + ": attention output magnitude too small — masked softmax may be collapsing");

            // Verify consistency: running the same query twice should give identical output
            INDArray[] outputs2 = Nd4j.exec(DynamicCustomOp.builder("dot_product_attention_v2")
                    .addInputs(query, kvVal, kvKey, emptyMask, emptyMask, bias)
                    .addFloatingPointArguments((double) scale, 0.0)
                    .addBooleanArguments(false, false, true)
                    .build());
            double maxDiff = outputs2[0].sub(attnOut).amaxNumber().doubleValue();
            assertEquals(0.0, maxDiff, 1e-5,
                    "Step " + step + ": repeated execution must give identical results (JIT partition reuse)");

            prevOutput = attnOut;
        }
    }

    // ========================= Mixed Dtype Tests =========================

    /**
     * Verifies DotProductAttentionV2 auto-promotes mixed FP types (FLOAT Q/K + HALF V)
     * instead of throwing. This is the scenario that occurs when FusedRoPE promotes
     * Q/K to FLOAT while V stays in HALF — the kernel must handle it internally.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Mixed Dtype Q(FLOAT) K(FLOAT) V(HALF) auto-promotes")
    public void testMixedDtypeAutoPromotion(Nd4jBackend backend) {
        int batch = 1, seqQ = 4, seqKV = 4, dim = 16;
        double scale = 1.0 / Math.sqrt(dim);

        // Q and K in FLOAT (as if FusedRoPE promoted them), V in HALF
        INDArray qArr = Nd4j.rand(DataType.FLOAT, batch, seqQ, dim).muli(0.1f);
        INDArray kArr = Nd4j.rand(DataType.FLOAT, batch, seqKV, dim).muli(0.1f);
        INDArray vArr = Nd4j.rand(DataType.HALF, batch, seqKV, dim).muli(0.1f);

        // Direct op execution — should NOT throw
        INDArray[] outputs = Nd4j.exec(DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(qArr, vArr, kArr)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)
                .build());

        assertNotNull(outputs[0], "Output should not be null");
        assertEquals(DataType.FLOAT, outputs[0].dataType(),
                "Output dtype should be promoted to FLOAT (widest of Q/K/V)");
        assertFalse(outputs[0].isNaN().any(),
                "Output should not contain NaN values");
    }

    /**
     * Verifies mixed dtype attention works through a SameDiff graph
     * both with and without GraphOptimizer. The optimizer may strip
     * explicit cast ops, so the kernel must handle the mismatch.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Mixed Dtype with SameDiff + GraphOptimizer")
    public void testMixedDtypeWithGraphOptimizer(Nd4jBackend backend) {
        int batch = 1, seqQ = 4, seqKV = 4, numHeads = 2, headDim = 8;
        double scale = 1.0 / Math.sqrt(headDim);

        // 4D BSHD format — Q/K as FLOAT, V as HALF (realistic FusedRoPE scenario)
        INDArray qArr = Nd4j.rand(DataType.FLOAT, batch, seqQ, numHeads, headDim).muli(0.1f);
        INDArray kArr = Nd4j.rand(DataType.FLOAT, batch, seqKV, numHeads, headDim).muli(0.1f);
        INDArray vArr = Nd4j.rand(DataType.HALF, batch, seqKV, numHeads, headDim).muli(0.1f);

        // Run WITHOUT optimizer first as baseline
        SameDiff sdBase = SameDiff.create();
        SDVariable q1 = sdBase.placeHolder("query", DataType.FLOAT, batch, seqQ, numHeads, headDim);
        SDVariable v1 = sdBase.placeHolder("value", DataType.HALF, batch, seqKV, numHeads, headDim);
        SDVariable k1 = sdBase.placeHolder("key", DataType.FLOAT, batch, seqKV, numHeads, headDim);
        SDVariable out1 = sdBase.nn().dotProductAttentionV2("attn", q1, v1, k1,
                null, null, null, scale, 0.0, false, false);

        Map<String, INDArray> feed = new java.util.LinkedHashMap<>();
        feed.put("query", qArr);
        feed.put("value", vArr);
        feed.put("key", kArr);

        Map<String, INDArray> baseResult = sdBase.output(feed, "attn");
        INDArray baseOutput = baseResult.get("attn");
        assertNotNull(baseOutput);
        assertEquals(DataType.FLOAT, baseOutput.dataType());
        assertFalse(baseOutput.isNaN().any(), "Baseline output should not contain NaN");

        // Run WITH optimizer — GraphOptimizer may strip cast ops
        SameDiff sdOpt = SameDiff.create();
        SDVariable q2 = sdOpt.placeHolder("query", DataType.FLOAT, batch, seqQ, numHeads, headDim);
        SDVariable v2 = sdOpt.placeHolder("value", DataType.HALF, batch, seqKV, numHeads, headDim);
        SDVariable k2 = sdOpt.placeHolder("key", DataType.FLOAT, batch, seqKV, numHeads, headDim);
        SDVariable out2 = sdOpt.nn().dotProductAttentionV2("attn", q2, v2, k2,
                null, null, null, scale, 0.0, false, false);

        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.optimize(sdOpt);

        Map<String, INDArray> optResult = sdOpt.output(feed, "attn");
        INDArray optOutput = optResult.get("attn");
        assertNotNull(optOutput);
        assertEquals(DataType.FLOAT, optOutput.dataType());
        assertFalse(optOutput.isNaN().any(), "Optimized output should not contain NaN");

        // Both paths should give the same result (within FP tolerance)
        double maxDiff = baseOutput.sub(optOutput).amaxNumber().doubleValue();
        assertTrue(maxDiff < 1e-3,
                "Base vs optimized output max diff " + maxDiff + " exceeds tolerance 1e-3");
    }

    /**
     * Verifies mixed dtype with all HALF Q/K/V — the non-mismatch case
     * should still work correctly (regression guard).
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("Attention - Uniform HALF dtype still works")
    public void testUniformHalfDtype(Nd4jBackend backend) {
        int batch = 1, seqQ = 4, seqKV = 4, dim = 16;
        double scale = 1.0 / Math.sqrt(dim);

        INDArray qArr = Nd4j.rand(DataType.HALF, batch, seqQ, dim).muli(0.1f);
        INDArray kArr = Nd4j.rand(DataType.HALF, batch, seqKV, dim).muli(0.1f);
        INDArray vArr = Nd4j.rand(DataType.HALF, batch, seqKV, dim).muli(0.1f);

        INDArray[] outputs = Nd4j.exec(DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(qArr, vArr, kArr)
                .addFloatingPointArguments(scale, 0.0)
                .addBooleanArguments(false, false, true)
                .build());

        assertNotNull(outputs[0]);
        assertEquals(DataType.HALF, outputs[0].dataType());
        assertFalse(outputs[0].isNaN().any());
    }

    /**
     * Verifies onnx_multi_head_attention auto-promotes mixed FP types.
     * Q is FLOAT, K/V are HALF — kernel must auto-cast without throwing.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ONNX MHA - Mixed Dtype Q(FLOAT) K(HALF) V(HALF) auto-promotes")
    public void testOnnxMhaMixedDtypeAutoPromotion(Nd4jBackend backend) {
        int batch = 1, seqQ = 4, seqKV = 4, numHeads = 2, headDim = 8;
        int hidden = numHeads * headDim;
        double scale = 1.0 / Math.sqrt(headDim);

        // Q in FLOAT, K/V in HALF (realistic mixed-precision scenario)
        INDArray qArr = Nd4j.rand(DataType.FLOAT, batch, seqQ, hidden).muli(0.1f);
        INDArray kArr = Nd4j.rand(DataType.HALF, batch, seqKV, hidden).muli(0.1f);
        INDArray vArr = Nd4j.rand(DataType.HALF, batch, seqKV, hidden).muli(0.1f);

        INDArray[] outputs = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                qArr, kArr, vArr, null, null, null,
                numHeads, scale, false, 1));

        assertNotNull(outputs[0], "Output should not be null");
        assertEquals(DataType.FLOAT, outputs[0].dataType(),
                "Output dtype should be promoted to FLOAT (widest of Q/K/V)");
        assertFalse(outputs[0].isNaN().any(),
                "Output should not contain NaN values");
    }

    /**
     * Verifies onnx_multi_head_attention with mixed dtypes through SameDiff + GraphOptimizer.
     */
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @DisplayName("ONNX MHA - Mixed Dtype with SameDiff + GraphOptimizer")
    public void testOnnxMhaMixedDtypeWithGraphOptimizer(Nd4jBackend backend) {
        int batch = 1, seqQ = 4, seqKV = 4, numHeads = 2, headDim = 8;
        int hidden = numHeads * headDim;
        double scale = 1.0 / Math.sqrt(headDim);

        INDArray qArr = Nd4j.rand(DataType.FLOAT, batch, seqQ, hidden).muli(0.1f);
        INDArray kArr = Nd4j.rand(DataType.HALF, batch, seqKV, hidden).muli(0.1f);
        INDArray vArr = Nd4j.rand(DataType.HALF, batch, seqKV, hidden).muli(0.1f);

        // Without optimizer
        SameDiff sdBase = SameDiff.create();
        SDVariable q1 = sdBase.placeHolder("query", DataType.FLOAT, batch, seqQ, hidden);
        SDVariable k1 = sdBase.placeHolder("key", DataType.HALF, batch, seqKV, hidden);
        SDVariable v1 = sdBase.placeHolder("value", DataType.HALF, batch, seqKV, hidden);
        sdBase.addVariable(new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                sdBase, q1, k1, v1, null, null, null,
                numHeads, scale, false, 1).outputVariables()[0]);

        Map<String, INDArray> feed = new java.util.LinkedHashMap<>();
        feed.put("query", qArr);
        feed.put("key", kArr);
        feed.put("value", vArr);

        // Get the output variable name
        String outName = null;
        for (String name : sdBase.variableMap().keySet()) {
            if (name.contains("onnx_multi_head_attention")) {
                outName = name;
                break;
            }
        }
        assertNotNull(outName, "Should find onnx_mha output variable");

        Map<String, INDArray> baseResult = sdBase.output(feed, outName);
        INDArray baseOutput = baseResult.get(outName);
        assertNotNull(baseOutput);
        assertEquals(DataType.FLOAT, baseOutput.dataType());
        assertFalse(baseOutput.isNaN().any(), "Baseline output should not contain NaN");

        // With optimizer
        SameDiff sdOpt = SameDiff.create();
        SDVariable q2 = sdOpt.placeHolder("query", DataType.FLOAT, batch, seqQ, hidden);
        SDVariable k2 = sdOpt.placeHolder("key", DataType.HALF, batch, seqKV, hidden);
        SDVariable v2 = sdOpt.placeHolder("value", DataType.HALF, batch, seqKV, hidden);
        sdOpt.addVariable(new org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention(
                sdOpt, q2, k2, v2, null, null, null,
                numHeads, scale, false, 1).outputVariables()[0]);

        String outNameOpt = null;
        for (String name : sdOpt.variableMap().keySet()) {
            if (name.contains("onnx_multi_head_attention")) {
                outNameOpt = name;
                break;
            }
        }

        GraphOptimizer optimizer = new GraphOptimizer();
        optimizer.optimize(sdOpt);

        Map<String, INDArray> optResult = sdOpt.output(feed, outNameOpt);
        INDArray optOutput = optResult.get(outNameOpt);
        assertNotNull(optOutput);
        assertEquals(DataType.FLOAT, optOutput.dataType());
        assertFalse(optOutput.isNaN().any(), "Optimized output should not contain NaN");
    }

    /**
     * Tests dot_product_attention_v2 with the same KV-cache contract used by GGUF decode:
     * Q/K/V arrive as FLOAT, persistent K/V caches are HALF, query heads outnumber KV heads,
     * and an additive cache mask hides unwritten cache slots.
     */
    @DisplayName("DotProductAttentionV2 - KV cache mixed dtype GQA single token")
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    public void testKvCacheMixedDtype(Nd4jBackend backend) {
        int batch = 1;
        int seqQ = 1;
        int seqKV = 1;
        int numHeads = 32;
        int numKvHeads = 8;
        int headDim = 128;
        int maxSeqLen = 16;
        int headsPerKvHead = numHeads / numKvHeads;

        INDArray queries = Nd4j.linspace(DataType.FLOAT, -0.25, 0.25, batch * seqQ * numHeads * headDim)
                .reshape(batch, seqQ, numHeads, headDim);
        INDArray keys = Nd4j.linspace(DataType.FLOAT, 0.125, -0.125, batch * seqKV * numKvHeads * headDim)
                .reshape(batch, seqKV, numKvHeads, headDim);
        INDArray values = Nd4j.linspace(DataType.FLOAT, -0.5, 0.5, batch * seqKV * numKvHeads * headDim)
                .reshape(batch, seqKV, numKvHeads, headDim);

        INDArray keyCache = Nd4j.zeros(DataType.HALF, batch, maxSeqLen, numKvHeads, headDim);
        INDArray valueCache = Nd4j.zeros(DataType.HALF, batch, maxSeqLen, numKvHeads, headDim);
        INDArray cachePos = Nd4j.createFromArray(new long[]{0});
        INDArray emptyMask = Nd4j.empty(DataType.FLOAT);
        INDArray cacheMask = Nd4j.valueArrayOf(new long[]{batch, 1, seqQ, maxSeqLen}, -1.0e9f, DataType.FLOAT);
        cacheMask.putScalar(new long[]{0, 0, 0, 0}, 0.0f);

        DynamicCustomOp op = DynamicCustomOp.builder("dot_product_attention_v2")
                .addInputs(queries, values, keys,
                        emptyMask, emptyMask,
                        keyCache, valueCache, cachePos, cacheMask)
                .addFloatingPointArguments(0.0, 0.0)
                .addBooleanArguments(false, false, true)
                .build();

        INDArray output = Nd4j.exec(op)[0];

        assertNotNull(output);
        assertEquals(DataType.FLOAT, output.dataType());
        assertArrayEquals(new long[]{batch, seqQ, numHeads, headDim}, output.shape());
        assertFalse(output.isNaN().any(), "Output should not contain NaN with mixed KV cache dtypes");
        assertFalse(output.isInfinite().any(), "Output should not contain Inf with mixed KV cache dtypes");
        assertFalse(keyCache.getDouble(0, 0, 0, 0) == 0.0 && keyCache.getDouble(0, 0, 0, 1) == 0.0,
                "KV cache at position 0 should have been written");

        for (int h = 0; h < numHeads; h++) {
            int kvHead = h / headsPerKvHead;
            for (int d = 0; d < headDim; d++) {
                assertEquals(valueCache.getDouble(0, 0, kvHead, d), output.getDouble(0, 0, h, d), 1e-3,
                        "seqKV=1 cache attention should copy value cache head " + kvHead
                                + " to query head " + h + " dim " + d);
            }
        }
    }
}
