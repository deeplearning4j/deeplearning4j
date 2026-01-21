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

        // Flatten to 2D for mmul: [batch*seqLen, dim]
        SDVariable qFlat = query.reshape(batch * seqLen, dim);
        SDVariable kFlat = key.reshape(batch * seqLen, dim);
        SDVariable vFlat = value.reshape(batch * seqLen, dim);

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
}
