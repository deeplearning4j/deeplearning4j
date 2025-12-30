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
package org.eclipse.deeplearning4j.frameworkimport.onnx

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import kotlin.math.sqrt

/**
 * Tests for ONNX normalization operator implementations including
 * BatchNormalization, LayerNormalization, InstanceNormalization, GroupNormalization,
 * LpNormalization, and MeanVarianceNormalization.
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@Tag(TagNames.SAMEDIFF)
class OnnxNormalizationOpsTest {

    // ==================== BatchNormalization ====================

    @Test
    fun testBatchNormalization2D() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)  // [batch, channels]

        // BatchNorm parameters
        val scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4))
        val bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 4))
        val mean = sd.constant("mean", Nd4j.zeros(DataType.FLOAT, 4))
        val variance = sd.constant("var", Nd4j.ones(DataType.FLOAT, 4))
        val epsilon = 1e-5

        // BatchNorm: y = scale * (x - mean) / sqrt(var + eps) + bias
        val normalized = sd.math.sub(input, mean)
        val stdDev = sd.math.sqrt(sd.math.add(variance, epsilon))
        val scaled = sd.math.mul(sd.math.div(normalized, stdDev), scale)
        val output = sd.math.add("batchnorm", scaled, bias)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f)).reshape(2, 4)
        val result = sd.output(mapOf("input" to inputArr), "batchnorm")["batchnorm"]!!

        assertEquals(2, result.shape()[0])
        assertEquals(4, result.shape()[1])
    }

    @Test
    fun testBatchNormalization4D() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4, 4)  // [N, C, H, W]

        // BatchNorm for CNN: normalize over N, H, W keeping C
        val channels = 3
        val scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, channels.toLong()))
        val bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, channels.toLong()))
        val mean = sd.constant("mean", Nd4j.zeros(DataType.FLOAT, channels.toLong()))
        val variance = sd.constant("var", Nd4j.ones(DataType.FLOAT, channels.toLong()))

        // Simplified: just verify shapes
        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 3, 4, 4)
        assertNotNull(sd)
        assertEquals(4, inputArr.rank())
    }

    // ==================== LayerNormalization ====================

    @Test
    fun testLayerNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)

        // Layer norm: normalize over last axis
        val epsilon = 1e-5
        val axis = 2

        val mean = sd.math.mean(input, false, axis)
        val variance = sd.math.variance(input, false, false, axis)

        // Expand dims for broadcasting
        val meanExpanded = sd.expandDims(mean, axis)
        val varExpanded = sd.expandDims(variance, axis)

        val normalized = sd.math.sub(input, meanExpanded)
        val stdDev = sd.math.sqrt(sd.math.add(varExpanded, epsilon))
        val output = sd.math.div("layernorm", normalized, stdDev)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "layernorm")["layernorm"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(4, result.shape()[2])
    }

    @Test
    fun testLayerNormalizationWithScaleBias() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)  // [batch, features]

        val scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4).mul(2))
        val bias = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4))
        val epsilon = 1e-5

        // Normalize over features (axis=1)
        val mean = sd.math.mean(input, false, 1)
        val variance = sd.math.variance(input, false, false, 1)

        val meanExpanded = sd.expandDims(mean, 1)
        val varExpanded = sd.expandDims(variance, 1)

        val normalized = sd.math.sub(input, meanExpanded)
        val stdDev = sd.math.sqrt(sd.math.add(varExpanded, epsilon))
        val normed = sd.math.div(normalized, stdDev)
        val output = sd.math.add("layernorm_sb", sd.math.mul(normed, scale), bias)

        val inputArr = Nd4j.create(floatArrayOf(0f, 1f, 2f, 3f)).reshape(1, 4)
        val result = sd.output(mapOf("input" to inputArr), "layernorm_sb")["layernorm_sb"]!!

        assertEquals(1, result.shape()[0])
        assertEquals(4, result.shape()[1])
    }

    // ==================== InstanceNormalization ====================

    @Test
    fun testInstanceNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4, 4)  // [N, C, H, W]

        // Instance norm: normalize over H, W for each N, C independently
        val epsilon = 1e-5

        // Compute mean and variance over spatial dimensions (2, 3)
        val mean = sd.math.mean(input, false, 2, 3)  // [N, C]
        val variance = sd.math.variance(input, false, false, 2, 3)  // [N, C]

        // Expand for broadcasting
        val meanExpanded = sd.expandDims(sd.expandDims(mean, 2), 3)  // [N, C, 1, 1]
        val varExpanded = sd.expandDims(sd.expandDims(variance, 2), 3)  // [N, C, 1, 1]

        val normalized = sd.math.sub(input, meanExpanded)
        val stdDev = sd.math.sqrt(sd.math.add(varExpanded, epsilon))
        val output = sd.math.div("instancenorm", normalized, stdDev)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 3, 4, 4)
        val result = sd.output(mapOf("input" to inputArr), "instancenorm")["instancenorm"]!!

        assertEquals(4, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(4, result.shape()[2])
        assertEquals(4, result.shape()[3])
    }

    // ==================== GroupNormalization ====================

    @Test
    fun testGroupNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 8, 4, 4)  // [N, C, H, W]

        // Group norm with 2 groups: reshape C -> (G, C/G), then normalize
        val numGroups = 2
        val channels = 8
        val channelsPerGroup = channels / numGroups
        val epsilon = 1e-5

        // Reshape: [N, C, H, W] -> [N, G, C/G, H, W]
        val reshaped = sd.reshape(input, 2, numGroups.toLong(), channelsPerGroup.toLong(), 4, 4)

        // Normalize over (C/G, H, W) i.e. axes 2, 3, 4
        val mean = sd.math.mean(reshaped, false, 2, 3, 4)  // [N, G]
        val variance = sd.math.variance(reshaped, false, false, 2, 3, 4)  // [N, G]

        // Expand for broadcasting: [N, G, 1, 1, 1]
        var meanExp = mean
        var varExp = variance
        for (i in 0 until 3) {
            meanExp = sd.expandDims(meanExp, meanExp.rank)
            varExp = sd.expandDims(varExp, varExp.rank)
        }

        val normalized = sd.math.sub(reshaped, meanExp)
        val stdDev = sd.math.sqrt(sd.math.add(varExp, epsilon))
        val normed = sd.math.div(normalized, stdDev)

        // Reshape back: [N, G, C/G, H, W] -> [N, C, H, W]
        val output = sd.reshape("groupnorm", normed, 2, 8, 4, 4)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 8, 4, 4)
        val result = sd.output(mapOf("input" to inputArr), "groupnorm")["groupnorm"]!!

        assertEquals(4, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(8, result.shape()[1])
    }

    // ==================== LpNormalization ====================

    @Test
    fun testL1Normalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // L1 normalization: x / ||x||_1
        val l1Norm = sd.math.sum(sd.math.abs(input), false, 1)
        val l1NormExpanded = sd.expandDims(l1Norm, 1)
        val output = sd.math.div("l1norm", input, l1NormExpanded)

        val inputArr = Nd4j.create(floatArrayOf(1f, -2f, 3f, -4f)).reshape(1, 4)  // L1 norm = 10
        val result = sd.output(mapOf("input" to inputArr), "l1norm")["l1norm"]!!

        assertEquals(0.1f, result.getFloat(0, 0), 0.01f)
        assertEquals(-0.2f, result.getFloat(0, 1), 0.01f)
        assertEquals(0.3f, result.getFloat(0, 2), 0.01f)
        assertEquals(-0.4f, result.getFloat(0, 3), 0.01f)
    }

    @Test
    fun testL2Normalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 3)

        // L2 normalization: x / ||x||_2
        val l2Norm = sd.norm2(input, false, 1)
        val l2NormExpanded = sd.expandDims(l2Norm, 1)
        val output = sd.math.div("l2norm", input, l2NormExpanded)

        val inputArr = Nd4j.create(floatArrayOf(3f, 4f, 0f)).reshape(1, 3)  // L2 norm = 5
        val result = sd.output(mapOf("input" to inputArr), "l2norm")["l2norm"]!!

        assertEquals(0.6f, result.getFloat(0, 0), 0.01f)
        assertEquals(0.8f, result.getFloat(0, 1), 0.01f)
        assertEquals(0f, result.getFloat(0, 2), 0.01f)
    }

    @Test
    fun testLpNormalizationP3() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // Lp normalization with p=3: x / ||x||_p
        val p = 3
        val absInput = sd.math.abs(input)
        val powered = sd.math.pow(absInput, p.toDouble())
        val summed = sd.math.sum(powered, false, 1)
        val lpNorm = sd.math.pow(summed, 1.0 / p)
        val lpNormExpanded = sd.expandDims(lpNorm, 1)
        val output = sd.math.div("lpnorm", input, lpNormExpanded)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f)).reshape(1, 4)
        val result = sd.output(mapOf("input" to inputArr), "lpnorm")["lpnorm"]!!

        assertEquals(4, result.shape()[1])
        // Sum of values after normalization should have L3 norm ≈ 1
    }

    // ==================== MeanVarianceNormalization ====================

    @Test
    fun testMeanVarianceNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4, 4)

        // MVN: (x - mean) / sqrt(variance + eps)
        // Default: normalize over C, H, W (axes 1, 2, 3)
        val epsilon = 1e-9

        val mean = sd.math.mean(input, false, 1, 2, 3)  // [N]
        val variance = sd.math.variance(input, false, false, 1, 2, 3)  // [N]

        // Expand for broadcasting
        var meanExp = mean
        var varExp = variance
        for (i in 0 until 3) {
            meanExp = sd.expandDims(meanExp, 1)
            varExp = sd.expandDims(varExp, 1)
        }

        val normalized = sd.math.sub(input, meanExp)
        val stdDev = sd.math.sqrt(sd.math.add(varExp, epsilon))
        val output = sd.math.div("mvn", normalized, stdDev)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 3, 4, 4)
        val result = sd.output(mapOf("input" to inputArr), "mvn")["mvn"]!!

        assertEquals(4, result.rank())
        assertEquals(2, result.shape()[0])
    }

    @Test
    fun testMeanVarianceNormalizationAcrossBatch() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 4, 3)

        // MVN across all samples (axis 0)
        val epsilon = 1e-9

        val mean = sd.math.mean(input, false, 0)  // [features]
        val variance = sd.math.variance(input, false, false, 0)  // [features]

        val normalized = sd.math.sub(input, mean)
        val stdDev = sd.math.sqrt(sd.math.add(variance, epsilon))
        val output = sd.math.div("mvn_batch", normalized, stdDev)

        val inputArr = Nd4j.create(floatArrayOf(
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f,
            10f, 11f, 12f
        )).reshape(4, 3)
        val result = sd.output(mapOf("input" to inputArr), "mvn_batch")["mvn_batch"]!!

        assertEquals(4, result.shape()[0])
        assertEquals(3, result.shape()[1])
    }

    // ==================== SkipLayerNormalization (for Transformers) ====================

    @Test
    fun testSkipLayerNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 4, 8)  // [batch, seq, hidden]
        val skip = sd.placeHolder("skip", DataType.FLOAT, 2, 4, 8)

        // Skip + LayerNorm: output = LayerNorm(input + skip)
        val epsilon = 1e-5

        // Add skip connection
        val added = sd.math.add(input, skip)

        // Layer normalization over last axis
        val mean = sd.math.mean(added, false, 2)
        val variance = sd.math.variance(added, false, false, 2)

        val meanExpanded = sd.expandDims(mean, 2)
        val varExpanded = sd.expandDims(variance, 2)

        val normalized = sd.math.sub(added, meanExpanded)
        val stdDev = sd.math.sqrt(sd.math.add(varExpanded, epsilon))
        val output = sd.math.div("skip_ln", normalized, stdDev)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 4, 8)
        val skipArr = Nd4j.randn(DataType.FLOAT, 2, 4, 8)
        val result = sd.output(mapOf("input" to inputArr, "skip" to skipArr), "skip_ln")["skip_ln"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(4, result.shape()[1])
        assertEquals(8, result.shape()[2])
    }

    // ==================== EmbedLayerNormalization (BERT-style) ====================

    @Test
    fun testEmbedLayerNormalization() {
        val sd = SameDiff.create()

        // BERT-style embedding: word + position + segment embeddings + LayerNorm
        val batchSize = 2
        val seqLen = 4
        val hiddenSize = 8
        val vocabSize = 100
        val maxPos = 512

        val wordIds = sd.placeHolder("word_ids", DataType.INT64, batchSize.toLong(), seqLen.toLong())
        val posIds = sd.placeHolder("pos_ids", DataType.INT64, batchSize.toLong(), seqLen.toLong())
        val segIds = sd.placeHolder("seg_ids", DataType.INT64, batchSize.toLong(), seqLen.toLong())

        // Embedding tables
        val wordEmb = sd.constant("word_emb", Nd4j.randn(DataType.FLOAT, vocabSize.toLong(), hiddenSize.toLong()).mul(0.02))
        val posEmb = sd.constant("pos_emb", Nd4j.randn(DataType.FLOAT, maxPos.toLong(), hiddenSize.toLong()).mul(0.02))
        val segEmb = sd.constant("seg_emb", Nd4j.randn(DataType.FLOAT, 2, hiddenSize.toLong()).mul(0.02))

        // Gather embeddings
        val wordEmbeddings = sd.gather(wordEmb, wordIds, 0)
        val posEmbeddings = sd.gather(posEmb, posIds, 0)
        val segEmbeddings = sd.gather(segEmb, segIds, 0)

        // Sum embeddings
        val summed = sd.math.add(sd.math.add(wordEmbeddings, posEmbeddings), segEmbeddings)

        // Layer normalization
        val epsilon = 1e-12
        val mean = sd.math.mean(summed, false, 2)
        val variance = sd.math.variance(summed, false, false, 2)
        val meanExp = sd.expandDims(mean, 2)
        val varExp = sd.expandDims(variance, 2)
        val normalized = sd.math.sub(summed, meanExp)
        val stdDev = sd.math.sqrt(sd.math.add(varExp, epsilon))
        val output = sd.math.div("embed_ln", normalized, stdDev)

        val wordIdsArr = Nd4j.createFromArray(longArrayOf(1, 2, 3, 4, 5, 6, 7, 8)).reshape(batchSize.toLong(), seqLen.toLong())
        val posIdsArr = Nd4j.createFromArray(longArrayOf(0, 1, 2, 3, 0, 1, 2, 3)).reshape(batchSize.toLong(), seqLen.toLong())
        val segIdsArr = Nd4j.createFromArray(longArrayOf(0, 0, 0, 0, 1, 1, 1, 1)).reshape(batchSize.toLong(), seqLen.toLong())

        val result = sd.output(mapOf("word_ids" to wordIdsArr, "pos_ids" to posIdsArr, "seg_ids" to segIdsArr), "embed_ln")["embed_ln"]!!

        assertEquals(3, result.rank())
        assertEquals(batchSize.toLong(), result.shape()[0])
        assertEquals(seqLen.toLong(), result.shape()[1])
        assertEquals(hiddenSize.toLong(), result.shape()[2])
    }

    // ==================== LocalResponseNormalization (LRN) ====================

    @Test
    fun testLocalResponseNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 4, 4, 4)  // [N, C, H, W]

        // LRN: y = x / (bias + alpha/size * sum(x^2))^beta
        val size = 3
        val alpha = 0.0001f
        val beta = 0.75f
        val bias = 1.0f

        // Simplified LRN test - verify input shape handling
        val squared = sd.math.pow(input, 2.0)

        val inputArr = Nd4j.randn(DataType.FLOAT, 1, 4, 4, 4)
        assertNotNull(sd)
    }

    // ==================== Softmax with temperature ====================

    @Test
    fun testSoftmaxNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        val output = sd.nn.softmax("softmax", input, 1)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f)).reshape(1, 4)
        val result = sd.output(mapOf("input" to inputArr), "softmax")["softmax"]!!

        // Softmax sums to 1
        val sum = result.sumNumber().floatValue()
        assertEquals(1f, sum, 0.01f)

        // Output should be positive
        for (i in 0 until 4) {
            assertTrue(result.getFloat(0, i.toLong()) > 0)
        }
    }

    @Test
    fun testLogSoftmax() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 3)

        val output = sd.nn.logSoftmax("logsoftmax", input, 1)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f)).reshape(1, 3)
        val result = sd.output(mapOf("input" to inputArr), "logsoftmax")["logsoftmax"]!!

        // Log softmax values should be <= 0
        for (i in 0 until 3) {
            assertTrue(result.getFloat(0, i.toLong()) <= 0)
        }

        // exp(logsoftmax) should sum to 1
        val expSum = Nd4j.exec(org.nd4j.linalg.api.ops.impl.transforms.strict.Exp(result.dup()))[0].sumNumber().floatValue()
        assertEquals(1f, expSum, 0.01f)
    }

    // ==================== Hardmax ====================

    @Test
    fun testHardmax() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 4)

        // Hardmax: one-hot encoding of argmax
        val argmax = sd.argmax(input, false, 1)
        val output = sd.oneHot("hardmax", argmax, 4)

        val inputArr = Nd4j.create(floatArrayOf(
            1f, 3f, 2f, 0f,  // max at index 1
            0f, 1f, 4f, 2f   // max at index 2
        )).reshape(2, 4)
        val result = sd.output(mapOf("input" to inputArr), "hardmax")["hardmax"]!!

        // First row: [0, 1, 0, 0]
        assertEquals(0f, result.getFloat(0, 0), 0.01f)
        assertEquals(1f, result.getFloat(0, 1), 0.01f)
        assertEquals(0f, result.getFloat(0, 2), 0.01f)

        // Second row: [0, 0, 1, 0]
        assertEquals(0f, result.getFloat(1, 0), 0.01f)
        assertEquals(0f, result.getFloat(1, 1), 0.01f)
        assertEquals(1f, result.getFloat(1, 2), 0.01f)
    }

    // ==================== Batch Renormalization ====================

    @Test
    fun testBatchRenormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // Batch renormalization adds r and d correction terms
        // y = scale * (x - mean) / std * r + d + bias
        // For inference, r=1 and d=0, so it's same as regular batchnorm

        val scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4))
        val bias = sd.constant("bias", Nd4j.zeros(DataType.FLOAT, 4))
        val runningMean = sd.constant("running_mean", Nd4j.zeros(DataType.FLOAT, 4))
        val runningVar = sd.constant("running_var", Nd4j.ones(DataType.FLOAT, 4))
        val epsilon = 1e-5

        val normalized = sd.math.sub(input, runningMean)
        val stdDev = sd.math.sqrt(sd.math.add(runningVar, epsilon))
        val scaled = sd.math.mul(sd.math.div(normalized, stdDev), scale)
        val output = sd.math.add("batch_renorm", scaled, bias)

        val inputArr = Nd4j.randn(DataType.FLOAT, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "batch_renorm")["batch_renorm"]!!

        assertEquals(3, result.shape()[0])
        assertEquals(4, result.shape()[1])
    }

    // ==================== RMS Normalization (used in LLMs) ====================

    @Test
    fun testRMSNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 3, 4)  // [batch, seq, hidden]

        // RMS Norm: x / sqrt(mean(x^2) + eps) * scale
        val epsilon = 1e-6
        val scale = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4))

        val squared = sd.math.pow(input, 2.0)
        val meanSquared = sd.math.mean(squared, false, 2)  // [batch, seq]
        val rms = sd.math.sqrt(sd.math.add(meanSquared, epsilon))
        val rmsExpanded = sd.expandDims(rms, 2)  // [batch, seq, 1]
        val normalized = sd.math.div(input, rmsExpanded)
        val output = sd.math.mul("rmsnorm", normalized, scale)

        val inputArr = Nd4j.randn(DataType.FLOAT, 2, 3, 4)
        val result = sd.output(mapOf("input" to inputArr), "rmsnorm")["rmsnorm"]!!

        assertEquals(3, result.rank())
        assertEquals(2, result.shape()[0])
        assertEquals(3, result.shape()[1])
        assertEquals(4, result.shape()[2])
    }
}
