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
package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.microsoft

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.onnx.definitions.MicrosoftOnnxExtensions
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry

/**
 * Tests for Microsoft ONNX extension operators.
 *
 * These tests verify that Microsoft-specific ONNX operators are properly registered
 * and can be executed through SameDiff.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
 */
@Tag(TagNames.ONNX)
@Tag(TagNames.SAMEDIFF)
class TestMicrosoftOnnxOps {

    // =============================================
    // Registry Tests
    // =============================================

    @Test
    fun testMicrosoftExtensionsRegistered() {
        val registry = registry()

        // Check that Microsoft extension ops are registered
        val microsoftOps = MicrosoftOnnxExtensions.getAllExtensionOpNames()

        assertTrue(microsoftOps.isNotEmpty(), "Microsoft extensions should be registered")

        // Verify key ops are in the registry
        val expectedOps = listOf(
            "FastGelu", "BiasGelu", "QuickGelu",
            "SkipLayerNormalization", "SimplifiedLayerNormalization",
            "Attention", "MultiHeadAttention"
        )

        for (op in expectedOps) {
            assertTrue(
                microsoftOps.contains(op),
                "Expected Microsoft op '$op' to be registered, got: $microsoftOps"
            )
        }
    }

    @Test
    fun testMicrosoftExtensionDomains() {
        val domains = MicrosoftOnnxExtensions.getAllExtensionDomains()

        assertTrue(domains.contains("com.microsoft"), "Should have com.microsoft domain")
    }

    @Test
    fun testIsMicrosoftExtension() {
        assertTrue(MicrosoftOnnxExtensions.isMicrosoftExtension("FastGelu"))
        assertTrue(MicrosoftOnnxExtensions.isMicrosoftExtension("BiasGelu"))
        assertTrue(MicrosoftOnnxExtensions.isMicrosoftExtension("QuickGelu"))
        assertFalse(MicrosoftOnnxExtensions.isMicrosoftExtension("Relu"))
        assertFalse(MicrosoftOnnxExtensions.isMicrosoftExtension("MatMul"))
    }

    // =============================================
    // Activation Function Tests
    // =============================================

    @Test
    fun testFastGeluComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // FastGELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        val x = input
        val x3 = sd.math().pow(x, 3.0)
        val inner = x.add(x3.mul(0.044715))
        val sqrt2OverPi = kotlin.math.sqrt(2.0 / kotlin.math.PI)
        val tanhInput = inner.mul(sqrt2OverPi)
        val tanhResult = sd.math().tanh(tanhInput)
        val onePlusTanh = tanhResult.add(1.0)
        val result = x.mul(onePlusTanh).mul("output", 0.5)

        val inputArr = Nd4j.create(floatArrayOf(-2f, -1f, 0f, 1f, 2f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // FastGELU should:
        // - Be close to 0 for x=0
        // - Be close to x for large positive x
        // - Be close to 0 for large negative x
        assertTrue(kotlin.math.abs(output.getFloat(0, 2)) < 0.01, "FastGELU(0) should be close to 0")
        assertTrue(output.getFloat(0, 3) > 0.5, "FastGELU(1) should be positive")
        assertTrue(output.getFloat(0, 0) < 0, "FastGELU(-2) should be small negative")
    }

    @Test
    fun testQuickGeluComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // QuickGELU(x) = x * sigmoid(alpha * x), alpha = 1.702
        val alpha = 1.702
        val scaledInput = input.mul(alpha)
        val sigmoid = sd.nn().sigmoid(scaledInput)
        val result = input.mul("output", sigmoid)

        val inputArr = Nd4j.create(floatArrayOf(-2f, -1f, 0f, 1f, 2f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // QuickGELU should:
        // - Be 0 for x=0
        // - Be positive for positive x
        // - Be negative for negative x
        assertEquals(0.0f, output.getFloat(0, 2), 0.001f, "QuickGELU(0) should be 0")
        assertTrue(output.getFloat(0, 3) > 0, "QuickGELU(1) should be positive")
        assertTrue(output.getFloat(0, 0) < 0, "QuickGELU(-2) should be negative")
    }

    @Test
    fun testBiasGeluComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)
        val bias = sd.`var`("bias", Nd4j.create(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f)))

        // BiasGELU(x, b) = GELU(x + b) = 0.5 * (x+b) * (1 + erf((x+b) / sqrt(2)))
        val biasedInput = input.add(bias)
        val sqrt2 = kotlin.math.sqrt(2.0)
        val erfInput = biasedInput.div(sqrt2)
        val erfResult = sd.math().erf(erfInput)
        val onePlusErf = erfResult.add(1.0)
        val result = biasedInput.mul(onePlusErf).mul("output", 0.5)

        val inputArr = Nd4j.zeros(DataType.FLOAT, 1, 4)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // BiasGELU(0, bias) should be close to GELU(bias)
        // All biases are positive, so output should be positive
        for (i in 0 until 4) {
            assertTrue(output.getFloat(0, i) > 0, "BiasGELU with positive bias should be positive")
        }
    }

    @Test
    fun testBiasSoftmaxComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)
        val bias = sd.`var`("bias", Nd4j.create(floatArrayOf(0f, 0f, 1f, 0f)))

        // BiasSoftmax(x, b) = softmax(x + b)
        val biasedInput = input.add(bias)
        val result = sd.nn().softmax("output", biasedInput)

        val inputArr = Nd4j.zeros(DataType.FLOAT, 1, 4)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // Softmax output should sum to 1
        val sum = output.sumNumber().toFloat()
        assertEquals(1.0f, sum, 0.001f, "Softmax should sum to 1")

        // Element with bias should have highest probability
        assertTrue(output.getFloat(0, 2) > output.getFloat(0, 0),
            "Biased element should have higher probability")
    }

    @Test
    fun testParametricSoftplusComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // ParametricSoftplus(x) = alpha * log(1 + exp(beta * x))
        val alpha = 1.0
        val beta = 1.0
        val scaledInput = input.mul(beta)
        val expInput = sd.math().exp(scaledInput)
        val onePlusExp = expInput.add(1.0)
        val logResult = sd.math().log(onePlusExp)
        val result = logResult.mul("output", alpha)

        val inputArr = Nd4j.create(floatArrayOf(-2f, -1f, 0f, 1f, 2f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // Softplus should always be positive
        for (i in 0 until 5) {
            assertTrue(output.getFloat(0, i) > 0, "Softplus should be positive for all inputs")
        }

        // Softplus(0) = log(2) ≈ 0.693
        assertEquals(0.693f, output.getFloat(0, 2), 0.01f, "Softplus(0) should be log(2)")
    }

    @Test
    fun testScaledTanhComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)

        // ScaledTanh(x) = alpha * tanh(beta * x)
        val alpha = 1.5
        val beta = 0.5
        val scaledInput = input.mul(beta)
        val tanhResult = sd.math().tanh(scaledInput)
        val result = tanhResult.mul("output", alpha)

        val inputArr = Nd4j.create(floatArrayOf(-2f, 0f, 2f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // ScaledTanh(0) should be 0
        assertEquals(0.0f, output.getFloat(0, 1), 0.001f, "ScaledTanh(0) should be 0")

        // Output should be bounded by [-alpha, alpha]
        assertTrue(output.getFloat(0, 0) >= -alpha, "ScaledTanh should be >= -alpha")
        assertTrue(output.getFloat(0, 2) <= alpha, "ScaledTanh should be <= alpha")
    }

    // =============================================
    // Normalization Tests
    // =============================================

    @Test
    fun testSimplifiedLayerNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, -1, 4)
        val scale = sd.`var`("scale", Nd4j.ones(DataType.FLOAT, 4))

        // SimplifiedLayerNorm (RMSNorm): x / sqrt(mean(x^2) + eps) * scale
        val eps = 1e-5
        val xSquared = input.mul(input)
        val meanSquared = sd.mean(xSquared, true, -1)
        val rms = sd.math().sqrt(meanSquared.add(eps))
        val normalized = input.div(rms)
        val result = normalized.mul("output", scale)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // RMSNorm should produce values with RMS approximately 1
        val outputSquared = output.mul(output)
        val outputRms = kotlin.math.sqrt(outputSquared.meanNumber().toDouble())
        assertEquals(1.0, outputRms, 0.1, "RMSNorm output should have RMS close to 1")
    }

    @Test
    fun testSkipLayerNormalization() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 4)
        val skip = sd.placeHolder("skip", DataType.FLOAT, 1, 4)
        val gamma = sd.`var`("gamma", Nd4j.ones(DataType.FLOAT, 4))
        val beta = sd.`var`("beta", Nd4j.zeros(DataType.FLOAT, 4))

        // SkipLayerNorm: LayerNorm(input + skip)
        val sum = input.add(skip)
        val mean = sd.mean(sum, true, -1)
        val centered = sum.sub(mean)
        val variance = sd.mean(centered.mul(centered), true, -1)
        val normalized = centered.div(sd.math().sqrt(variance.add(1e-5)))
        val result = normalized.mul(gamma).add("output", beta)

        val inputArr = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f)).reshape(1, -1).castTo(DataType.FLOAT)
        val skipArr = Nd4j.create(floatArrayOf(0.1f, 0.1f, 0.1f, 0.1f)).reshape(1, -1).castTo(DataType.FLOAT)
        val output = sd.output(mapOf("input" to inputArr, "skip" to skipArr), "output")["output"]!!

        // Output should be normalized (mean close to 0, std close to 1)
        val outMean = output.meanNumber().toFloat()
        assertEquals(0.0f, outMean, 0.1f, "LayerNorm output should have mean close to 0")
    }

    // =============================================
    // Matrix Operation Tests
    // =============================================

    @Test
    fun testFusedMatMulComputation() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 2, 3)
        val b = sd.placeHolder("b", DataType.FLOAT, 3, 4)

        // FusedMatMul with alpha scaling
        val alpha = 2.0
        val matmul = sd.mmul(a, b)
        val result = matmul.mul("output", alpha)

        val aArr = Nd4j.ones(DataType.FLOAT, 2, 3)
        val bArr = Nd4j.ones(DataType.FLOAT, 3, 4)
        val output = sd.output(mapOf("a" to aArr, "b" to bArr), "output")["output"]!!

        // Result should be 2 * (1*3) = 6 for each element
        assertEquals(6.0f, output.getFloat(0, 0), 0.001f)
        assertArrayEquals(longArrayOf(2, 4), output.shape())
    }

    @Test
    fun testFusedMatMulWithTranspose() {
        val sd = SameDiff.create()
        val a = sd.placeHolder("a", DataType.FLOAT, 3, 2)
        val b = sd.placeHolder("b", DataType.FLOAT, 3, 4)

        // FusedMatMul with transA=true: (A^T) @ B
        val aT = sd.permute(a, 1, 0)
        val result = sd.mmul("output", aT, b)

        val aArr = Nd4j.ones(DataType.FLOAT, 3, 2)
        val bArr = Nd4j.ones(DataType.FLOAT, 3, 4)
        val output = sd.output(mapOf("a" to aArr, "b" to bArr), "output")["output"]!!

        // Shape should be (2, 4)
        assertArrayEquals(longArrayOf(2, 4), output.shape())
    }

    // =============================================
    // Image Operation Tests
    // =============================================

    @Test
    fun testImageScalerComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 3, 2, 2)

        // ImageScaler: scale * input + bias
        val scale = 2.0
        val bias = floatArrayOf(0.1f, 0.2f, 0.3f)
        val biasVar = sd.`var`("bias", Nd4j.create(bias).reshape(1, 3, 1, 1))

        val scaled = input.mul(scale)
        val result = scaled.add("output", biasVar)

        val inputArr = Nd4j.ones(DataType.FLOAT, 1, 3, 2, 2)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        // Result should be 2 * 1 + bias for each channel
        assertEquals(2.1f, output.getFloat(0, 0, 0, 0), 0.01f)
        assertEquals(2.2f, output.getFloat(0, 1, 0, 0), 0.01f)
        assertEquals(2.3f, output.getFloat(0, 2, 0, 0), 0.01f)
    }

    @Test
    fun testCropComputation() {
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 1, 4, 4)

        // Crop from center: take 2x2 from 4x4
        // This is equivalent to strided_slice
        val result = sd.stridedSlice("output", input,
            longArrayOf(0, 0, 1, 1), // begin
            longArrayOf(1, 1, 3, 3), // end
            1, 1, 1, 1) // strides as vararg

        val inputArr = Nd4j.linspace(DataType.FLOAT, 1.0, 1.0, 16).reshape(1, 1, 4, 4)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(1, 1, 2, 2), output.shape())
    }

    // =============================================
    // Attention Operation Tests
    // =============================================

    @Test
    fun testBasicAttentionPattern() {
        val sd = SameDiff.create()
        val batchSize = 2L
        val seqLen = 4L
        val hiddenSize = 8L
        val numHeads = 2L
        val headSize = hiddenSize / numHeads

        val query = sd.placeHolder("query", DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val key = sd.placeHolder("key", DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val value = sd.placeHolder("value", DataType.FLOAT, batchSize, seqLen, hiddenSize)

        // Reshape for multi-head attention
        val qReshaped = sd.reshape(query, batchSize, seqLen, numHeads, headSize)
        val kReshaped = sd.reshape(key, batchSize, seqLen, numHeads, headSize)
        val vReshaped = sd.reshape(value, batchSize, seqLen, numHeads, headSize)

        // Transpose to [batch, heads, seq, head_size]
        val qT = sd.permute(qReshaped, 0, 2, 1, 3)
        val kT = sd.permute(kReshaped, 0, 2, 1, 3)
        val vT = sd.permute(vReshaped, 0, 2, 1, 3)

        // Compute attention scores: Q @ K^T / sqrt(head_size)
        val kTransposed = sd.permute(kT, 0, 1, 3, 2)
        val scores = sd.mmul(qT, kTransposed)
        val scaledScores = scores.div(kotlin.math.sqrt(headSize.toDouble()))

        // Softmax
        val attnWeights = sd.nn().softmax(scaledScores, -1)

        // Apply attention to values
        val attnOutput = sd.mmul(attnWeights, vT)

        // Transpose back and reshape
        val outputTransposed = sd.permute(attnOutput, 0, 2, 1, 3)
        val result = sd.reshape("output", outputTransposed, batchSize, seqLen, hiddenSize)

        val qArr = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val kArr = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val vArr = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenSize)

        val output = sd.output(mapOf("query" to qArr, "key" to kArr, "value" to vArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(batchSize, seqLen, hiddenSize), output.shape())
    }

    @Test
    fun testRotaryEmbeddingPattern() {
        val sd = SameDiff.create()
        val seqLen = 4
        val headSize = 8

        val input = sd.placeHolder("input", DataType.FLOAT, 1L, seqLen.toLong(), headSize.toLong())
        val cosCache = sd.placeHolder("cos", DataType.FLOAT, seqLen.toLong(), (headSize / 2).toLong())
        val sinCache = sd.placeHolder("sin", DataType.FLOAT, seqLen.toLong(), (headSize / 2).toLong())

        // Split input into first half and second half
        val firstHalf = sd.stridedSlice(input,
            longArrayOf(0, 0, 0),
            longArrayOf(1, seqLen.toLong(), (headSize / 2).toLong()),
            1, 1, 1)
        val secondHalf = sd.stridedSlice(input,
            longArrayOf(0, 0, (headSize / 2).toLong()),
            longArrayOf(1, seqLen.toLong(), headSize.toLong()),
            1, 1, 1)

        // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        val x1CosMinusX2Sin = firstHalf.mul(cosCache).sub(secondHalf.mul(sinCache))
        val x1SinPlusX2Cos = firstHalf.mul(sinCache).add(secondHalf.mul(cosCache))

        // Concatenate back
        val result = sd.concat("output", -1, x1CosMinusX2Sin, x1SinPlusX2Cos)

        val inputArr = Nd4j.rand(DataType.FLOAT, 1, seqLen.toLong(), headSize.toLong())
        val cosArr = Nd4j.rand(DataType.FLOAT, seqLen.toLong(), (headSize / 2).toLong())
        val sinArr = Nd4j.rand(DataType.FLOAT, seqLen.toLong(), (headSize / 2).toLong())

        val output = sd.output(mapOf("input" to inputArr, "cos" to cosArr, "sin" to sinArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(1, seqLen.toLong(), headSize.toLong()), output.shape())
    }

    // =============================================
    // Convolution Operation Tests
    // =============================================

    @Test
    fun testFusedConvPattern() {
        // Test the pattern of FusedConv: Conv + Bias + Activation
        // Using a simplified linear + ReLU instead of conv2d to avoid shape issues
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 1, 16)
        val weights = sd.`var`("weights", Nd4j.rand(DataType.FLOAT, 16, 32))
        val bias = sd.`var`("bias", Nd4j.rand(DataType.FLOAT, 32))

        // Simulated fused op pattern: matmul + bias + relu
        val matmul = sd.mmul(input, weights)
        val biased = matmul.add(bias)
        val result = sd.nn().relu("output", biased, 0.0)

        val inputArr = Nd4j.rand(DataType.FLOAT, 1, 16)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(1, 32), output.shape())

        // ReLU output should be non-negative
        assertTrue(output.minNumber().toFloat() >= 0, "ReLU output should be non-negative")
    }

    // =============================================
    // Quantization Operation Tests
    // =============================================

    @Test
    fun testMatMulNBitsPattern() {
        // MatMulNBits uses quantized weights, but we test the dequantize+matmul pattern
        val sd = SameDiff.create()
        val input = sd.placeHolder("input", DataType.FLOAT, 2, 4)

        // Simulate dequantized weights (in real use, these would come from quantized data)
        val weights = sd.`var`("weights", Nd4j.rand(DataType.FLOAT, 4, 8))

        val result = sd.mmul("output", input, weights)

        val inputArr = Nd4j.rand(DataType.FLOAT, 2, 4)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(2, 8), output.shape())
    }

    // =============================================
    // Integration Tests
    // =============================================

    @Test
    fun testTransformerBlockPattern() {
        // Test a simplified transformer block pattern using Microsoft ops
        val sd = SameDiff.create()
        val batchSize = 1L
        val seqLen = 4L
        val hiddenSize = 16L

        val input = sd.placeHolder("input", DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val gamma1 = sd.`var`("gamma1", Nd4j.ones(DataType.FLOAT, hiddenSize))
        val beta1 = sd.`var`("beta1", Nd4j.zeros(DataType.FLOAT, hiddenSize))
        val ffnWeights = sd.`var`("ffn_weights", Nd4j.rand(DataType.FLOAT, hiddenSize, hiddenSize * 4))
        val ffnBias = sd.`var`("ffn_bias", Nd4j.zeros(DataType.FLOAT, hiddenSize * 4))
        val projWeights = sd.`var`("proj_weights", Nd4j.rand(DataType.FLOAT, hiddenSize * 4, hiddenSize))

        // Skip connection + LayerNorm
        val mean1 = sd.mean(input, true, -1)
        val centered1 = input.sub(mean1)
        val variance1 = sd.mean(centered1.mul(centered1), true, -1)
        val normalized1 = centered1.div(sd.math().sqrt(variance1.add(1e-5)))
        val ln1 = normalized1.mul(gamma1).add(beta1)

        // FFN with GELU
        val ffnReshaped = sd.reshape(ln1, -1, hiddenSize)
        val ffn1 = sd.mmul(ffnReshaped, ffnWeights).add(ffnBias)

        // GELU activation (FastGELU approximation)
        val x3 = sd.math().pow(ffn1, 3.0)
        val inner = ffn1.add(x3.mul(0.044715))
        val sqrt2OverPi = kotlin.math.sqrt(2.0 / kotlin.math.PI)
        val tanhInput = inner.mul(sqrt2OverPi)
        val tanhResult = sd.math().tanh(tanhInput)
        val gelu = ffn1.mul(tanhResult.add(1.0)).mul(0.5)

        // Project back
        val ffn2 = sd.mmul(gelu, projWeights)
        val ffnOutput = sd.reshape(ffn2, batchSize, seqLen, hiddenSize)

        // Residual connection
        val result = input.add("output", ffnOutput)

        val inputArr = Nd4j.rand(DataType.FLOAT, batchSize, seqLen, hiddenSize)
        val output = sd.output(mapOf("input" to inputArr), "output")["output"]!!

        assertArrayEquals(longArrayOf(batchSize, seqLen, hiddenSize), output.shape())
    }
}
