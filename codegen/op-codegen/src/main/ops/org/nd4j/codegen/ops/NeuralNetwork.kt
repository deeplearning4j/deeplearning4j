/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.AtLeast
import org.nd4j.codegen.api.DataType.*
import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.mixins.transformStrict

fun NN() = Namespace("NN") {
    val convPkg = "org.nd4j.linalg.api.ops.impl.layers.convolution"

    Op("batchNorm") {
        javaPackage = convPkg
        Input(NUMERIC, "input") { description = "Input variable." }
        Input(NUMERIC, "mean") { description = "Mean value. For 1d axis, this should match input.size(axis)" }
        Input(NUMERIC, "variance") { description = "Variance value. For 1d axis, this should match input.size(axis)" }
        Input(NUMERIC, "gamma") { description = "Gamma value. For 1d axis, this should match input.size(axis)" }
        Input(NUMERIC, "beta") { description = "Beta value. For 1d axis, this should match input.size(axis)" }
        Arg(NUMERIC, "epsilon") { description = "Epsilon constant for numerical stability (to avoid division by 0)" }
        Arg(INT, "axis") {
            count = AtLeast(1)
            description = "For 2d CNN activations: 1 for NCHW format activations, or 3 for NHWC format activations.\n" +
                    "For 3d CNN activations: 1 for NCDHW format, 4 for NDHWC\n" +
                    "For 1d/RNN activations: 1 for NCW format, 2 for NWC"
        }

        Output(NUMERIC, "output") { description = "variable for batch normalization" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Neural network batch normalization operation.
            For details, see <a href="https://arxiv.org/abs/1502.03167">https://arxiv.org/abs/1502.03167</a>
            """.trimIndent()
        }
    }

    Op("biasAdd") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.broadcast"
        Input(NUMERIC, "input") { description = "4d input variable" }
        Input(NUMERIC, "bias") { description = "1d bias" }
        Arg(BOOL, "nchw") { description = "The format - nchw=true means [minibatch, channels, height, width] format; nchw=false - [minibatch, height, width, channels].\n" +
                "Unused for 2d inputs" }

        Output(NUMERIC, "output") { description = "Output variable, after applying bias add operation" }

        Doc(Language.ANY, DocScope.ALL) {
            """
            Bias addition operation: a special case of addition, typically used with CNN 4D activations and a 1D bias vector
            """.trimIndent()
        }
    }

    Op("dropout") {
        javaPackage = "org.nd4j.linalg.api.ops.random.impl"
        javaOpClass = "CustomDropOut"
        Input(NUMERIC, "input") { description = "Input array" }
        Arg(BOOL, "inverted") { description = "Whether dropout should be inverted or not." }
        Arg(INT, "seed") { description = "the seed for dropout"; defaultValue = 0 }
        Arg(NUMERIC,"probabilityValue") { description = "the chance of dropping a value to 0. Maybe interpreted as 1 - p if inverted is true."}
        Output(NUMERIC, "output") { description = "Output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
                Dropout operation
            """.trimIndent()
        }
    }



    Op("elu", transformStrict) {
        javaOpClass = "ELU"
        legacy = false
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise exponential linear unit (ELU) function:
             out = x if x > 0
             out = a * (exp(x) - 1) if x <= 0
             with constant a = 1.0
             <p>
             See: <a href="https://arxiv.org/abs/1511.07289">https://arxiv.org/abs/1511.07289</a>
            """.trimIndent()
        }
    }

    Op("gelu", transformStrict) {
        javaOpClass = "GELU"

        Doc(Language.ANY, DocScope.ALL) {
            """
             GELU activation function - Gaussian Error Linear Units
             For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
             This method uses the sigmoid approximation
            """.trimIndent()
        }
    }

    Op("hardSigmoid", transformStrict) {
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise hard sigmoid function:
             out[i] = 0 if in[i] <= -2.5
             out[1] = 0.2*in[i]+0.5 if -2.5 < in[i] < 2.5
             out[i] = 1 if in[i] >= 2.5
            """.trimIndent()
        }
    }

    Op("hardTanh", transformStrict) {
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise hard tanh function:
             out[i] = -1 if in[i] <= -1
             out[1] = in[i] if -1 < in[i] < 1
             out[i] = 1 if in[i] >= 1
            """.trimIndent()
        }
    }

    Op("hardTanhDerivative") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.gradient"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output"){ description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             Derivative (dOut/dIn) of the element-wise hard Tanh function - hardTanh(%INPUT_TYPE%)
            """.trimIndent()
        }
    }

    Op("leakyRelu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        javaOpClass = "LeakyReLU"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(NUMERIC, "alpha") { description = "Cutoff - commonly 0.01" }

        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise leaky ReLU function:
             out = x if x >= 0.0
             out = alpha * x if x < cutoff
             Alpha value is most commonly set to 0.01
            """.trimIndent()
        }
    }

    Op("leakyReluDerivative") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.gradient"
        javaOpClass = "LeakyReLUDerivative"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Arg(FLOATING_POINT, "alpha") { description = "Cutoff - commonly 0.01" }

        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Leaky ReLU derivative: dOut/dIn given input.
            """.trimIndent()
        }
    }

    Op("CReLU") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CReLU"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Concatenates a ReLU which selects only the positive part of the activation with a ReLU which selects only the negative part of the activation. Note that as a result this non-linearity doubles the depth of the activations.
            """.trimIndent()
        }
    }

    Op("linear") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "XwPlusB"
        Input(NUMERIC, "input") { description = "Input data" }
        Input(NUMERIC, "weights") { description = "Weights variable, shape [nIn, nOut]" }
        Input(NUMERIC, "bias") { description = "Optional bias variable (may be null)" /*; optional = true*/ }
        Arg(BOOL,"transposeA") { description = "Whether to transpose input or not"; defaultValue= false}
        Arg(BOOL,"transposeB") { description = "Whether to transpose second input or not"; defaultValue= false}
        Arg(BOOL,"transposeC") { description = "Whether to transpose result or not"; defaultValue= false}
        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Linear layer operation: out = mmul(in,w) + bias
             Note that bias array is optional
            """.trimIndent()
        }
    }


    Op("logSigmoid", transformStrict) {
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise sigmoid function: out[i] = log(sigmoid(in[i]))
            """.trimIndent()
        }
    }

    Op("logSoftmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LogSoftMax"
        Input(NUMERIC, "x") { description = "" }
        Output(NUMERIC, "output") { description = "" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             Log softmax activation
            """.trimIndent()
        }
    }

    Op("logSoftmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LogSoftMax"
        Input(NUMERIC, "x") { description = "Input" }
        Arg(INT, "dimension") { description = "Dimension along which to apply log softmax" }
        Output(NUMERIC, "output") { description = "Output - log(softmax(input))" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Log softmax activation
            """.trimIndent()
        }
    }

    Op("relu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        javaOpClass = "RectifiedLinear"
        legacy = true
        Input(NUMERIC, "x") { description = "Input" }
        Arg(NUMERIC, "cutoff") { description = "Cutoff value for ReLU operation - x > cutoff ? x : 0. Usually 0" }
        Output(NUMERIC, "output") { description = "Output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise rectified linear function with specified cutoff:
             out[i] = in[i] if in[i] >= cutoff
             out[i] = 0 otherwise
            """.trimIndent()
        }
    }

    Op("relu6") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        legacy = true
        Input(NUMERIC, "x") { description = "Input" }
        Arg(NUMERIC, "cutoff") { description = "Cutoff value for ReLU operation. Usually 0" }
        Output(NUMERIC, "output") { description = "Output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise "rectified linear 6" function with specified cutoff:
             out[i] = min(max(in, cutoff), 6)
            """.trimIndent()
        }
    }

    Op("reluLayer") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms"
        Input(NUMERIC, "input") { description = "Input data" }
        Input(NUMERIC, "weights") { description = "Weights variable" }
        Input(NUMERIC, "bias") { description = " Bias variable" }
        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)
           
            """.trimIndent()
        }
    }

    Op("preciseGelu", transformStrict) {
        javaOpClass = "PreciseGELU"

        Doc(Language.ANY, DocScope.ALL) {
            """
             GELU activation function - Gaussian Error Linear Units
             For more details, see <i>Gaussian Error Linear Units (GELUs)</i> - <a href="https://arxiv.org/abs/1606.08415">https://arxiv.org/abs/1606.08415</a>
             This method uses the precise method
            """.trimIndent()
        }
    }

    Op("prelu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.scalar"
        javaOpClass = "PRelu"
        Input(NUMERIC, "input") { description = "Input data" }
        Input(NUMERIC, "alpha") { description = "The cutoff variable.  Note that the batch dimension (the 0th, whether it is batch or not) should not be part of alpha." }
        Arg(INT, "sharedAxes") { count = AtLeast(1); description = "Which axes to share cutoff parameters along." }

        Output(NUMERIC, "output") { description = "Output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             PReLU (Parameterized Rectified Linear Unit) operation.  Like LeakyReLU with a learnable alpha:
             out[i] = in[i] if in[i] >= 0
             out[i] = in[i] * alpha[i] otherwise
            
             sharedAxes allows you to share learnable parameters along axes.
             For example, if the input has shape [batchSize, channels, height, width]
             and you want each channel to have its own cutoff, use sharedAxes = [2, 3] and an
             alpha with shape [channels].
            """.trimIndent()
        }
    }

    Op("selu", transformStrict) {
        javaOpClass = "SELU"
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise SeLU function - Scaled exponential Lineal Unit: see <a href="https://arxiv.org/abs/1706.02515">Self-Normalizing Neural Networks</a>
             
             out[i] = scale * alpha * (exp(in[i])-1) if in[i]>0, or 0 if in[i] <= 0
             Uses default scale and alpha values.
            """.trimIndent()
        }
    }

    Op("sigmoid", transformStrict) {
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise sigmoid function: out[i] = 1.0/(1+exp(-in[i]))
            """.trimIndent()
        }
    }

    Op("sigmoidDerivative") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.gradient"
        Input(NUMERIC, "x") { description = "Input Variable" }
        Input(NUMERIC, "wrt") { description = "Gradient at the output - dL/dOut. Must have same shape as the input" }
        Output(NUMERIC, "output") { description = "Output (gradient at input of sigmoid)" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise sigmoid function derivative: dL/dIn given input and dL/dOut
            """.trimIndent()
        }
    }

    Op("softmax") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SoftMax"
        Input(NUMERIC, "x") { description = "Input" }
        Arg(INT, "dimension") { description = "Dimension along which to apply softmax"; defaultValue = -1 }
        Output(NUMERIC, "output") { description = "Output variable" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             Softmax activation, along the specified dimension
            """.trimIndent()
        }
    }



    Op("softplus", transformStrict) {
        javaOpClass = "SoftPlus"
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise softplus function: out = log(exp(x) + 1)
            """.trimIndent()
        }
    }

    Op("softsign", transformStrict) {
        javaOpClass = "SoftSign"
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise softsign function: out = x / (abs(x) + 1)
            """.trimIndent()
        }
    }

    Op("softsignDerivative") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.gradient"
        javaOpClass = "SoftSignDerivative"
        legacy = true
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output") { description = "Output" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise derivative (dOut/dIn) of the softsign function softsign(%INPUT_TYPE%)
            """.trimIndent()
        }
    }

    Op("swish", transformStrict) {
        Doc(Language.ANY, DocScope.ALL) {
            """
             Element-wise "swish" function: out = x * sigmoid(b*x) with b=1.0
             See: <a href="https://arxiv.org/abs/1710.05941">https://arxiv.org/abs/1710.05941</a>
            """.trimIndent()
        }
    }

    Op("layerNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val input = Input(NUMERIC, "input") { description = "Input variable" }
        val g = Input(NUMERIC, "gain") { description = "Gain" }
        Input(NUMERIC, "bias") { description = "Bias"; defaultValue = null}
        val ch = Arg(BOOL, "channelsFirst") { description = "For 2D input - unused. True for NCHW (minibatch, channels, height, width), false for NHWC data" }
        val dim = Arg(LONG, "dimensions") { count = AtLeast(1); description = "Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs" }

        Output(NUMERIC, "output") { description = "Output variable" }

        AllParamSignature()
        Signature(input, g, ch, dim)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Apply Layer Normalization
            
             y = gain * standardize(x) + bias
            """.trimIndent()
        }
    }

    Op("rmsNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "RmsNorm"
        val input = Input(NUMERIC, "input") { description = "Input variable" }
        val gamma = Input(NUMERIC, "gamma") { description = "Scale/gain vector"; defaultValue = null }
        val epsilon = Arg(FLOATING_POINT, "epsilon") { defaultValue = 1e-5; description = "Epsilon for numerical stability" }

        Output(NUMERIC, "output") { description = "RMS normalized output" }

        AllParamSignature()
        Signature(input, gamma)
        Signature(input, epsilon)
        Signature(input)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Root Mean Square Layer Normalization (RMSNorm):

             output = input * rsqrt(mean(input^2, axis=-1) + epsilon) * gamma

             If gamma is not provided, only RMS normalization is applied.
            """.trimIndent()
        }
    }


    Op("dotProductAttentionV2") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "queries") { description = "Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention" }
        val v = Input(NUMERIC, "values") { description = "Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim]" }
        val k = Input(NUMERIC, "keys") { description = "Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim]" }
        val queryMask = Input(NUMERIC, "queryMask") { description = "Query mask tensor (optional). Shape: [batchSize, numQueries]"; defaultValue = null }
        val valueMask = Input(NUMERIC, "valueMask") { description = "Value mask tensor (optional). Shape: [batchSize, numValues]"; defaultValue = null }
        val attentionBias = Input(NUMERIC, "attentionBias") { description = "Attention bias tensor (optional). Shape: [batchSize, numHeads, numQueries, numKeys] or broadcastable. Added to attention scores before softmax."; defaultValue = null }

        val s = Arg(FLOATING_POINT, "scaleFactor") { defaultValue = 0.0; description = "Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))" }
        val dropout = Arg(FLOATING_POINT, "dropoutProbability") { defaultValue = 0.0; description = "Dropout probability applied to attention weights" }
        val useCausalMask = Arg(BOOL, "useCausalMask") { defaultValue = false; description = "Whether to apply causal mask for autoregressive tasks" }
        val training = Arg(BOOL, "training") { defaultValue = false; description = "Whether in training mode (affects dropout)" }

        Output(NUMERIC, "output") { description = "Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim]" }

        // Standard signature without attention bias (backward compatible)
        Signature(q, v, k, queryMask, valueMask, s, dropout, useCausalMask, training)
        // Full signature with attention bias
        Signature(q, v, k, queryMask, valueMask, attentionBias, s, dropout, useCausalMask, training)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Dot product attention operation with flash attention and KV cache support.

             out = softmax(Q * K^T / scale + attentionBias) * V

             For 4D inputs [batch, seq, heads, dim], uses memory-efficient flash attention algorithm.
             For 2D/3D inputs, uses standard attention computation.

             Flash attention features:
             - O(N) memory complexity instead of O(N^2)
             - Tiled computation with online softmax
             - Supports grouped query attention (GQA) where numHeads > numKvHeads
             - Supports attention bias (relative position bias, ALiBi, etc.)

             KV Cache support for autoregressive generation:
             - Pass keyCache and valueCache tensors
             - Set kvCachePosition to current generation position
             - Cached keys/values are updated in-place

             See "Attention is all you need" (https://arxiv.org/abs/1706.03762)
             See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)
            """.trimIndent()
        }
    }

    Op("dotProductAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "queries") { description = "input 3D array \"queries\" of shape [batchSize, featureKeys, queryCount]\n" +
                "or 4D array of shape [batchSize, numHeads, featureKeys, queryCount]" }
        val k = Input(NUMERIC, "keys") { description = "input 3D array \"keys\" of shape [batchSize, featureKeys, timesteps]\n" +
                "or 4D array of shape [batchSize, numHeads, featureKeys, timesteps]" }
        val v = Input(NUMERIC, "values") { description = "input 3D array \"values\" of shape [batchSize, featureValues, timesteps]\n" +
                "or 4D array of shape [batchSize, numHeads, featureValues, timesteps]" }
        val m = Input(NUMERIC, "mask") { description = "OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]" }
        val s = Arg(BOOL, "scaled") { description = "normalization, false -> do not apply normalization, true -> apply normalization" }
        Arg(BOOL, "withWeights") { defaultValue = false; description = "withWeights return attention weights as well, false -> only one output, true -> two outputs" }

        Output(NUMERIC, "output") { description = " Attention result arrays of shape [batchSize, featureValues, queryCount] or [batchSize, numHeads, featureValues, queryCount],\n" +
                "(optionally) Attention Weights of shape [batchSize, timesteps, queryCount] or [batchSize, numHeads, timesteps, queryCount]" }

        Signature(q, k, v, m, s)

        Doc(Language.ANY, DocScope.ALL) {
            """
             This operation performs dot product attention on the given timeseries input with the given queries
             out = sum(similarity(k_i, q) * v_i)
            
             similarity(k, q) = softmax(k * q) where x * q is the dot product of x and q
            
             Optionally with normalization step:
             similarity(k, q) = softmax(k * q / sqrt(size(q))
            
             See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, p. 4, eq. 1)
            
             Note: This supports multiple queries at once, if only one query is available the queries vector still has to
             be 3D but can have queryCount = 1
            
             Note: keys and values usually is the same array. If you want to use it as the same array, simply pass it for
             both.
            
             Note: Queries, keys and values must either be all rank 3 or all rank 4 arrays. Mixing them doesn't work. The
             output rank will depend on the input rank.
            """.trimIndent()
        }
    }



    Op("multiHeadDotProductAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "queries") { description = "input 3D array \"queries\" of shape [batchSize, featureKeys, queryCount]" }
        val k = Input(NUMERIC, "keys") { description = "input 3D array \"keys\" of shape [batchSize, featureKeys, timesteps]" }
        val v = Input(NUMERIC, "values") { description = "input 3D array \"values\" of shape [batchSize, featureValues, timesteps]" }
        val wq = Input(NUMERIC, "Wq") { description = "input query projection weights of shape [numHeads, projectedKeys, featureKeys]" }
        val wk = Input(NUMERIC, "Wk") { description = "input key projection weights of shape [numHeads, projectedKeys, featureKeys]" }
        val wv = Input(NUMERIC, "Wv") { description = "input value projection weights of shape [numHeads, projectedValues, featureValues]" }
        val wo = Input(NUMERIC, "Wo") { description = "output projection weights of shape [numHeads * projectedValues, outSize]" }
        val m = Input(NUMERIC, "mask") { description = "OPTIONAL; array that defines which values should be skipped of shape [batchSize, timesteps]" }
        val s = Arg(BOOL, "scaled") { description = "normalization, false -> do not apply normalization, true -> apply normalization" }
        Arg(BOOL, "withWeights") { defaultValue = false; description = "return attention weights as well, false -> only one output, true -> two outputs" }

        Output(NUMERIC, "output") { description = "Attention result arrays of shape [batchSize, outSize, queryCount]\n" +
                "(optionally) Attention Weights of shape [batchSize, numHeads, timesteps, queryCount]" }

        Signature(q, k, v, wq, wk, wv, wo, m, s)

        Doc(Language.ANY, DocScope.ALL) {
            """
             This performs multi-headed dot product attention on the given timeseries input
             out = concat(head_1, head_2, ..., head_n) * Wo
             head_i = dot_product_attention(Wq_i*q, Wk_i*k, Wv_i*v)
            
             Optionally with normalization when calculating the attention for each head.
            
             See also "Attention is all you need" (https://arxiv.org/abs/1706.03762, pp. 4,5, "3.2.2 Multi-Head Attention")
            
             This makes use of dot_product_attention OP support for rank 4 inputs.
             see dotProductAttention(%INPUT_TYPE%, %INPUT_TYPE%, %INPUT_TYPE%, %INPUT_TYPE%, boolean, boolean)
            """.trimIndent()
        }
    }

    Op("flashAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "query") { description = "Query tensor. Shape: [batch, seqLen, numHeads, headDim]" }
        val k = Input(NUMERIC, "key") { description = "Key tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }
        val v = Input(NUMERIC, "value") { description = "Value tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }

        val scale = Arg(FLOATING_POINT, "scale") { defaultValue = 0.0; description = "Scaling factor. 0 = auto (1/sqrt(headDim))" }
        val isCausal = Arg(BOOL, "isCausal") { defaultValue = true; description = "Whether to apply causal masking" }
        val numHeads = Arg(INT, "numHeads") { description = "Number of query attention heads" }
        val numKvHeads = Arg(INT, "numKvHeads") { defaultValue = 0; description = "Number of KV heads (0 = same as numHeads, for GQA use smaller value)" }

        Output(NUMERIC, "output") { description = "Attention output. Shape: [batch, seqLen, numHeads, headDim]" }

        Signature(q, k, v, scale, isCausal, numHeads, numKvHeads)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Flash Attention - Memory-efficient attention computation.

             Uses tiled computation with online softmax to achieve O(N) memory complexity
             instead of O(N^2) for standard attention.

             Supports Grouped Query Attention (GQA) where numHeads > numKvHeads,
             allowing multiple query heads to share the same KV heads.

             out = softmax(Q * K^T / scale) * V

             See "FlashAttention: Fast and Memory-Efficient Exact Attention" (https://arxiv.org/abs/2205.14135)
            """.trimIndent()
        }
    }

    Op("groupedQueryAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "query") { description = "Query tensor. Shape: [batch, seqLen, numHeads, headDim]" }
        val k = Input(NUMERIC, "key") { description = "Key tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }
        val v = Input(NUMERIC, "value") { description = "Value tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }

        val scale = Arg(FLOATING_POINT, "scale") { defaultValue = 0.0; description = "Scaling factor. 0 = auto (1/sqrt(headDim))" }
        val isCausal = Arg(BOOL, "isCausal") { defaultValue = true; description = "Whether to apply causal masking" }
        val numHeads = Arg(INT, "numHeads") { description = "Number of query attention heads" }
        val numKvHeads = Arg(INT, "numKvHeads") { description = "Number of KV heads (must divide numHeads evenly)" }

        Output(NUMERIC, "output") { description = "Attention output. Shape: [batch, seqLen, numHeads, headDim]" }

        Signature(q, k, v, scale, isCausal, numHeads, numKvHeads)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Grouped Query Attention (GQA) - Efficient attention with shared KV heads.

             Multiple query heads share the same key-value heads, reducing memory and
             computation while maintaining model quality. Used in LLaMA 2, Mistral, etc.

             numHeads must be divisible by numKvHeads. Each KV head is repeated
             (numHeads / numKvHeads) times to match query heads.

             Special cases:
             - numKvHeads == numHeads: Standard Multi-Head Attention (MHA)
             - numKvHeads == 1: Multi-Query Attention (MQA)

             See "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"
            """.trimIndent()
        }
    }

    Op("kvCacheUpdate") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "KVCacheUpdate"
        val keyCache = Input(NUMERIC, "keyCache") { description = "Existing key cache. Shape: [batch, maxSeqLen, numKvHeads, headDim]" }
        val valueCache = Input(NUMERIC, "valueCache") { description = "Existing value cache. Shape: [batch, maxSeqLen, numKvHeads, headDim]" }
        val newKeys = Input(NUMERIC, "newKeys") { description = "New keys to insert. Shape: [batch, newSeqLen, numKvHeads, headDim]" }
        val newValues = Input(NUMERIC, "newValues") { description = "New values to insert. Shape: [batch, newSeqLen, numKvHeads, headDim]" }

        val startPosition = Arg(INT, "startPosition") { defaultValue = 0; description = "Position in cache where new keys/values should be inserted" }

        Output(NUMERIC, "updatedKeyCache") { description = "Updated key cache" }
        Output(NUMERIC, "updatedValueCache") { description = "Updated value cache" }

        Signature(keyCache, valueCache, newKeys, newValues, startPosition)

        Doc(Language.ANY, DocScope.ALL) {
            """
             KV Cache Update - Updates key-value cache for autoregressive generation.

             During LLM inference, past key-value pairs are cached to avoid redundant
             computation during token-by-token generation. This operation efficiently
             inserts new keys/values at the specified position.

             Usage pattern:
             1. Initialize cache with zeros: [batch, maxSeqLen, numKvHeads, headDim]
             2. For each new token, compute new K/V and update cache
             3. Use full cached K/V for attention computation

             Returns updated keyCache and valueCache tensors.
            """.trimIndent()
        }
    }

    Op("slidingWindowAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "query") { description = "Query tensor. Shape: [batch, seqLen, numHeads, headDim]" }
        val k = Input(NUMERIC, "key") { description = "Key tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }
        val v = Input(NUMERIC, "value") { description = "Value tensor. Shape: [batch, seqLen, numKvHeads, headDim]" }

        val windowSize = Arg(INT, "windowSize") { defaultValue = 4096; description = "Sliding window size - tokens can only attend to this many previous positions" }
        val numHeads = Arg(INT, "numHeads") { description = "Number of query attention heads" }
        val numKvHeads = Arg(INT, "numKvHeads") { defaultValue = 0; description = "Number of KV heads (0 = same as numHeads)" }
        val scale = Arg(FLOATING_POINT, "scale") { defaultValue = 0.0; description = "Scaling factor. 0 = auto (1/sqrt(headDim))" }

        Output(NUMERIC, "output") { description = "Attention output. Shape: [batch, seqLen, numHeads, headDim]" }

        Signature(q, k, v, windowSize, numHeads, numKvHeads, scale)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Sliding Window Attention - Efficient attention for long sequences.

             Each token only attends to a fixed window of previous tokens, enabling
             efficient processing of very long sequences. Used in Mistral and other
             modern LLMs for handling long contexts.

             Benefits:
             - O(N * windowSize) complexity instead of O(N^2)
             - Memory efficient for long sequences
             - Supports very long context lengths (e.g., 32K with 4K window)

             The attention mask is automatically applied to restrict each position
             to only attend to positions within [pos - windowSize, pos].
            """.trimIndent()
        }
    }

    Op("pad") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms"
        Input(NUMERIC, "input") { description = "Input tensor"}
        Input(NUMERIC, "padding") { description = "Padding value" }
        Arg(ENUM, "PadMode") { possibleValues = listOf("CONSTANT", "REFLECT", "SYMMETRIC"); description = "Padding format"; defaultValue="CONSTANT" }
        Arg(NUMERIC, "constant") { description = "Padding constant" }

        Output(NUMERIC, "output"){ description = "Padded input" }

        Doc(Language.ANY, DocScope.ALL){
            """
             Padding operation
            """.trimIndent()
        }
    }



    Op("topK") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        Input(NUMERIC, "input") { description = "Input data" }
        Arg(NUMERIC, "k") { description = "The number of values to return" }
        Arg(BOOL, "sorted") { description = "Whether to return the values sorted or not" }
        Output(NUMERIC, "output") { description = "the top k values in the input" }
        Output(NUMERIC, "indices") { description = "the indices of the top k values" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Find values and indices for the largest k entries along the last dimension.<br>
            """.trimIndent()
        }
    }

    Op("windowedAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "query") { description = "Query tensor. Shape: [batch, seqLen, numHeads, headDim] for 1D or [batch, height, width, numHeads, headDim] for 2D" }
        val k = Input(NUMERIC, "key") { description = "Key tensor. Same shape as query" }
        val v = Input(NUMERIC, "value") { description = "Value tensor. Same shape as query" }
        val rpb = Input(NUMERIC, "relativePositionBias") { description = "Optional relative position bias. Shape: [numHeads, windowSize, windowSize]"; defaultValue = null }
        val mask = Input(NUMERIC, "attentionMask") { description = "Optional attention mask"; defaultValue = null }

        val windowSize = Arg(INT, "windowSize") { description = "Size of attention window" }
        val numHeads = Arg(INT, "numHeads") { description = "Number of attention heads" }
        val shiftSize = Arg(INT, "shiftSize") { defaultValue = 0; description = "Shift size for shifted window attention (Swin style). 0 = no shift" }
        val scale = Arg(FLOATING_POINT, "scale") { defaultValue = 0.0; description = "Attention scale factor. 0 = auto (1/sqrt(headDim))" }
        val returnWeights = Arg(BOOL, "returnWeights") { defaultValue = false; description = "Whether to return attention weights" }

        Output(NUMERIC, "output") { description = "Attention output. Same shape as query" }

        Signature(q, k, v, windowSize, numHeads)
        Signature(q, k, v, rpb, mask, windowSize, numHeads, shiftSize, scale, returnWeights)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Windowed Attention - Local/Sliding Window Attention.

             Implements windowed attention mechanisms used in efficient transformers like
             Longformer, BigBird, Swin Transformer, and SAM (Segment Anything Model).

             Supports both:
             - 1D windowed attention: for sequences [batch, seqLen, heads, dim]
             - 2D windowed attention: for images [batch, height, width, heads, dim]

             Shifted window attention (shiftSize > 0) enables cross-window connections
             as used in Swin Transformer.

             Benefits:
             - O(N * windowSize) complexity instead of O(N^2)
             - Efficient for long sequences and high-resolution images
             - Supports relative position bias for position-aware attention
            """.trimIndent()
        }
    }

    Op("relativePositionBias") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val biasTable = Input(NUMERIC, "biasTable") { description = "Learned bias table. Shape: [numRelativePositions, numHeads] for learned mode, or scalar/tensor for ALiBi mode" }
        val relPosIndex = Input(NUMERIC, "relativePositionIndex") { description = "Optional precomputed relative position index. Shape: [windowSize^2, windowSize^2]"; defaultValue = null }

        val numHeads = Arg(INT, "numHeads") { description = "Number of attention heads" }
        val windowSize = Arg(INT, "windowSize") { defaultValue = 0; description = "Window size for 2D position encoding (used if generating index)" }
        val useAlibi = Arg(BOOL, "useAlibi") { defaultValue = false; description = "Use ALiBi (Attention with Linear Biases) instead of learned bias" }

        Output(NUMERIC, "output") { description = "Position bias. Shape: [numHeads, windowSize^2, windowSize^2] or [numHeads, seqLen, seqLen]" }

        Signature(biasTable, numHeads, windowSize)
        Signature(biasTable, relPosIndex, numHeads, windowSize)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Relative Position Bias - Compute relative position bias for attention.

             Supports two modes:
             1. Learned bias (Swin/SAM style): Looks up bias values from a learned table
                based on relative positions between query and key positions.

             2. ALiBi (Attention with Linear Biases): Computes linear position-based bias
                without learned parameters. More efficient for very long sequences.

             For learned bias mode:
             - biasTable shape: [(2*windowSize-1)^2, numHeads] for 2D
             - Output is gathered based on relative position indices

             For ALiBi mode:
             - biasTable can be sequence length (scalar) or input tensor
             - Computes m_h * |i - j| where m_h = 2^(-8*h/H)

             Reference: "Swin Transformer" (Liu et al., 2021)
                        "Train Short, Test Long" (Press et al., 2021) for ALiBi
            """.trimIndent()
        }
    }

    Op("mixtureOfExperts") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val input = Input(NUMERIC, "input") { description = "Input embeddings. Shape: [batch, seqLen, hiddenSize]" }
        val routerWeights = Input(NUMERIC, "routerWeights") { description = "Router projection weights. Shape: [hiddenSize, numExperts]" }
        val expertWeights = Input(NUMERIC, "expertWeights") { description = "Expert weight matrices. Shape: [numExperts, hiddenSize, expertHiddenSize]" }
        val expertBias = Input(NUMERIC, "expertBias") { description = "Optional expert biases. Shape: [numExperts, expertHiddenSize]"; defaultValue = null }

        val numExperts = Arg(INT, "numExperts") { description = "Total number of experts" }
        val topK = Arg(INT, "topK") { defaultValue = 2; description = "Number of experts to route to per token" }
        val normalizeProbs = Arg(BOOL, "normalizeProbs") { defaultValue = true; description = "Whether to normalize router probabilities for selected experts" }
        val capacityFactor = Arg(FLOATING_POINT, "capacityFactor") { defaultValue = 1.0; description = "Expert capacity factor for load balancing" }

        Output(NUMERIC, "output") { description = "Combined expert outputs. Shape: [batch, seqLen, expertHiddenSize]" }
        Output(NUMERIC, "routerProbs") { description = "Router probabilities. Shape: [batch, seqLen, numExperts]" }
        Output(NUMERIC, "expertIndices") { description = "Selected expert indices. Shape: [batch, seqLen, topK]" }

        Signature(input, routerWeights, expertWeights, numExperts, topK)
        Signature(input, routerWeights, expertWeights, expertBias, numExperts, topK, normalizeProbs, capacityFactor)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Mixture of Experts (MoE) Layer.

             Implements sparse MoE routing where each token is processed by only the top-k
             selected experts out of a larger pool. This enables scaling model capacity
             without proportionally increasing computation.

             Used in large language models like:
             - DeepSeek (DeepSeekMoE)
             - Mixtral (Mistral AI)
             - Switch Transformer (Google)
             - GShard (Google)

             The router computes expert selection probabilities:
             router_probs = softmax(input @ routerWeights)

             Top-k experts are selected and their outputs are weighted by normalized probs:
             output = sum(normalized_prob[i] * expert[i](input) for i in top_k)

             Benefits:
             - Scales model capacity with sublinear compute increase
             - Enables very large models with efficient inference
             - Supports expert parallelism across devices
            """.trimIndent()
        }
    }

    Op("moeSharedExperts") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MoeSharedExperts"
        val input = Input(NUMERIC, "input") { description = "Input embeddings. Shape: [batch, seqLen, hiddenSize]" }
        val routerWeights = Input(NUMERIC, "routerWeights") { description = "Router projection weights. Shape: [hiddenSize, numRoutedExperts]" }
        val routedExpertWeights = Input(NUMERIC, "routedExpertWeights") { description = "Routed expert weight matrices. Shape: [numRoutedExperts, hiddenSize, expertHidden]" }
        val sharedGateProj = Input(NUMERIC, "sharedGateProj") { description = "Shared expert gate projection. Shape: [hiddenSize, sharedIntermediateSize]" }
        val sharedUpProj = Input(NUMERIC, "sharedUpProj") { description = "Shared expert up projection. Shape: [hiddenSize, sharedIntermediateSize]" }
        val sharedDownProj = Input(NUMERIC, "sharedDownProj") { description = "Shared expert down projection. Shape: [sharedIntermediateSize, hiddenSize]" }
        val routedExpertBias = Input(NUMERIC, "routedExpertBias") { description = "Optional routed expert biases"; defaultValue = null }

        val numRoutedExperts = Arg(INT, "numRoutedExperts") { description = "Number of routed experts" }
        val topK = Arg(INT, "topK") { defaultValue = 2; description = "Number of experts to route to per token" }
        val normalizeProbs = Arg(BOOL, "normalizeProbs") { defaultValue = true; description = "Whether to normalize router probabilities for selected experts" }
        val capacityFactor = Arg(FLOATING_POINT, "capacityFactor") { defaultValue = 1.0; description = "Expert capacity factor for load balancing" }

        Output(NUMERIC, "output") { description = "Combined shared + routed expert outputs. Shape: [batch, seqLen, hiddenSize]" }
        Output(NUMERIC, "routerProbs") { description = "Router probabilities. Shape: [batch, seqLen, numRoutedExperts]" }
        Output(NUMERIC, "expertIndices") { description = "Selected expert indices. Shape: [batch, seqLen, topK]" }

        Signature(input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, numRoutedExperts, topK)
        Signature(input, routerWeights, routedExpertWeights, sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias, numRoutedExperts, topK, normalizeProbs, capacityFactor)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Mixture of Experts with Shared Experts (IBM Granite 4.0 pattern).

             Extends MoE with an always-on shared expert pathway. The shared expert
             processes every token unconditionally using SwiGLU activation, while
             routed experts are selected via top-K gating.

             output = shared_expert(input) + weighted_sum(routed_experts(input))

             where shared_expert uses SwiGLU:
             shared_out = down_proj(silu(gate_proj(x)) * up_proj(x))

             Used in:
             - IBM Granite 4.0 (granitemoeshared architecture)
             - DeepSeek V2/V3 (shared expert variant)
            """.trimIndent()
        }
    }

    Op("ctcGreedyDecoder") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CTCGreedyDecoder"
        val logits = Input(NUMERIC, "logits") { description = "Log probabilities from CTC output. Shape: [batch, timeSteps, numClasses]" }
        val sequenceLength = Input(NUMERIC, "sequenceLength") { description = "Optional actual sequence lengths. Shape: [batch]"; defaultValue = null }

        val mergeRepeated = Arg(BOOL, "mergeRepeated") { defaultValue = true; description = "Whether to merge repeated characters in output" }
        val blankIndex = Arg(INT, "blankIndex") { defaultValue = 0; description = "Index of the blank label in the vocabulary" }

        Output(NUMERIC, "decoded") { description = "Decoded sequences. Shape: [batch, timeSteps] (padded with blank)" }
        Output(NUMERIC, "logProbability") { description = "Log probability of decoded sequences. Shape: [batch]" }

        Signature(logits, mergeRepeated, blankIndex)
        Signature(logits, sequenceLength, mergeRepeated, blankIndex)

        Doc(Language.ANY, DocScope.ALL) {
            """
             CTC Greedy Decoder - Connectionist Temporal Classification decoding.

             Performs greedy (best path) decoding on CTC output. Used in:
             - OCR (Optical Character Recognition) - PaddleOCR, CRNN
             - Speech recognition - DeepSpeech, Wav2Vec
             - Handwriting recognition

             Algorithm:
             1. At each timestep, select the class with highest probability
             2. Optionally merge consecutive repeated characters
             3. Remove blank labels from the output

             For example, with mergeRepeated=true and blankIndex=0:
             Input:  [0, 1, 1, 0, 2, 2, 2, 0] (0=blank, 1='a', 2='b')
             Output: [1, 2] -> "ab"

             Note: This is greedy decoding. For better accuracy with language models,
             use beam search decoding instead.
            """.trimIndent()
        }
    }

    Op("emaUpdate") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "EmaUpdate"
        Input(NUMERIC, "model") { description = "Current model parameters (student)" }
        Input(NUMERIC, "shadow") { description = "EMA shadow parameters (teacher)" }
        Arg(NUMERIC, "decay") { description = "EMA decay factor (typically 0.996-0.9999)"; defaultValue = 0.999 }
        Output(NUMERIC, "output") { description = "Updated shadow parameters" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             Exponential Moving Average parameter update for DINOv2 teacher networks.
             Computes: output = decay * shadow + (1 - decay) * model
             Used in self-supervised learning to maintain a slowly-updated teacher model.
            """.trimIndent()
        }
    }

    Op("centerAndSharpen") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CenterAndSharpen"
        Input(NUMERIC, "input") { description = "Teacher output logits [batch, features]" }
        Input(NUMERIC, "center") { description = "Running center vector [features]" }
        Arg(NUMERIC, "temperature") { description = "Sharpening temperature (typically 0.04-0.07)"; defaultValue = 0.07 }
        Output(NUMERIC, "output") { description = "Sharpened probabilities [batch, features]" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             DINOv2 centering and sharpening operation.
             Prevents mode collapse in self-supervised learning by centering the teacher output
             and applying temperature-based sharpening:
               output = softmax((input - center) / temperature)
            """.trimIndent()
        }
    }

    Op("twoWayCrossAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "TwoWayCrossAttention"
        Input(NUMERIC, "tokenQuery") { description = "Token queries [batch, tokenSeqLen, embedDim]" }
        Input(NUMERIC, "tokenKey") { description = "Token keys [batch, tokenSeqLen, embedDim]" }
        Input(NUMERIC, "tokenValue") { description = "Token values [batch, tokenSeqLen, embedDim]" }
        Input(NUMERIC, "imageQuery") { description = "Image queries [batch, imageSeqLen, embedDim]" }
        Input(NUMERIC, "imageKey") { description = "Image keys [batch, imageSeqLen, embedDim]" }
        Input(NUMERIC, "imageValue") { description = "Image values [batch, imageSeqLen, embedDim]" }
        Arg(NUMERIC, "scale") { description = "Attention scale factor (default: 1/sqrt(embedDim))"; defaultValue = 0.0 }
        Output(NUMERIC, "tokenOutput") { description = "Attended token embeddings [batch, tokenSeqLen, embedDim]" }
        Output(NUMERIC, "imageOutput") { description = "Attended image embeddings [batch, imageSeqLen, embedDim]" }
        Doc(Language.ANY, DocScope.ALL) {
            """
             SAM-style Two-Way Cross Attention.
             Bidirectional cross-attention where tokens attend to image features and
             image features attend to tokens simultaneously:
               tokenOutput = softmax(tokenQ @ imageK^T * scale) @ imageV
               imageOutput = softmax(imageQ @ tokenK^T * scale) @ tokenV
            """.trimIndent()
        }
    }

    Op("tokenSample") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "TokenSample"
        val logits = Input(NUMERIC, "logits") { description = "Logits tensor. Shape: [vocabSize], [batch, vocabSize], or [batch, seqLen, vocabSize]. For rank-3, samples from the last sequence position." }
        val temperature = Arg(FLOATING_POINT, "temperature") { defaultValue = 0.0; description = "Temperature for sampling. 0 = greedy (argmax)" }
        val topK = Arg(INT, "topK") { defaultValue = 0; description = "Top-K filtering: keep only top K logits. 0 = disabled" }
        val topP = Arg(FLOATING_POINT, "topP") { defaultValue = 0.0; description = "Top-P (nucleus) filtering threshold. 0 = disabled" }
        val seed = Arg(LONG, "seed") { defaultValue = 0; description = "Random seed for sampling. 0 = random" }

        Output(LONG, "output") { description = "Sampled token indices. Shape: [batch] or scalar" }

        Signature(logits)
        Signature(logits, temperature, topK, topP, seed)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Token sampling for LLM inference.

             Full sampling pipeline in a single native GPU call:
               temperature scaling -> top-K filtering -> softmax -> top-P filtering -> sample/argmax

             For greedy decoding (temperature=0 or no top-k/top-p), performs GPU-side argmax
             with shared-memory reduction — avoids transferring the full logits tensor to host.

             Supports rank 1 [vocabSize], rank 2 [batch, vocabSize], and rank 3
             [batch, seqLen, vocabSize] inputs. For rank 3, the last sequence position
             is automatically extracted for sampling.
            """.trimIndent()
        }
    }

    Op("kvScatter") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "KvScatter"
        val present = Input(NUMERIC, "present") { description = "Present KV tensor from decoder output. Shape: [batch, heads, seqLen, dim]" }
        val staticBuf = Input(NUMERIC, "staticBuffer") { description = "Static KV cache buffer. Shape: [batch, heads, maxKvLen, dim]. Updated in-place." }

        val cachePos = Arg(LONG, "cachePos") { description = "Position in static buffer to write the new entry" }
        val numPairs = Arg(INT, "numPairs") { defaultValue = 1; description = "Number of present/static KV pairs. When > 1, inputs are [present_0..N-1, static_0..N-1]" }

        Output(LONG, "output") { description = "Scalar 0 on success" }

        Signature(present, staticBuf, cachePos)
        Signature(present, staticBuf, cachePos, numPairs)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Batch KV cache scatter update for LLM autoregressive decoding.

             Copies a single time-step slice from each present KV tensor into the
             corresponding static KV buffer at a given cache position. Replaces N
             individual Java view+assign calls with a single native kernel launch.

             The present tensor has shape [batch, heads, seqLen, dim] where the new
             token's KV entry is at the last sequence position. This entry is extracted
             and written into the static buffer at cachePos.

             For multiple pairs, inputs are ordered as:
             [present_0, ..., present_{N-1}, static_0, ..., static_{N-1}]
            """.trimIndent()
        }
    }

    Alias(Math(), "tanh")

    Op("fusedElementwiseChain") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedElementwiseChain"

        Input(NUMERIC, "input") { description = "Primary input array" }
        Input(NUMERIC, "secondaryInputs") {
            count = AtLeast(0)
            description = "Optional secondary input arrays for binary ops (add, sub, mul, div)"
        }
        Arg(INT, "opCodes") {
            count = AtLeast(1)
            description = "Op codes: 0=add, 1=sub, 2=mul, 3=div, 10=relu, 11=sigmoid, 12=tanh, 13=gelu, 14=exp, 15=log, 16=abs, 17=neg, 18=square, 19=sqrt, 20=swish, 21=silu, 22=mish, 30=clip, 31=leaky_relu"
        }

        Output(NUMERIC, "output"){ description = "Result of applying the fused element-wise chain" }

        Doc(Language.ANY, DocScope.ALL){
            """
                Executes a fused chain of element-wise operations in a single kernel pass.
                Intermediate values stay in registers instead of global memory. Replaces N separate kernel launches with 1.
            """.trimIndent()
        }
    }

    Op("rope") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "RoPE"
        Input(NUMERIC, "input") { description = "Input tensor [batch, seq_len, num_heads, head_dim]" }
        Arg(INT, "mode") { description = "RoPE mode (default 0)" }
        Arg(INT, "nPast") { description = "Number of past tokens (default 0)" }
        Arg(INT, "nDims") { description = "Dimension subset for rotation (default last dim)" }
        Arg(FLOATING_POINT, "freqBase") { description = "Frequency base (default 10000.0)" }
        Arg(FLOATING_POINT, "freqScale") { description = "Frequency scale (default 1.0)" }
        Output(NUMERIC, "output") { description = "Output with rotary position embeddings applied" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Applies Rotary Position Embedding (RoPE) to the input tensor.
                Encodes position information by rotating pairs of dimensions in the input.
            """.trimIndent()
        }
    }

    Op("silu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SiLU"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Output(NUMERIC, "output") { description = "SiLU(x) = x * sigmoid(x)" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                SiLU (Sigmoid Linear Unit) activation function, also known as Swish.
                Computes f(x) = x * sigmoid(x).
            """.trimIndent()
        }
    }

    Op("applyAlibi") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "ApplyAlibi"
        Input(NUMERIC, "scores") { description = "Attention scores [batch, num_heads, seq_len, kv_len]" }
        Arg(INT, "numHeads") { description = "Number of attention heads" }
        Output(NUMERIC, "output") { description = "Scores with ALiBi position bias applied" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Applies Attention with Linear Biases (ALiBi) position encoding to attention scores.
            """.trimIndent()
        }
    }

    Op("quantizedMatmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "QuantizedMatmul"
        Input(NUMERIC, "a") { description = "First matrix" }
        Input(NUMERIC, "b") { description = "Second matrix" }
        Output(NUMERIC, "output") { description = "Matrix product" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Quantized matrix multiplication. Supports mixed precision (float/int) inputs.
            """.trimIndent()
        }
    }

    Op("loraMatMul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LoraMatMul"
        Input(NUMERIC, "input") { description = "Input [batch, in_features]" }
        Input(NUMERIC, "weight") { description = "Base weight [out_features, in_features]" }
        Input(NUMERIC, "loraA") { description = "LoRA down-projection [r, in_features]" }
        Input(NUMERIC, "loraB") { description = "LoRA up-projection [out_features, r]" }
        Arg(FLOATING_POINT, "scaling") { description = "LoRA scaling factor (default 1.0)" }
        Arg(BOOL, "transposeWeight") { description = "Whether to transpose weight (default true)" }
        Output(NUMERIC, "output") { description = "Result: input @ weight^T + scaling * input @ loraA^T @ loraB^T" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Low-Rank Adaptation (LoRA) fused matrix multiplication.
                Computes base weight matmul + low-rank adapter in a single operation.
            """.trimIndent()
        }
    }

    Op("doraMatMul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "DoraMatMul"
        Input(NUMERIC, "input") { description = "Input [batch, in_features]" }
        Input(NUMERIC, "weight") { description = "Base weight [out_features, in_features]" }
        Input(NUMERIC, "loraA") { description = "LoRA down-projection [r, in_features]" }
        Input(NUMERIC, "loraB") { description = "LoRA up-projection [out_features, r]" }
        Input(NUMERIC, "magnitude") { description = "Per-output magnitude [out_features]" }
        Arg(FLOATING_POINT, "scaling") { description = "LoRA scaling factor (default 1.0)" }
        Output(NUMERIC, "output") { description = "DoRA result with weight-decomposed adaptation" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Weight-Decomposed Low-Rank Adaptation (DoRA) fused matrix multiplication.
                Decomposes weight into magnitude and direction, applies LoRA to direction only.
            """.trimIndent()
        }
    }

    Op("lohaMatMul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LohaMatMul"
        Input(NUMERIC, "input") { description = "Input [batch, in_features]" }
        Input(NUMERIC, "weight") { description = "Base weight [out_features, in_features]" }
        Input(NUMERIC, "lohaA1") { description = "First Hadamard factor A [dim, in_features]" }
        Input(NUMERIC, "lohaB1") { description = "First Hadamard factor B [out_features, dim]" }
        Input(NUMERIC, "lohaA2") { description = "Second Hadamard factor A [dim, in_features]" }
        Input(NUMERIC, "lohaB2") { description = "Second Hadamard factor B [out_features, dim]" }
        Arg(FLOATING_POINT, "scaling") { description = "Scaling factor (default 1.0)" }
        Arg(BOOL, "transposeWeight") { description = "Whether to transpose weight (default true)" }
        Output(NUMERIC, "output") { description = "LoHa result" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Low-Rank Hadamard Product (LoHa) fused matrix multiplication.
                Uses Hadamard product of two low-rank matrices as the adapter.
            """.trimIndent()
        }
    }

    Op("lokrMatMul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "LokrMatMul"
        Input(NUMERIC, "input") { description = "Input [batch, in_features]" }
        Input(NUMERIC, "weight") { description = "Base weight [out_features, in_features]" }
        Input(NUMERIC, "lokrC") { description = "Kronecker factor C [f1, f2]" }
        Input(NUMERIC, "lokrA") { description = "Kronecker factor A [dim, d2]" }
        Input(NUMERIC, "lokrB") { description = "Kronecker factor B [d1, dim]" }
        Arg(INT, "factor1") { description = "First Kronecker factor dimension" }
        Arg(INT, "factor2") { description = "Second Kronecker factor dimension" }
        Arg(FLOATING_POINT, "scaling") { description = "Scaling factor (default 1.0)" }
        Arg(BOOL, "transposeWeight") { description = "Whether to transpose weight (default true)" }
        Output(NUMERIC, "output") { description = "LoKr result" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Low-Rank Kronecker Product (LoKr) fused matrix multiplication.
                Uses Kronecker product of matrices as the adapter.
            """.trimIndent()
        }
    }

    Op("multiLoraMatmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MultiLoraMatmul"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "baseWeight") { description = "Base weight matrix" }
        Input(NUMERIC, "loraAWeights") { description = "Stacked LoRA A weights" }
        Input(NUMERIC, "loraBWeights") { description = "Stacked LoRA B weights" }
        Input(NUMERIC, "adapterIds") { description = "Adapter selection indices" }
        Arg(FLOATING_POINT, "scaling") { description = "LoRA scaling factor" }
        Output(NUMERIC, "output") { description = "Result with per-sample adapter selection" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Multi-adapter LoRA matrix multiplication. Selects different LoRA adapters per batch element.
            """.trimIndent()
        }
    }

    Op("kvCacheQuantize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "KVCacheQuantize"
        Input(NUMERIC, "input") { description = "Key or value tensor to quantize" }
        Arg(INT, "quantType") { description = "Quantization type" }
        Arg(INT, "groupSize") { description = "Group size for quantization" }
        Output(NUMERIC, "output") { description = "Quantized tensor" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Quantizes KV cache tensors for memory-efficient inference.
            """.trimIndent()
        }
    }

    Op("kvCacheDequantize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "KVCacheDequantize"
        Input(NUMERIC, "input") { description = "Quantized key or value tensor" }
        Input(NUMERIC, "scale") { description = "Quantization scales" }
        Arg(INT, "quantType") { description = "Quantization type" }
        Output(NUMERIC, "output") { description = "Dequantized tensor" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Dequantizes quantized KV cache tensors back to floating point.
            """.trimIndent()
        }
    }

    Op("fusedRmsNormSwiglu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedRmsNormSwiGLU"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "gamma") { description = "RMS norm scale" }
        Input(NUMERIC, "wGate") { description = "Gate projection weight" }
        Input(NUMERIC, "wUp") { description = "Up projection weight" }
        Arg(FLOATING_POINT, "epsilon") { description = "Epsilon for numerical stability" }
        Output(NUMERIC, "output") { description = "Result of fused RMSNorm + SwiGLU" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused RMSNorm + SwiGLU activation. Combines normalization and gated activation
                into a single kernel for better memory efficiency.
            """.trimIndent()
        }
    }

    Op("fusedRoPE") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedRoPE"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "ropeCache") { description = "Precomputed RoPE cache (cos/sin)" }
        Arg(INT, "startPosition") { description = "Start position for RoPE application" }
        Output(NUMERIC, "output") { description = "Input with RoPE applied" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused Rotary Position Embedding using precomputed cache.
            """.trimIndent()
        }
    }

    Op("fusedLayerNorm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedLayerNorm"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "gamma") { description = "Scale parameter" }
        Input(NUMERIC, "beta") { description = "Bias parameter" }
        Arg(FLOATING_POINT, "epsilon") { description = "Epsilon for numerical stability" }
        Output(NUMERIC, "output") { description = "Layer-normalized output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused layer normalization. Computes mean, variance, normalize, scale and shift in one pass.
            """.trimIndent()
        }
    }

    Op("fusedGelu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedGELU"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Output(NUMERIC, "output") { description = "GELU(x)" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused Gaussian Error Linear Unit (GELU) activation function.
            """.trimIndent()
        }
    }

    Op("swishMul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SwishMul"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "gate") { description = "Gate tensor" }
        Output(NUMERIC, "output") { description = "swish(input) * gate" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused Swish-Mul: computes swish(input) * gate in a single kernel.
                Used in SwiGLU and similar gated architectures.
            """.trimIndent()
        }
    }

    Op("fusedGemmSwiglu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedGemmSwiglu"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "wGate") { description = "Gate projection weight" }
        Input(NUMERIC, "wUp") { description = "Up projection weight" }
        Output(NUMERIC, "output") { description = "SwiGLU(input @ wGate, input @ wUp)" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused GEMM + SwiGLU: combines two matrix multiplications with gated activation.
            """.trimIndent()
        }
    }

    Op("fusedBiasDropoutResidual") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedBiasDropoutResidual"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "bias") { description = "Bias tensor" }
        Input(NUMERIC, "residual") { description = "Residual connection tensor" }
        Arg(FLOATING_POINT, "dropoutProb") { description = "Dropout probability" }
        Arg(BOOL, "training") { description = "Whether in training mode" }
        Output(NUMERIC, "output") { description = "dropout(input + bias) + residual" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused bias addition, dropout, and residual connection in a single kernel.
            """.trimIndent()
        }
    }

    Op("columnParallelLinear") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "ColumnParallelLinear"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "weight") { description = "Weight matrix" }
        Arg(INT, "tpRank") { description = "Tensor parallel rank" }
        Arg(INT, "tpSize") { description = "Tensor parallel world size" }
        Arg(BOOL, "gatherOutput") { description = "Whether to all-gather output" }
        Output(NUMERIC, "output") { description = "Column-parallel linear output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Column-parallel linear layer for tensor parallelism.
                Splits weight columns across tensor parallel ranks.
            """.trimIndent()
        }
    }

    Op("rowParallelLinear") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "RowParallelLinear"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "weight") { description = "Weight matrix" }
        Arg(INT, "tpRank") { description = "Tensor parallel rank" }
        Arg(INT, "tpSize") { description = "Tensor parallel world size" }
        Arg(BOOL, "reduceOutput") { description = "Whether to all-reduce output" }
        Output(NUMERIC, "output") { description = "Row-parallel linear output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Row-parallel linear layer for tensor parallelism.
                Splits weight rows across tensor parallel ranks.
            """.trimIndent()
        }
    }

    Op("moeGate") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MoeGate"
        Input(NUMERIC, "input") { description = "Input hidden states" }
        Input(NUMERIC, "gateWeights") { description = "Router gate weights" }
        Arg(INT, "numExperts") { description = "Number of experts" }
        Arg(INT, "topK") { description = "Top-K experts to select" }
        Output(NUMERIC, "output") { description = "Gating weights and expert indices" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Mixture of Experts (MoE) gating/routing function.
                Selects top-K experts and computes routing weights.
            """.trimIndent()
        }
    }

    Op("smoothQuant") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SmoothQuant"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "smoothScale") { description = "Smooth quantization scale" }
        Output(NUMERIC, "output") { description = "Smoothly quantized output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                SmoothQuant: migrates quantization difficulty from activations to weights.
            """.trimIndent()
        }
    }

    Op("selectiveScan") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SelectiveScan"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Output(NUMERIC, "output") { description = "Selective scan output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Selective scan operation for state space models (Mamba architecture).
            """.trimIndent()
        }
    }

    Op("gpuTopKSample") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GpuTopKSample"
        Input(NUMERIC, "logits") { description = "Logit scores" }
        Arg(INT, "k") { description = "Number of top candidates" }
        Arg(FLOATING_POINT, "temperature") { description = "Sampling temperature" }
        Output(NUMERIC, "output") { description = "Sampled token indices" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                GPU-accelerated top-K sampling for autoregressive text generation.
            """.trimIndent()
        }
    }

    Op("gpuTopPSample") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GpuTopPSample"
        Input(NUMERIC, "logits") { description = "Logit scores" }
        Arg(FLOATING_POINT, "p") { description = "Cumulative probability threshold (nucleus)" }
        Arg(FLOATING_POINT, "temperature") { description = "Sampling temperature" }
        Output(NUMERIC, "output") { description = "Sampled token indices" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                GPU-accelerated nucleus (top-P) sampling for autoregressive text generation.
            """.trimIndent()
        }
    }

    Op("meanSquare") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MeanSquare"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Output(NUMERIC, "output") { description = "Mean of squared values" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Computes the mean of squared values. Used in RMSNorm and similar operations.
            """.trimIndent()
        }
    }

    Op("decoderMaskedMha") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "DecoderMaskedMha"
        Input(NUMERIC, "query") { description = "Query tensor" }
        Input(NUMERIC, "key") { description = "Key tensor" }
        Input(NUMERIC, "value") { description = "Value tensor" }
        Arg(INT, "numHeads") { description = "Number of attention heads" }
        Arg(BOOL, "isCausal") { description = "Whether to apply causal mask" }
        Output(NUMERIC, "output") { description = "Attention output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Decoder-optimized masked multi-head attention.
                Optimized for autoregressive decoding with incremental KV cache.
            """.trimIndent()
        }
    }

    Op("mlaAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "MLAAttention"
        Input(NUMERIC, "input") { description = "Input hidden states" }
        Input(NUMERIC, "kvDownProj") { description = "KV down-projection weight" }
        Arg(INT, "numHeads") { description = "Number of attention heads" }
        Arg(INT, "latentDim") { description = "Latent dimension for compressed KV" }
        Output(NUMERIC, "output") { description = "Attention output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Multi-head Latent Attention (MLA) from DeepSeek-V2.
                Uses low-rank KV compression for efficient long-context inference.
            """.trimIndent()
        }
    }

    Op("awqMatmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "AwqMatmul"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "weightPacked") { description = "AWQ-packed weight" }
        Input(NUMERIC, "weightScale") { description = "Weight quantization scales" }
        Arg(INT, "groupSize") { description = "Quantization group size" }
        Output(NUMERIC, "output") { description = "Dequantized matmul result" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Activation-aware Weight Quantization (AWQ) matrix multiplication.
            """.trimIndent()
        }
    }

    Op("fp8Matmul") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Fp8Matmul"
        Input(NUMERIC, "a") { description = "First matrix" }
        Input(NUMERIC, "b") { description = "Second matrix" }
        Input(NUMERIC, "scaleA") { description = "Scale for matrix A" }
        Input(NUMERIC, "scaleB") { description = "Scale for matrix B" }
        Output(NUMERIC, "output") { description = "Scaled FP8 matmul result" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                FP8 matrix multiplication with per-tensor scaling.
            """.trimIndent()
        }
    }

    Op("fusedNormQuantize") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "FusedNormQuantize"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "gamma") { description = "Norm scale parameter" }
        Arg(FLOATING_POINT, "epsilon") { description = "Epsilon for normalization" }
        Arg(INT, "quantType") { description = "Quantization type" }
        Output(NUMERIC, "output") { description = "Normalized and quantized output" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Fused normalization + quantization in a single kernel.
            """.trimIndent()
        }
    }

    Op("linearCopy") {
        javaPackage = "org.nd4j.linalg.api.ops.custom"
        javaOpClass = "LinearCopy"
        Input(NUMERIC, "input") { description = "Source tensor" }
        Output(NUMERIC, "output") { description = "Contiguous copy of input" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Creates a contiguous copy of the input tensor with linear (row-major) memory layout.
            """.trimIndent()
        }
    }

    Op("reshapeNoCopy") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.shape"
        javaOpClass = "ReshapeNoCopy"
        Input(NUMERIC, "input") { description = "Input tensor" }
        Input(NUMERIC, "shape") { description = "Target shape" }
        Output(NUMERIC, "output") { description = "Reshaped view (no data copy)" }
        Doc(Language.ANY, DocScope.ALL) {
            """
                Reshapes a tensor without copying data. Returns a view if possible.
                If the reshape cannot be done without copying, this op will fail.
            """.trimIndent()
        }
    }

    Op("gatedDeltaRule") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GatedDeltaRule"
        val q = Input(NUMERIC, "q") { description = "Query tensor [batch, seqLen, numHeads, headDimK]" }
        val k = Input(NUMERIC, "k") { description = "Key tensor [batch, seqLen, numHeads, headDimK] (L2-normalized)" }
        val v = Input(NUMERIC, "v") { description = "Value tensor [batch, seqLen, numHeads, headDimV]" }
        val beta = Input(NUMERIC, "beta") { description = "Per-step learning rate [batch, seqLen, numHeads]" }
        val gate = Input(NUMERIC, "gate") { description = "Decay gate (pre-exp) [batch, seqLen, numHeads]" }
        val stateIn = Input(NUMERIC, "stateIn") { description = "Previous recurrent state [batch, numHeads, headDimK, headDimV]"; defaultValue = null }

        Output(NUMERIC, "output") { description = "Attention output [batch, seqLen, numHeads, headDimV]" }
        Output(NUMERIC, "stateOut") { description = "Final recurrent state [batch, numHeads, headDimK, headDimV]" }

        AllParamSignature()
        Signature(q, k, v, beta, gate)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Gated Delta Rule (arXiv:2412.06464, ICLR 2025, NVIDIA Research).

             Recurrent linear attention with gated exponential decay and delta update rule:
               S_t = exp(g_t) * S_{t-1} + beta_t * k_t (x) (v_t - exp(g_t) * S_{t-1}^T * k_t)
               output_t = S_t^T * q_t

             State shape: [batch, numHeads, headDimK, headDimV].
             Used in Gated Delta Networks (Qwen3.5 and other production models).
            """.trimIndent()
        }
    }

    Op("causalConv1d") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "CausalConv1d"
        val x = Input(NUMERIC, "x") { description = "Input sequence [batch, seqLen, dim]" }
        val weight = Input(NUMERIC, "weight") { description = "Depthwise conv weights [dim, kernelSize] (wFormat=0) or [kernelSize, dim] (wFormat=1)" }
        val bias = Input(NUMERIC, "bias") { description = "Bias [dim]"; defaultValue = null }
        val stateIn = Input(NUMERIC, "convStateIn") { description = "Conv state for autoregressive decode [batch, dim, kernelSize-1]"; defaultValue = null }
        val activation = Arg(INT, "activation") { description = "Activation function (0=none, 1=silu)"; defaultValue = 0 }
        val wFormat = Arg(INT, "wFormat") { description = "Weight format (0=[D,K] PyTorch/ONNX default, 1=[K,D] TensorFlow)"; defaultValue = 0 }

        Output(NUMERIC, "output") { description = "Convolved output [batch, seqLen, dim]" }
        Output(NUMERIC, "stateOut") { description = "Updated conv state [batch, dim, kernelSize-1]" }

        AllParamSignature()
        Signature(x, weight)
        Signature(x, weight, bias)
        Signature(x, weight, bias, stateIn)
        Signature(x, weight, bias, stateIn, activation)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Causal depthwise 1D convolution with state for autoregressive decoding.

             Performs a causal (left-padded) depthwise 1D convolution.
             Used in Gated Delta Networks (GDN) and Mamba architectures.
             The state output preserves the last (kernelSize-1) input elements
             for use as initial state in the next autoregressive step.
            """.trimIndent()
        }
    }

    Op("gatedDeltaNetBlock") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "GatedDeltaNetBlock"
        val x = Input(NUMERIC, "x") { description = "Input tensor [batch, seqLen, modelDim]" }
        val wqkv = Input(NUMERIC, "wqkv") { description = "QKV projection weights [modelDim, qkvDim]" }
        val wbeta = Input(NUMERIC, "wbeta") { description = "Beta projection weights [modelDim, numHeads]" }
        val wgate = Input(NUMERIC, "wgate") { description = "Gate projection weights [modelDim, numHeads]" }
        val wout = Input(NUMERIC, "wout") { description = "Output projection weights [numHeads*headDimV, modelDim]" }
        val convWeight = Input(NUMERIC, "convWeight") { description = "Causal conv1d weights [modelDim, kernelSize]" }
        val convBias = Input(NUMERIC, "convBias") { description = "Causal conv1d bias [modelDim]" }
        val stateIn = Input(NUMERIC, "recurrentStateIn") { description = "Previous recurrent state [batch, numHeads, headDimK, headDimV]"; defaultValue = null }
        val numHeads = Arg(INT, "numHeads") { description = "Number of attention heads (H)" }
        val headDimK = Arg(INT, "headDimK") { description = "Key head dimension (D_k)" }
        val headDimV = Arg(INT, "headDimV") { description = "Value head dimension (D_v)" }
        val rmsNormEpsilon = Arg(FLOATING_POINT, "rmsNormEpsilon") { description = "RMSNorm epsilon"; defaultValue = 1e-5 }

        Output(NUMERIC, "output") { description = "Layer output [batch, seqLen, modelDim]" }
        Output(NUMERIC, "recurrentStateOut") { description = "Final recurrent state [batch, numHeads, headDimK, headDimV]" }
        Output(NUMERIC, "convStateOut") { description = "Final conv state [batch, modelDim, kernelSize-1]" }

        AllParamSignature()
        Signature(x, wqkv, wbeta, wgate, wout, convWeight, convBias, numHeads, headDimK, headDimV)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Full Gated Delta Network (GDN) layer block.

             Fuses the complete GDN layer pipeline:
               1. Linear projection (QKV + beta + gate)
               2. Causal depthwise conv1d with SiLU activation
               3. Gated delta rule recurrent state update
               4. RMSNorm + Swish gate
               5. Output linear projection

             This is the building block for Gated Delta Network architectures
             (arXiv:2412.06464, ICLR 2025).
            """.trimIndent()
        }
    }

    Op("perLayerEmbedding") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "PerLayerEmbedding"
        val hiddenStates = Input(NUMERIC, "hiddenStates") { description = "Hidden states [batch, seqLen, hiddenDim]" }
        val pleWeight = Input(NUMERIC, "pleWeight") { description = "Per-layer embedding table [vocabSize, hiddenDim]" }
        val tokenIds = Input(INT, "tokenIds") { description = "Token IDs [batch, seqLen]" }
        val scale = Arg(FLOATING_POINT, "scale") { description = "Scale factor for the embedding addition"; defaultValue = 1.0 }

        Output(NUMERIC, "output") { description = "Output [batch, seqLen, hiddenDim]" }

        AllParamSignature()
        Signature(hiddenStates, pleWeight, tokenIds)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Per-Layer Embedding (Gemma 4).

             Adds a per-layer residual from a second embedding table to the hidden states:
               output = hiddenStates + pleWeight[tokenIds] * scale

             Each decoder layer receives a small additive signal from a dedicated
             embedding table indexed by the original token IDs. This is computed once
             before multimodal features merge into the embedding sequence, since PLE
             relies on token IDs that are lost once multimodal features replace placeholders.
            """.trimIndent()
        }
    }

    Op("sharedKvAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SharedKvAttention"
        val query = Input(NUMERIC, "query") { description = "Query [batch, seqLen, numHeads, headDim]" }
        val sharedKey = Input(NUMERIC, "sharedKey") { description = "Key from donor layer [batch, kvSeqLen, numKvHeads, headDim]" }
        val sharedValue = Input(NUMERIC, "sharedValue") { description = "Value from donor layer [batch, kvSeqLen, numKvHeads, headDim]" }
        val mask = Input(NUMERIC, "mask") { description = "Attention mask [batch, 1, seqLen, kvSeqLen]"; defaultValue = null }
        val numHeads = Arg(INT, "numHeads") { description = "Number of query heads" }
        val numKvHeads = Arg(INT, "numKvHeads") { description = "Number of key-value heads (for GQA)" }
        val causal = Arg(INT, "causal") { description = "Causal masking (0=bidirectional, 1=causal)"; defaultValue = 1 }
        val slidingWindowSize = Arg(INT, "slidingWindowSize") { description = "Sliding window size (0=disabled)"; defaultValue = 0 }
        val scale = Arg(FLOATING_POINT, "scale") { description = "Attention scale (0=auto: 1/sqrt(headDim))"; defaultValue = 0.0 }

        Output(NUMERIC, "output") { description = "Attention output [batch, seqLen, numHeads, headDim]" }

        AllParamSignature()
        Signature(query, sharedKey, sharedValue, numHeads, numKvHeads)
        Signature(query, sharedKey, sharedValue, mask, numHeads, numKvHeads)
        Signature(query, sharedKey, sharedValue, mask, numHeads, numKvHeads, causal, slidingWindowSize)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Shared KV Attention (Gemma 4).

             Grouped-query attention where K/V come from a donor layer rather than
             being projected from the current hidden state. The last N layers reuse
             K/V tensors produced by an earlier layer (the last non-shared layer of
             the same attention type: sliding or full). This reduces memory and
             compute with minimal quality impact.

             Supports causal masking and optional sliding window for local attention.
            """.trimIndent()
        }
    }

    Op("dualRoPE") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "DualRoPE"
        val input = Input(NUMERIC, "input") { description = "Input tensor [batch, seqLen, numHeads, headDim] - headDim must be even" }
        val attentionType = Arg(INT, "attentionType") { description = "Attention type (0=local/sliding-window, 1=global/full-context)"; defaultValue = 0 }
        val positionOffset = Arg(INT, "positionOffset") { description = "Position offset for KV cache continuation"; defaultValue = 0 }
        val localFreqBase = Arg(FLOATING_POINT, "localFreqBase") { description = "RoPE frequency base for local/sliding-window layers"; defaultValue = 10000.0 }
        val globalFreqBase = Arg(FLOATING_POINT, "globalFreqBase") { description = "RoPE frequency base for global/full-context layers"; defaultValue = 1000000.0 }
        val localFreqScale = Arg(FLOATING_POINT, "localFreqScale") { description = "RoPE frequency scale for local layers"; defaultValue = 1.0 }
        val globalFreqScale = Arg(FLOATING_POINT, "globalFreqScale") { description = "RoPE frequency scale for global layers"; defaultValue = 1.0 }

        Output(NUMERIC, "output") { description = "Output with rotary embeddings applied [batch, seqLen, numHeads, headDim]" }

        AllParamSignature()
        Signature(input)
        Signature(input, attentionType)
        Signature(input, attentionType, positionOffset)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Dual Rotary Position Embedding (Gemma 4).

             Applies two different RoPE configurations depending on attention type:
             - Standard RoPE (localFreqBase) for sliding-window (local) attention layers
             - Proportional RoPE (globalFreqBase) for global full-context attention layers

             This enables longer context windows by using different position encoding
             frequencies for local vs global attention. For each dimension pair (2i, 2i+1):
               theta_i = freqBase ^ (-2i / headDim) * freqScale
               output[2i]   = input[2i] * cos(pos * theta) - input[2i+1] * sin(pos * theta)
               output[2i+1] = input[2i] * sin(pos * theta) + input[2i+1] * cos(pos * theta)
            """.trimIndent()
        }
    }

    Op("turboQuantAttention") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "TurboQuantAttention"

        val query = Input(NUMERIC, "query") { description = "Query tensor [B, H, Sq, D]" }
        val kMse = Input(NUMERIC, "kMse") { description = "MSE-reconstructed keys [B, H, Sk, D]" }
        val qjlSigns = Input(INT, "qjlSigns") { description = "QJL sign bits [B, H, Sk, D] INT8" }
        val residualNorms = Input(NUMERIC, "residualNorms") { description = "Residual L2 norms [B, H, Sk]" }
        val qjlMatrix = Input(NUMERIC, "qjlMatrix") { description = "QJL projection matrix [D, D]" }
        val values = Input(NUMERIC, "values") { description = "Dequantized values [B, H, Sk, D]" }
        val attentionMask = Input(NUMERIC, "attentionMask") { description = "Attention mask [B, 1, 1, Sk]" }
        val numHeads = Arg(INT, "numHeads") { description = "Number of attention heads" }
        val headDim = Arg(INT, "headDim") { description = "Dimension per head" }
        val scale = Arg(FLOATING_POINT, "scale") { description = "Attention scale (0 = auto: 1/sqrt(headDim))"; defaultValue = 0.0 }

        Output(NUMERIC, "output") { description = "Attention output [B, H, Sq, D]" }

        AllParamSignature()
        Signature(query, kMse, qjlSigns, residualNorms, qjlMatrix, values, attentionMask, numHeads, headDim)

        Doc(Language.ANY, DocScope.ALL) {
            """
             TurboQuant asymmetric attention with compressed keys.

             Computes scaled dot-product attention using compressed key representations
             from TurboQuant's two-stage quantization (ICLR 2026). The asymmetric inner
             product estimator combines MSE reconstruction with QJL correction:

               score(q, k) ≈ <q, k_mse> + ||r|| * sqrt(π/2)/m * <S@q, signs>

             Keys use full two-stage compression (MSE + QJL) for asymmetric attention.
             Values use MSE-only decompression (error averages out in softmax-weighted sum).
            """.trimIndent()
        }
    }

    Op("squaredRelu") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "SquaredReLU"
        Input(NUMERIC, "x") { description = "Input variable" }
        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Squared ReLU activation function: out = max(0, x)^2.
             Used in Nemotron and other NVIDIA model architectures.
            """.trimIndent()
        }
    }

    Op("mamba2Ssm") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        javaOpClass = "Mamba2SSM"
        val x = Input(NUMERIC, "x") { description = "Input tensor [batch, seqLen, D] where D = numHeads * headDim" }
        val a = Input(NUMERIC, "A") { description = "Per-head scalar decay in log-space [numHeads]" }
        val b = Input(NUMERIC, "B") { description = "Input-dependent state expansion [batch, seqLen, stateDim]" }
        val c = Input(NUMERIC, "C") { description = "Input-dependent state contraction [batch, seqLen, stateDim]" }
        val dt = Input(NUMERIC, "dt") { description = "Discretization timestep (post-softplus) [batch, seqLen, numHeads]" }

        val numHeads = Arg(INT, "numHeads") { description = "Number of SSM heads (H)" }
        val headDim = Arg(INT, "headDim") { description = "Dimension per head (P = D/H)" }
        val stateDim = Arg(INT, "stateDim") { description = "State dimension (N)" }

        Output(NUMERIC, "output") { description = "SSM output [batch, seqLen, D]" }
        Output(NUMERIC, "stateOut") { description = "Final recurrent state [batch, numHeads, stateDim, headDim]" }

        Signature(x, a, b, c, dt, numHeads, headDim, stateDim)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Mamba-2 State Space Model (SSD) - head-structured recurrence.

             Implements the Mamba-2 SSD recurrence with scalar per-head decay:
               h_t = exp(A * dt) * h_{t-1} + (B * dt) outer x_t
               y_t = C * h_t + D * x_t

             Unlike Mamba-1 (selective_scan) which uses per-element diagonal state,
             Mamba-2 uses head-structured state for improved hardware utilization.

             See "Transformers are SSMs: Generalized Models and Efficient Algorithms Through
             Structured State Space Duality" (https://arxiv.org/abs/2405.21060)
            """.trimIndent()
        }
    }
}
