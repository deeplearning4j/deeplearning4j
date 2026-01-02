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


    Op("dotProductAttentionV2") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.custom"
        val q = Input(NUMERIC, "queries") { description = "Query tensor. Shape: [batchSize, numQueries, queryDim] or [batchSize, numQueries, numHeads, headDim] for flash attention" }
        val v = Input(NUMERIC, "values") { description = "Value tensor. Shape: [batchSize, numValues, valueDim] or [batchSize, numValues, numHeads, headDim]" }
        val k = Input(NUMERIC, "keys") { description = "Key tensor. Shape: [batchSize, numValues, keyDim] or [batchSize, numValues, numHeads, headDim]" }
        val queryMask = Input(NUMERIC, "queryMask") { description = "Query mask tensor (optional). Shape: [batchSize, numQueries]"; defaultValue = null }
        val valueMask = Input(NUMERIC, "valueMask") { description = "Value mask tensor (optional). Shape: [batchSize, numValues]"; defaultValue = null }

        val s = Arg(FLOATING_POINT, "scaleFactor") { defaultValue = 0.0; description = "Scaling factor applied to attention scores. 0 = auto (1/sqrt(headDim))" }
        val dropout = Arg(FLOATING_POINT, "dropoutProbability") { defaultValue = 0.0; description = "Dropout probability applied to attention weights" }
        val useCausalMask = Arg(BOOL, "useCausalMask") { defaultValue = false; description = "Whether to apply causal mask for autoregressive tasks" }
        val training = Arg(BOOL, "training") { defaultValue = false; description = "Whether in training mode (affects dropout)" }

        Output(NUMERIC, "output") { description = "Output tensor. Shape: [batchSize, numQueries, valueDim] or [batchSize, numQueries, numHeads, headDim]" }

        // Standard signature matching Java constructor
        Signature(q, v, k, queryMask, valueMask, s, dropout, useCausalMask, training)

        Doc(Language.ANY, DocScope.ALL) {
            """
             Dot product attention operation with flash attention and KV cache support.

             out = softmax(Q * K^T / scale) * V

             For 4D inputs [batch, seq, heads, dim], uses memory-efficient flash attention algorithm.
             For 2D/3D inputs, uses standard attention computation.

             Flash attention features:
             - O(N) memory complexity instead of O(N^2)
             - Tiled computation with online softmax
             - Supports grouped query attention (GQA) where numHeads > numKvHeads

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

    Alias(Math(), "tanh")
}
