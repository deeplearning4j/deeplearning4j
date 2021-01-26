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
        javaOpClass = "DropOut"
        legacy = true
        Input(NUMERIC, "input") { description = "Input array" }
        Arg(NUMERIC, "inputRetainProbability") { description = "Probability of retaining an input (set to 0 with probability 1-p)" }

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
        Input(NUMERIC, "bias") { description = "Optional bias variable (may be null)" /*; optional = true*/ }
        Output(NUMERIC, "output") { description = "Output variable" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             ReLU (Rectified Linear Unit) layer operation: out = relu(mmul(in,w) + bias)
             Note that bias array is optional
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

    Op("softmaxDerivative") {
        javaPackage = "org.nd4j.linalg.api.ops.impl.transforms.gradient"
        javaOpClass = "SoftmaxBp"
        Input(NUMERIC, "x") { description = "Softmax input" }
        Input(NUMERIC, "wrt") { description = "Gradient at output, dL/dx" }
        Arg(INT, "dimension"){description = "Softmax dimension"}

        Output(NUMERIC, "output") { description = "" }

        Doc(Language.ANY, DocScope.ALL) {
            """
                Softmax derivative function
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
        val dim = Arg(INT, "dimensions") { count = AtLeast(1); description = "Dimensions to perform layer norm over - dimension=1 for 2d/MLP data, dimension=1,2,3 for CNNs" }

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

    Alias(Math(), "tanh")
}
