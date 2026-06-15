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
package org.nd4j.samediff.frameworkimport.onnx.export

import onnx.Onnx
import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.Op
import org.nd4j.samediff.frameworkimport.onnx.*

/**
 * Mapper for converting SameDiff operations to ONNX operations.
 *
 * This class provides the reverse mapping of OnnxOpDeclarations,
 * enabling export of SameDiff models to ONNX format.
 *
 * @author Adam Gibson
 */
object SameDiffToOnnxOpMapper {

    /**
     * Mapping from SameDiff op names to ONNX op names.
     */
    val sameDiffToOnnxOps: Map<String, String> = mapOf(
        // Arithmetic operations
        "add" to "Add",
        "subtract" to "Sub",
        "multiply" to "Mul",
        "divide" to "Div",
        "mod" to "Mod",
        "pow_pairwise" to "Pow",
        "neg" to "Neg",
        "abs" to "Abs",
        "floor" to "Floor",
        "ceil" to "Ceil",
        "round" to "Round",
        "sqrt" to "Sqrt",
        "reciprocal" to "Reciprocal",

        // Trigonometric operations
        "sin" to "Sin",
        "cos" to "Cos",
        "tan" to "Tan",
        "asin" to "Asin",
        "acos" to "Acos",
        "atan" to "Atan",
        "sinh" to "Sinh",
        "cosh" to "Cosh",
        "tanh" to "Tanh",
        "asinh" to "Asinh",
        "acosh" to "Acosh",
        "atanh" to "Atanh",

        // Activation functions
        "relu" to "Relu",
        "sigmoid" to "Sigmoid",
        "softmax" to "Softmax",
        "log_softmax" to "LogSoftmax",
        "elu" to "Elu",
        "selu" to "Selu",
        "softplus" to "Softplus",
        "softsign" to "Softsign",
        "leakyrelu" to "LeakyRelu",
        "hard_sigmoid" to "HardSigmoid",
        "thresholdedrelu" to "ThresholdedRelu",
        "prelu" to "PRelu",

        // Pooling operations
        "maxpool2d" to "MaxPool",
        "avgpool2d" to "AveragePool",
        "global_avgpool2d" to "GlobalAveragePool",
        "global_maxpool2d" to "GlobalMaxPool",

        // Convolution operations
        "conv2d" to "Conv",
        "conv3d" to "Conv",
        "deconv2d" to "ConvTranspose",

        // Shape operations
        "reshape" to "Reshape",
        "flatten_2d" to "Flatten",
        "squeeze" to "Squeeze",
        "expand_dims" to "Unsqueeze",
        "shape_of" to "Shape",
        "size" to "Size",
        "concat" to "Concat",
        "tile" to "Tile",
        "depth_to_space" to "DepthToSpace",
        "space_to_depth" to "SpaceToDepth",
        "transpose" to "Transpose",
        "permute" to "Transpose",
        "split" to "Split",
        "slice" to "Slice",
        "strided_slice" to "Slice",
        "gather" to "Gather",
        "gather_nd" to "GatherND",
        "scatter_update" to "ScatterElements",

        // Reduction operations
        "reduce_sum" to "ReduceSum",
        "reduce_mean" to "ReduceMean",
        "reduce_max" to "ReduceMax",
        "reduce_min" to "ReduceMin",
        "reduce_prod" to "ReduceProd",
        "reduce_norm1" to "ReduceL1",
        "reduce_norm2" to "ReduceL2",
        "reduce_logsumexp" to "ReduceLogSumExp",
        "argmax" to "ArgMax",
        "argmin" to "ArgMin",

        // Math operations
        "exp" to "Exp",
        "log" to "Log",
        "erf" to "Erf",
        "sign" to "Sign",
        "clip_by_value" to "Clip",
        "matmul" to "MatMul",
        "mmul" to "MatMul",
        "batch_mmul" to "MatMul",

        // Comparison operations
        "equals" to "Equal",
        "not_equals" to "NotEqual",
        "greater" to "Greater",
        "greater_equal" to "GreaterOrEqual",
        "less" to "Less",
        "less_equal" to "LessOrEqual",

        // Logical operations
        "or" to "Or",
        "and" to "And",
        "not" to "Not",
        "bitwise_xor" to "Xor",

        // Special operations
        "identity" to "Identity",
        "isinf" to "IsInf",
        "isnan" to "IsNaN",
        "pad" to "Pad",
        "range" to "Range",
        "top_k" to "TopK",
        "lrn" to "LRN",
        "matrix_determinant" to "Det",
        "cast" to "Cast",
        "one_hot" to "OneHot",

        // Random operations
        "random_normal" to "RandomNormal",
        "randomuniform" to "RandomUniform",

        // Normalization (handled by hooks for complex cases)
        "batchnorm" to "BatchNormalization",
        "layer_norm" to "LayerNormalization",

        // LSTM (handled by hook)
        "lstmLayer" to "LSTM",

        // Non-max suppression
        "non_max_suppression_v3" to "NonMaxSuppression"
    )

    /**
     * Mapping of SameDiff input parameter names to ONNX input names.
     * Key is (opName, sameDiffInputName), value is onnxInputName.
     */
    val inputMappings: Map<Pair<String, String>, String> = mapOf(
        // Binary operations - most use (A, B) in ONNX
        Pair("add", "input") to "A",
        Pair("add", "y") to "B",
        Pair("subtract", "input") to "A",
        Pair("subtract", "y") to "B",
        Pair("multiply", "input") to "A",
        Pair("multiply", "y") to "B",
        Pair("divide", "input") to "A",
        Pair("divide", "y") to "B",

        // Unary operations - most use X in ONNX
        Pair("relu", "input") to "X",
        Pair("sigmoid", "input") to "X",
        Pair("tanh", "input") to "X",
        Pair("softmax", "input") to "input",
        Pair("abs", "input") to "X",
        Pair("neg", "input") to "X",
        Pair("sqrt", "input") to "X",
        Pair("exp", "input") to "X",
        Pair("log", "input") to "X",

        // Pooling
        Pair("maxpool2d", "input") to "X",
        Pair("avgpool2d", "input") to "X",

        // Conv
        Pair("conv2d", "input") to "X",
        Pair("conv2d", "weights") to "W",
        Pair("conv2d", "bias") to "B",

        // Reduction operations
        Pair("reduce_sum", "input") to "data",
        Pair("reduce_mean", "input") to "data",
        Pair("reduce_max", "input") to "data",
        Pair("reduce_min", "input") to "data",

        // Shape operations
        Pair("reshape", "input") to "data",
        Pair("reshape", "shape") to "shape",
        Pair("concat", "input") to "inputs",
        Pair("transpose", "input") to "data"
    )

    /**
     * Mapping of SameDiff attribute names to ONNX attribute names.
     */
    val attributeMappings: Map<Pair<String, String>, String> = mapOf(
        // Softmax
        Pair("softmax", "dimension") to "axis",
        Pair("log_softmax", "dimension") to "axis",

        // Conv/Pool attributes
        Pair("conv2d", "sH") to "strides",
        Pair("conv2d", "sW") to "strides",
        Pair("conv2d", "pH") to "pads",
        Pair("conv2d", "pW") to "pads",
        Pair("conv2d", "dH") to "dilations",
        Pair("conv2d", "dW") to "dilations",
        Pair("conv2d", "kH") to "kernel_shape",
        Pair("conv2d", "kW") to "kernel_shape",

        // Pool attributes
        Pair("maxpool2d", "kH") to "kernel_shape",
        Pair("maxpool2d", "kW") to "kernel_shape",
        Pair("maxpool2d", "sH") to "strides",
        Pair("maxpool2d", "sW") to "strides",
        Pair("maxpool2d", "pH") to "pads",
        Pair("maxpool2d", "pW") to "pads",

        // Reduction
        Pair("reduce_sum", "dimensions") to "axes",
        Pair("reduce_sum", "keepDims") to "keepdims",
        Pair("reduce_mean", "dimensions") to "axes",
        Pair("reduce_mean", "keepDims") to "keepdims",

        // Concat
        Pair("concat", "concatDimension") to "axis",

        // Leaky ReLU
        Pair("leakyrelu", "alpha") to "alpha",

        // ELU
        Pair("elu", "alpha") to "alpha",

        // LRN
        Pair("lrn", "depth") to "size",
        Pair("lrn", "alpha") to "alpha",
        Pair("lrn", "beta") to "beta",
        Pair("lrn", "bias") to "bias"
    )

    /**
     * Get the ONNX op name for a SameDiff operation.
     *
     * @param sameDiffOpName The SameDiff operation name
     * @return The ONNX operation name, or null if not found
     */
    fun getOnnxOpName(sameDiffOpName: String): String? {
        return sameDiffToOnnxOps[sameDiffOpName]
    }

    /**
     * Get the ONNX input name for a SameDiff input.
     *
     * @param opName The SameDiff operation name
     * @param inputName The SameDiff input name
     * @return The ONNX input name, or the original name if no mapping exists
     */
    fun getOnnxInputName(opName: String, inputName: String): String {
        return inputMappings[Pair(opName, inputName)] ?: inputName
    }

    /**
     * Get the ONNX attribute name for a SameDiff attribute.
     *
     * @param opName The SameDiff operation name
     * @param attrName The SameDiff attribute name
     * @return The ONNX attribute name, or the original name if no mapping exists
     */
    fun getOnnxAttributeName(opName: String, attrName: String): String {
        return attributeMappings[Pair(opName, attrName)] ?: attrName
    }

    /**
     * Check if a SameDiff operation is supported for export.
     *
     * @param sameDiffOpName The SameDiff operation name
     * @return true if the operation can be exported to ONNX
     */
    fun isSupported(sameDiffOpName: String): Boolean {
        return sameDiffToOnnxOps.containsKey(sameDiffOpName) ||
                PostExportHookRegistry.findHook(sameDiffOpName) != null
    }

    /**
     * Get all supported SameDiff operation names.
     */
    fun getSupportedOps(): Set<String> {
        val ops = sameDiffToOnnxOps.keys.toMutableSet()
        // Add ops handled by hooks
        PostExportHookRegistry.getAllHooks().forEach { hook ->
            sameDiffToOnnxOps.keys.filter { hook.canHandle(it) }.forEach { ops.add(it) }
        }
        return ops
    }

    /**
     * Create ONNX attributes for a SameDiff operation.
     *
     * @param sd The SameDiff instance
     * @param op The operation
     * @param opName The SameDiff operation name
     * @return List of ONNX attribute protos
     */
    fun createAttributes(
        sd: SameDiff,
        op: DifferentialFunction,
        opName: String
    ): List<Onnx.AttributeProto> {
        val attributes = mutableListOf<Onnx.AttributeProto>()

        when (op) {
            is DynamicCustomOp -> {
                // Handle integer arguments
                val iArgs = op.iArgs()
                val tArgs = op.tArgs()
                val bArgs = op.bArgs()

                // Create attributes based on the operation type
                when (opName) {
                    "softmax", "log_softmax" -> {
                        if (iArgs.isNotEmpty()) {
                            attributes.add(AttributeProto {
                                name = "axis"
                                i = iArgs[0]
                                type = Onnx.AttributeProto.AttributeType.INT
                            })
                        }
                    }
                    "leakyrelu" -> {
                        if (tArgs.isNotEmpty()) {
                            attributes.add(AttributeProto {
                                name = "alpha"
                                f = tArgs[0].toFloat()
                                type = Onnx.AttributeProto.AttributeType.FLOAT
                            })
                        }
                    }
                    "elu" -> {
                        if (tArgs.isNotEmpty()) {
                            attributes.add(AttributeProto {
                                name = "alpha"
                                f = tArgs[0].toFloat()
                                type = Onnx.AttributeProto.AttributeType.FLOAT
                            })
                        }
                    }
                    "concat" -> {
                        if (iArgs.isNotEmpty()) {
                            attributes.add(AttributeProto {
                                name = "axis"
                                i = iArgs[0]
                                type = Onnx.AttributeProto.AttributeType.INT
                            })
                        }
                    }
                    "reduce_sum", "reduce_mean", "reduce_max", "reduce_min", "reduce_prod" -> {
                        // keepdims is typically the first boolean arg
                        val keepDims = if (bArgs.isNotEmpty()) bArgs[0] else true
                        attributes.add(AttributeProto {
                            name = "keepdims"
                            i = if (keepDims) 1 else 0
                            type = Onnx.AttributeProto.AttributeType.INT
                        })
                    }
                    "transpose" -> {
                        if (iArgs.isNotEmpty()) {
                            attributes.add(AttributeProto {
                                name = "perm"
                                addAllInts(iArgs.toList())
                                type = Onnx.AttributeProto.AttributeType.INTS
                            })
                        }
                    }
                }
            }
        }

        return attributes
    }
}
