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
package org.nd4j.samediff.frameworkimport.tensorflow.definitions


import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.nameSpaceTensorFromNDarray
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder
import org.nd4j.samediff.frameworkimport.tensorflow.*
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcess
import org.tensorflow.framework.*

val tensorflowOpRegistry = OpMappingRegistry<GraphDef,NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue>(
        "tensorflow",OpDescriptorLoaderHolder.nd4jOpDescriptor)

fun registry(): OpMappingRegistry<GraphDef,NodeDef,OpDef, TensorProto,DataType, OpDef.AttrDef,AttrValue> {
        return tensorflowOpRegistry
}

val singleTransformArgs = mapOf(
        "Abs" to "abs",
        "Acos" to "acos",
        "Acosh" to "acosh",
        "Asin" to "asin",
        "Asinh" to "asinh",
        "Atan" to "atan",
        "Atanh" to "atanh",
        "Ceil" to "ceil",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "Erf" to "erf",
        "Erfc" to "erfc",
        "Exp" to "exp",
        "Expm1" to "expm1",
        "Floor" to "floor",
        "Log" to "log",
        "Log1p" to "log1p",
        "Neg" to "neg",
        "Rint" to "rint",
        "Round" to "round",
        "Rsqrt" to "rsqrt",
        "Sigmoid" to "sigmoid",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Square" to "square",
        "Sqrt" to "sqrt",
        "Tan" to "tan",
        "Tanh" to "tanh"
)

val elementWiseTransformOps = mapOf(
        "Add" to "add",
        "AddV2" to "add",
        "Div" to "divide",
        "Greater" to "greater",
        "GreaterEqual" to "greater_equal",
        "Less" to "less",
        "LessEqual" to "less_equal",
        "Mul" to "multiply",
        "Sub" to "subtract",
        "Maximum" to "maximum",
        "FloorDiv" to "floordiv",
        "Mod" to "mod",
        "FloorMod" to "floormod",
        "SquaredDifference" to "squaredsubtract",
        "NotEqual" to "not_equals",
        "RealDiv" to "realdiv",
        "RightShift" to "rshift_bits",
        "Atan2" to "tf_atan2",
        "TruncateDiv" to "truncatediv"
)


val reduceOps = mapOf(
        "All" to "all",
        "Any" to "any",
        "Mean" to "reduce_mean",
        "Prod" to "reduce_prod",
        "Sum" to "reduce_sum",
        "Min" to "reduce_min",
        "Max" to "reduce_max",

        )


val pairwiseReduceOps = mapOf(
        "EuclideanNorm" to "euclidean"
)


val addN = TensorflowMappingProcess(
        inputFrameworkOpName = "AddN",
        opName = "mergesum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs"))),
        opMappingRegistry = tensorflowOpRegistry
)


val assert = mapTensorNamesWithOp(inputFrameworkOpName = "Assert",opName = "Assert",
        tensorNames = mutableMapOf("input" to "condition"),
        tensorflowOpRegistry = tensorflowOpRegistry)


/**
 *
 * Note that angle only supports complex inputs and outputs.
 * We don't support complex in nd4j so we just output zeros.
 */
/*val angleRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Angle",
        opName = "zeros_like",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        opMappingRegistry = tensorflowOpRegistry
)*/

/*
val approxEqualRule = TensorflowMappingProcess(
        inputFrameworkOpName = "Equal",
        opName = "equals_with_eps",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = listOf(valueMapping(mapOf("eps" to "tolerance")),
                //TODO: note dimensions isn't on the TF op, need to investigate if there is a better default here
                intConstant(inputName = "dimensions",constantValue = 0 ,argumentIndex = 0)[0],
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0]))
*/


val argMaxRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMax",
        opName = "argmax",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0])

)

val argMinRule = TensorflowMappingProcess(
        inputFrameworkOpName = "ArgMin",
        opName = "argmin",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "keepDims",constantValue = false,argumentIndex = 0)[0])

)
/*
val reduceLogSumExp = TensorflowMappingProcess(
        inputFrameworkOpName = "CumulativeLogsumexp",
        opName = "reduce_logsumexp",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        attributeMappingRules = listOf(
                ndarrayToIntList(mutableMapOf("dimensions" to "axis")),
                booleanConstant(inputName = "keepDims",constantValue = true,argumentIndex = 0)[0])

)*/

/**
 * Note: Assign uses variables, not tensors. We will not test this.
 */
val assignOp = TensorflowMappingProcess(
        inputFrameworkOpName = "Assign",
        opName = "assign",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "ref","y" to "value")))
)

val adjustHue = TensorflowMappingProcess(
        inputFrameworkOpName = "AdjustHue",
        opName = "adjust_hue",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images","delta" to "delta"))),
        attributeMappingRules = listOf(
                intConstant(inputName= "dimC",constantValue = -1 ,argumentIndex = 0)[0])
)

val adjustSaturation = TensorflowMappingProcess(
        inputFrameworkOpName = "AdjustSaturation",
        opName = "adjust_saturation",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images","factor" to "scale"))),
        attributeMappingRules = listOf(
                intConstant(inputName= "dimC",constantValue = -1 ,argumentIndex = 0)[0])
)

val adjustContrast = TensorflowMappingProcess(
        inputFrameworkOpName = "AdjustContrastv2",
        opName = "adjust_contrast_v2",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images","factor" to "contrast_factor"))),
        attributeMappingRules = listOf()
)

val stopGradient = TensorflowMappingProcess(
        inputFrameworkOpName = "StopGradient",
        opName = "stop_gradient",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",argumentIndex = 0,constantValue = false)[0])
)

val polygamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Polygamma",
        opName = "polygamma",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("n" to "a","input" to "x"))),
        attributeMappingRules = listOf()
)


val avgPool = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool",
        opName = "avgpool2d",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value"))),
        attributeMappingRules = listOf(
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "pH"
                                int64Value = 0
                                argIndex = 4
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "pW"
                                int64Value = 0
                                argIndex = 5
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "dW"
                                int64Value = 1
                                argIndex = 6
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "dH"
                                int64Value = 1
                                argIndex = 7
                        },
                        ArgDescriptor {
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                name = "extraParam0"
                                int64Value = 0
                                argIndex = 9
                        }
                ))
        )
)

val avgPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "AvgPool3D",
        opName = "avgpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "extraParam0",constantValue = 0 ,argumentIndex = 13)[0],
                intConstant(inputName = "pD",constantValue = 0 ,argumentIndex = 6)[0],
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 8)[0],
                intConstant(inputName = "dD",constantValue = 1 ,argumentIndex = 9)[0],
                intConstant(inputName = "dH",constantValue = 1 ,argumentIndex = 10)[0],
                intConstant(inputName = "dW",constantValue = 1 ,argumentIndex = 11)[0],
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 14),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                listAttributeValueLookupToIndex(outputAttributeValue = "kH",inputAttributeValue = "ksize",idx = 3,argumentIndex = 2),
                listAttributeValueLookupToIndex(outputAttributeValue = "kW",inputAttributeValue = "ksize",idx = 2,argumentIndex = 1),
                listAttributeValueLookupToIndex(outputAttributeValue = "kD",inputAttributeValue = "ksize",idx = 1,argumentIndex = 0),
                listAttributeValueLookupToIndex(outputAttributeValue = "sH",inputAttributeValue = "strides",idx = 3,argumentIndex = 5),
                listAttributeValueLookupToIndex(outputAttributeValue = "sW",inputAttributeValue = "strides",idx = 2,argumentIndex = 4),
                listAttributeValueLookupToIndex(outputAttributeValue = "sD",inputAttributeValue = "strides",idx = 1,argumentIndex = 3),
        )
)

val batchToSpace = TensorflowMappingProcess(
        opName = "batch_to_space",
        inputFrameworkOpName = "BatchToSpace",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("blockSize" to "block_size"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops")))
)

val batchToSpaceND = TensorflowMappingProcess(
        opName = "batch_to_space_nd",
        inputFrameworkOpName = "BatchToSpaceND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("blocks" to "block_shape")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","crop" to "crops","blockShape" to "block_shape")))
)

val betaInc = TensorflowMappingProcess(
        opName = "betainc",
        inputFrameworkOpName = "Betainc",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "a","b" to "b","input" to "x"))),
        attributeMappingRules = emptyList()
)

val biasAddResult = defineBiasAdd(tensorflowOpRegistry = tensorflowOpRegistry)

val binCount = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "bincount",
        inputFrameworkOpName = "Bincount",
        tensorMappingRules =  listOf(mappingNDArrayInputs(mutableMapOf("weights" to "weights","values" to "arr"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "minLength"
                                argIndex = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                int64Value = 0
                        }
                )),
                convertNDArrayInputToNumericalAttr(mutableMapOf("maxLength" to "size")),
                valueMapping(mutableMapOf("outputType" to "T"))),
        inputIndexOverrides = mapOf(1 to 2,2 to 1))


val bitCast = TensorflowMappingProcess(
        opName = "bitcast",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "Bitcast",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(dataTypeToInt(mutableMapOf("newType" to "type")), valueMapping(mutableMapOf("dtype" to "type")))
)

val bitwiseAnd = TensorflowMappingProcess(
        opName = "bitwise_and",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseAnd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)

val bitwiseOr = TensorflowMappingProcess(
        opName = "bitwise_or",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseOr",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)



val bitwiseXOr = TensorflowMappingProcess(
        opName = "bitwise_xor",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BitwiseXor",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)

val broadcastDynamicShape = TensorflowMappingProcess(
        opName = "broadcast_dynamic_shape",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BroadcastArgs",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "s0","y" to "s1")))
)

//TODO: not implemented yet

/*val broadcastCatGradientArgs = TensorflowMappingProcess(
        opName = "broadcastgradientargs",
        inputFrameworkOpName = "BroadcastGradientArgs",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                intConstant(inputName = "dimension",constantValue = 0 ,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "s0","y" to "s1")))
)*/

val broadcastTo = TensorflowMappingProcess(
        opName = "broadcast_to",
        inputFrameworkOpName = "BroadcastTo",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","shape" to "shape")))
)


val copy2 = multipleNameMapping(
        inputFrameworkOpNames = listOf("Copy"),
        opName = "copy",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input")
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val checkNumerics = TensorflowMappingProcess(
        opName = "check_numerics",
        inputFrameworkOpName = "CheckNumerics",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(convertStringToInputNDArray(mutableMapOf("message" to "message"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "tensor")))
)

//only exists in tf2, tf-java can't run it

val checkNumericsV2 = TensorflowMappingProcess(
        opName = "check_numerics",
        inputFrameworkOpName = "CheckNumericsV2",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(convertStringToInputNDArray(mutableMapOf("message" to "message"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "tensor")))
)


val variable = mapTensorNamesWithOp(inputFrameworkOpName = "Variable",
        opName = "identity",
        tensorNames = mutableMapOf(),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dataType" to "dtype")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val variableV2 = mapTensorNamesWithOp(inputFrameworkOpName = "VariableV2",
        opName = "identity",
        tensorNames = mutableMapOf(),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dataType" to "dtype")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)



val identity2 = mapTensorNamesWithOp(inputFrameworkOpName = "Identity",
        opName = "identity",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dataType" to "T")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val const = mapTensorNamesWithOp(inputFrameworkOpName = "Const",
        opName = "identity",
        tensorNames = mutableMapOf(),
        attributeMappingRules = listOf(ndArrayAttributeToNDarrayInput(mutableMapOf("input" to "value")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]
                ,valueMapping(mutableMapOf("dataType" to "dtype")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val cholesky = TensorflowMappingProcess(
        opName = "cholesky",
        inputFrameworkOpName = "Cholesky",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = mapSameName(listOf("input"))
)


val clipByValue = TensorflowMappingProcess(
        opName = "ClipByValue",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "ClipByValue",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "t"))),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("clipValueMin" to "clip_value_min","clipValueMax" to "clip_value_max")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])
)


//TODO: our compare and bit pack operation seems to do something different than TFs?
/*
val compareAndBitPack = TensorflowMappingProcess(
        opName = "compare_and_bitpack",
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "CompareAndBitpack",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("threshold" to "threshold"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","y" to "threshold")))
)
*/


val concat = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "Concat",
        tensorMappingRules = listOf(mappingListNDArrays(mutableMapOf("input" to "values","concatDimension" to "concat_dim"))),
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("concatDimension" to "concat_dim")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = true,argumentIndex = 0)[0],valueMapping(mutableMapOf("dtype" to "T")))
)

val parallelConcat = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "ParallelConcat",
        tensorMappingRules = listOf(mappingListNDArrays(mutableMapOf("input" to "values"))),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "isDynamicAxis",constantValue = false,argumentIndex = 0)[0]
                ,valueMapping(mutableMapOf("dtype" to "T")),
                intConstant(inputName = "concatDimension",constantValue = 0,argumentIndex = 0)[0])
)

val tensorArrayV3 = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "tensorarrayv3",
        inputFrameworkOpName = "TensorArrayV3",
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf(dataTypeToInt(mutableMapOf("dataType" to "dtype")))
)


val mergeadd = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "mergeadd",
        inputFrameworkOpName = "AccumulateNV2",
        tensorMappingRules = listOf(mappingListNDArrays(mutableMapOf("inArrs" to "inputs"))),
        attributeMappingRules = listOf( booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]))

/**
 * Note that dynamic axis being true here is important.
 * The concat op in tensorflow may find constant nodes
 * that have an axis specified. When that's the case,
 * if dynamic axis is false, it will cause a serialization issue
 * where an input in to a concat op that may appear in the serialization
 * may not appear when reloaded. This breaks a ton of tests.
 * This is related to any getInputsForOp() call on any given variable.
 *
 */
val mergeAdd = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "ConcatV2",
        tensorMappingRules = listOf(mappingListNDArrays(mutableMapOf("input" to "values","concatDimension" to "axis"))),
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("concatDimension" to "axis")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = true,argumentIndex = 0)[0]))


/*val parallelConcat = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "concat",
        inputFrameworkOpName = "ParallelConcat",
        tensorMappingRules = listOf(mappingListNDArrays(mutableMapOf("input" to "values"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "concatDimension",constantValue = 0 ,argumentIndex = 0)[0],
                booleanConstant(inputName = "isDynamicAxis",constantValue = true,argumentIndex = 0)[0])
)*/

//TODO Reference ImportClassMapping.java
//TODO: ParallelDynamicStitch, map to dynamic stitch
//TODO: PollyGamma, map to pairwise transforms
//TODO: map QR

val cropAndResize = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "crop_and_resize",
        inputFrameworkOpName = "CropAndResize",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "image" to "image",
                "boxes" to "boxes",
                "boxIndexes" to "box_ind",
                "newImageSize" to "crop_size"))),
        attributeMappingRules = listOf(
                ndarrayStringToIndex(outputAttributeValue = "method",inputAttributeValue = "method",listOfValues = listOf("bilinear","nearest"),argumentIndex = 0),
                valueMapping(mapOf("extrapolationVal" to "extrapolation_value")))
)

val cumProd = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumprod",
        inputFrameworkOpName = "Cumprod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","dimensions" to "axis"))),
        attributeMappingRules = listOf(invertBooleanNumber(mutableMapOf("exclusive" to "exclusive","reverse" to "reverse"))))




val cumSum = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cumsum",
        inputFrameworkOpName = "Cumsum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","dimensions" to "axis"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mutableMapOf("exclusive" to "exclusive",
                        "reverse" to "reverse"))))




val cross = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        opName = "cross",
        inputFrameworkOpName = "Cross",
        tensorMappingRules = mapSameName(listOf("a","b"))
)

val depthToSpace = TensorflowMappingProcess(
        opName = "depth_to_space",
        inputFrameworkOpName = "DepthToSpace",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")),
                stringEqualsRule("isNHWC"
                        ,inputFrameworkAttributeName = "data_format",valueToTest = "NHWC",argumentIndex = 1)),
        opMappingRegistry = tensorflowOpRegistry
)

/**
 * depth_conv
 */
val depthWiseConv2d = TensorflowMappingProcess(
        opName = "depthwise_conv2d",
        inputFrameworkOpName = "DepthwiseConv2dNative",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 9),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 6),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 7),
                //NOTE: This is a dynamically resolved attribute at runtime.
                sizeAtRule(outputAttributeName = "kH",dimensionIndex = 0,inputFrameworkAttributeName = "filter",argumentIndex = 0),
                sizeAtRule(outputAttributeName = "kW",dimensionIndex = 1,inputFrameworkAttributeName = "filter",argumentIndex = 1),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "pH"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 4
                        },
                        ArgDescriptor {
                                name = "pW"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 5
                        },
                        ArgDescriptor {
                                name = "wFormat"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 10
                        }
                )))
)


val diag = TensorflowMappingProcess(
        inputFrameworkOpName = "Diag",
        opName = "diag",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "diagonal"))),
        opMappingRegistry = tensorflowOpRegistry
)


val diagPart = TensorflowMappingProcess(
        inputFrameworkOpName = "DiagPart",
        opName = "diag_part",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        opMappingRegistry = tensorflowOpRegistry
)

val lGamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Lgamma",
        opName = "lgamma",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)


val diGamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Digamma",
        opName = "digamma",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)

val iGamma = TensorflowMappingProcess(
        inputFrameworkOpName = "Igamma",
        opName = "igamma",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "a","y" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)



val iGammaC = TensorflowMappingProcess(
        inputFrameworkOpName = "Igammac",
        opName = "igammac",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "a","y" to "x"))),
        opMappingRegistry = tensorflowOpRegistry
)

val dilation2D = TensorflowMappingProcess(
        opName = "dilation2d",
        inputFrameworkOpName = "Dilation2D",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 0),
                listNumberToListNumber(outputAttributeValue = "rates",inputAttributeValue = "rates"),
                listNumberToListNumber(outputAttributeValue = "strides",
                        inputAttributeValue = "strides"))
)

val drawBoundingBoxes = TensorflowMappingProcess(
        inputFrameworkOpName = "DrawBoundingBoxesV2",
        inputFramework = "tensorflow",
        opName = "draw_bounding_boxes",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("images" to "images","boxes" to "boxes","colors" to "colors")))
)


/**
 * Note: -1 means dynamically resolved.
 */
val conv2d =  TensorflowMappingProcess(
        inputFrameworkOpName = "Conv2D",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 5)[0],
                intConstant(inputName = "wFormat",constantValue = 0 ,argumentIndex = 10)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 9),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 6),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 7),
                //NOTE: This is a dynamically resolved attribute at runtime.
                intConstant(inputName = "kH",constantValue = -1,argumentIndex = 0)[0],
                intConstant(inputName = "kW",constantValue = -1,argumentIndex = 1)[0]
        ),opMappingRegistry = tensorflowOpRegistry)

/**
 * Note: -1 means dynamically resolved.
 */
val deconv2d =  TensorflowMappingProcess(
        inputFrameworkOpName = "Conv2DBackpropInput",
        opName = "deconv2d_tf",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "gradIShape" to "input_sizes","weights" to "filter"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 5)[0],
                intConstant(inputName = "wFormat",constantValue = 0 ,argumentIndex = 10)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 9),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dH", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 6),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "dW", attributeNameOfListAttribute = "dilations", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 7),
                //NOTE: This is a dynamically resolved attribute at runtime.
                intConstant(inputName = "kH",constantValue = -1,argumentIndex = 0)[0],
                intConstant(inputName = "kW",constantValue = -1,argumentIndex = 1)[0]
        ),opMappingRegistry = tensorflowOpRegistry)


/**
 * Note: -1 means dynamically resolved.
 */
val conv3d =  TensorflowMappingProcess(
        inputFrameworkOpName = "Conv3D",
        opName = "conv3dnew",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "input","weights" to "filter"))),
        attributeMappingRules = listOf(
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 13),
                stringEqualsRule(outputAttribute = "paddingMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                sizeAtRule(dimensionIndex = 0,"kD",inputFrameworkAttributeName = "filter",argumentIndex = 0),
                sizeAtRule(dimensionIndex = 1,"kH",inputFrameworkAttributeName = "filter",argumentIndex = 1),
                sizeAtRule(dimensionIndex = 2,"kW",inputFrameworkAttributeName = "filter",argumentIndex = 2),
                listAttributeValueLookupToIndex(outputAttributeValue = "sD",inputAttributeValue = "strides",idx = 1,argumentIndex = 3),
                listAttributeValueLookupToIndex(outputAttributeValue = "sH",inputAttributeValue = "strides",idx = 2,argumentIndex = 4),
                listAttributeValueLookupToIndex(outputAttributeValue = "sW",inputAttributeValue = "strides",idx = 3,argumentIndex = 5),
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 8)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 6)[0],
                listAttributeValueLookupToIndex(outputAttributeValue = "dH",inputAttributeValue = "dilations",idx = 3,argumentIndex = 11),
                listAttributeValueLookupToIndex(outputAttributeValue = "dW",inputAttributeValue = "dilations",idx = 2,argumentIndex = 10),
                listAttributeValueLookupToIndex(outputAttributeValue = "dD",inputAttributeValue = "dilations",idx = 1,argumentIndex = 9),
        ),opMappingRegistry = tensorflowOpRegistry)




val divideNoNan = TensorflowMappingProcess(
        opName = "divide_no_nan",
        inputFrameworkOpName = "DivNoNan",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        opMappingRegistry = tensorflowOpRegistry
)

val dynamicPartition = TensorflowMappingProcess(
        opName = "dynamic_partition",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","indices" to "partitions"))),
        inputFrameworkOpName = "DynamicPartition",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mapOf("numPartitions" to "num_partitions")))
)



val dynamicStitch = TensorflowMappingProcess(
        opName = "dynamic_stitch",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("index" to "data","input" to "indices"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("numPartitions" to "N"))),
        inputFrameworkOpName = "DynamicStitch",
        opMappingRegistry = tensorflowOpRegistry
)
//ParallelDynamicStitch
val parallelDynamicStitch = TensorflowMappingProcess(
        opName = "dynamic_stitch",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("index" to "data","input" to "indices"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("numPartitions" to "N"))),
        inputFrameworkOpName = "ParallelDynamicStitch",
        opMappingRegistry = tensorflowOpRegistry
)

val empty = TensorflowMappingProcess(
        opName = "create",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "shape"))),
        inputFrameworkOpName = "Empty",
        attributeMappingRules = listOf(valueMapping(mapOf("init" to "init","outputType" to "dtype")),
                dataTypeToInt(mutableMapOf("outputType" to "dtype")),
                intConstant(inputName = "order",constantValue = 'c'.toInt() ,argumentIndex = 0)[0]),
        opMappingRegistry = tensorflowOpRegistry
)

val cast = TensorflowMappingProcess(
        opName = "cast",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x"))),
        inputFrameworkOpName = "Cast",
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("dtype" to "DstT")),
                dataTypeToInt(mutableMapOf("dst" to "DstT"))),
        opMappingRegistry = tensorflowOpRegistry
)


val elu = mapTensorNamesWithOp(inputFrameworkOpName = "Elu",opName = "elu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = listOf(argDescriptorConstant(
                listOf(
                        ArgDescriptor {
                                name = "alpha"
                                doubleValue = 1.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 0
                        }
                )
        ))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val enter = TensorflowMappingProcess(
        opName = "enter",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        inputFrameworkOpName = "Enter",
        attributeMappingRules = listOf(valueMapping(mapOf("isConstant" to "is_constant","frameName" to "frame_name"))),
        opMappingRegistry = tensorflowOpRegistry
)

val equal = TensorflowMappingProcess(
        opName = "equals",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        inputFrameworkOpName = "Equal",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = tensorflowOpRegistry
)

val approxEqual = TensorflowMappingProcess(
        opName = "equals",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","y" to "y"))),
        inputFrameworkOpName = "ApproximateEqual",
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        opMappingRegistry = tensorflowOpRegistry
)

val exit = TensorflowMappingProcess(
        opName = "exit",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        inputFrameworkOpName = "Exit",
        opMappingRegistry = tensorflowOpRegistry
)

val expandDims = TensorflowMappingProcess(
        opName = "expand_dims",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        inputFrameworkOpName = "ExpandDims",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(ndarrayNameToAttributeName = mutableMapOf("dimensions" to "dim"))
        ))

val extractImagesPatches = TensorflowMappingProcess(
        opName = "extract_image_patches",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "images"))),
        inputFrameworkOpName = "ExtractImagePatches",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                listAttributeValueLookupToIndex(outputAttributeValue = "ksizeRows",inputAttributeValue = "ksizes",idx =  1,argumentIndex = 0),
                listAttributeValueLookupToIndex(outputAttributeValue = "ksizeCols",inputAttributeValue = "ksizes",idx =  2,argumentIndex = 1),
                listAttributeValueLookupToIndex(outputAttributeValue = "kstrideRows",inputAttributeValue = "strides",idx =  1,argumentIndex = 2),
                listAttributeValueLookupToIndex(outputAttributeValue = "kstrideCols",inputAttributeValue = "strides",idx =  2,argumentIndex = 3),
                listAttributeValueLookupToIndex(outputAttributeValue = "krateRows",inputAttributeValue = "rates",idx =  1,argumentIndex = 4),
                listAttributeValueLookupToIndex(outputAttributeValue = "krateCols",inputAttributeValue = "rates",idx =  2,argumentIndex = 5),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 0),
                valueMapping(mutableMapOf("dtype" to "T")))
)



val fusedBatchnormV1 = TensorflowMappingProcess(
        opName = "fused_batch_norm",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","scale" to "scale",
                "offset" to "offset","mean" to "mean","variance" to "variance"))),
        inputFrameworkOpName = "FusedBatchNorm",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mutableMapOf("epsilon" to "epsilon","dtype" to "T")),
                invertBooleanNumber(mutableMapOf("isTraining" to "is_training")),
                stringEqualsRule(outputAttribute = "dataFormat",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 0))
)



val fusedBatchnormV2 = TensorflowMappingProcess(
        opName = "fused_batch_norm",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","scale" to "scale",
                "offset" to "offset","mean" to "mean","variance" to "variance"))),
        inputFrameworkOpName = "FusedBatchNormV2",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mutableMapOf("epsilon" to "epsilon","dtype" to "T")),
                invertBooleanNumber(mutableMapOf("isTraining" to "is_training")),
                stringEqualsRule(outputAttribute = "dataFormat",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 0))
)

//tf2 op
val fusedBatchnormV3 = TensorflowMappingProcess(
        opName = "fused_batch_norm",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x","scale" to "scale",
                "offset" to "offset","mean" to "mean","variance" to "variance"))),
        inputFrameworkOpName = "FusedBatchNormV3",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(valueMapping(mutableMapOf("epsilon" to "epsilon","dtype" to "T")),
                invertBooleanNumber(mutableMapOf("isTraining" to "is_training")),
                stringEqualsRule(outputAttribute = "dataFormat",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 0))
)



val gather = TensorflowMappingProcess(
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf()),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                intConstant(inputName = "dimensions",constantValue = 0 ,argumentIndex = 0)[0]),
        inputFrameworkOpName = "Gather",
        opMappingRegistry = tensorflowOpRegistry
)

val gatherV2 = TensorflowMappingProcess(
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                ndarrayToIntList(mutableMapOf("dimensions" to "axis"))),
        inputFrameworkOpName = "GatherV2",
        opMappingRegistry = tensorflowOpRegistry
)

val gatherNd = TensorflowMappingProcess(
        opName = "gather_nd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "params","indices" to "indices"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf()),
                booleanConstant(inputName = "checkIndices",constantValue = false,argumentIndex = 0)[0]),
        inputFrameworkOpName = "GatherNd",
        opMappingRegistry = tensorflowOpRegistry
)

val histogramFixedWidth = TensorflowMappingProcess(
        opName = "histogram_fixed_width",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "values","range" to "value_range","numBins" to "nbins"))),
        inputFrameworkOpName = "HistogramFixedWidth",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("nbins" to "nbins")))
)

val identity = multipleNameMapping(
        opName = "identity",
        inputFrameworkOpNames = listOf("DeepCopy"),
        tensorNames =  mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val identityCopyToHost = multipleNameMapping(
        opName = "identity",
        inputFrameworkOpNames = listOf("CopyHost"),
        tensorNames =  mutableMapOf("input" to "input"),
        attributeMappingRules =  booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorflowOpRegistry = tensorflowOpRegistry)

val identityN = TensorflowMappingProcess(
        opName = "identity_n",
        inputFrameworkOpName = "IdentityN",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules =  listOf(mappingNDArrayInputs(mutableMapOf("input" to "input")))
)

val ifOp = TensorflowMappingProcess(
        opName = "switch",
        inputFrameworkOpName = "If",
        attributeMappingRules = listOf(),
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","predicate" to "cond")))
)

val switchOp = TensorflowMappingProcess(
        opName = "switch",
        inputFrameworkOpName = "Switch",
        attributeMappingRules = listOf(),
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","predicate" to "pred")))
)

val fill = TensorflowMappingProcess(
        opName = "fill",
        inputFrameworkOpName = "Fill",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("value" to "value")),
                dataTypeToInt(mutableMapOf("outputDataType" to "T")),
                valueMapping(mutableMapOf("dtype" to "T"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("shapeArray" to "dims")))
)


val reciprocal = TensorflowMappingProcess(
        opName = "Reciprocal",
        inputFrameworkOpName = "Inv",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x")))
)

val reciprocal2 = TensorflowMappingProcess(
        opName = "Reciprocal",
        inputFrameworkOpName = "Reciprocal",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "x")))
)


val inTopKResults = multipleNameMapping(inputFrameworkOpNames = listOf("InTopK"),
        opName = "in_top_k",
        tensorNames = mutableMapOf("target" to "targets","predictions" to "predictions"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("k" to "k")),
                booleanConstant(inputName = "sorted",constantValue = true,argumentIndex = 0)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val inTopKResults2 = multipleNameMapping(inputFrameworkOpNames = listOf("InTopKV2"),
        opName = "in_top_k",
        tensorNames = mutableMapOf("target" to "targets","predictions" to "predictions"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("k" to "k")),
                booleanConstant(inputName = "sorted",constantValue = true,argumentIndex = 0)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val invert = mapTensorNamesWithOp(inputFrameworkOpName = "Invert",opName = "toggle_bits"
        ,tensorNames = mutableMapOf("input" to "x")
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val invertPermutation = mapTensorNamesWithOp(inputFrameworkOpName = "InvertPermutation",
        opName = "invert_permutation",tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dataType" to "T")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val isFinite = mapTensorNamesWithOp(inputFrameworkOpName = "IsFinite",opName = "isfinite"
        ,tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val isInf = mapTensorNamesWithOp(inputFrameworkOpName = "IsInf",opName = "isinf",
        tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val isNan = mapTensorNamesWithOp(inputFrameworkOpName = "IsNan",opName = "isnan",
        tensorNames = mutableMapOf("input" to "x"),attributeMappingRules = booleanConstant(inputName = "inPlace"
                ,constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: weird parameter values with config.getBias( and other similar names
val lrn = mapTensorNamesWithOp(inputFrameworkOpName = "LRN",opName = "lrn",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("depth" to "depth_radius","alpha" to "alpha",
                "bias" to "bias","beta" to "beta")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val leakyRelu = mapTensorNamesWithOp(inputFrameworkOpName = "LeakyRelu",opName = "leakyrelu",
        attributeMappingRules = listOf(valueMapping(mappings = mutableMapOf("alpha" to "alpha"))),
        tensorNames = mutableMapOf("input" to "features"),tensorflowOpRegistry = tensorflowOpRegistry)
//TODO: no input values found
val leftShift = mapTensorNamesWithOp(inputFrameworkOpName = "LeftShift",opName = "shift_bits",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val linspace = mapTensorNamesWithOp(inputFrameworkOpName = "LinSpace",opName = "lin_space",
        tensorNames = mutableMapOf("start" to "start","finish" to "stop","numOfElements" to "num"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "start" to "start",
                        "stop" to "stop")),
                valueMapping(mutableMapOf("dataType" to "T"))
        ),tensorflowOpRegistry = tensorflowOpRegistry)

//0=tanh, 1=relu, 2=sigmoid, 3=affine, 4=leaky relu, 5= thresholded relu, 6=scaled tanh, 7=hard sigmoid, 8=ELU, 9=softsign, 10=softplus

val lstmActivationMap = mapOf(
        "Relu" to 1,
        "Tanh" to 0,
        "Sigmoid" to 2,
        "Affine" to 3,
        "LeakyRelu" to 4,
        "ThresholdedRelu" to 5,
        "ScaledTanh" to 6,
        "HardSigmoid" to 7,
        "Elu" to 8,
        "Softsign" to 9,
        "Softplus" to 10
)

val lstmBlock = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BlockLSTM",
        opName = "lstmBlock",
        tensorMappingRules = listOf(
                mappingNDArrayInputs(mutableMapOf(
                        "maxTSLength" to "seq_len_max",
                        "input" to "x",
                        "cLast" to "cs_prev",
                        "yLast" to "h_prev",
                        "W" to "w",
                        "Wci" to "wci",
                        "Wcf" to "wcf",
                        "Wco" to "wco",
                        "b" to "b"))
        ),
        attributeMappingRules =  listOf(
                valueMapping(mutableMapOf("forgetBias" to "forget_bias","clippingCellValue" to "cell_clip")),
                invertBooleanNumber(mutableMapOf("peephole" to "use_peephole")),
                intConstant(inputName = "dataFormat",constantValue = 0 ,argumentIndex = 0)[0])
)

val lstmBlockV2 = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "BlockLSTMV2",
        opName = "lstmBlock",
        tensorMappingRules = listOf(
                mappingNDArrayInputs(mutableMapOf(
                        "maxTSLength" to "seq_len_max",
                        "input" to "x",
                        "cLast" to "cs_prev",
                        "yLast" to "h_prev",
                        "W" to "w",
                        "Wci" to "wci",
                        "Wcf" to "wcf",
                        "Wco" to "wco",
                        "b" to "b"))
        ),
        attributeMappingRules =  listOf(
                valueMapping(mutableMapOf("clippingCellValue" to "cell_clip")),
                invertBooleanNumber(mutableMapOf("peephole" to "use_peephole")),
                doubleConstant(inputName = "forgetBias",constantValue = 3.0,argumentIndex = 0)[0],
                intConstant(inputName = "dataFormat",constantValue = 0 ,argumentIndex = 0)[0])
)

val lstmBlockCell = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "LSTMBlockCell",
        opName = "lstmBlockCell",
        tensorMappingRules = listOf(
                mappingNDArrayInputs(mutableMapOf(
                        "xt" to "x",
                        "cLast" to "cs_prev",
                        "yLast" to "h_prev",
                        "W" to "w",
                        "Wci" to "wci",
                        "Wcf" to "wcf",
                        "Wco" to "wco",
                        "b" to "b"))
        ),
        attributeMappingRules =  listOf(
                valueMapping(mutableMapOf("forgetBias" to "forget_bias","clippingCellValue" to "cell_clip")),
                invertBooleanNumber(mutableMapOf("peephole" to "use_peephole")))
)

val gruCell = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "GRUBlockCell",
        opName = "gruCell",
        tensorMappingRules = listOf(
                mappingNDArrayInputs(mutableMapOf(
                        "input" to "x",
                        "hLast" to "h_prev",
                        "Wru" to "w_ru",
                        "Wc" to "w_c",
                        "bru" to "b_ru",
                        "bc" to "b_c"))
        )
)

val listDiff = mapTensorNamesWithOp(inputFrameworkOpName = "ListDiff",
        opName = "listdiff",
        tensorNames = mutableMapOf("values" to "x","keep" to "y")
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val logMatrixDeterminant = mapTensorNamesWithOp(
        inputFrameworkOpName = "LogMatrixDeterminant",
        opName = "log_matrix_determinant",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val logicalAnd = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalAnd",opName = "boolean_and",tensorNames = mutableMapOf("input" to "x","y" to "y")
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val logicalNot = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalNot",opName = "boolean_not",tensorNames = mutableMapOf("input" to "x")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val lu = mapTensorNamesWithOp(inputFrameworkOpName = "Lu",opName = "lu",tensorNames = mutableMapOf("input" to "input")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val gemm = multipleNameMapping(inputFrameworkOpNames = listOf("MatMul"),opName = "matmul",
        tensorNames = mutableMapOf("input" to "a","y" to "b"),
        attributeMappingRules =
        listOf(doubleConstant(inputName = "alpha",constantValue = 1.0,argumentIndex = 0)[0],
                doubleConstant(inputName = "beta",constantValue = 0.0,argumentIndex = 1)[0],
                invertBooleanNumber(mutableMapOf("transX" to "transpose_a","transY" to "transpose_b")),
                intConstant(inputName = "transZ",constantValue = 0 ,argumentIndex = 2)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val batchMatMul = multipleNameMapping(inputFrameworkOpNames = listOf("BatchMatMul"),opName = "matmul",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),
        attributeMappingRules =
        listOf(doubleConstant(inputName = "alpha",constantValue = 1.0,argumentIndex = 0)[0],
                doubleConstant(inputName = "beta",constantValue = 1.0,argumentIndex = 1)[0],
                invertBooleanNumber(mutableMapOf("transX" to "adj_x","transY" to "adj_y")),
                intConstant(inputName = "transZ",constantValue = 0 ,argumentIndex = 2)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val batchMatMulV2 = multipleNameMapping(inputFrameworkOpNames = listOf("BatchMatMulV2"),opName = "matmul",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),
        attributeMappingRules =
        listOf(doubleConstant(inputName = "alpha",constantValue = 1.0,argumentIndex = 0)[0],
                doubleConstant(inputName = "beta",constantValue = 1.0,argumentIndex = 1)[0],
                invertBooleanNumber(mutableMapOf("transX" to "adj_x","transY" to "adj_y")),
                intConstant(inputName = "transZ",constantValue = 0 ,argumentIndex = 2)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val matrixSetDiag = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixSetDiag","BatchMatrixSetDiag"),
        opName = "matrix_set_diag",
        tensorNames = mutableMapOf("input" to "input","diagonal" to "diagonal"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val matrixSetDiagPart = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixDiagPart"),
        opName = "matrix_diag_part",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input"),tensorflowOpRegistry = tensorflowOpRegistry)


val matrixDiag = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixDiag"),
        opName = "matrix_diag",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorNames = mutableMapOf("diagonal" to "diagonal"),tensorflowOpRegistry = tensorflowOpRegistry)


val matrixSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixSolve",opName = "solve"
        ,tensorNames = mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("useAdjoint" to "adjoint")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val matrixTriangularSolve = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixTriangularSolve"
        ,opName = "triangular_solve",tensorNames =
        mutableMapOf("a" to "matrix","b" to "rhs"),
        attributeMappingRules = listOf(valueMapping(mapOf("useAdjoint" to "adjoint","isLower" to "lower"))),
        tensorflowOpRegistry = tensorflowOpRegistry)


val matrixDeterminant = multipleNameMapping(inputFrameworkOpNames = listOf("BatchMatrixDeterminant","MatrixDeterminant")
        ,opName = "matrix_determinant",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val minPairWise = mapTensorNamesWithOp(inputFrameworkOpName = "Minimum",
        opName = "minimum",
        tensorNames = mutableMapOf("input" to "x","y" to "y")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val maxPool = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPool"),
        opName = "maxpool2d",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 5)[0],
                intConstant(inputName = "dW",constantValue = 1 ,argumentIndex = 6)[0],
                intConstant(inputName = "dH",constantValue = 1 ,argumentIndex = 7)[0],
                intConstant(inputName = "extraParam0",constantValue = 1 ,argumentIndex = 9)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1)
        )
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val maxPoolArgmax = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPoolWithArgmax"),
        opName = "max_pool_with_argmax",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                intConstant(inputName = "kH",constantValue = 1 ,argumentIndex = 0)[0],
                intConstant(inputName = "kW",constantValue = 1 ,argumentIndex = 1)[0],
                intConstant(inputName = "sH",constantValue = 1 ,argumentIndex = 2)[0],
                intConstant(inputName = "sW",constantValue = 1 ,argumentIndex = 3)[0],
                intConstant(inputName = "pH",constantValue = 1 ,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 1 ,argumentIndex = 5)[0],
                intConstant(inputName = "dH",constantValue = 1 ,argumentIndex = 6)[0],
                intConstant(inputName = "dW",constantValue = 1 ,argumentIndex = 7)[0],
                intConstant(inputName = "extraParam0",constantValue = 0 ,argumentIndex = 9)[0],
                intConstant(inputName = "isNHWC",argumentIndex = 10,constantValue = 1 )[0],
                intConstant(inputName = "sameMode",argumentIndex = 8,constantValue = 8 )[0],
                valueMapping(mutableMapOf("dtype" to "T"))
        )
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val maxPoolV2 = multipleNameMapping(
        inputFrameworkOpNames = listOf("MaxPoolV2"),
        opName = "maxpool2d",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                intConstant(inputName = "extraParam0",constantValue = 0 ,argumentIndex = 9)[0],
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 4)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 5)[0],
                intConstant(inputName = "dW",constantValue = 1 ,argumentIndex = 6)[0],
                intConstant(inputName = "dH",constantValue = 1 ,argumentIndex = 7)[0],
                stringNotEqualsRule(outputAttribute = "isNCHW",inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 10),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 8),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "sH", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 2),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "sW", attributeNameOfListAttribute = "strides", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 3),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "kH", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 2,
                        falseIndex = 1,inputFrameworkStringNameToTest = "data_format",argumentIndex = 0),
                conditionalFieldValueIntIndexNDArrayRule(outputAttribute = "kW", attributeNameOfListAttribute = "ksize", targetValue = "NCHW", trueIndex = 3,
                        falseIndex = 2,inputFrameworkStringNameToTest = "data_format",argumentIndex = 1)
        )
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val maxPool3d = TensorflowMappingProcess(
        inputFrameworkOpName = "MaxPool3D",
        opName = "maxpool3dnew",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "extraParam0",constantValue = 0 ,argumentIndex = 13)[0],
                intConstant(inputName = "pD",constantValue = 0 ,argumentIndex = 6)[0],
                intConstant(inputName = "pH",constantValue = 0 ,argumentIndex = 7)[0],
                intConstant(inputName = "pW",constantValue = 0 ,argumentIndex = 8)[0],
                intConstant(inputName = "dD",constantValue = 1 ,argumentIndex = 9)[0],
                intConstant(inputName = "dH",constantValue = 1 ,argumentIndex = 10)[0],
                intConstant(inputName = "dW",constantValue = 1 ,argumentIndex = 11)[0],
                stringEqualsRule(outputAttribute = "isNCDHW",inputFrameworkAttributeName = "data_format",valueToTest = "NDHWC",argumentIndex = 14),
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "padding",valueToTest = "SAME",argumentIndex = 12),
                listAttributeValueLookupToIndex(outputAttributeValue = "kH",inputAttributeValue = "ksize",idx = 3,argumentIndex = 2),
                listAttributeValueLookupToIndex(outputAttributeValue = "kW",inputAttributeValue = "ksize",idx = 2,argumentIndex = 1),
                listAttributeValueLookupToIndex(outputAttributeValue = "kD",inputAttributeValue = "ksize",idx = 1,argumentIndex = 0),
                listAttributeValueLookupToIndex(outputAttributeValue = "sH",inputAttributeValue = "strides",idx = 3,argumentIndex = 5),
                listAttributeValueLookupToIndex(outputAttributeValue = "sW",inputAttributeValue = "strides",idx = 2,argumentIndex = 4),
                listAttributeValueLookupToIndex(outputAttributeValue = "sD",inputAttributeValue = "strides",idx = 1,argumentIndex = 3),
        )
)


//TODO: Not likely correct. Need to figure out true mapping. Likely an implicit control flow op?
val loopCond = mapTensorNamesWithOp(inputFrameworkOpName = "LoopCond",opName = "loop_cond",tensorNames = mutableMapOf()
        ,tensorflowOpRegistry = tensorflowOpRegistry)
val merge = mapTensorNamesWithOp(inputFrameworkOpName = "Merge",opName = "merge",
        attributeMappingRules = listOf(),

        tensorNames = mutableMapOf("input" to "inputs")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val mirrorPadding = mapTensorNamesWithOp(inputFrameworkOpName = "MirrorPad",opName = "mirror_pad",
        tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),
        attributeMappingRules = listOf(stringNotEqualsRule(outputAttribute = "mode",
                inputFrameworkAttributeName = "mode",valueToTest = "REFLECT",argumentIndex = 0),
                booleanConstant(inputName = "isSymmetric",constantValue = true,argumentIndex = 0)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

/**
 * TODO: Need to add a constant mapping or something for NonMaxSuppression
 * v1 and 2 which do not have a scoreThreshold to map. V3 does.
 */

val matrixBandPart = mapTensorNamesWithOp(inputFrameworkOpName = "MatrixBandPart",opName = "matrix_band_part",
        tensorNames = mutableMapOf("input" to "input","minLowerT" to "num_lower",
        "maxUpperT" to "num_upper"),
        attributeMappingRules = listOf()
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val nonMaxSuppressionV1 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppression"),
        opName = "non_max_suppression",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "maxOutputSize" to "max_output_size"),
        attributeMappingRules = listOf(
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                doubleValue = 0.5
                                name = "scoreThreshold"
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        }
                )),
                valueMapping(mutableMapOf("iouThreshold" to "iou_threshold")),
                convertNDArrayInputToNumericalAttr(mutableMapOf("maxOutputSize" to "max_output_size")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)



val nonMaxSuppressionV2 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV2"),
        opName = "non_max_suppression",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "iouThreshold" to "iou_threshold","maxOutputSize" to "max_output_size"),
        attributeMappingRules = listOf(
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                doubleValue = 0.5
                                name = "scoreThreshold"
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        }
                )),
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "maxOutputSize" to "max_output_size"
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val nonMaxSuppressionV3 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV3"),
        opName = "non_max_suppression_v3",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "maxOutSize" to "max_output_size", "iouThreshold" to "iou_threshold", "scoreThreshold" to "score_threshold"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "maxOutputSize" to "max_output_size"
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val nonMaxSuppressionV4 = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionV4"),
        opName = "non_max_suppression_v3",
        tensorNames = mutableMapOf("boxes" to "boxes","scales" to "scores",
                "maxOutSize" to "max_output_size", "iouThreshold" to "iou_threshold", "scoreThreshold" to "score_threshold"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "maxOutputSize" to "max_output_size"
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val matrixInverse = multipleNameMapping(inputFrameworkOpNames = listOf("MatrixInverse","BatchMatrixInverse"),opName = "matrix_inverse",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = true,argumentIndex = 0),
        tensorNames = mutableMapOf("input" to "input")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: There might be a subtle difference in the way max threshold is interpreted.
//Tensorflow gives exact number back, whereas we may give back less.
//See the non_max_suppression_overlaps test case in TestTensorflowIR
val nonMaxSuppressionOverlaps = multipleNameMapping(inputFrameworkOpNames = listOf("NonMaxSuppressionWithOverlaps"),
        opName = "non_max_suppression_overlaps",
        tensorNames = mutableMapOf("scales" to "scores","boxes" to "overlaps"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf(
                        "maxOutputSize" to "max_output_size",
                        "overlapThreshold" to "overlap_threshold",
                        "scoreThreshold" to "score_threshold")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val nthElement = mapTensorNamesWithOp(inputFrameworkOpName = "NthElement",opName = "nth_element",
        tensorNames = mutableMapOf("n" to "n","input" to "input"),
        attributeMappingRules = listOf(invertBooleanNumber(mapOf("reverse" to "reverse"))),tensorflowOpRegistry = tensorflowOpRegistry)

val oneHot = mapTensorNamesWithOp(inputFrameworkOpName = "OneHot",opName = "onehot",
        tensorNames = mutableMapOf("input" to "indices"),
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("on" to "on_value","off" to "off_value"
                        ,"depth" to "depth")),
                valueMapping(mutableMapOf("dimensions" to "axis","dataType" to "T")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val or = mapTensorNamesWithOp(inputFrameworkOpName = "LogicalOr",opName = "boolean_or",
        tensorNames = mutableMapOf("input" to "x","y" to "y"),tensorflowOpRegistry = tensorflowOpRegistry)

val onesLike = mapTensorNamesWithOp(inputFrameworkOpName = "OnesLike",
        opName = "ones_as",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dataType" to "T"))),
        tensorNames = mutableMapOf("input" to "x"),tensorflowOpRegistry = tensorflowOpRegistry)



val pow = mapTensorNamesWithOp(inputFrameworkOpName = "Pow",opName = "Pow",
        attributeMappingRules = listOf(),
        tensorNames = mutableMapOf("input" to "x","y" to "y"),tensorflowOpRegistry = tensorflowOpRegistry
)

val rank = mapTensorNamesWithOp(inputFrameworkOpName = "Rank", opName = "rank",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(argDescriptorConstant(listOf(ArgDescriptor {
                name = "inPlace"
                boolValue = false
                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                argIndex = 0

        }))),tensorflowOpRegistry = tensorflowOpRegistry)

val relu6 = multipleNameMapping(inputFrameworkOpNames = listOf("Relu6"),opName = "relu6",
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("dtype" to "T")),
                        argDescriptorConstant(
                        listOf(ArgDescriptor {
                                name = "inPlace"
                                boolValue = false
                                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                argIndex = 0
                        },
                                ArgDescriptor {
                                        name = "cutoff"
                                        doubleValue = 0.0
                                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                        argIndex = 0
                                })
                        )),
        tensorNames = mutableMapOf("input" to "features")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val stack = multipleNameMapping(inputFrameworkOpNames = listOf("Pack"),opName = "stack",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dimensions" to "axis","dtype" to "T"))),
        tensorNames = mutableMapOf("input" to "values")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

/**
 * // in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements
if (INT_ARG(0) == 0) { // CONSTANT mode
if(block.width() > 2) {
REQUIRE_TRUE(input->dataType() == INPUT_VARIABLE(2)->dataType(), 0, "PAD op: data types of input and padValue arrays should be the same but got %i and %i correspondingly !", input->dataType(), INPUT_VARIABLE(2)->dataType());
padValue.assign(INPUT_VARIABLE(2)->e(0));
}
else if (!block.getTArguments()->empty())
padValue = T_ARG(0);
}
else if(INT_ARG(0) == 1) {		// REFLECT mode
for(int dim=0; dim < rank; ++dim)
REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= (input->shapeOf()[dim]-1) && paddings->e<Nd4jLong>(dim,1) <= (input->shapeOf()[dim]-1), 0, "PAD op: wrong content of paddings array for REFLECT mode !");
}
if(INT_ARG(0) == 2) {		// SYMMETRIC mode
for(int dim=0; dim < rank; ++dim)
REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= input->shapeOf()[dim] && paddings->e<Nd4jLong>(dim,1)  <= input->shapeOf()[dim], 0, "PAD op: wrong content of paddings array for SYMMETRIC mode !");
}
 */
val pad = multipleNameMapping(inputFrameworkOpNames = listOf("Pad"),
        opName = "pad",tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),attributeMappingRules =
        listOf(argDescriptorConstant(listOf(
                ArgDescriptor {
                        //note: tensorflow only supports constant mode
                        name = "mode"
                        int64Value = 0
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        argIndex = 0
                },
                ArgDescriptor {
                        name = "padValue"
                        doubleValue = 0.0
                        argIndex = 0
                        argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        argIndex = 0
                }
        )))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val padV2 = multipleNameMapping(inputFrameworkOpNames = listOf("PadV2"),
        opName = "pad",tensorNames = mutableMapOf("input" to "input","paddings" to "paddings"),
        attributeMappingRules =
        listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("padValue" to "constant_values")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                //note: tensorflow only supports constant mode
                                name = "mode"
                                int64Value = 0
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                argIndex = 0
                        }
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val randomCrop = mapTensorNamesWithOp(inputFrameworkOpName = "RandomCrop",opName = "random_crop",tensorNames = mutableMapOf("input" to "image","shape" to "size"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val placeHolder = mapTensorNamesWithOp(inputFrameworkOpName = "Placeholder",opName = "placeholder",
        tensorNames = mutableMapOf(),
        attributeMappingRules = listOf()
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val randomGamma = mapTensorNamesWithOp(inputFrameworkOpName = "RandomGamma",opName = "random_gamma",tensorNames = mutableMapOf("shape" to "shape","alpha" to "alpha"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed"))),tensorflowOpRegistry = tensorflowOpRegistry)


val rgbToHsv = mapTensorNamesWithOp(inputFrameworkOpName = "RGBToHSV",opName = "rgb_to_hsv",tensorNames = mutableMapOf("input" to "images"),
        attributeMappingRules = listOf()
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val hsvToRgb = mapTensorNamesWithOp(inputFrameworkOpName = "HSVToRGB",opName = "hsv_to_rgb",tensorNames = mutableMapOf("input" to "images"),
        attributeMappingRules = listOf()
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val randomPoisson = multipleNameMapping(inputFrameworkOpNames = listOf("RandomPoisson"),opName = "random_poisson",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed","dtype" to "dtype"))),
        tensorNames = mutableMapOf("shape" to "shape","lambda" to "rate")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val randomPoissonv2 = multipleNameMapping(inputFrameworkOpNames = listOf("RandomPoissonV2"),opName = "random_poisson",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seed" to "seed","dtype" to "dtype"))),
        tensorNames = mutableMapOf("shape" to "shape","lambda" to "rate")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val randomShuffle = mapTensorNamesWithOp(inputFrameworkOpName = "RandomShuffle",opName = "random_shuffle",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("seeds" to "seed"))),tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: Look at extra arguments generated like T_ARG(1));
val randomStandardNormal = multipleNameMapping(inputFrameworkOpNames = listOf("RandomStandardNormal"),opName = "random_normal",
        tensorNames = mutableMapOf("input" to "shape"),
        attributeMappingRules =  listOf(valueMapping(mutableMapOf("dtype" to "dtype")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//note: tensorflow hard codes the value at 0 to 1 while we allow customization here

val randomUniform = multipleNameMapping(
        inputFrameworkOpNames = listOf("RandomUniform"),
        opName = "randomuniform",
        tensorNames = mutableMapOf("shape" to "shape"),
        attributeMappingRules =  listOf(
                dataTypeToInt(mutableMapOf("dtype" to "dtype")),
                valueMapping(mutableMapOf("dataType" to "dtype")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "min"
                                doubleValue = 0.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 0
                        },
                        ArgDescriptor {
                                name = "max"
                                doubleValue = 1.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        },
                        ArgDescriptor {
                                name = "min"
                                argIndex = 1
                                inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                                argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        },
                        ArgDescriptor {
                                name = "max"
                                argIndex = 2
                                inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                                argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        }
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry
)


val statelessRandomUniform = multipleNameMapping(
        inputFrameworkOpNames = listOf("StatelessRandomUniform"),
        opName = "randomuniform",
        tensorNames = mutableMapOf("shape" to "shape"),
        attributeMappingRules =  listOf(
                dataTypeToInt(mutableMapOf("dtype" to "dtype")),
                valueMapping(mutableMapOf("dataType" to "dtype")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "min"
                                doubleValue = 0.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 0
                        },
                        ArgDescriptor {
                                name = "max"
                                doubleValue = 1.0
                                argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                argIndex = 1
                        },
                        ArgDescriptor {
                                name = "min"
                                argIndex = 1
                                inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                                argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        },
                        ArgDescriptor {
                                name = "max"
                                argIndex = 2
                                inputValue = nameSpaceTensorFromNDarray(Nd4j.scalar(1.0))
                                argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        }
                )))
        ,tensorflowOpRegistry = tensorflowOpRegistry
)


val randomUniformInt = TensorflowMappingProcess(
        inputFrameworkOpName = "RandomUniformInt",
        opName = "randomuniform",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("shape" to "shape","min" to "minval","max" to "maxval"))),
        attributeMappingRules =  listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("min" to "minval","max" to "maxval")),
                dataTypeToInt(mutableMapOf("dtype" to "Tout")),valueMapping(mutableMapOf("dataType" to "Tout"))
        ),
        opMappingRegistry = tensorflowOpRegistry
)


val range = multipleNameMapping(inputFrameworkOpNames = listOf("Range"),opName = "range",
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("from" to "start",
                        "to" to "limit","step" to "delta")),
                valueMapping(mutableMapOf("dtype" to "Tidx"))),
        tensorNames = mutableMapOf("from" to "start","to" to "limit","step" to "delta"),tensorflowOpRegistry = tensorflowOpRegistry)

val relu = mapTensorNamesWithOp(inputFrameworkOpName = "Relu",opName = "relu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = listOf(doubleConstant(inputName = "cutoff",constantValue = 0.0,argumentIndex = 0)[0],
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val reshape = multipleNameMapping(inputFrameworkOpNames = listOf("Reshape"),opName = "reshape",
        tensorNames = mutableMapOf("input" to "tensor","shape" to "shape"),
        attributeMappingRules = listOf(),tensorflowOpRegistry = tensorflowOpRegistry)

val resizeArea = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeArea"),opName = "resize_area",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"),tensorflowOpRegistry = tensorflowOpRegistry)

val resizeBiCubic = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBicubic"),opName = "resize_bicubic",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners","alignPixelCenters" to "half_pixel_centers"))),
        tensorNames = mutableMapOf("image" to "images","size" to "size"),tensorflowOpRegistry = tensorflowOpRegistry)

val resizeBiLinear = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeBilinear"),opName = "resize_bilinear",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners","halfPixelCenter" to "half_pixel_centers"))),
        tensorNames = mutableMapOf("image" to "images","newImageSize" to "size"),tensorflowOpRegistry = tensorflowOpRegistry)

val resizeNearestNeighbor = multipleNameMapping(inputFrameworkOpNames = listOf("ResizeNearestNeighbor"),opName = "resize_nearest_neighbor",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("alignCorners" to "align_corners","halfPixelCenter" to "half_pixel_centers"))),
        tensorNames = mutableMapOf("image" to "images","newImageSize" to "size")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val reverse = multipleNameMapping(inputFrameworkOpNames = listOf("ReverseV2"),opName = "reverse",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("dimensions" to "axis"))),
        tensorNames = mutableMapOf("input" to "tensor"),tensorflowOpRegistry = tensorflowOpRegistry)

val reverseSequence = multipleNameMapping(inputFrameworkOpNames = listOf("ReverseSequence"),opName = "reverse_sequence",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("batchDim" to "batch_dim","seqDim" to "seq_dim"))),
        tensorNames = mutableMapOf("input" to "input","seqLengths" to "seq_lengths")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val roll = multipleNameMapping(inputFrameworkOpNames = listOf("Roll"),opName = "roll",
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("shift" to "shift","dimensions" to "axis"))),
        tensorNames = mutableMapOf("input" to "input")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: verify usingLocking property, it's not showing up in descriptors
val tesnorScatterAdd = multipleNameMapping(inputFrameworkOpNames = listOf("TensorScatterAdd"),opName = "scatter_add",
        tensorNames = mutableMapOf("input" to "tensor","indices" to "indices","updates" to "updates"),
        attributeMappingRules =
        listOf(booleanConstant(inputName = "lock",constantValue = false,0)[0],
                booleanConstant(inputName = "checkIndices",constantValue = false,argumentIndex = 1)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val scatterAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterAdd"),opName = "scatter_add",
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"),
        attributeMappingRules =
        listOf(booleanConstant(inputName = "lock",constantValue = false,0)[0],
                booleanConstant(inputName = "checkIndices",constantValue = false,argumentIndex = 1)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val scatterDiv = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterDiv"),opName = "scatter_div",
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterMax = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMax"),opName = "scatter_max",
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorScatterMax = multipleNameMapping(inputFrameworkOpNames = listOf("TensorScatterMax"),opName = "scatter_max",
        tensorNames = mutableMapOf("input" to "tensor","indices" to "indices","updates" to "updates"),tensorflowOpRegistry = tensorflowOpRegistry)


val tensorScatterMin = multipleNameMapping(inputFrameworkOpNames = listOf("TensorScatterMin"),opName = "scatter_min",
        tensorNames = mutableMapOf("input" to "tensor","indices" to "indices","updates" to "updates"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterMin = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMin"),opName = "scatter_min",
        tensorNames = mutableMapOf("input" to "ref","indices" to "indices","updates" to "updates"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterMul = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterMul"),opName = "scatter_mul",
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterNd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNd"),opName = "scatter_nd",
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","shape" to "shape"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "lock",constantValue = false,argumentIndex = 0)[0],
                booleanConstant(inputName = "checkIndices",constantValue = false,argumentIndex = 1)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val scatterNdAdd = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdAdd"),opName = "scatter_nd_add",
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterNdSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdSub"),opName = "scatter_nd_sub",
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"),tensorflowOpRegistry = tensorflowOpRegistry)

val scatterNdUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterNdUpdate"),opName = "scatter_nd_update",
        tensorNames = mutableMapOf("indices" to "indices","updates" to "updates","input" to "ref"),tensorflowOpRegistry = tensorflowOpRegistry)


val tensorScatterSub = multipleNameMapping(inputFrameworkOpNames = listOf("TensorScatterSub"),
        opName = "scatter_sub",
        tensorNames = mutableMapOf("indices" to "indices",
                "updates" to "updates","input" to "tensor"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "lock",constantValue = false,
                        argumentIndex = 0)[0],
                booleanConstant(inputName = "checkIndices",constantValue = false,
                        argumentIndex = 1)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val scatterSub = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterSub"),
        opName = "scatter_sub",
        tensorNames = mutableMapOf("indices" to "indices",
                "updates" to "updates","input" to "ref"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "lock",constantValue = false,
                        argumentIndex = 0)[0],
                booleanConstant(inputName = "checkIndices",constantValue = false,
                        argumentIndex = 1)[0])
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: note: TF expects indices, we don't support them?
val scatterUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("ScatterUpdate"),opName = "scatter_upd",
        attributeMappingRules = listOf(),
        tensorNames = mutableMapOf("input" to "ref","updates" to "updates","indices" to "indices"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorScatterUpdate = multipleNameMapping(inputFrameworkOpNames = listOf("TensorScatterUpdate"),opName = "scatter_upd",
        attributeMappingRules = listOf(),
        tensorNames = mutableMapOf("input" to "tensor","updates" to "updates","indices" to "indices"),tensorflowOpRegistry = tensorflowOpRegistry)
//L2Loss
val l2Loss = multipleNameMapping(inputFrameworkOpNames = listOf("L2Loss"),opName = "l2_loss",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("input" to "t")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val select = mapTensorNamesWithOp(inputFrameworkOpName = "Select",opName = "select",tensorNames = mutableMapOf("cond" to "condition","input" to "t","y" to "e")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val segmentMean = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMean"),opName = "segment_mean",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"),tensorflowOpRegistry = tensorflowOpRegistry)

val segmentMin = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMin"),opName = "segment_min",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"),tensorflowOpRegistry = tensorflowOpRegistry)


val segmentMax = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentMax"),opName = "segment_max",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val segmentProd = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentProd"),opName = "segment_prod",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"),tensorflowOpRegistry = tensorflowOpRegistry)

val segmentSum = multipleNameMapping(inputFrameworkOpNames = listOf("SegmentSum"),opName = "segment_sum",
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"),tensorflowOpRegistry = tensorflowOpRegistry)

val size = TensorflowMappingProcess(
        opMappingRegistry = tensorflowOpRegistry,
        inputFrameworkOpName = "Size",
        opName = "size",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "out_type"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input")))
)

val slice = mapTensorNamesWithOp(inputFrameworkOpName = "Slice",opName = "slice",
        tensorNames = mutableMapOf("input" to "input","b" to "begin","e" to "size"),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("size" to "size"))),tensorflowOpRegistry = tensorflowOpRegistry)

val selu = mapTensorNamesWithOp(inputFrameworkOpName = "Selu",opName = "selu",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules =
        booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),tensorflowOpRegistry = tensorflowOpRegistry)

val shapeOf = mapTensorNamesWithOp(inputFrameworkOpName = "Shape",
        opName = "shape_of",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dtype" to "out_type"))),tensorflowOpRegistry = tensorflowOpRegistry)

val softPlus = mapTensorNamesWithOp(inputFrameworkOpName = "Softplus",opName = "softplus",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),tensorflowOpRegistry = tensorflowOpRegistry)
val softSign = mapTensorNamesWithOp(inputFrameworkOpName = "Softsign",opName = "softsign",tensorNames = mutableMapOf("input" to "features"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),tensorflowOpRegistry = tensorflowOpRegistry)

val shapeN = mapTensorNamesWithOp(inputFrameworkOpName = "ShapeN",opName = "shapes_of",tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex =  0),tensorflowOpRegistry = tensorflowOpRegistry)

val softMax = mapTensorNamesWithOp(inputFrameworkOpName = "Softmax",opName = "softmax",tensorNames = mutableMapOf("input" to "logits"),attributeMappingRules =
listOf(argDescriptorConstant(
        listOf(
                ArgDescriptor {
                        name = "dimension"
                        int64Value = 1
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        argIndex = 0
                },
                ArgDescriptor {
                        name = "inPlace"
                        boolValue = false
                        argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                        argIndex = 0
                }
        )
)),tensorflowOpRegistry = tensorflowOpRegistry)
val logSoftmax = mapTensorNamesWithOp(inputFrameworkOpName = "LogSoftmax",
        opName = "log_softmax",tensorNames = mutableMapOf("input" to "logits")
,tensorflowOpRegistry = tensorflowOpRegistry)

//FakeQuantWithMinMaxVars
//FakeQuantWithMinMaxVarsPerChannel
val fakeQuantWithMinMaxVars = TensorflowMappingProcess(
        opName = "fake_quant_with_min_max_vars",
        inputFrameworkOpName = "FakeQuantWithMinMaxVars",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("numBits" to "num_bits","narrowed" to "narrow_range"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs","min" to "min","max" to "max")))
)

val fakeQuantWithMinMaxVarsPerChannel = TensorflowMappingProcess(
        opName = "fake_quant_with_min_max_vars_per_channel",
        inputFrameworkOpName = "FakeQuantWithMinMaxVarsPerChannel",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("numBits" to "num_bits","narrowed" to "narrow_range"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs","min" to "min","max" to "max")))
)

val fakeQuantWithMinArgs = TensorflowMappingProcess(
        opName = "fake_quant_with_min_max_args",
        inputFrameworkOpName = "FakeQuantWithMinMaxArgs",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("min" to "min","max" to "max","numBits" to "num_bits","narrowRange" to "narrow_range"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs")))
)

val sparseSoftmax = TensorflowMappingProcess(
        opName = "sparse_softmax_cross_entropy_loss_with_logits",
        inputFrameworkOpName = "SparseSoftmaxCrossEntropyWithLogits",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("dtype" to "T"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("labels" to "labels","logits" to "features"))),
        inputIndexOverrides = mapOf(1 to 0,0 to 1)
)

//SoftmaxCrossEntropyWithLogits
val softmaxCrossEntryopyWithLogits = TensorflowMappingProcess(
        opName = "softmax_cross_entropy_loss_with_logits",
        inputFrameworkOpName = "SoftmaxCrossEntropyWithLogits",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("dtype" to "T"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("labels" to "labels","logits" to "features")))

)


val spaceToBatch = TensorflowMappingProcess(
        opName = "space_to_batch",
        inputFrameworkOpName = "SpaceToBatch",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                valueMapping(mapOf("blockSize" to "block_size"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","padding" to "paddings")))
)

val spaceToBatchNd = TensorflowMappingProcess(
        opName = "space_to_batch_nd",
        inputFrameworkOpName = "SpaceToBatchND",
        opMappingRegistry = tensorflowOpRegistry,
        attributeMappingRules = listOf(
                ndarrayToIntList(mutableMapOf("blocks" to "block_shape")),
                argDescriptorConstant(listOf(
                        ArgDescriptor {
                                name = "inPlace"
                                boolValue = false
                                argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                                argIndex = 0

                        }
                ))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","blockShape" to "block_shape","padding" to "paddings")))
)

val spaceToDepth = TensorflowMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mapOf("block_size" to "block_size")),
                stringEqualsRule("isNHWC",inputFrameworkAttributeName = "data_format",valueToTest = "NHWC",argumentIndex = 1)),
        opMappingRegistry = tensorflowOpRegistry
)

val split = TensorflowMappingProcess(
        opName = "split",
        inputFrameworkOpName = "Split",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("a" to "split_dim","b" to "value"))),
        attributeMappingRules = listOf(valueMapping(mapOf("numSplit" to "num_split"))
                , ndarrayToIntList(mutableMapOf("dimensions" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)


val splitV = TensorflowMappingProcess(
        opName = "split_v",
        inputFrameworkOpName = "SplitV",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "value",
                "sizes" to "size_splits",
                "_a"  to "split_dim"))),
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("numSplit" to "num_split")),
                convertNDArrayInputToNumericalAttr(mutableMapOf("dimensions" to "split_dim")),
                ndarrayToIntList(mutableMapOf("dimensions" to "split_dim"))),
        opMappingRegistry = tensorflowOpRegistry
)

val squeeze = TensorflowMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(
                listNumberToNDarray(mutableMapOf("a" to "squeeze_dims")),
                listNumberToListNumber(outputAttributeValue = "_a",inputAttributeValue = "squeeze_dims")),
        opMappingRegistry = tensorflowOpRegistry
)

val stridedSlice = TensorflowMappingProcess(
        opName = "strided_slice",
        inputFrameworkOpName = "StridedSlice",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input",
                "v_begin" to "begin",
                "v_end" to "end",
                "v_stride" to "strides"))),
        attributeMappingRules = listOf(
                valueMapping(mutableMapOf("begin_mask" to "begin_mask","end_mask" to "end_mask",
                        "ellipsis_mask" to "ellipsis_mask","new_axis_mask" to "new_axis_mask",
                        "shrink_axis_mask" to "shrink_axis_mask","dtype" to "T")))
)


val svd = TensorflowMappingProcess(
        opName = "svd",
        inputFrameworkOpName = "Svd",
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("computeUv" to "compute_uv","fullUV" to "full_matrices")),
                invertBooleanNumber(mutableMapOf("calcUV" to "compute_uv","fullUV" to "full_matrices")),
                intConstant(inputName = "switchNum",constantValue = 16,argumentIndex = 2)[0])
)



//TODO: revisit this, not sure why the ops are off
val tensorArrayConcat = multipleNameMapping(inputFrameworkOpNames =
listOf("TensorArrayConcat"),
        opName = "stack_list",
        tensorNames = mutableMapOf("list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)


val tensorArrayConcatV2 = multipleNameMapping(inputFrameworkOpNames =
listOf("TensorArrayConcatV2"),
        opName = "stack_list",
        tensorNames = mutableMapOf("list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayConcatV3 = multipleNameMapping(inputFrameworkOpNames =
listOf("TensorArrayConcatV3"),
        opName = "stack_list",
        tensorNames = mutableMapOf("list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayWriteV3 = multipleNameMapping(inputFrameworkOpNames =
listOf("TensorArrayWriteV3"),
        opName = "tensorarraywritev3",
        tensorNames = mutableMapOf("input" to "handle"),tensorflowOpRegistry = tensorflowOpRegistry)


//TODO: revisit this, not sure why the ops are off
val tensorArrayGather = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayGather"),
        opName = "gather_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "dtype"))),
        tensorNames = mutableMapOf("indices" to "indices","list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayGatherv2 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayGatherV2"),
        opName = "gather_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "dtype"))),
        tensorNames = mutableMapOf("indices" to "indices","list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayGatherv3 = multipleNameMapping(inputFrameworkOpNames = listOf( "TensorArrayGatherV3"),
        opName = "gather_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "dtype"))),
        tensorNames = mutableMapOf("indices" to "indices","list" to "flow_in"),tensorflowOpRegistry = tensorflowOpRegistry)


//TODO: revisit this, not sure why the ops are off
/*val tensorArrayPack = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayPack", "TensorArrayPackV2", "TensorArrayPackV3"),
        opName = "tensorarraypackv3",
        tensorNames = mutableMapOf("indices" to "indices"))*/
//TODO: revisit this, not sure why the ops are off

val tensorArrayRead = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayRead"),
        opName = "read_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("importDataType" to "dtype"))),
        tensorNames = mutableMapOf("list" to "handle"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayReadV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayReadV2"),
        opName = "read_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("importDataType" to "dtype"))),
        tensorNames = mutableMapOf("list" to "handle"),tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayReadV3 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayReadV3"),
        opName = "read_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("importDataType" to "dtype"))),
        tensorNames = mutableMapOf("list" to "handle"),tensorflowOpRegistry = tensorflowOpRegistry)
//
//
// TODO: revisit this, not sure why the ops are off

val tensorArrayScatter = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayScatter"),
        opName = "scatter_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("array" to "value","sizes" to "indices","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayScatterV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayScatterV2"),
        opName = "scatter_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("array" to "value","sizes" to "indices","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArrayScatterV3 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArrayScatterV3"),
        opName = "scatter_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("array" to "value","sizes" to "indices","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: revisit this, not sure why the ops are off

val tensorArraySize = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySize"),
        opName = "size_list",
        tensorNames = mutableMapOf("list" to "handle","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val tensorArraySizeV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySizeV2"),
        opName = "size_list",
        tensorNames = mutableMapOf("list" to "handle","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val tensorArraySizeV3 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySizeV3"),
        opName = "size_list",
        tensorNames = mutableMapOf("list" to "handle","list" to "flow_in")
        ,tensorflowOpRegistry = tensorflowOpRegistry)
//TODO: revisit this, not sure why the ops are off

val tensorArraySplit = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySplit"),
        opName = "split_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("sizes" to "lengths","list" to "value","array" to "value")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val tensorArraySplitV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySplitV2"),
        opName = "split_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("sizes" to "lengths","list" to "value","array" to "value")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val tensorArraySplitV3 = multipleNameMapping(inputFrameworkOpNames = listOf("TensorArraySplitV3"),
        opName = "split_list",
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T"))),
        tensorNames = mutableMapOf("sizes" to "lengths","list" to "value","array" to "value")
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val tile = mapTensorNamesWithOp(inputFrameworkOpName = "Tile",opName = "tile",
        attributeMappingRules = listOf(intConstant(inputName = "dimensions",constantValue = 0 ,argumentIndex = 0)[0],
                booleanConstant(inputName = "is_static_reps",constantValue = true,argumentIndex = 0)[0]),
        tensorNames = mutableMapOf("input" to "input","reps_vector" to "multiples")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val topk = multipleNameMapping(inputFrameworkOpNames = listOf("TopK"),opName = "top_k",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("needSort" to "sorted","k" to "k"))),tensorflowOpRegistry = tensorflowOpRegistry)

val topkV2 = multipleNameMapping(inputFrameworkOpNames = listOf("TopKV2"),opName = "top_k",
        tensorNames = mutableMapOf("input" to "input"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("needSort" to "sorted")),
                convertNDArrayInputToNumericalAttr(mutableMapOf("k" to "k"))),tensorflowOpRegistry = tensorflowOpRegistry)

val transpose = mapTensorNamesWithOp(
        inputFrameworkOpName = "Transpose",
        opName = "transpose",
        tensorNames = mutableMapOf("input" to "x","permutationVector" to "perm"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dtype" to "T")), ndarrayToIntList(mutableMapOf("permuteDims" to "perm")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


//note we don't allow unique with an axis argument
val unique = multipleNameMapping(
        inputFrameworkOpNames = listOf("Unique","UniqueV2"),
        opName = "unique",
        tensorNames = mutableMapOf("input" to "x")
        ,tensorflowOpRegistry = tensorflowOpRegistry
)


/**
 * NOTE: Ours only supports vectors, not 2d.
 */
val uniqueWithCounts = multipleNameMapping(
        inputFrameworkOpNames = listOf("UniqueWithCounts","UniqueWithCountsV2"),
        opName = "unique_with_counts",
        tensorNames = mutableMapOf("input" to "x")
        ,tensorflowOpRegistry = tensorflowOpRegistry
)

val unpack = multipleNameMapping(inputFrameworkOpNames = listOf("Unpack"),
        opName = "unstack",
        tensorNames = mutableMapOf("input" to "value"),
        attributeMappingRules = listOf(valueMapping(mutableMapOf("dimensions" to "axis","num" to "num")))
        ,tensorflowOpRegistry = tensorflowOpRegistry)


val unsortedSegmentMax = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMax",
        opName = "unsorted_segment_max",
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments","numSegments" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val unsortedSegmentMin = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentMin",
        opName = "unsorted_segment_min",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

val unsortedSegmentProd = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentProd",
        opName = "unsorted_segment_prod",
        attributeMappingRules = listOf(
                convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids"),tensorflowOpRegistry = tensorflowOpRegistry)


val unsortedSegmentSum = mapTensorNamesWithOp(inputFrameworkOpName = "UnsortedSegmentSum",
        opName = "unsorted_segment_sum",
        attributeMappingRules = listOf(convertNDArrayInputToNumericalAttr(mutableMapOf("numSegments" to "num_segments"))),
        tensorNames = mutableMapOf("input" to "data","idxSegments" to "segment_ids")
        ,tensorflowOpRegistry = tensorflowOpRegistry)

//TODO: Figure out if need to map
val nextIteration = mapTensorNamesWithOp(inputFrameworkOpName = "NextIteration",opName = "next_iteration",
        tensorNames = mutableMapOf("input" to "data"), tensorflowOpRegistry = tensorflowOpRegistry)

val noOp = mapTensorNamesWithOp(inputFrameworkOpName = "NoOp",opName = "noop",tensorNames = mutableMapOf()
        , tensorflowOpRegistry = tensorflowOpRegistry)

val where = mapTensorNamesWithOp(inputFrameworkOpName = "Where",opName = "Where",
        tensorNames = mutableMapOf("condition" to "input")
        , tensorflowOpRegistry = tensorflowOpRegistry
)

val whileOp = mapTensorNamesWithOp(inputFrameworkOpName = "While",opName = "While",
        tensorNames = mutableMapOf("condition" to "input"),
        attributeMappingRules = listOf(booleanConstant(inputName = "isConstant",constantValue = false,argumentIndex = 0)[0])
        , tensorflowOpRegistry = tensorflowOpRegistry
)

val zerosLike = mapTensorNamesWithOp(inputFrameworkOpName = "ZerosLike",opName = "zeroslike",
        tensorNames = mutableMapOf("input" to "x"),
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                valueMapping(mutableMapOf("dataType" to "T"))
        ),tensorflowOpRegistry = tensorflowOpRegistry)

val zeta = mapTensorNamesWithOp(inputFrameworkOpName = "Zeta",opName = "zeta",
        tensorNames = mutableMapOf("input" to "x","q" to "q"),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorflowOpRegistry = tensorflowOpRegistry)


object TensorflowOpDeclarations {
        init {
                val tensorflowOps = OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow")
                val groupedOps = tensorflowOps.values.groupBy { input -> input.name }
                val singleGroupedOps = HashMap<String,OpDef>()
                groupedOps.forEach { name, node ->
                        singleGroupedOps[name] = node[0]
                }

                OpRegistryHolder.registerOpList("tensorflow", singleGroupedOps)
                tensorflowOps.values.forEach {
                        tensorflowOpRegistry.registerInputFrameworkOpDef(it.name,it)
                }

                OpDescriptorLoaderHolder.nd4jOpDescriptor.opListList.forEach {
                        tensorflowOpRegistry.registerNd4jOpDef(it.name,it)
                }

                reduceOps.forEach { tensorflowOpName, nd4jOpName ->
                        defineSingularReduce(inputFrameworkOpName = tensorflowOpName,inputOpName = nd4jOpName,
                                tensorflowOpRegistry = tensorflowOpRegistry)
                }


                singleTransformArgs.forEach {
                        defineTensorflowSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value
                                ,tensorflowOpRegistry = tensorflowOpRegistry)
                }

                elementWiseTransformOps.forEach {
                        defineTensorflowPairwiseTransforms(opName = it.value,inputFrameworkOpName = it.key,
                                tensorflowOpRegistry)
                }

                OpRegistryHolder.registerOpMappingRegistry("tensorflow", tensorflowOpRegistry)

        }
}



val declarations = TensorflowOpDeclarations

