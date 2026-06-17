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
package org.nd4j.samediff.frameworkimport.onnx.definitions

import onnx.Onnx
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcess
import org.nd4j.samediff.frameworkimport.onnx.rule.tensor.NDArrayMappingRule
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder
import org.nd4j.samediff.frameworkimport.onnx.definitions.MicrosoftOnnxExtensions

val onnxOpRegistry = OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>("onnx",OpDescriptorLoaderHolder.nd4jOpDescriptor)
fun registry(): OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto> {
        return onnxOpRegistry
}


val names = mapOf(
        "Acos" to "acos",
        "Acosh" to "acosh",
        "Asin" to "asin",
        "Asinh" to "asinh",
        "Atan" to "atan",
        "Atanh" to "atanh",
        "Cos" to "cos",
        "Cosh" to "cosh",
        "Erf" to "erf",
        "Exp" to "exp",
        "Identity" to "identity",
        "Log" to "log",
        "Sign" to "sign",
        "Sin" to "sin",
        "Sinh" to "sinh",
        "Softsign" to "softsign",
        "Tan" to "tan",
        "Tanh" to "tanh",

)

val pairWiseNames = mapOf(
        "And" to "boolean_and")


val sub = OnnxMappingProcess(
        inputFrameworkOpName = "Sub",
        opName = "subtract",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)

val mul = OnnxMappingProcess(
        inputFrameworkOpName = "Mul",
        opName = "multiply",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)


val divide = OnnxMappingProcess(
        inputFrameworkOpName = "Div",
        opName = "divide",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)


val add = OnnxMappingProcess(
        inputFrameworkOpName = "Add",
        opName = "add",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)
val reciprocal = OnnxMappingProcess(
        inputFrameworkOpName = "Reciprocal",
        opName = "reciprocal",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false, argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)

//Adagrad
//Adam


//unmapped: select_last_index
val argMax = OnnxMappingProcess(
        opName = "argmax",
        inputFrameworkOpName = "ArgMax",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                valueMappings(mutableMapOf("dimensions" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)

//unmapped: select_last_index
val argMin = OnnxMappingProcess(
        opName = "argmin",
        inputFrameworkOpName = "ArgMin",
        tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                valueMappings(mutableMapOf("dimensions" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)


//Note:  weight formats are NCHW in ONNX
val avgPool = OnnxMappingProcess(
        inputFrameworkOpName = "AveragePool",
        opName = "avgpool2d",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(argDescriptorConstants = listOf(ArgDescriptor {
                        name = "isNCHW"
                        int64Value = 0
                        argIndex = 10
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                })),
                intConstant(inputName = "dH",constantValue = 1,argumentIndex = 6)[0],
                intConstant(inputName = "dW",constantValue = 1,argumentIndex = 7)[0],
                intConstant(inputName = "extraParam0",constantValue = 0,argumentIndex = 9)[0],
                stringContainsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME",argumentIndex = 8),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "pads",indexValue = 2,argumentIndex = 4, defaultValueIfNotFound = ArgDescriptor {
                        argIndex = 4
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        int64Value = 0
                        name = "pH"
                }),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "pads",indexValue = 3,argumentIndex = 5,defaultValueIfNotFound = ArgDescriptor {
                        argIndex = 5
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        int64Value = 0
                        name = "pW"
                }),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0,argumentIndex = 2,defaultValueIfNotFound = ArgDescriptor {
                        argIndex = 2
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        int64Value = 1
                        name = "sH"
                }),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1,argumentIndex = 3,defaultValueIfNotFound = ArgDescriptor {
                        argIndex = 3
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        int64Value = 1
                        name = "sW"
                }),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1,argumentIndex = 1),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0,argumentIndex = 0)))


// ──────────────────────────────────────────────────────────────────────────────
// noop MAPPING PATTERN
//
// opName = "noop" suppresses direct op dispatch so a PreImportHook can fully
// handle the node instead.  There are two categories:
//
//   1. HOOKED — a PreImportHook class annotated @PreHookRule(opNames = ["X"])
//      exists in the implementations/ directory. The hook rewrites the node into
//      one or more supported SameDiff ops.  All of these are correct and intentional.
//
//   2. FRAMEWORK-HANDLED — the importer resolves the node before op dispatch
//      (e.g. Constant, Placeholder). No hook is needed.
//
//   3. UNSUPPORTED — no hook exists and the importer does not handle the op.
//      These produce silent no-ops in the imported graph, which is incorrect.
//      Each unsupported entry is marked with "UNSUPPORTED" below.
// ──────────────────────────────────────────────────────────────────────────────

// HOOKED: PreImportHook exists in implementations/AliasWithName.kt
val aliasWithName = OnnxMappingProcess(
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "AliasWithName"
)

//note: this is handled by the batchnorm class now
val batchNorm = OnnxMappingProcess(
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "BatchNormalization"
)


val embedLayerNormalization = OnnxMappingProcess(
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "EmbedLayerNormalization"
)

val binarizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Binarizer",
        opMappingRegistry = onnxOpRegistry
)

val bitshift = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "BitShift",
        opMappingRegistry = onnxOpRegistry
)

val arrayFeatureExtractor = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ArrayFeatureExtractor",
        opMappingRegistry = onnxOpRegistry
)

val castMap = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CastMap",
        opMappingRegistry = onnxOpRegistry
)

val categoryMapper = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CategoryMapper",
        opMappingRegistry = onnxOpRegistry
)

val celu = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Celu",
        opMappingRegistry = onnxOpRegistry
)

val compress = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Compress",
        opMappingRegistry = onnxOpRegistry
)

// Concat is handled by PreImportHook (Concat.kt) to support rank broadcasting
// ONNX allows concat of tensors with different ranks, but nd4j requires same rank
// The hook automatically reshapes lower-rank inputs by prepending dimensions of size 1
val concat = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Concat",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)
val concatFromSequence = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ConcatFromSequence",
        opMappingRegistry = onnxOpRegistry
)

val convInteger = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ConvInteger",
        opMappingRegistry = onnxOpRegistry
)

val convTranspose = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ConvTranspose",
        opMappingRegistry = onnxOpRegistry
)

val cumSum = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CumSum",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

val depthToSpace = OnnxMappingProcess(
        opName = "depth_to_space",
        inputFrameworkOpName = "DepthToSpace",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        //note onnx is NCHW by default
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "blocksize")),
                intConstant(inputName = "isNHWC",constantValue = 1,argumentIndex = 1)[0]),
        opMappingRegistry = onnxOpRegistry
)

val dequantizeLinear = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DequantizeLinear",
        opMappingRegistry = onnxOpRegistry
)

val determinant = OnnxMappingProcess(
        opName = "matrix_determinant",
        inputFrameworkOpName = "Det",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry
)




val floor = OnnxMappingProcess(
        opName = "floor",
        inputFrameworkOpName = "Floor",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry
)

val round = OnnxMappingProcess(
        opName = "round",
        inputFrameworkOpName = "Round",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry
)

val mod = OnnxMappingProcess(
        opName = "mod",
        inputFrameworkOpName = "Mod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry
)


val sigmoid = OnnxMappingProcess(
        opName = "sigmoid",
        inputFrameworkOpName = "Sigmoid",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        opMappingRegistry = onnxOpRegistry
)


// LogSoftmax handled by PreImportHook (LogSoftmax.kt) to ensure axis defaults to -1 per ONNX opset 13+
val logSoftmax = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LogSoftmax",
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf(),
        opMappingRegistry = onnxOpRegistry
)
// Softmax handled by PreImportHook (Softmax.kt) to ensure axis defaults to -1 per ONNX opset 13+
val softmax = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Softmax",
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf(),
        opMappingRegistry = onnxOpRegistry
)


val sequenceConstruct = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceConstruct",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

val sequenceAt = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceAt",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)


val sequenceEmpty = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceEmpty",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

val sequenceErase = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceErase",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

val sequenceinsert = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceInsert",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

// SequenceRemove: UNSUPPORTED — removes an element from an ONNX sequence type at a given index.
// No PreImportHook exists. noop causes silent pass-through.
// TODO: implement PreImportHook (sequence types require special list-handling in SameDiff).
val sequenceRemove = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceRemove",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)


val sequenceLength = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SequenceLength",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf()
)

val dynamicQuantizeLinear = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DynamicQuantizeLinear",
        opMappingRegistry = onnxOpRegistry
)

val einsum = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Einsum",
        opMappingRegistry = onnxOpRegistry
)

val eyeLike = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "EyeLike",
        opMappingRegistry = onnxOpRegistry
)

val featureVectorizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "FeatureVectorizer",
        opMappingRegistry = onnxOpRegistry
)

val gelu = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Gelu",
        opMappingRegistry = onnxOpRegistry
)

val gridSample = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GridSample",
        opMappingRegistry = onnxOpRegistry
)

val gru = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GRU",
        opMappingRegistry = onnxOpRegistry
)

val gatherElements = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GatherElements",
        opMappingRegistry = onnxOpRegistry
)

val gatherNd = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "GatherND",
        opName = "noop"  // Actual implementation in GatherND.kt PreImportHook
)


val ifOp = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "If",
        opMappingRegistry = onnxOpRegistry
)

val loop = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Loop",
        opMappingRegistry = onnxOpRegistry
)



val clip = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Clip",
        opMappingRegistry = onnxOpRegistry
)



val roiAlign = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RoiAlign",
        opMappingRegistry = onnxOpRegistry
)



val nonZero = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "NonZero",
        opMappingRegistry = onnxOpRegistry
)


//uses the Gemm Rule implementation instead
val gemm = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gemm",
        opName = "noop")

//note: no ops are mostly just stubs for ops implemented as pre processors
//These are implemented using the PreImportHook found: https://github.com/eclipse/deeplearning4j/tree/master/nd4j/samediff-import/samediff-import-onnx/src/main/kotlin/org/nd4j/samediff/frameworkimport/onnx/definitions/implementations
val globalAveragePooling = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GlobalAveragePool",
        opMappingRegistry = onnxOpRegistry
)
val globalLpPool = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GlobalLpPool",
        opMappingRegistry = onnxOpRegistry
)

val globalMaxPooling = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GlobalMaxPool",
        opMappingRegistry = onnxOpRegistry
)

val groupNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GroupNormalization",
        opMappingRegistry = onnxOpRegistry
)

val cast = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Cast",
        opMappingRegistry = onnxOpRegistry
)


val dictVectorizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DictVectorizer",
        opMappingRegistry = onnxOpRegistry
)

//Dropout: Note https://github.com/eclipse/deeplearning4j/issues/5650
val dropout = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Dropout",
        opMappingRegistry = onnxOpRegistry
)


val resize = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Resize",
        opMappingRegistry = onnxOpRegistry
)

//pytorch op
val resizeNearest = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ResizeNearest",
        opMappingRegistry = onnxOpRegistry
)

val constantOfShape = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ConstantOfShape",
        opMappingRegistry = onnxOpRegistry
)

val unsqueeze = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Unsqueeze",
        opMappingRegistry = onnxOpRegistry
)

val slice = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Slice",
        opMappingRegistry = onnxOpRegistry
)
val expand = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Expand",
        opMappingRegistry = onnxOpRegistry
)


val min = OnnxMappingProcess(
        inputFrameworkOpName = "Min",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry)

val max = OnnxMappingProcess(
        inputFrameworkOpName = "Max",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry)



//TODO: Gradient
//TODO: GraphCall
val hardSigmoid = OnnxMappingProcess(
        opName =  "hard_sigmoid",
        inputFrameworkOpName = "HardSigmoid",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

val hardSwish = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "HardSwish",
        opMappingRegistry = onnxOpRegistry
)

//TODO: map is-negative,is-positive
val isInf = OnnxMappingProcess(
        opName = "isinf",
        inputFrameworkOpName = "IsInf",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)



val or = OnnxMappingProcess(
        opName = "or",
        inputFrameworkOpName = "Or",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(
                doubleConstant(inputName = "comparable", constantValue = 0.0,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "A","y" to "B"))))
)

val xor = OnnxMappingProcess(
        opName = "boolean_xor",
        inputFrameworkOpName = "Xor",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace", constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "A","y" to "B"))))
)



val hardmax = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Hardmax",
        opMappingRegistry = onnxOpRegistry
)

val imputer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Imputer",
        opMappingRegistry = onnxOpRegistry
)

val instanceNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "InstanceNormalization",
        opMappingRegistry = onnxOpRegistry
)

val lrn = OnnxMappingProcess(
        opName = "lrn",
        inputFrameworkOpName = "LRN",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha","beta" to "beta","bias" to "bias","depth" to "size")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])

)

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

val lstm = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "LSTM",
        opName = "lstmLayer",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "X",
                "Wx" to "W",
                "Wr" to "R",
                "Wp" to "P",
                "b" to "B",
                "seqLen" to "sequence_lens",
                "hI" to "initial_h",
                "cI" to "initial_c"))),
        attributeMappingRules =  listOf(valueMappings(mapOf("cellClip" to "clip")),
                stringToIndex(outputAttributeValue = "directionMode",
                        inputAttributeValue = "direction",
                        listOfValues = listOf("forward","reverse","bidirectional"),argumentIndex = 1),
                intConstant(inputName = "dataFormat",constantValue = 0,argumentIndex = 0)[0],
                booleanConstant(inputName = "hasBiases",constantValue = true,argumentIndex = 0)[0],
                booleanConstant(inputName = "hasSeqLen",constantValue = true,argumentIndex = 1)[0],
                booleanConstant(inputName = "hasInitH",constantValue = true,argumentIndex = 2)[0],
                booleanConstant(inputName = "hasInitC",constantValue = true,argumentIndex = 3)[0],
                booleanConstant(inputName = "hasPH",constantValue = true,argumentIndex = 4)[0],
                booleanConstant(inputName = "retFullSeq",constantValue = true,argumentIndex = 5)[0],
                booleanConstant(inputName = "retLastH",constantValue = true,argumentIndex = 6)[0],
                booleanConstant(inputName = "retLastC",constantValue = true,argumentIndex = 7)[0],
                listAttributeValueLookup(outputAttributeValue = "gateAlpha",inputAttributeValue = "activation_alpha",indexValue = 0,argumentIndex = 1),
                listAttributeValueLookup(outputAttributeValue = "cellAlpha",inputAttributeValue = "activation_alpha",indexValue = 1,argumentIndex = 3),
                listAttributeValueLookup(outputAttributeValue = "outAlpha",inputAttributeValue = "activation_alpha",indexValue = 2,argumentIndex = 5),
                listAttributeValueLookup(outputAttributeValue = "gateBeta",inputAttributeValue = "activation_beta",indexValue = 0,argumentIndex = 2),
                listAttributeValueLookup(outputAttributeValue = "cellBeta",inputAttributeValue = "activation_beta",indexValue = 1,argumentIndex = 4),
                listAttributeValueLookup(outputAttributeValue = "outBeta",inputAttributeValue = "activation_beta",indexValue = 2,argumentIndex = 6),
                mapStringToInt(outputAttributeValue = "gateAct",inputAttributeValue = "activations",argumentIndex = 2,mapOfValuesToInts = lstmActivationMap,lookupIndex = 0),
                mapStringToInt(outputAttributeValue = "cellAct",inputAttributeValue = "activations",argumentIndex = 3,mapOfValuesToInts =lstmActivationMap,lookupIndex = 1),
                mapStringToInt(outputAttributeValue = "outAct",inputAttributeValue = "activations",argumentIndex = 4,mapOfValuesToInts = lstmActivationMap,lookupIndex = 2))
)
val labelEncoder = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LabelEncoder",
        opMappingRegistry = onnxOpRegistry
)

val leakyRelu = OnnxMappingProcess(
        inputFrameworkOpName = "LeakyRelu",
        opName = "leakyrelu",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha")),
                booleanConstant("inPlace",false,argumentIndex = 0)[0]),
        opMappingRegistry = onnxOpRegistry
)
val linearClassifier = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LinearClassifier",
        opMappingRegistry = onnxOpRegistry
)

val linearRegressor = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LinearRegressor",
        opMappingRegistry = onnxOpRegistry
)

val lpNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LpNormalization",
        opMappingRegistry = onnxOpRegistry
)

val lpPool = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LpPool",
        opMappingRegistry = onnxOpRegistry
)

val matMulInteger = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MatMulInteger",
        opMappingRegistry = onnxOpRegistry
)

val mish = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Mish",
        opMappingRegistry = onnxOpRegistry
)

val maxPool = OnnxMappingProcess(
        inputFrameworkOpName = "MaxPool",
        opName = "maxpool2d",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(
                argDescriptorConstant(argDescriptorConstants = listOf(ArgDescriptor {
                        name = "isNCHW"
                        int64Value = 0
                        argIndex = 10
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                })),
                intConstant(inputName = "extraParam0",argumentIndex = 9,constantValue = 0)[0],
                //note this parameter can be 0 for valid, 1 for same, 2 for causal
                intConstant(inputName = "isSameMode",constantValue = 0,argumentIndex = 8)[0],
                //stringContainsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME",argumentIndex = 8),
                listAttributeValueLookup(outputAttributeValue = "dH",inputAttributeValue = "dilations",indexValue = 0,argumentIndex = 6,defaultValueIfNotFound = ArgDescriptor {
                        int64Value = 1
                        name = "dH"
                        argIndex = 6
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                }),
                listAttributeValueLookup(outputAttributeValue = "dW",inputAttributeValue = "dilations",indexValue = 1,argumentIndex = 7,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 1
                                name = "dW"
                                argIndex = 7
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "pads",indexValue = 2,argumentIndex = 4,defaultValueIfNotFound = ArgDescriptor {
                        int64Value = 0
                        name = "pads"
                        argIndex = 4
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                }),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "pads",indexValue = 3,argumentIndex = 5,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 0
                                name = "pads"
                                argIndex = 5
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0,argumentIndex = 2,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 1
                                name = "sH"
                                argIndex = 6
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1,argumentIndex = 3,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 1
                                name = "sW"
                                argIndex = 7
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0,argumentIndex = 0),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1,argumentIndex = 1)))


val maxRoiPool = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MaxRoiPool",
        opMappingRegistry = onnxOpRegistry
)

val maxUnpool = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MaxUnpool",
        opMappingRegistry = onnxOpRegistry
)

val meanVarianceNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MeanVarianceNormalization",
        opMappingRegistry = onnxOpRegistry
)

val multinomial = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Multinomial",
        opMappingRegistry = onnxOpRegistry
)

val negativeLogLikelihoodLoss = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "NegativeLogLikelihoodLoss",
        opMappingRegistry = onnxOpRegistry
)

val nonMaxSuppression = OnnxMappingProcess(
        inputFrameworkOpName = "NonMaxSuppression",
        opName = "non_max_suppression_v3",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf("maxOutputSize" to "max_output_boxes_per_class"))),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "boxes" to "boxes",
                "scales" to "scores",
                "maxOutSize" to "max_output_boxes_per_class",
                "iouThreshold" to "iou_threshold",
                "scoreThreshold" to "score_threshold")))
)
val normalizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Normalizer",
        opMappingRegistry = onnxOpRegistry
)

val oneHot = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "OneHot",
        opMappingRegistry = onnxOpRegistry
)

val oneHotEncoder = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "OneHotEncoder",
        opMappingRegistry = onnxOpRegistry
)

//note: this is handled by the PRelu class now
val pRelu = OnnxMappingProcess(
        inputFrameworkOpName = "PRelu",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry
)

val pad = OnnxMappingProcess(
        inputFrameworkOpName = "Pad",
        opMappingRegistry = onnxOpRegistry,
        opName = "pad",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data","paddings" to "pads"))),
        attributeMappingRules = listOf(
                stringToIndex(outputAttributeValue = "mode",inputAttributeValue = "mode",listOfValues = listOf("constant","reflect","edge"),argumentIndex = 0),
                doubleConstant(inputName = "padValue",constantValue = 0.0,argumentIndex = 0)[0])
)

val qLinearConv = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "QLinearConv",
        opMappingRegistry = onnxOpRegistry
)

val qLinearMatMul = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "QLinearMatMul",
        opMappingRegistry = onnxOpRegistry
)

val quantizeLinear = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "QuantizeLinear",
        opMappingRegistry = onnxOpRegistry
)

val rnn = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RNN",
        opMappingRegistry = onnxOpRegistry
)

val randomNormal = OnnxMappingProcess(
        inputFrameworkOpName = "RandomNormal",
        opName = "random_normal",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(listNumberToNDarray(outputAttributeValue = "input",inputAttributeValue = "shape"))
)


val randomNormalLike = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RandomNormalLike",
        opMappingRegistry = onnxOpRegistry
)

//TODO: Note that the attributes for random unifrom are wrong and needed to be discovered through other means.
//The combination of a lack of a java class + the c++ calling out to other functions which had the actual parameters
//names prevented resolution of the real parameter names. May have to look in to values that are passed inline in to functions and look up
//parameter names that way.

val randomUniform = OnnxMappingProcess(
        inputFrameworkOpName = "RandomUniform",
        opName = "randomuniform",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(
                valueMappings(mapOf("min" to "low","max" to "high")),
                intConstant(inputName = "seed",constantValue = 0,argumentIndex = 0)[0],
                listNumberToNDarray(outputAttributeValue = "shape",
                        inputAttributeValue = "shape"))
)

val randomUniformLike = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RandomUniformLike",
        opMappingRegistry = onnxOpRegistry
)

val range = OnnxMappingProcess(
        inputFrameworkOpName = "Range",
        opName = "range",
        opMappingRegistry = onnxOpRegistry,
        // Keep inputs as dynamic tensors - do NOT convert to scalar attributes
        // This allows Range to work with dynamic sequence lengths (e.g., from Shape ops)
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("from" to "start","to" to "limit","step" to "delta")))
)

val neg = OnnxMappingProcess(
        opName = "neg",
        inputFrameworkOpName = "Neg",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)


val norm1 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL1",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm1",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes"))

)

val norm2 = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceL2",
        opMappingRegistry = onnxOpRegistry,
        opName = "reduce_norm2",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes"))
)

val reduceLogSum = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ReduceLogSum",
        opMappingRegistry = onnxOpRegistry
)

val reduceLogSumExp = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceLogSumExp",
        opName = "noop",  // Actual implementation in ReduceLogSumExp.kt PreImportHook (handles axes as input tensor in opset 18+)
        opMappingRegistry = onnxOpRegistry
)
val reduceMax = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMax",
        opName = "noop",  // Actual implementation in ReduceMax.kt PreImportHook (handles axes as input tensor in opset 18+)
        opMappingRegistry = onnxOpRegistry
)
val reduceMean = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMean",
        opName = "noop",  // Actual implementation in ReduceMean.kt PreImportHook (handles axes as input tensor in opset 18+)
        opMappingRegistry = onnxOpRegistry
)
val reduceMin = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMin",
        opName = "noop",  // Actual implementation in ReduceMin.kt PreImportHook (handles axes as input tensor in opset 18+)
        opMappingRegistry = onnxOpRegistry
)
val reduceProd = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceProd",
        opName = "noop",  // Actual implementation in ReduceProd.kt PreImportHook (handles axes as input tensor in opset 18+)
        opMappingRegistry = onnxOpRegistry
)

val reduceSum = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceSum",
        opName = "noop",  // Actual implementation in ReduceSum.kt PreImportHook (handles axes as input tensor in opset 13+)
        opMappingRegistry = onnxOpRegistry
)

//flattenDims
val flatten = OnnxMappingProcess(
        inputFrameworkOpName = "Flatten",
        opName = "flatten_2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mutableMapOf("flattenDimension" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)

//note this is implemented by Reshape.kt instead
val reshape = OnnxMappingProcess(
        inputFrameworkOpName = "Reshape",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry
)

val reduceSumSquare = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ReduceSumSquare",
        opMappingRegistry = onnxOpRegistry
)

//for mapping indices see: https://github.com/eclipse/deeplearning4j/blob/228f6cda30e27999f0fea74badc8d98ee8fb0647/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/enums/ImageResizeMethod.java#L29

val reverseSequence = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ReverseSequence",
        opMappingRegistry = onnxOpRegistry
)

val svmClassifier = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SVMClassifier",
        opMappingRegistry = onnxOpRegistry
)

val svmRegressor = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SVMRegressor",
        opMappingRegistry = onnxOpRegistry
)

val scaler = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Scaler",
        opMappingRegistry = onnxOpRegistry
)

val scan = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Scan",
        opMappingRegistry = onnxOpRegistry
)

val scatter = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "ScatterElements",
        opName = "scatter_update",
        attributeMappingRules =   listOf(),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("operand" to "data","updates" to "updates","indices" to "indices")))
)



val scatterNd = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ScatterND",
        opMappingRegistry = onnxOpRegistry
)

//TODO: SequenceAt
//TODO: SequenceConstruct
//TODO: SequenceErase
//TODO: SequenceInsert
//TODO: SequenceLength
val shape = OnnxMappingProcess(
        opName = "shape_of",
        inputFrameworkOpName = "Shape",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)
val shrink = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Shrink",
        opMappingRegistry = onnxOpRegistry
)

val not = OnnxMappingProcess(
        opName = "not",
        inputFrameworkOpName = "Not",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = doubleConstant(inputName = "comparable",constantValue = 0.0,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)


// Pow is handled by PreImportHook in implementations/Pow.kt for proper broadcasting support

val size = OnnxMappingProcess(
        opName = "size",
        inputFrameworkOpName = "Size",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)




val softmaxCrossEntropyLoss = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SoftmaxCrossEntropyLoss",
        opMappingRegistry = onnxOpRegistry
)

val spaceToDepth = OnnxMappingProcess(
        opName = "space_to_depth",
        inputFrameworkOpName = "SpaceToDepth",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mapOf("block_size" to "blocksize")),
                argDescriptorConstant(listOf(ArgDescriptor {
                        name = "isNHWC"
                        int64Value = 1
                        argIndex = 1
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64

                }))),
        opMappingRegistry = onnxOpRegistry
)

val split = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Split",
        opMappingRegistry = onnxOpRegistry,
)

val transpose = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Transpose",

        opMappingRegistry = onnxOpRegistry
)


val sqrt = OnnxMappingProcess(
        opName = "sqrt",
        inputFrameworkOpName = "Sqrt",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)

val softplus = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Softplus",
        opName = "softplus",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X")))
)

val splitToSequence = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SplitToSequence",
        opMappingRegistry = onnxOpRegistry
)

val squeeze = OnnxMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf( "_a" to  "axes")))
)

val stringNormalizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "StringNormalizer",
        opMappingRegistry = onnxOpRegistry
)

val tfIdfVectorizer = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "TfIdfVectorizer",
        opMappingRegistry = onnxOpRegistry
)

val treeEnsembleClassifier = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "TreeEnsembleClassifier",
        opMappingRegistry = onnxOpRegistry
)

val treeEnsembleRegressor = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "TreeEnsembleRegressor",
        opMappingRegistry = onnxOpRegistry
)

val thresholdedRelu = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ThresholdedRelu",
        opMappingRegistry = onnxOpRegistry
)

val tile = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Tile",
        opName = "tile",
        attributeMappingRules = listOf(
                booleanConstant(inputName = "is_static_reps",constantValue = true,argumentIndex = 0)[0],
                intConstant(inputName = "dimensions",constantValue = 0,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input","reps_vector" to "repeats")))
)

val topK = OnnxMappingProcess(
        opName = "top_k",
        inputFrameworkOpName = "TopK",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mutableMapOf("needSort" to "sorted")),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "k",inputAttributeValue = "K")),
        opMappingRegistry = onnxOpRegistry
)





val abs = OnnxMappingProcess(
        opName = "abs", tensorMappingRules = listOf(NDArrayMappingRule(mappingNamesToPerform = mutableMapOf("input" to "X"))),
        inputFrameworkOpName = "Abs",
        inputFramework = "onnx",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)



val ceil = defOnnxSingleTransform(inputFrameworkOpName = "Ceil",opName = "ceil",inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)


// Constant: framework-level op — the importer resolves this before op dispatch.
// No PreImportHook needed; noop suppresses dispatch so the importer handles it directly.
val const = OnnxMappingProcess(
        inputFrameworkOpName = "Constant",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf())

// Placeholder: not a real ONNX op — used as a dummy node to indicate inputs in
// SameDiff/TensorFlow parlance. No PreImportHook needed; the importer handles it
// as a graph input, not an executable op.
val placeHolder = OnnxMappingProcess(
        inputFrameworkOpName = "Placeholder",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf())

val conv2d = OnnxMappingProcess(
        inputFramework = "onnx",
        inputFrameworkOpName = "Conv",
        opName = "conv2d",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "input" to "X","weights" to "W","bias" to "B"))),
        attributeMappingRules = listOf(
                intConstant(inputName = "isNCHW",constantValue = 0,argumentIndex = 9)[0],
                intConstant(inputName = "wFormat",constantValue = 1,argumentIndex = 10)[0],
                stringEqualsRule(outputAttribute = "isSameMode",inputFrameworkAttributeName = "auto_pad",valueToTest = "SAME",argumentIndex = 8),
                listAttributeValueLookup(outputAttributeValue = "dH",inputAttributeValue = "dilations",indexValue = 0,argumentIndex = 6,defaultValueIfNotFound = ArgDescriptor {
                        int64Value = 1
                        name = "dH"
                        argIndex = 6
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                }),
                listAttributeValueLookup(outputAttributeValue = "dW",inputAttributeValue = "dilations",indexValue = 1,argumentIndex = 7,defaultValueIfNotFound = ArgDescriptor {
                        int64Value = 1
                        name = "dW"
                        argIndex = 7
                        argType = OpNamespace.ArgDescriptor.ArgType.INT64
                }),
                listAttributeValueLookup(outputAttributeValue = "pH",inputAttributeValue = "pads",indexValue = 0,argumentIndex = 4,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 0
                                name = "padding"
                                argIndex = 4
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "pW",inputAttributeValue = "pads",indexValue = 1,argumentIndex = 5,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 0
                                name = "padding"
                                argIndex = 5
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "sH",inputAttributeValue = "strides",indexValue = 0,argumentIndex = 2,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 1
                                name = "strides"
                                argIndex = 2
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "sW",inputAttributeValue = "strides",indexValue = 1,argumentIndex = 3,
                        defaultValueIfNotFound = ArgDescriptor {
                                int64Value = 1
                                name = "strides"
                                argIndex = 3
                                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                        }),
                listAttributeValueLookup(outputAttributeValue = "kW",inputAttributeValue = "kernel_shape",indexValue = 1,argumentIndex = 0),
                listAttributeValueLookup(outputAttributeValue = "kH",inputAttributeValue = "kernel_shape",indexValue = 0,argumentIndex = 1)
        ),opMappingRegistry = onnxOpRegistry)

val elu = defOnnxSingleTransform(opName = "elu",inputFrameworkOpName = "Elu",outputName = "input",inputFrameworkInput = "X",
        attributeMappingRules = listOf(valueMappings(mutableMapOf("alpha" to "alpha"))))



val relu = defOnnxSingleTransform(inputFrameworkOpName = "Relu",opName = "relu",
        inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0],
                doubleConstant(inputName = "cutoff",constantValue = 0.0,argumentIndex = 0)[0]))

val isNan = defOnnxSingleTransform(inputFrameworkOpName = "IsNaN",opName = "isnan",inputFrameworkInput = "X",outputName = "input",
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)


val selu = defOnnxSingleTransform(inputFrameworkOpName = "Selu",opName = "selu",inputFrameworkInput = "X",outputName = "input",attributeMappingRules =
booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)
)

val zipMap = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ZipMap",
        opMappingRegistry = onnxOpRegistry
)

// OCR-related operators for PaddleOCR and DeepSeek-OCR support

val ctcGreedyDecoder = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CTCGreedyDecoder",
        opMappingRegistry = onnxOpRegistry
)

val ctcGreedyDecoderAlt = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CTC_greedy_decoder",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveAvgPool1d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveAvgPool1d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveAvgPool2d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveAvgPool2d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveAvgPool2dAlt = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "adaptive_avg_pool2d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveAvgPool3d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveAvgPool3d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveMaxPool1d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveMaxPool1d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveMaxPool2d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveMaxPool2d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveMaxPool2dAlt = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "adaptive_max_pool2d",
        opMappingRegistry = onnxOpRegistry
)

val adaptiveMaxPool3d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AdaptiveMaxPool3d",
        opMappingRegistry = onnxOpRegistry
)

val mixtureOfExperts = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MixtureOfExperts",
        opMappingRegistry = onnxOpRegistry
)

val moe = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MoE",
        opMappingRegistry = onnxOpRegistry
)

val sparseMoe = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SparseMoE",
        opMappingRegistry = onnxOpRegistry
)

val deformConv = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DeformConv",
        opMappingRegistry = onnxOpRegistry
)

val deformableConv2d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DeformableConv2d",
        opMappingRegistry = onnxOpRegistry
)

val modulatedDeformConv = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ModulatedDeformConv",
        opMappingRegistry = onnxOpRegistry
)

val windowedAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "WindowedAttention",
        opMappingRegistry = onnxOpRegistry
)

val windowAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "WindowAttention",
        opMappingRegistry = onnxOpRegistry
)

val localAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LocalAttention",
        opMappingRegistry = onnxOpRegistry
)

val swinAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SwinAttention",
        opMappingRegistry = onnxOpRegistry
)

val relativePositionBias = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RelativePositionBias",
        opMappingRegistry = onnxOpRegistry
)

val relativePosEmb = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RelativePosEmb",
        opMappingRegistry = onnxOpRegistry
)

val relativePositionEmbedding = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RelativePositionEmbedding",
        opMappingRegistry = onnxOpRegistry
)

// Additional ops for Docling model support (transformers, vision models)

val trilu = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Trilu",
        opMappingRegistry = onnxOpRegistry
)

val col2Im = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Col2Im",
        opMappingRegistry = onnxOpRegistry
)

val im2Col = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Im2Col",
        opMappingRegistry = onnxOpRegistry
)

val unfold = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Unfold",
        opMappingRegistry = onnxOpRegistry
)

val centerCropPad = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "CenterCropPad",
        opMappingRegistry = onnxOpRegistry
)

val affineGrid = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "AffineGrid",
        opMappingRegistry = onnxOpRegistry
)

val unique = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Unique",
        opMappingRegistry = onnxOpRegistry
)

val scatterElements = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "ScatterElements",
        opMappingRegistry = onnxOpRegistry
)

// Signal processing ops for audio models
val stft = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "STFT",
        opMappingRegistry = onnxOpRegistry
)

val dft = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "DFT",
        opMappingRegistry = onnxOpRegistry
)

// MelWeightMatrix: UNSUPPORTED — no PreImportHook exists. Generates a triangular
// filter-bank matrix for mel-spectrogram computation. noop causes silent pass-through.
// TODO: implement PreImportHook using sd.math() operations or a native mel-filter op.
val melWeightMatrix = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MelWeightMatrix",
        opMappingRegistry = onnxOpRegistry
)

val hannWindow = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "HannWindow",
        opMappingRegistry = onnxOpRegistry
)

val hammingWindow = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "HammingWindow",
        opMappingRegistry = onnxOpRegistry
)

val blackmanWindow = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "BlackmanWindow",
        opMappingRegistry = onnxOpRegistry
)

// Optional handling ops
val optionalGetElement = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "OptionalGetElement",
        opMappingRegistry = onnxOpRegistry
)

val optionalHasElement = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "OptionalHasElement",
        opMappingRegistry = onnxOpRegistry
)

val optional = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Optional",
        opMappingRegistry = onnxOpRegistry
)

// Quantization ops (Microsoft ONNX extensions — no PreImportHook exists for these)
// QAttention: UNSUPPORTED — quantized multi-head attention (INT8 weights).
// noop causes silent pass-through. TODO: implement via QLinearMatMul decomposition.
val qAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "QAttention",
        opMappingRegistry = onnxOpRegistry
)

// QOrderedMatMul: UNSUPPORTED — ordered quantized matrix multiply (INT8).
// noop causes silent pass-through. TODO: implement via QLinearMatMul hook.
val qOrderedMatMul = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "QOrderedMatMul",
        opMappingRegistry = onnxOpRegistry
)

// Grid sampling variants
// GridSample3d: UNSUPPORTED — 3D volumetric grid sampling. No PreImportHook exists.
// The 2D variant (GridSample) has a full hook. noop causes silent pass-through.
// TODO: implement PreImportHook reusing the GridSample hook's interpolation logic for 3D.
val gridSample3d = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GridSample3d",
        opMappingRegistry = onnxOpRegistry
)

// LayerNormalization variants
val layerNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "LayerNormalization",
        opMappingRegistry = onnxOpRegistry
)

// Upsample (deprecated but still used in some models)
val upsample = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Upsample",
        opMappingRegistry = onnxOpRegistry
)

// Microsoft ONNX Runtime transformer ops
val attention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Attention",
        opMappingRegistry = onnxOpRegistry
)

val multiHeadAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "MultiHeadAttention",
        opMappingRegistry = onnxOpRegistry
)

val groupQueryAttention = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GroupQueryAttention",
        opMappingRegistry = onnxOpRegistry
)

val rotaryEmbedding = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "RotaryEmbedding",
        opMappingRegistry = onnxOpRegistry
)

val simplifiedLayerNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SimplifiedLayerNormalization",
        opMappingRegistry = onnxOpRegistry
)

val skipLayerNormalization = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "SkipLayerNormalization",
        opMappingRegistry = onnxOpRegistry
)

object OnnxOpDeclarations {
        @Volatile
        private var initialized = false

        fun init() {
                if (initialized) return
                synchronized(this) {
                        if (initialized) return
                        val onnxops = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")
                        val groupedOps = onnxops.values.groupBy { input -> input.name }
                        val singleGroupedOps = HashMap<String,Onnx.NodeProto>()
                        groupedOps.forEach { name,node ->
                                singleGroupedOps[name] = node[0]
                        }

                        OpRegistryHolder.registerOpList("onnx", singleGroupedOps)

                        names.forEach {
                                defineOnnxSingleTransform(inputFrameworkOpName = it.key,inputOpName = it.value)
                        } ?: "Error initializing single defined transforms in onnx."

                        pairWiseNames.forEach {
                                defineOnnxPairwiseTransforms(opName = it.value,inputFrameworkOpName = it.key)
                        } ?: "Error initializing pair wise transforms"

                        onnxops.values.forEach {
                                onnxOpRegistry.registerInputFrameworkOpDef(it.name,it)
                        }

                        OpDescriptorLoaderHolder.nd4jOpDescriptor.opListList.forEach {
                                onnxOpRegistry.registerNd4jOpDef(it.name,it)
                        }

                        MicrosoftOnnxExtensions.registerMicrosoftExtensions(onnxOpRegistry)


                        OpRegistryHolder.registerOpMappingRegistry("onnx", onnxOpRegistry)
                        initialized = true
                }
        }

        init {
             init()

        }
}


val declarations = OnnxOpDeclarations