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
        "Tanh" to "tanh"

)

val pairWiseNames = mapOf(
        "And" to "boolean_and")

val equal = OnnxMappingProcess(
        inputFrameworkOpName = "Equal",
        opName = "equals",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B"))),
        attributeMappingRules = booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry)


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

//TODO: Binarizer
//TODO: Bitshift
//TODO: CastMap
//TODO: CategoryMapper
//TODO: Celu
//TODO: Compress
val concat = OnnxMappingProcess(
        opName = "concat",
        inputFrameworkOpName = "Concat",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "inputs"))),
        attributeMappingRules = listOf(valueMappings(mapOf("concatDimension" to "axis")),
                booleanConstant(inputName = "isDynamicAxis",constantValue = false,argumentIndex = 0)[0]),
        variableResolutionType = MapperNamespace.VariableResolutionType.DIRECT

)
//TODO: ConcatFromSequence


//TODO: ConvInteger
//TODO: ConvTranspose
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

//TODO: DequantizeLinear
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


val logSoftmax = OnnxMappingProcess(
        opName = "log_softmax",
        inputFrameworkOpName = "LogSoftmax",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mutableMapOf("dimension" to "axis"))),
        opMappingRegistry = onnxOpRegistry
)
val softmax = OnnxMappingProcess(
        opName = "softmax",
        inputFrameworkOpName = "Softmax",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "input"))),
        attributeMappingRules = listOf(valueMappings(mutableMapOf("dimension" to "axis")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
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

//TODO: DynamicQuantizeLinear
//TODO: Einsum
//TODO: EyeLike
//TODO: FeatureVectorizer
val gru = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GRU",
        opMappingRegistry = onnxOpRegistry
)
val gather = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "Gather",
        opName = "gather",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data"))),
        attributeMappingRules = listOf(valueMappings(mapOf("dimensions" to "axis")),
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0])
)
//TODO: GatherElements
val gatherNd = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "GatherND",
        opName = "gather_nd",
        attributeMappingRules = booleanConstant(inputName = "checkIndices",constantValue = true,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("indices" to "indices","input" to "data")))
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
//TODO: GlobalLpPool
val globalMaxPooling = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "GlobalMaxPool",
        opMappingRegistry = onnxOpRegistry
)

val cast = OnnxMappingProcess(
        opName = "noop",
        inputFrameworkOpName = "Cast",
        opMappingRegistry = onnxOpRegistry
)


//TODO: DictVectorizer
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
        opName = "bitwise_xor",
        inputFrameworkOpName = "Xor",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace", constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "A","y" to "B"))))
)



//TODO: Hardmax
//TODO: Imputer
//TODO: InstanceNormalization
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
//TODO: LabelEncoder
val leakyRelu = OnnxMappingProcess(
        inputFrameworkOpName = "LeakyRelu",
        opName = "leakyrelu",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "X"))),
        attributeMappingRules = listOf(valueMappings(mapOf("alpha" to "alpha")),
                booleanConstant("inPlace",false,argumentIndex = 0)[0]),
        opMappingRegistry = onnxOpRegistry
)
//TODO: LinearClassifier
//TODO: LinearRegressor
//TODO: Loop
//TODO: LpNormalization
//TODO: LpPool
val matMul = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "MatMul",
        opName = "matmul",
        attributeMappingRules = listOf(
                booleanConstant(inputName = "transX",constantValue = false,argumentIndex = 0)[0],
                booleanConstant(inputName = "transY",constantValue = false,argumentIndex = 1)[0],
                booleanConstant(inputName = "transZ",constantValue = false,argumentIndex = 2)[0],
                booleanConstant(inputName = "transposeX",constantValue = false,argumentIndex = 0)[0],
                booleanConstant(inputName = "transposeY",constantValue = false,argumentIndex = 1)[0],
                booleanConstant(inputName = "transposeZ",constantValue = false,argumentIndex = 2)[0],
                doubleConstant(inputName = "alpha",constantValue = 0.0,argumentIndex = 0)[0],
                doubleConstant(inputName = "beta",constantValue = 1.0,argumentIndex = 1)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "A","y" to "B")))
)


//TODO: MatMulInteger
//TODO: Max
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


//TODO: MaxRoiPool
//TODO: MaxUnpool
//TODO: name: "MeanVarianceNormalization"
//todo: Momentum
//TODO: Multinomial
//TODO: NegativeLogLikelihoodLoss
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
//TODO: NonZero PRIORITIZE
//TODO: Normalizer
//TODO: OneHot
//TODO: OneHotEncoder
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

//TODO: QLinearConv
//TODO: QLinearMatMul
//TODO: QuantizeLinear
//TODO: RNN PRIORITIZE
val randomNormal = OnnxMappingProcess(
        inputFrameworkOpName = "RandomNormal",
        opName = "random_normal",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(listNumberToNDarray(outputAttributeValue = "input",inputAttributeValue = "shape"))
)


//TODO: RandomNormalLike
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

//TODO: RandomUniformLike
val range = OnnxMappingProcess(
        inputFrameworkOpName = "Range",
        opName = "range",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("from" to "start","to" to "limit","step" to "delta"))),
        attributeMappingRules = listOf(
                convertNDArrayInputToScalarAttr(outputAttributeValue = "from",inputAttributeValue = "start"),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "to",inputAttributeValue = "limit"),
                convertNDArrayInputToScalarAttr(outputAttributeValue = "step",inputAttributeValue = "delta"))
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

//TODO: ReduceLogSum
val reduceLogSumExp = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceLogSumExp",
        opName = "reduce_logsumexp",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mutableMapOf("keepDims" to "keepdims")),
                valueMappings(mutableMapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMax = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMax",
        opName = "reduce_max",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMean = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMean",
        opName = "reduce_mean",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceMin = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceMin",
        opName = "reduce_min",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(
                invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)
val reduceProd = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceProd",
        opName = "reduce_prod",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                listNumberToListNumber(outputAttributeValue =  "dimensions",inputAttributeValue = "axes")),
        opMappingRegistry = onnxOpRegistry
)

val reduceSum = OnnxMappingProcess(
        inputFrameworkOpName = "ReduceSum",
        opName = "reduce_sum",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(invertBooleanNumber(mapOf("keepDims" to "keepdims")),
                ndarrayToIntList(mutableMapOf( "dimensions" to "axes"))),
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

//TODO: ReduceSumSquare
//for mapping indices see: https://github.com/eclipse/deeplearning4j/blob/228f6cda30e27999f0fea74badc8d98ee8fb0647/nd4j/nd4j-backends/nd4j-api-parent/nd4j-api/src/main/java/org/nd4j/enums/ImageResizeMethod.java#L29

//TODO: ReverseSequence
//TODO: RoiAlign
//TODO: SVMClassifier
//TODO: SVMRegressor
//TODO: Scaler
//TODO: Scan
val scatter = OnnxMappingProcess(
        opMappingRegistry = onnxOpRegistry,
        inputFrameworkOpName = "ScatterElements",
        opName = "scatter_update",
        attributeMappingRules =   listOf(),
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("operand" to "data","updates" to "updates","indices" to "indices")))
)



val scatterNd = OnnxMappingProcess(inputFrameworkOpName ="ScatterND",
        opName = "scatter_nd",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf(
                "indices" to "indices",
                "updates" to "updates","shape" to "data"))),
        attributeMappingRules = listOf()
        ,opMappingRegistry = onnxOpRegistry)

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
//TODO: Shrink

val not = OnnxMappingProcess(
        opName = "not",
        inputFrameworkOpName = "Not",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = doubleConstant(inputName = "comparable",constantValue = 0.0,argumentIndex = 0),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X"))))
)


val pow = OnnxMappingProcess(
        opName = "pow_pairwise",
        inputFrameworkOpName = "Pow",
        opMappingRegistry = onnxOpRegistry,
        attributeMappingRules = listOf(
                booleanConstant(inputName = "inPlace",constantValue = false,argumentIndex = 0)[0]),
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "X","y" to "Y"))))
)

val size = OnnxMappingProcess(
        opName = "size",
        inputFrameworkOpName = "Size",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs((mutableMapOf("input" to "data"))))
)




//TODO: SoftmaxCrossEntropyLoss
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

//TODO: SplitToSequence
val squeeze = OnnxMappingProcess(
        opName = "squeeze",
        inputFrameworkOpName = "Squeeze",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "data"))),
        attributeMappingRules = listOf(ndarrayToIntList(mutableMapOf( "_a" to  "axes")))
)

//TODO: StringNormalizer
//TODO: TfIdfVectorizer
//TODO: ThresholdedRelu
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


val where = OnnxMappingProcess(
        inputFrameworkOpName = "Where",
        opName = "Where",
        tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("condition" to "condition","input" to "X","y" to "Y"))),
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


val const = OnnxMappingProcess(
        inputFrameworkOpName = "Constant",
        opName = "noop",
        opMappingRegistry = onnxOpRegistry,
        tensorMappingRules = listOf(),
        attributeMappingRules = listOf())

//note: this is not a real onnx op and meant to be a dummy node to indicate placeholders
//samediff/tensorflow parlance
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


object OnnxOpDeclarations {
        init {
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

        }
}


val declarations = OnnxOpDeclarations