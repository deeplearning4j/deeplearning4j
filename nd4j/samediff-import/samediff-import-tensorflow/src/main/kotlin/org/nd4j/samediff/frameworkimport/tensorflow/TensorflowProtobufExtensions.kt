/* ******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.samediff.frameworkimport.tensorflow

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcess
import org.nd4j.samediff.frameworkimport.tensorflow.rule.attribute.TensorflowArgDescriptorConstant
import org.nd4j.samediff.frameworkimport.tensorflow.rule.tensor.NDArrayMappingRule
import org.nd4j.samediff.frameworkimport.tensorflow.rule.tensor.TensorflowMultiInputIndexMappingRule
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.nio.charset.Charset

fun GraphDef.nodeByName(name: String): NodeDef {
    val nodeNames = nodeList.map { node -> node.name }
    return nodeList.first { it.name == name }!!
}

fun ListAttrValue(vararg i: Long): AttrValue.ListValue =
    AttrValue.ListValue.newBuilder().apply {
        i.forEach { addI(it) }
    }.build()

fun TensorProto(block: TensorProto.Builder.() -> Unit): TensorProto {
    return TensorProto.newBuilder().apply(block).build()
}

fun TensorProto.Builder.RawData(byteArray: ByteArray) {
    this.tensorContent = ByteString.copyFrom(byteArray)
}

fun TensorProto.Builder.Shape(shape: List<Long>) {
    this.tensorShape = TensorShapeProto {
        Dims(shape)
    }
}

fun TensorProto.Builder.DataType(value: DataType) {
    this.dtype = value
}

fun TensorProto.Builder.String(value: String) {
    this.addStringVal(ByteString.copyFrom(value.toByteArray(Charset.defaultCharset())))
}

fun TensorProto.Builder.StringData(value: List<String>) {
    this.addAllStringVal(value.map { value -> ByteString.copyFrom(value.toByteArray(Charset.defaultCharset())) })
}

fun TensorProto.Builder.Boolean(value: Boolean) {
    this.addBoolVal(value)
}

fun TensorProto.Builder.BooleanData(value: List<Boolean>) {
    this.addAllBoolVal(value)
}

fun TensorProto.Builder.Double(value: Double) {
    this.addDoubleVal(value)
}

fun TensorProto.Builder.Int64Data(value: List<Long>) {
    this.addAllInt64Val(value)
}


fun TensorProto.Builder.Int32Data(value: List<Int>) {
    this.addAllIntVal(value)
}

fun TensorProto.Builder.DoubleData(value: List<Double>) {
    this.addAllDoubleVal(value)
}

fun TensorProto.Builder.Float(value: Float) {
    this.addFloatVal(value)
}

fun TensorProto.Builder.FloatData(value: List<Float>) {
    this.addAllFloatVal(value)
}

fun TensorShapeProto.Builder.Dim(name: String, size: Long) {
    this.addDim(TensorShapeProto.Dim.newBuilder().setName(name).setSize(size).build())
}

fun Dim(block: TensorShapeProto.Dim.Builder.() -> Unit): TensorShapeProto.Dim {
    return TensorShapeProto.Dim.newBuilder().apply(block).build()
}

fun TensorShapeProto.Builder.Dims(shape: List<Long>) {
    shape.forEachIndexed  { index, value ->  this.addDim(
        Dim {
            name = index.toString()
            size = value
        })
    }
}

fun TensorShapeProto(block: TensorShapeProto.Builder.() -> Unit): TensorShapeProto {
    return TensorShapeProto.newBuilder().apply(block).build()
}

fun AttrValue(block: AttrValue.Builder.() -> Unit): AttrValue {
    return AttrValue.newBuilder().apply(block).build()
}



fun AttrValue.Builder.ListDataType(listDataTypes: List<DataType>) {
    this.listBuilder.addAllType(listDataTypes)
}

fun AttrValue.Builder.ListInts(listInts: List<Long>) {
    this.listBuilder.addAllI(listInts)
}

fun AttrValue.Builder.LongVal(intVal: Long) {
    this.i = intVal
}

fun AttrValue.Builder.ListFloats(listFloats: List<Float>) {
    this.listBuilder.addAllF(listFloats)
}



fun GraphDef(block: GraphDef.Builder.() -> Unit): GraphDef {
    return GraphDef.newBuilder().apply(block).build()
}

fun GraphDef.Builder.Node(inputNode: NodeDef) {
    this.addNode(inputNode)
}

fun String.toByteString() = ByteString.copyFrom(this, Charset.defaultCharset())

fun OpDef(block: OpDef.Builder.() -> Unit): OpDef {
    return OpDef.newBuilder().apply(block).build()
}

fun NodeDef(block: NodeDef.Builder.() -> Unit): NodeDef {
    return NodeDef.newBuilder().apply(block).build()
}

fun ListValue(block: AttrValue.ListValue.Builder.() -> Unit): AttrValue.ListValue {
    return AttrValue.ListValue.newBuilder().apply(block).build()
}

fun AttrValue.ListValue.Builder.LongItems(value: List<Long>) {
    this.addAllI(value)
}

fun AttrValue.ListValue.Builder.IntItems(value: List<Int>) {
    this.addAllI(value.map { it.toLong() })
}

fun AttrValue.ListValue.Builder.IntItem(value: Long) {
    this.addI(value)
}

fun NodeDef.Builder.Input(name: String) {
    this.addInput(name)
}

fun NodeDef.Builder.Attribute(name: String, value: AttrValue) {
    this.putAttr(name, value)
}

fun OpList.findOp(name: String): OpDef {
    if(!this.opList.map { input -> input.name }.contains(name)) {
        throw IllegalArgumentException("Op $name not found!")
    }
    return this.opList.first { it.name == name }!!
}


fun mappingListNDArrays(inputs: MutableMap<String, String>) : TensorflowMultiInputIndexMappingRule {
    return TensorflowMultiInputIndexMappingRule(
        mappingNamesToPerform = inputs
    )
}

fun booleanConstant(inputName: String, constantValue: Boolean,argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            name = inputName
            boolValue = constantValue
            argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            argIndex = argumentIndex
        }
    )))
}

fun doubleConstant(inputName: String, constantValue: Double, argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
            name = inputName
            doubleValue = constantValue
            argIndex = argumentIndex
        }
    )))
}

fun intConstant(inputName: String, constantValue: Int, argumentIndex: Int): List<TensorflowArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            name = inputName
            int64Value = constantValue.toLong()
            argIndex = argumentIndex
        }
    )))
}

fun mappingNDArrayInputs(inputs: MutableMap<String, String>) : NDArrayMappingRule {
    return NDArrayMappingRule(
        mappingNamesToPerform = inputs
    )
}

fun mapSameName(names: List<String>): List<NDArrayMappingRule> {
    return listOf(mappingNDArrayInputs(names.map { name -> Pair(name, name) }.toMap().toMutableMap()))
}

fun mapTensorNamesWithOp(inputFrameworkOpName: String,
                         opName: String,
                         tensorflowOpRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>,
                         tensorNames: MutableMap<String,String>,
                         attributeMappingRules: List<AttributeMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>> = emptyList()): TensorflowMappingProcess {
    return TensorflowMappingProcess(
        opName = opName,
        inputFrameworkOpName = inputFrameworkOpName,
        opMappingRegistry = tensorflowOpRegistry,
        tensorMappingRules = listOf(mappingNDArrayInputs(tensorNames)),
        attributeMappingRules = attributeMappingRules.toMutableList()
    )

}

fun multipleNameMapping(inputFrameworkOpNames: List<String>,
                        tensorflowOpRegistry: OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>,
                        opName: String, tensorNames: MutableMap<String, String>,
                        attributeMappingRules: List<AttributeMappingRule<GraphDef, OpDef, NodeDef, OpDef.AttrDef, AttrValue,
                                TensorProto, DataType>> = emptyList()):
        List<TensorflowMappingProcess> {
    return inputFrameworkOpNames.map {
        val attrCopy = attributeMappingRules.toMutableList()
        attrCopy.forEach { attr ->
            attr.modifyInputFrameworkOpName(it)
        }
        mapTensorNamesWithOp(
            inputFrameworkOpName = it,
            opName = opName,
            tensorNames = tensorNames,
            attributeMappingRules = attrCopy,
            tensorflowOpRegistry = tensorflowOpRegistry
        )
    }
}

fun defineBiasAdd(names :List<String> =  listOf("BiasAdd"),
                  tensorflowOpRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>) {
    names.forEach {
        TensorflowMappingProcess(
            opName = "biasadd",
            inputFrameworkOpName = it,
            opMappingRegistry = tensorflowOpRegistry,
            tensorMappingRules = listOf(mappingNDArrayInputs(mutableMapOf("input" to "value", "bias" to "bias"))),
            attributeMappingRules = listOf(
                stringEqualsRule("nchw"
                ,inputFrameworkAttributeName = "data_format",valueToTest = "NCHW",argumentIndex = 0)
            ))

    }
}

fun defineTensorflowSingleTransform(inputOpName: String,
                                    inputFrameworkOpName: String,
                                    tensorflowOpRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>
): TensorflowMappingProcess {
    return TensorflowMappingProcess(
        opName = inputOpName,
        inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules = listOf(
            NDArrayMappingRule(
                mappingNamesToPerform = mutableMapOf("input" to "x")
            )
        ),
        attributeMappingRules = listOf(
            valueMapping(mutableMapOf("dataType" to "T")),
            argDescriptorConstant(
            listOf(
                ArgDescriptor {
                    name = "inPlace"
                    boolValue = false
                    argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                    argIndex = 0
                }
            )
        )),
        opMappingRegistry = tensorflowOpRegistry)

}

fun defineSingularReduce(inputFrameworkOpName: String,
                         inputOpName: String,
                         tensorflowOpRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType, OpDef.AttrDef,AttrValue>
): TensorflowMappingProcess {
    return mapTensorNamesWithOp(
        inputFrameworkOpName = inputFrameworkOpName,
        opName = inputOpName,
        attributeMappingRules = listOf(
            valueMapping(mutableMapOf("keepDims" to "keep_dims")),
            ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices"))
        ),
        tensorNames = mutableMapOf("input" to "input"),
        tensorflowOpRegistry = tensorflowOpRegistry
    )
}

fun definePairWiseReduce(inputFrameworkOpName: String, inputOpName: String,tensorflowOpRegistry: OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>): TensorflowMappingProcess {
    return mapTensorNamesWithOp(
        inputFrameworkOpName = inputFrameworkOpName,
        opName = inputOpName,
        attributeMappingRules = listOf(
            valueMapping(mutableMapOf("keepDims" to "keep_dims")),
            ndarrayToIntList(mutableMapOf("dimensions" to "reduction_indices"))
        ),
        tensorNames = mutableMapOf("input" to "input"),
        tensorflowOpRegistry = tensorflowOpRegistry
    )
}

fun defineTensorflowPairwiseTransforms(opName: String, inputFrameworkOpName: String,
                                       tensorflowOpRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType,
                                               OpDef.AttrDef,AttrValue>,
                                       firstOutputName: String = "input",
                                       secondOutputName: String = "y",
                                       firstInput: String = "x", secondInput: String = "y") : TensorflowMappingProcess {
    return TensorflowMappingProcess(
        opName = opName,
        tensorMappingRules = listOf(
            NDArrayMappingRule(
                mappingNamesToPerform = mutableMapOf(
                    firstOutputName to firstInput,
                    secondOutputName to secondInput
                )
            )
        ),
        inputFrameworkOpName = inputFrameworkOpName,
        inputFramework = "tensorflow",
        attributeMappingRules = listOf(booleanConstant(inputName = "inPlace", constantValue = false, argumentIndex = 0)[0]
            ,valueMapping(mutableMapOf("dataType" to "T"))),
        opMappingRegistry = tensorflowOpRegistry
    )
}
