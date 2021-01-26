package org.nd4j.codegen.ir.tensorflow

import org.nd4j.codegen.ir.*
import org.nd4j.ir.OpNamespace
import org.tensorflow.framework.*


class TensorflowConditionalFieldValueIntIndexNDArrayRule
    (mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    ConditionalFieldValueIntIndexNDArrayRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        (mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<
            GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }


}

fun conditionalFieldValueIntIndexNDArrayRule(outputAttribute: String,
                                             inputFrameworkStringNameToTest: String,
                                             targetValue: String,
                                             trueIndex: Int,
                                             falseIndex: Int,
                                             attributeNameOfListAttribute: String,
                                             argumentIndex: Int): TensorflowConditionalFieldValueIntIndexNDArrayRule {
    return TensorflowConditionalFieldValueIntIndexNDArrayRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkStringNameToTest),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = "targetValue"
                stringValue = targetValue
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "trueIndex"
                int32Value = trueIndex
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "falseIndex"
                int32Value = falseIndex
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "attributeNameOfListAttribute"
                stringValue = attributeNameOfListAttribute
                argIndex = argumentIndex
            }))
    )
}





class TensorflowConditionalFieldValueIntIndexArrayRule
    (mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    ConditionalFieldValueIntIndexArrayRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        (mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }

    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<
            GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }


}

fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkStringNameToTest: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int,
                                           attributeNameOfListAttribute: String,
                                           argumentIndex: Int): TensorflowConditionalFieldValueIntIndexArrayRule {
    return TensorflowConditionalFieldValueIntIndexArrayRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkStringNameToTest),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = "targetValue"
                stringValue = targetValue
                argIndex = argIndex
            },
            ArgDescriptor {
                name = "trueIndex"
                int32Value = trueIndex
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "falseIndex"
                int32Value = falseIndex
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "attributeNameOfListAttribute"
                stringValue = attributeNameOfListAttribute
                argIndex = argumentIndex
            }))
    )
}

class TensorflowNDArraySizeAt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
    NDArraySizeAtRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun sizeAtRule(dimensionIndex: Int,
               outputAttributeName: String,
               inputFrameworkAttributeName: String,
               argumentIndex: Int): TensorflowNDArraySizeAt {
    return TensorflowNDArraySizeAt(
        mappingNamesToPerform = mapOf(outputAttributeName to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttributeName to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            int32Value = dimensionIndex
            argIndex = argumentIndex
        }.build()))
    )
}

class TensorflowNDArrayExtractScalarValue(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    NDArrayExtractScalarValue<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun ndarrayExtractScalarValue(outputAttribute: String,
                              inputFrameworkAttributeName: String,
                              argumentIndex: Int,
                              scalarIndex: Int): TensorflowNDArrayExtractScalarValue {
    return TensorflowNDArrayExtractScalarValue(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = outputAttribute
                int64Value = scalarIndex.toLong()
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                argIndex = argumentIndex
            })))
}




class TensorflowStringEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                        transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    StringEqualsAdapterRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun stringEqualsRule(outputAttribute: String,
                     inputFrameworkAttributeName: String,
                     valueToTest: String,
                     argumentIndex: Int): TensorflowStringEqualsAdapterRule {
    return TensorflowStringEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
                argType = OpNamespace.ArgDescriptor.ArgType.STRING
                argIndex = argumentIndex
            })))
}


class TensorflowStringNotEqualsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                           transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    StringNotEqualsAdapterRule<GraphDef,OpDef, NodeDef,
            OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String,
                                               mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String,
                                                mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun stringNotEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String,argumentIndex: Int): TensorflowStringNotEqualsAdapterRule {
    return TensorflowStringNotEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
            argIndex = argumentIndex
        }.build())))
}


class TensorflowStringContainsAdapterRule(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    StringContainsAdapterRule<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun stringContainsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowStringContainsAdapterRule {
    return TensorflowStringContainsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
        }.build())))
}


class TensorflowAttributeScalarNDArrayAttribute(mappingNamesToPerform: Map<String, String> = emptyMap(),
                                                transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()) :
    AttributeScalarNDArrayAttribute<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>
        ( mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun attributeScalarToNDArrayInput(outputAttribute: String, inputFrameworkAttributeName: String): TensorflowAttributeScalarNDArrayAttribute {
    return TensorflowAttributeScalarNDArrayAttribute(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName))
}




class TensorflowValueMappingRule(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    ValueMapping<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun valueMapping(mappings: Map<String,String>): TensorflowValueMappingRule {
    return TensorflowValueMappingRule(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}

class TensorflowInvertBooleanNumber(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    InvertBooleanNumber<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef, attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun invertBooleanNumber(mappings: Map<String,String>): TensorflowInvertBooleanNumber {
    return TensorflowInvertBooleanNumber(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}





class TensorflowNDArrayToIntAttributeValue(mappingNamesToPerform: Map<String, String>) : NDArrayToIntAttributeValue<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform = mappingNamesToPerform,transformerArgs = emptyMap()) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(attrDef,attributeValueType)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun ndarrayToIntList(ndarrayNameToAttributeName: MutableMap<String,String>): TensorflowNDArrayToIntAttributeValue {
    return TensorflowNDArrayToIntAttributeValue(mappingNamesToPerform = ndarrayNameToAttributeName)
}

class TensorflowNdArrayToStringIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : StringToInt<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun ndarrayStringToIndex(outputAttributeValue: String,inputAttributeValue: String, listOfValues: List<String>,argumentIndex: Int): TensorflowNdArrayToStringIndex {
    return TensorflowNdArrayToStringIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = mapOf(outputAttributeValue to listOfValues.map {
            valueName -> ArgDescriptor {
        name = valueName
        stringValue = valueName
        argIndex = argumentIndex
    }
    }))
}


class TensorflowMapStringToInt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : MapStringToInt<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun mapStringToInt(outputAttributeValue: String, inputAttributeValue: String, mapOfValuesToInts: Map<String,Int>,argumentIndex: Int,lookupIndex:Int): TensorflowMapStringToInt {
    return TensorflowMapStringToInt(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs =
    mapOf(outputAttributeValue to mapOfValuesToInts.map {
            entry -> ArgDescriptor {
        name = entry.key
        int64Value = entry.value.toLong()
        argIndex = argumentIndex
    }
    },"index" to listOf(ArgDescriptor {
        name = "index"
        int64Value = lookupIndex.toLong()
    })))
}




class TensorflowListNumberToListNumber(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListNumberToListNumber<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun listNumberToListNumber(outputAttributeValue: String, inputAttributeValue: String): TensorflowListNumberToListNumber {
    return TensorflowListNumberToListNumber(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}

class TensorflowStringAttributeToNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : StringAttributeToNDArray<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun convertStringToInputNDArray(mappings: Map<String,String>): TensorflowStringAttributeToNDArray {
    return TensorflowStringAttributeToNDArray(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}








class TensorflowAttributeNumberListNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : AttributeNumberListNDArray<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun convertNumberListToInputNDArray(outputAttributeValue: String, inputAttributeValue: String): TensorflowAttributeNumberListNDArray {
    return TensorflowAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


class TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) : ListAttributeValueLookupToIndex<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun listAttributeValueLookupToIndex(outputAttributeValue: String, inputAttributeValue: String, idx: Int,argumentIndex: Int): TensorflowListAttributeValueLookupToIndex {
    return TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
        transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            int64Value = idx.toLong()
            name = "index"
            argIndex = argumentIndex
        })))
}





class TensorflowDataTypeToInt(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    DataTypeToInt<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun dataTypeToInt(mutableMap: MutableMap<String,String>): TensorflowDataTypeToInt {
    return TensorflowDataTypeToInt(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}




class TensorflowNDArrayInputToNumericalAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    NDArrayInputToNumericalAttribute<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun convertNDArrayInputToNumericalAttr(mutableMap: MutableMap<String,String>): TensorflowNDArrayInputToNumericalAttribute {
    return TensorflowNDArrayInputToNumericalAttribute(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}

class TensorflowListNumberToNDArray(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    ListNumberToNDArray<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun listNumberToNDarray(mutableMap: MutableMap<String,String>): TensorflowListNumberToNDArray {
    return TensorflowListNumberToNDArray(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


class TensorflowNDArrayAttributeToNDArrayInput(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>) :
    NDArrayAttributeToNDArrayInput<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun ndArrayAttributeToNDarrayInput(mutableMap: MutableMap<String,String>): TensorflowNDArrayAttributeToNDArrayInput {
    return TensorflowNDArrayAttributeToNDArrayInput(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


class TensorflowArgDescriptorConstant(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>)
    : ArgDescriptorConstant<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun argDescriptorConstant(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): TensorflowArgDescriptorConstant {
    return TensorflowArgDescriptorConstant(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}


class TensorflowAttributeNDArrayToScalarAttribute(mappingNamesToPerform: Map<String, String>, transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>)
    : AttributeNDArrayToScalarAttribute<GraphDef,OpDef, NodeDef, OpDef.AttrDef, AttrValue, TensorProto, DataType>(mappingNamesToPerform, transformerArgs) {

    override fun createIRAttribute(name: String, attrDef: OpDef.AttrDef, attributeValueType: AttrValue): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
        return TensorflowIRAttr(inputAttributeValue = attributeValueType,inputAttributeDef = attrDef)
    }

    override fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        TODO("Not yet implemented")
    }
    override fun isInputFrameworkTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowTensorName(name,opDef)
    }

    override fun isNd4jTensorName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isNd4jTensorName(name,nd4jOpDescriptor)
    }

    override fun isInputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return isTensorflowAttributeName(name,opDef)
    }

    override fun isOutputFrameworkAttributeName(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): Boolean {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return isOutputFrameworkAttributeName(name,nd4jOpDescriptor)
    }

    override fun argDescriptorType(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): OpNamespace.ArgDescriptor.ArgType {
        val nd4jOpDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        return argDescriptorType(name,nd4jOpDescriptor)
    }

    override fun attributeValueTypeFor(name: String, mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>): AttributeValueType {
        val opDef = tensorflowOps.findOp(mappingProcess.inputFrameworkOpName())
        return tensorflowAttributeValueTypeFor(attributeName = name,opDef = opDef)
    }
}

fun ndarrayAttributeToScalarAttribute(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): TensorflowAttributeNDArrayToScalarAttribute {
    return TensorflowAttributeNDArrayToScalarAttribute(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}