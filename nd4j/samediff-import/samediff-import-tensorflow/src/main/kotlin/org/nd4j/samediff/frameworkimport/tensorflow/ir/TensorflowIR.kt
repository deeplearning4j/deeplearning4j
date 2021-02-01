package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.apache.commons.io.IOUtils
import org.nd4j.common.io.ClassPathResource
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.shade.protobuf.TextFormat
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import java.nio.charset.Charset

fun loadTensorflowOps(): OpList {
    val string = IOUtils.toString(ClassPathResource("ops.proto").inputStream, Charset.defaultCharset())
    val tfListBuilder = OpList.newBuilder()
    TextFormat.merge(string, tfListBuilder)
    return tfListBuilder.build()
}



fun attrDefaultValue(): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
    return TensorflowIRAttr(AttrDef.getDefaultInstance(), AttrValue.getDefaultInstance())
}


fun convertToDataType(dataType: org.nd4j.linalg.api.buffer.DataType): DataType {
    return when (dataType) {
        org.nd4j.linalg.api.buffer.DataType.UINT16 -> DataType.DT_UINT16
        org.nd4j.linalg.api.buffer.DataType.UINT32 -> DataType.DT_UINT32
        org.nd4j.linalg.api.buffer.DataType.UINT64 -> DataType.DT_UINT64
        org.nd4j.linalg.api.buffer.DataType.BOOL -> DataType.DT_BOOL
        org.nd4j.linalg.api.buffer.DataType.BFLOAT16 -> DataType.DT_BFLOAT16
        org.nd4j.linalg.api.buffer.DataType.FLOAT -> DataType.DT_FLOAT
        org.nd4j.linalg.api.buffer.DataType.INT -> DataType.DT_INT32
        org.nd4j.linalg.api.buffer.DataType.LONG -> DataType.DT_INT64
        org.nd4j.linalg.api.buffer.DataType.BYTE -> DataType.DT_INT8
        org.nd4j.linalg.api.buffer.DataType.SHORT -> DataType.DT_INT16
        org.nd4j.linalg.api.buffer.DataType.DOUBLE -> DataType.DT_DOUBLE
        org.nd4j.linalg.api.buffer.DataType.UBYTE -> DataType.DT_UINT8
        org.nd4j.linalg.api.buffer.DataType.HALF -> DataType.DT_HALF
        org.nd4j.linalg.api.buffer.DataType.UTF8 -> DataType.DT_STRING
        else -> throw UnsupportedOperationException("Unknown TF data type: [" + dataType.name + "]")
    }
}


fun tensorflowAttributeValueTypeFor(attributeName: String, opDef: OpDef): AttributeValueType {
    val names = opDef.attrList.map { attrDef -> attrDef.name }
    if(!names.contains(attributeName) && !isTensorflowTensorName(attributeName,opDef)) {
        throw java.lang.IllegalArgumentException("Tensorflow op ${opDef.name} does not have attribute name $attributeName")
    } else if(isTensorflowTensorName(attributeName,opDef)) {
        //note we allows tensors here since sometimes input tensors in tensorflow become attributes in nd4j
        return AttributeValueType.TENSOR
    }
    val attrDef = opDef.attrList.first { attrDef -> attrDef.name == attributeName }
    return TensorflowIRAttr(attrDef, AttrValue.getDefaultInstance()).attributeValueType()
}



fun isTensorflowTensorName(name: String, opDef: OpDef): Boolean {
    return opDef.inputArgList.map {inputDef -> inputDef.name }.contains(name)
}


fun isTensorflowAttributeName(name: String, opDef: OpDef): Boolean {
    return opDef.attrList.map { attrDef -> attrDef.name }.contains(name)
}

/**
 * fun <NODE_TYPE : GeneratedMessageV3,
OP_DEF_TYPE : GeneratedMessageV3,
TENSOR_TYPE : GeneratedMessageV3,
ATTR_DEF_TYPE : GeneratedMessageV3,
ATTR_VALUE_TYPE : GeneratedMessageV3,
DATA_TYPE: ProtocolMessageEnum > initAttributes(
df: DifferentialFunction,
applied: Pair<MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor>,
mappingContext: MappingContext<NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>,
sd: SameDiff
)
 */
//fun initAttributesTensorflow()







/**
 * Get the shape from a TensorShapeProto
 *
 * @param tensorShapeProto Shape
 * @return Shape as long[]
 */
private fun shapeFromShapeProto(tensorShapeProto: TensorShapeProto): LongArray? {
    val shape = LongArray(tensorShapeProto.dimList.size)
    for (i in shape.indices) {
        shape[i] = tensorShapeProto.getDim(i).size
    }
    return shape
}

/**
 * Convert from TF proto datatype to ND4J datatype
 *
 * @param tfType TF datatype
 * @return ND4J datatype
 */
fun convertType(tfType: DataType?): org.nd4j.linalg.api.buffer.DataType {
    return when (tfType) {
        DataType.DT_DOUBLE -> org.nd4j.linalg.api.buffer.DataType.DOUBLE
        DataType.DT_FLOAT -> org.nd4j.linalg.api.buffer.DataType.FLOAT
        DataType.DT_HALF -> org.nd4j.linalg.api.buffer.DataType.HALF
        DataType.DT_BFLOAT16 -> org.nd4j.linalg.api.buffer.DataType.BFLOAT16
        DataType.DT_INT8 -> org.nd4j.linalg.api.buffer.DataType.BYTE
        DataType.DT_INT16 -> org.nd4j.linalg.api.buffer.DataType.SHORT
        DataType.DT_INT32 -> org.nd4j.linalg.api.buffer.DataType.INT
        DataType.DT_INT64 -> org.nd4j.linalg.api.buffer.DataType.LONG
        DataType.DT_UINT8 -> org.nd4j.linalg.api.buffer.DataType.UBYTE
        DataType.DT_STRING -> org.nd4j.linalg.api.buffer.DataType.UTF8
        DataType.DT_BOOL -> org.nd4j.linalg.api.buffer.DataType.BOOL
        else -> org.nd4j.linalg.api.buffer.DataType.UNKNOWN
    }
}

/**
 * @return True if the specified name represents a control dependency (starts with "^")
 */
fun isControlDep(name: String): Boolean {
    return name.startsWith("^")
}

/**
 * @return The specified name without the leading "^" character (if any) that appears for control dependencies
 */
fun stripControl(name: String): String {
    return if (name.startsWith("^")) {
        name.substring(1)
    } else name
}

/**
 * Remove the ":1" etc suffix for a variable name to get the op name
 *
 * @param varName Variable name
 * @return Variable name without any number suffix
 */
fun stripVarSuffix(varName: String): String {
    if (varName.matches(regex = Regex(".*:\\d+"))) {
        val idx = varName.lastIndexOf(':')
        return varName.substring(0, idx)
    }
    return varName
}

/**
 * Convert the tensor to an NDArray (if possible and if array is available)
 *
 * @param node Node to get NDArray from
 * @return NDArray
 */
fun getNDArrayFromTensor(node: NodeDef): INDArray? {
    //placeholder of some kind
    if (!node.attrMap.containsKey("value")) {
        return null
    }
    val tfTensor = node.getAttrOrThrow("value").tensor
    return mapTensorProto(tfTensor)
}

/**
 * Convert a TensorProto to an INDArray
 *
 * @param tfTensor Tensor proto
 * @return INDArray
 */
fun mapTensorProto(tfTensor: TensorProto): INDArray {
    val m = TFTensorMappers.newMapper(tfTensor) ?: throw RuntimeException("Not implemented datatype: " + tfTensor.dtype)
    return m.toNDArray()
}

fun attributeValueTypeForTensorflowAttribute(attributeDef: AttrDef): AttributeValueType {
    when(attributeDef.type) {
        "list(bool)" -> return AttributeValueType.LIST_BOOL
        "bool" -> return AttributeValueType.BOOL
        "string" -> return AttributeValueType.STRING
        "list(string)" -> return AttributeValueType.LIST_STRING
        "int" -> return AttributeValueType.INT
        "list(int)" -> return AttributeValueType.LIST_INT
        "float" -> return AttributeValueType.FLOAT
        "list(float)" -> return AttributeValueType.LIST_FLOAT
        "tensor" -> return AttributeValueType.TENSOR
        "list(tensor)" -> return AttributeValueType.LIST_TENSOR
        "type" -> return AttributeValueType.DATA_TYPE
    }

    return AttributeValueType.INVALID
}


