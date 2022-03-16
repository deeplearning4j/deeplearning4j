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
package org.nd4j.samediff.frameworkimport.onnx

import onnx.Onnx
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

import org.nd4j.samediff.frameworkimport.ir.*
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRAttr
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.shade.protobuf.ByteString
import java.nio.charset.Charset
import kotlin.collections.ArrayList




fun attrDefaultValue(): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
    return OnnxIRAttr(Onnx.AttributeProto.getDefaultInstance(), Onnx.AttributeProto.getDefaultInstance())
}


fun Onnx.GraphProto.nodeByName(name: String): Onnx.NodeProto {
    return this.nodeList.first { it.name == name }!!
}


fun onnxAttributeTypeFor(attributeName: String,opDef: Onnx.NodeProto): AttributeValueType {
    if(isOnnxTensorName(attributeName,opDef))
        return AttributeValueType.TENSOR
    return OnnxIRAttr(opDef.attributeList.first {
            attributeProto -> attributeProto.name == attributeName },
        Onnx.AttributeProto.getDefaultInstance()).attributeValueType()
}

fun isOnnxTensorName(name: String, opDef: Onnx.NodeProto): Boolean {
    return opDef.inputList.contains(name)
}


fun isOnnxAttributeName(name: String, opDef: Onnx.NodeProto): Boolean {
    return opDef.attributeList.map { attrDef -> attrDef.name }.contains(name)
}

fun prepareGraphForExecAndExport(graphDef: Onnx.GraphProto,outputNames: List<String>,opset: Long = 13L,ir: Long = 7): Onnx.ModelProto {
    //onnx runtime doesn't allow any outputs that aren't defined
    //already in the model, we need to dynamically modify the model at runtime
    //to allow things like intermediate results

    val modelProto = ModelProto {
        OpSetImport(OperatorSetIdProto {
            version = opset
        })

        irVersion = ir
        graph = graphDef
    }

    return modelProto
}


fun convertToOnnxDataType(dataType: DataType): Onnx.TensorProto.DataType {
    return when (dataType) {
        DataType.UINT16 -> Onnx.TensorProto.DataType.UINT16
        DataType.UINT32 ->  Onnx.TensorProto.DataType.UINT32
        DataType.UINT64 ->  Onnx.TensorProto.DataType.UINT64
        DataType.BOOL ->  Onnx.TensorProto.DataType.BOOL
        DataType.FLOAT ->  Onnx.TensorProto.DataType.FLOAT
        DataType.INT,DataType.INT32 ->  Onnx.TensorProto.DataType.INT32
        DataType.LONG,DataType.INT64 ->  Onnx.TensorProto.DataType.INT64
        DataType.BYTE,DataType.INT8 ->  Onnx.TensorProto.DataType.INT8
        DataType.SHORT,DataType.INT16 -> Onnx.TensorProto.DataType.INT16
        DataType.DOUBLE -> Onnx.TensorProto.DataType.DOUBLE
        DataType.UBYTE,DataType.UINT8 ->  Onnx.TensorProto.DataType.UINT8
        DataType.HALF,DataType.FLOAT16 ->  Onnx.TensorProto.DataType.FLOAT16
        DataType.UTF8 ->  Onnx.TensorProto.DataType.STRING
        else -> throw UnsupportedOperationException("Unknown Onnx data type: [" + dataType.name + "]")
    }
}





fun convertToOnnxTensor(inputArray: INDArray, name: String): Onnx.TensorProto {
    val dtype = convertToOnnxDataType(inputArray.dataType())
    val newBuilder = Onnx.TensorProto.newBuilder()
    newBuilder.dataType = dtype.ordinal
    newBuilder.addAllDims(inputArray.shape().toList())
    newBuilder.name = name
    when(dtype) {
        Onnx.TensorProto.DataType.STRING -> {
            return OnnxTensorProto {
                val stringList = ArrayList<String>()
                for (i in 0 until inputArray.length()) {
                    stringList.add(inputArray.getString(i))
                }

                newBuilder.addAllStringData(stringList.map { input -> ByteString.copyFrom(input.toByteArray(Charset.defaultCharset())) })
            }
        }

        Onnx.TensorProto.DataType.DOUBLE -> {
            newBuilder.addAllDoubleData(inputArray.data().asDouble().asList())
        }

        Onnx.TensorProto.DataType.FLOAT -> {
            newBuilder.addAllFloatData(inputArray.data().asFloat().asList())
        }

        Onnx.TensorProto.DataType.INT32 -> {
            newBuilder.addAllInt32Data(inputArray.data().asInt().asList())
        }

        Onnx.TensorProto.DataType.INT64 -> {
            newBuilder.addAllInt64Data(inputArray.data().asLong().asList())
        }


        else -> {
            newBuilder.rawData = ByteString.copyFrom(inputArray.data().asBytes())
        }
    }
    return newBuilder.build()

}


fun attributeValueTypeForOnnxAttribute(attributeDef: Onnx.AttributeProto) : AttributeValueType {
    when(attributeDef.type) {
        Onnx.AttributeProto.AttributeType.STRING -> return AttributeValueType.STRING
        Onnx.AttributeProto.AttributeType.STRINGS -> return AttributeValueType.LIST_STRING
        Onnx.AttributeProto.AttributeType.INT-> return AttributeValueType.INT
        Onnx.AttributeProto.AttributeType.INTS -> return AttributeValueType.LIST_INT
        Onnx.AttributeProto.AttributeType.FLOAT -> return AttributeValueType.FLOAT
        Onnx.AttributeProto.AttributeType.FLOATS -> return AttributeValueType.LIST_FLOAT
        Onnx.AttributeProto.AttributeType.TENSOR -> return AttributeValueType.TENSOR
        Onnx.AttributeProto.AttributeType.TENSORS -> return AttributeValueType.LIST_TENSOR
    }

    return AttributeValueType.INVALID
}



