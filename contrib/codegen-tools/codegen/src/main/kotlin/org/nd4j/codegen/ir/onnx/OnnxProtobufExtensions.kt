package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import onnx.OnnxOperators
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.shade.protobuf.ByteString
import java.nio.charset.Charset

fun NodeProto(block: Onnx.NodeProto.Builder.() -> Unit): Onnx.NodeProto {
    return Onnx.NodeProto.newBuilder().apply(block).build()
}

fun AttributeProto(block: Onnx.AttributeProto.Builder.() -> Unit) : Onnx.AttributeProto {
    return Onnx.AttributeProto.newBuilder().apply { block }.build()
}

fun Onnx.AttributeProto.Builder.TensorValue(inputValue: Onnx.TensorProto) {
    this.addTensors(inputValue)
}

fun Onnx.AttributeProto.Builder.StringValue(inputValue: String) {
    this.addStrings(ByteString.copyFrom(inputValue.toByteArray(Charset.defaultCharset())))
}

fun Onnx.NodeProto.Builder.Attribute(attribute: Onnx.AttributeProto) {
    this.addAttribute(attribute)
}

fun Onnx.NodeProto.Builder.Input(inputName: String) {
    this.addInput(inputName)
}

fun Onnx.NodeProto.Builder.Output(inputName: String) {
    this.addOutput(inputName)
}

fun Onnx.GraphProto.Builder.Initializer(tensor: Onnx.TensorProto) {
    this.addInitializer(tensor)
}

fun OperatorSetIdProto(block: Onnx.OperatorSetIdProto.Builder.() -> Unit): Onnx.OperatorSetIdProto {
    return Onnx.OperatorSetIdProto.newBuilder().apply(block).build()
}

fun OperatorSetProto(block: OnnxOperators.OperatorSetProto.Builder.() -> Unit): OnnxOperators.OperatorSetProto {
    return OnnxOperators.OperatorSetProto.newBuilder().apply { block }.build()
}

fun Onnx.ModelProto.Builder.OpSetImport(opSetImport: Onnx.OperatorSetIdProto) {
    this.addOpsetImport(opSetImport)
}

fun ModelProto(block: Onnx.ModelProto.Builder.() -> Unit): Onnx.ModelProto {
    return Onnx.ModelProto.newBuilder()
        .apply(block).build()
}

fun TensorDefinition(block: Onnx.TypeProto.Tensor.Builder.() -> Unit) : Onnx.TypeProto.Tensor {
    return Onnx.TypeProto.Tensor.newBuilder().apply(block).build()
}

fun TypeProto(block: Onnx.TypeProto.Builder.() -> Unit): Onnx.TypeProto {
    return Onnx.TypeProto.newBuilder().apply(block).build()
}

fun GraphProto(block: Onnx.GraphProto.Builder.() -> Unit): Onnx.GraphProto {
    return Onnx.GraphProto.newBuilder()
        .apply(block).build()
}

fun OnnxDim(block: Onnx.TensorShapeProto.Dimension.Builder.() -> Unit): Onnx.TensorShapeProto.Dimension {
    return Onnx.TensorShapeProto.Dimension.newBuilder().apply(block).build()
}


fun Onnx.TensorShapeProto.Builder.OnnxShape(dims: List<Long>) {
    this.addAllDim(dims.map { inputDim -> OnnxDim {
        dimValue = inputDim
    } })
}


fun OnnxShapeProto(block: Onnx.TensorShapeProto.Builder.() -> Unit): Onnx.TensorShapeProto {
    return Onnx.TensorShapeProto.newBuilder().apply(block).build()
}

fun ValueInfoProto(block: Onnx.ValueInfoProto.Builder.() -> Unit): Onnx.ValueInfoProto {
    return Onnx.ValueInfoProto.newBuilder()
        .apply(block).build()
}

fun Onnx.GraphProto.Builder.Output(input: Onnx.ValueInfoProto) {
    this.addOutput(input)
}


fun Onnx.GraphProto.Builder.Input(input: Onnx.ValueInfoProto) {
    this.addInput(input)
}

fun Onnx.GraphProto.Builder.Node(inputNode: Onnx.NodeProto) {
    this.addNode(inputNode)
}

fun Onnx.AttributeProto.Builder.Tensor(inputTensor: Onnx.TensorProto) {
    this.addTensors(inputTensor)
}

fun OnnxTensorProto(block: Onnx.TensorProto.Builder.() -> Unit): Onnx.TensorProto {
    return Onnx.TensorProto.newBuilder().apply { block }.build()
}

fun Onnx.TensorProto.Builder.OnnxDataType(value: Onnx.TensorProto.DataType) {
    this.dataType = value
}

fun Onnx.TensorProto.Builder.OnnxRawData(byteArray: ByteArray) {
    this.rawData = ByteString.copyFrom(byteArray)
}

fun Onnx.TensorProto.Builder.Shape(shape: List<Long>) {
    this.dimsList.clear()
    this.dimsList.addAll(shape)
}

fun Onnx.TensorProto.Builder.LongData(longData: List<Long>) {
    this.addAllInt64Data(longData)
}

fun Onnx.TensorProto.Builder.IntData(intData: List<Int>) {
    this.addAllInt32Data(intData)
}

fun Onnx.TensorProto.Builder.FloatData(floatData: List<Float>) {
    this.addAllFloatData(floatData)
}


fun Onnx.TensorProto.Builder.DoubleData(doubleData: List<Double>) {
    this.addAllDoubleData(doubleData)
}

fun Onnx.TensorProto.Builder.StringData(stringData: List<String>) {
    this.addAllStringData(stringData.map { ByteString.copyFrom(it.toByteArray(Charset.defaultCharset())) })
}

fun Onnx.TensorProto.Builder.BoolData(boolData: List<Boolean>) {
    this.addAllInt32Data(boolData.map { input -> if(input) 1 else 0  })
}


fun createValueInfoFromTensor(arr: INDArray,valueInfoName: String,useShape: Boolean = true): Onnx.ValueInfoProto {
    if(useShape)
        return ValueInfoProto {
            name = valueInfoName
            type = TypeProto {
                tensorType =  TensorDefinition {
                    elemType = convertToOnnxDataType(arr.dataType())
                    shape = OnnxShapeProto {
                        OnnxShape(arr.shape().toList())
                    }
                }

            }
        }
    else
        return ValueInfoProto {
            name = valueInfoName
            type = TypeProto {
                tensorType =  TensorDefinition {
                    elemType = convertToOnnxDataType(arr.dataType())
                }

            }
        }
}