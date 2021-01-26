package org.nd4j.codegen.ir.tensorflow

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

