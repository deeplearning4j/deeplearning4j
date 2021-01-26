package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import onnx.OnnxMl
import org.apache.commons.io.FileUtils
import org.nd4j.codegen.ir.*
import org.nd4j.codegen.ir.tensorflow.AttrValue
import org.nd4j.codegen.ir.tensorflow.TensorflowIRTensor
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray

import kotlin.collections.HashMap
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.TensorProto
import java.io.File
import java.lang.IllegalArgumentException
import java.nio.charset.Charset
import java.util.*
import kotlin.collections.ArrayList
import kotlin.math.min

fun loadOnnxOps(): List<Onnx.NodeProto> {
    val graphProto = Onnx.GraphProto.parseFrom(ClassPathResource("onnx-op-defs.pb").inputStream)
    return graphProto.nodeList
}

val onnxops = loadOnnxOps()

class OnnxIRTensor(input: Onnx.TensorProto): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return tensor.dimsList
    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(),'c').asList()
    }

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[tensor.dataType.ordinal])
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensor.dimsCount) {
            builder.addDims(tensor.getDims(i))
        }

        when(tensor.dataType) {
            Onnx.TensorProto.DataType.UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            Onnx.TensorProto.DataType.UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            Onnx.TensorProto.DataType.UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            Onnx.TensorProto.DataType.FLOAT16 -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            Onnx.TensorProto.DataType.STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            Onnx.TensorProto.DataType.FLOAT -> builder.dataType  = TensorNamespace.DataType.FLOAT.ordinal
            Onnx.TensorProto.DataType.DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            Onnx.TensorProto.DataType.BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            Onnx.TensorProto.DataType.INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            Onnx.TensorProto.DataType.INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            Onnx.TensorProto.DataType.INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            Onnx.TensorProto.DataType.COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            Onnx.TensorProto.DataType.COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            Onnx.TensorProto.DataType.UNDEFINED,Onnx.TensorProto.DataType.UNRECOGNIZED ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleDataList != null && tensor.doubleDataCount > 0) {
            builder.addAllDoubleData(tensor.doubleDataList)
        }

        if(tensor.stringDataList != null && tensor.stringDataCount > 0) {
            builder.addAllStringData(tensor.stringDataList)
        }

        if(tensor.floatDataList != null && tensor.floatDataCount > 0) {
            builder.addAllFloatData(tensor.floatDataList)
        }

        if(tensor.int32DataList != null && tensor.int32DataCount > 0) {
            builder.addAllInt32Data(tensor.int32DataList)
        }

        if(tensor.int64DataCount != null && tensor.int64DataCount > 0) {
            builder.addAllInt64Data(tensor.int64DataList)
        }

        if(tensor.uint64DataList != null && tensor.uint64DataCount > 0) {
            builder.addAllInt64Data(tensor.uint64DataList)
        }

        if(tensor.rawData != null) {
            builder.rawData = tensor.rawData
        }

        builder.dataType = tensor.dataType.ordinal

        return builder.build()
    }

    override fun rawValue(): Onnx.TensorProto {
        return tensor
    }

    override fun toNd4jNDArray(): INDArray {
        return ndarrayFromNameSpaceTensor(toArgTensor())
    }


}

class OnnxIRDataType(inputDataType: Onnx.TensorProto.DataType): IRDataType<Onnx.TensorProto.DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: Onnx.TensorProto.DataType): IRDataTypeValue {
        when(input) {
            Onnx.TensorProto.DataType.UINT64 ->  return IRDataTypeValue.DT_UINT64
            Onnx.TensorProto.DataType.UINT32 ->  return IRDataTypeValue.DT_UINT32
            Onnx.TensorProto.DataType.UINT16 ->  return IRDataTypeValue.DT_UINT16
            Onnx.TensorProto.DataType.FLOAT16 -> return IRDataTypeValue.DT_HALF
            Onnx.TensorProto.DataType.STRING -> return IRDataTypeValue.DT_STRING
            Onnx.TensorProto.DataType.FLOAT ->  return IRDataTypeValue.DT_FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return IRDataTypeValue.DT_DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return IRDataTypeValue.DT_BOOL
            Onnx.TensorProto.DataType.INT64 -> return IRDataTypeValue.DT_INT64
            Onnx.TensorProto.DataType.INT32 ->  return IRDataTypeValue.DT_INT32
            Onnx.TensorProto.DataType.INT16 -> return IRDataTypeValue.DT_INT16
            Onnx.TensorProto.DataType.COMPLEX64 ->  return IRDataTypeValue.DT_COMPLEX64
            Onnx.TensorProto.DataType.COMPLEX128 ->  return IRDataTypeValue.DT_COMPLEX128
            Onnx.TensorProto.DataType.UNDEFINED,Onnx.TensorProto.DataType.UNRECOGNIZED ->  TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }

        return IRDataTypeValue.DT_INVALID
    }

    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): Onnx.TensorProto.DataType {
        return this.dataType
    }

    override fun nd4jDataType(): DataType {
        when(this.dataType) {
            Onnx.TensorProto.DataType.UINT64 ->  return  return DataType.INT64
            Onnx.TensorProto.DataType.UINT32 ->  return return DataType.INT32
            Onnx.TensorProto.DataType.UINT16 ->  return return DataType.INT16
            Onnx.TensorProto.DataType.FLOAT16 -> return   return DataType.FLOAT16
            Onnx.TensorProto.DataType.STRING -> return  return DataType.UTF8
            Onnx.TensorProto.DataType.FLOAT ->  return  return DataType.FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return  return DataType.DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return  return DataType.BOOL
            Onnx.TensorProto.DataType.INT64 -> return  return DataType.INT64
            Onnx.TensorProto.DataType.INT32 ->  return  return DataType.INT32
            Onnx.TensorProto.DataType.INT16 -> return  return DataType.INT16

        }

        return  return DataType.UNKNOWN

    }

    override fun nameSpaceDataType(): TensorNamespace.DataType {
        when(this.dataType) {
            Onnx.TensorProto.DataType.UINT64 ->  return  return TensorNamespace.DataType.INT64
            Onnx.TensorProto.DataType.UINT32 ->  return return TensorNamespace.DataType.INT32
            Onnx.TensorProto.DataType.UINT16 ->  return return TensorNamespace.DataType.INT16
            Onnx.TensorProto.DataType.FLOAT16 -> return   return TensorNamespace.DataType.FLOAT16
            Onnx.TensorProto.DataType.STRING -> return  return TensorNamespace.DataType.STRING
            Onnx.TensorProto.DataType.FLOAT ->  return  TensorNamespace.DataType.FLOAT
            Onnx.TensorProto.DataType.DOUBLE -> return  TensorNamespace.DataType.DOUBLE
            Onnx.TensorProto.DataType.BOOL -> return  return TensorNamespace.DataType.BOOL
            Onnx.TensorProto.DataType.INT64 -> return  return TensorNamespace.DataType.INT64
            Onnx.TensorProto.DataType.INT32 ->  return  return TensorNamespace.DataType.INT32
            Onnx.TensorProto.DataType.INT16 -> return  return TensorNamespace.DataType.INT16

        }

        return  TensorNamespace.DataType.UNDEFINED
    }

}

fun attrDefaultValue(): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
    return OnnxIRAttr(Onnx.AttributeProto.getDefaultInstance(), Onnx.AttributeProto.getDefaultInstance())
}

class OnnxIRAttr(inputAttributeDef: Onnx.AttributeProto, inputAttributeValue: Onnx.AttributeProto):
    IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.floatsList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        return attributeValue.intsList
    }

    override fun boolValue(): Boolean {
        return attributeValue.i > 0
    }

    override fun listBoolValue(): List<Boolean> {
        TODO("Implement")
    }

    override fun attributeValueType(): AttributeValueType {
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



    override fun internalAttributeDef(): Onnx.AttributeProto {
        return attributeDef
    }

    override fun internalAttributeValue(): Onnx.AttributeProto {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<Onnx.TensorProto,Onnx.TensorProto.DataType>> {
        return attributeValue.tensorsList.map {
                input -> OnnxIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRTensor(attributeValue.t)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.stringsList.map { it.toStringUtf8() }
    }

    override fun dataTataTypeValue(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRDataType(Onnx.TensorProto.DataType.values()[attributeDef.t.dataType.ordinal])
    }

}

class OnnxIRArgDef(input: Onnx.NodeProto): IRArgDef<Onnx.NodeProto,Onnx.TensorProto.DataType> {
    private val argDefValue = input

    override fun dataType(): IRDataType<Onnx.TensorProto.DataType> {
        return OnnxIRArgDef(argDefValue).dataType()
    }

    override fun name(): String {
        return argDefValue.name
    }

    override fun description(): String {
        return argDefValue.docString
    }

    override fun internalValue(): Onnx.NodeProto {
        return argDefValue
    }

    override fun indexOf(): Integer {
        TODO("Not yet implemented")
    }

}

class OnnxIROp(input: Onnx.NodeProto): IROpDef<Onnx.GraphProto,Onnx.NodeProto, Onnx.TensorProto, Onnx.NodeProto,Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto> {

    val opDef = input

    override fun attributes(): List<IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>> {
        return opDef.attributeList.map {
            OnnxIRAttr(it, Onnx.AttributeProto.getDefaultInstance())
        }
    }

    override fun opName(): String {
        return opDef.name
    }

    override fun internalValue(): Onnx.NodeProto {
        return opDef
    }

    override fun inputArgs(): List<IRArgDef<Onnx.NodeProto, Onnx.TensorProto.DataType>> {
        return opDef.inputList.map {
            OnnxIRArgDef(opDef)
        }
    }

    override fun outputArgs(): List<IRArgDef<Onnx.NodeProto,Onnx.TensorProto.DataType>> {
        return opDef.outputList.map {
            OnnxIRArgDef(opDef)
        }
    }

}

class OnnxIRNode(inputNode: Onnx.NodeProto, inputOpDef: Onnx.NodeProto): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {

    private val nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attributeList)
    private val attrMap: Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>> =
        initAttrMapFromNode(inputNode)
    private val mappingProcess: MappingProcess<Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>
    init {
        mappingProcess = onnxOpRegistry.lookupOpMappingProcess(inputNode.opType)
    }

    private fun attrDefsByName(input: List<Onnx.AttributeProto>): Map<String,Onnx.AttributeProto> {
        val ret = HashMap<String,Onnx.AttributeProto>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: Onnx.NodeProto): Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        val ret = HashMap<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto,Onnx.TensorProto.DataType>>()
        input.attributeList.forEach {
            ret[it.name] = OnnxIRAttr(it,it)

        }
        return ret
    }

    override fun opName(): String {
        return nodeDef.opType
    }

    override fun nodeName(): String {
        return nodeDef.name
    }

    override fun inputAt(index: Int): String {
        if(mappingProcess.indexOverrides().containsKey(index))
            return nodeDef.getInput(mappingProcess.indexOverrides()[index]!!)
        return nodeDef.getInput(index)
    }

    override fun outputAt(index: Int): String {
        return opDef.getOutput(index)
    }



    override fun hasAttribute(inputName: String): Boolean {
        return nodeDef.attributeList.filter { it.name == inputName }.size > 0
    }

    override fun attributeMap(): Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<Onnx.TensorProto>): List<IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return inputData.map { OnnxIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<Onnx.TensorProto>): List<IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        return inputValues.map { OnnxIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return attrMap.getOrDefault(inputName, attrDefaultValue())
    }

    override fun internalValue(): Onnx.NodeProto {
        return nodeDef
    }

    override fun numInputs(): Int {
        return nodeDef.inputCount
    }

    override fun numOutputs(): Int {
        return nodeDef.outputCount
    }

    override fun inputs(): List<String> {
        return nodeDef.inputList
    }

    override fun outputs(): List<String> {
        return nodeDef.outputList
    }

    override fun numInputsForListOfTensors(name: String): Int {
        return nodeDef.inputCount
    }

    override fun inputNamesForListOfInputValues(inputListName: String): List<String> {
        return nodeDef.inputList
    }

    override fun computeAdjustedOffsetForInput(
        nd4jName: String,
        inputFrameworkName: String,
        tensorInputMappings: Map<String, String>
    ): Int {
        //onnx doesn't have lists of values like this
        return lookupIndexForArgDescriptor(
            argDescriptorName = nd4jName,
            opDescriptorName = this.opName(),
            argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
        )
    }

    override fun nd4jInputs(tensorMappings: Map<String, String>): List<String> {
        return nodeDef.inputList
    }

}


fun Onnx.GraphProto.nodeByName(name: String): Onnx.NodeProto {
    return this.nodeList.first { it.name == name }!!
}


class OnnxIRGraphRunner(graphDef: OnnxIRGraph,inputNames: List<String>,outputNames: List<String>): IRGraphRunner<
        Onnx.GraphProto,
        Onnx.NodeProto,
        Onnx.NodeProto,
        Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType> {
    val graphDef = graphDef
    val inputNames = inputNames
    val outputNames = outputNames
    val graphRunner: OnnxRuntimeRunner

    init {
        val uuid = UUID.randomUUID().toString()
        val tempFile = File("tempFile-$uuid.proto")

        val modelProto = ModelProto {
            OpSetImport(OperatorSetIdProto {
                version = 12
            })

            irVersion = 7
            graph = graph().internalValue()
        }

        FileUtils.writeByteArrayToFile(tempFile,modelProto.toByteArray())
        graphRunner = OnnxRuntimeRunner.builder()
            .modelUri(tempFile.absolutePath)
            .inputs(inputNames)
            .outputs(outputNames)
            .build()
        tempFile.deleteOnExit()
    }

    override fun graph(): IRGraph<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        return graphDef
    }

    override fun run(inputs: Map<String, INDArray>): Map<String, INDArray> {
        return graphRunner.exec(inputs)
    }

}


class OnnxIRGraph(graphDef: Onnx.GraphProto): IRGraph<
        Onnx.GraphProto,Onnx.NodeProto,
        Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,
        Onnx.TensorProto.DataType> {

    val graphDef = graphDef
    val opList = graphDef.nodeList
    var cachedNodeList = nodeList()
    override fun nodeByName(input: String): Onnx.NodeProto {
        return cachedNodeList.first { inputNode -> inputNode.nodeName() == input }.internalValue()
    }

    override fun nodeList(): List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>> {
        val ret2 = ArrayList<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>>()
        //add all inputs, outputs, initializers together as "nodes" similar to TF
        val identityOp = onnxops.first { op -> op.name == "Constant" }
        //for model import purposes, add identity ops as dummies similar to how tensorflow does placeholders/constants
        graphDef.inputList.forEach { input ->
            //note: this is not a real op name in onnx, this is purely for flagging for import to grab the node from the initializer
            //add dummy values for placeholders
            val nodeToAdd = NodeProto {
                opType = "Constant"
                name = input.name
                Attribute(Onnx.AttributeProto.newBuilder().setName("value").
                addTensors(Onnx.TensorProto.getDefaultInstance()).build())
            }

            ret2.add(OnnxIRNode(nodeToAdd,identityOp))
        }

        graphDef.nodeList.forEach {
            val opDefOrNull = onnxops.firstOrNull { opDef -> opDef.name == it.opType  }
            if(opDefOrNull == null) {
                throw IllegalArgumentException("Op def name ${it.opType} not found!")
            }

            ret2.add(OnnxIRNode(it,opDefOrNull!!))
        }

        //create dummy nodes by inferring which nodes have outputs
        //setup identity nodes that reflect the output to automatically
        //map index outputs to nodes that actually have outputs
        val outputNames = graphDef.outputList.map { input -> input.name }.toSet()
        val outputNodes = ArrayList<Onnx.NodeProto>()
        graphDef.nodeList.forEach { nodeProto ->
            val outputList = nodeProto.outputList.map { input -> input.toString() }.toSet()
            val containsAny = outputNames.intersect(outputList)
            if(containsAny.isNotEmpty()) {
                outputNodes.add(nodeProto)
            }
        }

        outputNodes.forEach { nodeProto ->
            nodeProto.outputList.forEachIndexed { index, outputName ->
                val indexOfOutput = if(index < 1) "" else ":$index"
                if(!ret2.map { node -> node.nodeName() }.contains(outputName)) {
                    val nodeToAdd = NodeProto {
                        opType = "Identity"
                        name = outputName
                        Input("${nodeProto.name}$indexOfOutput")
                    }

                    ret2.add(OnnxIRNode(nodeToAdd, identityOp))
                }
            }

        }



        graphDef.initializerList.forEach { initializer ->
            //note: this is not a real op name in onnx, this is purely for flagging for import to grab the node from the initializer
            val nodeToAdd = NodeProto {
                opType = "Constant"
                name = initializer.name
                Attribute(Onnx.AttributeProto.newBuilder().setName("value").
                addTensors(Onnx.TensorProto.getDefaultInstance()).build())
            }

            ret2.add(OnnxIRNode(nodeToAdd,identityOp))
        }

        return ret2
    }


    fun graphDef(): Onnx.GraphProto {
        return graphDef
    }

    override fun internalValue(): Onnx.GraphProto {
        return graphDef
    }



    override fun createMappingContext(
        opDef: Onnx.NodeProto,
        node: Onnx.NodeProto,
        dynamicVariables: Map<String, Onnx.TensorProto>
    ): MappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        return OnnxMappingContext(opDef = opDef,node =  node,graph = this,dynamicVariables = dynamicVariables)
    }

    override fun frameworkName(): String {
        return "onnx"
    }

    override fun nd4jNameForInternalOpName(name: String): String {
        return onnxOpRegistry.lookupOpMappingProcess(name).opName()
    }

    override fun isConstantOpName(name: String): Boolean {
        return name == "Constant" || name == "Placeholder"
    }

    override fun isConstant(opName: String): Boolean {
        return opName == "Constant"
    }

    override fun isPlaceHolder(opName: String): Boolean {
        return opName == "Placeholder"
    }

    override fun shapeOfInput(varName: String): LongArray? {
        val firstOrNull = graphDef.initializerList.firstOrNull { inputNode -> inputNode.name == varName }
        if(firstOrNull != null)
            return firstOrNull.dimsList.toLongArray()
        return null
    }

    override fun dataTypeForVariable(varName: String): IRDataType<Onnx.TensorProto.DataType> {
        val firstOrNull = graphDef.initializerList.firstOrNull {
                inputNode -> inputNode.name == varName }
        val input = graphDef.inputList.firstOrNull { input2 ->
            input2.name == varName
        }
        if(firstOrNull != null)
            return OnnxIRDataType(Onnx.TensorProto.DataType.values()[firstOrNull!!.dataType.ordinal])
        else if(input != null)
            return OnnxIRDataType(input.type.tensorType.elemType)
        else
            return OnnxIRDataType(Onnx.TensorProto.DataType.UNDEFINED)
    }

    override fun importInfoForEachNode(dynamicVariables: Map<String, Onnx.TensorProto>): Map<String, Pair<MappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>, OpNamespace.OpDescriptor>> {
        return importInfoForEachNodeInGraph(graph = this,dynamicVariables = dynamicVariables)
    }

    override fun nodeIsPlaceHolder(nodeName: String): Boolean {
        return graphDef.inputList.map { input -> input.name }.contains(nodeName)
    }
}


class OnnxMappingContext(opDef: Onnx.NodeProto, node: Onnx.NodeProto, graph:
IRGraph< Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto,
        Onnx.AttributeProto,
        Onnx.AttributeProto, Onnx.TensorProto.DataType>,dynamicVariables: Map<String,Onnx.TensorProto>) :
    AbstractMappingContext< Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto,
            Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>(opDef, node, graph,dynamicVariables) {

    override fun attrDef(name: String): Onnx.AttributeProto {
        val ret = opDef().attributeList.firstOrNull { it.name == name }
        return ret!!
    }

    override fun irAttributeValueForNode(valueName: String): IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType> {
        val attrDef = attrDef(valueName)
        var attrValue = node.attributeList.firstOrNull { it.name == valueName }
        if(attrValue == null && attrDef.name == "value" && opDef.opType == "Constant")
        //allow dummy values
            attrValue = Onnx.AttributeProto.newBuilder().setName("value").addTensors(Onnx.TensorProto.getDefaultInstance())
                .build()
        else if(attrValue == null)
            throw IllegalArgumentException("Unable to resolve attribute for name $valueName for node ${nodeName()} for op type ${opName()}")
        return OnnxIRAttr(inputAttributeDef = attrDef,inputAttributeValue = attrValue!!)

    }

    override fun tensorInputFor(name: String): IRTensor<Onnx.TensorProto,Onnx.TensorProto.DataType> {
        var foundIndex = -1
        opDef.inputList.forEachIndexed {
                index,argDef -> if(argDef == name) foundIndex = index
        }

        return tensorInputFromInputFrameworkName(name)
    }

    override fun opName(): String {
        return opDef.opType
    }

    override fun nodeName(): String {
        return opDef.name
    }

    override fun nd4jDataTypeFor(input: IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType>): DataType {
        return input.dataType().nd4jDataType()
    }

    override fun createIRTensorFromNDArray(ndarray: INDArray): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return OnnxIRTensor(convertToOnnxTensor(ndarray,"tensor"))
    }

    override fun tensorAttributeFor(name: String): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return irAttributeValueForNode(name).tensorValue()
    }

    override fun irNode(): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        return OnnxIRNode(node, onnxops.first { input -> input.name == node.opType })
    }

    override fun tensorInputFromInputFrameworkName(name: String): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        val castedGraph = graph as OnnxIRGraph
        val graphDef = castedGraph.graphDef()
        var foundIndex = opDef.inputList.map { input -> input.toString() }.indexOf(name)



        if(foundIndex < 0) {
            throw java.lang.IllegalArgumentException("Node with name ${nodeName()} for opdef with name ${opDef.name} did not contain a tensor with name ${name}")
        }

        /**
         * Use op definition name as 1 unified reference name in rules for static purposes, but
         * look up via index for specific node mappings.
         *
         * This is equivalent to the tf input position attribute value in the previous tensorflow import.
         */
        val graphNode = if(node.opType == "Constant") name else node.getInput(foundIndex)
        val attemptedTensor = graphDef.initializerList.firstOrNull { it.name == graphNode }

        //no value to be found on placeholder, return default instance
        //if no value exists it's an output from another node
        if(attemptedTensor == null) {
            println("Value for node $graphNode is not a constant! This method only works for constants. Consider replacing the Placeholder node with a Constant node. This will return an empty tensor.")
            if(!dynamicVariables.containsKey(graphNode))
                return OnnxIRTensor(Onnx.TensorProto.getDefaultInstance())
            else {
                val toConvert = dynamicVariables[graphNode]!!
                return OnnxIRTensor(toConvert)
            }
        }

        //value nodes are the values of attributes that are input nodes in a frozen graph
        if(attemptedTensor == null) {
            throw IllegalArgumentException("Name $name not found in initializer list.")
        }
        return OnnxIRTensor(attemptedTensor!!)
    }

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


fun convertToOnnxDataType(dataType: DataType): Onnx.TensorProto.DataType {
    return when (dataType) {
        DataType.UINT16 -> Onnx.TensorProto.DataType.UINT16
        DataType.UINT32 ->  Onnx.TensorProto.DataType.UINT32
        DataType.UINT64 ->  Onnx.TensorProto.DataType.UINT64
        DataType.BOOL ->  Onnx.TensorProto.DataType.BOOL
        DataType.FLOAT ->  Onnx.TensorProto.DataType.FLOAT
        DataType.INT ->  Onnx.TensorProto.DataType.INT32
        DataType.LONG ->  Onnx.TensorProto.DataType.INT64
        DataType.BYTE ->  Onnx.TensorProto.DataType.INT8
        DataType.SHORT -> Onnx.TensorProto.DataType.INT16
        DataType.DOUBLE -> Onnx.TensorProto.DataType.DOUBLE
        DataType.UBYTE ->  Onnx.TensorProto.DataType.UINT8
        DataType.HALF ->  Onnx.TensorProto.DataType.FLOAT16
        DataType.UTF8 ->  Onnx.TensorProto.DataType.STRING
        else -> throw UnsupportedOperationException("Unknown Onnx data type: [" + dataType.name + "]")
    }
}


fun convertToOnnxTensor(inputArray: INDArray, name: String): Onnx.TensorProto {
    val dtype = convertToOnnxDataType(inputArray.dataType())
    val newBuilder = Onnx.TensorProto.newBuilder()
    newBuilder.dataType = dtype
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



