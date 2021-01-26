package org.nd4j.codegen.ir.tensorflow

import org.apache.commons.io.IOUtils
import org.nd4j.codegen.ir.*
import org.nd4j.common.io.ClassPathResource
import org.nd4j.imports.graphmapper.tf.tensors.TFTensorMappers
import org.nd4j.ir.OpNamespace
import org.nd4j.ir.TensorNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.shade.protobuf.TextFormat
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner
import org.tensorflow.framework.*
import org.tensorflow.framework.OpDef.AttrDef
import java.nio.charset.Charset
import kotlin.collections.HashMap
import kotlin.math.min

fun loadTensorflowOps(): OpList {
    val string = IOUtils.toString(ClassPathResource("ops.proto").inputStream, Charset.defaultCharset())
    val tfListBuilder = OpList.newBuilder()
    TextFormat.merge(string, tfListBuilder)
    return tfListBuilder.build()
}

val tensorflowOps = loadTensorflowOps()




class TensorflowIRTensor(input: TensorProto): IRTensor<TensorProto, DataType> {

    val tensor = input


    override fun shape(): List<Long> {
        return  tensor.tensorShape.dimList.map { it.size }

    }

    override fun stride(): List<Long> {
        return Nd4j.getStrides(shape().toTypedArray().toLongArray(), 'c').asList()
    }

    override fun dataType(): IRDataType<DataType> {
        return TensorflowIRDataType(tensor.dtype)
    }

    override fun toArgTensor(): TensorNamespace.TensorProto {
        val builder = TensorNamespace.TensorProto.newBuilder()
            .setDataLocation(TensorNamespace.TensorProto.DataLocation.DEFAULT)

        for(i in 0 until tensor.tensorShape.dimCount) {
            builder.addDims(tensor.tensorShape.getDim(i).size)
        }

        when(tensor.dtype) {
            DataType.DT_UINT64 -> builder.dataType = TensorNamespace.DataType.UINT64.ordinal
            DataType.DT_UINT32 -> builder.dataType = TensorNamespace.DataType.UINT32.ordinal
            DataType.DT_UINT16 -> builder.dataType = TensorNamespace.DataType.UINT16.ordinal
            DataType.DT_HALF -> builder.dataType = TensorNamespace.DataType.FLOAT16.ordinal
            DataType.DT_STRING -> builder.dataType = TensorNamespace.DataType.STRING.ordinal
            DataType.DT_FLOAT -> builder.dataType = TensorNamespace.DataType.FLOAT.ordinal
            DataType.DT_DOUBLE -> builder.dataType = TensorNamespace.DataType.DOUBLE.ordinal
            DataType.DT_BOOL -> builder.dataType = TensorNamespace.DataType.BOOL.ordinal
            DataType.DT_INT64 -> builder.dataType = TensorNamespace.DataType.INT64.ordinal
            DataType.DT_INT32 -> builder.dataType = TensorNamespace.DataType.INT32.ordinal
            DataType.DT_INT16 -> builder.dataType = TensorNamespace.DataType.INT16.ordinal
            DataType.DT_BFLOAT16 -> builder.dataType = TensorNamespace.DataType.BFLOAT16.ordinal
            DataType.DT_COMPLEX64 -> builder.dataType = TensorNamespace.DataType.COMPLEX64.ordinal
            DataType.DT_COMPLEX128 -> builder.dataType = TensorNamespace.DataType.COMPLEX128.ordinal
            DataType.UNRECOGNIZED -> builder.dataType = TensorNamespace.DataType.UNRECOGNIZED.ordinal

        }


        if(tensor.doubleValList != null && tensor.doubleValCount > 0) {
            builder.addAllDoubleData(tensor.doubleValList)
        }

        if(tensor.stringValList != null && tensor.stringValCount > 0) {
            builder.addAllStringData(tensor.stringValList)
        }

        if(tensor.floatValList != null && tensor.floatValCount > 0) {
            builder.addAllFloatData(tensor.floatValList)
        }

        if(tensor.intValList != null && tensor.intValCount > 0) {
            builder.addAllInt32Data(tensor.intValList)
        }

        if(tensor.uint64ValList != null && tensor.uint64ValCount > 0) {
            builder.addAllInt64Data(tensor.uint64ValList)
        }

        if(tensor.int64ValList != null && tensor.int64ValCount > 0) {
            builder.addAllInt64Data(tensor.int64ValList)
        }

        if(tensor.tensorContent != null) {
            builder.rawData = tensor.tensorContent
        }


        return builder.build()
    }

    override fun rawValue(): TensorProto {
        return tensor
    }

    override fun toNd4jNDArray(): INDArray {
        if(tensor.dtype == DataType.UNRECOGNIZED || tensor.dtype == DataType.DT_INVALID)
            return Nd4j.empty()
        return TFTensorMappers.newMapper(tensor).toNDArray()
    }
}

class TensorflowIRDataType(inputDataType: DataType): IRDataType<DataType> {
    val dataType = inputDataType

    override fun convertToDataType(input: DataType): IRDataTypeValue {
        when(input) {
            DataType.DT_BOOL,DataType.DT_BOOL_REF -> return IRDataTypeValue.DT_BOOL
            DataType.DT_BFLOAT16,DataType.DT_BFLOAT16_REF -> return IRDataTypeValue.DT_BFLOAT16
            DataType.DT_COMPLEX128,DataType.DT_COMPLEX128_REF -> return IRDataTypeValue.DT_COMPLEX128
            DataType.DT_COMPLEX64,DataType.DT_COMPLEX64_REF -> return IRDataTypeValue.DT_COMPLEX64
            DataType.DT_DOUBLE, DataType.DT_DOUBLE_REF -> return IRDataTypeValue.DT_DOUBLE
            DataType.DT_FLOAT,DataType.DT_FLOAT_REF -> return IRDataTypeValue.DT_FLOAT
            DataType.DT_HALF,DataType.DT_HALF_REF -> return IRDataTypeValue.DT_HALF
            DataType.DT_INT16,DataType.DT_INT16_REF -> return IRDataTypeValue.DT_INT16
            DataType.DT_INT32,DataType.DT_INT32_REF -> return IRDataTypeValue.DT_INT32
            DataType.DT_INT64, DataType.DT_INT64_REF -> return IRDataTypeValue.DT_INT64
            DataType.DT_QINT8,DataType.DT_QINT8_REF -> return IRDataTypeValue.DT_QINT8
            DataType.DT_QINT16, DataType.DT_QINT16_REF -> return IRDataTypeValue.DT_QINT16
            DataType.DT_QINT32, DataType.DT_QINT32_REF -> return IRDataTypeValue.DT_QINT32
            DataType.DT_STRING,DataType.DT_STRING_REF -> return IRDataTypeValue.DT_STRING
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return IRDataTypeValue.DT_UINT16
            DataType.DT_UINT32,DataType.DT_UINT32_REF -> return IRDataTypeValue.DT_UINT32
            DataType.DT_UINT64,DataType.DT_UINT64_REF -> return IRDataTypeValue.DT_UINT64

        }

        return IRDataTypeValue.DT_INVALID
    }



    override fun dataType(): IRDataTypeValue {
        return convertToDataType(this.dataType)
    }

    override fun internalValue(): DataType {
        return this.dataType
    }

    override fun nd4jDataType(): org.nd4j.linalg.api.buffer.DataType {
        when(this.dataType) {
            DataType.DT_BOOL,DataType.DT_BOOL_REF -> return org.nd4j.linalg.api.buffer.DataType.BOOL
            DataType.DT_FLOAT,DataType.DT_FLOAT_REF -> return org.nd4j.linalg.api.buffer.DataType.FLOAT
            DataType.DT_STRING, DataType.DT_STRING_REF -> return org.nd4j.linalg.api.buffer.DataType.UTF8
            DataType.DT_BFLOAT16,DataType.DT_BFLOAT16_REF -> return org.nd4j.linalg.api.buffer.DataType.BFLOAT16
            DataType.DT_INT64,DataType.DT_INT64_REF -> return org.nd4j.linalg.api.buffer.DataType.INT64
            DataType.DT_HALF,DataType.DT_HALF_REF -> return org.nd4j.linalg.api.buffer.DataType.FLOAT16
            DataType.DT_INT16,DataType.DT_INT16_REF -> return org.nd4j.linalg.api.buffer.DataType.INT16
            DataType.DT_INT32,DataType.DT_INT32_REF -> return org.nd4j.linalg.api.buffer.DataType.INT32
            DataType.DT_DOUBLE,DataType.DT_DOUBLE_REF -> return org.nd4j.linalg.api.buffer.DataType.DOUBLE
            DataType.DT_UINT16, DataType.DT_UINT16_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT16
            DataType.DT_UINT32,DataType.DT_UINT32_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT32
            DataType.DT_UINT64, DataType.DT_UINT64_REF -> return org.nd4j.linalg.api.buffer.DataType.UINT64
        }

        return org.nd4j.linalg.api.buffer.DataType.UNKNOWN
    }

    override fun nameSpaceDataType(): TensorNamespace.DataType {
        when(this.dataType) {
            DataType.DT_BOOL,DataType.DT_BOOL_REF -> return TensorNamespace.DataType.BOOL
            DataType.DT_FLOAT,DataType.DT_FLOAT_REF -> return TensorNamespace.DataType.FLOAT
            DataType.DT_STRING,DataType.DT_STRING_REF -> return TensorNamespace.DataType.STRING
            DataType.DT_BFLOAT16,DataType.DT_BFLOAT16_REF -> return TensorNamespace.DataType.BFLOAT16
            DataType.DT_INT64,  DataType.DT_INT64_REF -> return TensorNamespace.DataType.INT64
            DataType.DT_HALF,DataType.DT_HALF_REF-> return TensorNamespace.DataType.FLOAT16
            DataType.DT_INT16,DataType.DT_INT16_REF -> return TensorNamespace.DataType.INT16
            DataType.DT_INT32,DataType.DT_INT32_REF -> return TensorNamespace.DataType.INT32
            DataType.DT_DOUBLE,DataType.DT_DOUBLE_REF -> return TensorNamespace.DataType.DOUBLE
            DataType.DT_UINT16,DataType.DT_UINT16_REF -> return TensorNamespace.DataType.UINT16
            DataType.DT_UINT32, DataType.DT_UINT32_REF -> return TensorNamespace.DataType.UINT32
            DataType.DT_UINT64,DataType.DT_UINT64_REF -> return TensorNamespace.DataType.UINT64
        }

        return TensorNamespace.DataType.UNDEFINED
    }

}

fun attrDefaultValue(): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
    return TensorflowIRAttr(AttrDef.getDefaultInstance(), AttrValue.getDefaultInstance())
}

class TensorflowIRAttr(inputAttributeDef: AttrDef, inputAttributeValue: AttrValue): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {

    private val attributeDef = inputAttributeDef
    private val attributeValue = inputAttributeValue

    override fun name(): String {
        return attributeDef.name
    }

    override fun floatValue(): Float {
        return attributeValue.f
    }

    override fun listFloatValue(): List<Float> {
        return attributeValue.list.fList
    }


    override fun intValue(): Long {
        return attributeValue.i
    }

    override fun listIntValue(): List<Long> {
        return attributeValue.list.iList
    }

    override fun boolValue(): Boolean {
        return attributeValue.b
    }

    override fun listBoolValue(): List<Boolean> {
        return attributeValue.list.bList
    }

    override fun attributeValueType(): AttributeValueType {
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



    override fun internalAttributeDef(): AttrDef {
        return attributeDef
    }

    override fun internalAttributeValue(): AttrValue {
        return attributeValue
    }

    override fun listTensorValue(): List<IRTensor<TensorProto, DataType>> {
        return attributeValue.list.tensorList.map { input -> TensorflowIRTensor(input)
        }
    }

    override fun tensorValue(): IRTensor<TensorProto, DataType> {
        return TensorflowIRTensor(attributeValue.tensor)
    }

    override fun stringValue(): String {
        return attributeValue.s.toStringUtf8()
    }

    override fun listStringValue(): List<String> {
        return attributeValue.list.sList.map { it.toStringUtf8() }
    }

    override fun dataTataTypeValue(): IRDataType<DataType> {
        return TensorflowIRDataType(attributeValue.type)
    }

}

class TensorflowIRArgDef(input: OpDef.ArgDef): IRArgDef<OpDef.ArgDef, DataType> {
    private val argDefValue = input

    override fun dataType(): IRDataType<DataType> {
        return TensorflowIRArgDef(argDefValue).dataType()
    }

    override fun name(): String {
        return argDefValue.name
    }

    override fun description(): String {
        return argDefValue.description
    }

    override fun internalValue(): OpDef.ArgDef {
        return argDefValue
    }

    override fun indexOf(): Integer {
        TODO("Not yet implemented")
    }

}

class TensorflowIROp(input: OpDef): IROpDef<GraphDef,OpDef, TensorProto, OpDef.ArgDef, DataType, AttrDef, AttrValue> {

    val opDef = input

    override fun attributes(): List<IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
        return opDef.attrList.map {
            TensorflowIRAttr(it, AttrValue.getDefaultInstance())
        }
    }

    override fun opName(): String {
        return opDef.name
    }

    override fun internalValue(): OpDef {
        return opDef
    }

    override fun inputArgs(): List<IRArgDef<OpDef.ArgDef, DataType>> {
        return opDef.inputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

    override fun outputArgs(): List<IRArgDef<OpDef.ArgDef, DataType>> {
        return opDef.outputArgList.map {
            TensorflowIRArgDef(it)
        }
    }

}

class TensorflowIRNode(inputNode: NodeDef, inputOpDef: OpDef): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, DataType> {

    private val nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attrList)
    private val attrMap: Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> = initAttrMapFromNode(inputNode)
    private val opDescriptor: OpNamespace.OpDescriptor
    private val mappingProcess: MappingProcess<GraphDef,OpDef, NodeDef, TensorProto, OpDef.AttrDef,
    AttrValue, DataType> = tensorflowOpRegistry.lookupOpMappingProcess(inputNode.op)
    //private val inputs: List<OpNamespace.ArgDescriptor>
    //private val outputs: List<OpNamespace.ArgDescriptor>

    init {
        opDescriptor = nd4jOpDescriptors.findOp(mappingProcess.opName())
        // inputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
        // outputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR }

    }

    private fun attrDefsByName(input: List<AttrDef>): Map<String, AttrDef> {
        val ret = HashMap<String, AttrDef>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: NodeDef): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
        val ret = HashMap<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>>()
        input.attrMap.forEach { (key, value) ->
            ret[key] =  TensorflowIRAttr(attrDefsMap.getOrDefault(key, AttrDef.getDefaultInstance()), value)
        }

        return ret
    }

    override fun opName(): String {
        return nodeDef.op
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
        return opDef.outputArgList[index].name
    }



    override fun hasAttribute(inputName: String): Boolean {
        return nodeDef.containsAttr(inputName)
    }

    override fun attributeMap(): Map<String, IRAttribute<AttrDef, AttrValue, TensorProto, DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputData.map { TensorflowIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputValues.map { TensorflowIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
        return attrMap.getOrDefault(inputName, attrDefaultValue())
    }

    override fun internalValue(): NodeDef {
        return nodeDef
    }

    override fun numInputs(): Int {
        return nodeDef.inputCount
    }

    override fun numOutputs(): Int {
        return opDef.outputArgCount
    }

    override fun inputs(): List<String> {
        return nodeDef.inputList
    }

    override fun outputs(): List<String> {
        return opDef.outputArgList.map { input -> input.name }
    }

    /**
     * Get the list of tensors given an OpDef name (note: this is no tthe name of the input, but instead the op name, we use this to look up
     * the number attribute value and thus the number of inputs for a particular definition name.)
     * Tensorflow also allows multiple sets of lists of tensors as inputs, so we need to make sure to disambiguate which list of inputs we are looking up.
     */
    override fun numInputsForListOfTensors(name: String): Int {
        return nodeDef.getAttrOrThrow(opDef.inputArgList.first { input -> input.name == name }.numberAttr).i.toInt()
    }

    override fun inputNamesForListOfInputValues(inputListName: String): List<String> {
        val inputArgNames = opDef.inputArgList.map { argDef -> argDef.name }
        val indexOfDef = inputArgNames.indexOf(inputListName)
        if(indexOfDef < 0)
            return emptyList()
        var totalAmount: Long = 0
        for(i in 0 .. indexOfDef) {
            if(opDef.getInputArg(i).numberAttr.isNotEmpty()) {
                val numToAdd = nodeDef.getAttrOrDefault(opDef.getInputArg(i).numberAttr, AttrValue {
                    LongVal(1)
                }).i
                totalAmount += numToAdd
            }
            else
                totalAmount++
        }
        //note: this is inclusive
        return nodeDef.inputList.subList(indexOfDef,totalAmount.toInt())
    }

    override fun computeAdjustedOffsetForInput(
        nd4jName: String,
        inputFrameworkName: String,
        tensorInputMappings: Map<String, String>
    ): Int {
        val baseIndex = lookupIndexForArgDescriptor(
            argDescriptorName = nd4jName,
            opDescriptorName = this.opDescriptor.name,
            argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
        )

        val inputs = opDescriptor.argDescriptorList.filter { input -> input.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
        var totalAmount: Long = 0
        for(i in 0 until baseIndex) {
            val nd4jNameAtIndex = inputs.first {descriptor -> descriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR && descriptor.argIndex == i}.name
            val inputFrameworkName = tensorInputMappings[nd4jNameAtIndex]!!
            val totalNames = inputNamesForListOfInputValues(inputFrameworkName).size
            totalAmount += totalNames
        }

        if(totalAmount < 1)
            return baseIndex
        return (baseIndex + totalAmount.toInt()) - 1
    }

    override fun nd4jInputs(tensorMappings: Map<String, String>): List<String> {
        val ret = ArrayList<String>()
        val indicesToNames = HashMap<Int,List<String>>()
        tensorMappings.forEach { (nd4jName,inputFrameworkName) ->
            val idx = computeAdjustedOffsetForInput(nd4jName,inputFrameworkName,tensorMappings)
            val inputNamesForCurrInput = inputNamesForListOfInputValues(inputFrameworkName)
            indicesToNames[idx] = inputNamesForCurrInput
        }

        indicesToNames.toSortedMap().forEach { idx, names ->
            ret.addAll(names.filter {!ret.contains(it)})
        }

        return ret
    }

}


class TensorflowIRGraphRunner(irGraph: TensorflowIRGraph,inputNames: List<String>,outputNames: List<String>): IRGraphRunner<GraphDef,NodeDef,OpDef,TensorProto,AttrDef,AttrValue,DataType> {

    val irGraph = irGraph
    val graphRunner: GraphRunner
    init {
        graphRunner = GraphRunner.builder()
            .graphBytes(irGraph.graphDef.toByteArray())
            .inputNames(inputNames)
            .outputNames(outputNames)
            .build()
    }


    override fun graph(): IRGraph<GraphDef, NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType> {
        return irGraph
    }

    override fun run(inputs: Map<String, INDArray>): Map<String, INDArray> {
        return graphRunner.run(inputs)
    }

}

class TensorflowIRGraph(graphDef: GraphDef, opDef: OpList): IRGraph<
        GraphDef,
        NodeDef,
        OpDef,
        TensorProto,
        AttrDef,
        AttrValue,
        DataType> {

    val graphDef = graphDef
    val opDef = opDef
    override fun nodeByName(input: String): NodeDef {
        return graphDef.nodeByName(input)
    }


    override fun nodeList(): List<IRNode<NodeDef, TensorProto, AttrDef, AttrValue, DataType>> {
        return graphDef.nodeList.map {
                inputNode -> TensorflowIRNode(inputNode, tensorflowOps.findOp(inputNode.op))
        }
    }

    override fun internalValue(): GraphDef {
        return graphDef
    }



    override fun createMappingContext(
        opDef: OpDef,
        node: NodeDef,
        dynamicVariables: Map<String, TensorProto>
    ): MappingContext<GraphDef, NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType> {
        return TensorflowMappingContext(opDef = opDef,graph = this,node = node,dynamicVariables = dynamicVariables)
    }

    override fun frameworkName(): String {
        return "tensorflow"
    }

    override fun nd4jNameForInternalOpName(name: String): String {
        return tensorflowOpRegistry.lookupOpMappingProcess(name).opName()
    }

    override fun isConstantOpName(name: String): Boolean {
        return name == "Const" || name == "Placeholder"
    }

    override fun isConstant(opName: String): Boolean {
        return opName == "Const"
    }

    override fun isPlaceHolder(opName: String): Boolean {
        return opName == "Placeholder" || opName == "PlaceholderWithDefault"
    }

    override fun shapeOfInput(varName: String): LongArray? {
        val attrMap = nodeByName(varName).attrMap
        val shapeAvailable = attrMap.containsKey("shape")
        var shape: LongArray?
        shape = if (shapeAvailable) {
            attrMap["shape"]!!.list.iList.toLongArray()

        } else {
            //Some placeholders don't have any shape restrictions - i.e., accept anything...
            null
        }

        return shape
    }

    override fun dataTypeForVariable(varName: String): IRDataType<DataType> {
        val node = nodeByName(varName)
        val attrMap = node.attrMap
        if(!attrMap.containsKey("dtype")) {
            val retSet =  attrMap.values.filter { attrValue -> attrValue.type != DataType.DT_INVALID }
            if(retSet.isEmpty()) {
                return TensorflowIRDataType(DataType.DT_INVALID)
            } else {
                return TensorflowIRDataType(attrMap.values.filter { attrValue -> attrValue.type != DataType.DT_INVALID }.first().type)
            }
        } else if(attrMap.containsKey("dtype")) {
            return TensorflowIRDataType(attrMap["dtype"]!!.type)
        }

        return TensorflowIRDataType(DataType.DT_INVALID)
    }

    override fun importInfoForEachNode(dynamicVariables: Map<String, TensorProto>): Map<String, Pair<MappingContext<GraphDef, NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>, OpNamespace.OpDescriptor>> {
        return importInfoForEachNodeInGraph(graph = this,dynamicVariables = dynamicVariables)
    }

    override fun nodeIsPlaceHolder(nodeName: String): Boolean {
        return isPlaceHolder(nodeByName(nodeName).op)
    }


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


class TensorflowMappingContext(opDef: OpDef, node: NodeDef, graph: IRGraph<GraphDef,NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>,dynamicVariables: Map<String,TensorProto>) :
    AbstractMappingContext<GraphDef,NodeDef, OpDef, TensorProto, AttrDef, AttrValue, DataType>(opDef, node, graph,dynamicVariables) {

    override fun attrDef(name: String): AttrDef {
        if(opDef().attrCount < 1) {
            throw IllegalArgumentException("No attributes found for op def with name ${opDef.name}")
        }

        val ret =  opDef().attrList.firstOrNull { it.name == name } ?: error("No attribute found with name $name")
        return ret!!
    }

    override fun irAttributeValueForNode(valueName: String): IRAttribute<AttrDef, AttrValue, TensorProto, DataType> {
        val attrDef = attrDef(valueName)
        val attrValue = node.getAttrOrDefault(valueName, attrDef.defaultValue)
        return TensorflowIRAttr(inputAttributeDef = attrDef, inputAttributeValue = attrValue)

    }

    override fun tensorInputFor(name: String): IRTensor<TensorProto, DataType> {
        var foundIndex = -1
        /**
         * Use op definition name as 1 unified reference name in rules for static purposes, but
         * look up via index for specific node mappings.
         *
         * This is equivalent to the tf input position attribute value in the previous tensorflow import.
         */
        var baseIndexOffset: Int = 0
        opDef.inputArgList.forEachIndexed { index, argDef ->
            if(argDef.numberAttr.isNotEmpty()) {
                var totalNum = node.getAttrOrDefault(argDef.numberAttr,AttrValue {
                    i = 0
                })

                baseIndexOffset += totalNum.i.toInt()
            }

            if(argDef.name == name)
                foundIndex = min(index + baseIndexOffset,node.inputCount - 1)
        }


        if(foundIndex < 0) {
            throw java.lang.IllegalArgumentException("Node with name ${nodeName()} for opdef with name ${opDef.name} did not contain a tensor with name ${name}")
        }

        val graphNode = node.getInput(foundIndex)
        return tensorInputFromInputFrameworkName(graphNode)
    }

    override fun opName(): String {
        return node.op
    }

    override fun nodeName(): String {
        return node.name
    }

    override fun nd4jDataTypeFor(input: IRTensor<TensorProto, DataType>): org.nd4j.linalg.api.buffer.DataType {
        return input.dataType().nd4jDataType()
    }

    override fun createIRTensorFromNDArray(ndarray: INDArray): IRTensor<TensorProto, DataType> {
        val tensorProto = TensorProto {
            RawData(ndarray.data().asBytes())
            Shape(ndarray.shape().toList())
            DataType(convertToDataType(ndarray.dataType()))
        }

        return TensorflowIRTensor(tensorProto)
    }

    override fun tensorAttributeFor(name: String): IRTensor<TensorProto, DataType> {
        return TensorflowIRTensor(node.getAttrOrThrow(name).tensor)
    }

    override fun irNode(): IRNode<NodeDef, TensorProto, AttrDef, AttrValue, DataType> {
        return TensorflowIRNode(node, tensorflowOps.findOp(node.op))
    }

    override fun tensorInputFromInputFrameworkName(name: String): IRTensor<TensorProto, DataType> {
        val searchedNode = graph.nodeByName(stripVarSuffix(name))
        //no value to be found on placeholder, return default instance
        //if no value exists it's an output from another node
        if("Placeholder" in searchedNode.op || !searchedNode.containsAttr("value")) {
            println("Value for node $name is not a constant! This method only works for constants. Consider replacing the Placeholder node with a Constant node. This will return an empty tensor.")
            if(!dynamicVariables.containsKey(name))
                return TensorflowIRTensor(TensorProto.getDefaultInstance())
            else {
                val toConvert = dynamicVariables[name]!!
                return TensorflowIRTensor(toConvert)
            }
        }

        //value nodes are the values of attributes that are input nodes in a frozen graph
        return TensorflowIRTensor(searchedNode.getAttrOrThrow("value").tensor)
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


