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
package org.nd4j.samediff.frameworkimport.onnx.ir

import onnx.Onnx
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.importInfoForEachNodeInGraph
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.context.OnnxMappingContext
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.stripVarSuffix
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException

class OnnxIRGraph(graphDef: Onnx.GraphProto,opMappingRegistry: OpMappingRegistry<Onnx.GraphProto,
        Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,
        Onnx.AttributeProto>): IRGraph<
        Onnx.GraphProto, Onnx.NodeProto,
        Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto,
        Onnx.TensorProto.DataType> {

    var graphDef = graphDef
    val opList = graphDef.nodeList
    val opMappingRegistry = opMappingRegistry
    var cachedNodeList: List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>>
    var inputList = ArrayList<String>()
    var outputList = ArrayList<String>()
    var variableList = ArrayList<String>()
    var initializerSet = HashSet<String>()
    val nodeNames: Set<String>
    val inputsOutputs = HashSet<String>()


    override fun nodeByName(input: String): Onnx.NodeProto {
        //sometimes models exported from onnx will have tensorflow's var suffix
        val input2 = stripVarSuffix(input)
        if(!cachedNodeList.map { input -> input.nodeName() }.contains(input2)) {
            throw IllegalStateException("No input found for node name $input")
        }
        return cachedNodeList.first { inputNode -> inputNode.nodeName() == input2 }.internalValue()
    }

    init {
        //sometimes onnx nodes will have empty names, ensure that each node has a deterministically generated name
        val indexToNode = HashMap<Int,Onnx.NodeProto>()
        val opTypes = HashMap<String,String>()

        nodeNames = HashSet()
        preProcessZeroSuffixes()

        cachedNodeList = nodeList()


        cachedNodeList.forEachIndexed { index,node ->
            if(node.nodeName().isEmpty()) {
                val newNodeBuilder = node.internalValue().toBuilder()
                if(node.numOutputs() > 1) {
                    println("Found node with no name and > 1 input.  Node was $node. Using first output as name.")
                }
                val newName = node.outputAt(0)
                newNodeBuilder.name = newName.replace(":0","")
                val newNode = newNodeBuilder.build()
                indexToNode[index] = newNode
            }

            node.inputs().forEach { inputsOutputs.add(it.replace(":0","")) }
            node.outputs().forEach { inputsOutputs.add(it.replace(":0","")) }
            nodeNames.add(node.nodeName().replace(":0",""))
            opTypes[node.nodeName()] = node.opName()


        }


        val initializers = this.graphDef.initializerList.map { input -> input.name.replace(":0","") }
        println(initializers)
        val inputList = this.graphDef.inputList.filter { input -> !opTypes.containsKey(input.name.replace(":0","")) && !initializers.contains(input.name.replace(":0",""))}.map { input -> input.name.replace(":0","") }
        val varList = this.graphDef.inputList.filter { input -> initializers.contains(input.name.replace(":0","")) }.map { input -> input.name.replace(":0","") }
        println("Inputs $inputList")
        println("Variables $varList")
        this.inputList.addAll(inputList)
        this.variableList.addAll(inputList)
        initializerSet.addAll(initializers)
        outputList.addAll(this.graphDef.outputList.filter { valueInfo -> !valueInfo.name.contains(valueInfo.name) }
            .map { input -> input.name.replace(":0","") })
    }

    /**
     * Handle zero suffixes such that the suffixes are removed.
     * This is for when you import a tensorflow model or import a model
     * from tf onnx and need to handle the :0 edge case which is pretty common
     * when interacting with anything that came from tensorflow.
     */
    fun preProcessZeroSuffixes() {
        val graphDefBuilder = graphDef.toBuilder()
        val initializerList = ArrayList<Onnx.TensorProto>()
        //ensure we prune all :0 suffixes which may come from tf onnx
        for(i in 0 until graphDefBuilder.initializerCount) {
            val currInitializer = graphDefBuilder.initializerList[0]
            val builder = currInitializer.toBuilder()
            builder.name = currInitializer.name.replace(":0","")
            initializerList.add(builder.build())
            graphDefBuilder.removeInitializer(0)
        }


        graphDefBuilder.nodeBuilderList.forEach {
            it.name = it.name.replace(":0","")
            val inputList = it.inputList.toMutableList()
            val outputList = it.outputList.toMutableList()
            for(i in 0 until it.inputCount) {
                it.clearInput()
            }
            for(i in 0 until it.outputCount) {
                it.clearOutput()
            }

            it.addAllInput(inputList.map { input -> input.replace(":0","") })
            it.addAllOutput(outputList.map { input -> input.replace(":0","") })

        }

        initializerList.forEach { graphDefBuilder.addInitializer(it) }
        this.graphDef = graphDefBuilder.build()
    }


    override fun nodeList(): List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>> {
        if(cachedNodeList != null) {
            return cachedNodeList
        }

        val ret2 =
            ArrayList<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>>()
        //add all inputs, outputs, initializers together as "nodes" similar to TF

        val identityOp =  OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")["Constant"]!!

        //for model import purposes, add identity ops as dummies similar to how tensorflow does placeholders/constants
        val initializerListNames = graphDef.initializerList.map { input -> input.name.replace(":0","") }
        graphDef.inputList.filter { input -> !initializerListNames.contains(input.name.replace(":0","")) }.forEach { input ->
            //note: this is not a real op name in onnx, this is purely for flagging for import to grab the node from the initializer
            //add dummy values for placeholders
            val tensorBuilder = Onnx.TensorProto.newBuilder()
            tensorBuilder.name = input.name
            tensorBuilder.dataType = input.type.tensorType.elemType
            input.type.tensorType.shape.dimList.forEach {
                tensorBuilder.addDims(it.dimValue)
            }
            val nodeToAdd = NodeProto {
                opType = "Placeholder"
                name = input.name.replace(":0","")
                Attribute(
                    Onnx.AttributeProto.newBuilder().setName("value")
                        .addTensors(tensorBuilder.build())
                        .build()
                )
            }

            ret2.add(OnnxIRNode(nodeToAdd, identityOp,opMappingRegistry))
        }

        //add inputs and outputs for use cases like placeholder detection
        inputList.addAll(graphDef.inputList.filter { input -> !initializerListNames.contains(input.name) }.map { input -> input.name })
        outputList.addAll(graphDef.outputList.filter { valueInfo -> !outputList.contains(valueInfo.name) }.map { input -> input.name })
        val frameworkList =  OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")
        graphDef.nodeList.forEach {
            val opDefOrNull = if(!frameworkList.containsKey(it.opType)) {
                //use Constant as a placeholder for any op that resolves to noop, this is probably an op handled by the custom implementation
                frameworkList["Constant"]!!
            } else {
                frameworkList[it.opType]!!
            }
            ret2.add(OnnxIRNode(it, opDefOrNull!!,opMappingRegistry))
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




        this.cachedNodeList = ret2
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
        dynamicVariables: MutableMap<String, Onnx.TensorProto>
    ): MappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        return OnnxMappingContext(opDef = opDef, node = node, graph = this, dynamicVariables = dynamicVariables)
    }

    override fun frameworkName(): String {
        return "onnx"
    }

    override fun nd4jNameForInternalOpName(name: String): String {
        return opMappingRegistry.lookupOpMappingProcess(name).opName()
    }

    override fun isConstantOpName(name: String): Boolean {
        return name == "Constant"
    }

    override fun isConstant(opName: String): Boolean {
        return opName == "Constant"
    }

    override fun isPlaceHolder(opName: String): Boolean {
        //note: this is a dummy op only used for import, it's not a real onnx op
        return opName == "Placeholder"
    }

    override fun variableNames(): List<String> {
        return inputsOutputs.toList()
    }

    override fun shapeOfInput(varName: String): LongArray? {
        val firstOrNull = graphDef.initializerList.firstOrNull { inputNode -> inputNode.name == varName }
        if(firstOrNull != null)
            return firstOrNull.dimsList.toLongArray()
        else if(nodeIsPlaceHolder(stripVarSuffix(varName))) {
            val placeHolder = irNodeByName(stripVarSuffix(varName))
            val attrValue = placeHolder.attributeMap()["value"]!!.tensorValue().shape()
            val ret =  attrValue.toLongArray()
            for(i in ret.indices) {
                //missing dimension, probably dynamic, infer as -1 to match dynamic shape behavior in samediff
                if(ret[i] == 0L) {
                    ret[i] = -1
                }
            }

            return ret
        }
        return null
    }

    override fun dataTypeForVariable(varName: String): IRDataType<Onnx.TensorProto.DataType> {
        val varNameStripped = stripVarSuffix(varName)
        val firstOrNull = graphDef.initializerList.firstOrNull {
                inputNode -> inputNode.name == varNameStripped }
        val input = graphDef.inputList.firstOrNull { input2 ->
            input2.name == varNameStripped
        }
        if(firstOrNull != null)
            return OnnxIRDataType(Onnx.TensorProto.DataType.values()[firstOrNull!!.dataType])
        else if(nodeIsPlaceHolder(varNameStripped)) {
            if(input != null && input.type.hasTensorType()) {
                return OnnxIRDataType(Onnx.TensorProto.DataType.forNumber(input.type.tensorType.elemType))
            } else if(input != null && input.type.hasSequenceType()) {
                return OnnxIRDataType(Onnx.TensorProto.DataType.forNumber(input.type.sequenceType.elemType.tensorType.elemType))

            }

            val placeHolder = irNodeByName(varNameStripped)
            return placeHolder.attributeMap()["value"]!!.tensorValue().dataType()
        }
        else if(input != null)
            return OnnxIRDataType(Onnx.TensorProto.DataType.forNumber(input.type.tensorType.elemType))
        else
            return OnnxIRDataType(Onnx.TensorProto.DataType.UNDEFINED)
    }

    override fun importInfoForEachNode(dynamicVariables: MutableMap<String, Onnx.TensorProto>): Map<String, Pair<MappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>, OpNamespace.OpDescriptor>> {
        return importInfoForEachNodeInGraph(graph = this,dynamicVariables = dynamicVariables)
    }

    override fun nodeIsPlaceHolder(nodeName: String): Boolean {
        val realName = if(nodeName.endsWith(":0")) {
            nodeName.replace(":0","")
        } else {
            nodeName
        }


        return this.inputList.contains(realName) || this.inputList.contains("$realName:0")
    }

    override fun opMappingRegistry(): OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto> {
        return opMappingRegistry
    }

    override fun addConstantNode(name: String, value: INDArray) {
        val graphBuilder = graphDef.toBuilder()
        val converted = convertToOnnxTensor(value,name)
        graphBuilder.addInitializer(converted)

        val tensorShapeInfo = TensorTypeProto {
            shape = OnnxShapeProto {
                OnnxShape(value.shape().toList())
            }

        }

        val valueType = TypeProto {
            tensorType = tensorShapeInfo
        }

        val newValueInfo = ValueInfoProto {
            Type(valueType)
        }

        graphBuilder.addValueInfo(newValueInfo)
        this.graphDef = graphBuilder.build()
    }

    override fun updateNode(node: IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>) {
        val graphBuilder = graphDef.toBuilder()
        val indexOfNode = graphBuilder.nodeList.map { input -> input.name }.indexOf(node.nodeName())
        if(indexOfNode < 0) {
            throw IllegalStateException("No node of name ${node.nodeName()} was found")
        }
        graphBuilder.setNode(indexOfNode,node.internalValue())
        this.graphDef = graphBuilder.build()
    }

    override fun graphOutputs(): List<String> {
        return outputList
    }

    override fun outputAt(index: Int): String {
        return outputList[index]
    }

    override fun setOutputs(outputs: List<String>) {
        this.outputList = outputList
    }

    override fun graphInputs(): List<String> {
        return inputList
    }

    override fun inputAt(index: Int): String {
        return inputList[index]
    }

    override fun setInputs(inputs: List<String>) {
        this.inputList = inputs as ArrayList<String>
    }

    override fun isVariable(nodeName: String): Boolean {
        val realName = if(nodeName.endsWith(":0")) {
            nodeName.replace(":0","")
        } else {
            nodeName
        }

        return variableList.contains(realName) || variableList.contains("$realName:0")
    }

    override fun isVariableOpName(name: String): Boolean {
        return name != "Constant"
    }

    override fun getConstantArrayForName(name: String): INDArray {
        val check = graphDef.initializerList.map { input ->input.name }
        if(!check.contains(name)) {
            //initializer not found, see if there is a constant node
            if (this.nodeNames.contains(name)) {
                val constNode = nodeByName(name)
                if (constNode.opType == "Constant") {
                    //every constant should have a tensor value
                    val getValue = constNode.getAttribute(0).t
                    return OnnxIRTensor(getValue).toNd4jNDArray()
                } else {
                    throw IllegalArgumentException("Constant of name $name not found!")

                }

            }
        }

        return OnnxIRTensor(graphDef.initializerList.first { input -> input.name == name }).toNd4jNDArray()
    }

    override fun hasConstantInitializer(name: String): Boolean {
        return initializerSet.contains(name)
    }

    override fun indexOfNode(input: String): Int {
        return cachedNodeList.map { inputNode -> inputNode.nodeName() }.indexOf(input)
    }
    override fun nodesWithInput(name: String): List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>> {
        return cachedNodeList.filter { input -> input.inputs().contains(name) }
    }

    override fun irNodeByName(input: String): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        val node = nodeByName(input)
        return OnnxIRNode(node,opMappingRegistry.lookupInputFrameworkOpDef(node.opType),opMappingRegistry)
    }

    override fun hasNode(nodeName: String): Boolean {
        return nodeNames.contains(nodeName)
    }

    override fun addGraphOutputsAsProcessingNodes(): Boolean {
        return true
    }

    override fun convertToNDArray(tensorTypeInput: Onnx.TensorProto): INDArray {
        return OnnxIRTensor(tensorTypeInput).toNd4jNDArray()
    }

    override fun isInputOrOutput(name: String): Boolean {
        val realName = if(name.endsWith(":0")) {
            name.replace(":0","")
        } else {
            name
        }

        return inputsOutputs.contains(name) || inputsOutputs.contains(realName)
    }

    override fun updateNodeCacheWith(nodeList: List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>>) {
        this.cachedNodeList = nodeList
        val graphDefBuilder = graphDef.toBuilder()
        for(i in 0 until graphDefBuilder.nodeCount) {
            graphDefBuilder.removeNode(0)
        }
        nodeList.forEach {
            graphDefBuilder.addNode(it.internalValue())
        }

        this.graphDef = graphDefBuilder.build()
    }

    override fun convertToTensor(ndarrayInput: INDArray, tensorName: String): Onnx.TensorProto {
        return convertToOnnxTensor(ndarrayInput,tensorName)
    }
}