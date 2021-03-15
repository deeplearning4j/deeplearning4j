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
import org.apache.commons.lang3.StringUtils
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
    override fun nodeByName(input: String): Onnx.NodeProto {
        return cachedNodeList.first { inputNode -> inputNode.nodeName() == input }.internalValue()
    }

    init {
        //sometimes onnx nodes will have empty names, ensure that each node has a deterministically generated name
        val indexToNode = HashMap<Int,Onnx.NodeProto>()
        val graphDefBuilder = graphDef.toBuilder()
        val opTypes = HashMap<String,String>()
        graphDef.nodeList.forEachIndexed { index,node ->
            if(node.name.isEmpty()) {
                val newNodeBuilder = node.toBuilder()
                if(node.outputCount > 1) {
                    println("Found node with no name and > 1 input.  Node was $node. Using first output as name.")
                }
                val newName = node.getOutput(0)
                newNodeBuilder.name = newName
                val newNode = newNodeBuilder.build()
                indexToNode[index] = newNode
            }
        }

        if(indexToNode.isNotEmpty()) {
            indexToNode.forEach { (index, node) ->
                graphDefBuilder.setNode(index,node)
            }

            this.graphDef = graphDefBuilder.build()
        }

        this.graphDef.nodeList.forEach { node ->
            opTypes[node.name] = node.opType
        }

        val initializers = this.graphDef.initializerList.map { input -> input.name }
        println(initializers)
        cachedNodeList = nodeList()
        val inputList = this.graphDef.inputList.filter { input -> !opTypes.containsKey(input.name) && !initializers.contains(input.name)}.map { input -> input.name }
        val varList = this.graphDef.inputList.filter { input -> initializers.contains(input.name) }.map { input -> input.name }
        println("Inputs $inputList")
        println("Variables $varList")
        this.inputList.addAll(inputList)
        this.variableList.addAll(inputList)
        outputList.addAll(this.graphDef.outputList.map { input -> input.name })
    }


    override fun nodeList(): List<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>> {
        val ret2 =
            ArrayList<IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>>()
        //add all inputs, outputs, initializers together as "nodes" similar to TF

        val identityOp =  OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")["Constant"]!!
        //for model import purposes, add identity ops as dummies similar to how tensorflow does placeholders/constants
        graphDef.inputList.forEach { input ->
            //note: this is not a real op name in onnx, this is purely for flagging for import to grab the node from the initializer
            //add dummy values for placeholders
            val nodeToAdd = NodeProto {
                opType = "Constant"
                name = input.name
                Attribute(
                    Onnx.AttributeProto.newBuilder().setName("value").addTensors(Onnx.TensorProto.getDefaultInstance())
                        .build()
                )
            }

            ret2.add(OnnxIRNode(nodeToAdd, identityOp,opMappingRegistry))
        }

        graphDef.nodeList.forEach {

            val opDefOrNull =  OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")[it.opType]!!
            if(opDefOrNull == null) {
                throw IllegalArgumentException("Op def name ${it.opType} not found!")
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

        outputNodes.forEach { nodeProto ->
            nodeProto.outputList.forEachIndexed { index, outputName ->
                val indexOfOutput = if(index < 1) "" else ":$index"
                if(!ret2.map { node -> node.nodeName() }.contains(outputName)) {
                    val nodeToAdd = NodeProto {
                        opType = "Identity"
                        name = outputName
                        Input("${nodeProto.name}$indexOfOutput")
                    }

                    ret2.add(OnnxIRNode(nodeToAdd, identityOp,opMappingRegistry))
                }
            }

        }



        graphDef.initializerList.forEach { initializer ->
            //note: this is not a real op name in onnx, this is purely for flagging for import to grab the node from the initializer
            val nodeToAdd = NodeProto {
                opType = "Constant"
                name = initializer.name
                Attribute(
                    Onnx.AttributeProto.newBuilder().setName("value").addTensors(Onnx.TensorProto.getDefaultInstance())
                        .build()
                )
            }

            ret2.add(OnnxIRNode(nodeToAdd, identityOp,opMappingRegistry))
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
        return name == "Constant" || name == "Placeholder"
    }

    override fun isConstant(opName: String): Boolean {
        return opName == "Constant"
    }

    override fun isPlaceHolder(opName: String): Boolean {
        return opName == "Constant"
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

    override fun importInfoForEachNode(dynamicVariables: MutableMap<String, Onnx.TensorProto>): Map<String, Pair<MappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>, OpNamespace.OpDescriptor>> {
        return importInfoForEachNodeInGraph(graph = this,dynamicVariables = dynamicVariables)
    }

    override fun nodeIsPlaceHolder(nodeName: String): Boolean {
        return this.inputList.contains(nodeName)
    }

    override fun opMappingRegistry(): OpMappingRegistry<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto> {
        return opMappingRegistry
    }

    override fun addConstantNode(name: String, value: INDArray) {
        val graphBuilder = graphDef.toBuilder()
        val converted = convertToOnnxTensor(value,name)
        graphBuilder.addInitializer(converted)
        this.graphDef = graphBuilder.build()
    }

    override fun updateNode(node: IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>) {
        val nodeByName = nodeByName(node.nodeName())
        val graphBuilder = graphDef.toBuilder()
        val indexOfNode = graphBuilder.nodeList.indexOf(nodeByName)
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
        this.outputList = outputList as ArrayList<String>
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
        return variableList.contains(nodeName)
    }

    override fun isVariableOpName(name: String): Boolean {
        return name != "Constant"
    }

    override fun getConstantArrayForName(name: String): INDArray {
       return OnnxIRTensor(graphDef.initializerList.first { input -> input.name == name }).toNd4jNDArray()
    }
}