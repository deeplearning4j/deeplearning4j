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
package org.nd4j.samediff.frameworkimport.tensorflow.ir

import org.nd4j.common.primitives.Counter
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRDataType
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.importInfoForEachNodeInGraph
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.Attribute
import org.nd4j.samediff.frameworkimport.tensorflow.NodeDef
import org.nd4j.samediff.frameworkimport.tensorflow.context.TensorflowMappingContext
import org.nd4j.samediff.frameworkimport.tensorflow.convertNDArrayToTensorflowTensor
import org.nd4j.samediff.frameworkimport.tensorflow.nodeByName
import org.tensorflow.framework.*

class TensorflowIRGraph(graphDef: GraphDef, opDef: OpList
                        ,tensorflowOpMappingRegistry: OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType,OpDef.AttrDef,AttrValue>): IRGraph<
        GraphDef,
        NodeDef,
        OpDef,
        TensorProto,
        OpDef.AttrDef,
        AttrValue,
        DataType> {

    var graphDef = graphDef
    val opDef = opDef
    val tensorflowOpRegistry = tensorflowOpMappingRegistry
    var inputs = ArrayList<String>()
    var outputs = ArrayList<String>()
    var cachedNodeList : List<IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>>
    val nodeNames: Set<String>
    val inputsOutputs = HashSet<String>()

    init {
        val graphInputTo = Counter<String>()
        cachedNodeList = nodeList()
        nodeNames = HashSet()
        cachedNodeList.forEach {
            it.inputs().forEach { inputName -> graphInputTo.incrementCount(inputName,1.0) }
            it.inputs().forEach { input -> inputsOutputs.add(input) }
            it.outputs().forEach { output -> inputsOutputs.add(output) }
            //all placeholders are considered inputs
            if(it.opName().contains("Placeholder"))
                inputs.add(it.nodeName())
            //node not input in to anything
            if(graphInputTo.getCount(it.nodeName()) < 1) {
                outputs.add(it.nodeName())
            }

            nodeNames.add(it.nodeName())
        }

    }

    override fun nodeByName(input: String): NodeDef {
        return graphDef.nodeByName(input)
    }


    override fun nodeList(): List<IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>> {
        return graphDef.nodeList.map {
                inputNode ->
            TensorflowIRNode(inputNode, OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow").values.first { input ->
                input.name ==  inputNode.op
            },tensorflowOpRegistry)
        }
    }

    override fun internalValue(): GraphDef {
        return graphDef
    }



    override fun createMappingContext(
        opDef: OpDef,
        node: NodeDef,
        dynamicVariables: MutableMap<String, TensorProto>
    ): MappingContext<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {
        return TensorflowMappingContext(opDef = opDef, graph = this, node = node, dynamicVariables = dynamicVariables)
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

    override fun variableNames(): List<String> {
        return nodeNames.toList()
    }

    override fun shapeOfInput(varName: String): LongArray? {
        val attrMap = nodeByName(varName).attrMap
        val shapeAvailable = attrMap.containsKey("shape")
        var shape: LongArray?
        shape = if (shapeAvailable) {
            attrMap["shape"]!!.shape.dimList.map { input -> input.size }.toLongArray()
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
                return TensorflowIRDataType(attrMap.values.filter { attrValue -> attrValue.type != DataType.DT_INVALID }
                    .first().type)
            }
        } else if(attrMap.containsKey("dtype")) {
            return TensorflowIRDataType(attrMap["dtype"]!!.type)
        }

        return TensorflowIRDataType(DataType.DT_INVALID)
    }

    override fun importInfoForEachNode(dynamicVariables: MutableMap<String, TensorProto>): Map<String, Pair<MappingContext<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>, OpNamespace.OpDescriptor>> {
        return importInfoForEachNodeInGraph(graph = this,dynamicVariables = dynamicVariables)
    }

    override fun nodeIsPlaceHolder(nodeName: String): Boolean {
        return isPlaceHolder(nodeByName(nodeName).op)
    }

    override fun opMappingRegistry(): OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue> {
        return tensorflowOpRegistry
    }

    override fun addConstantNode(nodeName: String, value: INDArray) {
        val graphBuilder = graphDef.toBuilder()
        val convertedTensor = convertNDArrayToTensorflowTensor(value)
        val constant = NodeDef {
            name = nodeName
            op = "Const"
            Attribute("value", org.nd4j.samediff.frameworkimport.tensorflow.AttrValue {
                tensor = convertedTensor
            })
        }

        graphBuilder.addNode(constant)
        this.graphDef = graphBuilder.build()
    }

    override fun updateNode(node: IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>) {
        val nodeByName = nodeByName(node.nodeName())
        val internalGraphDefBuilder = graphDef.toBuilder()
        val indexOfNode = internalGraphDefBuilder.nodeList.indexOf(nodeByName)
        internalGraphDefBuilder.setNode(indexOfNode,node.internalValue())
        this.graphDef = internalGraphDefBuilder.build()
    }

    override fun graphOutputs(): List<String> {
        return outputs
    }

    override fun outputAt(index: Int): String {
        return outputs[index]
    }

    override fun graphInputs(): List<String> {
        return inputs
    }

    override fun setOutputs(outputs: List<String>) {
        this.outputs = outputs as ArrayList<String>
    }

    override fun inputAt(index: Int): String {
        return inputs[index]
    }

    override fun setInputs(inputs: List<String>) {
        this.inputs = inputs as ArrayList<String>
    }

    override fun isVariable(nodeName: String): Boolean {
        return isVariableOpName(nodeByName(nodeName).op)
    }

    override fun isVariableOpName(name: String): Boolean {
        return name == "Variable" || name == "VariableV2"
    }

    override fun getConstantArrayForName(name: String): INDArray {
        val node = nodeByName(name)
        if(!node.op.contains("Const")) {
            throw IllegalArgumentException("Illegal op found ${node.op} for name $name")
        }

        return TensorflowIRTensor(node.getAttrOrThrow("value").tensor).toNd4jNDArray()
    }

    override fun hasConstantInitializer(name: String): Boolean {
        if(!cachedNodeList.map { input -> input.nodeName() }.contains(name)) {
            return false
        }
        val node = nodeByName(name)
        return node != null && node.op == "Const"
    }

    override fun indexOfNode(input: String): Int {
        return cachedNodeList.map { input -> input.nodeName() }.indexOf(input)
    }

    override fun nodesWithInput(name: String): List<IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>> {
        return cachedNodeList.filter { input -> input.inputs().contains(name) }
    }

    override fun irNodeByName(input: String): IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {
        val node = nodeByName(input)
        return TensorflowIRNode(node,tensorflowOpRegistry.lookupInputFrameworkOpDef(node.op),opMappingRegistry())
    }

    override fun hasNode(nodeName: String): Boolean {
        return nodeNames.contains(nodeName)
    }

    override fun addGraphOutputsAsProcessingNodes(): Boolean {
        return false
    }

    override fun convertToNDArray(tensorTypeInput: TensorProto): INDArray {
        return TensorflowIRTensor(tensorTypeInput).toNd4jNDArray()
    }

    override fun isInputOrOutput(name: String): Boolean {
        return inputsOutputs.contains(name)
    }

    override fun updateNodeCacheWith(nodeList: List<IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>>) {
        this.cachedNodeList = nodeList
    }

    override fun convertToTensor(ndarrayInput: INDArray, tensorName: String): TensorProto {
        return convertNDArrayToTensorflowTensor(ndarrayInput)
    }


}