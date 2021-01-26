/* Copyright (c) 2021 Deeplearning4j Contributors
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/
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
import org.nd4j.samediff.frameworkimport.tensorflow.*
import org.nd4j.samediff.frameworkimport.tensorflow.context.TensorflowMappingContext
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


    init {
        val graphInputTo = Counter<String>()
        graphDef.nodeList.forEach {
            it.inputList.forEach { inputName -> graphInputTo.incrementCount(inputName,1.0) }
            //all placeholders are considered inputs
            if(it.op.contains("Placeholder"))
                inputs.add(it.name)
        }

        graphDef.nodeList.forEach {
            //node not input in to anything
            if(graphInputTo.getCount(it.name) < 1) {
                outputs.add(it.name)
            }
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


}