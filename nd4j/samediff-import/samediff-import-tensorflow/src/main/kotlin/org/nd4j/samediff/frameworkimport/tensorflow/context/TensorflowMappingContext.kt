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
package org.nd4j.samediff.frameworkimport.tensorflow.context

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.context.AbstractMappingContext
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.tensorflow.*
import org.nd4j.samediff.frameworkimport.tensorflow.ir.*
import org.tensorflow.framework.*
import kotlin.math.min

class TensorflowMappingContext(opDef: OpDef, node: NodeDef, graph: IRGraph<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>, dynamicVariables: MutableMap<String, TensorProto>) :
    AbstractMappingContext<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>(opDef, node, graph,dynamicVariables) {

    override fun attrDef(name: String): OpDef.AttrDef {
        if(opDef().attrCount < 1) {
            throw IllegalArgumentException("No attributes found for op def with name ${opDef.name}")
        }

        val ret =  opDef().attrList.firstOrNull { it.name == name } ?: error("No attribute found with name $name")
        return ret!!
    }

    override fun irAttributeValueForNode(valueName: String): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
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
                var totalNum = node.getAttrOrDefault(argDef.numberAttr, AttrValue {
                    i = 0
                })

                baseIndexOffset += totalNum.i.toInt()
            }

            if(argDef.name == name)
                foundIndex = min(index + baseIndexOffset, node.inputCount - 1)
        }


        if(foundIndex < 0) {
            throw IllegalArgumentException("Node with name ${nodeName()} for opdef with name ${opDef.name} did not contain a tensor with name ${name}")
        }

        var graphNode = node.getInput(foundIndex)
        if(graphNode.endsWith("/read"))
            graphNode = graphNode.replace("/read","")
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

    override fun irNode(): IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {
        return TensorflowIRNode(node,  OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow").values.first { input ->
            input.name == node.op
        },graph.opMappingRegistry())
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

    override fun nodeInputNameForOpDefInputName(name: String): String {
        val inputNameIdx  = opDef.inputArgList.map { input -> input.name  }.indexOf(name)
        if(inputNameIdx < 0) {
            throw java.lang.IllegalArgumentException("No name ${name} found on op def with name ${opDef.name}")
        }
        return node.getInput(inputNameIdx)
    }


}