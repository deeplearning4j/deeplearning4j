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
package org.nd4j.samediff.frameworkimport.onnx.context

import onnx.Onnx
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.context.AbstractMappingContext
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxTensor
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRAttr
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRNode
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRTensor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException
import java.lang.IllegalStateException

class OnnxMappingContext(opDef: Onnx.NodeProto, node: Onnx.NodeProto, graph:
IRGraph<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto,
        Onnx.AttributeProto,
        Onnx.AttributeProto, Onnx.TensorProto.DataType>, dynamicVariables: MutableMap<String, Onnx.TensorProto>) :
    AbstractMappingContext<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto,
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
            attrValue = Onnx.AttributeProto.newBuilder()
                .setName("value").addTensors(Onnx.TensorProto.getDefaultInstance())
                .build()
        else if(attrValue == null) {
            attrValue = Onnx.AttributeProto.newBuilder()
                .setName(valueName)
                .build()
            println("Unable to resolve attribute for name $valueName for node ${nodeName()} for op type ${opName()}")
        }
        return OnnxIRAttr(inputAttributeDef = attrDef, inputAttributeValue = attrValue!!)

    }

    override fun tensorInputFor(name: String): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
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
        return OnnxIRTensor(convertToOnnxTensor(ndarray, "tensor"))
    }

    override fun tensorAttributeFor(name: String): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        return irAttributeValueForNode(name).tensorValue()
    }

    override fun irNode(): IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {
        // Use the registry's lookup method instead of bypassing it
        val opRegistry = graph.opMappingRegistry()
        val opDefToUse = if(node.opType == "Placeholder") {
            opRegistry.lookupInputFrameworkOpDef("Constant")
        } else {
            opRegistry.lookupInputFrameworkOpDef(node.opType)
        }
        
        return OnnxIRNode(node, opDefToUse, opRegistry)
    }

    override fun tensorInputFromInputFrameworkName(name: String): IRTensor<Onnx.TensorProto, Onnx.TensorProto.DataType> {
        val castedGraph = graph as OnnxIRGraph
        val graphDef = castedGraph.graphDef()
        var foundIndex = opDef.inputList.map { input -> input.toString() }.indexOf(name)
        //optional or unknown tensors
        if(foundIndex < 0 || foundIndex >= node.inputCount) {
            println("Node with name ${nodeName()} for opdef with name ${opDef.name} did not contain a tensor with name ${name}, returning empty tensor")
            return OnnxIRTensor(Onnx.TensorProto.getDefaultInstance())
        }

        /**
         * Use op definition name as 1 unified reference name in rules for static purposes, but
         * look up via index for specific node mappings.
         *
         * This is equivalent to the tf input position attribute value in the previous tensorflow import.
         */
        val graphNode = if(node.opType == "Constant") name else node.getInput(foundIndex)
        val attemptedTensor = graphDef.initializerList.firstOrNull { it.name == graphNode }
            ?: return if(!dynamicVariables.containsKey(graphNode))
                OnnxIRTensor(Onnx.TensorProto.getDefaultInstance())
            else {
                val toConvert = dynamicVariables[graphNode]!!
                OnnxIRTensor(toConvert)
            }

        //no value to be found on placeholder, return default instance
        //if no value exists it's an output from another node

        //value nodes are the values of attributes that are input nodes in a frozen graph
        if(attemptedTensor == null) {
            throw IllegalArgumentException("Name $name not found in initializer list.")
        }
        return OnnxIRTensor(attemptedTensor!!)
    }

    override fun nodeInputNameForOpDefInputName(name: String): String {
        var foundIndex = opDef.inputList.map { input -> input.toString() }.indexOf(name)
        if(foundIndex < 0) {
            throw IllegalArgumentException("No name ${name} found on op def with name ${opDef.name}")
        }

        if(foundIndex >= node.inputCount) {
            throw IllegalStateException("Node with name $name was found at index $foundIndex but was inconsistent with number of inputs for node ${node.name}")
        }

        return node.getInput(foundIndex)
    }

    override fun hasInput(name: String): Boolean {
        var foundIndex = opDef.inputList.map { input -> input.toString() }.indexOf(name)
        return foundIndex >= 0 && foundIndex < node.inputCount
    }

    override fun preProcessNode() {
        val onnxIRNode = OnnxIRNode(node,opDef, registry())
        relevantNodePreProcessingHooks.forEach { hook ->
            hook.modifyNode(onnxIRNode,graph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
        }

        //post processed, we need to update the references in the node
       if(relevantNodePreProcessingHooks.isNotEmpty()) {
           this.node = onnxIRNode.internalValue()
           this.graph.updateNode(onnxIRNode)
       }

    }

}