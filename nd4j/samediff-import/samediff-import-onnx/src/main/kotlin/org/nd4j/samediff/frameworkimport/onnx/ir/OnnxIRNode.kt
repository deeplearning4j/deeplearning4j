/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.samediff.frameworkimport.onnx.attrDefaultValue
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import java.util.HashMap

class OnnxIRNode(inputNode: Onnx.NodeProto, inputOpDef: Onnx.NodeProto,opMappingRegistry: OpMappingRegistry<Onnx.GraphProto,
        Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto,
        Onnx.AttributeProto>
):
    IRNode<Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType> {

    private var nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attributeList)
    private val attrMap: Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> =
        initAttrMapFromNode(inputNode)
    private val mappingProcess: MappingProcess<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>
    private val opMappingRegistry = opMappingRegistry
    init {
        mappingProcess = opMappingRegistry.lookupOpMappingProcess(inputNode.opType)
    }

    private fun attrDefsByName(input: List<Onnx.AttributeProto>): Map<String, Onnx.AttributeProto> {
        val ret = HashMap<String, Onnx.AttributeProto>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: Onnx.NodeProto): Map<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> {
        val ret =
            HashMap<String, IRAttribute<Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>>()
        input.attributeList.forEach {
            ret[it.name] = OnnxIRAttr(it, it)

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

    override fun addInput(inputName: String) {
        val nodeBuilder = nodeDef.toBuilder()
        nodeBuilder.addInput(inputName)
        this.nodeDef = nodeBuilder.build()
    }

}