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

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.ir.IRTensor
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.AttrValue
import org.nd4j.samediff.frameworkimport.tensorflow.LongVal
import org.tensorflow.framework.*

class TensorflowIRNode(inputNode: NodeDef, inputOpDef: OpDef,tensorflowOpMappingRegistry: OpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>):
    IRNode<NodeDef, TensorProto, OpDef.AttrDef, AttrValue, DataType> {

    private var nodeDef = inputNode
    private val opDef = inputOpDef
    private val attrDefsMap = attrDefsByName(inputOpDef.attrList)
    private val attrMap: Map<String, IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> = initAttrMapFromNode(inputNode)
    private val opDescriptor: OpNamespace.OpDescriptor
    private val tensorflowOpRegistry = tensorflowOpMappingRegistry
    private val mappingProcess: MappingProcess<GraphDef, OpDef, NodeDef, TensorProto, OpDef.AttrDef,
            AttrValue, DataType> = tensorflowOpRegistry.lookupOpMappingProcess(inputNode.op)
    //private val inputs: List<OpNamespace.ArgDescriptor>
    //private val outputs: List<OpNamespace.ArgDescriptor>

    init {
        opDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingProcess.opName())
        // inputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR }
        // outputs = opDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR }

    }

    private fun attrDefsByName(input: List<OpDef.AttrDef>): Map<String, OpDef.AttrDef> {
        val ret = HashMap<String, OpDef.AttrDef>()
        input.forEach {
            ret[it.name] = it
        }
        return ret
    }

    private fun initAttrMapFromNode(input: NodeDef): Map<String, IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        val ret = HashMap<String, IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>>()
        input.attrMap.forEach { (key, value) ->
            ret[key] = TensorflowIRAttr(attrDefsMap.getOrDefault(key, OpDef.AttrDef.getDefaultInstance()), value)
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

    override fun attributeMap(): Map<String, IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType>> {
        return attrMap
    }

    override fun createInputsFrom(inputData: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputData.map { TensorflowIRTensor(it) }
    }

    override fun createOutputsFrom(inputValues: List<TensorProto>): List<IRTensor<TensorProto, DataType>> {
        return inputValues.map { TensorflowIRTensor(it) }
    }

    override fun getAttribute(inputName: String): IRAttribute<OpDef.AttrDef, AttrValue, TensorProto, DataType> {
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
            if(!tensorInputMappings.containsKey(nd4jNameAtIndex)) {
                throw IllegalArgumentException("Tensor input mapping with key $nd4jNameAtIndex not found! Keys were ${tensorInputMappings.keys}")
            }
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
        val indicesToNames = HashMap<Int, List<String>>()
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

    override fun addInput(inputName: String) {
        val newNode = nodeDef.toBuilder()
        newNode.addInput(inputName)
        this.nodeDef = newNode.build()

    }

}