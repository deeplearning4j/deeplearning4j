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
package org.nd4j.samediff.frameworkimport.ir

import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface IRNode<NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE>
        where  DATA_TYPE: ProtocolMessageEnum {


    /**
     * Removes the attribute with the specified name
     * from this node. Returns the attribute value that was removed.
     */
    fun removeAttribute(attributeName: String): ATTRIBUTE_VALUE_TYPE

    fun nd4jInputs(tensorMappings: Map<String, String>): List<String>

    fun computeAdjustedOffsetForInput(
        nd4jName: String,
        inputFrameworkName: String,
        tensorInputMappings: Map<String, String>
    ): Int

    /**
     * Get the list of inputs from the node that represent a particular
     * [OpDef] input list name.
     */
    fun inputNamesForListOfInputValues(inputListName: String): List<String>

    /**
     * Compute the number of inputs
     * for a list of tensors that reflect 1 or more inputs
     * as 1 name.
     */
    fun numInputsForListOfTensors(name: String): Int

    /**
     * List of inputs in to the node
     * @return the list of input names for this node
     */
    fun  createInputsFrom(inputData: List<TENSOR_TYPE>): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    /**
     * List of outputs
     * @return the list of output names for this node
     */
    fun createOutputsFrom(inputValues: List<TENSOR_TYPE>): List<IRTensor<TENSOR_TYPE, DATA_TYPE>>

    /**
     * Op name
     */
    fun opName(): String

    /**
     * The name of the node
     * @return the name of the node
     */
    fun nodeName(): String

    /**
     * Sets this node's name to  be the new name
     */
    fun setNodeName(name: String)

    /**
     * Dynamically add an input to the node

     */
    fun addInput(inputName: String)


    /**
     * Sets the input at a particular index
     */
    fun setInputAt(index: Int,name: String)


    /**
     * Sets the output name at a particular index
     */
    fun setOutputAt(index: Int,name: String)

    /**
     * List of input names
     */
    fun inputs(): List<String>

    /**
     * List of output names
     */
    fun outputs(): List<String>

    /**
     * The input at a particular index
     * @return the name at the particular index
     */
    fun inputAt(index: Int): String

    /**
     * The output at a particular index
     */
    fun outputAt(index: Int): String

    fun isControlflowOp(): Boolean

    /**
     * The number of inputs for this node
     */
    fun numInputs(): Int

    /**
     * The number of outputs for this node
     */
    fun numOutputs(): Int

    fun attributeMap(): Map<String, IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>
    fun getAttribute(inputName: String): IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
    fun hasAttribute(inputName: String): Boolean

    fun internalValue(): NODE_TYPE

    fun internalValueString(): String {
        return internalValue().toString()
    }
}