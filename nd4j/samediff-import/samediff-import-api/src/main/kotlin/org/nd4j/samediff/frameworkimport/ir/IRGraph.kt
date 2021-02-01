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

import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface IRGraph<
        GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        ATTRIBUTE_TYPE: GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
        DATA_TYPE : ProtocolMessageEnum> {

    fun addConstantNode(name: String,value: INDArray)

    fun importInfoForEachNode(dynamicVariables: MutableMap<String, TENSOR_TYPE>): Map<String, Pair<MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor>>

    fun shapeOfInput(varName: String): LongArray?

    fun dataTypeForVariable(varName: String): IRDataType<DATA_TYPE>

    fun isConstant(opName: String): Boolean

    fun nodeIsPlaceHolder(nodeName: String): Boolean

    fun isPlaceHolder(opName: String): Boolean

    fun isVariable(nodeName: String): Boolean

    fun isConstantOpName(name: String): Boolean

    fun isVariableOpName(name: String): Boolean

    fun nodeByName(input: String): NODE_TYPE

    fun nodeList(): List<IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>>

    fun internalValue(): GRAPH_TYPE

    fun opMappingRegistry(): OpMappingRegistry<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,DATA_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE>

    fun updateNode(node: IRNode<NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>)

    fun createMappingContext(
        opDef: OP_DEF_TYPE,
        node: NODE_TYPE,
        dynamicVariables: MutableMap<String, TENSOR_TYPE>
    ): MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>

    fun frameworkName(): String

    fun nd4jNameForInternalOpName(name: String): String

    fun graphOutputs(): List<String>

    fun outputAt(index: Int): String

    fun setOutputs(outputs: List<String>)

    fun graphInputs(): List<String>

    fun inputAt(index: Int): String

    fun setInputs(inputs: List<String>)

    fun getConstantArrayForName(name: String): INDArray
}