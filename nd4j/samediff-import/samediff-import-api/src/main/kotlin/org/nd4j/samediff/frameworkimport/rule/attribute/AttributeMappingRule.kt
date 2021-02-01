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
package org.nd4j.samediff.frameworkimport.rule.attribute

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRAttribute
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface AttributeMappingRule<GRAPH_TYPE: GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,NODE_DEF_TYPE: GeneratedMessageV3,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>
        where DATA_TYPE: ProtocolMessageEnum {

    fun initWithMappingProcess(mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>)

    fun mappingNamesToPerform(): Map<String,String>

    fun mappingTransformerArgs(): Map<String, List<OpNamespace.ArgDescriptor>>

    fun setMappingTransformerArgs(args: Map<String,List<OpNamespace.ArgDescriptor>>)

    fun name(): String

    fun modifyName(name: String)

    fun modifyInputFrameworkOpName(name: String)

    fun serialize(): MapperNamespace.MappingRule

    fun convertAttributes(mappingCtx: MappingContext<GRAPH_TYPE, NODE_DEF_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor>

    fun convertAttributesReverse(allInputArguments: List<OpNamespace.ArgDescriptor>, inputArgumentsToProcess: List<OpNamespace.ArgDescriptor>): List<IRAttribute<ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>


    fun isInputFrameworkTensorName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): Boolean

    fun isNd4jTensorName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): Boolean

    fun isInputFrameworkAttributeName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): Boolean

    fun isOutputFrameworkAttributeName(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): Boolean

    fun argDescriptorType(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): OpNamespace.ArgDescriptor.ArgType

    fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean

    fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean

    fun attributeValueTypeFor(name: String,mappingProcess: MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>): AttributeValueType

    fun argDescriptorTypesForOutputName(
        name: String, mappingProcess:
        MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE,
                ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    ): List<OpNamespace.ArgDescriptor.ArgType>
}