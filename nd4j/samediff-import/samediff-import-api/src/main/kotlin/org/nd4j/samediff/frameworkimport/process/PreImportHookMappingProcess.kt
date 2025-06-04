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
package org.nd4j.samediff.frameworkimport.process

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A dummy mapping process that serves as a placeholder for operations that are handled
 * entirely by PreImportHook implementations. This allows the registry system to have
 * a valid MappingProcess reference while the actual import logic is handled by the
 * PreImportHook mechanism.
 * 
 * @author Adam Gibson
 */
class PreImportHookMappingProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum>(
    private val inputFrameworkName: String,
    private val inputFrameworkOpName: String
) : MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {

    override fun arrayResolutionType(): MapperNamespace.VariableResolutionType = 
        MapperNamespace.VariableResolutionType.DIRECT

    override fun inputOpDefValueTypes(): Map<String, AttributeValueType> = emptyMap()

    override fun opName(): String = "noop"

    override fun frameworkVersion(): String = "1.0"

    override fun inputFramework(): String = inputFrameworkName

    override fun inputFrameworkOpName(): String = inputFrameworkOpName

    override fun attributeMappingRules(): List<AttributeMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> = 
        emptyList()

    override fun tensorMappingRules(): List<TensorMappingRule<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>> = 
        emptyList()

    override fun applyProcess(
        mappingCtx: MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    ): Pair<MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor> {
        // Create a minimal op descriptor for noop
        // The actual operation will be handled by the PreImportHook
        val opDescriptor = OpNamespace.OpDescriptor.newBuilder()
            .setName("noop")
            .setOpDeclarationType(OpNamespace.OpDescriptor.OpDeclarationType.CUSTOM_OP_IMPL)
            .build()
            
        return Pair(mappingCtx, opDescriptor)
    }

    override fun applyProcessReverse(input: OpNamespace.OpDescriptor): IRNode<NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        throw UnsupportedOperationException("Reverse process not supported for PreImportHook placeholder operations")
    }

    override fun indexOverrides(): Map<Int, Int> = emptyMap()

    override fun serialize(): MapperNamespace.MapperDeclaration {
        return MapperNamespace.MapperDeclaration.newBuilder()
            .setFrameworkName(inputFrameworkName)
            .setInputFrameworkOpName(inputFrameworkOpName)
            .setOpName("noop")
            .setVariableResolutionType(MapperNamespace.VariableResolutionType.DIRECT)
            .build()
    }

    override fun toString(): String {
        return "PreImportHookMappingProcess(framework='$inputFrameworkName', op='$inputFrameworkOpName')"
    }
}