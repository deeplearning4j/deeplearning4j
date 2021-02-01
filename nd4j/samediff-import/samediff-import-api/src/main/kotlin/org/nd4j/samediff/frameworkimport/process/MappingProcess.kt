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
package org.nd4j.samediff.frameworkimport.process

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeValueType
import org.nd4j.samediff.frameworkimport.ir.IRNode
import org.nd4j.samediff.frameworkimport.rule.tensor.TensorMappingRule
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

interface MappingProcess<
        GRAPH_TYPE: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> {



    fun inputOpDefValueTypes(): Map<String, AttributeValueType>

    fun opName(): String

    fun frameworkVersion(): String

    fun inputFramework(): String

    fun inputFrameworkOpName(): String

    fun attributeMappingRules(): List<AttributeMappingRule<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_DEF_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            TENSOR_TYPE,
            DATA_TYPE>>

    fun tensorMappingRules():  List<TensorMappingRule<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_DEF_TYPE,
            ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>>

    fun applyProcess(mappingCtx: MappingContext<GRAPH_TYPE, NODE_DEF_TYPE, OP_DEF_TYPE,
            TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    ):
            Pair<MappingContext<GRAPH_TYPE,
                    NODE_DEF_TYPE,
                    OP_DEF_TYPE,
                    TENSOR_TYPE, ATTRIBUTE_TYPE,
                    ATTRIBUTE_VALUE_TYPE,
                    DATA_TYPE>, OpNamespace.OpDescriptor>

    fun applyProcessReverse(input: OpNamespace.OpDescriptor): IRNode<NODE_DEF_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>


    fun indexOverrides() : Map<Int,Int>

    fun serialize(): MapperNamespace.MapperDeclaration


}