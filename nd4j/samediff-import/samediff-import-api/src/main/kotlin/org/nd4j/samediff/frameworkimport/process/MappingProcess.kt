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


    /**
     * Returns the array resolution type.
     * This configures how a mapping process should resolve variables.
     * Sometimes there are differences between the number of input variables
     * in an nd4j op and the op that is being imported from.
     * This could include things like attributes being converted to inputs, the reverse
     * or other situations like multiple variables.
     *
     * There are 3 values:
     * DIRECT: map as is. This means import the inputs exactly as described.
     * OVERRIDE: use nd4j's descriptor and prioritize the variables resolved from the op descriptor
     * ERROR_ON_NOT_EQUAL: throw an exception if the resolved and the import aren't an exact match.
     * This can be used when debugging or just as an assertion about an ops state upon import.
     */
    fun arrayResolutionType(): MapperNamespace.VariableResolutionType

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