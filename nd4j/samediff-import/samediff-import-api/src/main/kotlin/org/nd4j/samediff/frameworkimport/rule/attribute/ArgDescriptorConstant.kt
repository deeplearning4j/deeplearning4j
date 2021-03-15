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
package org.nd4j.samediff.frameworkimport.rule.attribute

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class ArgDescriptorConstant<
        GRAPH_DEF : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum>(
    mappingNamesToPerform: Map<String, String> = emptyMap(),
    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>
) :
    BaseAttributeExtractionRule<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (
        name = "argdescriptorconstant",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return true
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return true
    }

    override fun convertAttributes(
        mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF,
                ATTR_VALUE_TYPE, DATA_TYPE>
    ): List<OpNamespace.ArgDescriptor> {
        return transformerArgs.flatMap {
            it.value.map { descriptor ->
                ArgDescriptor {
                    name = descriptor.name
                    argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = descriptor.name,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = descriptor.argType
                    )
                    argType = descriptor.argType
                    boolValue = descriptor.boolValue
                    floatValue = descriptor.floatValue
                    doubleValue = descriptor.doubleValue
                    int32Value = descriptor.int32Value
                    int64Value = descriptor.int64Value
                    stringValue = descriptor.stringValue
                    inputValue = descriptor.inputValue
                    outputValue = descriptor.outputValue

                }
            }
        }
    }
}