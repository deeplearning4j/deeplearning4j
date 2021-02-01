/* ******************************************************************************
 * Copyright (c) 2021 Deeplearning4j Contributors
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/
package org.nd4j.samediff.frameworkimport.rule.attribute

import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class StringNotEqualsAdapterRule<
        GRAPH_DEF: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE>(
    mappingNamesToPerform: Map<String, String> = emptyMap(),
    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>> = emptyMap()):
    BaseAttributeExtractionRule<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "stringnotequalsadapterrule",
        mappingNamesToPerform =  mappingNamesToPerform,
        transformerArgs = transformerArgs)
        where DATA_TYPE: ProtocolMessageEnum {

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorForName = transformerArgs[k]
            val compString = descriptorForName!![0].stringValue
            val testValue = mappingCtx.irAttributeValueForNode(v).stringValue()
            val argDescriptorTypeList = mappingCtx.argDescriptorTypeForName(k)
            argDescriptorTypeList.forEach { argDescriptorType ->
                when(argDescriptorType) {
                    OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                        ret.add(ArgDescriptor {
                            name = k
                            argType = argDescriptorType
                            int64Value = if (testValue != compString) 1 else 0
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                            )

                        })
                    }

                    OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                        ret.add(ArgDescriptor {
                            name = k
                            argType = argDescriptorType
                            boolValue = testValue != compString
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.BOOL
                            )

                        })
                    }
                }
            }


        }
        return ret
    }

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.STRING
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.BOOL) ||
                argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INT64)
    }
}