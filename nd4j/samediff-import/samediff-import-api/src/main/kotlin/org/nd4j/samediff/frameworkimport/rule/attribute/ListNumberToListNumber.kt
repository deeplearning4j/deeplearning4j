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

abstract class ListNumberToListNumber<
        GRAPH_DEF : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum>(
    mappingNamesToPerform: Map<String, String>,
    transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>
) :
    BaseAttributeExtractionRule<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (
        name = "listnumbertolistnumber",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {
    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.INT ||
                argDescriptorType == AttributeValueType.FLOAT ||
                argDescriptorType == AttributeValueType.LIST_INT ||
                argDescriptorType == AttributeValueType.LIST_FLOAT
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INT64) ||
                argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for ((k, v) in mappingNamesToPerform()) {

            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when (irAttribute.attributeValueType()) {
                AttributeValueType.LIST_INT -> {
                    val baseIndex = if(mappingCtx.descriptorsSoFar().isEmpty()) lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                    ) else mappingCtx.descriptorsSoFar().size
                    val listInts = irAttribute.listIntValue()
                    listInts.forEachIndexed { index, element ->
                        val finalName = if (index > 0) k + "$index" else k
                        val argDescriptor = ArgDescriptor {
                            name = finalName
                            int64Value = element
                            argType = OpNamespace.ArgDescriptor.ArgType.INT64
                            argIndex = baseIndex + index
                        }

                        ret.add(argDescriptor)
                    }
                }
                AttributeValueType.LIST_FLOAT -> {
                    val baseIndex = if(mappingCtx.descriptorsSoFar().isEmpty()) lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                    ) else mappingCtx.descriptorsSoFar().size

                    val listFloats = irAttribute.listFloatValue()
                    listFloats.forEachIndexed { index, element ->
                        val finalName = if (index > 0) k + "$index" else k
                        val argDescriptor = ArgDescriptor {
                            name = finalName
                            doubleValue = element.toDouble()
                            argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                            argIndex = baseIndex + index
                        }

                        ret.add(argDescriptor)
                    }
                }
            }
        }

        return ret
    }
}