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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class ValueMapping<
        GRAPH_DEF: GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        NODE_TYPE: GeneratedMessageV3,
        ATTR_DEF : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum>(mappingNamesToPerform: Map<String, String>,
                                                                          transformerArgs: Map<String, List<OpNamespace.ArgDescriptor>>):
    BaseAttributeExtractionRule<GRAPH_DEF, OP_DEF_TYPE, NODE_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, TENSOR_TYPE, DATA_TYPE>
        (name = "valuemapping", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {
    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType != AttributeValueType.TENSOR
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return !argDescriptorType.containsAll(listOf(
            OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR,
            OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR, OpNamespace.ArgDescriptor.ArgType.DATA_TYPE
        ))
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for((k, v) in mappingNamesToPerform()) {
            val descriptorBuilder = OpNamespace.ArgDescriptor.newBuilder()
            descriptorBuilder.name = k
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when(irAttribute.attributeValueType()) {
                AttributeValueType.INT -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.INT64
                    descriptorBuilder.int64Value = irAttribute.intValue()
                    descriptorBuilder.argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                    )

                }

                AttributeValueType.FLOAT -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                    //DO NOT REMOVE work around for numerical underflow that happens at the JVM level, this does a safe cast allowing us to get the real value out
                    val realValue = Nd4j.scalar(irAttribute.floatValue()).castTo(DataType.DOUBLE)
                    descriptorBuilder.doubleValue =  realValue.getDouble(0)
                    descriptorBuilder.argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                    )

                }

                AttributeValueType.BOOL -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                    descriptorBuilder.boolValue = irAttribute.boolValue()
                    descriptorBuilder.argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.BOOL
                    )
                }

                AttributeValueType.STRING -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.STRING
                    descriptorBuilder.stringValue = irAttribute.stringValue()
                    descriptorBuilder.argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.STRING
                    )
                }

                AttributeValueType.DATA_TYPE -> {
                    descriptorBuilder.argType = OpNamespace.ArgDescriptor.ArgType.DATA_TYPE
                    descriptorBuilder.dataTypeValue = irAttribute.dataTataTypeValue().nameSpaceDataType()
                    descriptorBuilder.argIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = OpNamespace.ArgDescriptor.ArgType.DATA_TYPE
                    )
                }

                else -> {
                    throw IllegalArgumentException("Unable to map value $k. Please use different rule for list values and tensors.")
                }
            }


            ret.add(descriptorBuilder.build())

        }
        return ret
    }
}