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

abstract class ListAttributeValueLookupToIndex<
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
        name = "listattributevaluelookuptoindex",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.LIST_FLOAT ||
                argDescriptorType == AttributeValueType.LIST_INT ||
                argDescriptorType == AttributeValueType.LIST_STRING ||
                argDescriptorType == AttributeValueType.LIST_TENSOR ||
                argDescriptorType == AttributeValueType.LIST_BOOL ||
                argDescriptorType == AttributeValueType.INT
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return !argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR)
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for ((k, v) in mappingNamesToPerform()) {
            val index = (transformerArgs[k] ?: error(""))[0]!!.int64Value
            val listOfValues = mappingCtx.irAttributeValueForNode(v)
            when (listOfValues.attributeValueType()) {
                AttributeValueType.LIST_FLOAT -> {
                    val listFloat = listOfValues.listFloatValue()
                    if(!listFloat.isEmpty()) {
                        val argDescriptor = ArgDescriptor {
                            name = k
                            doubleValue = listFloat[index.toInt()].toDouble()
                            argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                            )
                        }

                        ret.add(argDescriptor)
                    } else if(transformerArgs[k]!!.size > 1) {
                        val args = transformerArgs[k]!![1]!!
                        ret.add(args)
                    }

                }
                AttributeValueType.LIST_INT -> {
                    val listInt = listOfValues.listIntValue()
                    if(!listInt.isEmpty()) {
                        val argDescriptor = ArgDescriptor {
                            name = k
                            int64Value = listInt[index.toInt()]
                            argType = OpNamespace.ArgDescriptor.ArgType.INT64
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                            )
                        }

                        ret.add(argDescriptor)
                    } else if(transformerArgs[k]!!.size > 1) {
                        val args = transformerArgs[k]!![1]!!
                        ret.add(args)
                    }

                }

                AttributeValueType.LIST_STRING -> {
                    val listString = listOfValues.listStringValue()
                    if(!listString.isEmpty()) {
                        val argDescriptor = ArgDescriptor {
                            name = k
                            stringValue = listString[index.toInt()]
                            argType = OpNamespace.ArgDescriptor.ArgType.STRING
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.STRING
                            )
                        }

                        ret.add(argDescriptor)
                    } else if(transformerArgs[k]!!.size > 1) {
                        val args = transformerArgs[k]!![1]!!
                        ret.add(args)
                    }

                }

                AttributeValueType.LIST_TENSOR -> {
                    val listTensor = listOfValues.listTensorValue()
                    if(!listTensor.isEmpty()) {
                        val argDescriptor = ArgDescriptor {
                            name = k
                            inputValue = listTensor[index.toInt()].toArgTensor()
                            argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                            )
                        }

                        ret.add(argDescriptor)
                    }  else if(transformerArgs[k]!!.size > 1) {
                        val args = transformerArgs[k]!![1]!!
                        ret.add(args)
                    }

                }

                AttributeValueType.LIST_BOOL -> {
                    val listBool = listOfValues.listBoolValue()
                    if(!listBool.isEmpty()) {
                        val argDescriptor = ArgDescriptor {
                            name = k
                            boolValue = listBool[index.toInt()]
                            argType = OpNamespace.ArgDescriptor.ArgType.BOOL
                            argIndex = lookupIndexForArgDescriptor(
                                argDescriptorName = k,
                                opDescriptorName = mappingCtx.nd4jOpName(),
                                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.BOOL
                            )
                        }

                        ret.add(argDescriptor)
                    } else if(transformerArgs[k]!!.size > 1) {
                        val args = transformerArgs[k]!![1]!!
                        ret.add(args)
                    }

                }

            }


        }

        return ret
    }
}