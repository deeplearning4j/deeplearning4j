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

import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.samediff.frameworkimport.nameSpaceTensorFromNDarray
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class AttributeNumberListNDArray<
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
        name = "convertinputnumberlisttondarray",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {


    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.LIST_FLOAT ||
                argDescriptorType == AttributeValueType.LIST_INT
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR)
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for ((k, v) in mappingNamesToPerform()) {
            val irAttribute = mappingCtx.irAttributeValueForNode(v)
            when (irAttribute.attributeValueType()) {
                AttributeValueType.LIST_FLOAT -> {
                    val listArr = irAttribute.listFloatValue().toFloatArray()
                    val ndarray = Nd4j.create(listArr)
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(ndarray)
                        argIndex = lookupIndexForArgDescriptor(
                            argDescriptorName = k,
                            opDescriptorName = mappingCtx.nd4jOpName(),
                            argDescriptorType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                        )
                    })
                }

                AttributeValueType.LIST_INT -> {
                    val intArr = irAttribute.listIntValue().toLongArray()
                    val strides = Nd4j.getStrides(1, 4).toList().map { it.toLong() }.toLongArray()
                    val ndarray =
                        Nd4j.create(intArr, longArrayOf(1, intArr.size.toLong()), strides, 'c', DataType.INT64)
                    ret.add(ArgDescriptor {
                        argType = OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
                        name = k
                        inputValue = nameSpaceTensorFromNDarray(ndarray)
                        argIndex = lookupIndexForArgDescriptor(
                            argDescriptorName = k,
                            opDescriptorName = mappingCtx.nd4jOpName(),
                            argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                        )
                    })
                }

            }

        }

        return ret
    }
}