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

import org.nd4j.common.util.ArrayUtil
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class FlattenDims<
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
        name = "flattendims",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.LIST_INT ||
                argDescriptorType == AttributeValueType.TENSOR
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INT64) || argDescriptorType.contains(
            OpNamespace.ArgDescriptor.ArgType.INT32)
    }

    override fun convertAttributes(
        mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF,
                ATTR_VALUE_TYPE, DATA_TYPE>
    ): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        for ((k, v) in mappingNamesToPerform()) {
            val attr = mappingCtx.irAttributeValueForNode(v)
            val transformerArgs = transformerArgs[k]
            val baseIndex  = lookupIndexForArgDescriptor(
                argDescriptorName = k,
                opDescriptorName = mappingCtx.nd4jOpName(),
                argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
            )

            val axis = transformerArgs!![0]!!.int64Value

            when(attr.attributeValueType()) {
                AttributeValueType.TENSOR -> {
                    val inputTensor = mappingCtx.tensorInputFor(v)
                    val asAxisList = inputTensor.toNd4jNDArray().toLongVector().toMutableList()
                    addToList(ret,k,baseIndex,axis,asAxisList)

                }

                AttributeValueType.LIST_INT -> {
                    val axis = transformerArgs!![0]!!.int64Value
                    val axisList = mappingCtx.irAttributeValueForNode(v).listIntValue()
                    addToList(ret,k,baseIndex,axis,axisList)
                }
            }

        }

        return ret
    }

    fun addToList(ret: MutableList<OpNamespace.ArgDescriptor>,k: String,baseIndex: Int,axis: Long,axisList: List<Long>) {
        val beforeAccessProdValue = if(axis.toInt() == 0) 1L else ArrayUtil.prodLong(axisList.subList(0,axis.toInt()))
        val prodValue = ArrayUtil.prodLong(axisList.subList(axis.toInt(),axisList.size - 1))

        ret.add(ArgDescriptor {
            int64Value = beforeAccessProdValue
            argIndex = baseIndex
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            name = k
        })

        ret.add(ArgDescriptor {
            int64Value = prodValue
            argIndex = baseIndex + 1
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            name = k
        })
    }

}