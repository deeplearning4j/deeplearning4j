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

abstract class MapStringToInt<
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
        (name = "mapstringtoindex", mappingNamesToPerform = mappingNamesToPerform, transformerArgs = transformerArgs) {

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.LIST_STRING
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INT64)
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        val indexOfValue = transformerArgs["index"]!![0].int64Value
        for ((k, v) in mappingNamesToPerform()) {

            val stringVal = mappingCtx.irAttributeValueForNode(v).listStringValue()[indexOfValue.toInt()]
            val activationInt = (transformerArgs[k] ?: error("Unable to map value $v to a type string for op name ${mappingCtx.nd4jOpName()} and input op name ${mappingCtx.opName()}"))
                .filter {argDescriptor -> argDescriptor.name == stringVal }
                .map { argDescriptor -> argDescriptor.int64Value }.first()
            val argDescriptor = ArgDescriptor {
                name = k
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                argIndex = lookupIndexForArgDescriptor(
                    argDescriptorName = k,
                    opDescriptorName = mappingCtx.nd4jOpName(),
                    argDescriptorType = OpNamespace.ArgDescriptor.ArgType.INT64
                )
                int64Value = activationInt
            }

            ret.add(argDescriptor)

        }

        return ret
    }
}