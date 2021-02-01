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
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.lookupIndexForArgDescriptor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

abstract class NDArrayInputToNumericalAttribute<
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
        name = "ndarrayinputtonumericalattribute",
        mappingNamesToPerform = mappingNamesToPerform,
        transformerArgs = transformerArgs
    ) {

    override fun acceptsInputType(argDescriptorType: AttributeValueType): Boolean {
        return argDescriptorType == AttributeValueType.TENSOR
    }

    override fun outputsType(argDescriptorType: List<OpNamespace.ArgDescriptor.ArgType>): Boolean {
        return argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.DOUBLE)
                || argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.INT64) ||
                argDescriptorType.contains(OpNamespace.ArgDescriptor.ArgType.FLOAT)
    }

    override fun convertAttributes(mappingCtx: MappingContext<GRAPH_DEF, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF, ATTR_VALUE_TYPE, DATA_TYPE>): List<OpNamespace.ArgDescriptor> {
        val ret = ArrayList<OpNamespace.ArgDescriptor>()
        val realDescriptor =  OpDescriptorLoaderHolder.nd4jOpDescriptor.findOp(mappingCtx.nd4jOpName())
        for ((k, v) in mappingNamesToPerform()) {
            val inputTensor = mappingCtx.tensorInputFor(v).toNd4jNDArray()
            realDescriptor.argDescriptorList.filter { argDescriptor -> argDescriptor.name == k &&
                    argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.INT64 && argDescriptor.name == k ||
                    argDescriptor.argType == OpNamespace.ArgDescriptor.ArgType.DOUBLE && argDescriptor.name == k}
                .forEach { argDescriptor ->
                    val baseIndex = lookupIndexForArgDescriptor(
                        argDescriptorName = k,
                        opDescriptorName = mappingCtx.nd4jOpName(),
                        argDescriptorType = argDescriptor.argType
                    )
                    for (i in 0 until 1) {
                        val nameToUse = if (i > 0) k + "$i" else k
                        val get = if(inputTensor.length() > 0) inputTensor.getDouble(i) else 0.0
                        when (argDescriptor.argType) {
                            OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                                ret.add(ArgDescriptor {
                                    name = nameToUse
                                    argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
                                    doubleValue = get
                                    argIndex = baseIndex + i
                                })
                            }

                            OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                                ret.add(ArgDescriptor {
                                    name = nameToUse
                                    argType = OpNamespace.ArgDescriptor.ArgType.INT64
                                    int64Value = get.toLong()
                                    argIndex = baseIndex + i
                                })
                            }
                        }

                    }
                }

        }

        return ret
    }
}