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
package org.nd4j.samediff.frameworkimport.runner

import org.nd4j.autodiff.functions.DifferentialFunction
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.common.io.ReflectionUtils
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.Op
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.context.MappingContext
import org.nd4j.samediff.frameworkimport.convertNd4jDataTypeFromNameSpaceTensorDataType
import org.nd4j.samediff.frameworkimport.ndarrayFromNameSpaceTensor
import org.nd4j.samediff.frameworkimport.setNameForFunctionFromDescriptors
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.lang.IllegalArgumentException
import java.lang.reflect.Modifier

class DefaultImportRunner<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE : GeneratedMessageV3,
        TENSOR_TYPE : GeneratedMessageV3,
        ATTR_DEF_TYPE : GeneratedMessageV3,
        ATTR_VALUE_TYPE : GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum> : ImportRunner<GRAPH_TYPE,
        NODE_TYPE,
        OP_DEF_TYPE,
        TENSOR_TYPE,
        ATTR_DEF_TYPE,
        ATTR_VALUE_TYPE,
        DATA_TYPE> {
    override fun <GRAPH_TYPE : GeneratedMessageV3, NODE_TYPE : GeneratedMessageV3, OP_DEF_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTR_DEF_TYPE : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum> initAttributes(
        df: DifferentialFunction,
        sd: SameDiff,
        descriptorAndContext: Pair<MappingContext<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE, DATA_TYPE>, OpNamespace.OpDescriptor>
    ) {

        val applied = descriptorAndContext
        val mappingContext = applied.first
        when (df.opType()) {
            Op.Type.CUSTOM,Op.Type.LOGIC -> {
                val dynamicCustomOp = df as DynamicCustomOp
                val grouped = descriptorAndContext.second.argDescriptorList.groupBy { descriptor ->
                    descriptor.argType
                }

                val sortedMap = HashMap<OpNamespace.ArgDescriptor.ArgType, List<OpNamespace.ArgDescriptor>>()
                grouped.forEach { (argType, list) ->
                    sortedMap[argType] = list.sortedBy { arg -> arg.argIndex }
                }

                sortedMap.forEach { (argType, listOfArgsSortedByIndex) ->
                    when (argType) {
                        OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR -> {
                            if(df.opType() != Op.Type.LOGIC) {
                                val args = dynamicCustomOp.args()
                                val arraysToAdd = ArrayList<INDArray>()
                                listOfArgsSortedByIndex.forEachIndexed { index, argDescriptor ->
                                    val convertedTensor = ndarrayFromNameSpaceTensor(argDescriptor.inputValue)
                                    if (index < args.size) {
                                        val arg = args[index]
                                        if (arg.variableType != VariableType.ARRAY) {
                                            if (arg.shape == null) {
                                                val emptyLongArray = LongArray(0)
                                                arg.setShape(*emptyLongArray)
                                            }

                                            arraysToAdd.add(convertedTensor)

                                        }
                                    }

                                }

                                //note we don't add arrays one at a time because addInputArgument requires all the input arrays to be added at once
                                //dynamicCustomOp.addInputArgument(*arraysToAdd.toTypedArray())


                            }

                        }

                        OpNamespace.ArgDescriptor.ArgType.INT64, OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                            listOfArgsSortedByIndex.forEach { dynamicCustomOp.addIArgument(it.int64Value) }
                        }

                        OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                            listOfArgsSortedByIndex.forEach { dynamicCustomOp.addTArgument(it.doubleValue) }
                        }

                        OpNamespace.ArgDescriptor.ArgType.OUTPUT_TENSOR -> {
                            listOfArgsSortedByIndex.forEach {
                                val convertedTensor = ndarrayFromNameSpaceTensor(it.inputValue)
                                dynamicCustomOp.addOutputArgument(convertedTensor)
                            }
                        }

                        //allow strings, but only for cases of setting a value in java
                        OpNamespace.ArgDescriptor.ArgType.STRING -> {}

                        OpNamespace.ArgDescriptor.ArgType.BOOL -> {
                            listOfArgsSortedByIndex.forEach {
                                dynamicCustomOp.addBArgument(it.boolValue)
                            }
                        }

                        OpNamespace.ArgDescriptor.ArgType.DATA_TYPE -> {
                            listOfArgsSortedByIndex.forEach {
                                val dtype = convertNd4jDataTypeFromNameSpaceTensorDataType(it.dataTypeValue!!)
                                val dtypeJavaClass = Class.forName("org.nd4j.linalg.api.buffer.DataType")
                                dynamicCustomOp.addDArgument(dtype)
                                df.javaClass.declaredFields.forEach { field ->
                                    if (!Modifier.isStatic(field.modifiers) && !Modifier.isFinal(field.modifiers)
                                        && dtypeJavaClass.isAssignableFrom(field.type)
                                    ) {
                                        field.isAccessible = true
                                        ReflectionUtils.setField(field, df, dtype)
                                    }
                                }
                            }
                        }
                        else -> {
                            throw IllegalArgumentException("Illegal type")
                        }

                    }

                    //set any left over fields if they're found
                    setNameForFunctionFromDescriptors(listOfArgsSortedByIndex, df)
                }


            }
            Op.Type.SCALAR -> {
                applied.second.argDescriptorList.forEach { argDescriptor ->
                    val field = ReflectionUtils.findField(df.javaClass, argDescriptor.name)
                    if (field != null) {
                        field.isAccessible = true
                        when (argDescriptor.name) {
                            "x", "y", "z" -> {
                                val createdNDArray = mappingContext.tensorInputFor(argDescriptor.name).toNd4jNDArray()
                                ReflectionUtils.setField(field, df, createdNDArray)
                            }
                            else -> {
                                val scalarField = ReflectionUtils.findField(df.javaClass, "scalarValue")
                                scalarField.isAccessible = true
                                //access the first input (should have been set) and make sure the scalar type is the
                                //the same
                                val firstValue = sd.variables().first()
                                val dtype = firstValue.dataType()
                                when (argDescriptor.argType) {
                                    OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                                        val nd4jScalarValue = Nd4j.scalar(argDescriptor.doubleValue).castTo(dtype)
                                        ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                    }
                                    OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                                        val nd4jScalarValue = Nd4j.scalar(argDescriptor.floatValue).castTo(dtype)
                                        ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                    }
                                    OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                                        val nd4jScalarValue = Nd4j.scalar(argDescriptor.int32Value).castTo(dtype)
                                        ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                    }
                                    OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                                        val nd4jScalarValue = Nd4j.scalar(argDescriptor.int64Value).castTo(dtype)
                                        ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                    }
                                }
                            }
                        }

                    } else {
                        if (argDescriptor.argType in listOf(
                                OpNamespace.ArgDescriptor.ArgType.INT64,
                                OpNamespace.ArgDescriptor.ArgType.DOUBLE, OpNamespace.ArgDescriptor.ArgType.INT32,
                                OpNamespace.ArgDescriptor.ArgType.FLOAT
                            )
                        ) {
                            val scalarField = ReflectionUtils.findField(df.javaClass, "scalarValue")
                            scalarField.isAccessible = true
                            //access the first input (should have been set) and make sure the scalar type is the
                            //the same
                            val irNode = mappingContext.irNode()
                            val firstValue = sd.getVariable(irNode.inputAt(0))
                            val dtype = firstValue.dataType()
                            when (argDescriptor.argType) {
                                OpNamespace.ArgDescriptor.ArgType.DOUBLE -> {
                                    val nd4jScalarValue = Nd4j.scalar(argDescriptor.doubleValue).castTo(dtype)
                                    ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                }
                                OpNamespace.ArgDescriptor.ArgType.FLOAT -> {
                                    val nd4jScalarValue = Nd4j.scalar(argDescriptor.floatValue).castTo(dtype)
                                    ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                }
                                OpNamespace.ArgDescriptor.ArgType.INT32 -> {
                                    val nd4jScalarValue = Nd4j.scalar(argDescriptor.int32Value).castTo(dtype)
                                    ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                }
                                OpNamespace.ArgDescriptor.ArgType.INT64 -> {
                                    val nd4jScalarValue = Nd4j.scalar(argDescriptor.int64Value).castTo(dtype)
                                    ReflectionUtils.setField(scalarField, df, nd4jScalarValue)

                                }
                            }
                        }
                    }


                }

                //set any left over fields if they're found
                setNameForFunctionFromDescriptors(applied.second.argDescriptorList, df)
            }
            else -> {
                var hasDimensions = false
                applied.second.argDescriptorList.forEach { argDescriptor ->
                    if (argDescriptor.name == "dimensions")
                        hasDimensions = true
                    val field = ReflectionUtils.findField(df.javaClass, argDescriptor.name)
                    if (field != null) {
                        field.isAccessible = true
                        when (argDescriptor.name) {
                            "x", "y", "z" -> {
                                val createdNDArray = mappingContext.tensorInputFor(argDescriptor.name).toNd4jNDArray()
                                ReflectionUtils.setField(field, df, createdNDArray)
                            }
                            "keepDims" -> ReflectionUtils.setField(field, df, argDescriptor.boolValue)
                            else -> {
                            }
                        }
                    }
                }

                if (hasDimensions) {
                    //dimensions sorted by index
                    val dimArgs =
                        applied.second.argDescriptorList.filter { argDescriptor -> argDescriptor.name.contains("dimensions") }
                            .sortedBy { argDescriptor -> argDescriptor.argIndex }
                            .map { argDescriptor -> argDescriptor.int64Value.toInt() }.toIntArray()
                    val dimensionsField = ReflectionUtils.findField(df.javaClass, "dimensions")
                    val dimensionzField = ReflectionUtils.findField(df.javaClass, "dimensionz")
                    if (dimensionsField != null) {
                        dimensionsField.isAccessible = true
                        if (intArrayOf(0).javaClass.isAssignableFrom(dimensionsField.type)) {
                            ReflectionUtils.setField(dimensionsField, df, dimArgs)
                        }
                    }

                    if (dimensionzField != null) {
                        dimensionzField.isAccessible = true
                        if (INDArray::class.java.isAssignableFrom(dimensionzField.type)) {
                            val buffer = Nd4j.createBuffer(dimArgs)
                            val createdArr = Nd4j.create(buffer)
                            ReflectionUtils.setField(dimensionzField, df, createdArr)
                        }
                    }

                }

                //set any left over fields if they're found
                setNameForFunctionFromDescriptors(applied.second.argDescriptorList, df)
            }
        }
    }
}