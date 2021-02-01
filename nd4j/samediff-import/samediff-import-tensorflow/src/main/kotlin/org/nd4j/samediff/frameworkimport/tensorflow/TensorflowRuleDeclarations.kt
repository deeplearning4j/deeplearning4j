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
package org.nd4j.samediff.frameworkimport.tensorflow

import org.nd4j.common.util.ArrayUtil
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.tensorflow.rule.attribute.*
import org.tensorflow.framework.DataType
import org.tensorflow.framework.TensorProto


fun convertNDArrayToTensorflowTensor(arrayToConvert: INDArray): TensorProto {
    if(arrayToConvert.data() == null)
        return TensorProto.getDefaultInstance()
    when(arrayToConvert.dataType()) {
        org.nd4j.linalg.api.buffer.DataType.FLOAT -> {
            return TensorProto {
                FloatData(arrayToConvert.data().asFloat().toList())
                Shape(arrayToConvert.shape().toList())
                dtype = DataType.DT_FLOAT
            }
        }
        org.nd4j.linalg.api.buffer.DataType.INT32 -> {
            return  TensorProto {
                Int32Data(arrayToConvert.data().asInt().toList())
                Shape(arrayToConvert.shape().toList())
                dtype = DataType.DT_INT32
            }
        }
        org.nd4j.linalg.api.buffer.DataType.INT64 -> {
            return  TensorProto {
                Int64Data(arrayToConvert.data().asLong().toList())
                Shape(arrayToConvert.shape().toList())
                dtype = DataType.DT_INT64
            }
        }
        org.nd4j.linalg.api.buffer.DataType.DOUBLE -> {
            return  TensorProto {
                DoubleData(arrayToConvert.data().asDouble().toList())
                Shape(arrayToConvert.shape().toList())
                dtype = DataType.DT_DOUBLE
            }
        }
        org.nd4j.linalg.api.buffer.DataType.UTF8 -> {
            return  TensorProto {
                val totalLength = ArrayUtil.prod(*arrayToConvert.shape())
                val stringList = ArrayList<String>()
                for(i in 0 until totalLength) {
                    val currString = arrayToConvert.getString(i.toLong())
                    stringList.add(currString)
                }

                StringData(stringList)
                Shape(arrayToConvert.shape().toList())
                dtype = DataType.DT_STRING
            }
        }

        else -> {
            return  TensorProto {
                dtype = convertNd4jDataTypeToTensorflow(arrayToConvert.dataType())
                RawData(arrayToConvert.data().asBytes())
                Shape(arrayToConvert.shape().toList())

            }
        }

    }
}

fun convertNd4jDataTypeToTensorflow(dataType: org.nd4j.linalg.api.buffer.DataType) : DataType {
    when(dataType) {
        org.nd4j.linalg.api.buffer.DataType.DOUBLE -> return DataType.DT_DOUBLE
        org.nd4j.linalg.api.buffer.DataType.FLOAT16 -> return DataType.DT_HALF
        org.nd4j.linalg.api.buffer.DataType.FLOAT -> return DataType.DT_FLOAT
        org.nd4j.linalg.api.buffer.DataType.INT32 -> return DataType.DT_INT32
        org.nd4j.linalg.api.buffer.DataType.UINT32 -> return DataType.DT_UINT32
        org.nd4j.linalg.api.buffer.DataType.INT64 -> return DataType.DT_INT64
        org.nd4j.linalg.api.buffer.DataType.UINT64 -> return DataType.DT_UINT64
        org.nd4j.linalg.api.buffer.DataType.BOOL -> return DataType.DT_BOOL
        org.nd4j.linalg.api.buffer.DataType.INT8 -> return DataType.DT_INT8
        org.nd4j.linalg.api.buffer.DataType.INT16 -> return DataType.DT_INT16
        org.nd4j.linalg.api.buffer.DataType.BFLOAT16 -> return DataType.DT_BFLOAT16
        org.nd4j.linalg.api.buffer.DataType.UTF8 -> return DataType.DT_STRING
        else -> {
            return DataType.UNRECOGNIZED
        }
    }
}

fun conditionalFieldValueIntIndexNDArrayRule(outputAttribute: String,
                                             inputFrameworkStringNameToTest: String,
                                             targetValue: String,
                                             trueIndex: Int,
                                             falseIndex: Int,
                                             attributeNameOfListAttribute: String,
                                             argumentIndex: Int): TensorflowConditionalFieldValueIntIndexNDArrayRule {
    return TensorflowConditionalFieldValueIntIndexNDArrayRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkStringNameToTest),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = "targetValue"
                stringValue = targetValue
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "trueIndex"
                int64Value = trueIndex.toLong()
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "falseIndex"
                int64Value = falseIndex.toLong()
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "attributeNameOfListAttribute"
                stringValue = attributeNameOfListAttribute
                argIndex = argumentIndex
            }))
    )
}


fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkStringNameToTest: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int,
                                           attributeNameOfListAttribute: String,
                                           argumentIndex: Int): TensorflowConditionalFieldValueIntIndexArrayRule {
    return TensorflowConditionalFieldValueIntIndexArrayRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkStringNameToTest),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = "targetValue"
                stringValue = targetValue
                argIndex = argIndex
            },
            ArgDescriptor {
                name = "trueIndex"
                int64Value = trueIndex.toLong()
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "falseIndex"
                int64Value = falseIndex.toLong()
                argIndex = argumentIndex
            },
            ArgDescriptor {
                name = "attributeNameOfListAttribute"
                stringValue = attributeNameOfListAttribute
                argIndex = argumentIndex
            }))
    )
}

fun sizeAtRule(dimensionIndex: Int,
               outputAttributeName: String,
               inputFrameworkAttributeName: String,
               argumentIndex: Int): TensorflowNDArraySizeAt {
    return TensorflowNDArraySizeAt(
        mappingNamesToPerform = mapOf(outputAttributeName to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttributeName to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            int64Value = dimensionIndex.toLong()
            argIndex = argumentIndex
        }.build()))
    )
}

fun ndarrayExtractScalarValue(outputAttribute: String,
                              inputFrameworkAttributeName: String,
                              argumentIndex: Int,
                              scalarIndex: Int): TensorflowNDArrayExtractScalarValue {
    return TensorflowNDArrayExtractScalarValue(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = outputAttribute
                int64Value = scalarIndex.toLong()
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                argIndex = argumentIndex
            })))
}


fun stringEqualsRule(outputAttribute: String,
                     inputFrameworkAttributeName: String,
                     valueToTest: String,
                     argumentIndex: Int): TensorflowStringEqualsAdapterRule {
    return TensorflowStringEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = inputFrameworkAttributeName
                stringValue = valueToTest
                argType = OpNamespace.ArgDescriptor.ArgType.STRING
                argIndex = argumentIndex
            })))
}


fun stringNotEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String,argumentIndex: Int): TensorflowStringNotEqualsAdapterRule {
    return TensorflowStringNotEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
            argIndex = argumentIndex
        }.build())))
}


fun stringContainsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String): TensorflowStringContainsAdapterRule {
    return TensorflowStringContainsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(OpNamespace.ArgDescriptor.newBuilder().apply {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
        }.build())))
}


fun attributeScalarToNDArrayInput(outputAttribute: String, inputFrameworkAttributeName: String): TensorflowAttributeScalarNDArrayAttribute {
    return TensorflowAttributeScalarNDArrayAttribute(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName))
}


fun valueMapping(mappings: Map<String,String>): TensorflowValueMappingRule {
    return TensorflowValueMappingRule(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}

fun invertBooleanNumber(mappings: Map<String,String>): TensorflowInvertBooleanNumber {
    return TensorflowInvertBooleanNumber(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


fun ndarrayToIntList(ndarrayNameToAttributeName: MutableMap<String,String>): TensorflowNDArrayToIntAttributeValue {
    return TensorflowNDArrayToIntAttributeValue(mappingNamesToPerform = ndarrayNameToAttributeName)
}

fun ndarrayStringToIndex(outputAttributeValue: String,inputAttributeValue: String, listOfValues: List<String>,argumentIndex: Int): TensorflowNdArrayToStringIndex {
    return TensorflowNdArrayToStringIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = mapOf(outputAttributeValue to listOfValues.map {
            valueName -> ArgDescriptor {
        name = valueName
        stringValue = valueName
        argIndex = argumentIndex
    }
    }))
}


fun mapStringToInt(outputAttributeValue: String, inputAttributeValue: String, mapOfValuesToInts: Map<String,Int>,argumentIndex: Int,lookupIndex:Int): TensorflowMapStringToInt {
    return TensorflowMapStringToInt(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs =
    mapOf(outputAttributeValue to mapOfValuesToInts.map {
            entry -> ArgDescriptor {
        name = entry.key
        int64Value = entry.value.toLong()
        argIndex = argumentIndex
    }
    },"index" to listOf(ArgDescriptor {
        name = "index"
        int64Value = lookupIndex.toLong()
    })))
}


fun listNumberToListNumber(outputAttributeValue: String, inputAttributeValue: String): TensorflowListNumberToListNumber {
    return TensorflowListNumberToListNumber(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}

fun convertStringToInputNDArray(mappings: Map<String,String>): TensorflowStringAttributeToNDArray {
    return TensorflowStringAttributeToNDArray(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


fun convertNumberListToInputNDArray(outputAttributeValue: String, inputAttributeValue: String): TensorflowAttributeNumberListNDArray {
    return TensorflowAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun listAttributeValueLookupToIndex(outputAttributeValue: String, inputAttributeValue: String, idx: Int,argumentIndex: Int,defaultValueIfNotFound: OpNamespace.ArgDescriptor? = null): TensorflowListAttributeValueLookupToIndex {
   if(defaultValueIfNotFound != null)
    return TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
        transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            int64Value = idx.toLong()
            name = "index"
            argIndex = argumentIndex
        },defaultValueIfNotFound!!)))
    else
       return TensorflowListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
           transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
               argType = OpNamespace.ArgDescriptor.ArgType.INT64
               int64Value = idx.toLong()
               name = "index"
               argIndex = argumentIndex
           })))
}


fun dataTypeToInt(mutableMap: MutableMap<String,String>): TensorflowDataTypeToInt {
    return TensorflowDataTypeToInt(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


fun convertNDArrayInputToNumericalAttr(mutableMap: MutableMap<String,String>): TensorflowNDArrayInputToNumericalAttribute {
    return TensorflowNDArrayInputToNumericalAttribute(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}

fun listNumberToNDarray(mutableMap: MutableMap<String,String>): TensorflowListNumberToNDArray {
    return TensorflowListNumberToNDArray(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


fun ndArrayAttributeToNDarrayInput(mutableMap: MutableMap<String,String>): TensorflowNDArrayAttributeToNDArrayInput {
    return TensorflowNDArrayAttributeToNDArrayInput(mappingNamesToPerform = mutableMap,transformerArgs = emptyMap())
}


fun argDescriptorConstant(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): TensorflowArgDescriptorConstant {
    return TensorflowArgDescriptorConstant(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}


fun ndarrayAttributeToScalarAttribute(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): TensorflowAttributeNDArrayToScalarAttribute {
    return TensorflowAttributeNDArrayToScalarAttribute(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}