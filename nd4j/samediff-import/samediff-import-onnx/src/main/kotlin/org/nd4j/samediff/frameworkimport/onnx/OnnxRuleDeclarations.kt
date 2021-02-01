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
package org.nd4j.samediff.frameworkimport.onnx

import onnx.Onnx
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.onnx.definitions.onnxOpRegistry
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcess
import org.nd4j.samediff.frameworkimport.onnx.rule.attribute.*
import org.nd4j.samediff.frameworkimport.onnx.rule.tensor.NDArrayMappingRule
import org.nd4j.samediff.frameworkimport.onnx.rule.tensor.OnnxMultiInputIndexMappingRule
import org.nd4j.samediff.frameworkimport.rule.attribute.AttributeMappingRule

fun mappingNDArrayInputs(inputs: MutableMap<String,String>) : NDArrayMappingRule {
    return NDArrayMappingRule(
        mappingNamesToPerform = inputs)
}


fun mappingListNDArrays(inputs: MutableMap<String,String>) : OnnxMultiInputIndexMappingRule {
    return OnnxMultiInputIndexMappingRule(
        mappingNamesToPerform = inputs)
}

fun conditionalFieldValueIntIndexNDArrayRule(outputAttribute: String,
                                             inputFrameworkAttributeName: String,
                                             targetValue: String,
                                             trueIndex: Int,
                                             falseIndex: Int,
                                             argumentIndex: Int): OnnxConditionalFieldValueIntIndexNDArrayRule {
    return OnnxConditionalFieldValueIntIndexNDArrayRule(
        mappingNamesToPerform = mutableMapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
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
            }))
    )
}


fun ndarrayExtractScalarValue(outputAttribute: String,
                              inputFrameworkAttributeName: String,
                              argumentIndex: Int,
                              scalarIndex: Int) : OnnxNDArrayExtractScalarValue {
    return OnnxNDArrayExtractScalarValue(
        mappingNamesToPerform = mutableMapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = inputFrameworkAttributeName
                int64Value = scalarIndex.toLong()
                argType = OpNamespace.ArgDescriptor.ArgType.INT64
                argIndex = argumentIndex
            })))
}


fun conditionalFieldValueIntIndexArrayRule(outputAttribute: String,
                                           inputFrameworkAttributeName: String,
                                           targetValue: String,
                                           trueIndex: Int,
                                           falseIndex: Int,
                                           argumentIndex: Int): OnnxConditionalFieldValueIntIndexArrayRule {
    return OnnxConditionalFieldValueIntIndexArrayRule(
        mappingNamesToPerform = mutableMapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
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
            }))
    )
}

fun valueMappings(mappings: Map<String,String>): OnnxValueMapping {
    return OnnxValueMapping(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}

fun ndarrayAttributeToNDArrayInput(mappings: Map<String,String>): OnnxNDArrayAttributeToNDArrayInput {
    return OnnxNDArrayAttributeToNDArrayInput(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}

/**
 * This will change a boolean to a number and a number to a boolean
 */
fun invertBooleanNumber(mappings: Map<String,String>): OnnxInvertBooleanNumber {
    return OnnxInvertBooleanNumber(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


fun dataTypeToInt(mappings: Map<String,String>): OnnxDataTypeToInt {
    return OnnxDataTypeToInt(mappingNamesToPerform = mappings,transformerArgs = emptyMap())
}


fun sizeAtRule(dimensionIndex: Int, outputAttributeName: String, inputFrameworkAttributeName: String,argumentIndex: Int): OnnxNDArraySizeAt {
    return OnnxNDArraySizeAt(
        mappingNamesToPerform = mapOf(outputAttributeName to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttributeName to listOf(ArgDescriptor {
            name = inputFrameworkAttributeName
            int64Value = dimensionIndex.toLong()
            argIndex = argumentIndex
        }))
    )
}

fun stringEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String,argumentIndex: Int): OnnxStringEqualsAdapterRule {
    return OnnxStringEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
            argIndex = argumentIndex
        })))
}


fun stringContainsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String,argumentIndex: Int): OnnxStringContainsAdapterRule {
    return OnnxStringContainsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
            argIndex = argumentIndex
        })))
}


fun stringNotEqualsRule(outputAttribute: String, inputFrameworkAttributeName: String, valueToTest: String,argumentIndex: Int): OnnxStringNotEqualsAdapterRule {
    return OnnxStringNotEqualsAdapterRule(
        mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(ArgDescriptor {
            name = inputFrameworkAttributeName
            stringValue = valueToTest
            argIndex = argumentIndex
        })))
}


fun ndarrayToIntList(ndarrayNameToAttributeName: MutableMap<String,String>): OnnxNDArrayToIntAttributeValue {
    return OnnxNDArrayToIntAttributeValue(mappingNamesToPerform = ndarrayNameToAttributeName)
}

fun sizeThreshold(outputAttribute: String, inputFrameworkAttributeName: String, sizeThreshold: Long, index: Long, fallbackIndex: Long,argumentIndex: Int): OnnxSizeThresholdIntArrayIntIndexRule {
    return OnnxSizeThresholdIntArrayIntIndexRule(mappingNamesToPerform = mapOf(outputAttribute to inputFrameworkAttributeName),
        transformerArgs = mapOf(outputAttribute to listOf(
            ArgDescriptor {
                name = "index"
                int64Value = index
                argIndex = argIndex
            },
            ArgDescriptor {
                name = "sizeThreshold"
                int64Value = sizeThreshold
                argIndex = argIndex
            },
            ArgDescriptor {
                name = "fallbackIndex"
                int64Value = fallbackIndex
                argIndex = argumentIndex
            })))
}


fun stringToIndex(outputAttributeValue: String, inputAttributeValue: String, listOfValues: List<String>,argumentIndex: Int): OnnxStringToIndex {
    return OnnxStringToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs =
    mapOf(outputAttributeValue to listOfValues.map {
            valueName -> ArgDescriptor {
        name = outputAttributeValue
        stringValue = valueName
        argIndex = argumentIndex
    }
    }))
}


fun mapStringToInt(outputAttributeValue: String, inputAttributeValue: String, mapOfValuesToInts: Map<String,Int>,argumentIndex: Int,lookupIndex: Int): OnnxMapStringToInt {
    return OnnxMapStringToInt(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs =
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


fun listAttributeValueLookup(outputAttributeValue: String, inputAttributeValue: String, indexValue: Int,argumentIndex: Int,defaultValueIfNotFound: OpNamespace.ArgDescriptor? = null): OnnxListAttributeValueLookupToIndex {
    if(defaultValueIfNotFound != null)
        return OnnxListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
            transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
                name = inputAttributeValue
                int64Value = indexValue.toLong()
                argIndex = argumentIndex
            },defaultValueIfNotFound!!)
            ))
    else
        return OnnxListAttributeValueLookupToIndex(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),
            transformerArgs = mapOf(outputAttributeValue to listOf(ArgDescriptor {
                name = inputAttributeValue
                int64Value = indexValue.toLong()
                argIndex = argumentIndex
            })
            ))
}

fun listNumberToListNumber(outputAttributeValue: String, inputAttributeValue: String): OnnxListNumberToListNumber {
    return OnnxListNumberToListNumber(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun convertStringToNDArray(outputAttributeValue: String, inputAttributeValue: String): OnnxStringAttributeToNDArray {
    return OnnxStringAttributeToNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun convertNumericalListToNDArray(outputAttributeValue: String, inputAttributeValue: String): OnnxAttributeNumberListNDArray {
    return OnnxAttributeNumberListNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}




fun listNumberToNDarray(outputAttributeValue: String, inputAttributeValue: String): OnnxListNumberToNDArray {
    return OnnxListNumberToNDArray(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun convertNDArrayInputToScalarAttr(outputAttributeValue: String, inputAttributeValue: String): OnnxNDArrayInputToNumericalAttribute {
    return OnnxNDArrayInputToNumericalAttribute(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun ndarrayAttributeToScalarAttribute(outputAttributeValue: String, inputAttributeValue: String): OnnxAttributeNDArrayToScalarAttribute {
    return OnnxAttributeNDArrayToScalarAttribute(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}

fun attributeScalarToNDArrayInput(outputAttributeValue: String, inputAttributeValue: String): OnnxAttributeScalarNDArrayAttribute {
    return OnnxAttributeScalarNDArrayAttribute(mappingNamesToPerform = mapOf(outputAttributeValue to inputAttributeValue),transformerArgs = emptyMap())
}


fun argDescriptorConstant(argDescriptorConstants: List<OpNamespace.ArgDescriptor>): OnnxArgDescriptorConstant {
    return OnnxArgDescriptorConstant(mappingNamesToPerform = emptyMap(),transformerArgs = mapOf("value" to argDescriptorConstants))
}


//TODO: TreeEnsembleClassifier
//TODO: TreeEnsembleRegressor
//TODO: Unique PRIORITIZE
//TODO: Unsqueeze PRIORITIZE
//TODO: Upsample PRIORITIZE
//TODO: Where PRIORITIZE
//TODO: ZipMap
fun defOnnxSingleTransform(opName: String, inputFrameworkOpName: String, outputName: String, inputFrameworkInput: String = "input", attributeMappingRules: List<AttributeMappingRule<Onnx.GraphProto, Onnx.NodeProto, Onnx.NodeProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto, Onnx.TensorProto.DataType>> = emptyList()): OnnxMappingProcess {
    return OnnxMappingProcess(
        opName = opName,
        tensorMappingRules = listOf(
            NDArrayMappingRule(mappingNamesToPerform = mutableMapOf(outputName to inputFrameworkInput))
        ),
        inputFrameworkOpName = inputFrameworkOpName,
        inputFramework = "onnx",
        attributeMappingRules = attributeMappingRules,
        opMappingRegistry = onnxOpRegistry
    )
}

fun defineOnnxPairwiseTransforms(opName: String, inputFrameworkOpName: String,
                                 firstOutputName: String = "input",
                                 secondOutputName: String = "y",
                                 firstInput: String = "A", secondInput: String = "B") : OnnxMappingProcess {
    return OnnxMappingProcess(
        opName = opName,
        tensorMappingRules = listOf(
            NDArrayMappingRule(
                mappingNamesToPerform = mutableMapOf(
                    firstOutputName to firstInput,
                    secondOutputName to secondInput
                )
            )
        ),
        inputFrameworkOpName = inputFrameworkOpName,
        inputFramework = "onnx",
        opMappingRegistry = onnxOpRegistry
    )
}

fun defineOnnxSingleTransform(inputOpName: String, inputFrameworkOpName: String): OnnxMappingProcess {
    return OnnxMappingProcess(
        opName = inputOpName,
        inputFrameworkOpName = inputFrameworkOpName, tensorMappingRules = listOf(
            NDArrayMappingRule(
                mappingNamesToPerform = mutableMapOf("input" to "input")
            )
        ),
        attributeMappingRules = booleanConstant(inputName = "inPlace", constantValue = false, argumentIndex = 0),
        opMappingRegistry = onnxOpRegistry
    )

}

fun flattenDims(outputName: String, inputFrameworkName: String, axis: Long, argumentIndex: Int): OnnxFlattenDims {
    return OnnxFlattenDims(
        mutableMapOf(outputName to inputFrameworkName),
        mapOf(outputName to listOf(ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            int64Value = axis
            name = outputName
            argIndex = argumentIndex
        }))
    )
}

fun booleanConstant(inputName: String, constantValue: Boolean, argumentIndex: Int): List<OnnxArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.BOOL
            name = inputName
            argIndex = argumentIndex
            boolValue = constantValue
        }
    )))
}

fun doubleConstant(inputName: String, constantValue: Double, argumentIndex: Int): List<OnnxArgDescriptorConstant> {

    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.DOUBLE
            name = inputName
            argIndex = argumentIndex
            doubleValue = constantValue
        }
    )))
}

fun intConstant(inputName: String, constantValue: Int, argumentIndex: Int): List<OnnxArgDescriptorConstant> {
    return listOf(argDescriptorConstant(listOf(
        ArgDescriptor {
            argType = OpNamespace.ArgDescriptor.ArgType.INT64
            name = inputName
            argIndex = argumentIndex
            int64Value = constantValue.toLong()
        }
    )))
}