package org.nd4j.codegen.ir.onnx

import org.junit.jupiter.api.Test
import org.nd4j.codegen.ir.ArgDescriptor
import org.nd4j.codegen.ir.onnx.attributeScalarToNDArrayInput
import org.nd4j.codegen.ir.onnx.conditionalFieldValueIntIndexArrayRule
import org.nd4j.codegen.ir.onnx.convertNDArrayInputToScalarAttr
import org.nd4j.ir.TensorNamespace
import org.nd4j.shade.protobuf.ByteString
import java.nio.charset.Charset
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TestOnnxRuleDeclarations {

  /*  @Test
    fun testArgConstant() {
        val opDef = onnxops.first { it.name == "Dilation2D" }
        val intItems = listOf(2,1,1,1)
        val valueNodeDef = NodeProto {
            opType = "Dilation2D"
            name = "inputs"
            AttributeProto {
                
            }
        }

        val shape = listOf(1,1).map { it.toLong() }
        val valueNodeDef2 = NodeProto {
            opType = "Constant"
            name = "inputs"
            AttributeProto {
                
            }
            Attribute(name = "value",value =  AttributeProto {
                tensor = TensorProto {
                    Shape(shape)
                    DoubleData(listOf(1.0))
                }
            })
        }



        val graphDef = GraphProto {
            Node(valueNodeDef)
            Node(valueNodeDef2)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val convertNumberListToInputNDArrayRule = org.nd4j.codegen.ir.tensorflow.argDescriptorConstant(listOf(ArgDescriptor {
            name = "value"
            int32Value = 1
        }))

        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)

        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(1,convertNumberListToInputNDArrayResult[0].int32Value)
    }



    @Test
    fun testConvertNDArrayInputToScalarAttr() {
        val opDef = onnxops.findOp("Dilation2D")
        val intItems = listOf(2,1,1,1)
        val valueNodeDef = NodeProto {
            op = "Dilation2D"
            name = "inputs"
            Attribute(name = "strides",value =  AttributeProto {
                list = ListValue {
                    IntItems(intItems)
                }
            })
        }

        val shape = listOf(1,1).map { it.toLong() }
        val valueNodeDef2 = NodeProto {
            op = "Constant"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                tensor = TensorProto {
                    Shape(shape)
                    DoubleData(listOf(1.0))
                }
            })
        }



        val graphDef = GraphProto {
            Node(valueNodeDef)
            Node(valueNodeDef2)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val convertNumberListToInputNDArrayRule = convertNDArrayInputToScalarAttr(mutableMapOf("output" to "inputs "))
        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)
        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(2,convertNumberListToInputNDArrayResult[0].int64Value)
    }

    @Test
    fun testListAttributeValueLookupToIndex() {
        val opDef = onnxops.findOp("Dilation2D")
        val intItems = listOf(2,1,1,1)
        val valueNodeDef = NodeDef {
            op = "Dilation2D"
            name = "inputs"
            Attribute(name = "strides",value =  AttrValue {
                list = ListValue {
                    IntItems(intItems)
                }
            })
        }


        val graphDef = GraphDef {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef, onnxops)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val convertNumberListToInputNDArrayRule = listAttributeValueLookupToIndex(outputAttributeValue = "output", inputAttributeValue = "strides", idx = 0)
        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)
        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(2,convertNumberListToInputNDArrayResult[0].int64Value)
    }


    @Test
    fun testConvertNumberListToInputNDArray() {
        val opDef = onnxops.findOp("Dilation2D")
        val intItems = listOf(1,1,1,1)
        val valueNodeDef = NodeProto {
            op = "Dilation2D"
            name = "inputs"
            Attribute(name = "strides",value =  AttrValue {
                list = ListValue {
                    IntItems(intItems)
                }
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val convertNumberListToInputNDArrayRule = convertNumberListToInputNDArray(outputAttributeValue = "output", inputAttributeValue = "strides")
        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)
        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        val inputVal = convertNumberListToInputNDArrayResult[0].inputValue
        assertEquals(2,inputVal.dimsCount)
        val testList = inputVal.int64DataList
        testList.forEach {
            assertEquals(1,it)
        }
    }

    @Test
    fun testValueMapping() {
        val opDef = onnxops.findOp("CudnnRNN")
        val valueNodeDef = NodeProto {
            op = "CudnnRNN"
            name = "inputs"
            Attribute(name = "is_training",value =  AttrValue {
                b = true
            })
            Attribute(name = "seed",value =  AttrValue {
                i = 1
            })
            Attribute(name = "dropout",value =  AttrValue {
                f = 1.0f
            })
            Attribute(name = "direction",value =  AttrValue {
                s = ByteString.copyFrom("unidirectional".toByteArray(Charset.defaultCharset()))
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val booleanToInt = valueMapping(mapOf("output" to "is_training","output2" to "seed","output3" to "dropout","output4" to "direction"))
        val booleanToIntResult = booleanToInt.convertAttributes(mappingContext)
        assertEquals(4,booleanToIntResult.size)
        val boolValue = booleanToIntResult.first { it.name == "output" }.boolValue
        val intValue = booleanToIntResult.first {it.name == "output2" }.int64Value
        val floatValue = booleanToIntResult.first {it.name == "output3"}.floatValue
        val stringVal = booleanToIntResult.first {it.name == "output4" }.stringValue
        assertEquals(true,boolValue)
        assertEquals(1,intValue)
        assertEquals(1.0f,floatValue)
        assertEquals("unidirectional",stringVal)
    }

    @Test
    fun testBooleanToInt() {
        val opDef = onnxops.findOp("CudnnRNN")
        val valueNodeDef = NodeProto {
            op = "CudnnRNN"
            name = "inputs"
            Attribute(name = "is_training",value =  AttrValue {
                b = true
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef, onnxops)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val booleanToInt = org.nd4j.codegen.ir.tensorflow.booleanToInt(mapOf("output" to "is_training"))
        val booleanToIntResult = booleanToInt.convertAttributes(mappingContext)
        assertEquals(1,booleanToIntResult.size)
        val boolValue = booleanToIntResult[0].int64Value
        assertEquals(1,boolValue)
    }

    @Test
    fun testAttributeScalarToNDArrayInputRuleDouble() {
        val opDef = onnxops.findOp("CudnnRNN")
        val valueNodeDef = NodeProto {
            op = "CudnnRNN"
            name = "inputs"
            Attribute(name = "dropout",value =  AttrValue {
                f = 1.0f
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val ndarrScalarRule = attributeScalarToNDArrayInput(outputAttribute = "output",inputFrameworkAttributeName = "dropout")
        val ndarrScalarRuleResult = ndarrScalarRule.convertAttributes(mappingContext)
        assertEquals(1,ndarrScalarRuleResult.size)
        assertTrue {ndarrScalarRuleResult[0].hasInputValue()}
        val tensorValue = ndarrScalarRuleResult[0].inputValue
        assertEquals(2,tensorValue.dimsCount)
        assertEquals(TensorNamespace.DataType.FLOAT.ordinal,tensorValue.dataType)
        val floatValue = tensorValue.floatDataList[0]
        assertEquals(1.0f,floatValue)
    }

    @Test
    fun testAttributeScalarToNDArrayInputRuleInt() {
        val opDef = onnxops.findOp("CountUpTo")
        val valueNodeDef = NodeProto {
            op = "CountUpTo"
            name = "inputs"
            Attribute(name = "limit",value =  AttrValue {
                i = 1
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        val ndarrScalarRule = attributeScalarToNDArrayInput(outputAttribute = "output",inputFrameworkAttributeName = "limit")
        val ndarrScalarRuleResult = ndarrScalarRule.convertAttributes(mappingContext)
        assertEquals(1,ndarrScalarRuleResult.size)
        assertTrue {ndarrScalarRuleResult[0].hasInputValue()}
        val tensorValue = ndarrScalarRuleResult[0].inputValue
        assertEquals(2,tensorValue.dimsCount)
        assertEquals(TensorNamespace.DataType.INT64.ordinal,tensorValue.dataType)
        val intValue = tensorValue.int64DataList[0]
        assertEquals(1,intValue)
    }

    @Test
    fun testStringNotEqualsRule() {
        val opDef = onnxops.findOp("Const")
        val valueNodeDef = NodeProto {
            op = "Const"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                s = ByteString.copyFrom("value".toByteArray(Charset.defaultCharset()))
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)
        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        listOf("value","notValue").zip(listOf(false,true)).forEach { (valueToTest,assertionResult) ->
            val stringNotEqualsRule = org.nd4j.codegen.ir.tensorflow.stringNotEqualsRule(outputAttribute = "output", inputFrameworkAttributeName = "value", valueToTest = valueToTest)
            val stringEqualsResult = stringNotEqualsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testStringContainsRule() {
        val opDef = onnxops.findOp("Const")
        val valueNodeDef = NodeProto {
            op = "Const"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                s = ByteString.copyFrom("value".toByteArray(Charset.defaultCharset()))
            })
        }


        val graphDef = GraphProto {
            Node(valueNodeDef)

        }

        val tfGraph = OnnxIRGraph(graphDef)
        val mappingContext = OnnxMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        listOf("value","notValue").zip(listOf(true,false)).forEach { (valueToTest,assertionResult) ->
            val stringContainsRule = org.nd4j.codegen.ir.tensorflow.stringContainsRule(outputAttribute = "output", inputFrameworkAttributeName = "value", valueToTest = valueToTest)
            val stringEqualsResult = stringContainsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testStringEqualsRule() {
        val opDef = onnxops.findOp("Const")
        val valueNodeDef = NodeDef {
            op = "Const"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                s = ByteString.copyFrom("value".toByteArray(Charset.defaultCharset()))
            })
        }


        val graphDef = GraphDef {
            Node(valueNodeDef)

        }

        val tfGraph = TensorflowIRGraph(graphDef, onnxops)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph)
        listOf("value","notValue").zip(listOf(true,false)).forEach { (valueToTest,assertionResult) ->
            val stringEqualsRule = org.nd4j.codegen.ir.tensorflow.stringEqualsRule(outputAttribute = "output", inputFrameworkAttributeName = "value", valueToTest = valueToTest)
            val stringEqualsResult = stringEqualsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testNDArraySizeAtRule() {
        val opDef = onnxops.findOp("AddN")
        val nodeDef = NodeDef {
            op = "AddN"
            Input("inputs")
            Input("y")
            name = "test"
        }

        val shape = listOf(1,2).map { it.toLong() }

        val valueNodeDef = NodeDef {
            op = "Constant"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                tensor = TensorProto {
                    Shape(shape)
                    DoubleData(listOf(1.0,2.0))
                }
            })
        }


        val graphDef = GraphDef {
            Node(nodeDef)
            Node(valueNodeDef)

        }

        val tfGraph = TensorflowIRGraph(graphDef, onnxops)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)
        shape.forEachIndexed { i,value ->
            val sizeAtRule = org.nd4j.codegen.ir.tensorflow.sizeAtRule(dimensionIndex = i, outputAttributeName = "output", inputFrameworkAttributeName = "inputs")
            val sizeAtRuleResult = sizeAtRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,sizeAtRuleResult.size)
            assertEquals(value,sizeAtRuleResult[0].int64Value)

        }

    }


    @Test
    fun testConditionalIndex() {

        val opDef = onnxops.findOp("AddN")
        val strings = listOf("value","falseValue")
        //when item is equal to value return element at index 0
        //when item is not equal to value return element at index 1
        val assertionValue = mapOf("value" to 1,"falseValue" to 0)
        val trueIndex = 0
        val falseIndex = 1
        val listOfItemsForTesting = listOf(1,0,2,3)
        //true and false case with index 1
        for(string in strings) {
            val nodeDef = NodeDef {
                op = "AddN"
                Input("inputs")
                Input("y")
                name = "test"
                Attribute(name = "N",value = AttrValue {
                    name = "N"
                    list = ListValue {
                        IntItems(listOfItemsForTesting)
                    }
                })
                Attribute(name = "T",value = AttrValue {
                    name = "T"
                    s = ByteString.copyFrom(string.toByteArray(Charset.defaultCharset()))
                })
            }

            val graphDef = GraphDef {
                Node(nodeDef)
            }

            val tfGraph = TensorflowIRGraph(graphDef, onnxops)


            val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph)

            val conditionalIndex = conditionalFieldValueIntIndexArrayRule(
                    outputAttribute = "N",
                    attributeNameOfListAttribute = "N",
                    targetValue = "value", trueIndex = trueIndex, falseIndex = falseIndex,
                    inputFrameworkStringNameToTest = "T")

            val ret = conditionalIndex.convertAttributes(mappingContext)
            assertEquals(1,ret.size)
            assertEquals((assertionValue[string] ?:
            error("No value found with string value $string")).toLong(),ret[0].int64Value)
            assertEquals("N",ret[0].name)

        }

    }*/
}