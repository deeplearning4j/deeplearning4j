package org.nd4j.samediff.frameworkimport.tensorflow

import org.junit.jupiter.api.Test
import org.nd4j.ir.TensorNamespace
import org.nd4j.samediff.frameworkimport.ArgDescriptor
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.context.TensorflowMappingContext
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph
import org.nd4j.shade.protobuf.ByteString
import org.tensorflow.framework.*
import java.nio.charset.Charset
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class TestTensorflowRuleDeclarations {
    val tensorflowOps =  {
        val input = OpList.newBuilder()
        OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow").values.forEach {
            input.addOp(it)
        }

        input.build()
    }.invoke()

    val tensorflowOpRegistry = OpMappingRegistry<GraphDef, NodeDef,OpDef, TensorProto, DataType,OpDef.AttrDef, AttrValue>("tensorflow",OpDescriptorLoaderHolder.nd4jOpDescriptor)


    @Test
    fun testArgConstant() {
        val opDef = tensorflowOps.findOp("Dilation2D")
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

        val shape = listOf(1,1).map { it.toLong() }
        val valueNodeDef2 = NodeDef {
            op = "Constant"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                tensor = TensorProto {
                    Shape(shape)
                    DoubleData(listOf(1.0))
                }
            })
        }



        val graphDef = GraphDef {
            Node(valueNodeDef)
            Node(valueNodeDef2)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph
            ,dynamicVariables = HashMap())
        val convertNumberListToInputNDArrayRule = argDescriptorConstant(listOf(ArgDescriptor {
            name = "value"
            int32Value = 1
        }))

        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)

        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(1,convertNumberListToInputNDArrayResult[0].int32Value)
    }



    @Test
    fun testConvertNDArrayInputToScalarAttr() {
        val opDef = tensorflowOps.findOp("Dilation2D")
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

        val shape = listOf(1,1).map { it.toLong() }
        val valueNodeDef2 = NodeDef {
            op = "Constant"
            name = "inputs"
            Attribute(name = "value",value =  AttrValue {
                tensor = TensorProto {
                    Shape(shape)
                    DoubleData(listOf(1.0))
                }
            })
        }



        val graphDef = GraphDef {
            Node(valueNodeDef)
            Node(valueNodeDef2)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        val convertNumberListToInputNDArrayRule = convertNDArrayInputToNumericalAttr(mutableMapOf("output" to "inputs "))
        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)
        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(2,convertNumberListToInputNDArrayResult[0].int64Value)
    }

    @Test
    fun testListAttributeValueLookupToIndex() {
        val opDef = tensorflowOps.findOp("Dilation2D")
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        val convertNumberListToInputNDArrayRule = listAttributeValueLookupToIndex(outputAttributeValue = "output", inputAttributeValue = "strides", idx = 0,argumentIndex = 0)
        val convertNumberListToInputNDArrayResult = convertNumberListToInputNDArrayRule.convertAttributes(mappingContext)
        assertEquals(1,convertNumberListToInputNDArrayResult.size)
        assertEquals(2,convertNumberListToInputNDArrayResult[0].int64Value)
    }


    @Test
    fun testConvertNumberListToInputNDArray() {
        val opDef = tensorflowOps.findOp("Dilation2D")
        val intItems = listOf(1,1,1,1)
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        val convertNumberListToInputNDArrayRule = convertNumberListToInputNDArray(outputAttributeValue = "output",inputAttributeValue = "strides")
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
        val opDef = tensorflowOps.findOp("CudnnRNN")
        val valueNodeDef = NodeDef {
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


        val graphDef = GraphDef {
            Node(valueNodeDef)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
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
        val opDef = tensorflowOps.findOp("CudnnRNN")
        val valueNodeDef = NodeDef {
            op = "CudnnRNN"
            name = "inputs"
            Attribute(name = "is_training",value =  AttrValue {
                b = true
            })
        }


        val graphDef = GraphDef {
            Node(valueNodeDef)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        val booleanToInt = invertBooleanNumber(mapOf("output" to "is_training"))
        val booleanToIntResult = booleanToInt.convertAttributes(mappingContext)
        assertEquals(1,booleanToIntResult.size)
        val boolValue = booleanToIntResult[0].int64Value
        assertEquals(1,boolValue)
    }

    @Test
    fun testAttributeScalarToNDArrayInputRuleDouble() {
        val opDef = tensorflowOps.findOp("CudnnRNN")
        val valueNodeDef = NodeDef {
            op = "CudnnRNN"
            name = "inputs"
            Attribute(name = "dropout",value =  AttrValue {
                f = 1.0f
            })
        }


        val graphDef = GraphDef {
            Node(valueNodeDef)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
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
        val opDef = tensorflowOps.findOp("CountUpTo")
        val valueNodeDef = NodeDef {
            op = "CountUpTo"
            name = "inputs"
            Attribute(name = "limit",value =  AttrValue {
                i = 1
            })
        }


        val graphDef = GraphDef {
            Node(valueNodeDef)
        }

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
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
        val opDef = tensorflowOps.findOp("Const")
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        listOf("value","notValue").zip(listOf(false,true)).forEach { (valueToTest,assertionResult) ->
            val stringNotEqualsRule = stringNotEqualsRule(outputAttribute = "output",inputFrameworkAttributeName = "value",valueToTest = valueToTest,argumentIndex = 0)
            val stringEqualsResult = stringNotEqualsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testStringContainsRule() {
        val opDef = tensorflowOps.findOp("Const")
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        listOf("value","notValue").zip(listOf(true,false)).forEach { (valueToTest,assertionResult) ->
            val stringContainsRule = stringContainsRule(outputAttribute = "output",inputFrameworkAttributeName = "value",valueToTest = valueToTest)
            val stringEqualsResult = stringContainsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testStringEqualsRule() {
        val opDef = tensorflowOps.findOp("Const")
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = valueNodeDef,graph = tfGraph,dynamicVariables = HashMap())
        listOf("value","notValue").zip(listOf(true,false)).forEach { (valueToTest,assertionResult) ->
            val stringEqualsRule = stringEqualsRule(outputAttribute = "output",inputFrameworkAttributeName = "value",valueToTest = valueToTest,argumentIndex = 0)
            val stringEqualsResult = stringEqualsRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,stringEqualsResult.size)
            assertEquals(assertionResult,stringEqualsResult[0].boolValue)

        }


    }


    @Test
    fun testNDArraySizeAtRule() {
        val opDef = tensorflowOps.findOp("AddN")
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

        val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph,dynamicVariables = HashMap())
        shape.forEachIndexed { i,value ->
            val sizeAtRule = sizeAtRule(dimensionIndex = i,outputAttributeName = "output",inputFrameworkAttributeName = "inputs",argumentIndex = 0)
            val sizeAtRuleResult = sizeAtRule.convertAttributes(mappingCtx = mappingContext)
            assertEquals(1,sizeAtRuleResult.size)
            assertEquals(value,sizeAtRuleResult[0].int64Value)

        }

    }


    fun OpList.findOp(name: String): OpDef {
        if(!this.opList.map { input -> input.name }.contains(name)) {
            throw IllegalArgumentException("Op $name not found!")
        }
        return this.opList.first { it.name == name }!!
    }

    @Test
    fun testConditionalIndex() {

        val opDef = tensorflowOps.findOp("AddN")
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

            val tfGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)


            val mappingContext = TensorflowMappingContext(opDef = opDef,node = nodeDef,graph = tfGraph,dynamicVariables = HashMap())

            val conditionalIndex = conditionalFieldValueIntIndexArrayRule(
                outputAttribute = "N",
                attributeNameOfListAttribute = "N",
                targetValue = "value", trueIndex = trueIndex, falseIndex = falseIndex,
                inputFrameworkStringNameToTest = "T",argumentIndex = 0)

            val ret = conditionalIndex.convertAttributes(mappingContext)
            assertEquals(1,ret.size)
            assertEquals((assertionValue[string] ?:
            error("No value found with string value $string")).toLong(),ret[0].int64Value)
            assertEquals("N",ret[0].name)

        }

    }
}




