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

package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx


import onnx.Onnx
import org.eclipse.deeplearning4j.modelimport.onnx.OnnxTestUtils
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.config.SDValue
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.common.util.ArrayUtil
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.factory.ops.NDBitwise
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.onnx.*
import org.nd4j.samediff.frameworkimport.onnx.definitions.OnnxOpDeclarations
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraphRunner
import kotlin.test.assertTrue

data class OnnxGraphInput(var graphDef: Onnx.GraphProto, var inputNames: List<String>, var outputNames: List<String>,
                          var inputArrays: Map<String, INDArray>, var dynamicArrays: Map<String, INDArray>)

@Tag(TagNames.ONNX)
@Disabled
class TestOnnxIR {
    var declarations = OnnxOpDeclarations

    @Test
    fun testSequenceConstruct() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Output(createSequenceValueInfoFromTensors(arrayOf(inputTensor),"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf("W" to w,"x" to inputTensor)),onnxOpRegistry)
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor),"W" to SDValue.create(w))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2,listOf("y"))
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertion["y"],result["y"])
    }




    @Test
    fun testSequenceErase() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var insert = Nd4j.ones(DataType.FLOAT,1)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var index = Nd4j.ones(1).castTo(DataType.INT32)
        var indexTwo = Nd4j.ones(1).castTo(DataType.INT32)

        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            Input(createValueInfoFromTensor(insert,"insert",true))
            Input(createValueInfoFromTensor(index,"index",true))
            Input(createValueInfoFromTensor(indexTwo,"indexTwo",true))

            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Node(NodeProto {
                Input("y")
                Input("indexTwo")
                Output("sequenceInsert")
                name = "sequenceInsert"
                opType = "SequenceErase"

            })

            Node(NodeProto {
                Input("y")
                Input("index")
                Output("sequenceAt")
                name = "sequenceAt"
                opType = "SequenceAt"

            })

            Output(createValueInfoFromTensor(inputTensor,"sequenceAt",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W","index","indexTwo","insert"),listOf("sequenceAt"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf(
            "W" to w,"x" to inputTensor,"index" to index,"indexTwo" to indexTwo,"insert" to insert)),onnxOpRegistry)
        println(importedGraph.summary())
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w),"index" to arrayOf(index),"indexTwo" to arrayOf(indexTwo),"insert" to arrayOf(insert))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor)
            ,"W" to SDValue.create(w),
            "index" to SDValue.create(index),
            "indexTwo" to SDValue.create(indexTwo),"insert" to SDValue.create(insert))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2,listOf("sequenceAt"))
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertion["sequenceAt"],result["sequenceAt"])
    }



    @Test
    fun testSequenceInsert() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var insert = Nd4j.ones(DataType.FLOAT,1)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var index = Nd4j.ones(1).castTo(DataType.INT32)
        var indexTwo = Nd4j.ones(1).castTo(DataType.INT32).addi(1)

        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            Input(createValueInfoFromTensor(insert,"insert",true))
            Input(createValueInfoFromTensor(index,"index",true))
            Input(createValueInfoFromTensor(indexTwo,"indexTwo",true))

            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Node(NodeProto {
                Input("y")
                Input("insert")
                Input("indexTwo")
                Output("sequenceInsert")
                name = "sequenceInsert"
                opType = "SequenceInsert"

            })

            Node(NodeProto {
                Input("y")
                Input("index")
                Output("sequenceAt")
                name = "sequenceAt"
                opType = "SequenceAt"

            })

            Output(createValueInfoFromTensor(inputTensor,"sequenceAt",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W","index","indexTwo","insert"),listOf("sequenceAt"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf(
            "W" to w,"x" to inputTensor,"index" to index,"indexTwo" to indexTwo,"insert" to insert)),onnxOpRegistry)
        println(importedGraph.summary())
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w),"index" to arrayOf(index),"indexTwo" to arrayOf(indexTwo),"insert" to arrayOf(insert))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor),"W" to
                SDValue.create(w),"index" to SDValue.create(index),"indexTwo" to SDValue.create(indexTwo),
            "insert" to SDValue.create(insert))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2,listOf("sequenceAt"))
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertion["sequenceAt"],result["sequenceAt"])
    }



    @Test
    fun testSequenceLength() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)

        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))

            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Node(NodeProto {
                Input("y")
                Output("sequenceLength")
                name = "sequenceLength"
                opType = "SequenceLength"

            })

            Output(createValueInfoFromTensor(inputTensor.castTo(DataType.INT64),"sequenceLength",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("sequenceLength"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf(
            "W" to w,"x" to inputTensor)),onnxOpRegistry)
        println(importedGraph.summary())
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor),"W" to SDValue.create(w))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2,listOf("sequenceLength"))
        var assertionArr = assertion["sequenceLength"]!!
        var resultArr = assertion["sequenceLength"]!!
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertionArr,resultArr)
    }


    @Test
    fun testSequenceAt() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var index = Nd4j.ones(1).castTo(DataType.INT32)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            Input(createValueInfoFromTensor(index,"index",true))
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Node(NodeProto {
                Input("y")
                Input("index")
                Output("sequenceAt")
                name = "sequenceAt"
                opType = "SequenceAt"

            })

            Output(createValueInfoFromTensor(inputTensor,"sequenceAt",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("sequenceAt"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf("W" to w,"x" to inputTensor,"index" to index)),onnxOpRegistry)
        println(importedGraph.summary())
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w),"index" to arrayOf(index))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor),"W" to SDValue.create(w),"index" to SDValue.create(index))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2,listOf("sequenceAt"))
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertion["sequenceAt"],result["sequenceAt"])
    }




    @Test
    fun testSequenceRemove() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var index = Nd4j.ones(1).castTo(DataType.INT32)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            Input(createValueInfoFromTensor(index,"index",true))
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "SequenceConstruct"

            })

            Node(NodeProto {
                Input("y")
                Input("index")
                Output("sequenceRemove")
                name = "sequenceRemove"
                opType = "SequenceErase"

            })

            Output(createEmptySequence(convertToOnnxDataType( inputTensor.dataType()),"sequenceRemove"))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("sequenceRemove"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf("W" to w,"x" to inputTensor,"index" to index)),onnxOpRegistry)
        println(importedGraph.summary())
        var inputs = mapOf("x" to arrayOf(inputTensor),"W" to arrayOf(w),"index" to arrayOf(index))
        var inputs2 = mapOf("x" to SDValue.create(inputTensor),"W" to SDValue.create(w),"index" to SDValue.create(index))
        var assertion = onnxGraphRunner.runSequence(inputs2)
        var result = importedGraph.outputValues(inputs2 ,listOf("sequenceRemove"))
        //assert equals doesn't know how to deal with arrays within maps
        assertEquals(assertion["sequenceRemove"],result["sequenceRemove"])
    }



    @Test
    fun testConvPaddingSame() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(1,1,1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(1)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf("W" to w,"x" to inputTensor)),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }


    @Test
    fun testLoop() {
        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)

        var bitWise = NDBitwise()
        var and = bitWise.and(Nd4j.ones(DataType.INT64,1),Nd4j.ones(DataType.INT64,1))

        var axes = Nd4j.createFromArray(0L)


        var inputArr = Nd4j.create(floatArrayOf(1.0f,2.0f,3.0f,4.0f,5.0f))
        var y = Nd4j.create(floatArrayOf(-2.0f))



        var onnxOpRegistry = registry()


        var tripCount = Nd4j.createFromArray(2).castTo(DataType.INT64)
        var resY = Nd4j.createFromArray(13f)
        var cond = Nd4j.createFromArray(true)
        var resScan = Nd4j.createFromArray(-1,1,4,8,13).castTo(DataType.FLOAT).reshape(5,1)


        var modelLoad = """
            ir_version: 8
            producer_name: "backend-test"
            graph {
              node {
                input: "trip_count"
                input: "cond"
                input: "seq_empty"
                output: "seq_res"
                op_type: "Loop"
                name: "loop_body"
                attribute {
                  name: "body"
                  g {
                    node {
                      name: "cond_out"
                      input: "cond_in"
                      output: "cond_out"
                      op_type: "Identity"
                    }
                    node {
                      name: "x"
                      output: "x"
                      op_type: "Constant"
                      attribute {
                        name: "value"
                        t {
                          dims: 5
                          data_type: 1
                          float_data: 1.0
                          float_data: 2.0
                          float_data: 3.0
                          float_data: 4.0
                          float_data: 5.0
                          name: "const_tensor_x"
                        }
                        type: TENSOR
                      }
                    }
                    node {
                      name: "one"
                      output: "one"
                      op_type: "Constant"
                      attribute {
                        name: "value"
                        t {
                          data_type: 7
                          int64_data: 1
                          name: "const_tensor_one"
                        }
                        type: TENSOR
                      }
                    }
                    node {
                      name: "slice_start"
                      output: "slice_start"
                      op_type: "Constant"
                      attribute {
                        name: "value"
                        t {
                          dims: 1
                          data_type: 7
                          int64_data: 0
                          name: "const_tensor_zero"
                        }
                        type: TENSOR
                      }
                    }
                    node {
                      name: "add"
                      input: "iter_count"
                      input: "one"
                      output: "end"
                      op_type: "Add"
                    }
                    node {
                      name: "axes"
                      output: "axes"
                      op_type: "Constant"
                      attribute {
                        name: "value"
                        t {
                          data_type: 7
                          int64_data: 0
                          name: "const_tensor_axes"
                        }
                        type: TENSOR
                      }
                    }
                    node {
                      name: "slice_end"
                      input: "end"
                      input: "axes"
                      output: "slice_end"
                      op_type: "Unsqueeze"
                    }
                    node {
                      name: "slice_out"
                      input: "x"
                      input: "slice_start"
                      input: "slice_end"
                      output: "slice_out"
                      op_type: "Slice"
                    }
                    node {
                      name: "seq_out"
                      input: "seq_in"
                      input: "slice_out"
                      output: "seq_out"
                      op_type: "SequenceInsert"
                    }
                    name: "loop_body"
                    input {
                      name: "iter_count"
                      type {
                        tensor_type {
                          elem_type: 7
                          shape {
                          }
                        }
                      }
                    }
                    input {
                      name: "cond_in"
                      type {
                        tensor_type {
                          elem_type: 9
                          shape {
                          }
                        }
                      }
                    }
                    input {
                      name: "seq_in"
                      type {
                        sequence_type {
                          elem_type {
                            tensor_type {
                              elem_type: 1
                            }
                          }
                        }
                      }
                    }
                    output {
                      name: "cond_out"
                      type {
                        tensor_type {
                          elem_type: 9
                          shape {
                          }
                        }
                      }
                    }
                    output {
                      name: "seq_out"
                      type {
                        sequence_type {
                          elem_type {
                            tensor_type {
                              elem_type: 1
                            }
                          }
                        }
                      }
                    }
                  }
                  type: GRAPH
                }
              }
              name: "test_loop13_seq"
              input {
                name: "trip_count"
                type {
                  tensor_type {
                    elem_type: 7
                    shape {
                    }
                  }
                }
              }
              input {
                name: "cond"
                type {
                  tensor_type {
                    elem_type: 9
                    shape {
                    }
                  }
                }
              }
              input {
                name: "seq_empty"
                type {
                  sequence_type {
                    elem_type {
                      tensor_type {
                        elem_type: 1
                        shape {
                        }
                      }
                    }
                  }
                }
              }
              output {
                name: "seq_res"
                type {
                  sequence_type {
                    elem_type {
                      tensor_type {
                        elem_type: 1
                      }
                    }
                  }
                }
              }
            }
            opset_import {
              domain: ""
              version: 13
            }

        """.trimIndent()

        var graph = OnnxTestUtils.loadFromString(modelLoad)
        var onnxIRGraph = OnnxIRGraph(graph.graph,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("trip_count","cond","seq_empty"),listOf("res_y","res_scan"))
        var inputs = mapOf("trip_count" to tripCount,"cond" to cond,"y" to y,"begin_axes" to axes,"end_axes" to axes,"iter_count" to Nd4j.create(Nd4j.createBuffer(longArrayOf(1))))
        var sequenceInputValues = mapOf("trip_count" to SDValue.create(tripCount),
            "cond" to SDValue.create(cond),
            "y" to SDValue.create(y),
            "begin_axes" to SDValue.create(axes),"end_axes" to SDValue.create(axes),
            "seq_empty" to SDValue.create(mutableListOf(Nd4j.ones(DataType.FLOAT,1))))


        var inputsOnnx = convertToOnnxTensors(inputs)

        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,inputsOnnx,onnxOpRegistry)
        var assertion = onnxGraphRunner.runSequence(sequenceInputValues)



        println(importedGraph.summary())

        var result = importedGraph.outputValues(sequenceInputValues,mutableListOf("seq_res"))
        assertEquals(assertion,result)

    }

    @Test
    fun testEager() {
        var sd = SameDiff.create()
        sd.isEagerMode = true
        var result = sd.math().add(sd.constant(Nd4j.ones(1)),sd.constant(Nd4j.ones(1)))
        var result2 = sd.math().add(result,1.0)
        sd.outputAll(emptyMap())
        println(result2)
    }


    @Test
    fun testConvPaddingGroups() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.ones(1,32,224,224).castTo(DataType.FLOAT)
        var w = Nd4j.ones(32,1,3,3).castTo(DataType.FLOAT)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(1,1,1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(32)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var inputsOnnx = mutableMapOf("x" to convertToOnnxTensor(inputTensor,"x"),"W" to convertToOnnxTensor(w,"W"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,inputsOnnx,onnxOpRegistry)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }


    @Test
    fun testConvPadding() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5).castTo(DataType.FLOAT)
        var w = Nd4j.ones(1,1,3,3).castTo(DataType.FLOAT)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(1,1,1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(1)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(mutableMapOf("x" to inputTensor,"W" to w)),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }


    @Test
    fun testConvNoPadding() {
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,25,25).reshape(1,1,5,5)
        var w = Nd4j.ones(1,1,3,3)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(0,0,0,0))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(1)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
            convertToOnnxTensors(mutableMapOf("x" to inputTensor,"W" to w)),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }


    @Test
    fun testConvStridesPadding() {
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,34,35).reshape(1,1,7,5)
        var w = Nd4j.ones(1,1,3,3)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(1,1,1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(2,2))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(1)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
            convertToOnnxTensors(mutableMapOf("x" to inputTensor,"W" to w)),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }


    @Test
    @Disabled("See: https://github.com/eclipse/deeplearning4j/issues/9525 we need to support asymmetrics padding")
    fun testConvStridesAsymmetricPadding() {
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.linspace(0,34,35).reshape(1,1,7,5)
        var w = Nd4j.ones(1,1,3,3)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))
            Input(createValueInfoFromTensor(w,"W",true))
            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Input("W")
                Output("y")
                name = "y"
                opType = "Conv"
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "kernel_shape"
                    ListInts(listOf(3,3))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "pads"
                    ListInts(listOf(1,0,1,0))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "strides"
                    ListInts(listOf(2,2))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INTS
                    name = "dilations"
                    ListInts(listOf(1,1))
                })
                Attribute(AttributeProto {
                    type = Onnx.AttributeProto.AttributeType.INT
                    name = "group"
                    IntValue(1)
                })
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","W"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,HashMap(),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor,"W" to w)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)

    }



    @Test
    fun testOpExecutionHooks() {
        var onnxOpRegistry = registry()
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var inputTensor = Nd4j.ones(1,3,5,5)
        var graphToRun = GraphProto {
            Input(createValueInfoFromTensor(inputTensor,"x",true))


            //Initializer(convertedTensor)
            Node(NodeProto {
                Input("x")
                Output("y")
                name = "y"
                opType = "GlobalAveragePool"
            })

            Output(createValueInfoFromTensor(inputTensor,"y",false))
        }


        var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x"),listOf("y"))
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
            convertToOnnxTensors(mutableMapOf("x" to inputTensor)),onnxOpRegistry)
        var inputs = mapOf("x" to inputTensor)
        var assertion = onnxGraphRunner.run(inputs)
        var result = importedGraph.output(inputs,"y")
        assertEquals(assertion,result)
    }


    @Test
    fun testExpand() {
        var declarations = OnnxOpDeclarations
        var onnxOpRegistry = registry()
        var shape = longArrayOf(3,1)
        var newShape = longArrayOf(2,1,6)
        var inputNewShape = Nd4j.create(Nd4j.createBuffer(newShape))
        var inputs = mapOf("data" to Nd4j.arange(1.0, ArrayUtil.prod(*shape).toDouble() + 1.0).reshape(*shape),
            "newShape" to inputNewShape)
        var inputNames = listOf("data","newShape")
        var outputs = listOf("expanded")
        var graph = createSingleNodeGraph(op = "Expand",inputs = inputs, attributes = emptyMap(),outputs = outputs,inputNames = inputNames)
        runAssertion(graph,inputs,outputs)

    }

    @Test
    fun testSlice() {

        /**
         * Note that this test case is manual due to subtle differences in
         * how onnxruntime and tensorflow appear to interpret their nearest neighbor results.
         * In our test case here, we are verifying against tensorflow-onnx as the implementation.
         *
         */
        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)

        var x = Nd4j.linspace(1,1000,1000).reshape(20,10,5)
        var starts = Nd4j.zeros(2).castTo(DataType.INT64)
        var ends = Nd4j.create(Nd4j.createBuffer(longArrayOf(3,10))).reshape(2)
        var axes = Nd4j.create(Nd4j.createBuffer(longArrayOf(0,1))).reshape(2)
        var steps = Nd4j.ones(2).castTo(DataType.INT64).reshape(2)

        var input = mapOf("x" to x,"starts" to starts,"ends" to ends,"axes" to axes,"steps" to steps)

        var outputs = listOf("y")
        var attributes = emptyMap<String,Any>()
        var inputs = listOf("x","starts","ends","axes","steps")
        var graph = createSingleNodeGraph(input,"Slice",attributes,outputs,inputs)
        assertEquals(input.size,graph.inputCount)
        assertEquals(1,graph.outputCount)
        runAssertion(graph,input,outputs)
    }




    @Test
    fun testClip() {
        var declarations = OnnxOpDeclarations
        var inputs = mutableMapOf("input" to Nd4j.linspace(1,4,4).castTo(DataType.DOUBLE),
            "min" to Nd4j.scalar(1.0).castTo(DataType.DOUBLE), "max" to Nd4j.scalar(2.0).castTo(DataType.DOUBLE))
        var output = listOf("output")
        var createdGraph = createSingleNodeGraph(inputs,"Clip",emptyMap(),output,inputs.keys.toList())
        runAssertion(createdGraph,inputs,output)

    }


    @Test
    fun testNonZero() {
        var declarations = OnnxOpDeclarations
        var inputs = mutableMapOf("input" to Nd4j.linspace(1,4,4).castTo(DataType.DOUBLE))
        var onnxOpRegistry = registry()

        var output = listOf("output")
        var createdGraph = createSingleNodeGraph(inputs,"NonZero",emptyMap(),output,inputs.keys.toList(),templateTensor = Nd4j.ones(DataType.INT64))
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var onnxIRGraph = OnnxIRGraph(createdGraph,onnxOpRegistry)
        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(inputs),onnxOpRegistry)
        var result = importedGraph.output(inputs,output)

        //runAssertion(createdGraph,inputs,output)

    }


    @Test
    fun testIf() {
        var thenOut = convertToOnnxTensor(Nd4j.ones(DataType.FLOAT,5),"then_out")
        var elseOut = convertToOnnxTensor(Nd4j.ones(DataType.FLOAT,5),"else_out")
        var x = Nd4j.linspace(1,5,5).castTo(DataType.FLOAT)
        var y = Nd4j.create(floatArrayOf(5.0f,4.0f,3.0f,2.0f,1.0f))
        var elseGraph = createSingleNodeGraph(emptyMap(),"Constant",mapOf("value" to elseOut),listOf("else_out"),listOf(),x)
        var thenGraph = createSingleNodeGraph(emptyMap(),"Constant",mapOf("value" to thenOut),listOf("then_out"),listOf(),x)
        var thenGraphAttr = AttributeProto {
            name = "then_branch"
            g = thenGraph
            type = Onnx.AttributeProto.AttributeType.GRAPH
        }
        var elseAttr = AttributeProto {
            name = "else_branch"
            g = elseGraph
            type = Onnx.AttributeProto.AttributeType.GRAPH
        }
        var ifNode = NodeProto {
            opType = "If"
            name = "ifNode"
            Input("cond")
            Output("res")
            Attribute(thenGraphAttr)
            Attribute(elseAttr)
        }

        var graph = GraphProto {
            name = "ifGraph"
            Input(createValueInfoFromTensor(Nd4j.ones(1).castTo(DataType.BOOL),"cond",true))
            Node(ifNode)
            Output(createValueInfoFromTensor(y,"res",true))
        }

        runAssertion(graph,mapOf("cond" to (Nd4j.ones(1).castTo(DataType.BOOL))),listOf("res"))
    }



    @Test
    fun testRoiAligned() {
        var xArr =   arrayOf(
            doubleArrayOf(
                0.2764,
                0.7150,
                0.1958,
                0.3416,
                0.4638,
                0.0259,
                0.2963,
                0.6518,
                0.4856,
                0.7250,
            ),
            doubleArrayOf(
                0.9637,
                0.0895,
                0.2919,
                0.6753,
                0.0234,
                0.6132,
                0.8085,
                0.5324,
                0.8992,
                0.4467,
            ),
            doubleArrayOf(
                0.3265,
                0.8479,
                0.9698,
                0.2471,
                0.9336,
                0.1878,
                0.4766,
                0.4308,
                0.3400,
                0.2162,
            ),
            doubleArrayOf(
                0.0206,
                0.1720,
                0.2155,
                0.4394,
                0.0653,
                0.3406,
                0.7724,
                0.3921,
                0.2541,
                0.5799,
            ),
            doubleArrayOf(
                0.4062,
                0.2194,
                0.4473,
                0.4687,
                0.7109,
                0.9327,
                0.9815,
                0.6320,
                0.1728,
                0.6119,
            ),
            doubleArrayOf(
                0.3097,
                0.1283,
                0.4984,
                0.5068,
                0.4279,
                0.0173,
                0.4388,
                0.0430,
                0.4671,
                0.7119,
            ),
            doubleArrayOf(
                0.1011,
                0.8477,
                0.4726,
                0.1777,
                0.9923,
                0.4042,
                0.1869,
                0.7795,
                0.9946,
                0.9689,
            ),
            doubleArrayOf(
                0.1366,
                0.3671,
                0.7011,
                0.6234,
                0.9867,
                0.5585,
                0.6985,
                0.5609,
                0.8788,
                0.9928,
            ),
            doubleArrayOf(
                0.5697,
                0.8511,
                0.6711,
                0.9406,
                0.8751,
                0.7496,
                0.1650,
                0.1049,
                0.1559,
                0.2514,
            ),
            doubleArrayOf(
                0.7012,
                0.4056,
                0.7879,
                0.3461,
                0.0415,
                0.2998,
                0.5094,
                0.3727,
                0.5482,
                0.0502,
            ))
        var roiX = Nd4j.create(xArr).reshape(1,1,10,10).castTo(DataType.FLOAT)
        var rois = Nd4j.create(Nd4j.createBuffer(longArrayOf(0,0,9,9,0,5,4,9,5,5,9,9))).reshape(3,4).castTo(DataType.FLOAT)
        var batchIndices = Nd4j.create(Nd4j.createBuffer(longArrayOf(0,0,0))).reshape(3)
        var y = Nd4j.create(Nd4j.createBuffer(doubleArrayOf(0.4664,0.4466,0.3405,0.5688,0.6068,0.3714,0.4296,0.3835,0.5562,0.351
            ,0.2768,0.4883,0.5222,0.5528,0.4171,0.4713,0.4844,0.6904,0.492,0.8774
            ,0.6239,0.7125,0.6289,0.3355,0.3495,0.3022,0.4305,0.4696,0.3978,0.5423
            ,0.3656,0.705,0.5165,0.3172,0.7015,0.2912,0.5059,0.6476,0.6235,0.8299
            ,0.5916,0.7389,0.7048,0.8372,0.8893,0.6227,0.6153,0.7097,0.6154,0.4585
            ,0.2384,0.3379,0.3717,0.61,0.7601,0.3767,0.3785,0.7147,0.9243,0.9727
            ,0.5749,0.5826,0.5709,0.7619,0.877,0.5355,0.2566,0.2141,0.2796,0.36
            ,0.4365,0.3504,0.2887,0.3661,0.2349))).reshape(3,1,5,5)

        var outputs = listOf("y")
        var inputs = mapOf("X" to roiX,"rois" to rois,"batch_indices" to batchIndices)
        var attributes = mapOf("spatial_scale" to 1.0f,"output_height" to 5,"output_width" to 5,"sampling_ratio" to 2)
        var createdGraph = createSingleNodeGraph(inputs,"RoiAlign",attributes,outputs,inputs.keys.toList())
        runAssertion(createdGraph,inputs,outputs)

    }

    @Test
    fun testMaximum() {
        var declarations = OnnxOpDeclarations
        var inputs = mutableMapOf<String,INDArray>()
        for(i in 0 until 5) {
            inputs["$i"] = Nd4j.zeros(2).addi(i)
        }

        var output = listOf("output")
        var createdGraph = createSingleNodeGraph(inputs,"Max",emptyMap(),output,inputs.keys.toList())
        runAssertion(createdGraph,inputs,output)

    }


    @Test
    fun testMinimum() {
        var declarations = OnnxOpDeclarations
        var inputs = mutableMapOf<String,INDArray>()
        for(i in 0 until 5) {
            inputs["$i"] = Nd4j.zeros(2).addi(i)
        }

        var output = listOf("output")
        var createdGraph = createSingleNodeGraph(inputs,"Min",emptyMap(),output,inputs.keys.toList())
        runAssertion(createdGraph,inputs,output)

    }


    @Test
    fun testUnsqueeze() {
        var declarations = OnnxOpDeclarations

        /**
         * Note that this test case is manual due to subtle differences in
         * how onnxruntime and tensorflow appear to interpret their nearest neighbor results.
         * In our test case here, we are verifying against tensorflow-onnx as the implementation.
         *
         */
        var onnxOpRegistry = registry()
        var inputData = Nd4j.linspace(1,15,15).reshape(1,3,1,5)
        var axes = Nd4j.create(floatArrayOf(-2.0f)).castTo(DataType.INT64)
        var input = mapOf("x" to inputData,"axes" to axes)

        var outputs = listOf("y")
        var attributes = emptyMap<String,Any>()
        var inputs = listOf("x","axes")
        var graph = createSingleNodeGraph(input,"Unsqueeze",attributes,outputs,inputs)
        assertEquals(input.size,graph.inputCount)
        assertEquals(1,graph.outputCount)
        var onnxIRGraph = OnnxIRGraph(graph,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,input.keys.toList(),outputs)
        var assertion = onnxGraphRunner.run(input)
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()

        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(input),onnxOpRegistry)
        var result = importedGraph.output(input,outputs)
        //TODO: add coefficients for better eps comparison, see: https://github.com/eclipse/deeplearning4j/issues/9467
        assertTrue(assertion["y"]!!.equalsWithEps(result["y"],1e-1))
    }




    @Test
    fun testCast() {
        var declarations = OnnxOpDeclarations

        /**
         * Note that this test case is manual due to subtle differences in
         * how onnxruntime and tensorflow appear to interpret their nearest neighbor results.
         * In our test case here, we are verifying against tensorflow-onnx as the implementation.
         *
         */
        var onnxOpRegistry = registry()
        var startInput = Nd4j.ones(2).castTo(DataType.DOUBLE)
        var inputData = mapOf("x" to startInput)
        var outputs = listOf("y")
        var attributes = mapOf("to" to Onnx.TensorProto.DataType.FLOAT.ordinal)
        var graph = createSingleNodeGraph(inputData,"Cast",attributes,outputs,listOf("x"),Nd4j.ones(2).castTo(DataType.FLOAT))
        runAssertion(graph,inputData,outputs)
    }

    @Test
    fun testResize() {
        var declarations = OnnxOpDeclarations

        /**
         * Note that this test case is manual due to subtle differences in
         * how onnxruntime and tensorflow appear to interpret their nearest neighbor results.
         * In our test case here, we are verifying against tensorflow-onnx as the implementation.
         *
         */
        var onnxOpRegistry = registry()
        var inputData = Nd4j.linspace(1,16,16).reshape(1,1,4,4)
        var scales = Nd4j.create(floatArrayOf(1.0f,1.0f,0.8f,0.8f))
        var input = mapOf("x" to inputData,"scales" to scales,"roi-empty" to Nd4j.zeros(1,1,1,1))

        var outputs = listOf("y")
        var attributes = mapOf("mode" to "cubic")
        var inputs = listOf("x","roi-empty","scales")
        var graph = createSingleNodeGraph(input,"Resize",attributes,outputs,inputs)
        assertEquals(input.size,graph.inputCount)
        assertEquals(1,graph.outputCount)
        var onnxIRGraph = OnnxIRGraph(graph,onnxOpRegistry)
        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,input.keys.toList(),outputs)
        var assertion = onnxGraphRunner.run(input)
        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()

        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null, convertToOnnxTensors(input),onnxOpRegistry)
        var result = importedGraph.output(input,outputs)
        //TODO: add coefficients for better eps comparison, see: https://github.com/eclipse/deeplearning4j/issues/9467
        assertTrue(assertion["y"]!!.equalsWithEps(result["y"],1e-1))

    }


    @Test
    fun testOpExecution() {
        var onnxOpRegistry = registry()

        var scalarInputs = mapOf(
            "abs" to -1.0,
            "copy" to 1.0,
            "erfc" to 1.0,
            "exp" to 1.0,
            "identity" to 1.0,
            "neg" to 1.0,
            "ones_as" to 1.0,
            "relu6" to 1.0,
            "round" to 1.0,
            "sign" to 1.0,
            "sin" to 1.0,
            "square" to 1.0,
            "sqrt" to 1.0)

        var scalarFloatOps = mapOf(
            "acos" to 1.0f,
            "asin" to 1.0f,
            "acosh" to 1.0f,
            "asinh" to 1.0f,
            "atan" to 1.0f,
            "atanh" to 0.5f,
            "ceil" to 1.0f,
            "cosh" to 1.0f,
            "cos" to 1.0f,
            "erf" to 1.0f,
            "hard_sigmoid" to 1.0f,
            "floor" to 1.0f,
            "log" to 1.0f,
            "round" to 1.0f,
            "relu" to 1.0f,
            "selu" to 1.0f,
            "sinh" to 1.0f,
            "sigmoid" to 1.0f,
            "softplus" to 1.0f,
            "softsign" to 1.0f,
            "tan" to 1.0f,
            "tanh" to 1.0f
        )


        var singleInputOps = scalarInputs.keys
        var singleInputBooleanOps = mapOf(
            "not" to false
        )

        var singleOutputBooleanOps = mapOf(
            "isfinite" to 1.0f,
            "isinf" to 1.0f,
            "isnan" to 1.0f,
        )

        var pairWiseBooleanOps = mapOf(
            "min" to listOf(1.0,2.0),
            "max" to listOf(1.0,2.0),
            "equals" to listOf(2.0,2.0),
            "greater" to listOf(2.0,1.0),
            "greater_equal" to listOf(2.0,1.0),
            "less" to listOf(2.0,1.0),
            "less_equal" to listOf(2.0,1.0))


        var singleInputIntOutput = mapOf(
            "size" to Nd4j.linspace(1,4,4).reshape(2,2),
            "shape_of" to Nd4j.linspace(1,4,4).reshape(2,2)
        )

        var pairWiseBooleanInputs = mapOf(
            "or" to listOf(true,false),
            "and" to listOf(false,false),
            "xor" to listOf(false,true)
        )


        var singleReduceOps = mapOf(
            "reduce_sum" to Nd4j.linspace(1,4,4).reshape(2,2),
            // "reduce_logsumexp" to Nd4j.linspace(1,4,4).reshape(2,2)
        )


        var pairwise = mapOf(
            "add" to listOf(1.0,1.0),
            "subtract" to listOf(2.0,1.0),
            "multiply" to listOf(2.0,1.0),
            "divide" to listOf(2.0,1.0),
            "pow" to listOf(2.0,1.0)
        )

        var mappedOps = setOf("elu","transpose","argmin","argmax","leakyrelu","prelu","flatten_2d")//,"top_k")

        /**
         * NOTE WHEN WRITING TESTS, IF YOU SEE AN ERROR like:
         * java.lang.RuntimeException: Could not find an implementation for the node output:Cos(7)
         *
         * Check the supported data types for each op here:
         * https://github.com/microsoft/onnxruntime/blob/master/docs/OperatorKernels.md
         */

        var importGraph = ImportGraph<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
        var finishedOps = HashSet<String>()
        onnxOpRegistry.mappingProcessNames()
            .filter { onnxOpRegistry.hasMappingOpProcess(it) }
            .map { onnxOpRegistry.lookupOpMappingProcess(it) }.forEach { mappingProcess ->
                var nd4jOpDef = onnxOpRegistry.lookupNd4jOpDef(mappingProcess.opName())
                var onnxOpDef = onnxOpRegistry.lookupInputFrameworkOpDef(mappingProcess.inputFrameworkOpName())
                if(scalarInputs.containsKey(nd4jOpDef.name)) {
                    print("Running op $nd4jOpDef.name")
                    var input = Nd4j.scalar(scalarInputs[mappingProcess.opName()]).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(input,"input"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("input")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(input,"output"))
                    }

                    var inputs = mapOf("input" to input)
                    var onnxInputs = convertToOnnxTensors(inputs)
                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("input"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,onnxInputs,onnxOpRegistry)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    assertEquals(assertion["output"]!!.reshape(1,1),result["output"]!!.reshape(1,1),"Function ${nd4jOpDef.name} failed with input $input")
                    finishedOps.add(nd4jOpDef.name)

                } else if(scalarFloatOps.containsKey(nd4jOpDef.name)) {
                    print("Running op $nd4jOpDef.name")
                    var input = Nd4j.scalar(scalarFloatOps[mappingProcess.opName()]).castTo(DataType.FLOAT)

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(input,"input"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("input")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(input,"output"))
                    }

                    var inputs = mapOf("input" to input)
                    var dynamicVariables = convertToOnnxTensors(inputs)
                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("input"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,dynamicVariables,onnxOpRegistry)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    assertEquals(assertion["output"]!!.reshape(1,1),result["output"]!!.reshape(1,1),"Function ${nd4jOpDef.name} failed with input $input")
                    finishedOps.add(nd4jOpDef.name)

                }

                else if(singleOutputBooleanOps.containsKey(nd4jOpDef.name)) {
                    print("Running op $nd4jOpDef.name")
                    var input = Nd4j.scalar(singleOutputBooleanOps[mappingProcess.opName()]).castTo(DataType.FLOAT)
                    var convertedTensor = convertToOnnxTensor(input,"input")
                    var convertedOutputTensor = convertToOnnxTensor(input,"output")

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(input,"input"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("input")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(Nd4j.create(booleanArrayOf(true)).reshape(),"output"))
                    }

                    var inputs = mapOf("input" to input)
                    var convertedTensors = convertToOnnxTensors(inputs)
                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("input"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,convertedTensors,onnxOpRegistry)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    assertEquals(assertion["output"]!!.reshape(1,1),result["output"]!!.reshape(1,1),"Function ${nd4jOpDef.name} failed with input $input")
                    finishedOps.add(nd4jOpDef.name)

                }


                else if(pairwise.containsKey(nd4jOpDef.name)) {
                    print("Running op def $nd4jOpDef.name")
                    var x = Nd4j.scalar(pairwise[mappingProcess.opName()]!![0]!!).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                    var y = Nd4j.scalar(pairwise[mappingProcess.opName()]!![1]!!).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(x,"x"))
                        Input(createValueInfoFromTensor(y,"y"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("x")
                            Input("y")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(x,"output"))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","y"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
                        hashMapOf("x" to convertToOnnxTensor(x,"x"),"y" to convertToOnnxTensor(y,"y")),onnxOpRegistry)
                    var inputs = mapOf("x" to x,"y" to y)
                    var result = importedGraph.output(inputs,"output")
                    var assertion = onnxGraphRunner.run(inputs)
                    assertEquals(assertion["output"]!!.getDouble(0),result["output"]!!.getDouble(0),"Function ${nd4jOpDef.name} failed with input $x $y")
                    finishedOps.add(nd4jOpDef.name)

                }  else if(pairWiseBooleanInputs.containsKey(nd4jOpDef.name)) {
                    print("Running op def $nd4jOpDef.name")
                    var x = Nd4j.scalar(pairWiseBooleanInputs[mappingProcess.opName()]!![0]!!).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)
                    var y = Nd4j.scalar(pairWiseBooleanInputs[mappingProcess.opName()]!![1]!!).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(x,"x"))
                        Input(createValueInfoFromTensor(y,"y"))
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("x")
                            Input("y")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(x,"output"))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","y"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
                        hashMapOf("x" to convertToOnnxTensor(x,"x"),"y" to convertToOnnxTensor(y,"y")),onnxOpRegistry)
                    var inputs = mapOf("x" to x,"y" to y)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    assertEquals(assertion["output"]!!.getDouble(0),result["output"]!!.getDouble(0),"Function ${nd4jOpDef.name} failed with input $x $y")
                    finishedOps.add(nd4jOpDef.name)

                } else if(pairWiseBooleanOps.containsKey(nd4jOpDef.name)) {
                    print("Running op def $nd4jOpDef.name")
                    var x = Nd4j.scalar(pairWiseBooleanOps[mappingProcess.opName()]!![0]!!).castTo(DataType.FLOAT)
                    var y = Nd4j.scalar(pairWiseBooleanOps[mappingProcess.opName()]!![1]!!).castTo(DataType.FLOAT)
                    var output = Nd4j.scalar(pairWiseBooleanOps[mappingProcess.opName()]!![1]!!).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(x,"x"))
                        Input(createValueInfoFromTensor(y,"y"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("x")
                            Input("y")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(output,"output"))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","y"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
                        hashMapOf("x" to convertToOnnxTensor(x,"x"),"y" to convertToOnnxTensor(y,"y")),onnxOpRegistry)
                    var inputs = mapOf("x" to x,"y" to y)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    assertEquals(assertion["output"]!!.getDouble(0),result["output"]!!.getDouble(0),"Function ${nd4jOpDef.name} failed with input $x $y")
                    finishedOps.add(nd4jOpDef.name)

                }

                else if(singleInputBooleanOps.containsKey(nd4jOpDef.name)) {
                    print("Running op def $nd4jOpDef.name")
                    var x = Nd4j.create(booleanArrayOf(singleInputBooleanOps[mappingProcess.opName()]!!)).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)
                    var output = Nd4j.create(booleanArrayOf(singleInputBooleanOps[mappingProcess.opName()]!!)).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)

                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(x,"x"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("x")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(output,"output"))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,hashMapOf("x" to convertToOnnxTensor(x,"x")),onnxOpRegistry)
                    var inputs = mapOf("x" to x)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    finishedOps.add(nd4jOpDef.name)

                    //assertEquals("Function ${nd4jOpDef.name} failed with input $x",assertion["output"]!!.reshape(1,1),result["output"]!!.reshape(1,1))
                }

                else if(singleReduceOps.containsKey(nd4jOpDef.name)) {
                    print("Running op def $nd4jOpDef.name")
                    var x = singleReduceOps[mappingProcess.opName()]!!.castTo(DataType.FLOAT)
                    var axes = Nd4j.zeros(1).castTo(DataType.INT64)
                    var output = x.mean(0).reshape(2)
                        .castTo(DataType.FLOAT)


                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(x,"x"))
                        Input(createValueInfoFromTensor(axes,"axes",true))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("x")
                            Input("axes")
                            Output("output")
                            Attribute(Onnx.AttributeProto.newBuilder()
                                .setType(Onnx.AttributeProto.AttributeType.INT)
                                .setI(0)
                                .setName("keepdims").build())

                        })

                        Output(createValueInfoFromTensor(output,"output",false))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var inputs = mapOf("x" to x,"axes" to axes)
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,
                        hashMapOf("x" to convertToOnnxTensor(x,"x"),"axes" to convertToOnnxTensor(axes,"axes")),onnxOpRegistry)
                    var result = importedGraph.output(inputs,"output")
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("x","axes"),listOf("output"))
                    var assertion = onnxGraphRunner.run(inputs)
                    assertEquals(assertion["output"]!!.reshape(1,2),result["output"]!!.reshape(1,2),"Function ${nd4jOpDef.name} failed with input $x")
                    finishedOps.add(nd4jOpDef.name)

                } else if(mappedOps.contains(nd4jOpDef.name)){
                    var graphForOp = graphForOp(nd4jOpDef.name)
                    graphForOp.forEach { graph ->
                        var onnxIRGraph = OnnxIRGraph(graph.graphDef,onnxOpRegistry)
                        var inputs =graph.inputArrays
                        var convertedArrays = HashMap<String,Onnx.TensorProto>()
                        graph.inputArrays.forEach { name, arr ->
                            convertedArrays[name] = convertToOnnxTensor(arr,name)
                        }
                        var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,convertedArrays,onnxOpRegistry)
                        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,graph.inputNames,graph.outputNames)
                        var assertion = onnxGraphRunner.run(inputs)
                        var result = importedGraph.output(inputs,graph.outputNames)
                        assertEquals(assertion.keys,result.keys)
                        result.forEach { name,arr ->
                            if(arr.length().toInt() == 1) {
                                assertEquals(assertion[name]!!.getDouble(0),arr.getDouble(0),1e-3,"Function ${nd4jOpDef.name} failed with input ${graph.inputNames}")
                            }
                            else {
                                assertEquals(assertion[name],arr,"Function ${nd4jOpDef.name} failed with input ${graph.inputNames}")
                            }
                        }

                        finishedOps.add(nd4jOpDef.name)


                    }


                } else   if(singleInputIntOutput.containsKey(nd4jOpDef.name)) {
                    print("Running op $nd4jOpDef.name")
                    var input = singleInputIntOutput[mappingProcess.opName()]!!.castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
                    var graphToRun = GraphProto {
                        Input(createValueInfoFromTensor(input,"input"))
                        //Initializer(convertedTensor)
                        Node(NodeProto {
                            name = "output"
                            opType = onnxOpDef.opType
                            Input("input")
                            Output("output")

                        })

                        Output(createValueInfoFromTensor(input,"output",false ))
                    }


                    var onnxIRGraph = OnnxIRGraph(graphToRun,onnxOpRegistry)
                    var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,listOf("input"),listOf("output"))
                    var importedGraph = importGraph.importGraph(onnxIRGraph,null,null,HashMap(),onnxOpRegistry)
                    var inputs = mapOf("input" to input)
                    var assertion = onnxGraphRunner.run(inputs)
                    var result = importedGraph.output(inputs,"output")
                    if(assertion["output"]!!.length() == 1L)
                        assertEquals(assertion["output"]!!.reshape(1,1),result["output"]!!.reshape(1,1),"Function ${nd4jOpDef.name} failed with input $input")
                    else
                        assertEquals(assertion["output"]!!.ravel(),result["output"]!!.ravel(),"Function ${nd4jOpDef.name} failed with input $input")
                    finishedOps.add(nd4jOpDef.name)

                }
            }

        println("Finished ops totaling ${finishedOps.size} out of ${onnxOpRegistry.mappedNd4jOpNames().size}")
    }



    fun graphForOp(opName: String): List<OnnxGraphInput> {
        when(opName) {
            "non_max_suppression_v3" -> {
                /**
                 * TODO: Add pre and post processing for each node.
                 * Our NMS requires 2d, but onnx is 3d. Attempt to see
                 * if generalized pre/post processing node additions as part of a mapping process can work.
                 *
                 */
                print("Running op def $opName")
                var boxesvar = Nd4j.create(arrayOf(
                    floatArrayOf(0f,0f,1f,1f),
                    floatArrayOf(0f,0.1f,1f,1.1f),
                    floatArrayOf(0f,-0.1f,1f,0.9f),
                    floatArrayOf(0f,10f,1f,11f)
                )).reshape(1,4,4).castTo(DataType.FLOAT)

                var scoresvar = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                    .reshape(1,1,4)
                    .castTo(DataType.FLOAT)
                var maxOutputSize = Nd4j.scalar(4.0).castTo(DataType.INT64)
                var iouThreshold = Nd4j.scalar(0.5).castTo(DataType.FLOAT)
                var scoreThreshold = Nd4j.scalar(0.0).castTo(DataType.FLOAT)

                var inputs = mapOf("boxes" to boxesVal,"scores" to scoresVal,"max_output_boxes_per_class" to maxOutputSize,
                    "iou_threshold" to iouThreshold,"score_threshold" to scoreThreshold)
                var output = Nd4j.scalar(1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(boxesVal,"boxes",false))
                    Input(createValueInfoFromTensor(scoresVal,"scores",false))
                    Input(createValueInfoFromTensor(maxOutputSize,"max_output_boxes_per_class",false))
                    Input(createValueInfoFromTensor(iouThreshold,"iou_threshold",false))
                    Input(createValueInfoFromTensor(scoreThreshold,"score_threshold",false))

                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        Input("boxes")
                        Input("scores")
                        Input("max_output_boxes_per_class")
                        Input("iou_threshold")
                        Input("score_threshold")
                        Output("output")
                        name = "output"
                        opType = "NonMaxSuppression"



                    })

                    Output(createValueInfoFromTensor(output,"output",false))
                }

                return listOf(OnnxGraphInput(graphToRun,listOf("boxes","scores","max_output_boxes_per_class","iou_threshold","score_threshold"),listOf("output"),inputs,inputs))
            }
            "argmin","argmax" -> {
                print("Running op def $opName")
                var x = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var output = x.mean(0).reshape(2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(x,"x"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = if(opName == "argmin") "ArgMin" else "ArgMax"
                        Input("x")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setName("axis").setI(0).build())
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(0)
                            .setName("keepdims").build())
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(1)
                            .setName("select_last_index").build())

                    })

                    Output(createValueInfoFromTensor(output,"output",false))
                }

                var inputMap = mapOf("x" to x)
                return listOf(OnnxGraphInput(graphToRun,listOf("x"),listOf("output"),inputMap,inputMap))
            }
            "top_k" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var k = Nd4j.scalar(2.0).castTo(DataType.INT64).reshape(1)
                var output = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    Input(createValueInfoFromTensor(k,"k"))
                    Node(NodeProto {
                        name = "output"
                        opType = "TopK"
                        Input("input")
                        Input("k")
                        Output("output")
                        Output("indices")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(0)
                            .setName("axis").build())
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(1)
                            .setName("sorted").build())
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(1)
                            .setName("largest").build())

                    })


                    Output(createValueInfoFromTensor(input,"output",false))
                    Output(createValueInfoFromTensor(output,"indices",false))
                }

                var inputMap = mapOf("input" to input,"k" to k)
                return listOf(OnnxGraphInput(graphToRun,listOf("input","k"),listOf("output","indices"),inputMap,inputMap))

            }
            "transpose" -> {
                var input = Nd4j.linspace(1,6,6).reshape(3,2).castTo(DataType.FLOAT)
                var output = Nd4j.linspace(1,6,6).reshape(2,3).castTo(DataType.FLOAT)

                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "Transpose"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .addInts(1).addInts(0)
                            .setName("perm").build())

                    })

                    Output(createValueInfoFromTensor(output,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }
            "prelu" -> {
                var input = Nd4j.randn(3,4,5).castTo(DataType.FLOAT)
                var alpha = Nd4j.zeros(1,1,5).addi(0.1).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input",false))
                    Input(createValueInfoFromTensor(input,"slope",false))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "PRelu"
                        Input("input")
                        Input("slope")
                        Output("output")

                    })

                    Output(createValueInfoFromTensor(input,"output",false))
                }

                var inputMap = mapOf("input" to input,"slope" to alpha)
                return listOf(OnnxGraphInput(graphToRun,listOf("input","slope"),listOf("output"),inputMap,inputMap))

            }



            "reduce_norm1" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "ReduceL1"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .setInts(0,0)
                            .setName("axes").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }



            "reduce_prod" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "ReduceProd"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .setInts(0,0)
                            .setName("axes").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }


            "reduce_norm2" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "ReduceL2"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .setInts(0,0)
                            .setName("axes").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }


            "reduce_mean" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "ReduceMean"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .setInts(0,0)
                            .setName("axes").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }

            "reduce_max" -> {
                var input = Nd4j.linspace(1,4,4).reshape(2,2).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "ReduceMax"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INTS)
                            .setInts(0,0)
                            .setName("axes").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }


            "elu","leakyrelu" -> {
                var input = Nd4j.scalar(1.0f).castTo(DataType.FLOAT)
                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(input,"input"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = if(name == "elu") "Elu" else "LeakyRelu"
                        Input("input")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.FLOAT)
                            .setF(1.0f)
                            .setName("alpha").build())

                    })

                    Output(createValueInfoFromTensor(input,"output"))
                }

                var inputMap = mapOf("input" to input)
                return listOf(OnnxGraphInput(graphToRun,listOf("input"),listOf("output"),inputMap,inputMap))

            }

            "flatten_2d" -> {
                var x = Nd4j.randn(2,3,4).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(x,"x"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "Flatten"
                        Input("x")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(1)
                            .setName("axis").build())
                    })

                    Output(createValueInfoFromTensor(x,"output",false))
                }

                var inputMap = mapOf("x" to x)
                return listOf(OnnxGraphInput(graphToRun,listOf("x"),listOf("output"),inputMap,inputMap))


            }

            "mod" -> {
                var x = Nd4j.scalar(2.0).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                var y = Nd4j.scalar(2.0).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                var graphToRun = GraphProto {
                    Input(createValueInfoFromTensor(x,"x"))
                    Input(createValueInfoFromTensor(y,"y"))
                    //Initializer(convertedTensor)
                    Node(NodeProto {
                        name = "output"
                        opType = "Mod"
                        Input("x")
                        Input("y")
                        Output("output")
                        Attribute(Onnx.AttributeProto.newBuilder()
                            .setType(Onnx.AttributeProto.AttributeType.INT)
                            .setI(1)
                            .setName("fmod").build())
                    })

                    Output(createValueInfoFromTensor(x,"output"))
                }

                var inputMap = mapOf("x" to x,"y" to y)
                return listOf(OnnxGraphInput(graphToRun,listOf("x","y"),listOf("output"),inputMap,inputMap))


            }
            else -> {
                throw IllegalArgumentException("Illegal op name $opName")
            }

        }
    }

}