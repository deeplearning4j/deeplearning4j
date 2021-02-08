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

package org.nd4j.samediff.frameworkimport.tensorflow

import junit.framework.Assert.assertEquals
import junit.framework.Assert.assertTrue
import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import org.junit.Ignore
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.io.ClassPathResource
import org.nd4j.imports.graphmapper.tf.TFGraphMapper
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.custom.Roll
import org.nd4j.linalg.api.ops.impl.transforms.BinCount
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.profiler.ProfilerConfig
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.registry.OpRegistryHolder
import org.nd4j.samediff.frameworkimport.tensorflow.context.TensorflowMappingContext
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraphRunner
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRNode
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRTensor
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.shade.protobuf.TextFormat
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner
import org.tensorflow.framework.*
import java.io.File
import java.lang.IllegalStateException
import java.nio.charset.Charset
import kotlin.math.max

data class GraphInput(val graphDef: GraphDef,val inputNames: List<String>,val outputNames: List<String>,
                      val inputArrays: Map<String,INDArray>,val dynamicArrays: Map<String,INDArray>)

class TestTensorflowIR {

    val tensorflowOps =  {
        val input = OpList.newBuilder()
        OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow").values.forEach {
            input.addOp(it)
        }

        input.build()
    }.invoke()



    @Test
    @Ignore
    fun manualTest() {
        val manualGraph = FileUtils.readFileToString(File("test.pbtxt"),Charset.defaultCharset())
        val parsedGraph = GraphDef.newBuilder()
        TextFormat.merge(manualGraph,parsedGraph)
        val textGraph = parsedGraph.build()
        println(textGraph)
        val tfImporter = TensorflowFrameworkImporter()
        //with names [image] and shapes {image=[4, 2, 28, 28, 3]}
        Nd4j.getEnvironment().isDebug = true
        Nd4j.getEnvironment().isVerbose = true
        //TFGraphMapper.importGraph(textGraph)
        // val inputMap = mapOf("input_1" to Nd4j.zeros(10).castTo(org.nd4j.linalg.api.buffer.DataType.INT32),"input_2" to Nd4j.zeros(1,8).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE))
        //val inputMap = mapOf("image" to Nd4j.ones(1,128,128,4))
        val inputMap = emptyMap<String,INDArray>()
        val tensorflowIRGraph = TensorflowIRGraph(textGraph,tensorflowOps,tfImporter.registry)
        val outputList = tensorflowIRGraph.nodeList().map { input -> input.nodeName() }.toMutableSet()
        outputList.add("FusedBatchNormV3:1")
        outputList.add("FusedBatchNormV3:2")
        val tfGraphRunner = TensorflowIRGraphRunner(tensorflowIRGraph, inputMap.keys.toList(), outputList.toList())
        val importedGraph = TFGraphMapper.importGraph(textGraph)
        val graph = tfImporter.importFromGraph(textGraph,inputMap)
        val tfOutput = tfGraphRunner.run(inputMap)
        val output = graph.outputAll(inputMap)
        val output2 = importedGraph.outputAll(inputMap)


        //assertEquals(tfOutput.keys,outputList)
        //assertEquals(tfOutput.keys,output2.keys)
        val names = tensorflowIRGraph.nodeList().map { input -> input.nodeName() }
        val skipValidation = setOf("parallel_stack/ExpandDims/dim")
        //assertEquals(output.keys,output2.keys)
    /*    val notEquals = HashSet<String>()
        names.forEach {
            val value = output[it]
            val value2 = output2[it]
            if(value!! != (value2!!)) {
                val oldOps = importedGraph.ops[it]
                val newOps = graph.ops[it]
                val oldVar = importedGraph.variables[it]
                val newVar = graph.variables[it]
                notEquals.add(it)
            }
        }*/

        //println(notEquals)

        // assertEquals(output,output2)
        //assertEquals(tfOutput,output)
    }

    @Test
    @Ignore
    fun manualTest2() {
        val manualGraph = FileUtils.readFileToString(File("test.pbtxt"),Charset.defaultCharset())
        val parsedGraph = GraphDef.newBuilder()
        TextFormat.merge(manualGraph,parsedGraph)
        val textGraph = parsedGraph.build()
        println(textGraph)
        val tfImporter = TensorflowFrameworkImporter()
        //with names [image] and shapes {image=[4, 2, 28, 28, 3]}
        val inputs = Nd4j.linspace(1,18816,18816).reshape(4, 2, 28, 28, 3)
        val importedGraph = TFGraphMapper.importGraph(textGraph)
        val output = importedGraph.outputAll(emptyMap())
        println(output.entries.map { (k,v) -> "$k,${v.shapeInfoToString()}" })

    }





    @Test
    fun loadModelTest() {
        val tensorflowOpRegistry = registry()
        val importGraph = ImportGraph<GraphDef,NodeDef,OpDef,TensorProto,OpDef.AttrDef,AttrValue,DataType>()
        val inputs = listOf("input_0", "input_1")
        val content = IOUtils.toByteArray(ClassPathResource("lenet_frozen.pb").inputStream)
        val graphDef = GraphDef.parseFrom(content)
        val irGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
        val importedModel = importGraph.importGraph(irGraph = irGraph,importOverride = null,opFilter = null,opMappingRegistry = OpRegistryHolder.tensorflow())
        println(importedModel)
    }


    @Test
    fun testRegistry() {
        val tensorflowOpRegistry = registry()
        val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess("Conv2D")
        println(mappingProcess)
    }



    @Test
    @Ignore
    fun testTensorflowMappingContext() {
        val tensorflowOpRegistry = registry()

        val absOpDef = tensorflowOpRegistry.lookupOpMappingProcess("Abs")
        val opDef = tensorflowOps.findOp("Abs")
        val absNodeDef = NodeDef {
            name = "input"
            Input("input1")
            op = "Abs"
        }

        val graph = GraphDef {
            Node(absNodeDef)
        }

        val tfIRGraph = TensorflowIRGraph(graphDef = graph,opDef = tensorflowOps,tensorflowOpMappingRegistry = tensorflowOpRegistry)

        val tfMappingCtx = TensorflowMappingContext(
            opDef =opDef,
            node = absNodeDef,
            graph = tfIRGraph,dynamicVariables = HashMap())

        assertEquals(opDef,tfMappingCtx.opDef)

    }




    @Test
    fun testInputOutputNames() {
        val tensorflowOpRegistry = registry()
        val tensorflowOpNames = tensorflowOpRegistry.inputFrameworkOpNames()
        val nd4jOpNames = tensorflowOpRegistry.nd4jOpNames()
        tensorflowOpRegistry.mappingProcessNames().map {
            tensorflowOpRegistry.lookupOpMappingProcess(it)
        }.forEach {
            println("Beginning processing of op ${it.inputFrameworkOpName()} and nd4j op ${it.opName()}")
            assertTrue(tensorflowOpNames.contains(it.inputFrameworkOpName()))
            assertTrue(nd4jOpNames.contains(it.opName()))
            val nd4jOpDef = tensorflowOpRegistry.lookupNd4jOpDef(it.opName())
            val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(it.inputFrameworkOpName())
            val inputNameArgDefs = nd4jOpDef.argDescriptorList.filter {
                    argDef -> argDef.argType == OpNamespace.ArgDescriptor.ArgType.INPUT_TENSOR
            }.map { argDef -> argDef.name }

            val inputFrameworkOpDefNames = tensorflowOpDef.inputArgList.map { tfOpDef -> tfOpDef.name}

            val nd4jArgDefNames = nd4jOpDef.argDescriptorList.map { nd4jArgDef -> nd4jArgDef.name }
            val tfAttrNames = tensorflowOpDef.attrList.map { tfAttr -> tfAttr.name }
            it.tensorMappingRules().forEach { tensorRules ->
                println("Running tensor mapping rule ${tensorRules.name()} for op ${it.inputFrameworkOpName()} and nd4j op name ${it.opName()}")
                run {
                    tensorRules.mappingNamesToPerform().forEach { tensorRule ->
                        run {
                            println("Testing assertion for nd4j name ${tensorRule.key} and input name ${tensorRule.value}")
                            assertTrue(inputNameArgDefs.contains(tensorRule.key)) ?: error("Failed on inputArgName ${tensorRule.key}")
                            assertTrue(inputFrameworkOpDefNames.contains(tensorRule.value)) ?: error("Failed on inputArgName ${tensorRule.value}")
                        }

                    }
                }

            }

            println("Running attribute mapping rules for ${it.opName()} and input op name ${it.inputFrameworkOpName()}")
            it.attributeMappingRules().forEach { attrRule ->
                run {
                    attrRule.mappingNamesToPerform().forEach { attrMapping ->
                        run {
                            println("Testing nd4j name  ${attrMapping.key} and input framework name ${attrMapping.value}")
                            assertTrue(nd4jArgDefNames.contains(attrMapping.key) || inputNameArgDefs.contains(attrMapping.key))
                            assertTrue(tfAttrNames.contains(attrMapping.value)  || inputFrameworkOpDefNames.contains(attrMapping.value))
                        }

                    }
                }
            }

        }
    }


    @Test
    @Ignore
    @org.junit.jupiter.api.Disabled
    fun testOpExecution() {
        Nd4j.getRandom().setSeed(12345)
        Nd4j.getEnvironment().isDebug = true
        Nd4j.getEnvironment().isVerbose = true
        Nd4j.getEnvironment().isProfiling = true
        val scalarInputs = mapOf(
            "abs" to -1.0,
            "acos" to 1.0,
            "acosh" to 1.0,
            "asin" to 1.0,
            "asinh" to 1.0,
            "atan" to 1.0,
            "atanh" to 0.5,
            "ceil" to 1.0,
            "copy" to 1.0,
            "cos" to 1.0,
            "cosh" to 1.0,
            "erf" to 1.0,
            "elu" to 1.0,
            "erfc" to 1.0,
            "exp" to 1.0,
            "expm1" to 1.0,
            "floor" to 1.0,
            "identity" to 1.0,
            "isfinite" to 1.0,
            "isinf" to 1.0,
            "isnan" to 1.0,
            //"identity_n" to 1.0,
            "log" to 1.0,
            "log1p" to 1.0,
            "neg" to 1.0,
            "ones_as" to 1.0,
            "Reciprocal" to 1.0,
            "rank" to 1.0,
            "relu6" to 1.0,
            "rint" to 1.0,
            "round" to 1.0,
            "rsqrt" to 1.0,
            "sigmoid" to 1.0,
            "sign" to 1.0,
            "size" to 1.0,
            "sin" to 1.0,
            "sinh" to 1.0,
            "square" to 1.0,
            "sqrt" to 1.0,
            "tan" to 1.0,
            "tanh" to 1.0,
            "selu" to 1.0,
            "softsign" to 1.0,
            "softplus" to 1.0,
            "zeroslike" to 1.0)

        val singleInputOps = scalarInputs.keys

        val pairWiseInputs = mapOf(
            "add" to listOf(1.0,1.0),
            "divide" to listOf(1.0,1.0),
            "greater" to listOf(1.0,1.0),
            "less" to listOf(1.0,1.0),
            "less_equal" to listOf(1.0,1.0),
            "multiply" to listOf(1.0,1.0),
            "floordiv" to listOf(1.0,1.0),
            "mod" to listOf(1.0,1.0),
            "squaredsubtract" to listOf(1.0,1.0),
            "not_equals" to listOf(1.0,1.0),
            "realdiv" to listOf(1.0,1.0),
            "tf_atan2" to listOf(1.0,1.0),
            "maximum" to listOf(0.0,1.0),
            "min_pairwise" to listOf(1.0,1.0),
            "greater_equal" to listOf(1.0,1.0),
            "equals" to listOf(1.0,1.0),
            "min_pairwise" to listOf(1.0,1.0),
            "divide_no_nan" to listOf(1.0,1.0),
            "zeta" to listOf(2.0,3.0)


        )





        /**
         * Control flow ops
         */

        /**
         * Random distribution ops
         */


        /**
         * Creation ops
         * Empty
         * CopyHost
         * Linspace
         * OnesLike
         */

        /**
         * Scatter ops:
         * scatter_div
         * scatter_add
         * scatter_sub
         * scatter_min
         * scatter_mul
         * scatter_update
         * scatter_nd
         * scatter_nd_add
         * scatter_nd_sub
         * scatter_nd_update
         */




        val pairWiseIntOps = mapOf(
            "fmod" to listOf(1,1),
            "rshift_bits" to listOf(1,1),
            "truncatediv" to listOf(1,1),
            "bitwise_and" to listOf(1,1),
            "bitwise_or" to listOf(1,1),
            "bitwise_xor" to listOf(1,1),
            "shift_bits" to listOf(1,1)
        )

        val pairWiseNames = pairWiseInputs.keys


        val booleanReduceOps = mapOf(
            "all" to Nd4j.create(listOf(true,false,true,false).toBooleanArray()).reshape(2,2),
            "any" to Nd4j.create(listOf(true,false,true,false).toBooleanArray()).reshape(2,2)
        )

        val singularReduceOps = mapOf(
            "reduce_mean" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_prod" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_min" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_sum" to Nd4j.linspace(1,4,4).reshape(2,2),
            "reduce_max" to Nd4j.linspace(1,4,4).reshape(2,2)
        )




        val mappedOps = setOf(
            "Assert",
            "gather_nd",
            "lstmBlock",
            "lstmBlockCell",
            "cast",
            "gruCell",
            "igamma",
            "igammac",
            "lgamma",
            "mergeadd",
            "reduce_logsumexp",
            "check_numerics",
            "adjust_hue",
            "adjust_saturation",
            "adjust_contrast_v2",
            "reverse_sequence",
            "depthwise_conv2d",
            "resize_nearest_neighbor",
            "scatter_nd",
            "resize_area",
            "rgb_to_hsv",
            "resize_bicubic",
            "resize_bilinear",
            "listdiff",
            "mirror_pad",
            "histogram_fixed_width",
            "extract_image_patches",
            "ClipByValue",
            "crop_and_resize",
            "broadcast_dynamic_shape",
            "broadcastgradientargs",
            "lrn",
            "batch_to_space_nd",
            "space_to_batch_nd",
            "draw_bounding_boxes",
            "fused_batch_norm",
            "conv3dnew",
            "avgpool3dnew",
            "maxpool3dnew",
            "create",
            "slice",
            "strided_slice",
            "select",
            "compare_and_bitpack",
            "bincount",
            "broadcast_to",
            "biasadd",
            "condition",
            "avgpool2d",
            "maxpool2d",
            "conv2d",
            "dilation2d",
            "batch_to_space",
            "space_to_batch",
            "dynamic_partition",
            "dynamic_stitch",
            "softmax",
            "mergesum",
            "matrix_set_diag",
            "matrix_diag_part",
            "identity_n",
            "split",
            "split_v",
            "shapes_of",
            "squeeze",
            "bitcast",
            "merge_sum",
            "tile",
            "matmul",
            "range",
            "lin_space",
            "gather",
            "betainc",
            "concat",
            "stack",
            "unstack",
            "merge",
            "leakyrelu",
            "shape_of",
            "roll",
            "reverse",
            "relu",
            "relu6",
            "argmin",
            "argmax",
            "cross",
            "cumsum",
            "cumprod",
            "diag",
            "diag_part",
            "digamma",
            "depth_to_space",
            "expand_dims",
            "toggle_bits",
            "invert_permutation",
            //"enter", TODO: deal with frames or maybe ignore?
            //"exit",
            "in_top_k",
            "top_k",
            "lu",
            "matrix_inverse",
            "matrix_determinant",
            "solve",
            "triangular_solve",
            "log_matrix_determinant",
            "cholesky",
            "reshape",
            "noop",
            "nth_element",
            "non_max_suppression_overlaps",
            "non_max_suppression",
            "non_max_suppression_v3",
            "onehot",
            "pad",
            "pow",
            "transpose",
            "space_to_depth",
            "Where",
            "unsorted_segment_max",
            "unsorted_segment_min",
            "unsorted_segment_prod",
            "unsorted_segment_sum",
            "unique_with_counts",
            "unique",
            "boolean_and",
            "boolean_not",
            "boolean_or",
            "segment_mean",
            "segment_min",
            "segment_max",
            "segment_prod",
            "segment_sum"

            //"scatter_add", Skipping due to different op validation
            //"scatter_sub", Skipping due to different op validation
            //"scatter_update", Skipping due to different op validation
            //"scatter_nd" Skipping due to different op validation
        )




        //Skipping due to using references rather than tensors
        //"scatter_nd_add",
        //"scatter_nd_sub",
        // "scatter_nd_update"
        // //"scatter_min",
        //            //"scatter_mul",)

        val singularReduceNames = singularReduceOps.keys
        val testedOps = HashSet<String>()
        //skip testing control flow
        val controlFlowOps = setOf("Switch","While","placeholder","next_iteration","enter","exit","loop_cond")
        val resourceOps = setOf("stack_list","size_list","scatter_list","read_list","split_list","gather_list")
        val refOps = setOf("assign","scatter_add","scatter_sub","scatter_update")
        val randomOps = setOf("random_gamma","random_crop","random_normal","random_poisson","random_shuffle","randomuniform")
        testedOps.addAll(randomOps)
        testedOps.addAll(controlFlowOps)
        testedOps.addAll(resourceOps)
        testedOps.addAll(refOps)
        val importGraph = ImportGraph<GraphDef,NodeDef,OpDef,TensorProto,OpDef.AttrDef,AttrValue,DataType>()
        val tensorflowOpRegistry = registry()
        tensorflowOpRegistry.mappingProcessNames().map { name ->
            tensorflowOpRegistry.lookupOpMappingProcess(name)
        }.forEach { mappingProcess ->
            val nd4jOpDef = tensorflowOpRegistry.lookupNd4jOpDef(mappingProcess.opName())
            val tensorflowOpDef = tensorflowOpRegistry.lookupInputFrameworkOpDef(mappingProcess.inputFrameworkOpName())

            if(singleInputOps.contains(nd4jOpDef.name) && tensorflowOpDef.name != "Variable" && tensorflowOpDef.name != "VariableV2" && tensorflowOpDef.name != "Const") {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("x")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }
                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
                val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,HashMap(),OpRegistryHolder.tensorflow()).enableDebugMode()!!
                Nd4j.getExecutioner().setProfilingConfig(ProfilerConfig.builder()
                    .stackTrace(true).build())
                val xVal =  Nd4j.scalar(scalarInputs[mappingProcess.opName()]).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal)
                if(!mappedGraph.hasVariable("output"))
                    throw IllegalStateException("No output variable found. Variables include ${mappedGraph.variables}")
                val tfResults = tensorflowRunner.run(inputs)
                val results = mappedGraph.output(inputs,"output")
                val tfOutput = tfResults["output"]!!
                assertTrue(tfOutput.isScalar)
                val nd4jOutput = results["output"]!!
                assertTrue(nd4jOutput.isScalar)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",nd4jOutput.getDouble(0), tfOutput.getDouble(0),1e-3)
                testedOps.add(nd4jOpDef.name)
            }
            else if(singularReduceNames.contains(nd4jOpDef.name)) {
                listOf(listOf(0),listOf(-1),listOf(0,1)).forEach { dimensions ->
                    listOf(true,false).forEach { keepDim ->
                        val tensorNode = NodeDef {
                            name = "x"
                            op = "Placeholder"
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                        }

                        val opNode = NodeDef {
                            Input("x")
                            Input("dimensions")
                            op = tensorflowOpDef.name
                            name = "output"
                            Attribute("T",AttrValue {
                                type = DataType.DT_DOUBLE
                            })
                            Attribute("Tidx",AttrValue {
                                type = DataType.DT_INT32
                            })
                            Attribute("keep_dims",AttrValue {
                                b = keepDim
                            })
                        }

                        val tensorNode2 = NodeDef {
                            op = "Const"
                            name = "dimensions"
                            Attribute("value",AttrValue {
                                tensor = TensorProto {
                                    Int32Data(dimensions)
                                    dtype = DataType.DT_INT32
                                    tensorShape = TensorShapeProto {
                                        Dims(listOf(1,dimensions.size.toLong()))
                                    }
                                }
                            })
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_INT32
                            })
                        }

                        val graphDef = GraphDef {
                            Node(tensorNode)
                            Node(tensorNode2)
                            Node(opNode)
                        }

                        val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                        val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,HashMap(),tensorflowOpRegistry)!!
                        val xVal =  singularReduceOps[mappingProcess.opName()]!!.castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x"),outputNames = listOf("output"))
                        val inputs = mapOf("x" to xVal)
                        val results = mappedGraph.output(inputs,"output")
                        val tfResults = tensorflowRunner.run(inputs)
                        //2 dimensions means sum the whole array, sometimes there are subtle differences in the shape like 1,1 vs a zero length array which is effectively the same thing
                        if(dimensions.size < 2)
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!, results["output"]!!)
                        else
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))

                    }

                }

                testedOps.add(nd4jOpDef.name)

            } else if(booleanReduceOps.keys.contains(nd4jOpDef.name)) {
                listOf(listOf(0),listOf(-1),listOf(0,1)).forEach { dimensions ->
                    listOf(true,false).forEach { keepDim ->
                        val tensorNode = NodeDef {
                            name = "x"
                            op = "Placeholder"
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_BOOL
                            })
                        }

                        val opNode = NodeDef {
                            Input("x")
                            Input("dimensions")
                            op = tensorflowOpDef.name
                            name = "output"

                            Attribute("Tidx",AttrValue {
                                type = DataType.DT_INT32
                            })
                            Attribute("keep_dims",AttrValue {
                                b = keepDim
                            })
                        }

                        val tensorNode2 = NodeDef {
                            op = "Const"
                            name = "dimensions"
                            Attribute("value",AttrValue {
                                tensor = TensorProto {
                                    Int32Data(dimensions)
                                    dtype = DataType.DT_INT32
                                    tensorShape = TensorShapeProto {
                                        Dims(listOf(1,dimensions.size.toLong()))
                                    }
                                }
                            })
                            Attribute("dtype",AttrValue {
                                type = DataType.DT_INT32
                            })
                        }

                        val graphDef = GraphDef {
                            Node(tensorNode)
                            Node(tensorNode2)
                            Node(opNode)
                        }

                        val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                        val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,HashMap(),OpRegistryHolder.tensorflow())!!
                        val xVal =  booleanReduceOps[mappingProcess.opName()]!!
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x"),outputNames = listOf("output"))
                        val inputs = mapOf("x" to xVal)
                        val results = mappedGraph.output(inputs,"output")
                        val tfResults = tensorflowRunner.run(inputs)
                        //2 dimensions means sum the whole array, sometimes there are subtle differences in the shape like 1,1 vs a zero length array which is effectively the same thing
                        if(dimensions.size < 2)
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!, results["output"]!!)
                        else
                            assertEquals("Function ${nd4jOpDef.name} failed with input $xVal and dimension ${dimensions}",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))

                    }

                }

                testedOps.add(nd4jOpDef.name)

            } else if(pairWiseNames.contains(nd4jOpDef.name)) {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "y"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }


                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val mappingProcess = tensorflowOpRegistry.lookupOpMappingProcess(tensorflowOpDef.name)
                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
                val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicVariables = hashMapOf("y" to TensorProto {
                    dtype = DataType.DT_DOUBLE
                    DoubleData(listOf(1.0))
                    Shape(listOf(1,1))
                }),OpRegistryHolder.tensorflow())!!

                val xVal =  Nd4j.scalar(pairWiseInputs[mappingProcess.opName()]!![0])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                val yVal =  Nd4j.scalar(pairWiseInputs[mappingProcess.opName()]!![1])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x","y"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal,"y" to yVal)
                val results = mappedGraph.output(inputs,"output")
                val tfResults = tensorflowRunner.run(inputs)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))
                testedOps.add(nd4jOpDef.name)

            } else if(pairWiseIntOps.contains(nd4jOpDef.name)) {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "y"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("y")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val tensorflowGraph = TensorflowIRGraph(graphDef, tensorflowOps,tensorflowOpRegistry)
                val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,HashMap(),OpRegistryHolder.tensorflow())!!
                val xVal =  Nd4j.scalar(pairWiseIntOps[mappingProcess.opName()]!![0])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val yVal =  Nd4j.scalar(pairWiseIntOps[mappingProcess.opName()]!![1])
                    .reshape(1,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val tensorflowRunner = TensorflowIRGraphRunner(irGraph =   tensorflowGraph,inputNames = listOf("x","y"),outputNames = listOf("output"))
                val inputs = mapOf("x" to xVal,"y" to yVal)
                val results = mappedGraph.output(inputs,"output")
                val tfResults = tensorflowRunner.run(inputs)
                assertEquals("Function ${nd4jOpDef.name} failed with input $xVal",tfResults["output"]!!.reshape(1,1), results["output"]!!.reshape(1,1))
                testedOps.add(nd4jOpDef.name)

            } else if(mappedOps.contains(mappingProcess.opName())) {
                val graphInputList = graphForOp(nd4jOpName = mappingProcess.opName(),inputFrameworkOpName = mappingProcess.inputFrameworkOpName())
                graphInputList.forEach { graphInput ->
                    val tensorflowGraph = TensorflowIRGraph(graphInput.graphDef, tensorflowOps,tensorflowOpRegistry)
                    val dynamicOpsMap = HashMap<String,TensorProto>()
                    graphInput.inputArrays.forEach { k, v ->
                        dynamicOpsMap[k] = convertNDArrayToTensorflowTensor(v)
                    }

                    //NOTE: The output name here is different than the output names from samediff because we want every array from tensorflow for assertion purposes.
                    //The outputs from samediff might be slightly different (eg: not have every output tensorflow does or more)

                    //tf2 ops don't currently work in nd4j-tensorflow and can't be verified
                    val tf2Ops = setOf("CheckNumericsV2","FusedBatchNormV3","ParallelConcat","FusedBatchNorm","FusedBatchNormV2")
                    //these ops reflect ops that should generally be tested other ways and are usually tested down below
                    val bannedOps = setOf("noop","unique","unique_with_counts","matrix_determinant","log_matrix_determinant","Assert","split_v","identity_n","dynamic_partition","dynamic_stitch","draw_bounding_boxes","fused_batch_norm")
                    if(!bannedOps.contains(mappingProcess.opName()) && !tf2Ops.contains(mappingProcess.inputFrameworkOpName())) {
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        if(mappingProcess.opName() == "bincount") {
                            val inputVal = Nd4j.create(doubleArrayOf(1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0))
                                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                            val sizeVal = Nd4j.create(doubleArrayOf(3.0))
                                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                            val weightVal = Nd4j.create(doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                            println(Nd4j.getExecutioner().exec(DynamicCustomOp.builder("bincount").addInputs(inputVal,weightVal).addIntegerArguments(0,3).build())[0])
                            println()
                        }
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames} " +
                                "with tfValue of shape ${tfResults.values.first().shapeInfoToString()} and nd4j ${results.values.first().shapeInfoToString()} and ${graphInput}"
                            ,tfResults.values.first(), results.values.first())
                    } else if(mappingProcess.opName() == "unique_with_counts" || mappingProcess.opName() == "unique") {
                        //note: this is a separate case since the results are equal, minus dimensions
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults.values.first().ravel(), results.values.first().ravel())
                    }//slight difference in scalar result, doesn't matter in practice
                    else if(mappingProcess.opName() == "matrix_determinant" || mappingProcess.opName() == "log_matrix_determinant") {
                        //note: this is a separate case since the results are equal, minus dimensions
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        if(mappingProcess.opName() == "matrix_determinant") {
                            val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                            val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                            assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults["output"]!!.ravel().getDouble(0), results["output"]!!.ravel().getDouble(0),1e-3)

                        }
                    }
                    else if(mappingProcess.opName() == "split_v" || mappingProcess.opName() == "identity_n" || mappingProcess.opName() == "dynamic_partition"|| mappingProcess.opName() == "dynamic_stitch") {
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults, results)

                    } else if(mappingProcess.opName() == "draw_bounding_boxes") {
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)
                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())
                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults, results)

                    }
                    else if(mappingProcess.opName() == "fused_batch_norm" && !tf2Ops.contains(mappingProcess.inputFrameworkOpName())) {
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults["y"], results["y"])

                    }

                    else  if(!bannedOps.contains(mappingProcess.opName()) && !tf2Ops.contains(mappingProcess.inputFrameworkOpName())) {
                        //note that log outputs 2 results and the 2nd one is the one we need. The first result is a sign.
                        val tensorflowRunner = TensorflowIRGraphRunner(irGraph =  tensorflowGraph,inputNames = graphInput.inputNames,outputNames = graphInput.outputNames)


                        val mappedGraph = importGraph.importGraph(tensorflowGraph,null,null,dynamicOpsMap,OpRegistryHolder.tensorflow())
                        assertEquals("Input name mismatch with input array elements",graphInput.inputArrays.keys,graphInput.inputNames.toSet())

                        val tfResults = tensorflowRunner.run(graphInput.inputArrays)
                        val results = mappedGraph!!.output(graphInput.inputArrays,graphInput.outputNames)
                        assertEquals("Function ${nd4jOpDef.name} failed with input ${graphInput.inputNames}",tfResults["finalResult"]!!.ravel().getDouble(0), results["finalResult"]!!.ravel().getDouble(0),1e-3)

                    }

                }

                testedOps.add(nd4jOpDef.name)

            }
        }

        val differenceOfSet = tensorflowOpRegistry.mappedNd4jOpNames() - testedOps
        println("Ops left to test is ${differenceOfSet.size} and ops are $differenceOfSet with total ops ran ${testedOps.size}")
        println("Note we skipped ${controlFlowOps.size} testing control flow ops named $controlFlowOps")
        println("Note we skipped ${resourceOps.size} testing resource ops named $resourceOps due to resources being handled differently than normal tensors")
        println("Note we skipped ${refOps.size} testing resource ops named $refOps due to references being handled differently than normal tensors")
        println("Note we skipped ${randomOps.size} testing resource ops named $randomOps due to random not being consistently testable. This may change in the short term.")

    }

}