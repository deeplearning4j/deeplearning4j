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


import org.apache.commons.io.IOUtils
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.api.ops.DynamicCustomOp
import org.nd4j.linalg.api.ops.impl.transforms.BinCount
import org.nd4j.linalg.api.ops.impl.transforms.floating.RSqrt
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.profiler.ProfilerConfig
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.registry
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner
import org.tensorflow.framework.*
import java.lang.IllegalStateException
import java.nio.charset.Charset
import kotlin.math.max

fun graphForOp(nd4jOpName: String,inputFrameworkOpName: String): List<GraphInput> {
    val tensorflowOpDef = registry().lookupInputFrameworkOpDef(inputFrameworkOpName)
    when (nd4jOpName) {
        "check_numerics" -> {
            val tensor = NodeDef {
                name = "tensor"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("tensor")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("message",AttrValue {
                    s = ByteString.copyFrom("test message".toByteArray(Charset.defaultCharset()))
                })
            }

            val graphDef = GraphDef {
                Node(tensor)
                Node(opNode)
            }



            val xVal = Nd4j.create(floatArrayOf(1.0f,2.0f,3.0f))
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("tensor" to xVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("tensor"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "gruCell" -> {
            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val hPrev = NodeDef {
                name = "h_prev"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val wRu = NodeDef {
                name = "w_ru"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val wC = NodeDef {
                name = "w_c"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val bRu = NodeDef {
                name = "b_ru"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val bc = NodeDef {
                name = "b_c"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("x")
                Input("h_prev")
                Input("w_ru")
                Input("w_c")
                Input("b_ru")
                Input("b_c")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val r = NodeDef {
                name = "r"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val u = NodeDef {
                name = "u"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val c = NodeDef {
                name = "c"
                Input("output:2")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val h = NodeDef {
                name = "h"
                Input("output:3")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val graphDef = GraphDef {
                Node(x)
                Node(hPrev)
                Node(wRu)
                Node(wC)
                Node(bRu)
                Node(bc)
                Node(opNode)
                Node(r)
                Node(u)
                Node(c)
                Node(h)
            }




            val xVal = Nd4j.linspace(1,20,20).reshape(2,10)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val hPrevVal = Nd4j.linspace(1,8,8).reshape(2,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val wRuVal = Nd4j.linspace(1,112,112).reshape(14,8)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val wcVal = Nd4j.linspace(1,56,56).reshape(14,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val bRuVal = Nd4j.linspace(1,8,8).reshape(8)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val bcVal = Nd4j.linspace(1,4,4).reshape(4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("x" to xVal,"h_prev" to hPrevVal,"w_ru" to wRuVal,"w_c" to wcVal,"b_ru" to bRuVal,"b_c" to bcVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("x","h_prev","w_ru","w_c","b_ru","b_c"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "lstmBlockCell" -> {
            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val csPrev = NodeDef {
                name = "cs_prev"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val hPrev = NodeDef {
                name = "h_prev"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val w = NodeDef {
                name = "w"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val wci = NodeDef {
                name = "wci"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val wcf = NodeDef {
                name = "wcf"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val wco = NodeDef {
                name = "wco"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val bias = NodeDef {
                name = "b"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }
            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("x")
                Input("cs_prev")
                Input("h_prev")
                Input("w")
                Input("wci")
                Input("wcf")
                Input("wco")
                Input("b")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("forget_bias",AttrValue {
                    f = 2.0f
                })

                Attribute("use_peephole",AttrValue {
                    b = false
                })
            }


            val i = NodeDef {
                name = "i"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val cs = NodeDef {
                name = "cs"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val f = NodeDef {
                name = "f"
                Input("output:2")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val o = NodeDef {
                name = "o"
                Input("output:3")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val ci = NodeDef {
                name = "ci"
                Input("output:4")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val h = NodeDef {
                name = "h"
                Input("output:5")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(x)
                Node(csPrev)
                Node(hPrev)
                Node(w)
                Node(wci)
                Node(wcf)
                Node(wco)
                Node(bias)
                Node(opNode)
                Node(i)
                Node(cs)
                Node(f)
                Node(o)
                Node(ci)
                Node(h)
            }




            val xVal = Nd4j.linspace(1,5,5).reshape(1,5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val csPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val hPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val wVal = Nd4j.linspace(1,96,96).reshape(8,12)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val wciVal = Nd4j.linspace(1,3,3).reshape(3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val wcfVal = Nd4j.linspace(1,3,3).reshape(3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val wcoVal = Nd4j.linspace(1,3,3).reshape(3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val bVal = Nd4j.zeros(12)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)




            val inputs = mapOf("x" to xVal,"cs_prev" to csPrevVal,"h_prev" to hPrevVal,"w" to wVal,"wci" to wciVal,"wcf" to wcfVal,"wco" to wcoVal,"b" to bVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("x","cs_prev","h_prev","w","wci","wcf","wco","b"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "lstmBlock" -> {
            if(inputFrameworkOpName == "BlockLSTM") {
                val seqLenMax = NodeDef {
                    name = "seq_len_max"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val x = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val csPrev = NodeDef {
                    name = "cs_prev"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val hPrev = NodeDef {
                    name = "h_prev"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val w = NodeDef {
                    name = "w"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val wci = NodeDef {
                    name = "wci"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val wcf = NodeDef {
                    name = "wcf"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val wco = NodeDef {
                    name = "wco"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val bias = NodeDef {
                    name = "b"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }
                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("seq_len_max")
                    Input("x")
                    Input("cs_prev")
                    Input("h_prev")
                    Input("w")
                    Input("wci")
                    Input("wcf")
                    Input("wco")
                    Input("b")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("forget_bias", AttrValue {
                        f = 2.0f
                    })
                    Attribute("forget_bias", AttrValue {
                        f = 3.0f
                    })
                    Attribute("use_peephole", AttrValue {
                        b = false
                    })
                }


                val i = NodeDef {
                    name = "i"
                    Input("output:0")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val cs = NodeDef {
                    name = "cs"
                    Input("output:1")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val f = NodeDef {
                    name = "f"
                    Input("output:2")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val o = NodeDef {
                    name = "o"
                    Input("output:3")
                    op = "Identity"
                    Attribute("T",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val ci = NodeDef {
                    name = "ci"
                    Input("output:4")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val h = NodeDef {
                    name = "h"
                    Input("output:5")
                    op = "Identity"
                    Attribute("T",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val graphDef = GraphDef {
                    Node(seqLenMax)
                    Node(x)
                    Node(csPrev)
                    Node(hPrev)
                    Node(w)
                    Node(wci)
                    Node(wcf)
                    Node(wco)
                    Node(bias)
                    Node(opNode)
                    Node(i)
                    Node(cs)
                    Node(f)
                    Node(o)
                    Node(ci)
                    Node(h)
                }



                val seqLenVal = Nd4j.scalar(5.0)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

                val xVal = Nd4j.linspace(1,20,20).reshape(5,1,4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val csPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val hPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wVal = Nd4j.linspace(1,84,84).reshape(7,12)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wciVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wcfVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


                val wcoVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


                val bVal = Nd4j.zeros(12)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)




                val inputs = mapOf("seq_len_max" to seqLenVal,"x" to xVal,"cs_prev" to csPrevVal,"h_prev" to hPrevVal,"w" to wVal,"wci" to wciVal,"wcf" to wcfVal,"wco" to wcoVal,"b" to bVal)


                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("seq_len_max","x","cs_prev","h_prev","w","wci","wcf","wco","b"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else { //BlockLSTMV2
                val seqLenMax = NodeDef {
                    name = "seq_len_max"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val x = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val csPrev = NodeDef {
                    name = "cs_prev"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val hPrev = NodeDef {
                    name = "h_prev"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val w = NodeDef {
                    name = "w"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val wci = NodeDef {
                    name = "wci"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val wcf = NodeDef {
                    name = "wcf"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val wco = NodeDef {
                    name = "wco"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val bias = NodeDef {
                    name = "b"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }
                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("seq_len_max")
                    Input("x")
                    Input("cs_prev")
                    Input("h_prev")
                    Input("w")
                    Input("wci")
                    Input("wcf")
                    Input("wco")
                    Input("b")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })

                    Attribute("use_peephole", AttrValue {
                        b = false
                    })
                }


                val i = NodeDef {
                    name = "i"
                    Input("output:0")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val cs = NodeDef {
                    name = "cs"
                    Input("output:1")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val f = NodeDef {
                    name = "f"
                    Input("output:2")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val o = NodeDef {
                    name = "o"
                    Input("output:3")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val ci = NodeDef {
                    name = "ci"
                    Input("output:4")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val h = NodeDef {
                    name = "h"
                    Input("output:5")
                    op = "Identity"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val graphDef = GraphDef {
                    Node(seqLenMax)
                    Node(x)
                    Node(csPrev)
                    Node(hPrev)
                    Node(w)
                    Node(wci)
                    Node(wcf)
                    Node(wco)
                    Node(bias)
                    Node(opNode)
                    Node(i)
                    Node(cs)
                    Node(f)
                    Node(o)
                    Node(ci)
                    Node(h)
                }



                val seqLenVal = Nd4j.scalar(5.0)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

                val xVal = Nd4j.linspace(1,20,20).reshape(5,1,4)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val csPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val hPrevVal = Nd4j.linspace(1,3,3).reshape(1,3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wVal = Nd4j.linspace(1,84,84).reshape(7,12)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wciVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val wcfVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


                val wcoVal = Nd4j.linspace(1,3,3).reshape(3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


                val bVal = Nd4j.zeros(12)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)




                val inputs = mapOf("seq_len_max" to seqLenVal,"x" to xVal,"cs_prev" to csPrevVal,"h_prev" to hPrevVal,"w" to wVal,"wci" to wciVal,"wcf" to wcfVal,"wco" to wcoVal,"b" to bVal)


                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("seq_len_max","x","cs_prev","h_prev","w","wci","wcf","wco","b"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }
        }



        "adjust_hue","adjust_saturation" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val delta = NodeDef {
                name = "delta"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("delta")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(input)
                Node(delta)
                Node(opNode)
            }



            val xVal = Nd4j.zeros(3,3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val deltaVal = Nd4j.scalar(0.5).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val inputs = mapOf("input" to xVal,"delta" to deltaVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","delta"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "adjust_contrast_v2" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val delta = NodeDef {
                name = "contrast_factor"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("contrast_factor")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(input)
                Node(delta)
                Node(opNode)
            }



            val xVal = Nd4j.zeros(3,3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val deltaVal = Nd4j.scalar(0.5).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val inputs = mapOf("input" to xVal,"contrast_factor" to deltaVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","contrast_factor"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "rgb_to_hsv" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }



            val xVal = Nd4j.zeros(3,3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("input" to xVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "reverse_sequence" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val seqLengths = NodeDef {
                name = "seq_lengths"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("seq_lengths")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("Tlen",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("seq_dim",AttrValue {
                    i = 2
                })
                Attribute("batch_dim",AttrValue {
                    i = 1
                })
            }

            val graphDef = GraphDef {
                Node(input)
                Node(seqLengths)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,60,60).reshape(3,4,5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(4f,4f,4f,4f))
                .reshape(4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("input" to xVal,"seq_lengths" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","seq_lengths"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "resize_nearest_neighbor" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("size")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(size)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,36,36).reshape(1,3,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(6f,6f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("images" to xVal,"size" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","size"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "resize_bilinear" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("size")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(size)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,36,36).reshape(1,3,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(6f,6f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("images" to xVal,"size" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","size"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "resize_bicubic" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("size")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(size)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,36,36).reshape(1,3,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(6f,6f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("images" to xVal,"size" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","size"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "resize_area" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("size")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(size)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,36,36).reshape(1,3,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(6f,6f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("images" to xVal,"size" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","size"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "mirror_pad" -> {
            val mirrorPadRet = ArrayList<GraphInput>()
            listOf("REFLECT","SYMMETRIC").forEach { mode ->
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val paddings = NodeDef {
                    name = "paddings"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("input")
                    Input("paddings")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("mode", AttrValue {
                        s = ByteString.copyFrom(mode.toByteArray(Charset.defaultCharset()))
                    })
                    Attribute("Tpaddings", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val graphDef = GraphDef {
                    Node(input)
                    Node(paddings)
                    Node(opNode)
                }



                val xVal = Nd4j.linspace(1,5,5).reshape(5)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val yVal = Nd4j.create(floatArrayOf(1f,1f))
                    .reshape(1,2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



                val inputs = mapOf("input" to xVal,"paddings" to yVal)


                mirrorPadRet.add(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("input","paddings"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

            return mirrorPadRet
        }

        "listdiff" -> {
            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val y = NodeDef {
                name = "y"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
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
                Node(x)
                Node(y)
                Node(opNode)
            }



            val xVal = Nd4j.linspace(1,4,4).reshape(4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val yVal = Nd4j.create(floatArrayOf(3f,1f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)



            val inputs = mapOf("x" to xVal,"y" to yVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("x","y"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "histogram_fixed_width" -> {
            val values = NodeDef {
                name = "values"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val valueRange = NodeDef {
                name = "value_range"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val nBins = NodeDef {
                name = "nbins"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("values")
                Input("value_range")
                Input("nbins")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(values)
                Node(valueRange)
                Node(nBins)
                Node(opNode)
            }



            val valuesVal = Nd4j.ones(2,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val valueRangeVal = Nd4j.create(floatArrayOf(0f,5f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val nbinsVal = Nd4j.scalar(5f)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("values" to valuesVal,"value_range" to valueRangeVal,"nbins" to nbinsVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("values","value_range","nbins"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "extract_image_patches" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("images")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("ksizes",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("rates",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
            }
            val graphDef = GraphDef {
                Node(images)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2


            val imagesVal = Nd4j.ones(2,4,4,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)



            val inputs = mapOf("images" to imagesVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "crop_and_resize" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val boxes = NodeDef {
                name = "boxes"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val boxesI = NodeDef {
                name = "boxesI"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val cropSize = NodeDef {
                name = "cropSize"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("boxes")
                Input("boxesI")
                Input("cropSize")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(boxes)
                Node(boxesI)
                Node(cropSize)
                Node(opNode)
            }



            val imagesVal = Nd4j.create(floatArrayOf(1f,2f,3f,4f))
                .reshape(1,2,2,1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val boxesVal = Nd4j.create(floatArrayOf(0f,0f,1f,1f))
                .reshape(1,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val boxesIVal = Nd4j.create(floatArrayOf(0f))
                .reshape(1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val cropSizeVal = Nd4j.create(floatArrayOf(1f,1f))
                .reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("images" to imagesVal,"boxes" to boxesVal,"boxesI" to boxesIVal,"cropSize" to cropSizeVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","boxes","boxesI","cropSize"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "broadcastgradientargs" -> {
            val s0 = NodeDef {
                name = "s0"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val s1 = NodeDef {
                name = "s1"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("s0")
                Input("s1")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(s0)
                Node(s1)
                Node(opNode)
            }



            val s0Val = Nd4j.create(floatArrayOf(2f,2f,2f))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val s1Val = Nd4j.create(floatArrayOf(2f,1f,2f))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("s0" to s0Val,"s1" to s1Val)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("s0","s1"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "broadcast_dynamic_shape" -> {
            val s0 = NodeDef {
                name = "s0"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val s1 = NodeDef {
                name = "s1"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("s0")
                Input("s1")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(s0)
                Node(s1)
                Node(opNode)
            }



            val s0Val = Nd4j.create(floatArrayOf(2f,2f,2f))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val s1Val = Nd4j.create(floatArrayOf(2f,1f,2f))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("s0" to s0Val,"s1" to s1Val)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("s0","s1"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "lrn" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("depth_radius",AttrValue {
                    i = 5
                })
                Attribute("bias",AttrValue {
                    f = 1f
                })
                Attribute("alpha",AttrValue {
                    f = 0.5f
                })
                Attribute("beta",AttrValue {
                    f = 0.5f
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2

            //1, 1,2,2,1, 1,2,2,1

            val inputVal = Nd4j.linspace(1,16,16).reshape(2,2,2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("input" to inputVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "fused_batch_norm" -> {
            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val scale = NodeDef {
                name = "scale"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val offset = NodeDef {
                name = "offset"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val mean = NodeDef {
                name = "mean"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val variance = NodeDef {
                name = "variance"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            val epsilon = 0.0001f
            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("x")
                Input("scale")
                Input("offset")
                Input("mean")
                Input("variance")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("U",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("is_training",AttrValue {
                    b = false
                })
                Attribute("data_format",AttrValue {
                    s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                })
                Attribute("epsilon",AttrValue {
                    f = epsilon
                })
            }


            val y = NodeDef {
                name = "y"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val batchMean = NodeDef {
                name = "batch_mean"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val batchVariance = NodeDef {
                name = "batch_variance"
                Input("output:2")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(x)
                Node(scale)
                Node(mean)
                Node(offset)
                Node(variance)
                Node(opNode)
                Node(y)
                Node(batchMean)
                Node(batchVariance)
            }



            val xVal = Nd4j.ones(2,2,2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val scaleVal = Nd4j.zeros(2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)
            val offsetVal = Nd4j.zeros(2).addi(2).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            //    xAffected *= (*variance + epsilon).transform(transform::RSqrt) * (*scale) + (*offset);
            val testResult = Nd4j.ones(8,2).muli(Nd4j.exec(RSqrt(Nd4j.scalar(epsilon)))).muli(scaleVal).addi(offsetVal)
            val meanVal = Nd4j.zeros(2)
            val varianceVal = Nd4j.zeros(2)
            val otherResult = xVal.sub(meanVal).div(varianceVal.add(epsilon)).mul(scaleVal).add(offsetVal)
            // (batch - self.moving_mean) / (self.moving_var + epsilon) * gamma + beta.

            val inputs = mapOf("x" to xVal,"scale" to scaleVal,"mean" to meanVal,"offset" to offsetVal,"variance" to varianceVal)

            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("x","scale","offset","mean","variance"),
                outputNames = listOf("y","batch_mean","batch_variance"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "conv3dnew" -> {
            // int bS=2, iD=3,iH=4,iW=3,  iC=4,oC=3,  kD=2,kH=3,kW=2,  sD=1,sH=1,sW=1,  pD=0,pH=0,pW=0,  dD=1,dH=1,dW=1;
            // int paddingMode = 1;             // 1-SAME,  0-VALID;
            //int dataFormat  = 1;             // 1-NDHWC, 0-NCDHW
            //2,3,4,3,4
            //2,3,2,4,3
            //auto input    = NDArrayFactory::create<TypeParam>('c', {bS, iD, iH, iW, iC});
            //auto weights  = NDArrayFactory::create<TypeParam>('c', {kD, kH, kW, iC, oC});
//, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW, dD,dH,dW, paddingMode, 1, dataFormat}
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val filter = NodeDef {
                name = "filter"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            //, {kD,kH,kW,  sD,sH,sW,  pD,pH,pW, dD,dH,dW, paddingMode, 1, dataFormat}

            val opNode = NodeDef {
                Input("input")
                Input("filter")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
                Attribute("data_format",AttrValue {
                    s = ByteString.copyFrom("NDHWC".toByteArray(Charset.defaultCharset()))
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(filter)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2


            val inputVal = Nd4j.ones(2,3,4,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val filterVal = Nd4j.ones(2,3,2,4,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"filter" to filterVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","filter"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "avgpool3dnew","maxpool3dnew" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("ksize",AttrValue {
                    ListInts(listOf(1,1,1,1,1))
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
                Attribute("data_format",AttrValue {
                    s = ByteString.copyFrom("NDHWC".toByteArray(Charset.defaultCharset()))
                })
            }


            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }

            //2,3,3,43
            val inputVal = Nd4j.ones(2,3,3,4,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("input" to inputVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "draw_bounding_boxes" -> {
            val images = NodeDef {
                name = "images"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val boxes = NodeDef {
                name = "boxes"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val colors = NodeDef {
                name = "colors"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("images")
                Input("boxes")
                Input("colors")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val graphDef = GraphDef {
                Node(images)
                Node(boxes)
                Node(colors)
                Node(opNode)
            }



            val imagesVal = Nd4j.linspace(1,120,120).reshape(2,4,5,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val boxesVal = Nd4j.linspace(1,16,16).reshape(2,2,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)
            val colorVal = Nd4j.create(floatArrayOf(201f, 202f, 203f, 127f, 128f, 129f)).reshape(2,3)

            val inputs = mapOf("images" to imagesVal,"boxes" to boxesVal,"colors" to colorVal)

            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("images","boxes","colors"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "create" -> {
            val shape = NodeDef {
                name = "shape"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("shape")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("init",AttrValue {
                    b = true
                })
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val graphDef = GraphDef {
                Node(shape)
                Node(opNode)
            }



            val shapeVal = Nd4j.create(doubleArrayOf(1.0,2.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val inputs = mapOf("shape" to shapeVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("shape"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "select" -> {
            val condition = NodeDef {
                name = "condition"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val t = NodeDef {
                name = "t"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val e = NodeDef {
                name = "e"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("condition")
                Input("t")
                Input("e")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }
            val graphDef = GraphDef {
                Node(condition)
                Node(t)
                Node(e)
                Node(opNode)
            }



            val conditionVal = Nd4j.create(booleanArrayOf(true,false,false))
                .castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)

            val tVal = Nd4j.linspace(1,9,9).reshape(3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val eVal = Nd4j.create(doubleArrayOf(9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0))
                .reshape(3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("condition" to conditionVal,"t" to tVal,"e" to eVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("condition","t","e"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "compare_and_bitpack" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val threshold = NodeDef {
                name = "threshold"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }



            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("threshold")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(threshold)
                Node(opNode)
            }



            val inputVal = Nd4j.create(floatArrayOf(-12f, -11f, -10f, -9f, -8f, -7f, -6f, -5f, -4f, -3f, -2f, -1f, 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f)).reshape(2,3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val thresholdVal = Nd4j.scalar(2.0)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"threshold" to thresholdVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","threshold"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "strided_slice" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val begin = NodeDef {
                name = "begin"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val end = NodeDef {
                name = "end"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val strides = NodeDef {
                name = "strides"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("begin")
                Input("end")
                Input("strides")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Index",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("shrink_axis_mask",AttrValue {
                    i = 1
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(begin)
                Node(end)
                Node(strides)
                Node(opNode)
            }



            val inputVal = Nd4j.linspace(1,10,10).reshape(5,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val beginVal = Nd4j.create(doubleArrayOf(0.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val endVal = Nd4j.create(doubleArrayOf(1.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val strideVal = Nd4j.create(doubleArrayOf(1.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("input" to inputVal,"begin" to beginVal, "end" to endVal,"strides" to strideVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","begin","end","strides"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "bincount" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val weights = NodeDef {
                name = "weights"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("size")
                Input("weights")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(size)
                Node(weights)
                Node(opNode)
            }



            val inputVal = Nd4j.create(doubleArrayOf(1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 2.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val sizeVal = Nd4j.create(doubleArrayOf(3.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val weightVal = Nd4j.create(doubleArrayOf(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"size" to sizeVal, "weights" to weightVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","size","weights"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "broadcast_to" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val shape = NodeDef {
                name = "shape"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("shape")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tidx",AttrValue {
                    type = DataType.DT_INT64
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(shape)
                Node(opNode)
            }



            val inputVal = Nd4j.create(doubleArrayOf(2.0))
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val shapeVal = Nd4j.zeros(2).addi(4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

            val inputs = mapOf("input" to inputVal,"shape" to shapeVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","shape"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "condition" -> {
            val condition = NodeDef {
                name = "condition"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val t = NodeDef {
                name = "t"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val e = NodeDef {
                name = "e"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("condition")
                Input("t")
                Input("e")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }
            val graphDef = GraphDef {
                Node(condition)
                Node(t)
                Node(e)
                Node(opNode)
            }



            val conditionVal = Nd4j.create(booleanArrayOf(true,true,false,false)).reshape(2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.BOOL)

            val tVal = Nd4j.linspace(1,4,4).reshape(2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val eVal = Nd4j.linspace(1,4,4).reshape(2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("condition" to conditionVal,"t" to tVal,"e" to eVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("condition","t","e"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "biasadd" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val bias = NodeDef {
                name = "bias"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("bias")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(bias)
                Node(opNode)
            }



            val inputVal = Nd4j.linspace(1,2 * 3 * 3 * 2,2 * 3 * 3 * 2).reshape(2,3,3,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val biasVal = Nd4j.linspace(1,2,2).reshape(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"bias" to biasVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","bias"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "dilation2d" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val filter = NodeDef {
                name = "filter"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("input")
                Input("filter")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("rates",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(filter)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2

            //1, 1,2,2,1, 1,2,2,1

            val inputVal = Nd4j.linspace(1,2 * 6 * 6 * 3,2 * 6 * 6 * 3).reshape(2,6,6,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val filterVal = Nd4j.linspace(1,3 * 2 * 3,3 * 2 * 3).reshape(3,2,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"filter" to filterVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","filter"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "depthwise_conv2d" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val filter = NodeDef {
                name = "filter"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("input")
                Input("filter")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
                Attribute("data_format",AttrValue {
                    s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(filter)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2


            val inputVal = Nd4j.ones(2,4,3,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val filterVal = Nd4j.ones(3,2,2,2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"filter" to filterVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","filter"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "conv2d" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val filter = NodeDef {
                name = "filter"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
            val opNode = NodeDef {
                Input("input")
                Input("filter")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("strides",AttrValue {
                    ListInts(listOf(1,1,1,1))
                })
                Attribute("padding",AttrValue {
                    s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                })
                Attribute("data_format",AttrValue {
                    s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                })
            }
            val graphDef = GraphDef {
                Node(input)
                Node(filter)
                Node(opNode)
            }

            //1,2,5,4

            //3,2,2,2


            val inputVal = Nd4j.ones(1,4,1,1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val filterVal = Nd4j.ones(1,1,1,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"filter" to filterVal)


            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("input","filter"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "avgpool2d","maxpool2d" -> {
            if(tensorflowOpDef.name == "AvgPool" || tensorflowOpDef.name == "MaxPool") {
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
                val opNode = NodeDef {
                    Input("input")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("ksize", AttrValue {
                        ListInts(listOf(1, 1, 1, 1))
                    })
                    Attribute("strides", AttrValue {
                        ListInts(listOf(1, 1, 1, 1))
                    })
                    Attribute("padding", AttrValue {
                        s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                    })
                    Attribute("data_format", AttrValue {
                        s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                    })
                }


                val graphDef = GraphDef {
                    Node(input)
                    Node(opNode)
                }

                val inputVal = Nd4j.ones(2,4,4,2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                val inputs = mapOf("input" to inputVal)


                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("input"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else { //MaxPoolV2
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val ksize = NodeDef {
                    name = "ksize"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val stride = NodeDef {
                    name = "stride"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                println("Running test import process for op ${tensorflowOpDef.name}")
                // {2, 2, 2, 2, 0, 0, 1, 1, 1, 1, 1}
                val opNode = NodeDef {
                    Input("input")
                    Input("ksize")
                    Input("stride")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })

                    Attribute("padding", AttrValue {
                        s = ByteString.copyFrom("SAME".toByteArray(Charset.defaultCharset()))
                    })
                    Attribute("data_format", AttrValue {
                        s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                    })
                }


                val graphDef = GraphDef {
                    Node(input)
                    Node(ksize)
                    Node(stride)
                    Node(opNode)
                }

                val inputVal = Nd4j.ones(2,4,4,2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                val ksizeVal = Nd4j.create(floatArrayOf(1.0f,2.0f,2.0f,1.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                val strideVal = Nd4j.create(floatArrayOf(1.0f,2.0f,2.0f,1.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                val inputs = mapOf("input" to inputVal,"ksize" to ksizeVal,"stride" to strideVal)


                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("input","ksize","stride"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

        }


        "space_to_batch" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val paddings = NodeDef {
                name = "paddings"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("paddings")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tpaddings",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("block_size",AttrValue {
                    i = 2
                })
            }


            val graphDef = GraphDef {
                Node(input)
                Node(paddings)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,12,12).reshape(1,2,2,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val paddingsVal = Nd4j.zeros(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"paddings" to paddingsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","paddings"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "batch_to_space_nd" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val blockShape = NodeDef {
                name = "block_shape"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("shape",AttrValue {
                    shape = TensorShapeProto {
                        Dims(listOf(3))
                    }
                })
            }

            val crops = NodeDef {
                name = "crops"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("block_shape")
                Input("crops")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tblock_shape",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("Tcrops",AttrValue {
                    type = DataType.DT_INT32
                })

            }


            val graphDef = GraphDef {
                Node(input)
                Node(blockShape)
                Node(crops)
                Node(opNode)
            }

            val tVal = Nd4j.linspace(1,24,24).reshape(8,1,1,1,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val blockShapeVal = Nd4j.zeros(3).addi(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val cropsVal = Nd4j.zeros(3,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to tVal,"block_shape" to blockShapeVal,"crops" to cropsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","block_shape","crops"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }

        "space_to_batch_nd" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val blockShape = NodeDef {
                name = "block_shape"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("shape",AttrValue {
                    shape = TensorShapeProto {
                        Dims(listOf(3))
                    }
                })
            }

            val paddings = NodeDef {
                name = "paddings"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("block_shape")
                Input("paddings")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tblock_shape",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("Tpaddings",AttrValue {
                    type = DataType.DT_INT32
                })

            }


            val graphDef = GraphDef {
                Node(input)
                Node(blockShape)
                Node(paddings)
                Node(opNode)
            }

            val tVal = Nd4j.linspace(1,48,48).reshape(2,2,4,3,1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val blockShapeVal = Nd4j.create(floatArrayOf(2.0f,2.0f,3f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val paddingsVal = Nd4j.create(floatArrayOf(0f,0f,0f,2f,2f,1f)).reshape(3,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to tVal,"block_shape" to blockShapeVal,"paddings" to paddingsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","block_shape","paddings"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "batch_to_space" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val crops = NodeDef {
                name = "crops"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("crops")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tidx",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("block_size",AttrValue {
                    i = 2
                })
            }


            val graphDef = GraphDef {
                Node(input)
                Node(crops)
                Node(opNode)
            }

            val tVal = Nd4j.linspace(1,12,12).reshape(4,1,1,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val cropsVal = Nd4j.zeros(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to tVal,"crops" to cropsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","crops"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "slice" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val begin = NodeDef {
                name = "begin"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val size = NodeDef {
                name = "size"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("begin")
                Input("size")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Index",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(input)
                Node(begin)
                Node(size)
                Node(opNode)
            }

            val tVal = Nd4j.linspace(1,12,12).reshape(3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val beginVal = Nd4j.create(doubleArrayOf(0.0,1.0)).reshape(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
            val sizeVal = Nd4j.create(doubleArrayOf(0.0,1.0)).reshape(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to tVal,"begin" to beginVal,"size" to sizeVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","begin","size"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "ClipByValue" -> {
            val t = NodeDef {
                name = "t"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val clipValueMin = NodeDef {
                name = "clip_value_min"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val clipValueMax = NodeDef {
                name = "clip_value_max"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("t")
                Input("clip_value_min")
                Input("clip_value_max")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val graphDef = GraphDef {
                Node(t)
                Node(clipValueMin)
                Node(clipValueMax)
                Node(opNode)
            }

            val tVal = Nd4j.linspace(1,12,12).reshape(3,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val clipValueMinVal = Nd4j.scalar(0.0).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
            val clipValueMaxVal = Nd4j.scalar(1.0).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


            val inputs = mapOf("t" to tVal,"clip_value_min" to clipValueMinVal,"clip_value_max" to clipValueMaxVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("t","clip_value_min","clip_value_max"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }




        "squeeze" -> {
            val value = NodeDef {
                name = "value"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("value")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("squeeze_dims",AttrValue {
                    ListInts(listOf(2))
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(value)
                Node(opNode)
            }

            val valuesVal = Nd4j.linspace(1,12,12).reshape(3,4,1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("value" to valuesVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("value"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "identity_n" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val input2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("input2")
                op = tensorflowOpDef.name
                name = "output"

                Attribute("T",AttrValue {
                    ListDataType(listOf(DataType.DT_INT64,DataType.DT_INT64))
                })
            }


            val out0 = NodeDef {
                name = "out0"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val out1 = NodeDef {
                name = "out1"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(input)
                Node(input2)
                Node(opNode)
                Node(out0)
                Node(out1)
            }


            val inputVal = Nd4j.linspace(1,4,4)
                .reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","input2"),
                outputNames = listOf("out0","out1"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "shapes_of" -> {
            val input1 = NodeDef {
                name = "input1"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val input2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val opNode = NodeDef {
                Input("input1")
                Input("input2")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("N",AttrValue {
                    i = 2
                })

                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("out_type",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val out0 = NodeDef {
                name = "out0"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val out1 = NodeDef {
                name = "out1"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }



            val graphDef = GraphDef {
                Node(input1)
                Node(input2)
                Node(opNode)
                Node(out0)
                Node(out1)
            }

            val input1Val = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
            val input2Val = Nd4j.linspace(1,6,6).reshape(2,3).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input1" to input1Val,"input2" to input2Val)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input1","input2"),
                outputNames = listOf("out0","out1"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }



        "dynamic_stitch" -> {
            val indices1 = NodeDef {
                name = "indices"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val indices2 = NodeDef {
                name = "indices2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val data0 = NodeDef {
                name = "data0"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val data1 = NodeDef {
                name = "data1"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("indices")
                Input("indices2")
                Input("data0")
                Input("data1")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("N",AttrValue {
                    i = 2
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }





            val graphDef = GraphDef {
                Node(indices1)
                Node(indices2)
                Node(data0)
                Node(data1)
                Node(opNode)
            }

            val testGraph = GraphRunner.builder().graphBytes(graphDef.toByteArray()).build()

            val indicesVal = Nd4j.create(floatArrayOf(1.0f,3.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val indices2Val = Nd4j.create(floatArrayOf(5.0f,0.0f,2.0f,4.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val dataVal = Nd4j.create(floatArrayOf(-1f,-1f)).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
            val data2Val = Nd4j.create(floatArrayOf(0.1f,5.2f,4.3f,7.4f)).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("indices" to indicesVal,"indices2" to indices2Val,"data0" to dataVal,"data1" to data2Val)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("indices","indices2","data0","data1"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }

        "dynamic_partition" -> {
            val data = NodeDef {
                name = "data"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val partitions = NodeDef {
                name = "partitions"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("data")
                Input("partitions")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("num_partitions",AttrValue {
                    i = 2
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val out0 = NodeDef {
                name = "out0"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val out1 = NodeDef {
                name = "out1"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }



            val graphDef = GraphDef {
                Node(data)
                Node(partitions)
                Node(opNode)
                Node(out0)
                Node(out1)
            }

            val testGraph = GraphRunner.builder().graphBytes(graphDef.toByteArray()).build()

            val partitionsVal = Nd4j.create(floatArrayOf(0f,0f,1f,1f,0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val dataVal = Nd4j.create(floatArrayOf(10f, 20f, 30f, 40f, 50f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

            val inputs = mapOf("data" to dataVal,"partitions" to partitionsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("data","partitions"),
                outputNames = listOf("out0","out1"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "split_v" -> {
            val splitDim = NodeDef {
                name = "split_dim"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val sizeSplits = NodeDef {
                name = "size_splits"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val value = NodeDef {
                name = "value"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("value")
                Input("size_splits")
                Input("split_dim")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("num_split",AttrValue {
                    i = 2
                })
                Attribute("Tlen",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val out0 = NodeDef {
                name = "out0"
                Input("output:0")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val out1 = NodeDef {
                name = "out1"
                Input("output:1")
                op = "Identity"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(value)
                Node(sizeSplits)
                Node(splitDim)
                Node(opNode)
                Node(out0)
                Node(out1)
            }

            val splitDimVal = Nd4j.scalar(-2.0).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val sizeSplitsVal = Nd4j.create(floatArrayOf(5f,3f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

            val valuesVal = Nd4j.linspace(1,56,56)
                .reshape(8,7).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("split_dim" to splitDimVal,"value" to valuesVal,"size_splits" to sizeSplitsVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("value","size_splits","split_dim"),
                outputNames = listOf("out0","out1"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }

        "split" -> {
            val splitDim = NodeDef {
                name = "split_dim"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val value = NodeDef {
                name = "value"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("split_dim")
                Input("value")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("num_split",AttrValue {
                    i = 2
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(splitDim)
                Node(value)
                Node(opNode)
            }

            val concatDimVal = Nd4j.scalar(0.0).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val valuesVal = Nd4j.create(floatArrayOf(0f,1f,0f,1f))
                .reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("split_dim" to concatDimVal,"value" to valuesVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("split_dim","value"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }


        "matmul" -> {
            val mmulInput = ArrayList<GraphInput>()
            listOf(false,true).forEach { transA ->
                listOf(false,true).forEach { transB ->
                    val a = NodeDef {
                        name = "a"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val bNode = NodeDef {
                        name = "b"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    println("Running test import process for op ${tensorflowOpDef.name}")
                    val opNode = NodeDef {
                        Input("a")
                        Input("b")

                        op = tensorflowOpDef.name
                        name = "output"
                        Attribute("T",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                        Attribute("transpose_a",AttrValue {
                            b = transA
                        })
                        Attribute("transpose_b",AttrValue {
                            b = transB
                        })
                    }


                    val graphDef = GraphDef {
                        Node(a)
                        Node(bNode)
                        Node(opNode)
                    }

                    val aVal = Nd4j.linspace(1,4,4).reshape(2,2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
                    val bVal = Nd4j.create(floatArrayOf(0f,1f,0f,1f))
                        .reshape(2,2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                    val inputs = mapOf("a" to aVal,"b" to bVal)
                    mmulInput.add(GraphInput(
                        graphDef =graphDef,
                        inputNames = listOf("a","b"),
                        outputNames = listOf("output"),
                        inputArrays = inputs,
                        dynamicArrays = inputs
                    ))
                }
            }


            return mmulInput


        }

        "range" -> {
            val start = NodeDef {
                name = "start"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val limit = NodeDef {
                name = "limit"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val delta = NodeDef {
                name = "delta"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("start")
                Input("limit")
                Input("delta")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("Tidx", AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val graphDef = GraphDef {
                Node(start)
                Node(limit)
                Node(delta)
                Node(opNode)
            }

            val startVal = Nd4j.scalar(1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val limitVal = Nd4j.scalar(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val deltaVal = Nd4j.scalar(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("start" to startVal, "limit" to limitVal, "delta" to deltaVal)


            return listOf(
                GraphInput(
                    graphDef = graphDef, inputNames = listOf("start", "limit", "delta"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                )
            )

        }

        "lin_space" -> {
            val start = NodeDef {
                name = "start"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val stop = NodeDef {
                name = "stop"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val num = NodeDef {
                name = "num"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("start")
                Input("stop")
                Input("num")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T", AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tidx", AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(start)
                Node(stop)
                Node(num)
                Node(opNode)
            }

            val startVal = Nd4j.scalar(1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
            val limitVal = Nd4j.scalar(1).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)
            val deltaVal = Nd4j.scalar(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("start" to startVal,"stop" to
                    limitVal, "num" to deltaVal)


            return listOf(
                GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("start", "stop","num"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = mapOf("limit" to limitVal)
                )
            )

        }

        "gather","gather_nd" -> {
            if(tensorflowOpDef.name != "GatherV2") {
                val params = NodeDef {
                    name = "params"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val indices = NodeDef {
                    name = "indices"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("params")
                    Input("indices")

                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("Tparams",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("Tindices",AttrValue {
                        type = DataType.DT_INT64
                    })
                }


                val graphDef = GraphDef {
                    Node(params)
                    Node(indices)
                    Node(opNode)
                }

                val paramsVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
                val indicesVal = Nd4j.create(floatArrayOf(0f,1f,0f,1f))
                    .reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


                val inputs = mapOf("params" to paramsVal,"indices" to indicesVal.dup())


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("params","indices"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else {
                val params = NodeDef {
                    name = "params"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val indices = NodeDef {
                    name = "indices"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val axis = NodeDef {
                    name = "axis"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }


                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("params")
                    Input("indices")
                    Input("axis")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("Tparams",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("Tindices",AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("Taxis",AttrValue {
                        type = DataType.DT_INT64
                    })
                }


                val graphDef = GraphDef {
                    Node(params)
                    Node(indices)
                    Node(axis)
                    Node(opNode)
                }

                val paramsVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
                val indicesVal = Nd4j.create(floatArrayOf(0f,1f,0f,1f))
                    .reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
                val axisVal = Nd4j.scalar(0).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

                val inputs = mapOf("params" to paramsVal,"indices" to indicesVal.dup(),"axis" to axisVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("params","indices","axis"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }


        }
        "stack" -> {
            val concat1 = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val concat2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("input2")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("N",AttrValue {
                    i = 2
                })
                Attribute("axis",AttrValue {
                    i = 0
                })
            }


            val graphDef = GraphDef {
                Node(concat1)
                Node(concat2)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","input2"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "unstack" -> {
            val concat1 = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("num",AttrValue {
                    i = 2
                })
                Attribute("axis",AttrValue {
                    i = 0
                })
            }


            val graphDef = GraphDef {
                Node(concat1)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "mergesum" -> {
            val concat1 = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val concat2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("input2")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("N",AttrValue {
                    i = 2
                })
            }


            val graphDef = GraphDef {
                Node(concat1)
                Node(concat2)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","input2"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "merge" -> {
            val concat1 = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val concat2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("input2")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("N",AttrValue {
                    i = 2
                })
            }


            val graphDef = GraphDef {
                Node(concat1)
                Node(concat2)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","input2"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "mergeadd" -> {
            val concat1 = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val concat2 = NodeDef {
                name = "input2"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                Input("input2")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("N",AttrValue {
                    i = 2
                })
                Attribute("shape",AttrValue {
                    shape = TensorShapeProto {
                        Dims(listOf(4,2))
                    }
                })
            }


            val graphDef = GraphDef {
                Node(concat1)
                Node(concat2)
                Node(opNode)
            }

            val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","input2"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "concat" -> {
            if(inputFrameworkOpName == "Concat") {
                val concatDim = NodeDef {
                    name = "concat_dim"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            dtype = DataType.DT_INT32
                            Int32Data(listOf(0))
                            Shape(listOf())
                        }
                    })
                }
                val concat1 = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val concat2 = NodeDef {
                    name = "input2"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("concat_dim")
                    Input("input")
                    Input("input2")

                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("N", AttrValue {
                        i = 2
                    })
                }


                val graphDef = GraphDef {
                    Node(concatDim)
                    Node(concat1)
                    Node(concat2)
                    Node(opNode)
                }

                val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


                val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("input","input2"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else { //ConcatV2
                val concatDim = NodeDef {
                    name = "concat_dim"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            dtype = DataType.DT_INT32
                            Int32Data(listOf(0))
                            Shape(listOf())
                        }
                    })
                }
                val concat1 = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                val concat2 = NodeDef {
                    name = "input2"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT64
                    })
                }

                println("Running test import process for op ${tensorflowOpDef.name}")
                val opNode = NodeDef {
                    Input("input")
                    Input("input2")
                    Input("concat_dim")

                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT64
                    })
                    Attribute("N", AttrValue {
                        i = 2
                    })
                }


                val graphDef = GraphDef {
                    Node(concat1)
                    Node(concat2)
                    Node(concatDim)
                    Node(opNode)
                }

                val inputVal = Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


                val inputs = mapOf("input" to inputVal,"input2" to inputVal.dup())


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("input","input2"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

        }

        "shape_of" -> {
            val tensorNode = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("out_type",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val graphDef = GraphDef {
                Node(tensorNode)
                Node(opNode)
            }

            val inputVal = Nd4j.create(floatArrayOf(1.0f,0.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT64)


            val inputs = mapOf("input" to inputVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "toggle_bits","invert_permutation" -> {
            val tensorNode = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            println("Running test import process for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val graphDef = GraphDef {
                Node(tensorNode)
                Node(opNode)
            }

            val inputVal = Nd4j.create(floatArrayOf(1.0f,0.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to inputVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "reverse" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })

            }

            val axis = NodeDef {
                name = "axis"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val opNode = NodeDef {
                Input("input")
                Input("axis")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("Tidx",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val graphDef = GraphDef {
                Node(input)
                Node(axis)
                Node(opNode)
            }


            val inputVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val axisVal = Nd4j.zeros(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to inputVal,"axis" to axisVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","axis"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "roll" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })

            }

            val shift = NodeDef {
                name = "shift"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val axis = NodeDef {
                name = "axis"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val opNode = NodeDef {
                Input("input")
                Input("shift")
                Input("axis")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("Tshift",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("Taxis",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val graphDef = GraphDef {
                Node(input)
                Node(shift)
                Node(axis)
                Node(opNode)
            }


            val inputVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val shiftVal = Nd4j.zeros(2).addi(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val axisVal = Nd4j.zeros(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to inputVal,"shift" to shiftVal,"axis" to axisVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","shift","axis"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "tile" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })

            }

            val multiples = NodeDef {
                name = "multiples"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val opNode = NodeDef {
                Input("input")
                Input("multiples")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("Tmultiples",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val graphDef = GraphDef {
                Node(input)
                Node(multiples)
                Node(opNode)
            }


            val inputVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val multiplesVal = Nd4j.zeros(2).addi(2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


            val inputs = mapOf("input" to inputVal,"multiples" to multiplesVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input","multiples"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "leakyrelu" -> {
            val a = NodeDef {
                name = "a"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })

            }

            val opNode = NodeDef {
                Input("a")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("alpha",AttrValue {
                    f = 0.1f
                })
            }



            val graphDef = GraphDef {
                Node(a)
                Node(opNode)
            }


            val aVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)



            val inputs = mapOf("a" to aVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("a"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "betainc" -> {
            val a = NodeDef {
                name = "a"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val b = NodeDef {
                name = "b"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val opNode = NodeDef {
                Input("a")
                Input("b")
                Input("x")

                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            val graphDef = GraphDef {
                Node(a)
                Node(b)
                Node(x)
                Node(opNode)
            }


            val aVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val bVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val xVal = Nd4j.zeros(2,2).addi(0.5)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("a" to aVal,"b" to bVal,"x" to xVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("a","b","x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }
        "top_k" -> {
            if(tensorflowOpDef.name == "TopK") {
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }



                val opNode = NodeDef {
                    Input("input")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("k", AttrValue {
                        i = 2
                    })
                }



                val graphDef = GraphDef {
                    Node(input)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)



                val inputs = mapOf("input" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("input"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else { //TopKV2
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }


                val k = NodeDef {
                    name = "k"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(2))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val opNode = NodeDef {
                    Input("input")
                    Input("k")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_FLOAT
                    })

                }



                val graphDef = GraphDef {
                    Node(input)
                    Node(k)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)



                val inputs = mapOf("input" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("input"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

        }
        "enter" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("is_constant",AttrValue {
                    b = false
                })
                Attribute("frame_name",AttrValue {
                    s = ByteString.copyFrom("hello".toByteArray(Charset.defaultCharset()))
                })

            }

            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 6, 6)
                .reshape(2, 3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("input" to xVal)

            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "Assert" -> {
            val condition = NodeDef {
                name = "condition"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val input = NodeDef {
                name = "input"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        FloatData(listOf(0.0f))
                        dtype = DataType.DT_FLOAT

                    }
                })
            }


            val opNode = NodeDef {
                Input("condition")
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    ListDataType(listOf(DataType.DT_FLOAT))
                })

            }

            val graphDef = GraphDef {
                Node(condition)
                Node(input)
                Node(opNode)
            }


            val xVal = Nd4j.create(listOf(true,true,true,true).toBooleanArray())

            val inputs = mapOf("condition" to xVal)

            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("condition"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }



        "bitcast" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }


            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("type",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }


            val xVal = Nd4j.zeros(2,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)

            val inputs = mapOf("input" to xVal)

            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "exit" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 6, 6)
                .reshape(2, 3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("input" to xVal)

            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "expand_dims" -> {
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val n = NodeDef {
                name = "dimension"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        Int32Data(listOf(0))
                        dtype = DataType.DT_INT32

                    }
                })
            }

            val opNode = NodeDef {
                Input("input")
                Input("dimension")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_INT32
                })

            }

            val graphDef = GraphDef {
                Node(input)
                Node(n)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 6, 6)
                .reshape(2, 3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

            val inputs = mapOf("input" to xVal)

            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "non_max_suppression","non_max_suppression_v3" -> {
            if(inputFrameworkOpName == "NonMaxSuppression") {
                val overlaps = NodeDef {
                    name = "overlaps"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val scores = NodeDef {
                    name = "scores"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val maxOutputSize = NodeDef {
                    name = "maxOutputSize"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(1))
                            dtype = DataType.DT_INT32

                        }
                    })
                }



                val opNode = NodeDef {
                    Input("overlaps")
                    Input("scores")
                    Input("maxOutputSize")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("iou_threshold", AttrValue {
                        f = 0.5f
                    })
                }

                val graphDef = GraphDef {
                    Node(overlaps)
                    Node(scores)
                    Node(maxOutputSize)
                    Node(opNode)
                }



                val overlapsVal = Nd4j.create(arrayOf(
                    floatArrayOf(0f,0f,1f,1f),
                    floatArrayOf(0f,0.1f,1f,1.1f),
                    floatArrayOf(0f,-0.1f,1f,0.9f),
                    floatArrayOf(0f,10f,1f,11f)
                )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("overlaps","scores"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }
            else if(inputFrameworkOpName == "NonMaxSuppressionV2") {
                val overlaps = NodeDef {
                    name = "overlaps"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val scores = NodeDef {
                    name = "scores"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val maxOutputSize = NodeDef {
                    name = "maxOutputSize"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(1))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val iouThreshold = NodeDef {
                    name = "iouThreshold"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            FloatData(listOf(0.5f))
                            dtype = DataType.DT_FLOAT

                        }
                    })
                }



                val opNode = NodeDef {
                    Input("overlaps")
                    Input("scores")
                    Input("maxOutputSize")
                    Input("iouThreshold")
                    op = tensorflowOpDef.name
                    name = "output"

                }

                val graphDef = GraphDef {
                    Node(overlaps)
                    Node(scores)
                    Node(iouThreshold)
                    Node(maxOutputSize)
                    Node(opNode)
                }



                val overlapsVal = Nd4j.create(arrayOf(
                    floatArrayOf(0f,0f,1f,1f),
                    floatArrayOf(0f,0.1f,1f,1.1f),
                    floatArrayOf(0f,-0.1f,1f,0.9f),
                    floatArrayOf(0f,10f,1f,11f)
                )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("overlaps","scores"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else {
                //V3 and later
                val overlaps = NodeDef {
                    name = "overlaps"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val scores = NodeDef {
                    name = "scores"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val maxOutputSize = NodeDef {
                    name = "maxOutputSize"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(1))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val overlapThreshold = NodeDef {
                    name = "iouThreshold"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            FloatData(listOf(0.5f))
                            dtype = DataType.DT_FLOAT

                        }
                    })
                }

                val scoreThreshold = NodeDef {
                    name = "scoreThreshold"
                    op = "Const"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            FloatData(listOf(0.5f))
                            dtype = DataType.DT_FLOAT

                        }
                    })
                }

                val opNode = NodeDef {
                    Input("overlaps")
                    Input("scores")
                    Input("maxOutputSize")
                    Input("iouThreshold")
                    Input("scoreThreshold")
                    op = tensorflowOpDef.name
                    name = "output"

                }

                val graphDef = GraphDef {
                    Node(overlaps)
                    Node(scores)
                    Node(scoreThreshold)
                    Node(overlapThreshold)
                    Node(maxOutputSize)
                    Node(opNode)
                }



                val overlapsVal = Nd4j.create(arrayOf(
                    floatArrayOf(0f,0f,1f,1f),
                    floatArrayOf(0f,0.1f,1f,1.1f),
                    floatArrayOf(0f,-0.1f,1f,0.9f),
                    floatArrayOf(0f,10f,1f,11f)
                )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

                return listOf(GraphInput(
                    graphDef = graphDef,
                    inputNames = listOf("overlaps","scores"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }
        }

        "non_max_suppression_overlaps" -> {
            val overlaps = NodeDef {
                name = "overlaps"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val scores = NodeDef {
                name = "scores"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val maxOutputSize = NodeDef {
                name = "maxOutputSize"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        Int32Data(listOf(1))
                        dtype = DataType.DT_INT32

                    }
                })
            }

            val overlapThreshold = NodeDef {
                name = "overlapThreshold"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        FloatData(listOf(2.0f))
                        dtype = DataType.DT_FLOAT

                    }
                })
            }

            val scoreThreshold = NodeDef {
                name = "scoreThreshold"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        FloatData(listOf(0.5f))
                        dtype = DataType.DT_FLOAT

                    }
                })
            }

            val opNode = NodeDef {
                Input("overlaps")
                Input("scores")
                Input("maxOutputSize")
                Input("overlapThreshold")
                Input("scoreThreshold")
                op = tensorflowOpDef.name
                name = "output"

            }

            val graphDef = GraphDef {
                Node(overlaps)
                Node(scores)
                Node(scoreThreshold)
                Node(overlapThreshold)
                Node(maxOutputSize)
                Node(opNode)
            }



            val overlapsVal = Nd4j.create(arrayOf(
                floatArrayOf(0f,0f,1f,1f),
                floatArrayOf(0f,0.1f,1f,1.1f),
                floatArrayOf(0f,-0.1f,1f,0.9f),
                floatArrayOf(0f,10f,1f,11f)
            )).castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val scoresVal = Nd4j.create(listOf(0.9f,0.75f,0.6f,0.95f).toFloatArray())
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val inputs = mapOf("overlaps" to overlapsVal,"scores" to scoresVal)

            return listOf(GraphInput(
                graphDef = graphDef,
                inputNames = listOf("overlaps","scores"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs

            ))
        }

        "nth_element" -> {
            val ret = ArrayList<GraphInput>()
            listOf(true,false).forEach { reverse ->
                val input = NodeDef {
                    name = "input"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val n = NodeDef {
                    name = "n"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value",AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(2))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val opNode = NodeDef {
                    Input("input")
                    Input("n")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT32
                    })

                    Attribute("reverse", AttrValue {
                        type = DataType.DT_BOOL
                        b = reverse
                    })

                }

                val graphDef = GraphDef {
                    Node(input)
                    Node(n)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 6, 6)
                    .reshape(2, 3)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val inputs = mapOf("input" to xVal)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("input"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

            return ret
        }


        "cholesky" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


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


            val xVal = Nd4j.create(floatArrayOf(4f,12f,-16f, 12f ,37f,-43f, -16f, -43f, 98f))
                .reshape(3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }


        "matrix_diag_part"  -> {
            val retSolve = ArrayList<GraphInput>()
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }




            val opNode = NodeDef {
                Input("input")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })


            }

            val graphDef = GraphDef {
                Node(input)
                Node(opNode)
            }


            val inputVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


            val inputs = mapOf("input" to inputVal)


            retSolve.add(GraphInput(
                graphDef = graphDef, inputNames = listOf("input"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))


            return retSolve

        }


        "matrix_set_diag","matrix_diag_part"  -> {
            val retSolve = ArrayList<GraphInput>()
            val input = NodeDef {
                name = "input"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val diagonal = NodeDef {
                name = "diagonal"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }



            val opNode = NodeDef {
                Input("input")
                Input("diagonal")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })


            }

            val graphDef = GraphDef {
                Node(input)
                Node(diagonal)
                Node(opNode)
            }


            val inputVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val diagonalVal = Nd4j.zeros(2).addi(1)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("input" to inputVal,"diagonal" to diagonalVal)


            retSolve.add(GraphInput(
                graphDef = graphDef, inputNames = listOf("input","diagonal"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))


            return retSolve

        }

        "solve","triangular_solve" -> {
            val retSolve = ArrayList<GraphInput>()
            listOf(false,true).forEach { useAdjoint ->
                val a = NodeDef {
                    name = "a"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val bNode = NodeDef {
                    name = "b"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val opNode = NodeDef {
                    Input("a")
                    Input("b")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("adjoint", AttrValue {
                        b = useAdjoint
                    })

                }

                val graphDef = GraphDef {
                    Node(a)
                    Node(bNode)
                    Node(opNode)
                }


                val aVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val bVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("a" to aVal,"b" to bVal)


                retSolve.add(GraphInput(
                    graphDef = graphDef, inputNames = listOf("a","b"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

            return retSolve

        }

        "matrix_determinant","log_matrix_determinant" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val opNode = NodeDef {
                Input("x")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })

            }

            val finalResult = NodeDef {
                Input("output:1")
                op = "Identity"
                name = "finalResult"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })

            }

            if(nd4jOpName == "log_matrix_determinant") {
                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(finalResult)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef = graphDef, inputNames = listOf("x"),
                    outputNames = listOf("finalResult"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))

            } else {
                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }
        }


        "lu" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


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


            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "matrix_inverse" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


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


            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "in_top_k" -> {
            if(tensorflowOpDef.name == "InTopK") {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val predictions = NodeDef {
                    name = "predictions"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("predictions")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("k", AttrValue {
                        i = 2
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(predictions)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val predictionsArr = Nd4j.linspace(1, 2, 2)
                    .reshape(2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                val inputs = mapOf("x" to xVal,"predictions" to predictionsArr)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x","predictions"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            } else {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_FLOAT
                    })
                }

                val predictions = NodeDef {
                    name = "predictions"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val k = NodeDef {
                    name = "k"
                    op = "Const"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            Int32Data(listOf(2))
                            dtype = DataType.DT_INT32

                        }
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("predictions")
                    Input("k")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(predictions)
                    Node(k)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

                val predictionsArr = Nd4j.linspace(1, 2, 2)
                    .reshape(2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                val inputs = mapOf("x" to xVal,"predictions" to predictionsArr)


                return listOf(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x","predictions"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }



        }


        "onehot" -> {
            val indices = NodeDef {
                name = "indices"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
            }

            val depth = NodeDef {
                name = "depth"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        dtype = DataType.DT_INT32
                        Int32Data(listOf(1))

                    }
                })
            }

            val onValue = NodeDef {
                name = "on"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        dtype = DataType.DT_INT64
                        Int64Data(listOf(1))

                    }
                })
            }


            val offValue = NodeDef {
                name = "off"
                op = "Const"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        dtype = DataType.DT_INT64
                        Int64Data(listOf(0))

                    }
                })
            }


            val opNode = NodeDef {
                Input("indices")
                Input("depth")
                Input("on")
                Input("off")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("TI",AttrValue {
                    type = DataType.DT_INT64
                })
                Attribute("T",AttrValue {
                    type = DataType.DT_INT64
                })

                Attribute("axis",AttrValue {
                    i = 0
                })
            }



            val graphDef = GraphDef {
                Node(indices)
                Node(depth)
                Node(onValue)
                Node(offValue)
                Node(opNode)
            }


            val indicesVal = Nd4j.linspace(1, 4, 4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.INT64)
            val inputs = mapOf("indices" to indicesVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("indices"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "cross" -> {
            val a = NodeDef {
                name = "a"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val b = NodeDef {
                name = "b"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val opNode = NodeDef {
                Input("a")
                Input("b")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })

            }



            val graphDef = GraphDef {
                Node(a)
                Node(b)
                Node(opNode)
            }


            val aVal = Nd4j.linspace(1, 27, 27)
                .reshape(3,3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)

            val bVal = Nd4j.linspace(1, 27, 27)
                .reshape(3,3,3)
                .castTo(org.nd4j.linalg.api.buffer.DataType.FLOAT)


            val inputs = mapOf("a" to aVal,"b" to bVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("a","b"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "transpose" ->  {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val tensorNode2 = NodeDef {
                op = "Const"
                name = "perm"
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        Int32Data(listOf(0,1))
                        Shape(listOf(2))
                        dtype = DataType.DT_INT32
                    }
                })
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val opNode = NodeDef {
                Input("x")
                Input("perm")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tperm",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val graphDef = GraphDef {
                Node(tensorNode)
                Node(tensorNode2)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)



            val inputs = mapOf("x" to xVal)


            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

        }
        "relu", "relu6" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val opNode = NodeDef {
                Input("x")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)

            return listOf(GraphInput(
                graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "depth_to_space","space_to_depth" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val opNode = NodeDef {
                Input("x")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T", AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("data_format", AttrValue {
                    s = ByteString.copyFrom("NHWC".toByteArray(Charset.defaultCharset()))
                })
                Attribute("block_size", AttrValue {
                    i = 2
                })
            }

            val xVal = Nd4j.linspace(1, 256, 256)
                .reshape(4, 4,4,4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)

            return listOf(GraphInput(
                graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "softmax","digamma","diag","diag_part","lgamma" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            val opNode = NodeDef {
                Input("x")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T", AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)

            return listOf(GraphInput(
                graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "cumsum","cumprod" -> {
            val ret = ArrayList<GraphInput>()
            listOf(false,true).forEach { reverse ->
                listOf(false,true).forEach { exclusive ->
                    val inputNames = listOf("x")
                    val tensorNode = NodeDef {
                        name = "x"
                        op = "Placeholder"
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_DOUBLE
                        })
                    }

                    val dimensions = listOf(1)
                    val tensorNode2 = NodeDef {
                        op = "Const"
                        name = "dimensions"
                        Attribute("value",AttrValue {
                            tensor = TensorProto {
                                Int32Data(dimensions)
                                dtype = DataType.DT_INT32
                                tensorShape = TensorShapeProto {
                                    Dims(listOf())
                                }
                            }
                        })
                        Attribute("dtype",AttrValue {
                            type = DataType.DT_INT32
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
                        Attribute("exclusive",AttrValue {
                            b = exclusive
                        })

                        Attribute("reverse",AttrValue {
                            b = reverse
                        })
                    }



                    val graphDef = GraphDef {
                        Node(tensorNode)
                        Node(tensorNode2)
                        Node(opNode)
                    }

                    val xVal = Nd4j.linspace(1, 4, 4)
                        .reshape(2, 2)
                        .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


                    val inputs = mapOf("x" to xVal)
                    ret.add(GraphInput(
                        graphDef =graphDef, inputNames = inputNames,
                        outputNames = listOf("output"),
                        inputArrays = inputs,
                        dynamicArrays = inputs
                    ))
                }
            }

            return ret

        }

        "Assert" -> {
            val tensorNode = NodeDef {
                name = "condition"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val tensorNode2 = NodeDef {
                op = "Placeholder"
                name = "data"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            println("Running op def for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                Input("condition")
                Input("data")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    ListDataType(listOf(DataType.DT_DOUBLE))
                })
            }



            val graphDef = GraphDef {
                Node(tensorNode)
                Node(opNode)
                Node(tensorNode2)
            }

            val inputs = mapOf("data" to Nd4j.linspace(1,4,4).castTo(
                org.nd4j.linalg.api.buffer.DataType.DOUBLE
            ),"condition" to Nd4j.ones(2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.BOOL))
            return listOf(GraphInput(graphDef = graphDef,
                inputNames = listOf("condition","data"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }


        "Where" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }


            println("Running op def for op ${tensorflowOpDef.name}")
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

            val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                org.nd4j.linalg.api.buffer.DataType.DOUBLE
            ))
            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }



        "boolean_or" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val inputNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }


            val secondNode = NodeDef {
                name = "y"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val opNode = NodeDef {
                Input("x")
                Input("y")
                name = "and"
                op = tensorflowOpDef.name
            }


            val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.BOOL
            ), "y" to Nd4j.zeros(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.BOOL
            ))


            val graphDef = GraphDef {
                Node(inputNode)
                Node(secondNode)
                Node(opNode)
            }

            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","y"),
                outputNames = listOf("and"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }


        "boolean_and" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val inputNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }


            val secondNode = NodeDef {
                name = "y"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }

            val opNode = NodeDef {
                Input("x")
                Input("y")
                name = "and"
                op = tensorflowOpDef.name
            }


            val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.BOOL
            ), "y" to Nd4j.zeros(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.BOOL
            ))


            val graphDef = GraphDef {
                Node(inputNode)
                Node(secondNode)
                Node(opNode)
            }

            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","y"),
                outputNames = listOf("and"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }


        "igamma","igammac" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val a = NodeDef {
                name = "a"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val x = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }



            val opNode = NodeDef {
                Input("a")
                Input("x")
                name = "igamma"
                op = tensorflowOpDef.name
                Attribute("T",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }


            val inputs = mapOf("a" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.FLOAT
            ),"x" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.FLOAT
            ))

            val graphDef = GraphDef {
                Node(a)
                Node(x)
                Node(opNode)
            }

            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("a","x"),
                outputNames = listOf("igamma"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }

        "boolean_not" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val inputNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_BOOL
                })
            }




            val opNode = NodeDef {
                Input("x")
                name = "not"
                op = tensorflowOpDef.name
            }


            val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.BOOL
            ))

            val graphDef = GraphDef {
                Node(inputNode)
                Node(opNode)
            }

            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),
                outputNames = listOf("not"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }



        "cast" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val inputNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_FLOAT
                })
            }

            val opNode = NodeDef {
                Input("x")
                name = "output"
                op = tensorflowOpDef.name
                Attribute("SrcT",AttrValue {
                    type = DataType.DT_FLOAT
                })
                Attribute("DstT",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }



            val graphDef = GraphDef {
                Node(inputNode)
                Node(opNode)
            }

            val inputs = mapOf("x" to Nd4j.ones(2,2).castTo(
                org.nd4j.linalg.api.buffer.DataType.FLOAT
            ))

            return listOf(GraphInput(graphDef = graphDef,
                inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = emptyMap()))
        }

        "noop" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val opNode = NodeDef {
                name = "noop"
                op = tensorflowOpDef.name
            }



            val graphDef = GraphDef {
                Node(opNode)
            }

            return listOf(GraphInput(graphDef = graphDef,
                inputNames = listOf(),
                outputNames = listOf(),
                inputArrays = emptyMap(),
                dynamicArrays = emptyMap()))
        }

        "While" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    ListDataType(listOf(DataType.DT_DOUBLE))
                })
            }


            val opNode = NodeDef {
                Input("x")
                name = "while"
                op = tensorflowOpDef.name
            }



            val graphDef = GraphDef {
                Node(tensorNode)
                Node(opNode)
            }

            val inputs = mapOf("x" to Nd4j.scalar(1.0))

            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }

        "unique_with_counts","unique" -> {
            println("Running op def for op ${tensorflowOpDef.name}")
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            if(tensorflowOpDef.name == "UniqueWithCountsV2" || tensorflowOpDef.name == "UniqueV2") {
                val axis = NodeDef {
                    name = "axis"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT64
                    })
                }


                val opNode = NodeDef {
                    Input("x")
                    Input("axis")
                    name = "output"
                    op = tensorflowOpDef.name
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(axis)
                    Node(opNode)
                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).reshape(2,2).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE),
                    "axis" to Nd4j.scalar(1).reshape(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT64))

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","axis"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs))
            }
            else {
                val opNode = NodeDef {
                    Input("x")
                    name = "output"
                    op = tensorflowOpDef.name
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE))

                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs))
            }

        }


        "pad" -> {
            if(tensorflowOpDef.name == "Pad") {
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "paddings"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("paddings")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("Tpaddings", AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(tensorNode2)
                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                    org.nd4j.linalg.api.buffer.DataType.DOUBLE
                ),"paddings" to Nd4j.ones(1,2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","paddings"),outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs))
            } else if(tensorflowOpDef.name == "PadV2"){
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "paddings"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val constantValues = NodeDef {
                    op = "Const"
                    name = "constant_values"
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            DoubleData(listOf(1.0))
                            dtype = DataType.DT_DOUBLE
                            tensorShape = TensorShapeProto {
                                Dims(listOf())
                            }
                        }
                    })
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                }

                val opNode = NodeDef {
                    Input("x")
                    Input("paddings")
                    Input("constant_values")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T",AttrValue {
                        type = DataType.DT_DOUBLE
                    })
                    Attribute("Tpaddings",AttrValue {
                        type = DataType.DT_INT32
                    })
                }



                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(opNode)
                    Node(constantValues)
                    Node(tensorNode2)

                }

                val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                    org.nd4j.linalg.api.buffer.DataType.DOUBLE
                ),"paddings" to Nd4j.ones(1,2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
                return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","paddings"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs))
            } else {
                throw IllegalArgumentException("Illegal mapping for padding op $tensorflowOpDef.name")
            }

        }


        "reshape" -> {
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val tensorNode2 = NodeDef {
                op = "Placeholder"
                name = "shape"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val opNode = NodeDef {
                Input("x")
                Input("shape")
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

            val inputs = mapOf("x" to Nd4j.linspace(1,4,4).castTo(
                org.nd4j.linalg.api.buffer.DataType.DOUBLE
            ),"shape" to Nd4j.ones(2).addi(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32))
            return listOf(GraphInput(graphDef = graphDef,inputNames = listOf("x","shape"),outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs))
        }

        "reduce_logsumexp" -> {
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
                Attribute("exclusive",AttrValue {
                    b = false
                })
            }

            val dimensions = listOf(0)
            val tensorNode2 = NodeDef {
                op = "Const"
                name = "dimensions"
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        Int32Data(dimensions)
                        dtype = DataType.DT_INT32
                        tensorShape = TensorShapeProto {
                            Dims(listOf())
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


            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)

            return listOf(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))
        }

        "argmin", "argmax" -> {
            val ret = ArrayList<GraphInput>()
            listOf(true, false).forEach { keepDim ->
                val tensorNode = NodeDef {
                    name = "x"
                    op = "Placeholder"
                    Attribute("dtype", AttrValue {
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
                }

                val dimensions = listOf(0)
                val tensorNode2 = NodeDef {
                    op = "Const"
                    name = "dimensions"
                    Attribute("value", AttrValue {
                        tensor = TensorProto {
                            Int32Data(dimensions)
                            dtype = DataType.DT_INT32
                            tensorShape = TensorShapeProto {
                                Dims(listOf())
                            }
                        }
                    })
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val graphDef = GraphDef {
                    Node(tensorNode)
                    Node(tensorNode2)
                    Node(opNode)
                }


                val xVal = Nd4j.linspace(1, 4, 4)
                    .reshape(2, 2)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

                val inputs = mapOf("x" to xVal)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("x"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }

            return ret
        }

        "pow" -> {
            val ret = ArrayList<GraphInput>()
            val tensorNode = NodeDef {
                name = "x"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val tensorNode2 = NodeDef {
                op = "Const"
                name = "y"
                Attribute("value",AttrValue {
                    tensor = TensorProto {
                        DoubleData(listOf(1.0))
                        dtype = DataType.DT_DOUBLE
                        tensorShape = TensorShapeProto {
                            Dims(listOf())
                        }
                    }
                })
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
                Node(tensorNode2)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 4, 4)
                .reshape(2, 2)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)

            val inputs = mapOf("x" to xVal)

            ret.add(GraphInput(
                graphDef =graphDef, inputNames = listOf("x"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))

            return ret
        }



        //scatter_div
        //TODO: Revisit. TF op validation seems to be different than ours.
        "scatter_add","scatter_sub","scatter_min","scatter_sub","scatter_min","scatter_mul","scatter_update","scatter_nd","scatter_nd_add","scatter_nd_sub","scatter_nd_update" -> {
            val ret = ArrayList<GraphInput>()
            listOf(true,false).forEach { lock ->
                val xRef = NodeDef {
                    name = "shape"
                    op = "Placeholder"
                    Attribute("dtype",AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                val tensorNode2 = NodeDef {
                    op = "Placeholder"
                    name = "indices"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                val updates2 = NodeDef {
                    op = "Placeholder"
                    name = "updates"
                    Attribute("dtype", AttrValue {
                        type = DataType.DT_INT32
                    })
                }

                val opNode = NodeDef {
                    Input("indices")
                    Input("updates")
                    Input("shape")
                    op = tensorflowOpDef.name
                    name = "output"
                    Attribute("T", AttrValue {
                        type = DataType.DT_INT32
                    })
                    Attribute("Tindices", AttrValue {
                        type = DataType.DT_INT32
                    })
                }


                val graphDef = GraphDef {
                    Node(xRef)
                    Node(tensorNode2)
                    Node(updates2)
                    Node(opNode)
                }


                //from testScatterOpGradients.
                val shape = Nd4j.scalar(8).reshape(1).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
                val indices = Nd4j.create(floatArrayOf(4f,3f,1f,7f)).reshape(4,1)
                    .castTo(org.nd4j.linalg.api.buffer.DataType.INT32)

                val updates = Nd4j.linspace(1,4,4).reshape(4).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)


                val inputs = mapOf("shape" to shape,"updates" to updates,"indices" to indices)

                ret.add(GraphInput(
                    graphDef =graphDef, inputNames = listOf("indices","updates","shape"),
                    outputNames = listOf("output"),
                    inputArrays = inputs,
                    dynamicArrays = inputs
                ))
            }






            return ret
        }




        "segment_mean", "segment_min","segment_max","segment_prod","segment_sum" -> {
            val ret = ArrayList<GraphInput>()
            val tensorNode = NodeDef {
                name = "data"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val segmentIds = NodeDef {
                op = "Placeholder"
                name = "segment_ids"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }


            val opNode = NodeDef {
                Input("data")
                Input("segment_ids")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tindices",AttrValue {
                    type = DataType.DT_INT32
                })

            }



            val graphDef = GraphDef {
                Node(tensorNode)
                Node(segmentIds)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 12, 12)
                .reshape(3, 4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


            val indices = Nd4j.create(floatArrayOf(1.0f,2.0f,3.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val inputs = mapOf("data" to xVal,"segment_ids" to indices)

            ret.add(GraphInput(
                graphDef =graphDef, inputNames = listOf("data","segment_ids"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))


            return ret
        }


        "unsorted_segment_sum", "unsorted_segment_prod","unsorted_segment_min","unsorted_segment_max" -> {
            val ret = ArrayList<GraphInput>()
            val tensorNode = NodeDef {
                name = "data"
                op = "Placeholder"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_DOUBLE
                })
            }

            val segmentIds = NodeDef {
                op = "Placeholder"
                name = "segment_ids"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })
            }

            val numSegmentsNode = NodeDef {
                op = "Const"
                name = "num_segments"
                Attribute("dtype",AttrValue {
                    type = DataType.DT_INT32
                })

                Attribute(name = "value",value = AttrValue {
                    tensor = TensorProto {
                        Shape(listOf())
                        Int32Data(listOf(2))
                        DataType(DataType.DT_INT32)
                    }
                })
            }

            val opNode = NodeDef {
                Input("data")
                Input("segment_ids")
                Input("num_segments")
                op = tensorflowOpDef.name
                name = "output"
                Attribute("T",AttrValue {
                    type = DataType.DT_DOUBLE
                })
                Attribute("Tindices",AttrValue {
                    type = DataType.DT_INT32
                })
                Attribute("Tnumsegments",AttrValue {
                    type = DataType.DT_INT32
                })
            }



            val graphDef = GraphDef {
                Node(tensorNode)
                Node(segmentIds)
                Node(numSegmentsNode)
                Node(opNode)
            }


            val xVal = Nd4j.linspace(1, 12, 12)
                .reshape(3, 4)
                .castTo(org.nd4j.linalg.api.buffer.DataType.DOUBLE)


            val indices = Nd4j.create(floatArrayOf(0.0f,1.0f,0.0f)).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val numSegments = Nd4j.scalar(2).castTo(org.nd4j.linalg.api.buffer.DataType.INT32)
            val inputs = mapOf("data" to xVal,"segment_ids" to indices,"num_segments" to numSegments)

            ret.add(GraphInput(
                graphDef =graphDef, inputNames = listOf("data","segment_ids","num_segments"),
                outputNames = listOf("output"),
                inputArrays = inputs,
                dynamicArrays = inputs
            ))


            return ret
        }


        else -> {
            throw IllegalArgumentException("Illegal op name $inputFrameworkOpName")
        }
    }
}
