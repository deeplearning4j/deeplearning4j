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

package org.nd4j.linalg.custom

import junit.framework.Assert.assertEquals
import org.junit.Ignore
import org.junit.Test
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ops.impl.image.CropAndResize
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.tensorflow.*
import org.nd4j.samediff.frameworkimport.tensorflow.importer.TensorflowFrameworkImporter
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraphRunner

class CustomOpTensorflowInteropTests {

    @Test
    @Ignore("Tensorflow expects different shape")
    fun testCropAndResize() {
        val image = Nd4j.createUninitialized(DataType.FLOAT, 1, 2, 2, 1)
        val boxes = Nd4j.createFromArray(*floatArrayOf(1f, 2f, 3f, 4f)).reshape(1, 4)
        val box_indices = Nd4j.createFromArray(*intArrayOf(0))
        val crop_size = Nd4j.createFromArray(*intArrayOf(1, 2)).reshape( 2)
        val imageNode = NodeDef {
            op = "Placeholder"
            name = "image"
            Attribute("dtype", AttrValue {
                type = org.tensorflow.framework.DataType.DT_FLOAT
            })
        }

        val boxesNode = NodeDef {
            op = "Placeholder"
            name = "boxes"
            Attribute("dtype", AttrValue {
                type = org.tensorflow.framework.DataType.DT_FLOAT
            })
        }

        val boxIndicesNode = NodeDef {
            op = "Placeholder"
            name = "boxIndices"
            Attribute("dtype", AttrValue {
                type = org.tensorflow.framework.DataType.DT_INT32
            })
        }

        val cropSizesNode = NodeDef {
            op = "Placeholder"
            name = "cropSize"
            Attribute("dtype", AttrValue {
                type = org.tensorflow.framework.DataType.DT_INT32
            })
        }


        val opNode = NodeDef {
            op = "CropAndResize"
            name = "output"
            Input("image")
            Input("boxes")
            Input("boxIndices")
            Input("cropSize")
            Attribute("extrapolation_value", AttrValue {
                f = 0.5f
            })
            Attribute("T", AttrValue {
                type = org.tensorflow.framework.DataType.DT_FLOAT
            })
        }

        val graph = GraphDef {
            Node(imageNode)
            Node(boxesNode)
            Node(boxIndicesNode)
            Node(cropSizesNode)
            Node(opNode)

        }

        val importer = TensorflowFrameworkImporter()
        val irGraph = TensorflowIRGraph(graph,importer.opDefList,importer.registry)
        val runner = TensorflowIRGraphRunner(irGraph,listOf("image","boxes","boxIndices","cropSize"),listOf("output"))
        val tfResult = runner.run(mapOf("image" to image,"boxes" to boxes,"boxIndices" to box_indices,"cropSize" to crop_size))
        val outputArr = tfResult["output"]
        //Output shape mismatch - TF [2, 2, 1, 1] vs SD: [1, 2, 1, 1]
        val output = Nd4j.create(DataType.FLOAT, 2, 2, 1, 1)
        Nd4j.exec(
            CropAndResize(
                image, boxes, box_indices, crop_size, CropAndResize.Method.BILINEAR, 0.5,
                output
            )
        )

        assertEquals(outputArr,output)
    }


}