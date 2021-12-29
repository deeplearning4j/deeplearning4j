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

/*******************************************************************************
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
package org.nd4j.samediff.frameworkimport.onnx.modelzoo

import onnx.Onnx
import org.apache.commons.io.FileUtils
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import org.nd4j.common.resources.Downloader
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraphRunner
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRTensor
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.io.File
import java.net.URI

data class InputDataset(val dataSetIndex: Int,val inputPaths: List<String>,val outputPaths: List<String>)
class TestPretrainedModels {

    val modelBaseUrl = "https://media.githubusercontent.com/media/onnx/models/master"
    public val modelDirectory = File(File(System.getProperty("user.home")),"models/")
    public val importer = OnnxFrameworkImporter()
    val runOnly = emptySet<String>()
    val dontRunRegexes = setOf("")
    val modelPaths = setOf("vision/body_analysis/age_gender/models/age_googlenet.onnx",
        "vision/body_analysis/age_gender/models/gender_googlenet.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx",
        //"vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "vision/body_analysis/ultraface/models/version-RFB-320.onnx",
        "vision/classification/alexnet/model/bvlcalexnet-9.onnx",
        "vision/classification/caffenet/model/caffenet-9.onnx",
        "vision/classification/densenet-121/model/densenet-9.onnx",
        "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        "vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
        "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        "vision/classification/mnist/model/mnist-8.onnx",
        "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
        "vision/classification/resnet/model/resnet101-v1-7.onnx",
        "vision/classification/resnet/model/resnet152-v2-7.onnx",
        "vision/classification/resnet/model/resnet18-v1-7.onnx",
        "vision/classification/resnet/model/resnet18-v2-7.onnx",
        "vision/classification/resnet/model/resnet34-v1-7.onnx",
        "vision/classification/resnet/model/resnet34-v2-7.onnx",
        "vision/classification/shufflenet/model/shufflenet-9.onnx",
        "vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
        "vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
        "vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        "vision/classification/vgg/model/vgg16-7.onnx",
        "vision/classification/vgg/model/vgg16-bn-7.onnx",
        "vision/classification/vgg/model/vgg19-7.onnx",
        "vision/classification/vgg/model/vgg19-caffe2-9.onnx",
        "vision/classification/zfnet-512/model/zfnet512-9.onnx",
        "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx",
        "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx",
        "vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.onnx",
        "vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx",
        "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx",
        "vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx",
        "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
        "vision/object_detection_segmentation/ssd/model/ssd-10.onnx",
        "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx",
        "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx",
        "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
        "vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx",
        "vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
        "vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
        "vision/style_transfer/fast_neural_style/model/candy-8.onnx",
        "vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        "/vision/style_transfer/fast_neural_style/model/mosaic-8.onnx",
        "vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        "vision/style_transfer/fast_neural_style/model/pointilism-8.onnx",
        "vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        "vision/style_transfer/fast_neural_style/model/rain-princess-8.onnx",
        "vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
        "vision/style_transfer/fast_neural_style/model/udnie-8.onnx",
        "vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
        "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
        "text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        "text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
        "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx",
        "text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
        "text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx",
        "text/machine_comprehension/roberta/model/roberta-base-11.onnx",
        "text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx",
        "text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.onnx",
        "text/machine_comprehension/t5/model/t5-encoder-12.onnx"

    )


    init {
    }




    @Test
    fun test() {
        modelPaths.forEach {
            pullModel(it)
        }

        if(!runOnly.isEmpty()) {
            runOnly.forEach { path ->
                testModel(path)
            }
        }

        else
            modelPaths.forEach { path ->
                testModel(path)
            }
    }

    fun testModel(path: String) {

        val modelArchive = File(modelDirectory,filenameFromPath(path))
        pullModel(path)
        val onnxImporter = OnnxFrameworkImporter()
        val loadedGraph = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(modelArchive))
        val onnxIRGraph = OnnxIRGraph(loadedGraph.graph,onnxImporter.registry)
        val toPrint = StringBuilder()
        loadedGraph.graph.initializerList.forEach {
            toPrint.append(it.name)
            toPrint.append(it.dimsList.toString())
            toPrint.append("\n")
        }

        loadedGraph.graph.inputList.forEach {
            println(it)
        }

        println("Loaded initializers  $toPrint")
        println("Running model from model path $path")

        val onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,loadedGraph.graph.inputList.map { input -> input.name },loadedGraph.graph.outputList.map { input -> input.name })
        val dynamicVariables =
            importer.suggestDynamicVariables(onnxIRGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
        val outputs = onnxGraphRunner.run(dynamicVariables)
        val debugPrint = StringBuilder()
        outputs.forEach { name, array ->
            debugPrint.append("$name and shape ${array.shapeInfoToString()}\n")
        }
        println(debugPrint)
        val imported = onnxImporter.runImport(modelArchive.absolutePath,dynamicVariables)
        println(imported.summary())
        val batchOutput = imported.batchOutput()
        batchOutput.placeholders = dynamicVariables
        batchOutput.outputs = onnxIRGraph.graphOutputs()
        batchOutput.outputSingle()
        //assertEquals("Onnx runtime outputs not equal to list of assertions pre provided",outputs,outputAssertions)
        // assertEquals("Onnx runtime outputs not equal to nd4j outputs",outputAssertions,nd4jOutputs)
    }


    fun filenameFromPath(modelPath: String): String {
        return modelPath.split("/").last()
    }


    fun modelDatasetsForArchive(modelPath: String): List<InputDataset> {
        val modelArchive = File(modelDirectory,filenameFromPath(modelPath))
        val listedFiles = ArchiveUtils.tarGzListFiles(modelArchive).filter { input -> input.contains("test_data_set") }
        val mapOfInputsDataSets = HashMap<Int,MutableList<String>>()
        val mapOfOutputDataSets = HashMap<Int,MutableList<String>>()
        val numDatasets = numTestDataSets(modelPath)
        val ret = ArrayList<InputDataset>()
        listedFiles.forEach { name ->
            if(name.contains("/test_data_set")) {
                val index = name.split("/").filter { input -> input.isNotEmpty() && input.matches("test_data_set.*".toRegex())}
                    .map { input ->
                        Integer.parseInt(input.replace("test_data_set_","")) }.first()
                if(!mapOfInputsDataSets.containsKey(index)) {
                    val newList = ArrayList<String>()
                    mapOfInputsDataSets[index] = newList
                }

                if(!mapOfOutputDataSets.containsKey(index)) {
                    val newList = ArrayList<String>()
                    mapOfOutputDataSets[index] = newList
                }

                val finalName = name.split("/").last()
                if(finalName.matches("input_\\d+\\.pb".toRegex())) {
                    mapOfInputsDataSets[index]!!.add(name)
                }

                if(finalName.matches("output_\\d+\\.pb".toRegex())) {
                    mapOfOutputDataSets[index]!!.add(name)
                }
            }

        }

        for(i in 0 until numDatasets) {
            val inputs = mapOfInputsDataSets[i]!!
            val outputs = mapOfOutputDataSets[i]!!
            val inputDataset = InputDataset(i,inputs,outputs)
            ret.add(inputDataset)
        }

        return ret

    }


    fun numTestDataSets(modelPath: String): Int {
        val listOfFiles = ArchiveUtils.tarGzListFiles(File(modelDirectory,filenameFromPath(modelPath)))
        var currentMax = 0
        listOfFiles.filter { input -> input.contentEquals("test_data_set") }.forEach { name ->
            val num = Integer.parseInt(name.replace("test_data_set_",""))
            currentMax = num.coerceAtLeast(currentMax)
        }

        //0 index based
        return (currentMax + 1)
    }

    fun pullModel(modelPath: String) {
        val modelUrl = URI.create("$modelBaseUrl/$modelPath").toURL()
        println("Download model $modelPath from $modelUrl")
        val fileName = modelPath.split("/").last()
        val modelFileArchive =  File(modelDirectory,fileName)
        if(modelFileArchive.exists()) {
            println("Skipping archive ${modelFileArchive.absolutePath}")
            return
        }
        FileUtils.copyURLToFile( modelUrl,modelFileArchive, Downloader.DEFAULT_CONNECTION_TIMEOUT, Downloader.DEFAULT_CONNECTION_TIMEOUT)
        if(modelFileArchive.endsWith(".gz"))
            println("Files in archive  ${ArchiveUtils.tarGzListFiles(modelFileArchive)}")
    }

}

