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
import org.nd4j.common.tests.tags.ExpensiveTest
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraphRunner
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRTensor
import java.io.File
import java.net.URI

data class InputDataset(val dataSetIndex: Int,val inputPaths: List<String>,val outputPaths: List<String>)
@ExpensiveTest
class TestPretrainedModels {

    val modelBaseUrl = "https://media.githubusercontent.com/media/onnx/models/master"
    val modelDirectory = File(File(System.getProperty("user.home")),"models/")
    val runOnly = emptySet<String>()
    val dontRunRegexes = setOf("")
    val modelPaths = setOf("vision/body_analysis/age_gender/models/age_googlenet.onnx",
        "vision/body_analysis/age_gender/models/gender_googlenet.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx",
        "vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx",
        //"vision/body_analysis/arcface/model/arcfaceresnet100-8.tar.gz",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.tar.gz",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.tar.gz",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.tar.gz",
        "vision/body_analysis/ultraface/models/version-RFB-320.onnx",
       // "vision/classification/alexnet/model/bvlcalexnet-3.tar.gz",
       // "vision/classification/alexnet/model/bvlcalexnet-6.tar.gz",
        //"vision/classification/alexnet/model/bvlcalexnet-7.tar.gz",
       // "vision/classification/alexnet/model/bvlcalexnet-8.tar.gz",
        "vision/classification/alexnet/model/bvlcalexnet-9.tar.gz",
       // "vision/classification/caffenet/model/caffenet-3.tar.gz",
       // "vision/classification/caffenet/model/caffenet-6.tar.gz",
       // "vision/classification/caffenet/model/caffenet-7.tar.gz",
       // "vision/classification/caffenet/model/caffenet-8.tar.gz",
        "vision/classification/caffenet/model/caffenet-9.tar.gz",
       // "vision/classification/densenet-121/model/densenet-3.tar.gz",
        //"vision/classification/densenet-121/model/densenet-6.tar.gz",
        //"vision/classification/densenet-121/model/densenet-7.tar.gz",
        //"vision/classification/densenet-121/model/densenet-8.tar.gz",
        "vision/classification/densenet-121/model/densenet-9.tar.gz",
        "vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.tar.gz",
       // "vision/classification/inception_and_googlenet/googlenet/model/googlenet-3.tar.gz",
       // "vision/classification/inception_and_googlenet/googlenet/model/googlenet-6.tar.gz",
       // "vision/classification/inception_and_googlenet/googlenet/model/googlenet-7.tar.gz",
       // "vision/classification/inception_and_googlenet/googlenet/model/googlenet-8.tar.gz",
        "vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.tar.gz",
     //   "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-3.tar.gz",
     //   "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-6.tar.gz",
     //   "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-7.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-8.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-9.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-3.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-6.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-7.tar.gz",
      //  "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-8.tar.gz",
        "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.tar.gz",
        //"vision/classification/mnist/model/mnist-1.tar.gz",
        //"vision/classification/mnist/model/mnist-7.tar.gz",
        "vision/classification/mnist/model/mnist-8.tar.gz",
        "vision/classification/mobilenet/model/mobilnetv2-7.tar.gz",
        //"vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-3.tar.gz",
        //"vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-6.tar.gz",
        //"vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-7.tar.gz",
        //"vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-8.tar.gz",
        "vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.tar.gz",
        "vision/classification/resnet/model/resnet101-v1-7.tar.gz",
        "vision/classification/resnet/model/resnet152-v2-7.tar.gz",
        "vision/classification/resnet/model/resnet18-v1-7.tar.gz",
        "vision/classification/resnet/model/resnet18-v2-7.tar.gz",
        "vision/classification/resnet/model/resnet34-v1-7.tar.gz",
        "vision/classification/resnet/model/resnet34-v2-7.tar.gz",
        //"vision/classification/resnet/model/resnet50-caffe2-v1-3.tar.gz",
        //"vision/classification/resnet/model/resnet50-caffe2-v1-6.tar.gz",
        //"vision/classification/resnet/model/resnet50-caffe2-v1-7.tar.gz",
        //"vision/classification/resnet/model/resnet50-caffe2-v1-8.tar.gz",
        //"vision/classification/resnet/model/resnet50-caffe2-v1-9.tar.gz",
        //"vision/classification/resnet/model/resnet50-v1-7.tar.gz",
        //"vision/classification/resnet/model/resnet50-v2-7.tar.gz",
        //"vision/classification/shufflenet/model/shufflenet-3.tar.gz",
        //"vision/classification/shufflenet/model/shufflenet-6.tar.gz",
        //"vision/classification/shufflenet/model/shufflenet-7.tar.gz",
        //"vision/classification/shufflenet/model/shufflenet-8.tar.gz",
        "vision/classification/shufflenet/model/shufflenet-9.tar.gz",
        "vision/classification/shufflenet/model/shufflenet-v2-10.tar.gz",
        //"vision/classification/squeezenet/model/squeezenet1.0-3.tar.gz",
        //"vision/classification/squeezenet/model/squeezenet1.0-6.tar.gz",
        //"vision/classification/squeezenet/model/squeezenet1.0-7.tar.gz",
        //"vision/classification/squeezenet/model/squeezenet1.0-8.tar.gz",
        "vision/classification/squeezenet/model/squeezenet1.0-9.tar.gz",
        "vision/classification/squeezenet/model/squeezenet1.1-7.tar.gz",
        "vision/classification/vgg/model/vgg16-7.tar.gz",
        "vision/classification/vgg/model/vgg16-bn-7.tar.gz",
        "vision/classification/vgg/model/vgg19-7.tar.gz",
       // "vision/classification/vgg/model/vgg19-caffe2-3.tar.gz",
       // "vision/classification/vgg/model/vgg19-caffe2-6.tar.gz",
       // "vision/classification/vgg/model/vgg19-caffe2-7.tar.gz",
       // "vision/classification/vgg/model/vgg19-caffe2-8.tar.gz",
        "vision/classification/vgg/model/vgg19-caffe2-9.tar.gz",
       // "vision/classification/zfnet-512/model/zfnet512-3.tar.gz",
       // "vision/classification/zfnet-512/model/zfnet512-6.tar.gz",
       // "vision/classification/zfnet-512/model/zfnet512-7.tar.gz",
       // "vision/classification/zfnet-512/model/zfnet512-8.tar.gz",
        "vision/classification/zfnet-512/model/zfnet512-9.tar.gz",
        "vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.tar.gz",
        "vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.tar.gz",
        "vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.tar.gz",
        "vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.tar.gz",
        "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.tar.gz",
        "vision/object_detection_segmentation/retinanet/model/retinanet-9.tar.gz",
        "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.tar.gz",
        "vision/object_detection_segmentation/ssd/model/ssd-10.tar.gz",
        "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.tar.gz",
        "vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.tar.gz",
        "vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.tar.gz",
        "vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx",
        "vision/object_detection_segmentation/yolov3/model/yolov3-10.tar.gz",
        "vision/object_detection_segmentation/yolov4/model/yolov4.tar.gz",
        "vision/style_transfer/fast_neural_style/model/candy-8.tar.gz",
        "vision/style_transfer/fast_neural_style/model/candy-9.tar.gz",
        "/vision/style_transfer/fast_neural_style/model/mosaic-8.tar.gz",
        "vision/style_transfer/fast_neural_style/model/mosaic-9.tar.gz",
        "vision/style_transfer/fast_neural_style/model/pointilism-8.tar.gz",
        "vision/style_transfer/fast_neural_style/model/pointilism-9.tar.gz",
        "vision/style_transfer/fast_neural_style/model/rain-princess-8.tar.gz",
        "vision/style_transfer/fast_neural_style/model/rain-princess-9.tar.gz",
        "vision/style_transfer/fast_neural_style/model/udnie-8.tar.gz",
        "vision/style_transfer/fast_neural_style/model/udnie-9.tar.gz",
        "vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz",
        "text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz",
        "text/machine_comprehension/bert-squad/model/bertsquad-8.tar.gz",
        "text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.tar.gz",
        "text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz",
        "text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.tar.gz",
        "text/machine_comprehension/roberta/model/roberta-base-11.tar.gz",
        "text/machine_comprehension/roberta/model/roberta-sequence-classification-9.tar.gz",
        "text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.tar.gz",
        "text/machine_comprehension/t5/model/t5-encoder-12.tar.gz"

    )


    init {
    }

    fun shouldRun(path: String): Boolean {
        if(path.contains(".onnx"))
            return false
        else if(!dontRunRegexes.isEmpty()) {
            for(regex in dontRunRegexes) {
                if(path.matches(regex.toRegex()))
                    return false
            }
        }

        return true
    }


    @Test
    @Disabled
    fun test() {
        modelPaths.forEach {
            pullModel(it)
        }

        if(!runOnly.isEmpty()) {
            runOnly.filter { input -> shouldRun(input) }.forEach { path ->
                testModel(path)
            }
        }

        else
            modelPaths.filter { input -> shouldRun(input) }.forEach { path ->
                testModel(path)
            }
    }

    fun testModel(path: String) {
        val modelArchive = File(modelDirectory,filenameFromPath(path))
        pullModel(path)
        val modelFile = modelFromArchive(path)
        val onnxImporter = OnnxFrameworkImporter()
        val inputDatasets = modelDatasetsForArchive(path)
        val loadedGraph = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(modelFile))
        val separateUpdatedGraph = Onnx.ModelProto.parseFrom(FileUtils.readFileToByteArray(File("C:\\Users\\agibs\\models\\model-12.onnx")))
        val upgradedGraph = OnnxIRGraph(separateUpdatedGraph.graph, onnxImporter.registry)
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
        inputDatasets.forEach {  input ->
            val dynamicVariables = HashMap<String,INDArray>()
            val outputAssertions = HashMap<String,INDArray>()
            val listOfInputNames = ArrayList<String>()
            val listOfOutputNames = ArrayList<String>()
            input.inputPaths.forEachIndexed { index,inputTensorPath ->
                val tensorName = "input_tensor_$index.pb"
                val inputFile = File(modelDirectory,tensorName)
                ArchiveUtils.tarGzExtractSingleFile(modelArchive,inputFile,inputTensorPath)
                val bytesForTensor = FileUtils.readFileToByteArray(inputFile)
                val tensor = Onnx.TensorProto.parseFrom(bytesForTensor)
                val converted = OnnxIRTensor(tensor).toNd4jNDArray()
                println("Converted to $converted")
                dynamicVariables[onnxIRGraph.inputAt(index)] = converted
                listOfInputNames.add(onnxIRGraph.inputAt(index))
            }

            input.outputPaths.forEachIndexed { index,outputTensorPath ->
                val tensorName = "output_tensor_$index.pb"
                val inputFile = File(modelDirectory,tensorName)
                ArchiveUtils.tarGzExtractSingleFile(modelArchive,inputFile,outputTensorPath)
                val bytesForTensor = FileUtils.readFileToByteArray(inputFile)
                val tensor = Onnx.TensorProto.parseFrom(bytesForTensor)
                outputAssertions[onnxIRGraph.outputAt(index)] = OnnxIRTensor(tensor).toNd4jNDArray()
                listOfOutputNames.add(onnxIRGraph.outputAt(index))
            }

            val debugOutputNames = listOf("conv2_1","norm2_1","pool2_1","conv1_1","norm1_1","pool1_1")
            val onnxGraphRunner = OnnxIRGraphRunner(upgradedGraph,listOfInputNames,debugOutputNames)
            val outputs = onnxGraphRunner.run(dynamicVariables)
            val debugPrint = StringBuilder()
            outputs.forEach { name, array ->
                debugPrint.append("$name and shape ${array.shapeInfoToString()}\n")
            }
            println(debugPrint)
            val imported = onnxImporter.runImport(modelFile.absolutePath,dynamicVariables)
            println(imported.summary())
            val batchOutput = imported.batchOutput()
            batchOutput.placeholders = dynamicVariables
            batchOutput.outputs = onnxIRGraph.graphOutputs()
            batchOutput.outputSingle()
            //assertEquals("Onnx runtime outputs not equal to list of assertions pre provided",outputs,outputAssertions)
           // assertEquals("Onnx runtime outputs not equal to nd4j outputs",outputAssertions,nd4jOutputs)
        }

    }

    fun filenameFromPath(modelPath: String): String {
        return modelPath.split("/").last()
    }

    fun modelFromArchive(modelPath: String): File {
        val modelArchive = File(modelDirectory,filenameFromPath(modelPath))
        val modelPathInArchive = ArchiveUtils.tarGzListFiles(modelArchive).first { input -> input.contains(".onnx")
                && !input.split("/").last().startsWith("._")}
        val modelName = modelPathInArchive.split("/").last()
        val finalModelPath = File(modelDirectory,modelName)
        ArchiveUtils.tarGzExtractSingleFile(modelArchive,finalModelPath,modelPathInArchive)
        return finalModelPath
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