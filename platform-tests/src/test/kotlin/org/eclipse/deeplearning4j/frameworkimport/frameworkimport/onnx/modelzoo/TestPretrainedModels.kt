

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
package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.modelzoo

import onnx.Onnx
import org.apache.commons.io.FileUtils
import org.eclipse.deeplearning4j.testinit.convertedModelDirectory
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import org.nd4j.common.resources.Downloader
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.common.util.ArchiveUtils
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.onnx.OnnxConverter
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraphRunner
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.io.File
import java.net.URI
import java.nio.file.Path
import javax.annotation.concurrent.NotThreadSafe

data class InputDataset(var dataSetIndex: Int,var inputPaths: List<String>,var outputPaths: List<String>)
@Tag(TagNames.ONNX)
@NotThreadSafe
@Disabled("Crashes in multi threaded context. Run manually.")
class TestPretrainedModels {

    var modelBaseUrl = "https://media.githubusercontent.com/media/onnx/models/master"
    private var modelDirectory = File(File(System.getProperty("user.home")),"models/")
    var importer = OnnxFrameworkImporter()
    var converter = OnnxConverter()
    var runOnly = emptySet<String>()
    var dontRunRegexes = setOf("")


    var modelPaths = setOf(//"vision/body_analysis/age_gender/models/age_googlenet.onnx",
        //"vision/body_analysis/age_gender/models/gender_googlenet.onnx",
        //seems to fail in converter due to bad memory alloc
        //6g of ram
        //"vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx",
        //8g of ram
        //"vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_imdb_wiki.onnx",
        //"vision/body_analysis/age_gender/models/vgg_ilsvrc_16_gender_imdb_wiki.onnx",
        //java.lang.RuntimeException:  This is an invalid model. Error in Node:bn0 : Unrecognized attribute: spatial for operator BatchNormalization
        //"vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-2.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-7.onnx",
        //"vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        //broken on softmax node in ORT: axis == 2 is invalid. Output shape is invalid?
        //"vision/body_analysis/ultraface/models/version-RFB-320.onnx",
        //ode returned while running Reshape node. Name:'' Status Message: /__w/javacpp-presets/javacpp-presets/onnxruntime/cppbuild/linux-x86_64/onnxruntime/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:42 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, std::vector<long int>&, bool) gsl::narrow_cast<int64_t>(input_shape.Size()) == size was false. The input tensor cannot be reshaped to the requested shape. Input shape:{1,1000}, requested shape:{}
        //"vision/classification/alexnet/model/bvlcalexnet-9.onnx",
        //Non-zero status code returned while running Reshape node. Name:'' Status Message: /__w/javacpp-presets/javacpp-presets/onnxruntime/cppbuild/linux-x86_64/onnxruntime/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:42 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, std::vector<long int>&, bool) gsl::narrow_cast<int64_t>(input_shape.Size()) == size was false. The input tensor cannot be reshaped to the requested shape. Input shape:{1,1000}, requested shape:{}
        //"vision/classification/caffenet/model/caffenet-9.onnx",
        //java.lang.IllegalStateException: Node name was empty!
        //"vision/classification/densenet-121/model/densenet-9.onnx",
        //:/__w/javacpp-presets/javacpp-presets/onnxruntime/cppbuild/linux-x86_64/onnxruntime/onnxruntime/core/graph/graph.cc:1143 void onnxruntime::Graph::InitializeStateFromModelFileGraphProto() node_arg was false. Graph ctor should have created NodeArg for initializer. Missing:efficientnet-lite4/model/head/tpu_batch_normalization/ReadVariableOp_1:0
        //"vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx",
        //le running Reshape node. Name:'' Status Message: /__w/javacpp-presets/javacpp-presets/onnxruntime/cppbuild/linux-x86_64/onnxruntime/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:42 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, std::vector<long int>&, bool) gsl::narrow_cast<int64_t>(input_shape.Size()) == size was false. The input tensor cannot be reshaped to the requested shape. Input shape:{1,1000}, requested shape:{}
        //"vision/classification/inception_and_googlenet/googlenet/model/googlenet-9.onnx",
        // while running Reshape node. Name:'' Status Message: /__w/javacpp-presets/javacpp-presets/onnxruntime/cppbuild/linux-x86_64/onnxruntime/onnxruntime/core/providers/cpu/tensor/reshape_helper.h:42 onnxruntime::ReshapeHelper::ReshapeHelper(const onnxruntime::TensorShape&, std::vector<long int>&, bool) gsl::narrow_cast<int64_t>(input_shape.Size()) == size was false. The input tensor cannot be reshaped to the requested shape. Input shape:{1,1000}, requested shape:{}
        //"vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-9.onnx",
        //"vision/classification/mnist/model/mnist-8.onnx",
        //java.lang.IllegalStateException: Node name was empty!
        //"vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-9.onnx",
        "vision/classification/resnet/model/resnet101-v1-7.onnx",
        //"vision/classification/resnet/model/resnet152-v2-7.onnx",
        //"vision/classification/resnet/model/resnet18-v1-7.onnx",
        //"vision/classification/resnet/model/resnet18-v2-7.onnx",
        //"vision/classification/resnet/model/resnet34-v1-7.onnx",
        //"vision/classification/resnet/model/resnet34-v2-7.onnx",
        //"vision/classification/shufflenet/model/shufflenet-9.onnx",
        //"vision/classification/shufflenet/model/shufflenet-v2-10.onnx",
        //"vision/classification/squeezenet/model/squeezenet1.0-9.onnx",
        //"vision/classification/squeezenet/model/squeezenet1.1-7.onnx",
        //"vision/classification/vgg/model/vgg16-7.onnx",
        //"vision/classification/vgg/model/vgg16-bn-7.onnx",
        //"vision/classification/vgg/model/vgg19-7.onnx",
        //"vision/classification/vgg/model/vgg19-caffe2-9.onnx",
        //"vision/classification/zfnet-512/model/zfnet512-9.onnx",
        //"vision/object_detection_segmentation/duc/model/ResNet101-DUC-7.onnx",
        //"vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx",
        //"vision/object_detection_segmentation/fcn/model/fcn-resnet101-11.onnx",
        //"vision/object_detection_segmentation/fcn/model/fcn-resnet50-11.onnx",
        //"vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-10.onnx",
        //"vision/object_detection_segmentation/retinanet/model/retinanet-9.onnx",
        //"vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",
        //"vision/object_detection_segmentation/ssd/model/ssd-10.onnx",
        //"vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-7.onnx",
        //"vision/object_detection_segmentation/tiny-yolov2/model/tinyyolov2-8.onnx",
        //"vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx",
        //"vision/object_detection_segmentation/yolov2-coco/model/yolov2-coco-9.onnx",
        //"vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx",
        //"vision/object_detection_segmentation/yolov4/model/yolov4.onnx",
        //"vision/style_transfer/fast_neural_style/model/candy-8.onnx",
        //"vision/style_transfer/fast_neural_style/model/candy-9.onnx",
        //"/vision/style_transfer/fast_neural_style/model/mosaic-8.onnx",
        //"vision/style_transfer/fast_neural_style/model/mosaic-9.onnx",
        //"vision/style_transfer/fast_neural_style/model/pointilism-8.onnx",
        //"vision/style_transfer/fast_neural_style/model/pointilism-9.onnx",
        //"vision/style_transfer/fast_neural_style/model/rain-princess-8.onnx",
        //"vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx",
        //"vision/style_transfer/fast_neural_style/model/udnie-8.onnx",
        //"vision/style_transfer/fast_neural_style/model/udnie-9.onnx",
        //"vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx",
        //"text/machine_comprehension/bert-squad/model/bertsquad-10.onnx",
        //"text/machine_comprehension/bert-squad/model/bertsquad-8.onnx",
        //"text/machine_comprehension/bidirectional_attention_flow/model/bidaf-9.onnx",
        //"text/machine_comprehension/gpt-2/model/gpt2-10.onnx",
        //"text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx",
        //"text/machine_comprehension/roberta/model/roberta-base-11.onnx",
        //"text/machine_comprehension/roberta/model/roberta-sequence-classification-9.onnx",
        //"text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.onnx",
        //"text/machine_comprehension/t5/model/t5-encoder-12.onnx"

    )


    init {
    }




    @Test
    fun test(@TempDir tempPath: Path) {
        if(runOnly.isNotEmpty()) {
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

        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)
        var newModel = File(convertedModelDirectory,path.split("/").last())
        if(!newModel.exists()) {
            println("Model at path $newModel does not exist!")
            return
        }
        var onnxImporter = OnnxFrameworkImporter()
        var loadedGraph = Onnx.ModelProto.parseFrom(newModel.readBytes())


        var onnxIRGraph = OnnxIRGraph(loadedGraph.graph,onnxImporter.registry)
        var toPrint = StringBuilder()
        loadedGraph.graph.initializerList.forEach {
            toPrint.append(it.name)
            toPrint.append(it.dimsList.toString())
            toPrint.append("\n")
        }

        loadedGraph.graph.inputList.forEach {
            println(it)
        }

        var inputList = loadedGraph.graph.inputList.map { input -> input.name }

        var appendAllOutputs = false
        var outputList = if(appendAllOutputs) {
            loadedGraph.graph.nodeList.map { input -> input.name }
        } else {
            loadedGraph.graph.outputList.map { input -> input.name }
        }

        loadedGraph = null
        System.gc()





        var outputListMutable = ArrayList(outputList)

        println("Loaded initializers  $toPrint")
        println("Running model from model path $path")

        var onnxGraphRunner = OnnxIRGraphRunner(onnxIRGraph,inputList,outputListMutable)
        var dynamicVariables =
            importer.suggestDynamicVariables(onnxIRGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
        var outputs = onnxGraphRunner.run(dynamicVariables)
        var debugPrint = StringBuilder()
        outputs.forEach { (name, array) ->
            debugPrint.append("$name and shape ${array.shapeInfoToString()}\n")
        }
        println(debugPrint)
        var imported = onnxImporter.runImport(newModel.absolutePath,dynamicVariables)
        println(imported.summary())
        var batchOutput = imported.batchOutput()
        batchOutput.placeholders = dynamicVariables
        batchOutput.outputs = onnxIRGraph.graphOutputs()
        batchOutput.output()
        //assertEquals("Onnx runtime outputs not equal to list of assertions pre provided",outputs,outputAssertions)
        // assertEquals("Onnx runtime outputs not equal to nd4j outputs",outputAssertions,nd4jOutputs)
    }


    fun filenameFromPath(modelPath: String): String {
        return modelPath.split("/").last()
    }


    fun modelDatasetsForArchive(modelPath: String): List<InputDataset> {
        var modelArchive = File(modelDirectory,filenameFromPath(modelPath))
        var listedFiles = ArchiveUtils.tarGzListFiles(modelArchive).filter { input -> input.contains("test_data_set") }
        var mapOfInputsDataSets = HashMap<Int,MutableList<String>>()
        var mapOfOutputDataSets = HashMap<Int,MutableList<String>>()
        var numDatasets = numTestDataSets(modelPath)
        var ret = ArrayList<InputDataset>()
        listedFiles.forEach { name ->
            if(name.contains("/test_data_set")) {
                var index = name.split("/").filter { input -> input.isNotEmpty() && input.matches("test_data_set.*".toRegex())}
                    .map { input ->
                        Integer.parseInt(input.replace("test_data_set_","")) }.first()
                if(!mapOfInputsDataSets.containsKey(index)) {
                    var newList = ArrayList<String>()
                    mapOfInputsDataSets[index] = newList
                }

                if(!mapOfOutputDataSets.containsKey(index)) {
                    var newList = ArrayList<String>()
                    mapOfOutputDataSets[index] = newList
                }

                var finalName = name.split("/").last()
                if(finalName.matches("input_\\d+\\.pb".toRegex())) {
                    mapOfInputsDataSets[index]!!.add(name)
                }

                if(finalName.matches("output_\\d+\\.pb".toRegex())) {
                    mapOfOutputDataSets[index]!!.add(name)
                }
            }

        }

        for(i in 0 until numDatasets) {
            var inputs = mapOfInputsDataSets[i]!!
            var outputs = mapOfOutputDataSets[i]!!
            var inputDataset = InputDataset(i,inputs,outputs)
            ret.add(inputDataset)
        }

        return ret

    }


    fun numTestDataSets(modelPath: String): Int {
        var listOfFiles = ArchiveUtils.tarGzListFiles(File(modelDirectory,filenameFromPath(modelPath)))
        var currentMax = 0
        listOfFiles.filter { input -> input.contentEquals("test_data_set") }.forEach { name ->
            var num = Integer.parseInt(name.replace("test_data_set_",""))
            currentMax = num.coerceAtLeast(currentMax)
        }

        //0 index based
        return (currentMax + 1)
    }

    fun pullModel(modelPath: String) {
        var modelUrl = URI.create("$modelBaseUrl/$modelPath").toURL()
        println("Download model $modelPath from $modelUrl")
        var fileName = modelPath.split("/").last()
        var modelFileArchive =  File(modelDirectory,fileName)
        if(modelFileArchive.exists()) {
            println("Skipping archive ${modelFileArchive.absolutePath}")
            return
        }
        FileUtils.copyURLToFile( modelUrl,modelFileArchive, Downloader.DEFAULT_CONNECTION_TIMEOUT, Downloader.DEFAULT_CONNECTION_TIMEOUT)
        if(modelFileArchive.endsWith(".gz"))
            println("Files in archive  ${ArchiveUtils.tarGzListFiles(modelFileArchive)}")
    }

}

