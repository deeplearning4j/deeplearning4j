
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
package org.eclipse.deeplearning4j.testinit

import onnx.Onnx
import org.apache.commons.io.IOUtils
import org.nd4j.samediff.frameworkimport.onnx.OnnxConverter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import kotlin.system.exitProcess

var convertedModelDirectory = File(File(System.getProperty("user.home")),".nd4jtests/onnx-pretrained-converted/")


fun main(args: Array<String>) {
   var inputModelPath = File(modelDirectory, args[0].split("/").last())
   var newModel = File(convertedModelDirectory, inputModelPath.name)
   if(newModel.exists()) {
      println("New model ${newModel.absolutePath} already exists. Exiting.")
      exitProcess(0)
   }

   var onnxConverter = OnnxConverter()
   var modelProto = loadAndPreProcess(inputModelPath,onnxConverter)
   var tempModelFile = File(System.getProperty("java.io.tmpdir"),inputModelPath.name)
   tempModelFile.deleteOnExit()
   IOUtils.write(modelProto.toByteArray(),FileOutputStream(tempModelFile))
   println("Write temp model at ${tempModelFile.absolutePath}")
   if(!convertedModelDirectory.exists())
      convertedModelDirectory.mkdirs()
   onnxConverter.convertModel(tempModelFile, newModel)
}

fun loadAndPreProcess(inputModel: File,onnxConverter: OnnxConverter): Onnx.ModelProto {
   var modelProto: Onnx.ModelProto = Onnx.ModelProto.parseFrom(FileInputStream(inputModel))
   var graphProto = onnxConverter.addConstValueInfoToGraph(modelProto.graph)
   modelProto = modelProto.toBuilder().setGraph(graphProto).build()
   return modelProto
}