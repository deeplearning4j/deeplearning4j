
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

import com.google.common.io.Files
import org.apache.commons.io.FileUtils
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.resources.Downloader
import org.nd4j.common.util.ArchiveUtils
import java.io.File
import java.net.URI
import java.nio.charset.Charset


val modelBaseUrl = "https://media.githubusercontent.com/media/onnx/models/master"
val modelDirectory = File(File(System.getProperty("user.home")),".nd4jtests/onnx-pretrained/")


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


fun main(args: Array<String>) {
    val modelPathList = File(args[0])
    val readLines = Files.readLines(modelPathList, Charset.defaultCharset())
    readLines.forEach { line ->
        pullModel(line)
    }

}