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
package org.nd4j.samediff.frameworkimport.reflect

import io.github.classgraph.ClassGraph
import io.github.classgraph.ScanResult
import org.apache.commons.io.FileUtils
import org.apache.commons.io.IOUtils
import org.nd4j.common.config.ND4JSystemProperties
import org.nd4j.common.io.ClassPathResource
import java.io.File
import java.nio.charset.Charset

object ClassGraphHolder {
    var scannedClasses =   ClassGraph()
        .disableModuleScanning()
        .enableMethodInfo()
        .enableAnnotationInfo()
        .enableClassInfo()
        .scan()

    init {
        scannedClasses =   ClassGraph()
            .disableModuleScanning()
            .enableMethodInfo()
            .enableAnnotationInfo()
            .enableClassInfo()
            .scan()
        if(System.getProperties().containsKey(ND4JSystemProperties.CLASS_GRAPH_SCAN_RESOURCES)) {
            val resource = ClassPathResource(System.getProperty(ND4JSystemProperties.CLASS_GRAPH_SCAN_RESOURCES))
            val content = resource.inputStream
            val contentString = IOUtils.toString(content, Charset.defaultCharset())
            scannedClasses = ScanResult.fromJSON(contentString)
        }
    }


    @JvmStatic
    fun saveScannedClasses(inputPath: File) {
        FileUtils.write(inputPath,scannedClasses.toJSON(), Charset.defaultCharset())
    }

    @JvmStatic
    fun loadFromJson(inputJson: String): ScanResult {
        scannedClasses = ScanResult.fromJSON(inputJson)
        return scannedClasses
    }


}