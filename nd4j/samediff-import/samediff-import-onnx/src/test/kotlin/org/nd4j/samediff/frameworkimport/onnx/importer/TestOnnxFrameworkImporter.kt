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
package org.nd4j.samediff.frameworkimport.onnx.importer

import GraphPreProcessRunner
import onnx.Onnx
import org.junit.Test
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.nd4j.autodiff.samediff.TrainingConfig
import org.nd4j.common.io.ClassPathResource
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.samediff.frameworkimport.onnx.NodeProto
import org.nd4j.samediff.frameworkimport.onnx.createSingleNodeGraph
import org.nd4j.samediff.frameworkimport.onnx.definitions.onnxOpRegistry
import org.nd4j.samediff.frameworkimport.onnx.definitions.registry
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraphRunner
import org.nd4j.samediff.frameworkimport.onnx.runAssertion
import java.io.File
import java.io.FileInputStream
import java.util.*

class TestOnnxFrameworkImporter {


    @Test
    fun testUnsqueezeUpgrade() {
        val importer = OnnxFrameworkImporter()
        val inputData = Nd4j.linspace(1,15,15).reshape(1,3,1,5)
        val input = mapOf("x" to inputData)

        val outputs = listOf("y")
        val attributes = mapOf("axes" to listOf(-2))
        val inputs = listOf("x")
        val graph = createSingleNodeGraph(input,"Unsqueeze",attributes,outputs,inputs)
        val preProcessRunner = GraphPreProcessRunner()
        val irGraph = OnnxIRGraph(graph, registry())
        preProcessRunner.preProcessGraph(irGraph)
        val internalValue = irGraph.internalValue()
        runAssertion(internalValue,input,outputs)
    }

    @Test
    fun testSuggestedVariables() {
        val importer = OnnxFrameworkImporter()
        val file = ClassPathResource("mobilenet.onnx").file
        val suggestedVariables = importer.suggestDynamicVariables(file.absolutePath)
        assertTrue(suggestedVariables.containsKey("input.1"))
        val shape = suggestedVariables["input.1"]!!.shape()
        assertArrayEquals(longArrayOf(1,3,224,224),shape)

    }

    @Test
    fun testMobileNet() {
        Nd4j.getExecutioner().enableVerboseMode(true)
        Nd4j.getExecutioner().enableDebugMode(true)
        val importer = OnnxFrameworkImporter()
        val file = ClassPathResource("mobilenet.onnx").file
        val result  = importer.runImport(file.absolutePath, emptyMap(),suggestDynamicVariables = true)
        result.outputAll(Collections.singletonMap("input.1",Nd4j.ones(1,3,224,224)))
    }

    @Test
    fun testLenet() {
        Nd4j.getExecutioner().enableDebugMode(true)
        Nd4j.getExecutioner().enableVerboseMode(true)
        val importer = OnnxFrameworkImporter()
        val file = ClassPathResource("lenet.onnx").file
        val result  = importer.runImport(file.absolutePath, emptyMap())
        val labelsVar = result.placeHolder("label", DataType.FLOAT,1,10)
        val output = result.getVariable("raw_output___13")!!
        result.loss().softmaxCrossEntropy("loss",labelsVar,output,null)
        val arr = Nd4j.ones(1,1,28,28)
        val labels = Nd4j.ones(1,10,5,1)
        result.convertConstantsToVariables()
        val trainingConfig = TrainingConfig.builder()
        trainingConfig.updater(Adam())
        trainingConfig.dataSetFeatureMapping("import/Placeholder")
        trainingConfig.dataSetLabelMapping("label")
        trainingConfig.minimize("loss")
        result.trainingConfig = trainingConfig.build()

        val inputData = DataSet(arr,labels)
        result.fit(inputData)
    }



}