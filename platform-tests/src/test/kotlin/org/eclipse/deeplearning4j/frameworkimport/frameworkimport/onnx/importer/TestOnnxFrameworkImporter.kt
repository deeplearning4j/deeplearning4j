package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.TrainingConfig
import org.nd4j.common.io.ClassPathResource
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import java.util.*

class TestOnnxFrameworkImporter {



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
        val arr = Nd4j.ones(1,1,28,28)
        val inputs = mapOf("import/Placeholder" to arr)
        val importer = OnnxFrameworkImporter()
        val file = ClassPathResource("lenet.onnx").file
        val result  = importer.runImport(file.absolutePath, inputs)
        val labelsVar = result.placeHolder("label", DataType.FLOAT,1,10,1,1)
        val output = result.getVariable("raw_output___13")!!
        result.loss().softmaxCrossEntropy("loss",labelsVar,output,null)
        val labels = Nd4j.ones(1,10,1,1)
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