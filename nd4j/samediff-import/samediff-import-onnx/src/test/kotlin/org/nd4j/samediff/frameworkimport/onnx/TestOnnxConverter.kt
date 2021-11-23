package org.nd4j.samediff.frameworkimport.onnx

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource
import java.io.File

class TestOnnxConverter {

    @Test
    fun upgradeModel() {
        val onnxConverter = OnnxConverter()
        val inputModel = ClassPathResource("lenet.onnx").file
        val outputFile = File(System.getProperty("java.io.tmpdir"),"output.onnx")
        onnxConverter.convertModel(inputModel,outputFile)
        assertTrue(outputFile.exists())
        outputFile.delete()
    }

}