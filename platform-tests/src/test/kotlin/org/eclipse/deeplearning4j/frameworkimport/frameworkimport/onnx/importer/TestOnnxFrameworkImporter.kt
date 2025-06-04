package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.resources.Resources
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import java.util.*

class TestOnnxFrameworkImporter {





    @Test
    fun testOther() {
        val importer = OnnxFrameworkImporter()
        importer.runImport("/home/agibsonccc/Documents/GitHub/kompile/./anserini-models/bge-base-en-v1.5/bge-base-en-v1.5.onnx")
    }



    @Test
    fun testConstantInitialization() {
        val importer = OnnxFrameworkImporter()
        val file = Resources.asFile("onnx_graphs/output_cnn_mnist.onnx")
        //tests model import with constant initializers where an output of a constant node is
        //defined
        val output = importer.runImport(file.absolutePath, suggestDynamicVariables = true)
        //ensure that the graph with an eager mode can automatically import the model
        assertNotNull(output)
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
        val result  = importer.runImport(file.absolutePath, emptyMap(), suggestDynamicVariables = true)
        result.outputAll(Collections.singletonMap("input.1",Nd4j.ones(1,3,224,224)))
    }




}