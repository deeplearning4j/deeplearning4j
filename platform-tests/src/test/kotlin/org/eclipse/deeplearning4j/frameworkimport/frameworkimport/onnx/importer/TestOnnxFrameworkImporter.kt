package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.serde.SDZSerializer
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.resources.Resources
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import java.io.File
import java.util.*

class TestOnnxFrameworkImporter {





    @Test
    fun testOther() {
        Nd4j.getEnvironment().isVariableTracingEnabled = true
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport("/home/agibsonccc/Documents/GitHub/deeplearning4j/bge-base-en-v1.5-optimized.onnx")
        SDZSerializer.save(imported, File("bge-base-en-v1.5.sdz"),true,Collections.emptyMap())
        val sd = SDZSerializer.load(File("bge-base-en-v1.5.sdz"),true)
        println(sd.summary())
    }




}