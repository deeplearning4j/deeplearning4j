package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.serde.SDZSerializer
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.resources.Resources
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import java.io.File
import java.util.*

class TestOnnxFrameworkImporter {


    @Test
    fun testOther() {
        Nd4j.getEnvironment().isVariableTracingEnabled = true
        Nd4j.getEnvironment().isLogNativeNDArrayCreation = true
        Nd4j.getEnvironment().isFuncTracePrintJavaOnly = true
        val importer = OnnxFrameworkImporter()
        val filePath = "/home/agibsonccc/Documents/GitHub/deeplearning4j/bge-base-en-v1.5-optimized.onnx";
        val runner = OnnxRuntimeRunner(filePath)
        val imported = importer.runImport(filePath);

        val arr = Nd4j.readBinary(File("inputs.bin"))
        val inputPlaceHolder = mutableMapOf<String, INDArray>()
        inputPlaceHolder["input_ids"] = arr
        val initMatmUl = runner.getConstantOrInitializer("onnx::MatMul_1509")
        val runnerOutput = runner.exec(
            inputPlaceHolder, listOf(
                "/encoder/layer.0/attention/self/Reshape_3_output_0",
                "1492"
            )
        )
        val output = imported.output(inputPlaceHolder, "1492")
        val dryRun = imported.dryRunExecutionDAG("1492")
        val finalFile = File("bge-base-en-v1.5.sdz")
        SDZSerializer.save(imported, finalFile, true, Collections.emptyMap())
        val sd = SDZSerializer.load(finalFile, true)

        println(sd.summary())
        println()
    }
}




