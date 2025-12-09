package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter

@Tag(TagNames.ONNX)
class TestOnnxFrameworkImporter {

    @Test
    fun testOther() {
        Nd4j.getRandom().setSeed(12345)
        Nd4j.getEnvironment().isFuncTracePrintJavaOnly = true

        val modelFile = ClassPathResource("mnist.onnx").file
        val importer = OnnxFrameworkImporter()

        OnnxRuntimeRunner(modelFile.absolutePath).use { runner ->
            val imported = importer.runImport(modelFile.absolutePath)

            val inputInfo = runner.inputs.first()
            val inputName = inputInfo.name
            val inputShape = inputInfo.type.tensorType.shape.dimList
                .map { dim -> if (dim.hasDimValue()) dim.dimValue else 1L }
                .toLongArray()

            val input = Nd4j.rand(DataType.FLOAT, *inputShape)
            val inputs = mutableMapOf(inputName to input)

            val outputName = runner.outputs.first().name
            val runnerOutput = runner.exec(inputs, listOf(outputName))[outputName]!!
            val sameDiffOutput = imported.output(inputs, outputName)[outputName]!!

            assertArrayEquals(runnerOutput.shape(), sameDiffOutput.shape())
            assertTrue(runnerOutput.equalsWithEps(sameDiffOutput, 1e-4))
        }
    }
}
