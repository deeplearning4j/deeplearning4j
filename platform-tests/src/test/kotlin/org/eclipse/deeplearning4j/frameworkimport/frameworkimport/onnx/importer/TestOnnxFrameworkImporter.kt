package org.eclipse.deeplearning4j.frameworkimport.frameworkimport.onnx.importer

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.nd4j.autodiff.samediff.serde.SDZSerializer
import org.nd4j.common.config.ND4JSystemProperties
import org.nd4j.common.io.ClassPathResource
import org.nd4j.common.tests.tags.TagNames
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.onnxruntime.runner.OnnxRuntimeRunner
import org.nd4j.samediff.frameworkimport.onnx.importer.OnnxFrameworkImporter
import java.io.BufferedInputStream
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.nio.channels.Channels
import java.util.*

@Tag(TagNames.ONNX)
class TestOnnxFrameworkImporter {
    @Test
    fun testOther2() {
        Nd4j.getEnvironment().isFuncTracePrintJavaOnly = true
        val importer = OnnxFrameworkImporter()
        val filePath = "/home/agibsonccc/Documents/GitHub/deeplearning4j/bge-base-en-v1.5-optimized.onnx";
        val runner = OnnxRuntimeRunner(filePath)
        val imported = importer.runImport(filePath);

        // DEBUG: Print variable counts after import
        println("=== DEBUG: After ONNX Import ===")
        println("Total variables: ${imported.variables().size}")
        println("Constants: ${imported.variables().filter { it.isConstant }.size}")
        println("Placeholders: ${imported.variables().filter { it.isPlaceHolder }.size}")
        println("Variables with arrays: ${imported.variables().filter { it.arr != null }.size}")
        println("Variable names (first 20): ${imported.variables().take(20).map { it.name() }}")

        // Check internal variables map
        println("Internal variables map size: ${imported.getVariables().size}")
        println("Ops count: ${imported.ops.size}")

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

        // DEBUG: Print counts before save
        println("=== DEBUG: Before Save ===")
        println("Total variables: ${imported.variables().size}")
        println("Constants: ${imported.variables().filter { it.isConstant }.size}")
        println("Variables with arrays: ${imported.variables().filter { it.arr != null }.size}")

        val finalFile = File("bge-base-en-v1.5.sdz")
        SDZSerializer.save(imported, finalFile, true, Collections.emptyMap())

        println("=== DEBUG: After Save, Before Load ===")
        println("SDZ file size: ${finalFile.length()} bytes")

        val sd = SDZSerializer.load(finalFile, true)

        println("=== DEBUG: After Load ===")
        println("Loaded variables: ${sd.variables().size}")
        println("Loaded constants: ${sd.variables().filter { it.isConstant }.size}")
        println("Loaded variables with arrays: ${sd.variables().filter { it.arr != null }.size}")

        println(sd.summary())
        println()
    }

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

    // Cross-Encoder Model Download and Import Tests
    // HuggingFace Source URLs:
    // - ms-marco-MiniLM-L-6-v2:      https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2
    // - ms-marco-MiniLM-L-12-v2:     https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2
    // - stsb-TinyBERT-L-4:           https://huggingface.co/cross-encoder/stsb-TinyBERT-L-4
    // - mmarco-mMiniLMv2-L12-H384-v1: https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
    // - qnli-distilroberta-base:     https://huggingface.co/cross-encoder/qnli-distilroberta-base
    //
    // ONNX versions (Xenova conversions):
    // - https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2
    // - https://huggingface.co/Xenova/ms-marco-MiniLM-L-12-v2

    companion object {
        // Cross-Encoder ONNX models (Xenova conversions for Transformers.js compatibility)
        val CROSS_ENCODER_ONNX_URLS = mapOf(
            "ms-marco-MiniLM-L-6-v2" to "https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model.onnx",
            "ms-marco-MiniLM-L-12-v2" to "https://huggingface.co/Xenova/ms-marco-MiniLM-L-12-v2/resolve/main/onnx/model.onnx",
            "ms-marco-TinyBERT-L-2-v2" to "https://huggingface.co/Xenova/ms-marco-TinyBERT-L-2-v2/resolve/main/onnx/model.onnx"
        )

        // Dense Encoder ONNX models (Xenova conversions)
        val DENSE_ENCODER_ONNX_URLS = mapOf(
            "bge-base-en-v1.5" to "https://huggingface.co/Xenova/bge-base-en-v1.5/resolve/main/onnx/model.onnx",
            "bge-small-en-v1.5" to "https://huggingface.co/Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx",
            "bge-large-en-v1.5" to "https://huggingface.co/Xenova/bge-large-en-v1.5/resolve/main/onnx/model.onnx",
            "all-MiniLM-L6-v2" to "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx",
            "all-MiniLM-L12-v2" to "https://huggingface.co/Xenova/all-MiniLM-L12-v2/resolve/main/onnx/model.onnx"
        )

        val CROSS_ENCODER_VOCAB_URLS = mapOf(
            "ms-marco-MiniLM-L-6-v2" to "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main/vocab.txt",
            "ms-marco-MiniLM-L-12-v2" to "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/resolve/main/vocab.txt",
            "ms-marco-TinyBERT-L-2-v2" to "https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2/resolve/main/vocab.txt",
            "stsb-TinyBERT-L-4" to "https://huggingface.co/cross-encoder/stsb-TinyBERT-L-4/resolve/main/vocab.txt",
            "mmarco-mMiniLMv2-L12-H384-v1" to "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/sentencepiece.bpe.model",
            "qnli-distilroberta-base" to "https://huggingface.co/cross-encoder/qnli-distilroberta-base/resolve/main/vocab.json"
        )

        val DENSE_ENCODER_VOCAB_URLS = mapOf(
            "bge-base-en-v1.5" to "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/vocab.txt",
            "bge-small-en-v1.5" to "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/vocab.txt",
            "bge-large-en-v1.5" to "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/vocab.txt",
            "all-MiniLM-L6-v2" to "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt",
            "all-MiniLM-L12-v2" to "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/vocab.txt"
        )

        val CROSS_ENCODER_CONFIG_URLS = mapOf(
            "ms-marco-MiniLM-L-6-v2" to "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2/resolve/main/config.json",
            "ms-marco-MiniLM-L-12-v2" to "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2/resolve/main/config.json",
            "ms-marco-TinyBERT-L-2-v2" to "https://huggingface.co/cross-encoder/ms-marco-TinyBERT-L-2-v2/resolve/main/config.json",
            "stsb-TinyBERT-L-4" to "https://huggingface.co/cross-encoder/stsb-TinyBERT-L-4/resolve/main/config.json",
            "mmarco-mMiniLMv2-L12-H384-v1" to "https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/resolve/main/config.json",
            "qnli-distilroberta-base" to "https://huggingface.co/cross-encoder/qnli-distilroberta-base/resolve/main/config.json"
        )

        val DENSE_ENCODER_CONFIG_URLS = mapOf(
            "bge-base-en-v1.5" to "https://huggingface.co/BAAI/bge-base-en-v1.5/resolve/main/config.json",
            "bge-small-en-v1.5" to "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main/config.json",
            "bge-large-en-v1.5" to "https://huggingface.co/BAAI/bge-large-en-v1.5/resolve/main/config.json",
            "all-MiniLM-L6-v2" to "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
            "all-MiniLM-L12-v2" to "https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2/resolve/main/config.json"
        )

        /**
         * Parse timeout value from system property, with validation and default fallback
         * @param propertyName The system property name to read
         * @param defaultValue The default timeout value in milliseconds
         * @return The parsed timeout value, or default if property is not set or invalid
         */
        private fun parseTimeoutProperty(propertyName: String, defaultValue: Int): Int {
            val propertyValue = System.getProperty(propertyName) ?: return defaultValue
            
            // Use toLongOrNull to safely parse - returns null if not a valid number
            val timeoutLong = propertyValue.toLongOrNull() ?: return defaultValue
            
            // Ensure it's within valid Int range and positive (HttpURLConnection requires positive int)
            return if (timeoutLong in 1..Int.MAX_VALUE) {
                timeoutLong.toInt()
            } else {
                defaultValue
            }
        }

        fun downloadFile(urlString: String, targetFile: File) {
            // Use configurable timeouts via system properties, with defaults of 60 seconds
            // This follows the pattern used in ResourceFile and Downloader classes
            val connectTimeout = parseTimeoutProperty(
                ND4JSystemProperties.RESOURCES_CONNECTION_TIMEOUT, 
                60000  // Default: 60 seconds (matching Downloader.DEFAULT_CONNECTION_TIMEOUT and codebase standard)
            )
            
            val readTimeout = parseTimeoutProperty(
                ND4JSystemProperties.RESOURCES_READ_TIMEOUT,
                60000  // Default: 60 seconds (matching Downloader.DEFAULT_READ_TIMEOUT and codebase standard)
            )
            
            try {
                val url = URL(urlString)
                val connection = url.openConnection() as HttpURLConnection
                connection.setRequestProperty("User-Agent", "Mozilla/5.0 Kompile-Model-Manager/1.0")
                connection.instanceFollowRedirects = true
                connection.connectTimeout = connectTimeout
                connection.readTimeout = readTimeout

                val responseCode = connection.responseCode
                if (responseCode != 200) {
                    throw RuntimeException("HTTP $responseCode for $urlString")
                }

                BufferedInputStream(connection.inputStream).use { input ->
                    Channels.newChannel(input).use { rbc ->
                        FileOutputStream(targetFile).use { fos ->
                            fos.channel.transferFrom(rbc, 0, Long.MAX_VALUE)
                        }
                    }
                }
            } catch (e: SocketTimeoutException) {
                throw RuntimeException("Timeout downloading $urlString (connect timeout: ${connectTimeout}ms, read timeout: ${readTimeout}ms). " +
                    "If network is slow, increase timeouts via system properties: " +
                    "${ND4JSystemProperties.RESOURCES_CONNECTION_TIMEOUT} and ${ND4JSystemProperties.RESOURCES_READ_TIMEOUT}", e)
            } catch (e: Exception) {
                throw RuntimeException("Failed to download $urlString: ${e.message}", e)
            }
        }
    }

    @Test
    fun testDownloadMsMarcoMiniLML6Onnx() {
        val modelId = "ms-marco-MiniLM-L-6-v2"
        val onnxUrl = CROSS_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        println("Downloading $modelId from $onnxUrl")

        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }

        assertTrue(modelFile.exists(), "ONNX model should exist")
        assertTrue(modelFile.length() > 1_000_000, "ONNX model should be > 1MB, got ${modelFile.length()}")
        println("Downloaded $modelId: ${modelFile.length() / (1024.0 * 1024.0)} MB")
    }

    @Test
    fun testDownloadMsMarcoMiniLML12Onnx() {
        val modelId = "ms-marco-MiniLM-L-12-v2"
        val onnxUrl = CROSS_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        println("Downloading $modelId from $onnxUrl")

        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }

        assertTrue(modelFile.exists(), "ONNX model should exist")
        assertTrue(modelFile.length() > 1_000_000, "ONNX model should be > 1MB, got ${modelFile.length()}")
        println("Downloaded $modelId: ${modelFile.length() / (1024.0 * 1024.0)} MB")
    }

    @Test
    fun testImportMsMarcoMiniLML6ToSameDiff() {
        val modelId = "ms-marco-MiniLM-L-6-v2"
        val onnxUrl = CROSS_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 1_000_000, "SDZ file should be > 1MB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testImportMsMarcoMiniLML12ToSameDiff() {
        val modelId = "ms-marco-MiniLM-L-12-v2"
        val onnxUrl = CROSS_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 1_000_000, "SDZ file should be > 1MB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testImportMsMarcoTinyBERTL2ToSameDiff() {
        val modelId = "ms-marco-TinyBERT-L-2-v2"
        val onnxUrl = CROSS_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 100_000, "SDZ file should be > 100KB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testImportBgeBaseEnV15ToSameDiff() {
        val modelId = "bge-base-en-v1.5"
        val onnxUrl = DENSE_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "dense-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 1_000_000, "SDZ file should be > 1MB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testImportBgeSmallEnV15ToSameDiff() {
        val modelId = "bge-small-en-v1.5"
        val onnxUrl = DENSE_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "dense-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 100_000, "SDZ file should be > 100KB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testImportAllMiniLML6V2ToSameDiff() {
        val modelId = "all-MiniLM-L6-v2"
        val onnxUrl = DENSE_ENCODER_ONNX_URLS[modelId]!!
        val tempDir = File(System.getProperty("java.io.tmpdir"), "dense-encoder-tests")
        tempDir.mkdirs()

        val modelFile = File(tempDir, "$modelId.onnx")
        val sdzFile = File(tempDir, "$modelId.sdz")

        println("Downloading $modelId from $onnxUrl")
        if (!modelFile.exists()) {
            downloadFile(onnxUrl, modelFile)
        }
        println("Downloaded: ${modelFile.length() / (1024.0 * 1024.0)} MB")

        println("Importing ONNX to SameDiff...")
        val importer = OnnxFrameworkImporter()
        val imported = importer.runImport(modelFile.absolutePath)

        println("Saving to SDZ format...")
        SDZSerializer.save(imported, sdzFile, true, Collections.emptyMap())

        assertTrue(sdzFile.exists(), "SDZ file should exist")
        assertTrue(sdzFile.length() > 100_000, "SDZ file should be > 100KB")
        println("Saved $modelId.sdz: ${sdzFile.length() / (1024.0 * 1024.0)} MB")

        // Verify we can load it back
        val loaded = SDZSerializer.load(sdzFile, true)
        println("Loaded SDZ summary:")
        println(loaded.summary())
    }

    @Test
    fun testDownloadAllDenseEncoderVocabs() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "dense-encoder-tests")
        tempDir.mkdirs()

        for ((modelId, vocabUrl) in DENSE_ENCODER_VOCAB_URLS) {
            val vocabFile = File(tempDir, "$modelId-vocab.txt")

            println("Downloading vocab for $modelId from $vocabUrl")
            try {
                if (!vocabFile.exists()) {
                    downloadFile(vocabUrl, vocabFile)
                }
                assertTrue(vocabFile.exists(), "Vocab file should exist for $modelId")
                assertTrue(vocabFile.length() > 100, "Vocab file should have content for $modelId")
                println("Downloaded $modelId vocab: ${vocabFile.length()} bytes")
            } catch (e: Exception) {
                println("Warning: Failed to download vocab for $modelId: ${e.message}")
            }
        }
    }

    @Test
    fun testDownloadAllDenseEncoderConfigs() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "dense-encoder-tests")
        tempDir.mkdirs()

        for ((modelId, configUrl) in DENSE_ENCODER_CONFIG_URLS) {
            val configFile = File(tempDir, "$modelId-config.json")

            println("Downloading config for $modelId from $configUrl")
            try {
                if (!configFile.exists()) {
                    downloadFile(configUrl, configFile)
                }
                assertTrue(configFile.exists(), "Config file should exist for $modelId")
                assertTrue(configFile.length() > 100, "Config file should have content for $modelId")
                println("Downloaded $modelId config: ${configFile.length()} bytes")
                println("Content: ${configFile.readText().take(500)}")
            } catch (e: Exception) {
                println("Warning: Failed to download config for $modelId: ${e.message}")
            }
        }
    }

    @Test
    fun testDownloadAllCrossEncoderVocabs() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        for ((modelId, vocabUrl) in CROSS_ENCODER_VOCAB_URLS) {
            val extension = when {
                vocabUrl.endsWith(".json") -> ".json"
                vocabUrl.endsWith(".model") -> ".model"
                else -> ".txt"
            }
            val vocabFile = File(tempDir, "$modelId-vocab$extension")

            println("Downloading vocab for $modelId from $vocabUrl")
            try {
                if (!vocabFile.exists()) {
                    downloadFile(vocabUrl, vocabFile)
                }
                assertTrue(vocabFile.exists(), "Vocab file should exist for $modelId")
                assertTrue(vocabFile.length() > 100, "Vocab file should have content for $modelId")
                println("Downloaded $modelId vocab: ${vocabFile.length()} bytes")
            } catch (e: Exception) {
                println("Warning: Failed to download vocab for $modelId: ${e.message}")
            }
        }
    }

    @Test
    fun testDownloadAllCrossEncoderConfigs() {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "cross-encoder-tests")
        tempDir.mkdirs()

        for ((modelId, configUrl) in CROSS_ENCODER_CONFIG_URLS) {
            val configFile = File(tempDir, "$modelId-config.json")

            println("Downloading config for $modelId from $configUrl")
            if (!configFile.exists()) {
                downloadFile(configUrl, configFile)
            }

            assertTrue(configFile.exists(), "Config file should exist for $modelId")
            assertTrue(configFile.length() > 100, "Config file should have content for $modelId")
            println("Downloaded $modelId config: ${configFile.length()} bytes")
            println("Content: ${configFile.readText().take(500)}")
        }
    }
}
