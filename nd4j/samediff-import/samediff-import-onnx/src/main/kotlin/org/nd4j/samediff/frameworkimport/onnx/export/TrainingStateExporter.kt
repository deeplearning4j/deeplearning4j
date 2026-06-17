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
package org.nd4j.samediff.frameworkimport.onnx.export

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.TrainingConfig
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.linalg.learning.config.*
import org.nd4j.shade.jackson.databind.ObjectMapper
import org.nd4j.shade.jackson.databind.SerializationFeature
import org.slf4j.LoggerFactory
import java.io.*
import java.util.zip.ZipEntry
import java.util.zip.ZipInputStream
import java.util.zip.ZipOutputStream

/**
 * Exporter for SameDiff training state to accompany ONNX model exports.
 *
 * Since ONNX doesn't natively support training state, this exporter creates
 * a companion file that stores:
 * - Optimizer configuration (type, learning rate, momentum, etc.)
 * - Training metadata
 *
 * The companion file uses a ZIP format containing:
 * - metadata.json: Optimizer configuration
 *
 * Note: Full updater state export is a future enhancement.
 *
 * @author Adam Gibson
 */
object TrainingStateExporter {

    private val logger = LoggerFactory.getLogger(TrainingStateExporter::class.java)
    private val objectMapper = ObjectMapper().apply {
        enable(SerializationFeature.INDENT_OUTPUT)
    }

    /**
     * Export training state to a companion file.
     *
     * @param sameDiff The SameDiff instance with training configuration
     * @param outputFile The output file (typically model.onnx.training)
     */
    @JvmStatic
    fun export(sameDiff: SameDiff, outputFile: File) {
        val trainingConfig = sameDiff.trainingConfig
            ?: throw IllegalStateException("SameDiff has no training configuration")

        ZipOutputStream(FileOutputStream(outputFile)).use { zos ->
            // Write metadata
            val metadata = buildMetadata(sameDiff, trainingConfig)
            writeJsonEntry(zos, "metadata.json", metadata)

            logger.info("Exported training state to: {}", outputFile.absolutePath)
        }
    }

    /**
     * Import training state from a companion file.
     *
     * @param sameDiff The SameDiff instance to restore training state to
     * @param inputFile The training state file
     */
    @JvmStatic
    fun import(sameDiff: SameDiff, inputFile: File) {
        if (!inputFile.exists()) {
            throw FileNotFoundException("Training state file not found: ${inputFile.absolutePath}")
        }

        ZipInputStream(FileInputStream(inputFile)).use { zis ->
            var metadata: TrainingStateMetadata? = null

            var entry: ZipEntry? = zis.nextEntry
            while (entry != null) {
                when {
                    entry.name == "metadata.json" -> {
                        val json = zis.readBytes().decodeToString()
                        metadata = objectMapper.readValue(json, TrainingStateMetadata::class.java)
                    }
                }
                entry = zis.nextEntry
            }

            // Restore training config if not already set
            if (metadata != null && sameDiff.trainingConfig == null) {
                val restoredConfig = restoreTrainingConfig(metadata)
                sameDiff.trainingConfig = restoredConfig
            }

            logger.info("Imported training state from: {}", inputFile.absolutePath)
        }
    }

    /**
     * Build metadata from training configuration.
     */
    private fun buildMetadata(sameDiff: SameDiff, config: TrainingConfig): TrainingStateMetadata {
        val updaterConfig = config.updater

        val updaterParams = mutableMapOf<String, Any>()
        when (updaterConfig) {
            is Adam -> {
                updaterParams["type"] = "Adam"
                updaterParams["learningRate"] = updaterConfig.learningRate
                updaterParams["beta1"] = updaterConfig.beta1
                updaterParams["beta2"] = updaterConfig.beta2
                updaterParams["epsilon"] = updaterConfig.epsilon
            }
            is Sgd -> {
                updaterParams["type"] = "Sgd"
                updaterParams["learningRate"] = updaterConfig.learningRate
            }
            is Nesterovs -> {
                updaterParams["type"] = "Nesterovs"
                updaterParams["learningRate"] = updaterConfig.learningRate
                updaterParams["momentum"] = updaterConfig.momentum
            }
            is RmsProp -> {
                updaterParams["type"] = "RmsProp"
                updaterParams["learningRate"] = updaterConfig.learningRate
                updaterParams["rmsDecay"] = updaterConfig.rmsDecay
                updaterParams["epsilon"] = updaterConfig.epsilon
            }
            is AdaGrad -> {
                updaterParams["type"] = "AdaGrad"
                updaterParams["learningRate"] = updaterConfig.learningRate
                updaterParams["epsilon"] = updaterConfig.epsilon
            }
            is AdaDelta -> {
                updaterParams["type"] = "AdaDelta"
                updaterParams["rho"] = updaterConfig.rho
                updaterParams["epsilon"] = updaterConfig.epsilon
            }
            else -> {
                updaterParams["type"] = updaterConfig?.javaClass?.simpleName ?: "Unknown"
            }
        }

        // Get label mapping from config (used as loss variable targets)
        val lossVars = config.dataSetLabelMapping?.toList() ?: emptyList()

        // Get trainable variables
        val trainableVars = sameDiff.variables.values
            .filter { it.variable.variableType == VariableType.VARIABLE }
            .map { it.variable.name() }

        return TrainingStateMetadata(
            optimizerType = updaterParams["type"] as String,
            optimizerParams = updaterParams,
            iterationCount = 0L,  // Placeholder - would need to track this
            epochCount = 0L,      // Placeholder - would need to track this
            lossVariables = lossVars,
            trainableVariables = trainableVars
        )
    }

    /**
     * Restore TrainingConfig from metadata.
     */
    private fun restoreTrainingConfig(metadata: TrainingStateMetadata): TrainingConfig {
        val updater = when (metadata.optimizerType) {
            "Adam" -> Adam(
                metadata.optimizerParams["learningRate"] as? Double ?: 0.001,
                metadata.optimizerParams["beta1"] as? Double ?: 0.9,
                metadata.optimizerParams["beta2"] as? Double ?: 0.999,
                metadata.optimizerParams["epsilon"] as? Double ?: 1e-8
            )
            "Sgd" -> Sgd(metadata.optimizerParams["learningRate"] as? Double ?: 0.01)
            "Nesterovs" -> Nesterovs(
                metadata.optimizerParams["learningRate"] as? Double ?: 0.01,
                metadata.optimizerParams["momentum"] as? Double ?: 0.9
            )
            "RmsProp" -> RmsProp(
                metadata.optimizerParams["learningRate"] as? Double ?: 0.001,
                metadata.optimizerParams["rmsDecay"] as? Double ?: 0.95,
                metadata.optimizerParams["epsilon"] as? Double ?: 1e-8
            )
            "AdaGrad" -> AdaGrad(
                metadata.optimizerParams["learningRate"] as? Double ?: 0.01,
                metadata.optimizerParams["epsilon"] as? Double ?: 1e-8
            )
            "AdaDelta" -> AdaDelta(
                metadata.optimizerParams["rho"] as? Double ?: 0.95,
                metadata.optimizerParams["epsilon"] as? Double ?: 1e-6
            )
            else -> throw IllegalArgumentException("Unknown optimizer type: ${metadata.optimizerType}")
        }

        val builder = TrainingConfig.builder()
            .updater(updater)

        // Add loss variables if available
        if (metadata.lossVariables.isNotEmpty()) {
            builder.dataSetFeatureMapping(*metadata.lossVariables.toTypedArray())
        }

        return builder.build()
    }

    /**
     * Write a JSON entry to the ZIP file.
     */
    private fun writeJsonEntry(zos: ZipOutputStream, entryName: String, data: Any) {
        val entry = ZipEntry(entryName)
        zos.putNextEntry(entry)
        val json = objectMapper.writeValueAsString(data)
        zos.write(json.toByteArray(Charsets.UTF_8))
        zos.closeEntry()
    }
}

/**
 * Data class for training state metadata.
 */
data class TrainingStateMetadata(
    val optimizerType: String = "",
    val optimizerParams: Map<String, Any> = emptyMap(),
    val iterationCount: Long = 0L,
    val epochCount: Long = 0L,
    val lossVariables: List<String> = emptyList(),
    val trainableVariables: List<String> = emptyList(),
    val version: String = "1.0"
)
