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

import onnx.Onnx
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.VariableType
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.onnx.OnnxTensorUtils
import org.nd4j.samediff.frameworkimport.onnx.*
import org.slf4j.LoggerFactory
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStream

/**
 * Exporter for converting SameDiff models to ONNX format.
 *
 * This class provides the main functionality for exporting SameDiff computation graphs
 * to ONNX ModelProto format.
 *
 * Example usage:
 * ```kotlin
 * val sd = SameDiff.create()
 * // ... build graph ...
 * OnnxExporter.export(sd, File("model.onnx"))
 * ```
 *
 * @author Adam Gibson
 */
object OnnxExporter {

    private val logger = LoggerFactory.getLogger(OnnxExporter::class.java)

    /**
     * Export a SameDiff graph to an ONNX file.
     *
     * @param sameDiff The SameDiff instance to export
     * @param outputFile The output file
     * @param config Export configuration
     */
    @JvmStatic
    @JvmOverloads
    fun export(
        sameDiff: SameDiff,
        outputFile: File,
        config: OnnxExportConfig = OnnxExportConfig.DEFAULT
    ) {
        FileOutputStream(outputFile).use { fos ->
            export(sameDiff, fos, config)
        }

        // Export training state if requested
        if (config.includeTrainingState && sameDiff.trainingConfig != null) {
            val trainingFile = File(outputFile.absolutePath + ".training")
            TrainingStateExporter.export(sameDiff, trainingFile)
        }
    }

    /**
     * Export a SameDiff graph to an output stream.
     *
     * @param sameDiff The SameDiff instance to export
     * @param outputStream The output stream
     * @param config Export configuration
     */
    @JvmStatic
    @JvmOverloads
    fun export(
        sameDiff: SameDiff,
        outputStream: OutputStream,
        config: OnnxExportConfig = OnnxExportConfig.DEFAULT
    ) {
        val model = toModelProto(sameDiff, config)
        model.writeTo(outputStream)
    }

    /**
     * Convert a SameDiff graph to ONNX bytes.
     *
     * @param sameDiff The SameDiff instance to convert
     * @param config Export configuration
     * @return The ONNX model as bytes
     */
    @JvmStatic
    @JvmOverloads
    fun toBytes(
        sameDiff: SameDiff,
        config: OnnxExportConfig = OnnxExportConfig.DEFAULT
    ): ByteArray {
        val baos = ByteArrayOutputStream()
        export(sameDiff, baos, config)
        return baos.toByteArray()
    }

    /**
     * Convert a SameDiff graph to ONNX ModelProto.
     *
     * @param sameDiff The SameDiff instance to convert
     * @param config Export configuration
     * @return The ONNX ModelProto
     */
    @JvmStatic
    @JvmOverloads
    fun toModelProto(
        sameDiff: SameDiff,
        config: OnnxExportConfig = OnnxExportConfig.DEFAULT
    ): Onnx.ModelProto {
        val graphProto = buildGraph(sameDiff, config)

        return ModelProto {
            irVersion = config.irVersion
            OpSetImport(OperatorSetIdProto {
                version = config.opsetVersion
            })
            producerName = config.producerName
            producerVersion = config.producerVersion
            if (config.domain.isNotEmpty()) {
                domain = config.domain
            }
            modelVersion = config.modelVersion
            if (config.docString != null) {
                docString = config.docString
            }
            graph = graphProto
        }
    }

    /**
     * Build the ONNX GraphProto from a SameDiff instance.
     */
    private fun buildGraph(sameDiff: SameDiff, config: OnnxExportConfig): Onnx.GraphProto {
        val graphBuilder = Onnx.GraphProto.newBuilder()
        graphBuilder.name = config.graphName

        // Get topologically sorted operations
        val sortedOps = getTopologicallySortedOps(sameDiff)

        // Build graph inputs (placeholders)
        buildGraphInputs(sameDiff, graphBuilder)

        // Build initializers (constants and variables with values)
        buildInitializers(sameDiff, graphBuilder)

        // Build nodes (operations)
        buildNodes(sameDiff, sortedOps, graphBuilder, config)

        // Build graph outputs
        buildGraphOutputs(sameDiff, graphBuilder)

        return graphBuilder.build()
    }

    /**
     * Get operations in topological order.
     */
    private fun getTopologicallySortedOps(sameDiff: SameDiff): List<SameDiffOp> {
        val ops = sameDiff.ops
        val visited = mutableSetOf<String>()
        val result = mutableListOf<SameDiffOp>()

        fun visit(opName: String) {
            if (opName in visited) return
            visited.add(opName)

            val op = ops[opName] ?: return

            // Visit input ops first
            op.inputsToOp?.forEach { inputVarName ->
                val inputVar = sameDiff.variables[inputVarName]
                inputVar?.inputsForOp?.forEach { inputOpName ->
                    if (!inputOpName.isNullOrEmpty()) {
                        visit(inputOpName)
                    }
                }
            }

            result.add(op)
        }

        ops.keys.forEach { visit(it) }
        return result
    }

    /**
     * Build graph inputs from placeholders.
     */
    private fun buildGraphInputs(sameDiff: SameDiff, graphBuilder: Onnx.GraphProto.Builder) {
        sameDiff.variables.values
            .filter { it.variable.variableType == VariableType.PLACEHOLDER }
            .forEach { varMeta ->
                val variable = varMeta.variable
                val shape = variable.shape ?: longArrayOf()
                val dataType = variable.dataType() ?: DataType.FLOAT

                val valueInfo = createValueInfoProto(variable.name(), shape, dataType)
                graphBuilder.addInput(valueInfo)

                logger.debug("Added input: {} with shape {} and type {}", variable.name(), shape.contentToString(), dataType)
            }
    }

    /**
     * Build initializers from constants and variables with values.
     */
    private fun buildInitializers(sameDiff: SameDiff, graphBuilder: Onnx.GraphProto.Builder) {
        sameDiff.variables.values
            .filter {
                it.variable.variableType == VariableType.CONSTANT ||
                        it.variable.variableType == VariableType.VARIABLE
            }
            .forEach { varMeta ->
                val variable = varMeta.variable
                val arr = variable.arr

                if (arr != null) {
                    val tensorProto = OnnxTensorUtils.toTensorProto(arr, variable.name())
                    graphBuilder.addInitializer(tensorProto)

                    // Also add to inputs for compatibility with some runtimes
                    val valueInfo = createValueInfoProto(
                        variable.name(),
                        arr.shape(),
                        arr.dataType()
                    )
                    graphBuilder.addInput(valueInfo)

                    logger.debug("Added initializer: {} with shape {}", variable.name(), arr.shape().contentToString())
                }
            }
    }

    /**
     * Build nodes from operations.
     */
    private fun buildNodes(
        sameDiff: SameDiff,
        sortedOps: List<SameDiffOp>,
        graphBuilder: Onnx.GraphProto.Builder,
        config: OnnxExportConfig
    ) {
        for (op in sortedOps) {
            val opName = op.op.opName()

            // Check for post-export hook
            val hook = PostExportHookRegistry.findHook(opName)
            if (hook != null) {
                val nodes = hook.export(sameDiff, op, graphBuilder, config)
                nodes.forEach { graphBuilder.addNode(it) }

                // Add any additional initializers from the hook
                hook.getInitializers(sameDiff, op).forEach { graphBuilder.addInitializer(it) }
                continue
            }

            // Get ONNX op name from mapper
            val onnxOpName = SameDiffToOnnxOpMapper.getOnnxOpName(opName)
            if (onnxOpName == null) {
                logger.warn("No ONNX mapping found for operation: {}. Skipping.", opName)
                continue
            }

            // Build the node
            val nodeProto = buildNode(sameDiff, op, onnxOpName, config)
            graphBuilder.addNode(nodeProto)

            logger.debug("Added node: {} (ONNX: {})", op.name, onnxOpName)
        }
    }

    /**
     * Build a single ONNX node from a SameDiff operation.
     */
    private fun buildNode(
        sameDiff: SameDiff,
        op: SameDiffOp,
        onnxOpName: String,
        config: OnnxExportConfig
    ): Onnx.NodeProto {
        return NodeProto {
            name = op.name
            opType = onnxOpName

            // Add inputs
            op.inputsToOp?.forEach { inputName ->
                addInput(inputName)
            }

            // Add outputs
            op.outputsOfOp?.forEach { outputName ->
                addOutput(outputName)
            }

            // Add attributes
            val attributes = SameDiffToOnnxOpMapper.createAttributes(sameDiff, op.op, op.op.opName())
            attributes.forEach { attr ->
                addAttribute(attr)
            }
        }
    }

    /**
     * Build graph outputs.
     */
    private fun buildGraphOutputs(sameDiff: SameDiff, graphBuilder: Onnx.GraphProto.Builder) {
        // Get explicitly marked outputs
        val outputVariables = sameDiff.outputs()

        if (outputVariables.isNullOrEmpty()) {
            // If no outputs are explicitly set, use all variables that are not inputs to other ops
            val allOutputVars = sameDiff.variables.values
                .filter { varMeta ->
                    val isOutput = varMeta.outputOfOp != null &&
                            (varMeta.inputsForOp.isNullOrEmpty() ||
                                    varMeta.inputsForOp?.all { it.isNullOrEmpty() } == true)
                    isOutput
                }
                .map { it.variable }

            allOutputVars.forEach { addOutput(it, graphBuilder) }
        } else {
            outputVariables.forEach { varName ->
                val variable = sameDiff.getVariable(varName)
                if (variable != null) {
                    addOutput(variable, graphBuilder)
                }
            }
        }
    }

    /**
     * Add an output variable to the graph.
     */
    private fun addOutput(variable: SDVariable, graphBuilder: Onnx.GraphProto.Builder) {
        val shape = variable.shape ?: longArrayOf()
        val dataType = variable.dataType() ?: DataType.FLOAT

        val valueInfo = createValueInfoProto(variable.name(), shape, dataType)
        graphBuilder.addOutput(valueInfo)

        logger.debug("Added output: {} with shape {} and type {}", variable.name(), shape.contentToString(), dataType)
    }

    /**
     * Create a ValueInfoProto for a variable.
     */
    private fun createValueInfoProto(
        name: String,
        shape: LongArray,
        dataType: DataType
    ): Onnx.ValueInfoProto {
        return ValueInfoProto {
            this.name = name
            type = TypeProto {
                tensorType = TensorTypeProto {
                    elemType = convertToOnnxDataType(dataType).ordinal
                    if (shape.isNotEmpty()) {
                        this.shape = OnnxShapeProto {
                            OnnxShape(shape.toList())
                        }
                    }
                }
            }
        }
    }

    /**
     * Convert ND4J DataType to ONNX DataType.
     */
    private fun convertToOnnxDataType(dataType: DataType): Onnx.TensorProto.DataType {
        return when (dataType) {
            DataType.FLOAT -> Onnx.TensorProto.DataType.FLOAT
            DataType.DOUBLE -> Onnx.TensorProto.DataType.DOUBLE
            DataType.INT -> Onnx.TensorProto.DataType.INT32
            DataType.LONG -> Onnx.TensorProto.DataType.INT64
            DataType.BOOL -> Onnx.TensorProto.DataType.BOOL
            DataType.INT8, DataType.BYTE -> Onnx.TensorProto.DataType.INT8
            DataType.INT16, DataType.SHORT -> Onnx.TensorProto.DataType.INT16
            DataType.UINT8, DataType.UBYTE -> Onnx.TensorProto.DataType.UINT8
            DataType.UINT16 -> Onnx.TensorProto.DataType.UINT16
            DataType.UINT32 -> Onnx.TensorProto.DataType.UINT32
            DataType.UINT64 -> Onnx.TensorProto.DataType.UINT64
            DataType.HALF -> Onnx.TensorProto.DataType.FLOAT16
            DataType.BFLOAT16 -> Onnx.TensorProto.DataType.BFLOAT16
            DataType.UTF8 -> Onnx.TensorProto.DataType.STRING
            else -> throw IllegalArgumentException("Unsupported data type for ONNX export: $dataType")
        }
    }

    /**
     * Validate that all operations in the graph can be exported.
     *
     * @param sameDiff The SameDiff instance to validate
     * @return List of unsupported operation names
     */
    @JvmStatic
    fun validateForExport(sameDiff: SameDiff): List<String> {
        val unsupported = mutableListOf<String>()

        sameDiff.ops.values.forEach { op ->
            val opName = op.op.opName()
            if (!SameDiffToOnnxOpMapper.isSupported(opName)) {
                unsupported.add(opName)
            }
        }

        return unsupported
    }

    /**
     * Check if a SameDiff graph can be fully exported to ONNX.
     *
     * @param sameDiff The SameDiff instance to check
     * @return true if all operations are supported
     */
    @JvmStatic
    fun canExport(sameDiff: SameDiff): Boolean {
        return validateForExport(sameDiff).isEmpty()
    }
}
