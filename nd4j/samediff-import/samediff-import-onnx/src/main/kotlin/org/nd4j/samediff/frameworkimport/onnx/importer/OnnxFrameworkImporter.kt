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

import onnx.Onnx
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.InferenceSession
import org.nd4j.common.config.ND4JSystemProperties
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.FrameworkImporter
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.onnx.OnnxImportGraph
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxTensors
import org.nd4j.samediff.frameworkimport.onnx.definitions.OnnxOpDeclarations
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.opdefs.OnnxOpDescriptorLoader
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.shade.protobuf.ByteString
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.slf4j.LoggerFactory
import java.io.File
import java.io.RandomAccessFile
import java.nio.file.Files

class OnnxFrameworkImporter: FrameworkImporter {

    companion object {
        private val log = LoggerFactory.getLogger(OnnxFrameworkImporter::class.java)
    }

    val onnxImporter = OnnxImportGraph()
    val loader = OpDescriptorLoaderHolder.listForFramework<Onnx.NodeProto>("onnx")
    val onnxOpDescriptorLoader = OnnxOpDescriptorLoader()
    val registry = onnxOpDescriptorLoader.createOpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.AttributeProto,Onnx.AttributeProto,Onnx.TensorProto.DataType>()
    val loadedGraphBuilder = Onnx.GraphProto.newBuilder()

    init {
        OnnxOpDeclarations.init()
        loader.values.forEach { loadedGraphBuilder.addNode(it) }
    }

    val opDefs = loadedGraphBuilder.build()

    fun loadGraph(fileName: String): OnnxIRGraph {
        val modelFile = File(fileName)
        val modelDir = modelFile.parentFile
        // Use streaming parsing instead of reading entire file into memory
        val loadGraph = Files.newInputStream(modelFile.toPath()).buffered(65536).use { stream ->
            Onnx.ModelProto.parseFrom(stream)
        }
        // Resolve external data tensors — ONNX models can split weights into separate files
        val resolvedModel = resolveExternalData(loadGraph, modelDir)
        return OnnxIRGraph(resolvedModel.graph, registry)
    }

    /**
     * Resolves ONNX external data tensors by reading weight data from sibling files.
     *
     * ONNX models can store tensor data externally (data_location == EXTERNAL) with
     * key-value pairs in external_data describing the file location, byte offset, and length.
     * This is common for large models (>2GB) like bge-m3 where model.onnx contains only
     * the graph structure and model.onnx_data contains the actual weights.
     *
     * After resolution, all tensors have their raw_data inlined and data_location set to DEFAULT
     * so the rest of the import pipeline processes them normally.
     */
    private fun resolveExternalData(model: Onnx.ModelProto, modelDir: File?): Onnx.ModelProto {
        val graph = model.graph
        val initializers = graph.initializerList
        val hasExternal = initializers.any { it.dataLocation == Onnx.TensorProto.DataLocation.EXTERNAL }
        if (!hasExternal) {
            return model
        }

        log.info("ONNX model has external data tensors, resolving from directory: {}", modelDir?.absolutePath)
        val resolvedInitializers = mutableListOf<Onnx.TensorProto>()
        var resolvedCount = 0

        for (tensor in initializers) {
            if (tensor.dataLocation == Onnx.TensorProto.DataLocation.EXTERNAL) {
                resolvedInitializers.add(resolveExternalTensor(tensor, modelDir))
                resolvedCount++
            } else {
                resolvedInitializers.add(tensor)
            }
        }

        log.info("Resolved {} external data tensors", resolvedCount)

        // Rebuild graph with resolved initializers
        val newGraph = Onnx.GraphProto.newBuilder(graph)
            .clearInitializer()
            .addAllInitializer(resolvedInitializers)
            .build()

        return Onnx.ModelProto.newBuilder(model)
            .setGraph(newGraph)
            .build()
    }

    /**
     * Resolves a single external data tensor by reading its bytes from the referenced file.
     *
     * Per the ONNX spec, external_data contains key-value pairs:
     * - "location" (required): POSIX path relative to model directory
     * - "offset" (optional): byte offset into the file (default 0)
     * - "length" (optional): number of bytes to read (default: rest of file from offset)
     */
    private fun resolveExternalTensor(tensor: Onnx.TensorProto, modelDir: File?): Onnx.TensorProto {
        val extData = mutableMapOf<String, String>()
        for (entry in tensor.externalDataList) {
            extData[entry.key] = entry.value
        }

        val location = extData["location"]
            ?: throw IllegalStateException("External data tensor '${tensor.name}' missing required 'location' key")
        val offset = extData["offset"]?.toLongOrNull() ?: 0L
        val length = extData["length"]?.toLongOrNull()

        val dataFile = if (modelDir != null) File(modelDir, location) else File(location)
        if (!dataFile.exists()) {
            throw IllegalStateException(
                "External data file not found: ${dataFile.absolutePath} " +
                "(referenced by tensor '${tensor.name}')"
            )
        }

        log.debug("Reading external data for tensor '{}': file={}, offset={}, length={}",
            tensor.name, dataFile.name, offset, length ?: "rest-of-file")

        val bytes = RandomAccessFile(dataFile, "r").use { raf ->
            raf.seek(offset)
            val readLength = length ?: (raf.length() - offset)
            val buf = ByteArray(readLength.toInt())
            raf.readFully(buf)
            buf
        }

        // Rebuild the tensor with inlined raw data
        return Onnx.TensorProto.newBuilder(tensor)
            .setRawData(ByteString.copyFrom(bytes))
            .clearExternalData()
            .setDataLocation(Onnx.TensorProto.DataLocation.DEFAULT)
            .build()
    }

    override fun runImport(
        fileName: String,
        dynamicVariables: Map<String, INDArray>,
        suggestDynamicVariables: Boolean,
        trackVariableChanges: Boolean
    ): SameDiff {
        // Disable DSP and CUDA graphs during model import. Importing loads model constants
        // to GPU — DSP compilation and CUDA graph capture add memory pressure that causes OOM.
        val dspWasEnabled = InferenceSession.isDynamicShapePlanEnabled()
        val prevCudaGraphs = System.getProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED)
        InferenceSession.setDynamicShapePlanEnabled(false)
        System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, "false")
        try {
            val loadGraph = loadGraph(fileName)
            if(suggestDynamicVariables) {
                // CONTRACT WARNING: suggested inputs are dummy MINIMAL-shape tensors and
                // import-time eager evaluation FOLDS shape math over them into constants.
                // For dynamic-sequence models (autoregressive decoders, variable-length
                // inputs) this silently bakes 1-token geometry into the graph: KV present
                // outputs collapse to seq-dim 1 and logits are garbage at non-final
                // positions. Pass suggestDynamicVariables=false for such models.
                log.warn("suggestDynamicVariables=true: import will CONSTANT-FOLD shape math " +
                        "over minimal dummy input shapes. Dynamic-sequence models (decoders, " +
                        "variable-length inputs) MUST use false — baked shapes silently corrupt " +
                        "multi-token execution.")
                val newDynamicVariables  = suggestDynamicVariables(loadGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
                val dynamicVariablesConverted = convertToOnnxTensors(newDynamicVariables)
                // Add ONNX initializers to dynamicVariables so PreImportHooks can access constant tensors
                addInitializersToDynamicVariables(loadGraph, dynamicVariablesConverted)
                val ret = onnxImporter.importGraph(loadGraph, null, null, dynamicVariablesConverted, registry, trackVariableChanges)
                ret.outputs().addAll(loadGraph.outputList)
                return ret
            } else {
                val dynamicVariablesConverted = convertToOnnxTensors(dynamicVariables)
                // Add ONNX initializers to dynamicVariables so PreImportHooks can access constant tensors
                addInitializersToDynamicVariables(loadGraph, dynamicVariablesConverted)
                val ret = onnxImporter.importGraph(loadGraph, null, null, dynamicVariablesConverted, registry, trackVariableChanges)
                ret.outputs().addAll(loadGraph.outputList)
                return ret
            }
        } finally {
            InferenceSession.setDynamicShapePlanEnabled(dspWasEnabled)
            if (prevCudaGraphs != null) {
                System.setProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED, prevCudaGraphs)
            } else {
                System.clearProperty(ND4JSystemProperties.DSP_CUDA_GRAPHS_ENABLED)
            }
        }
    }

    /**
     * Add all ONNX initializers and Constant node outputs to the dynamicVariables map.
     * This allows PreImportHooks to access constant tensor values (like axes for Unsqueeze)
     * which are stored as ONNX initializers or Constant nodes.
     */
    private fun addInitializersToDynamicVariables(
        graph: OnnxIRGraph,
        dynamicVariables: MutableMap<String, Onnx.TensorProto>
    ) {
        val graphDef = graph.graphDef()

        // Add all initializers (direct tensor constants)
        for (initializer in graphDef.initializerList) {
            val name = initializer.name
            // Only add if not already present (don't override user-provided values)
            if (!dynamicVariables.containsKey(name)) {
                dynamicVariables[name] = initializer
            }
        }

        // Add all Constant node outputs (opset 13+ uses Constant nodes for some constants)
        for (node in graphDef.nodeList) {
            if (node.opType == "Constant") {
                // Get the output name (what other nodes reference this constant by)
                val outputNames = node.outputList

                // Extract the tensor from the Constant node's attributes
                val tensor = extractTensorFromConstantNode(node)
                if (tensor != null) {
                    for (outputName in outputNames) {
                        val cleanName = outputName.replace(":0", "")
                        if (!dynamicVariables.containsKey(cleanName)) {
                            dynamicVariables[cleanName] = tensor
                        }
                        if (!dynamicVariables.containsKey(outputName)) {
                            dynamicVariables[outputName] = tensor
                        }
                    }
                }
            }
        }
    }

    /**
     * Extract the tensor value from a Constant node's attributes.
     */
    private fun extractTensorFromConstantNode(node: Onnx.NodeProto): Onnx.TensorProto? {
        for (attr in node.attributeList) {
            when (attr.name) {
                "value" -> return attr.t
                "value_int" -> {
                    // Build a scalar INT64 tensor
                    return Onnx.TensorProto.newBuilder()
                        .setDataType(Onnx.TensorProto.DataType.INT64_VALUE)
                        .addInt64Data(attr.i)
                        .build()
                }
                "value_ints" -> {
                    // Build a 1D INT64 tensor
                    return Onnx.TensorProto.newBuilder()
                        .setDataType(Onnx.TensorProto.DataType.INT64_VALUE)
                        .addAllInt64Data(attr.intsList)
                        .addDims(attr.intsCount.toLong())
                        .build()
                }
                "value_float" -> {
                    // Build a scalar FLOAT tensor
                    return Onnx.TensorProto.newBuilder()
                        .setDataType(Onnx.TensorProto.DataType.FLOAT_VALUE)
                        .addFloatData(attr.f)
                        .build()
                }
                "value_floats" -> {
                    // Build a 1D FLOAT tensor
                    return Onnx.TensorProto.newBuilder()
                        .setDataType(Onnx.TensorProto.DataType.FLOAT_VALUE)
                        .addAllFloatData(attr.floatsList)
                        .addDims(attr.floatsCount.toLong())
                        .build()
                }
            }
        }
        // Fallback: try first attribute's tensor
        if (node.attributeCount > 0 && node.getAttribute(0).hasT()) {
            return node.getAttribute(0).t
        }
        return null
    }


    override fun suggestDynamicVariables(fileName: String): Map<String, INDArray> {
        val graph = loadGraph(fileName)
        return suggestDynamicVariables(graph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
    }

    override fun suggestDynamicVariables(irGraph: IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>): Map<String, INDArray> {
        val graph = irGraph as OnnxIRGraph
        val ret = HashMap<String,INDArray>()
        for(i in 0 until graph.inputList.size) {
            if(irGraph.shapeOfInput(graph.inputAt(i)) == null) {
                throw IllegalArgumentException("Unable to suggest dynamic variables. No shape found for input $i named ${graph.inputAt(i)}")
            }
        }


        for(i in 0 until graph.inputList.size) {
            var inputShape = graph.shapeOfInput(graph.inputAt(i))
            val dType = graph.dataTypeForVariable(graph.inputAt(i))
            if(inputShape != null) {
                inputShape = graph.shapeOfInput(graph.inputAt(i))!!.map { input -> if(input < 0) 1 else input }.toLongArray()
                ret[graph.inputAt(i)] = Nd4j.ones(dType.nd4jDataType(),*inputShape)
            } else {
                ret[graph.inputAt(i)] = Nd4j.ones(dType.nd4jDataType())

            }

        }

        return ret
    }
}