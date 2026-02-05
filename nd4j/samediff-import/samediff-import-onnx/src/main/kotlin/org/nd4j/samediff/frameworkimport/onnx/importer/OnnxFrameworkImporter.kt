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
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import java.io.File
import java.nio.file.Files

class OnnxFrameworkImporter: FrameworkImporter {

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
        // Use streaming parsing instead of reading entire file into memory
        val loadGraph = Files.newInputStream(File(fileName).toPath()).buffered(65536).use { stream ->
            Onnx.ModelProto.parseFrom(stream)
        }
        return OnnxIRGraph(loadGraph.graph, registry)
    }

    override fun runImport(
        fileName: String,
        dynamicVariables: Map<String, INDArray>,
        suggestDynamicVariables: Boolean,
        trackVariableChanges: Boolean
    ): SameDiff {
        val loadGraph = loadGraph(fileName)
        if(suggestDynamicVariables) {
            val newDynamicVariables  = suggestDynamicVariables(loadGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
            val dynamicVariablesConverted = convertToOnnxTensors(newDynamicVariables)
            // Add ONNX initializers to dynamicVariables so PreImportHooks can access constant tensors
            addInitializersToDynamicVariables(loadGraph, dynamicVariablesConverted)
            val ret =   onnxImporter.importGraph(loadGraph, null, null, dynamicVariablesConverted, registry, trackVariableChanges)
            ret.outputs().addAll(loadGraph.outputList)
            return ret
        } else {
            val dynamicVariablesConverted = convertToOnnxTensors(dynamicVariables)
            // Add ONNX initializers to dynamicVariables so PreImportHooks can access constant tensors
            addInitializersToDynamicVariables(loadGraph, dynamicVariablesConverted)
            val ret =  onnxImporter.importGraph(loadGraph, null, null, dynamicVariablesConverted, registry, trackVariableChanges)
            ret.outputs().addAll(loadGraph.outputList)
            return ret

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