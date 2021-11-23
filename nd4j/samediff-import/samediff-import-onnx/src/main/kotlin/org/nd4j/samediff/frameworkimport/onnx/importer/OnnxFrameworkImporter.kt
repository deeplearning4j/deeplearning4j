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
        loader.values.forEach { loadedGraphBuilder.addNode(it) }
    }

    val opDefs = loadedGraphBuilder.build()

    fun loadGraph(fileName: String): OnnxIRGraph {
        val loadGraph = Onnx.ModelProto.parseFrom(Files.readAllBytes(File(fileName).toPath()))
        return OnnxIRGraph(loadGraph.graph, registry)
    }

    override fun runImport(fileName: String, dynamicVariables: Map<String, INDArray>,suggestDynamicVariables: Boolean): SameDiff {
        val loadGraph = loadGraph(fileName)
        if(suggestDynamicVariables) {
            val newDynamicVariables  = suggestDynamicVariables(loadGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
            val dynamicVariablesConverted = convertToOnnxTensors(newDynamicVariables)
            return onnxImporter.importGraph(loadGraph,null,null, dynamicVariablesConverted,registry)
        } else {
            val dynamicVariablesConverted = convertToOnnxTensors(dynamicVariables)
            return onnxImporter.importGraph(loadGraph,null,null, dynamicVariablesConverted,registry)
        }

    }


    override fun suggestDynamicVariables(fileName: String): Map<String, INDArray> {
        val graph = loadGraph(fileName)
        return suggestDynamicVariables(graph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
    }

    override fun suggestDynamicVariables(irGraph: IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>): Map<String, INDArray> {
        val graph = irGraph as OnnxIRGraph
        val ret = HashMap<String,INDArray>()
        for(i in 0 until graph.inputList.size) {
            var inputShape = graph.shapeOfInput(graph.inputAt(i))
            val dType = graph.dataTypeForVariable(graph.inputAt(i))
            if(inputShape != null) {
                graph.shapeOfInput(graph.inputAt(i))!!.map { input -> if(input < 0) 1 else input }.toLongArray()
                ret[graph.inputAt(i)] = Nd4j.ones(dType.nd4jDataType(),*inputShape)
            }
        }

        return ret
    }
}