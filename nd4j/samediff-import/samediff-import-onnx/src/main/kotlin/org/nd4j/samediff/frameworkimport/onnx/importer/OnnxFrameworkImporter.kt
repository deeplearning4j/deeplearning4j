/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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
import org.nd4j.samediff.frameworkimport.FrameworkImporter
import org.nd4j.samediff.frameworkimport.onnx.OnnxImportGraph
import org.nd4j.samediff.frameworkimport.onnx.convertToOnnxTensor
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRGraph
import org.nd4j.samediff.frameworkimport.onnx.opdefs.OnnxOpDescriptorLoader
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
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

    override fun runImport(fileName: String, dynamicVariables: Map<String, INDArray>): SameDiff {
        val loadGraph = Onnx.ModelProto.parseFrom(Files.readAllBytes(File(fileName).toPath()))
        val irGraph = OnnxIRGraph(loadGraph.graph,registry)
        val dynamicVariablesConverted = HashMap<String,Onnx.TensorProto>()
        dynamicVariables.forEach { name, array ->
            dynamicVariablesConverted[name] = convertToOnnxTensor(array,name)
        }
        return onnxImporter.importGraph(irGraph,null,null, dynamicVariablesConverted,registry)
    }
}