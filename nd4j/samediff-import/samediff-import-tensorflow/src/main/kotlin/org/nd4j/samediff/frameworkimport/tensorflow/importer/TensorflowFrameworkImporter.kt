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
package org.nd4j.samediff.frameworkimport.tensorflow.importer

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.imports.graphmapper.tf.TFGraphMapper
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.FrameworkImporter
import org.nd4j.samediff.frameworkimport.ir.IRGraph
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.tensorflow.TensorflowImportGraph
import org.nd4j.samediff.frameworkimport.tensorflow.convertNDArrayToTensorflowTensor
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.gruCell
import org.nd4j.samediff.frameworkimport.tensorflow.definitions.tensorflowOpRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.ir.TensorflowIRGraph
import org.nd4j.samediff.frameworkimport.tensorflow.opdefs.TensorflowOpDescriptorLoader
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.tensorflow.framework.*
import java.io.File
import java.nio.file.Files

class TensorflowFrameworkImporter: FrameworkImporter {

    val tfImporter = TensorflowImportGraph()
    val loader = OpDescriptorLoaderHolder.listForFramework<OpDef>("tensorflow")
    val tfOpDescriptorLoader = TensorflowOpDescriptorLoader()
    val opDefListBuilder = OpList.newBuilder()
    val opDefList = opDefListBuilder.build()
    val registry =
        tfOpDescriptorLoader.createOpMappingRegistry<GraphDef, NodeDef, OpDef, TensorProto, OpDef.AttrDef, AttrValue, DataType>()

    init {
        loader.values.forEach { opDef -> opDefListBuilder.addOp(opDef) }

    }

    fun importFromGraph(graphDef: GraphDef, dynamicVariables: Map<String, INDArray>): SameDiff {
        val dynamicVariablesConverted = HashMap<String, TensorProto>()
        dynamicVariables.forEach { (name, array) ->
            dynamicVariablesConverted[name] = convertNDArrayToTensorflowTensor(array)
        }
        val irGraph = TensorflowIRGraph(graphDef, opDefList, registry)
        return tfImporter.importGraph(irGraph, null, null, dynamicVariablesConverted, tensorflowOpRegistry)

    }

    override fun runImport(fileName: String, dynamicVariables: Map<String, INDArray>,suggestDynamicVariables: Boolean): SameDiff {
        val loadGraph = GraphDef.parseFrom(Files.readAllBytes(File(fileName).toPath()))
        val irGraph = TensorflowIRGraph(loadGraph,opDefList,registry)
        return if(suggestDynamicVariables) {
            val newDynamicVariables  = suggestDynamicVariables(irGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)
            importFromGraph(graphDef = loadGraph, dynamicVariables = newDynamicVariables)
        } else {
            importFromGraph(graphDef = loadGraph, dynamicVariables = dynamicVariables)

        }
    }

    override fun suggestDynamicVariables(fileName: String): Map<String, INDArray> {
        val loadGraph = GraphDef.parseFrom(Files.readAllBytes(File(fileName).toPath()))
        val irGraph = TensorflowIRGraph(loadGraph,opDefList,registry)
        return suggestDynamicVariables(irGraph as IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>)

    }

    override fun suggestDynamicVariables(irGraph: IRGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>): Map<String, INDArray> {
        val irGraph = irGraph as TensorflowIRGraph
        val ret = HashMap<String,INDArray>()
        for(i in 0 until irGraph.inputs.size) {
            val shape = irGraph.shapeOfInput(irGraph.inputs[i])
            if(shape != null) {
                val dtype = irGraph.dataTypeForVariable(irGraph.inputAt(i))
                ret[irGraph.inputAt(i)] = Nd4j.ones(dtype.nd4jDataType(),*shape)
            }
        }
        return ret
    }
}