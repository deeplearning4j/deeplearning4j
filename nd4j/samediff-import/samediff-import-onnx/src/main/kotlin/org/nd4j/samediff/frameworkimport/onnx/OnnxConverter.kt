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
package org.nd4j.samediff.frameworkimport.onnx

import lombok.SneakyThrows
import onnx.Onnx
import org.apache.commons.io.IOUtils
import org.bytedeco.javacpp.BytePointer
import org.bytedeco.onnx.DefaultVersionConverter
import org.bytedeco.onnx.ModelProto
import org.bytedeco.onnx.OpSetID
import java.io.BufferedInputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer

class OnnxConverter {

    @SneakyThrows
    fun convertModel(inputModel: File,outputModelFilePath: File)  {
        val converter = DefaultVersionConverter()
        val bytes = ByteBuffer.wrap(IOUtils.toByteArray(BufferedInputStream(FileInputStream(inputModel))))
        val bytePointer = BytePointer(bytes)
        val proto = ModelProto(bytes.capacity().toLong())
        //val operatorSet = Onnx.OperatorSetIdProto()
        proto.ParseFromString(bytePointer)
        val initialId = OpSetID(0)
        for(i in 0 until proto.opset_import_size()) {
            val opSetImport = proto.opset_import(i)
            if(!opSetImport.has_domain() || opSetImport.domain().string == "ai.onnx") {
                //approximates default opset from https://github.com/onnx/onnx/blob/master/onnx/version_converter/convert.cc#L14
                initialId.setVersion(opSetImport.version().toInt())
                break

            }
        }

        val convertVersion = converter.convert_version(proto, initialId, OpSetID(13))
        val save = convertVersion.SerializeAsString()
        IOUtils.write(save.stringBytes, FileOutputStream(outputModelFilePath))

    }


    fun addConstValueInfoToGraph(graph: Onnx.GraphProto): Onnx.GraphProto {
        val inputs = graph.inputList.map { input -> input.name }
        val existingInfoNames = graph.valueInfoList.map { input -> input.name to input}.toMap()
        val graphBuilder = graph.toBuilder()
        for(init in graphBuilder.initializerList) {
            if(inputs.contains(init.name)) {
                continue
            }

            val elemType = init.dataType
            val shape = init.dimsList
            val vi = if(existingInfoNames.containsKey(init.name)) {
                existingInfoNames[init.name]!!
            } else {
                val newAdd = graphBuilder.addValueInfoBuilder()
                newAdd.name = init.name
                newAdd.build()
            }

            if(!inputs.contains(init.name)) {
                graphBuilder.addInput(vi)
            }

            val ttElem = vi.type.tensorType
            val ttDType = vi.type.tensorType.elemType
            if(ttDType == Onnx.TensorProto.DataType.UNDEFINED) {
                ttElem.toBuilder().elemType = ttElem.elemType
            }

            if(!ttElem.hasShape()) {
                for(dim in shape) {
                    ttElem.toBuilder().shape.toBuilder().addDimBuilder().dimValue = dim
                }
            }

        }


        for(node in graphBuilder.nodeList) {
            for(attr in node.attributeList) {
                if(attr.name != "") {
                    if(attr.type == Onnx.AttributeProto.AttributeType.GRAPH) {
                        attr.toBuilder().g = addConstValueInfoToGraph(attr.g)
                    }
                    if(attr.type == Onnx.AttributeProto.AttributeType.GRAPHS) {
                        for(i in 0 until attr.graphsCount) {
                            attr.toBuilder().setGraphs(i,addConstValueInfoToGraph(attr.getGraphs(i)))
                        }
                    }
                }
            }
        }

        return graphBuilder.build()
    }



}