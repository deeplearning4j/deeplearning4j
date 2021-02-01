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
package org.nd4j.samediff.frameworkimport.onnx.opdefs

import onnx.Onnx
import org.apache.commons.io.IOUtils
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.onnx.process.OnnxMappingProcessLoader
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoader
import org.nd4j.samediff.frameworkimport.opdefs.nd4jFileNameTextDefault
import org.nd4j.samediff.frameworkimport.opdefs.nd4jFileSpecifierProperty
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.shade.protobuf.TextFormat
import java.nio.charset.Charset

class OnnxOpDescriptorLoader: OpDescriptorLoader<Onnx.NodeProto> {


    val onnxFileNameTextDefault = "onnx-op-defs.pb"
    val onnxFileSpecifierProperty = "samediff.import.onnxdescriptors"


    val onnxMappingRulSetDefaultFile = "onnx-mapping-ruleset.pbtxt"
    val onnxRulesetSpecifierProperty = "samediff.import.onnxmappingrules"
    val nd4jOpDescriptors = nd4jOpList()
    var mapperDefSet: MapperNamespace.MappingDefinitionSet? = mappingProcessDefinitionSet()
    var cachedOpDefs:  Map<String,Onnx.NodeProto>?  = inputFrameworkOpDescriptorList()


    override fun frameworkName(): String {
        return "onnx"
    }


    override fun nd4jOpList(): OpNamespace.OpDescriptorList {
        val fileName = System.getProperty(nd4jFileSpecifierProperty, nd4jFileNameTextDefault)
        val nd4jOpDescriptorResourceStream = ClassPathResource(fileName).inputStream
        val resourceString = IOUtils.toString(nd4jOpDescriptorResourceStream, Charset.defaultCharset())
        val descriptorListBuilder = OpNamespace.OpDescriptorList.newBuilder()
        TextFormat.merge(resourceString,descriptorListBuilder)
        val ret = descriptorListBuilder.build()
        val mutableList = ArrayList(ret.opListList)
        mutableList.sortBy { it.name }

        val newResultBuilder = OpNamespace.OpDescriptorList.newBuilder()
        newResultBuilder.addAllOpList(mutableList)
        return newResultBuilder.build()
    }

    override fun inputFrameworkOpDescriptorList(): Map<String,Onnx.NodeProto> {
        if(cachedOpDefs != null)
            return cachedOpDefs!!
        val fileName = System.getProperty(onnxFileSpecifierProperty, onnxFileNameTextDefault)
        val stream = ClassPathResource(fileName).inputStream
        val ret = HashMap<String,Onnx.NodeProto>()
        val graphProto = Onnx.GraphProto.parseFrom(stream)

        graphProto.nodeList.forEach { opDef ->
            ret[opDef.name] = opDef
        }

        cachedOpDefs =  ret
        return ret
    }

    override fun mappingProcessDefinitionSet(): MapperNamespace.MappingDefinitionSet {
        if(mapperDefSet != null)
            return mapperDefSet!!
        val fileName = System.getProperty(onnxRulesetSpecifierProperty, onnxMappingRulSetDefaultFile)
        val string = IOUtils.toString(ClassPathResource(fileName).inputStream, Charset.defaultCharset())
        val declarationBuilder = MapperNamespace.MappingDefinitionSet.newBuilder()
        TextFormat.merge(string,declarationBuilder)
        val ret =  declarationBuilder.build()
        this.mapperDefSet = ret
        return ret
    }

    override fun <GRAPH_TYPE : GeneratedMessageV3, NODE_TYPE : GeneratedMessageV3, OP_DEF_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTR_DEF_TYPE : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum> createOpMappingRegistry(): OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE> {
        val onnxMappingRegistry =  OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto>("onnx",nd4jOpDescriptors)
        val loader = OnnxMappingProcessLoader(onnxMappingRegistry)
        val mappingProcessDefs = mappingProcessDefinitionSet()
        onnxMappingRegistry.loadFromDefinitions(mappingProcessDefs,loader)
        return onnxMappingRegistry as OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>
    }


}