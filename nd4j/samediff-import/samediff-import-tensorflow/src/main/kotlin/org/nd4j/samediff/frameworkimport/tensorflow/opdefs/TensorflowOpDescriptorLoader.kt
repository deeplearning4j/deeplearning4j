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
package org.nd4j.samediff.frameworkimport.tensorflow.opdefs

import org.apache.commons.io.IOUtils
import org.nd4j.common.config.ND4JClassLoading
import org.nd4j.common.io.ClassPathResource
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoader
import org.nd4j.samediff.frameworkimport.opdefs.nd4jFileNameTextDefault
import org.nd4j.samediff.frameworkimport.opdefs.nd4jFileSpecifierProperty
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.samediff.frameworkimport.tensorflow.process.TensorflowMappingProcessLoader
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.nd4j.shade.protobuf.TextFormat
import org.tensorflow.framework.*
import java.lang.Exception
import java.nio.charset.Charset

class TensorflowOpDescriptorLoader: OpDescriptorLoader<OpDef> {

    val tensorflowFileNameTextDefault = "/tensorflow-op-def.pbtxt"
    val tensorflowFileSpecifierProperty = "samediff.import.tensorflowdescriptors"

    val tensorflowMappingRulSetDefaultFile = "/tensorflow-mapping-ruleset.pbtxt"
    val tensorflowRulesetSpecifierProperty = "samediff.import.tensorflowmappingrules"
    val nd4jOpDescriptors = nd4jOpList()
    var mapperDefSet: MapperNamespace.MappingDefinitionSet? = mappingProcessDefinitionSet()
    var cachedOpList: Map<String,OpDef>? = inputFrameworkOpDescriptorList()
    override fun frameworkName(): String {
        return "tensorflow"
    }

    override fun nd4jOpList(): OpNamespace.OpDescriptorList {
        val fileName = System.getProperty(nd4jFileSpecifierProperty, nd4jFileNameTextDefault)
        val nd4jOpDescriptorResourceStream = ClassPathResource(fileName,ND4JClassLoading.getNd4jClassloader()).inputStream
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

    override fun inputFrameworkOpDescriptorList(): Map<String,OpDef> {
        if(cachedOpList != null) {
            return cachedOpList!!
        }
        val fileName = System.getProperty(tensorflowFileSpecifierProperty, tensorflowFileNameTextDefault)
        val string = IOUtils.toString(ClassPathResource(fileName,ND4JClassLoading.getNd4jClassloader()).inputStream, Charset.defaultCharset())
        val tfListBuilder = OpList.newBuilder()
        TextFormat.merge(string, tfListBuilder)
        val ret = HashMap<String,OpDef>()
        tfListBuilder.build().opList.forEach { opDef ->
            ret[opDef.name] = opDef
        }

        return ret
    }



    override fun mappingProcessDefinitionSet(): MapperNamespace.MappingDefinitionSet {
        if(mapperDefSet != null)
            return mapperDefSet!!
        val fileName = System.getProperty(tensorflowRulesetSpecifierProperty, tensorflowMappingRulSetDefaultFile)
        val string = IOUtils.toString(ClassPathResource(fileName,ND4JClassLoading.getNd4jClassloader()).inputStream, Charset.defaultCharset())
        val declarationBuilder = MapperNamespace.MappingDefinitionSet.newBuilder()
       try {
           TextFormat.merge(string,declarationBuilder)
       } catch(e: Exception) {
           println("Unable to parse mapper definitions for file file $fileName")
       }

        return declarationBuilder.build()
    }

    override fun <GRAPH_TYPE : GeneratedMessageV3, NODE_TYPE : GeneratedMessageV3, OP_DEF_TYPE : GeneratedMessageV3, TENSOR_TYPE : GeneratedMessageV3, ATTR_DEF_TYPE : GeneratedMessageV3, ATTR_VALUE_TYPE : GeneratedMessageV3, DATA_TYPE : ProtocolMessageEnum> createOpMappingRegistry(): OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE> {
        val tensorflowOpMappingRegistry =  OpMappingRegistry<GraphDef,NodeDef,OpDef,TensorProto,DataType,OpDef.AttrDef,AttrValue>("tensorflow",nd4jOpDescriptors)
        val loader = TensorflowMappingProcessLoader(tensorflowOpMappingRegistry)
        val mappingProcessDefs = mappingProcessDefinitionSet()
        tensorflowOpMappingRegistry.loadFromDefinitions(mappingProcessDefs,loader)
        return tensorflowOpMappingRegistry as OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTR_DEF_TYPE, ATTR_VALUE_TYPE>
    }
}