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
package org.nd4j.samediff.frameworkimport.registry

import onnx.Onnx
import org.apache.commons.collections4.MultiSet
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.apache.commons.collections4.MultiValuedMap
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.apache.commons.io.FileUtils

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.findOp
import org.nd4j.samediff.frameworkimport.opdefs.OpDescriptorLoaderHolder
import org.nd4j.samediff.frameworkimport.process.MappingProcess
import org.nd4j.samediff.frameworkimport.process.MappingProcessLoader
import org.nd4j.samediff.frameworkimport.process.PreImportHookMappingProcess
import org.nd4j.samediff.frameworkimport.reflect.ImportReflectionCache
import org.nd4j.shade.protobuf.TextFormat
import java.io.File
import java.lang.IllegalArgumentException
import java.nio.charset.Charset


class OpMappingRegistry<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>(inputFrameworkName: String,nd4jOpDescriptors: OpNamespace.OpDescriptorList) {

    val registeredOps: MultiValuedMap<String, MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE,
            TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>> = HashSetValuedHashMap()

    val opDefList = HashMap<String,OP_DEF_TYPE>()
    val nd4jOpDefs = HashMap<String,OpNamespace.OpDescriptor>()
    val inputFrameworkName = inputFrameworkName
    val nd4jOpDescriptors = nd4jOpDescriptors
    val cache = ImportReflectionCache

    fun mappedNd4jOpNames(): Set<String> {
        return registeredOps.values().map { input -> input.opName() }.toSortedSet()!!
    }

    fun mappingProcessNames(): MultiSet<String> {
        return registeredOps.keys()!!
    }

    fun nd4jOpNames(): Set<String> {
        return nd4jOpDefs.keys
    }

    fun inputFrameworkOpNames(): Set<String> {
        return opDefList.keys
    }

    fun lookupNd4jOpDef(name:String): OpNamespace.OpDescriptor {
        return nd4jOpDefs[name]!!
    }

    fun registerOpDefs(opDefList: Map<String,OP_DEF_TYPE>) {
        opDefList.forEach { (name,inputOpDef) ->
            registerInputFrameworkOpDef(name,inputOpDef)
        }
    }

    fun registerNd4jOpDef(name:String, opDef: OpNamespace.OpDescriptor) {
        nd4jOpDefs[name] = opDef
    }

    fun lookupInputFrameworkOpDef(name:String): OP_DEF_TYPE {
        if(opDefList.isEmpty()) {
            val opList =  OpDescriptorLoaderHolder.listForFramework<OP_DEF_TYPE>(inputFrameworkName)
            opList.forEach { (name, opDefType) ->
                opDefList[name] = opDefType
            }
        }

        //workaround for placeholder not being defined, only used in limited circumstances
        if(name == "Placeholder" && !opDefList.containsKey("Placeholder"))
            return opDefList["Constant"]!!
            
        // Check if this is a PreImportHook operation before failing
        if (!opDefList.containsKey(name)) {
            val hasPreHookRules = cache.preProcessRuleImplementationsByOp.cellSet().any { cell ->
                cell.rowKey == inputFrameworkName && cell.columnKey == name
            }
            
            if (hasPreHookRules) {
                // Create a dummy op definition for PreImportHook operations
                val dummyOpDef = createDummyOpDefinition(name)
                opDefList[name] = dummyOpDef
                return dummyOpDef
            }
        }
        
        // Check if the op definition exists
        if (!opDefList.containsKey(name)) {
            throw IllegalArgumentException("No op definition found for '$name' in framework '$inputFrameworkName'. Available ops: ${opDefList.keys.sorted()}")
        }
        
        return opDefList[name]!!
    }

    @Suppress("UNCHECKED_CAST")
    private fun createDummyOpDefinition(opName: String): OP_DEF_TYPE {
        // For ONNX, create a minimal NodeProto
        return when (inputFrameworkName) {
            "onnx" -> {
                Onnx.NodeProto.newBuilder()
                    .setOpType(opName)
                    .setDomain("com.microsoft") // Default domain for most PreImportHook ops
                    .setName("${opName}_prehook")
                    .build() as OP_DEF_TYPE
            }
            else -> {
                throw IllegalArgumentException("Dummy op definition creation not implemented for framework: $inputFrameworkName")
            }
        }
    }

    fun registerInputFrameworkOpDef(name: String,opDef: OP_DEF_TYPE) {
        opDefList[name] = opDef
    }

    fun registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister)
    }

    fun hasMappingOpProcess(inputFrameworkOpName: String): Boolean {
        return registeredOps.containsKey(inputFrameworkOpName)
    }

    fun  lookupOpMappingProcess(inputFrameworkOpName: String): MappingProcess<
            GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE> {

        if(!registeredOps.containsKey(inputFrameworkOpName)) {
            val allRules = cache.preProcessRuleImplementationsByOp
            val hasPreHookRules = allRules.cellSet().any { cell ->
                cell.rowKey == inputFrameworkName && cell.columnKey == inputFrameworkOpName
            }
            
            if(hasPreHookRules) {
                // Create and register a dummy MappingProcess for this PreImportHook operation
                val dummyProcess = PreImportHookMappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>(
                    inputFrameworkName, inputFrameworkOpName
                )
                registeredOps.put(inputFrameworkOpName, dummyProcess)
            } else {
                throw IllegalArgumentException("No import process defined for $inputFrameworkOpName")
            }
        }

        return registeredOps[inputFrameworkOpName]!!.first()
    }

    fun opTypeForName(nd4jOpName: String): OpNamespace.OpDescriptor.OpDeclarationType {
        val descriptor = nd4jOpDescriptors.findOp(nd4jOpName)
        return descriptor.opDeclarationType
    }

    fun opHasRuleNoProcess(inputFrameworkOpName: String): Boolean {
        val allRules = cache.preProcessRuleImplementationsByOp
        val noRules = allRules.cellSet().filter { input -> input.rowKey == inputFrameworkOpName }.isEmpty()
        return noRules && !registeredOps.containsKey(inputFrameworkOpName)
    }


    /**
     * TODO: Make loading op mapping rules (both tensor and attribute), input framework op definitions casted as
     * OP_DEF_TYPE and op descriptors file.
     *
     * TODO: Get rid of static global constants (onnxops,tensorflow ops)
     * TODO: See if possible to genericize lists of ops
     */
    fun loadFromFile(mapperDeclarationsFile: String,
                     mappingProcessLoader: MappingProcessLoader<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>) {

        val loadedFile = File(mapperDeclarationsFile)
        val bytes = FileUtils.readFileToByteArray(loadedFile)
        val parsed = MapperNamespace.MappingDefinitionSet.newBuilder()
        val string = String(bytes, Charset.defaultCharset())
        TextFormat.merge(string,parsed)
        val defs = parsed.build()
        loadFromDefinitions(defs,mappingProcessLoader)
    }


    /**
     * TODO: Make loading op mapping rules (both tensor and attribute), input framework op definitions casted as
     * OP_DEF_TYPE and op descriptors file.
     *
     * TODO: Get rid of static global constants (onnxops,tensorflow ops)
     * TODO: See if possible to genericize lists of ops
     */
    fun loadFromDefinitions(mapperDeclarations: MapperNamespace.MappingDefinitionSet,
                            mappingProcessLoader: MappingProcessLoader<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>) {
        mapperDeclarations.mappingsList.forEach {
            val process = mappingProcessLoader.createProcess(it)
            this.registerMappingProcess(it.inputFrameworkOpName,process)
        }

    }


    fun saveProcessesAndRuleSet() {
        val mapperDeclarations = ArrayList<MapperNamespace.MapperDeclaration>()
        val bufferToWrite = StringBuilder()
        registeredOps.asMap().forEach { (_, listOfMappingProcesses) ->
            listOfMappingProcesses.forEach { mappingProcess ->
                mapperDeclarations.add(mappingProcess.serialize())
            }

            mapperDeclarations.map { input -> input.toString() }.forEach { processString ->
                bufferToWrite.append(processString + "\n")
            }

        }

        val mapperSet = MapperNamespace.MappingDefinitionSet.newBuilder()
        mapperSet.addAllMappings(mapperDeclarations)

        val finalSet = mapperSet.build()

        FileUtils.write(File("$inputFrameworkName-processes.pbtxt"),finalSet.toString(), Charset.defaultCharset())

    }

}