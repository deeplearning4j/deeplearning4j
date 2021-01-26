package org.nd4j.codegen.ir.registry

import org.apache.commons.collections4.MultiSet
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.apache.commons.collections4.MultiValuedMap
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.apache.commons.io.FileUtils
import org.nd4j.codegen.ir.MappingProcess
import org.nd4j.codegen.ir.findOp
import org.nd4j.codegen.ir.nd4jOpDescriptors
import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import java.io.File
import java.lang.IllegalArgumentException
import java.nio.charset.Charset


class OpMappingRegistry<GRAPH_TYPE: GeneratedMessageV3,
        NODE_TYPE : GeneratedMessageV3,
        OP_DEF_TYPE: GeneratedMessageV3,
        TENSOR_TYPE: GeneratedMessageV3,
        DATA_TYPE: ProtocolMessageEnum,
        ATTRIBUTE_TYPE : GeneratedMessageV3,
        ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>(inputFrameworkName: String) {

    val registeredOps: MultiValuedMap<String,MappingProcess<GRAPH_TYPE,OP_DEF_TYPE,NODE_TYPE,
            TENSOR_TYPE,ATTRIBUTE_TYPE,ATTRIBUTE_VALUE_TYPE,DATA_TYPE>> = HashSetValuedHashMap<
            String,MappingProcess<GRAPH_TYPE,
            OP_DEF_TYPE,
            NODE_TYPE,
            TENSOR_TYPE,
            ATTRIBUTE_TYPE,
            ATTRIBUTE_VALUE_TYPE,
            DATA_TYPE>>()

    val opDefList = HashMap<String,OP_DEF_TYPE>()
    val nd4jOpDefs = HashMap<String,OpNamespace.OpDescriptor>()
    val inputFrameworkName = inputFrameworkName



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
            val opList = OpRegistryHolder.opListForFramework<OP_DEF_TYPE>(inputFrameworkName)
            opList.forEach {  name,opDefType ->
                opDefList[name] = opDefType
            }
        }
        return opDefList[name]!!
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
            throw IllegalArgumentException("No import process defined for $inputFrameworkOpName")
        }
        return registeredOps[inputFrameworkOpName]!!.first()
    }

    fun opTypeForName(nd4jOpName: String): OpNamespace.OpDescriptor.OpDeclarationType {
        val descriptor = nd4jOpDescriptors.findOp(nd4jOpName)
        return descriptor.opDeclarationType
    }

    /**
     * TODO: Make loading op mapping rules (both tensor and attribute), input framework op definitions casted as
     * OP_DEF_TYPE and op descriptors file.
     *
     * TODO: Get rid of static global constants (onnxops,tensorflow ops)
     * TODO: See if possible to genericize lists of ops
     */
    fun loadFromFile(inputFrameworkOpDefsName: String,nd4jOpDescriptorsFile: String,inputFrameworkName: String,mapperDeclarationsFile: String) {
           //Nd4j op descriptors: OpNamespace.OpDescriptorList TextFormat.merge(..)
           //
    }

    fun saveProcessesAndRuleSet() {
        val mapperDeclarations = ArrayList<MapperNamespace.MapperDeclaration>()
        val bufferToWrite = StringBuilder()
        registeredOps.asMap().forEach { name, listOfMappingProcesses ->
            listOfMappingProcesses.forEach { mappingProcess ->
                mapperDeclarations.add(mappingProcess.serialize())
            }

            mapperDeclarations.map { input -> input.toString() }.forEach { processString ->
                bufferToWrite.append(processString + "\n")
            }

        }

        FileUtils.write(File("$inputFrameworkName-processes.pbtxt"),bufferToWrite.toString(), Charset.defaultCharset())

    }

}






