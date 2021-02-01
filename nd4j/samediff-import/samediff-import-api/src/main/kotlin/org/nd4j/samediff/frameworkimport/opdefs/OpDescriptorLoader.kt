package org.nd4j.samediff.frameworkimport.opdefs

import org.nd4j.ir.MapperNamespace
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

val nd4jFileNameTextDefault = "nd4j-op-def.pbtxt"
val nd4jFileSpecifierProperty = "samediff.import.nd4jdescriptors"

interface OpDescriptorLoader<OP_DEF_TYPE: GeneratedMessageV3> {

    fun frameworkName(): String

    fun nd4jOpList(): OpNamespace.OpDescriptorList

    fun mappingProcessDefinitionSet(): MapperNamespace.MappingDefinitionSet

    fun inputFrameworkOpDescriptorList(): Map<String,OP_DEF_TYPE>

    fun <GRAPH_TYPE: GeneratedMessageV3,
            NODE_TYPE : GeneratedMessageV3,
            OP_DEF_TYPE : GeneratedMessageV3,
            TENSOR_TYPE : GeneratedMessageV3,
            ATTR_DEF_TYPE : GeneratedMessageV3,
            ATTR_VALUE_TYPE : GeneratedMessageV3,
            DATA_TYPE: ProtocolMessageEnum>
            createOpMappingRegistry(): OpMappingRegistry<GRAPH_TYPE,NODE_TYPE,OP_DEF_TYPE,TENSOR_TYPE,DATA_TYPE,ATTR_DEF_TYPE,ATTR_VALUE_TYPE>

}