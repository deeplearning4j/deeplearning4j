package org.nd4j.codegen.ir.registry

import onnx.Onnx
import org.apache.commons.collections4.multimap.HashSetValuedHashMap
import org.nd4j.codegen.ir.MappingProcess
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum
import org.tensorflow.framework.*

object OpRegistryHolder {

    private val registeredOps = HashSetValuedHashMap<String, OpMappingRegistry<out GeneratedMessageV3,out GeneratedMessageV3, out GeneratedMessageV3, out GeneratedMessageV3, out ProtocolMessageEnum, out GeneratedMessageV3, out GeneratedMessageV3>>()
    private val opDefLists = HashMap<String,Map<String,GeneratedMessageV3>>()

    fun <GRAPH_TYPE: GeneratedMessageV3,
            NODE_TYPE: GeneratedMessageV3,
            OP_DEF_TYPE: GeneratedMessageV3,
            TENSOR_TYPE: GeneratedMessageV3,
            ATTRIBUTE_TYPE: GeneratedMessageV3,
            ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3,
            DATA_TYPE : ProtocolMessageEnum> opMappingRegistryForName(name: String) : OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE>{
        return registeredOps[name].first() as  OpMappingRegistry<GRAPH_TYPE, NODE_TYPE, OP_DEF_TYPE, TENSOR_TYPE, DATA_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE>

    }


    fun onnx(): OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,Onnx.NodeProto,Onnx.TensorProto,Onnx.TensorProto.DataType,Onnx.AttributeProto,Onnx.AttributeProto> {
        return registeredOps["onnx"].first() as OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.TensorProto.DataType, Onnx.AttributeProto, Onnx.AttributeProto>
    }

    fun tensorflow(): OpMappingRegistry<GraphDef,
            NodeDef,
            OpDef,
            TensorProto, DataType,OpDef.AttrDef, AttrValue> {
        return registeredOps["tensorflow"].first() as OpMappingRegistry<GraphDef,
                NodeDef, OpDef, TensorProto, DataType, OpDef.AttrDef, AttrValue>
    }

    fun <OP_DEF_TYPE: GeneratedMessageV3> opListForFramework(frameworkName: String): Map<String,OP_DEF_TYPE> {
        return opDefLists[frameworkName] as Map<String,OP_DEF_TYPE>
    }

    fun <OP_DEF_TYPE: GeneratedMessageV3> registerOpList(inputFrameworkName: String,opDefMap: Map<String,OP_DEF_TYPE>) {
        opDefLists[inputFrameworkName] = opDefMap

    }

    fun registerOpMappingRegistry(framework: String, registry: OpMappingRegistry<out GeneratedMessageV3,out GeneratedMessageV3, out GeneratedMessageV3, out GeneratedMessageV3, out ProtocolMessageEnum, out GeneratedMessageV3, out GeneratedMessageV3>) {
        registeredOps.put(framework,registry)
    }

    fun <GRAPH_TYPE: GeneratedMessageV3,NODE_TYPE : GeneratedMessageV3,OP_DEF_TYPE: GeneratedMessageV3,TENSOR_TYPE: GeneratedMessageV3, DATA_TYPE: ProtocolMessageEnum,ATTRIBUTE_TYPE : GeneratedMessageV3, ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>
            registerMappingProcess(inputFrameworkOpName: String, processToRegister: MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>) {
        registeredOps.put(inputFrameworkOpName,processToRegister as OpMappingRegistry<GeneratedMessageV3,GeneratedMessageV3, GeneratedMessageV3,
                GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>)
    }

    fun  <GRAPH_TYPE: GeneratedMessageV3,NODE_TYPE : GeneratedMessageV3,
            OP_DEF_TYPE: GeneratedMessageV3,
            TENSOR_TYPE: GeneratedMessageV3,
            DATA_TYPE: ProtocolMessageEnum,
            ATTRIBUTE_TYPE : GeneratedMessageV3,
            ATTRIBUTE_VALUE_TYPE: GeneratedMessageV3>
            lookupOpMappingProcess(inputFrameworkName: String, inputFrameworkOpName: String):
            MappingProcess<GRAPH_TYPE,OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE> {
        val mappingRegistry = registeredOps[inputFrameworkName].first()
        val lookup = mappingRegistry.lookupOpMappingProcess(inputFrameworkOpName = inputFrameworkOpName)
        return lookup as MappingProcess<GRAPH_TYPE, OP_DEF_TYPE, NODE_TYPE, TENSOR_TYPE, ATTRIBUTE_TYPE, ATTRIBUTE_VALUE_TYPE, DATA_TYPE>
    }
}
