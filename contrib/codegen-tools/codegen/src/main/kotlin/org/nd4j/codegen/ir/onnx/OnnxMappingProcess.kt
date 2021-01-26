package org.nd4j.codegen.ir.onnx

import onnx.Onnx
import org.nd4j.codegen.ir.AbstractMappingProcess
import org.nd4j.codegen.ir.AttributeMappingRule
import org.nd4j.codegen.ir.AttributeValueType
import org.nd4j.codegen.ir.TensorMappingRule
import org.nd4j.codegen.ir.registry.OpMappingRegistry

open class OnnxMappingProcess(inputFramework: String = "onnx",
                              frameworkVersion: String = "1.4",
                              inputFrameworkOpName: String,
                              opName: String,
                              opMappingRegistry: OpMappingRegistry<Onnx.GraphProto,Onnx.NodeProto,
                                      Onnx.NodeProto,
                                      Onnx.TensorProto,
                                      Onnx.TensorProto.DataType,
                                      Onnx.AttributeProto,
                                      Onnx.AttributeProto>,
                              tensorMappingRules: List<TensorMappingRule<Onnx.GraphProto,
                                      Onnx.NodeProto, Onnx.NodeProto,
                                      Onnx.AttributeProto, Onnx.AttributeProto,
                                      Onnx.TensorProto,Onnx.TensorProto.DataType>> = emptyList(),
                              inputIndexOverrides: Map<Int,Int> = emptyMap(),
                              attributeMappingRules: List<out AttributeMappingRule<Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto,Onnx.AttributeProto, Onnx.AttributeProto,
                                      Onnx.TensorProto, Onnx.TensorProto.DataType>> = emptyList())
    : AbstractMappingProcess<Onnx.GraphProto,Onnx.NodeProto, Onnx.NodeProto, Onnx.TensorProto, Onnx.AttributeProto, Onnx.AttributeProto, Onnx.TensorProto.DataType>(
    inputFramework,
    frameworkVersion,
    inputFrameworkOpName,
    inputIndexOverrides,
    opName,
    opMappingRegistry,
    tensorMappingRules,
    attributeMappingRules) {
    override fun inputOpDefValueTypes(): Map<String, AttributeValueType> {
        val opDef = opMappingRegistry.lookupInputFrameworkOpDef(inputFrameworkOpName)
        val ret = HashMap<String,AttributeValueType>()
        opDef.attributeList.forEach { attributeProto ->
              ret[attributeProto.name] = attributeValueTypeForOnnxAttribute(attributeProto)
        }

        return ret
    }

}

