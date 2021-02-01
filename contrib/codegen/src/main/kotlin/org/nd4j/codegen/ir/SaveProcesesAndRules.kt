package org.nd4j.codegen.ir

import org.nd4j.codegen.ir.onnx.OnnxOpDeclarations
import org.nd4j.codegen.ir.registry.OpMappingRegistry
import org.nd4j.codegen.ir.registry.OpRegistryHolder
import org.nd4j.codegen.ir.tensorflow.TensorflowOpDeclarations

class SaveProcesesAndRules {

    companion object {
        @JvmStatic fun main(args : Array<String>) {
            val tensorflowDeclarations = TensorflowOpDeclarations
            val onnxDeclarations = OnnxOpDeclarations
            OpRegistryHolder.tensorflow().saveProcessesAndRuleSet()
            OpRegistryHolder.onnx().saveProcessesAndRuleSet()
        }
    }


}