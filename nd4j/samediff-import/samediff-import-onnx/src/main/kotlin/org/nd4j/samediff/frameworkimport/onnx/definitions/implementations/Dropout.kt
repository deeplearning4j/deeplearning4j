package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Maps dropout
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Dropout"],frameworkName = "onnx")
class Dropout : PreImportHook  {


    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val p = if(attributes.containsKey("ratio")) {
            val fVal = attributes["ratio"] as Float
            fVal.toDouble()
        } else if(op.inputsToOp.size > 1) {
            val dropoutVar = sd.getVariable(op.inputsToOp[1]).arr.getDouble(0)
            dropoutVar
        } else {
            0.5
        }

        sd.ops.remove(op.name)

        val outputVar = sd.nn().dropoutInverted(outputNames[0],inputVariable,p)
        return mapOf(outputVar.name() to listOf(outputVar))

    }


}