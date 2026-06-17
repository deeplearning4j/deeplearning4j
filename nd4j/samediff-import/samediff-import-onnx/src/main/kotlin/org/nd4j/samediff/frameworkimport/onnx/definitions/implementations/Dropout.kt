package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import onnx.Onnx
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
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        var inputVariable = sd.getVariable(op.inputsToOp[0])

        // Get dropout ratio - prefer attribute, then input variable, then default
        val p = if(attributes.containsKey("ratio")) {
            val fVal = attributes["ratio"] as Float
            fVal.toDouble()
        } else if(op.inputsToOp.size > 1) {
            // Try to get value from dynamicVariables (ONNX TensorProto)
            val ratioVarName = op.inputsToOp[1]
            getDoubleFromTensorProto(dynamicVariables, ratioVarName) ?: 0.5
        } else {
            0.5
        }


        val outputVar = sd.nn().dropout(outputNames[0],inputVariable,true,p)
        return mapOf(outputVar.name() to listOf(outputVar))

    }

    /**
     * Extract double value from ONNX TensorProto in dynamicVariables.
     */
    private fun getDoubleFromTensorProto(
        dynamicVariables: Map<String, GeneratedMessageV3>,
        varName: String
    ): Double? {
        val tensorProto = dynamicVariables[varName] as? Onnx.TensorProto ?: return null
        return when {
            tensorProto.floatDataCount > 0 -> tensorProto.floatDataList[0].toDouble()
            tensorProto.doubleDataCount > 0 -> tensorProto.doubleDataList[0]
            tensorProto.rawData.size() > 0 -> {
                val buffer = tensorProto.rawData.asReadOnlyByteBuffer()
                buffer.order(java.nio.ByteOrder.LITTLE_ENDIAN)
                when (tensorProto.dataType) {
                    Onnx.TensorProto.DataType.FLOAT_VALUE -> buffer.asFloatBuffer().get().toDouble()
                    Onnx.TensorProto.DataType.DOUBLE_VALUE -> buffer.asDoubleBuffer().get()
                    else -> null
                }
            }
            else -> null
        }
    }
}