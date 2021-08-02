package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.common.util.ArrayUtil
import org.nd4j.ir.OpNamespace
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule

@PreHookRule(nodeNames = [],opNames = ["GlobalAMaxPool"],frameworkName = "onnx")
class GlobalMaxPooling: PreImportHook {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor
    ): HookResult {
        val inputVariable = sd.getVariable(op.inputsToOp[0])
        val rankOf = sd.rank(inputVariable)
        val rangeMin = minNum(rankOf.eval().getInt(0))
        val rankList = ArrayUtil.range(rangeMin as Long,rankOf as Long).map { input ->input.toInt() }.toIntArray()
        val output = sd.mean(inputVariable,true,*rankList)
        sd.ops.remove(op.name)
        sd.variables.remove(op.name)
        return HookResult(outputVariables = mapOf(output.name() to listOf(output)),
            proceedWithInit = false)
    }

    fun minNum (rank: Int): Int {
        if(rank < 4)
            return 2
        return 3
    }

}