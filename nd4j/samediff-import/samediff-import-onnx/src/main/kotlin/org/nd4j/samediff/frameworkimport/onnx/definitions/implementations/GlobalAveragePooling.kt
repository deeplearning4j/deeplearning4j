package org.nd4j.samediff.frameworkimport.onnx.definitions.implementations

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule

@PreHookRule(nodeNames = [],opNames = ["GlobalAveragePool"],frameworkName = "onnx")
class GlobalAveragePooling: PreImportHook {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>
    ): HookResult {
        val inputVariable = sd.getVariable(op.inputsToOp[0])
        val rankOf = sd.rank(inputVariable)
        val range = sd.range(sd.constant(2),rankOf,sd.constant(1),DataType.INT64)
        val output = sd.math.mean(outputNames[0],inputVariable,range,true)
        sd.ops.remove(op.name)
        return HookResult(outputVariables = mapOf(output.name() to listOf(output)),
            proceedWithInit = false)
    }

    fun trueBodyFor(rank: Int): SameDiff {
        val ret = SameDiff.create()
        if(rank == 2)
            ret.constant(Nd4j.create(Nd4j.createBuffer(intArrayOf(2))))
        else if(rank == 3) {
            ret.constant(Nd4j.create(Nd4j.createBuffer(intArrayOf(2,3))))
        } else if(rank == 4) {
            ret.constant(Nd4j.create(Nd4j.createBuffer(intArrayOf(2,3,4))))

        }
        return ret
    }

    fun rankTest(rank: Int): SameDiff {
        val ret = SameDiff.create()
        val input = ret.placeHolder("x", DataType.INT64)
        val const = ret.constant(rank)
        ret.eq(input,const)
        return ret
    }

    fun minNum (rank: Int): Int {
        if(rank < 4)
            return 2
        return 3
    }

}