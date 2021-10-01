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