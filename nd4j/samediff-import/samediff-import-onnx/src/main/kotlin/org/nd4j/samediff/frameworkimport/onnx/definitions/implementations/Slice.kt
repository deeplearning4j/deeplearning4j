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

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of slice.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/slice.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Slice"],frameworkName = "onnx")
class Slice : PreImportHook  {
    override fun preProcess(
        op: SameDiffOp,
        sd: SameDiff,
        attributes: Map<String, Any>,
        descriptor: OpNamespace.OpDescriptor,
        outputNames: List<String>,
        isFinalOutput: Boolean,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): HookResult {
        // Parameter docs below are from the onnx operator docs:
        // https://github.com/onnx/onnx/blob/master/docs/Operators.md#slice

        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val inputTensorShape = sd.shape(inputVariable)
        val starts = sd.getVariable(op.inputsToOp[1])
        val ends = sd.getVariable(op.inputsToOp[2])
        val axes = if(op.inputsToOp.size < 4) sd.range(sd.constant(0),sd.shape(starts),sd.constant(1),starts.dataType())
        else sd.getVariable(op.inputsToOp[3])
        val inputRank = sd.rank(inputVariable)
        val isAxesNegative = sd.lt(axes,sd.zerosLike(axes))
        val axesWhere = sd.where(axes.add(inputRank),axes,isAxesNegative)
        val sparseIndices = sd.castTo(sd.expandDims(axesWhere,-1),DataType.INT64)
        val sparseShape = sd.gatherNd(sd.shape(inputVariable),sparseIndices).castTo(ends.dataType())
        val startsMin = sd.min(starts,sparseShape)
        val endsMin = sd.min(ends,sparseShape)

        val isStartsNegative = sd.lt(startsMin,sd.zerosLike(startsMin))
        val startsFinal = sd.where(startsMin.add(sparseShape),startsMin,isStartsNegative)
        val isEndsNegative = sd.lt(endsMin,sd.zerosLike(endsMin))
        val endsFinal = sd.where(endsMin.add(sparseShape),endsMin,isEndsNegative)
        val outputShape = inputRank.castTo(DataType.INT64)
        val denseBegins = sd.sparseToDense(sparseIndices,outputShape,startsFinal)


        val denseEnds = sd.sparseToDense(sparseIndices,outputShape,endsFinal,sd.constant(Nd4j.create(
            floatArrayOf(-1.0f)).castTo(denseBegins.dataType())))
       val denseEnds2 = sd.where(inputTensorShape,denseEnds,sd.eq(denseEnds,sd.constant(-1).castTo(denseBegins.dataType())))

        val denseSteps: SDVariable = if(op.inputsToOp.size >= 5) {
            val inputVar = sd.getVariable(op.inputsToOp[4])
            sd.sparseToDense(sparseIndices,
                outputShape,inputVar,
               sd.constant(Nd4j.create(floatArrayOf(1.0f))
                   .castTo(inputVar.dataType())))
        } else {
            sd.onesLike(inputVariable.shape())
        }

        val outputVarName: String? = if(isFinalOutput) {
            outputNames[0]
        } else null

        if(outputVarName != null && sd.hasVariable(outputVarName)) {
            sd.variables.remove(outputVarName)
            sd.ops.remove(outputVarName)
        }

        val finalVal = sd.stridedSlice(outputVarName,inputVariable,denseBegins,denseEnds2,denseSteps,0,0,0,0,0)

        return HookResult(outputVariables = mapOf(finalVal.name() to listOf(finalVal)),
            proceedWithInit = false)


    }



}