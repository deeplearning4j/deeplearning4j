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

import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.common.util.ArrayUtil
import org.nd4j.enums.ImageResizeMethod
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import java.lang.IllegalArgumentException

/**
 * A port of resize.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/resize.py#L195
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
        isFinalOutput: Boolean
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
        val isAxesNegative = sd.lt("isAxesNegative",axes,sd.zerosLike(axes))
        val axesWhere = sd.where("axesWhere",axes.add(inputRank),axes,isAxesNegative)
        val sparseIndices = sd.castTo("sparseIndices",sd.expandDims(axesWhere,-1),DataType.INT64)
        val sparseShape = sd.gatherNd("sparseShape",sd.shape("inputVariableShape",inputVariable),sparseIndices).castTo(ends.dataType())
        val startsMin = sd.min("startsMin",starts,sparseShape)
        val endsMin = sd.min("endsMin",ends,sparseShape)

        val isStartsNegative = sd.lt("isStartsNegative",startsMin,sd.zerosLike(startsMin))
        val startsFinal = sd.where("startsWhere",startsMin.add("startsMinAdd",sparseShape),startsMin,isStartsNegative)
        val isEndsNegative = sd.lt("isEndsNegative",endsMin,sd.zerosLike("zerosLikeEndsMin",endsMin))
        val endsFinal = sd.where("endWhere",endsMin.add(sparseShape),endsMin,isEndsNegative)
        val outputShape = inputRank.castTo("outputShape",DataType.INT64)
        val denseBegins = sd.sparseToDense("denseBegins",sparseIndices,outputShape,startsFinal)


        val denseEnds = sd.sparseToDense("denseEnds",sparseIndices,outputShape,endsFinal,sd.constant(Nd4j.create(
            floatArrayOf(-1.0f)).castTo(denseBegins.dataType())))
      //TODO: double check when back
       val denseEnds2 = sd.where("denseEnds2",inputTensorShape,denseEnds,sd.eq(denseEnds,sd.constant(-1).castTo(denseBegins.dataType())))

        val denseSteps: SDVariable = if(op.inputsToOp.size >= 5) {
            val inputVar = sd.getVariable(op.inputsToOp[4])
            sd.sparseToDense("denseSteps",sparseIndices,
                outputShape,inputVar,
               sd.constant(Nd4j.create(floatArrayOf(1.0f))
                   .castTo(inputVar.dataType())))
        } else {
            sd.onesLike("denseSteps",inputVariable.shape())
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