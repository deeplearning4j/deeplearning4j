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

import onnx.Onnx
import org.nd4j.autodiff.samediff.SDIndex
import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.autodiff.samediff.internal.SameDiffOp
import org.nd4j.ir.OpNamespace
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.HookResult
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.onnx.ir.OnnxIRDataType
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of split.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/split.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Split"],frameworkName = "onnx")
class Split : PreImportHook  {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val splitDim = if(attributes.containsKey("axis")) {
            attributes["axis"] as Long
        } else {
            0 as Long
        }

        if(op.inputsToOp.size > 1) {
            val split = sd.getVariable(op.inputsToOp[1])
            val splitOutput = sd.split(inputVariable,split,splitDim.toInt())
            return mapOf(splitOutput[0].name() to splitOutput.toList())
        } else if(attributes.containsKey("split")) {
            val numSplits = attributes["split"] as List<Long>
            val splitOutput = sd.split(inputVariable,numSplits[0].toInt(),splitDim.toInt())
            return mapOf(splitOutput[0].name() to splitOutput.toList())
        } else {
            val inputShape = sd.shape(inputVariable)
            val numSplits = inputShape.get(SDIndex.point(splitDim.toLong())).div(outputNames.size.toDouble()).castTo(
                DataType.INT64)
            val splitOutput = sd.split(inputVariable,numSplits,splitDim.toInt())
            val retMap = mutableMapOf<String,List<SDVariable>>()
            splitOutput.toList().forEach { retMap[it.name()] = listOf(it) }
            return retMap
        }
    }


}