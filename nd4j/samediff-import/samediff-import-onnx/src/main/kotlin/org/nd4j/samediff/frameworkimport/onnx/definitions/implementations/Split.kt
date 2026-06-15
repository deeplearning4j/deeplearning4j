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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.guava.primitives.Ints
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
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val splitDim = if(attributes.containsKey("axis")) {
            attributes["axis"] as Long
        } else {
            0 as Long
        }

        if(op.inputsToOp.size > 1) {
            val split = sd.getVariable(op.inputsToOp[1])
            // Try to get split sizes from constant array, or use outputNames.size
            val splitArr = split.arr
            if (splitArr != null) {
                // Split sizes are known - use splitV with explicit num_split
                val splitOutput = sd.splitV(outputNames.toTypedArray(), inputVariable, split, splitArr.length().toInt(), splitDim.toInt())
                return retOutput(splitOutput)
            } else {
                // Split sizes are dynamic - use outputNames.size as num_split
                val splitOutput = sd.splitV(outputNames.toTypedArray(), inputVariable, split, outputNames.size, splitDim.toInt())
                return retOutput(splitOutput)
            }
        } else if(attributes.containsKey("split")) {
            val numSplits = attributes["split"] as List<Long>
            // Use createFromArray to ensure 1D shape, not [1, N] row vector
            val splitConst = sd.constant(Nd4j.createFromArray(*numSplits.toLongArray()))
            val splitOutput = sd.splitV(outputNames.toTypedArray(),inputVariable,splitConst,numSplits.size,splitDim.toInt())
            return retOutput(splitOutput)
        } else {
            // No split sizes given - split evenly based on number of outputs
            val numOutputs = outputNames.size
            // Create equal split sizes array
            val splitSizes = sd.constant(Nd4j.ones(DataType.INT64, numOutputs.toLong()))
            val splitOutput = sd.splitV(outputNames.toTypedArray(), inputVariable, splitSizes, numOutputs, splitDim.toInt())
            val retMap = mutableMapOf<String,List<SDVariable>>()
            splitOutput.toList().forEach { retMap[it.name()] = listOf(it) }
            return retMap
        }
    }

    fun retOutput(vars: Array<SDVariable>): Map<String,List<SDVariable>> {
        val ret = HashMap<String,List<SDVariable>>()
        for(sdVar in vars) {
            ret[sdVar.name()] = listOf(sdVar)
        }

        return ret
    }


}