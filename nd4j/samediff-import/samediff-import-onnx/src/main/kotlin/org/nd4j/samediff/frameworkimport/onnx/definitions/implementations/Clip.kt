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
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of clip.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/clip.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["Clip"],frameworkName = "onnx")
class Clip : PreImportHook  {


    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>
    ): Map<String, List<SDVariable>> {
        var inputVariable = sd.getVariable(op.inputsToOp[0])
        val min = if(op.inputsToOp.size > 1) {
            sd.getVariable(op.inputsToOp[1])
        } else {
            sd.minMax(inputVariable.dataType().toInt(),0)
        }

        val max = if(op.inputsToOp.size > 2) {
            sd.getVariable(op.inputsToOp[2])
        } else {
            sd.minMax(inputVariable.dataType().toInt(),1)
        }

        val clipped = sd.clipByValue(inputVariable,min,max)
        val outputVar = sd.castTo(outputNames[0],clipped,inputVariable.dataType())
        return mapOf(outputNames[0] to listOf(outputVar))

    }


}