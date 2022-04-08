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
import org.nd4j.linalg.api.ops.impl.shape.tensorops.TensorArray
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * A port of sequence_insert.py from onnx tensorflow for samediff:
 * https://github.com/onnx/onnx-tensorflow/blob/master/onnx_tf/handlers/backend/sequence_insert.py
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [],opNames = ["SequenceInsert"],frameworkName = "onnx")
class SequenceInsert : PreImportHook  {


    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val input = sd.getVariable(op.inputsToOp[0])
        val tensorToInsert = sd.getVariable(op.inputsToOp[1])
        val position = if(op.inputsToOp.size < 3) sd.constant(-1) else {
            sd.getVariable(op.inputsToOp[2])
        }

        val tensorArrayOp = sd.tensorArray(input)
        val written = tensorArrayOp.write(input,position,tensorToInsert)
        written.addControlDependency(position)
        written.addControlDependency(input)
        var outputVars = written
        outputVars = sd.updateVariableNameAndReference(outputVars,outputNames[0])
        val ret = mutableMapOf<String,List<SDVariable>>()
        ret[outputNames[0]] = listOf(outputVars)
        return ret
    }


    fun checkPositionInBounds(sd: SameDiff,inputSequence: SDVariable,position: SDVariable): SDVariable {
        val seqLength = inputSequence.shape().prod(0).castTo(position.dataType())
        val cond1 = sd.gte(position,sd.math().neg(seqLength))
        val cond2 = sd.lte(position,seqLength)
        return sd.all(sd.bitwise().and(cond1,cond2),0)
    }

}