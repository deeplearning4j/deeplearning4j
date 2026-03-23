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
package org.nd4j.samediff.frameworkimport.tensorflow.definitions.implementations

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
 * PreImportHook for TensorFlow GatedDeltaRule / GatedDeltaNet / DeltaNet operations.
 *
 * Maps TF custom ops to our fused gated_delta_rule op.
 *
 * Inputs:
 *   0: Q [B, L, H, D_k]
 *   1: K [B, L, H, D_k]
 *   2: V [B, L, H, D_v]
 *   3: beta [B, L, H]
 *   4: gate [B, L, H]
 *   5: state_in [B, H, D_k, D_v] (optional)
 *
 * Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GatedDeltaRule", "GatedDeltaNet", "DeltaNet"], frameworkName = "tensorflow")
class GatedDeltaRule : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val inputs = op.inputsToOp
        val q = sd.getVariable(inputs[0])
        val k = sd.getVariable(inputs[1])
        val v = sd.getVariable(inputs[2])
        val beta = sd.getVariable(inputs[3])
        val gate = sd.getVariable(inputs[4])

        val gdnOp = if (inputs.size > 5) {
            val stateIn = sd.getVariable(inputs[5])
            org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd, q, k, v, beta, gate, stateIn)
        } else {
            org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule(sd, q, k, v, beta, gate)
        }

        val results = sd.doCustomOp(gdnOp)

        val resultMap = mutableMapOf<String, List<SDVariable>>()
        if (outputNames.isNotEmpty()) {
            val output = results[0].rename(outputNames[0])
            resultMap[output.name()] = listOf(output)
        }
        if (outputNames.size > 1) {
            val stateOut = results[1].rename(outputNames[1])
            resultMap[stateOut.name()] = listOf(stateOut)
        }

        return resultMap
    }
}
