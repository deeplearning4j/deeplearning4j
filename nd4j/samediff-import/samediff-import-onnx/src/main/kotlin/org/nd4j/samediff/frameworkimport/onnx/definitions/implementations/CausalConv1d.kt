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
 * PreImportHook for ONNX CausalConv1D operations.
 *
 * Maps ONNX CausalConv1D (custom domain or decomposed) to our causal_conv1d op.
 *
 * Inputs:
 *   0: x [B, L, D]
 *   1: weight [D, K]
 *   2: bias [D] (optional)
 *   3: state_in [B, D, K-1] (optional)
 *
 * Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["CausalConv1D", "CausalConv1d"], frameworkName = "onnx")
class CausalConv1d : PreImportHook {

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
        val x = sd.getVariable(inputs[0])
        val weight = sd.getVariable(inputs[1])

        // Parse activation attribute (0=none, 1=silu)
        val activation = when (val actAttr = attributes["activation"]) {
            is Number -> actAttr.toInt()
            is String -> if (actAttr.lowercase() == "silu") 1 else 0
            else -> 0
        }

        val bias = if (inputs.size > 2) sd.getVariable(inputs[2]) else null
        val stateIn = if (inputs.size > 3) sd.getVariable(inputs[3]) else null

        val convOp = org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d(
            sd, x, weight, bias, stateIn, activation
        )

        val results = sd.doCustomOp(convOp)

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
