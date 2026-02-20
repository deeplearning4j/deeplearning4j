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
 * Implementation of ONNX HardSwish operation.
 *
 * ONNX HardSwish spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#HardSwish
 *
 * HardSwish activation function:
 * hardswish(x) = x * max(0, min(1, (x + 3) / 6))
 *             = x * clip((x + 3) / 6, 0, 1)
 *
 * This is a computationally efficient approximation of Swish.
 *
 * Inputs:
 * - X: Input tensor
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["HardSwish"], frameworkName = "onnx")
class HardSwish : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val opName = op.name

        // Get input
        val x = sd.getVariable(op.inputsToOp[0])

        // HardSwish: x * clip((x + 3) / 6, 0, 1)
        val xPlus3 = sd.math.add(x, 3.0)
        val divided = sd.math.div(xPlus3, 6.0)
        val clipped = sd.math.clipByValue("${opName}_clip", divided, 0.0, 1.0)
        val output = sd.math.mul("${opName}_hardswish", x, clipped)

        val finalOutput = output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
    }
}
