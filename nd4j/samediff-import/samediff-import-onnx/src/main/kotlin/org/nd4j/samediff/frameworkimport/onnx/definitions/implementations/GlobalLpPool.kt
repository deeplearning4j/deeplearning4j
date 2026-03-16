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
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX GlobalLpPool operation.
 *
 * ONNX GlobalLpPool spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalLpPool
 *
 * Applies Lp pooling across the entire spatial dimensions of the input tensor.
 * For input of shape [N, C, H, W], output is [N, C, 1, 1].
 *
 * Inputs:
 * - X: Input tensor of shape [N, C, D1, D2, ..., Dn]
 *
 * Attributes:
 * - p: Exponent of Lp norm (default: 2)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GlobalLpPool"], frameworkName = "onnx")
class GlobalLpPool : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])

        // Get p attribute (default: 2)
        val p = (attributes.getOrDefault("p", 2L) as Number).toInt()

        // For 4D input [N, C, H, W], reduce over dimensions 2 and 3 (H and W)
        val inputShape = input.shape ?: throw IllegalStateException("GlobalLpPool requires static input shape")
        val rank = inputShape.size
        
        // Compute Lp norm: (sum(|x|^p))^(1/p) over spatial dimensions
        // GlobalLpPool computes: (Σ|x_i|^p)^(1/p) over spatial dimensions
        val output = if (p == 1) {
            // L1: sum of absolute values over spatial dims
            sd.math.sum(outputNames[0], sd.math.abs(input), true, 2, 3)
        } else if (p == 2) {
            // L2: sqrt of sum of squares
            sd.math.sqrt(outputNames[0], sd.math.sum(sd.math.square(input), true, 2, 3))
        } else {
            // General Lp: (sum(|x|^p))^(1/p)
            val absInput = sd.math.abs(input)
            val powered = sd.math.pow("${opName}_pow", absInput, p.toDouble())
            val sumPow = sd.math.sum("${opName}_sum", powered, true, 2, 3)
            sd.math.pow(outputNames[0], sumPow, 1.0 / p)
        }

        return mapOf(outputNames[0] to listOf(output))
    }
}
