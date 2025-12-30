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
 * @author Eclipse Deeplearning4j Development Team
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
        
        // Get spatial dimensions (all dimensions except batch and channel)
        // For [N, C, D1, D2, ..., Dn], spatial dims are [2, 3, ..., rank-1]
        val rankOf = sd.rank(input)
        val range = sd.range(sd.constant(0), rankOf, sd.constant(1), DataType.INT64)
        val sizes = sd.concat(0, sd.constant(2).castTo(DataType.INT64), sd.prod(range.shape()).sub(2.0).castTo(DataType.INT64))
        val split = sd.splitV(range, sizes, 2, 0)
        val spatialDims = split[1]
        
        // Compute Lp norm: (sum(|x|^p))^(1/p)
        val output = if (p == 1) {
            // L1: sum of absolute values
            sd.math.mean(outputNames[0], sd.math.abs(input), spatialDims, true)
        } else if (p == 2) {
            // L2: sqrt of sum of squares (RMS with mean replaced by sum, then normalize)
            val sumSquares = sd.math.sum("${opName}_sumSq", sd.math.square(input), spatialDims, true)
            // Get count of spatial elements for proper Lp pooling
            val inputShape = sd.shape(input)
            val spatialSize = sd.prod(sd.gather(inputShape, spatialDims, 0))
            val avgSquares = sd.math.div("${opName}_avgSq", sumSquares, spatialSize.castTo(sumSquares.dataType()))
            sd.math.sqrt(outputNames[0], avgSquares)
        } else {
            // General Lp: (mean(|x|^p))^(1/p)
            val absInput = sd.math.abs(input)
            val powered = sd.math.pow("${opName}_pow", absInput, p.toDouble())
            val meanPow = sd.math.mean("${opName}_mean", powered, spatialDims, true)
            sd.math.pow(outputNames[0], meanPow, 1.0 / p)
        }
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
