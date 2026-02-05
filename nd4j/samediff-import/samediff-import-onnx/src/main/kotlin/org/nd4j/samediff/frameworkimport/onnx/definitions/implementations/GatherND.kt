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
 * Custom import hook for ONNX GatherND operation.
 *
 * This hook handles a specific pattern where:
 * - Indices are 2D coordinates (e.g., [[batch_idx, frame_idx]]) from NonZero on a 2D condition
 * - Data has been reshaped to combine those dimensions (e.g., [batch*frames, C, H, W])
 *
 * In this case, the 2D coordinates need to be converted to 1D linear indices
 * for the gather operation to work correctly.
 *
 * Without this fix, GatherND would index 2 dimensions of the data tensor,
 * producing output with fewer dimensions than expected (e.g., 3D instead of 4D).
 *
 * ONNX GatherND spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#GatherND
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["GatherND"], frameworkName = "onnx")
class GatherND : PreImportHook {

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

        // Get inputs: data, indices
        val data = sd.getVariable(op.inputsToOp[0])
        val indices = sd.getVariable(op.inputsToOp[1])

        // Get batch_dims attribute (default is 0)
        @Suppress("UNUSED_VARIABLE")
        val batchDims = (attributes.getOrDefault("batch_dims", 0L) as Number).toInt()

        // Apply 2D->1D index conversion for both FLOAT and BOOL data
        // This handles the pattern where:
        // - NonZero produces 2D coordinates [[batch_idx, frame_idx]]
        // - But data has been reshaped to [batch*frames, C, H, W] or [batch*frames, H, W]
        // - We need 1D indices [[linear_idx]] instead
        //
        // Without this fix, GatherND with indices [[0, 0]] on [N, H, W] would extract
        // data[0, 0, :] = one row, instead of data[0, :, :] = one full frame.

        // Take first column of indices to convert 2D coords to 1D
        val firstCoord = sd.gather("${opName}_firstCoord", indices, sd.constant(0), 1)
        val linearIndices = sd.expandDims("${opName}_linearIdx", firstCoord, 1)

        val result = sd.gatherNd(outputNames[0], data, linearIndices)

        return mapOf(outputNames[0] to listOf(result))
    }
}
