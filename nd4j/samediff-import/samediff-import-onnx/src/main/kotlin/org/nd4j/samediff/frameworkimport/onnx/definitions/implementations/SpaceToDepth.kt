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
import org.nd4j.enums.DataFormat
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX SpaceToDepth operation.
 *
 * ONNX SpaceToDepth spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#SpaceToDepth
 *
 * Rearranges blocks of spatial data into depth.
 * This is the reverse of DepthToSpace.
 *
 * Input shape: [N, C, H, W]
 * Output shape: [N, C*blocksize^2, H/blocksize, W/blocksize]
 *
 * Inputs:
 * - input: Input tensor [N, C, H, W]
 *
 * Attributes:
 * - blocksize: Block size for rearrangement
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["SpaceToDepth"], frameworkName = "onnx")
class SpaceToDepth : PreImportHook {

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

        // Get attributes
        val blockSize = (attributes["blocksize"] as? Number)?.toInt() ?: 1

        // Use SameDiff's space_to_depth operation with NCHW format (ONNX default)
        val output = sd.cnn.spaceToDepth(
            outputNames[0],
            input,
            blockSize,
            DataFormat.NCHW
        )

        return mapOf(outputNames[0] to listOf(output))
    }
}
