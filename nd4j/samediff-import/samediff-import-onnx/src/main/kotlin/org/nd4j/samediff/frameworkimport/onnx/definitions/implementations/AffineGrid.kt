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
 * Implementation of ONNX AffineGrid operation using native C++ op.
 *
 * AffineGrid generates a 2D or 3D flow field (sampling grid) given a batch
 * of affine transformation matrices theta. The grid can be used with
 * GridSample to perform spatial transformations.
 *
 * This is commonly used in:
 * - Spatial transformer networks
 * - Image registration
 * - Data augmentation
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#AffineGrid
 *
 * Inputs:
 * - theta: Affine transformation matrix [N, 2, 3] for 2D or [N, 3, 4] for 3D
 * - size: Target output size [N, C, H, W] for 2D or [N, C, D, H, W] for 3D
 *
 * Attributes:
 * - align_corners: If true, consider -1 and 1 to refer to the centers of
 *   corner pixels. Default is 0 (false).
 *
 * Output:
 * - grid: Sampling grid [N, H, W, 2] for 2D or [N, D, H, W, 3] for 3D
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["AffineGrid"], frameworkName = "onnx")
class AffineGrid : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val theta = sd.getVariable(op.inputsToOp[0])
        val size = sd.getVariable(op.inputsToOp[1])

        // Get align_corners attribute
        val alignCorners = (attributes.getOrDefault("align_corners", 0L) as Number).toInt() != 0

        // Use native affine_grid op
        val output = sd.image().affineGrid(outputNames[0], theta, size, alignCorners)

        return mapOf(outputNames[0] to listOf(output))
    }
}
