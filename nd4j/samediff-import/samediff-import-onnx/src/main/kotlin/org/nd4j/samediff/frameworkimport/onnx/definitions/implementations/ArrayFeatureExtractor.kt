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
 * Implementation of ONNX ArrayFeatureExtractor operation (ML domain).
 *
 * ONNX ArrayFeatureExtractor spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.ArrayFeatureExtractor
 *
 * Selects elements from the last axis of the input tensor based on the indices.
 * This is commonly used to select specific features from a feature vector.
 *
 * Inputs:
 * - X: Input tensor of shape [N, ..., F]
 * - Y: Indices tensor specifying which features to extract
 *
 * Output:
 * - Z: Selected features from input
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ArrayFeatureExtractor"], frameworkName = "onnx")
class ArrayFeatureExtractor : PreImportHook {

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

        // Get inputs
        val x = sd.getVariable(op.inputsToOp[0])  // Input features
        val y = sd.getVariable(op.inputsToOp[1])  // Indices to select

        // ArrayFeatureExtractor selects from the last axis
        // For this operation, we use axis -1 to always select from the last axis
        // This works dynamically regardless of input rank

        // Ensure indices are INT64 for gather
        val indices = y.castTo(DataType.INT64)

        // Use gather to select features along the last axis (axis = -1)
        val output = sd.gather(outputNames[0], x, indices, -1)

        return mapOf(outputNames[0] to listOf(output))
    }
}
