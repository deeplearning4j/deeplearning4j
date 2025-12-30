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
 * Implementation of ONNX CastMap operation (ML domain).
 *
 * ONNX CastMap spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.CastMap
 *
 * Converts a map to a tensor. The output is determined by cast_to attribute.
 * Since SameDiff doesn't have native map support, this implementation
 * treats the input as a tensor and casts to the target type.
 *
 * Inputs:
 * - X: Input map (represented as tensor in SameDiff)
 *
 * Attributes:
 * - cast_to: Target type ('TO_FLOAT', 'TO_STRING', 'TO_INT64')
 * - map_form: Form of the map ('DENSE', 'SPARSE')
 * - max_map: Maximum map size for sparse representation
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["CastMap"], frameworkName = "onnx")
class CastMap : PreImportHook {

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

        // Get cast_to attribute (default: TO_FLOAT)
        val castTo = attributes.getOrDefault("cast_to", "TO_FLOAT") as? String ?: "TO_FLOAT"

        // Determine target data type
        val targetType = when (castTo.uppercase()) {
            "TO_FLOAT" -> DataType.FLOAT
            "TO_INT64" -> DataType.INT64
            "TO_STRING" -> DataType.UTF8
            else -> DataType.FLOAT
        }

        // Cast the input to target type
        val output = if (targetType == DataType.UTF8) {
            // String casting is not directly supported, pass through as float
            sd.identity("${opName}_identity", x).castTo(DataType.FLOAT)
        } else {
            x.castTo("${opName}_cast", targetType)
        }

        val finalOutput = output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
    }
}
