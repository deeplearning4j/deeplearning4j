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
 * Implementation of ONNX FeatureVectorizer operation (ML domain).
 *
 * ONNX FeatureVectorizer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.FeatureVectorizer
 *
 * Concatenates input tensors into a single output tensor along axis 1.
 * Each input tensor is flattened to 2D [N, C] before concatenation.
 *
 * Inputs:
 * - X (variadic): One or more input tensors
 *
 * Attributes:
 * - inputdimensions: The size of each input feature
 *
 * Output:
 * - Y: Concatenated tensor [N, total_features]
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["FeatureVectorizer"], frameworkName = "onnx")
class FeatureVectorizer : PreImportHook {

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

        // Get all inputs
        val inputs = op.inputsToOp.filter { it.isNotEmpty() }.map { sd.getVariable(it) }

        if (inputs.isEmpty()) {
            throw IllegalArgumentException("FeatureVectorizer requires at least one input")
        }

        // Get input dimensions attribute (optional)
        val inputDimensions = getLongArray(attributes, "inputdimensions")

        // Process each input: flatten to 2D if necessary
        val processedInputs = inputs.mapIndexed { idx, input ->
            // Get the expected dimension for this input if specified
            val expectedDim = inputDimensions?.getOrNull(idx)

            // Flatten to 2D: [N, features]
            val flatInput = if (input.shape.size > 2) {
                // Reshape from [N, ...] to [N, -1]
                val batchSize = input.shape[0]
                sd.reshape("${opName}_flat_$idx", input, batchSize, -1)
            } else if (input.shape.size == 1) {
                // Expand dims to make it [N, 1] or [1, features]
                sd.expandDims("${opName}_expand_$idx", input, 0)
            } else {
                input
            }

            // Cast to float for consistency
            flatInput.castTo("${opName}_cast_$idx", DataType.FLOAT)
        }

        // Concatenate all inputs along axis 1 (feature dimension)
        val output = if (processedInputs.size == 1) {
            processedInputs[0]
        } else {
            sd.concat(outputNames[0], 1, *processedInputs.toTypedArray())
        }

        val finalOutput = if (output.name() != outputNames[0]) {
            output.rename(outputNames[0])
        } else {
            output
        }

        return mapOf(outputNames[0] to listOf(finalOutput))
    }

    private fun getLongArray(attributes: Map<String, Any>, key: String): LongArray? {
        val value = attributes[key] ?: return null
        return when (value) {
            is LongArray -> value
            is IntArray -> value.map { it.toLong() }.toLongArray()
            is List<*> -> value.map { (it as Number).toLong() }.toLongArray()
            else -> null
        }
    }
}
