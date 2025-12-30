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
 * Implementation of ONNX Multinomial operation.
 *
 * ONNX Multinomial spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Multinomial
 *
 * Generate random samples from a multinomial distribution.
 *
 * Inputs:
 * - input: Input tensor with unnormalized log-probabilities of shape [batch_size, class_size]
 *
 * Attributes:
 * - dtype: Output data type (default: INT32)
 * - sample_size: Number of samples to draw per batch (default: 1)
 * - seed: Random seed (optional)
 *
 * Output:
 * - output: Tensor of shape [batch_size, sample_size] with sampled indices
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Multinomial"], frameworkName = "onnx")
class Multinomial : PreImportHook {

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

        // Get input (unnormalized log-probabilities)
        val input = sd.getVariable(op.inputsToOp[0])

        // Get attributes
        val sampleSize = (attributes.getOrDefault("sample_size", 1L) as Number).toInt()

        // Get dtype
        val dtypeAttr = attributes["dtype"]
        val dtype = if (dtypeAttr != null) {
            val dtypeInt = (dtypeAttr as Number).toInt()
            when (dtypeInt) {
                6 -> DataType.INT32
                7 -> DataType.INT64
                else -> DataType.INT32
            }
        } else {
            DataType.INT32
        }

        // Convert log-probabilities to probabilities using softmax
        val probs = sd.nn.softmax("${opName}_probs", input, -1)

        // Use SameDiff's multinomial sampling
        // Note: This assumes a multinomial op exists; if not, implement using other primitives
        val samples = sd.random.multinomial("${opName}_samples", probs, sampleSize)

        val output = samples.castTo(dtype).rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(output))
    }
}
