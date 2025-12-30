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
 * Implementation of ONNX RandomUniformLike operation.
 *
 * ONNX RandomUniformLike spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomUniformLike
 *
 * Generate a tensor with random values drawn from a uniform distribution.
 * The shape and type of the output are the same as the input tensor.
 *
 * Inputs:
 * - input: Input tensor (only shape is used)
 *
 * Attributes:
 * - dtype: Data type (optional, defaults to input dtype)
 * - high: Upper bound of uniform distribution (default: 1.0)
 * - low: Lower bound of uniform distribution (default: 0.0)
 * - seed: Random seed (optional)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["RandomUniformLike"], frameworkName = "onnx")
class RandomUniformLike : PreImportHook {

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

        // Get input (only shape is used)
        val input = sd.getVariable(op.inputsToOp[0])

        // Get attributes
        val low = (attributes.getOrDefault("low", 0.0) as Number).toDouble()
        val high = (attributes.getOrDefault("high", 1.0) as Number).toDouble()

        // Get dtype - if not specified, use input dtype
        val dtypeAttr = attributes["dtype"]
        val dtype = if (dtypeAttr != null) {
            val dtypeInt = (dtypeAttr as Number).toInt()
            onnxDtypeToNd4j(dtypeInt)
        } else {
            input.dataType()
        }

        // Get shape of input
        val inputShape = sd.shape(input)

        // Generate uniform random in [0, 1) and transform to [low, high)
        // output = low + (high - low) * uniform
        val uniform = sd.random.uniform("${opName}_uniform", low, high, dtype, inputShape)

        val output = uniform.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(output))
    }

    private fun onnxDtypeToNd4j(onnxDtype: Int): DataType {
        return when (onnxDtype) {
            1 -> DataType.FLOAT
            10 -> DataType.FLOAT16
            11 -> DataType.DOUBLE
            14 -> DataType.BFLOAT16
            else -> DataType.FLOAT
        }
    }
}
