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
 * Implementation of ONNX RandomNormalLike operation.
 *
 * ONNX RandomNormalLike spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#RandomNormalLike
 *
 * Generate a tensor with random values drawn from a normal distribution.
 * The shape and type of the output are the same as the input tensor.
 *
 * Inputs:
 * - input: Input tensor (only shape is used)
 *
 * Attributes:
 * - dtype: Data type (optional, defaults to input dtype)
 * - mean: Mean of the normal distribution (default: 0.0)
 * - scale: Standard deviation (default: 1.0)
 * - seed: Random seed (optional)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["RandomNormalLike"], frameworkName = "onnx")
class RandomNormalLike : PreImportHook {

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
        val mean = (attributes.getOrDefault("mean", 0.0) as Number).toDouble()
        val scale = (attributes.getOrDefault("scale", 1.0) as Number).toDouble()

        // Get dtype - if not specified, use input dtype
        val dtypeAttr = attributes["dtype"]
        val dtype = if (dtypeAttr != null) {
            val dtypeInt = (dtypeAttr as Number).toInt()
            onnxDtypeToNd4j(dtypeInt)
        } else {
            input.dataType()
        }

        // Get shape of input - for RandomNormalLike we need a static shape
        // Since sd.random.normal expects long... shape, we'll use the input's static shape
        val inputShapeArr = input.shape ?: throw IllegalStateException("RandomNormalLike requires static input shape")

        // Generate normal distribution with specified mean and scale (stddev)
        val output = sd.random.normal("${opName}_normal", mean, scale, dtype, *inputShapeArr)
        output.rename(outputNames[0])

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
