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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Microsoft ONNX ImageScaler operation.
 *
 * ImageScaler applies scaling and bias to an image tensor:
 * output = scale * input + bias
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#ImageScaler
 *
 * Inputs:
 * - input: Input tensor (NCHW format)
 *
 * Attributes:
 * - scale: Scaling factor (default: 1.0)
 * - bias: Per-channel bias values (default: [0, 0, 0, ...])
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ImageScaler"], frameworkName = "onnx")
class ImageScaler : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val input = sd.getVariable(op.inputsToOp[0])

        // Get scale attribute (default 1.0)
        val scale = (attributes.getOrDefault("scale", 1.0) as Number).toDouble()

        // Get bias attribute (per-channel bias)
        val biasValues = getFloatListAttribute(attributes, "bias", listOf())

        // Apply scaling: output = scale * input
        var result = if (scale != 1.0) {
            sd.math.mul(input, scale)
        } else {
            input
        }

        // Apply per-channel bias if provided
        if (biasValues.isNotEmpty()) {
            // Bias is per-channel, needs to be reshaped for broadcasting
            // Input is NCHW, so bias should be [1, C, 1, 1]
            val biasArray = Nd4j.createFromArray(*biasValues.toFloatArray())
                .reshape(1, biasValues.size.toLong(), 1, 1)
            val bias = sd.constant("${op.name}_bias", biasArray)
            result = sd.math.add(result, bias)
        }

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }

    private fun getFloatListAttribute(attributes: Map<String, Any>, name: String, default: List<Float>): List<Float> {
        val value = attributes[name] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toFloat() }
            is FloatArray -> value.toList()
            is DoubleArray -> value.map { it.toFloat() }
            else -> default
        }
    }
}
