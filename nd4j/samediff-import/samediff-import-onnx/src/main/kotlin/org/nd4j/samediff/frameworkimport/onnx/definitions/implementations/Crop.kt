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
 * Implementation of Microsoft ONNX Crop operation.
 *
 * Crop extracts a region from the input tensor based on border values
 * or scale parameters.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#Crop
 *
 * Inputs:
 * - input: Input tensor (NCHW format)
 *
 * Attributes:
 * - border: Border values [left, top, right, bottom] for cropping
 * - scale: Scale for output size [height, width]
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Crop"], frameworkName = "onnx")
class Crop : PreImportHook {

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

        // Get border attribute [left, top, right, bottom]
        val border = getIntListAttribute(attributes, "border", listOf(0, 0, 0, 0))

        // Get optional scale attribute [height, width]
        val scale = getIntListAttribute(attributes, "scale", listOf())

        val left = border[0]
        val top = border[1]
        val right = if (border.size > 2) border[2] else 0
        val bottom = if (border.size > 3) border[3] else 0

        // Input shape is NCHW
        // Calculate slicing parameters
        val inputShape = input.shape

        // Calculate output height and width
        val outputH = if (scale.isNotEmpty()) scale[0] else -1
        val outputW = if (scale.size > 1) scale[1] else -1

        // Use strided slice to crop
        // For NCHW format: [N, C, H, W]
        val begins = longArrayOf(0, 0, top.toLong(), left.toLong())
        val ends = longArrayOf(
            Long.MAX_VALUE,  // N - keep all
            Long.MAX_VALUE,  // C - keep all
            if (outputH > 0) (top + outputH).toLong() else -bottom.toLong() - 1,
            if (outputW > 0) (left + outputW).toLong() else -right.toLong() - 1
        )
        val strides = longArrayOf(1, 1, 1, 1)

        val result = sd.stridedSlice(outputNames[0], input, begins, ends, *strides)

        return mapOf(outputNames[0] to listOf(result))
    }

    private fun getIntListAttribute(attributes: Map<String, Any>, name: String, default: List<Int>): List<Int> {
        val value = attributes[name] ?: return default
        return when (value) {
            is List<*> -> value.map { (it as Number).toInt() }
            is LongArray -> value.map { it.toInt() }
            is IntArray -> value.toList()
            else -> default
        }
    }
}
