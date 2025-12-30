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
 * Implementation of ONNX MeanVarianceNormalization operation.
 *
 * ONNX MeanVarianceNormalization spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MeanVarianceNormalization
 *
 * A MeanVarianceNormalization Function: Perform mean variance normalization on the input tensor X.
 * output = (input - mean) / sqrt(variance + epsilon)
 *
 * By default, mean and variance are computed across channels and spatial dims (axes=[0,2,3]).
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - axes: Axes along which to compute mean and variance (default: [0, 2, 3])
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["MeanVarianceNormalization"], frameworkName = "onnx")
class MeanVarianceNormalization : PreImportHook {

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

        // Get axes attribute (default: [0, 2, 3] for NCHW format - normalize over batch and spatial)
        val axesAttr = attributes["axes"]
        val axes: IntArray = when (axesAttr) {
            is List<*> -> axesAttr.map { (it as Number).toInt() }.toIntArray()
            is LongArray -> axesAttr.map { it.toInt() }.toIntArray()
            else -> intArrayOf(0, 2, 3) // Default axes
        }

        // Small epsilon for numerical stability
        val epsilon = 1e-9

        // Compute mean along specified axes (keep dims for broadcasting)
        val mean = sd.mean("${opName}_mean", input, true, *axes)

        // Compute variance along specified axes
        val variance = sd.variance("${opName}_var", input, false, true, *axes)

        // Normalize: (x - mean) / sqrt(variance + epsilon)
        val centered = sd.math.sub("${opName}_centered", input, mean)
        val stddev = sd.math.sqrt("${opName}_std", variance.add(epsilon))
        val output = sd.math.div(outputNames[0], centered, stddev)

        return mapOf(outputNames[0] to listOf(output))
    }
}
