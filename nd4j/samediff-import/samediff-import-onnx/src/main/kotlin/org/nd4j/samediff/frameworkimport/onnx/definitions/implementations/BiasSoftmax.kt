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
 * Implementation of Microsoft ONNX BiasSoftmax operation.
 *
 * BiasSoftmax combines bias addition with softmax:
 * output = softmax(input + bias, axis)
 *
 * This is commonly used in attention mechanisms where a bias
 * (such as attention mask) is added before softmax.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#BiasSoftmax
 *
 * Inputs:
 * - data: Input tensor
 * - bias: Bias tensor to add before softmax
 *
 * Attributes:
 * - axis: The axis along which to compute softmax (default: 1)
 * - is_inner_broadcast: If true, bias is broadcast as [1, 1, ..., bias_dim]
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["BiasSoftmax"], frameworkName = "onnx")
class BiasSoftmax : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val data = sd.getVariable(op.inputsToOp[0])
        val bias = sd.getVariable(op.inputsToOp[1])

        // Get axis attribute (default 1)
        val axis = (attributes.getOrDefault("axis", 1) as Number).toInt()

        // Add bias to data
        val biasedData = sd.math.add(data, bias)

        // Apply softmax along the specified axis
        val result = sd.nn.softmax(outputNames[0], biasedData, axis)

        return mapOf(outputNames[0] to listOf(result))
    }
}
