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
 * Implementation of ONNX ReverseSequence operation.
 *
 * ONNX ReverseSequence spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReverseSequence
 *
 * Reverses variable length slices. This operator first slices input tensor along the
 * batch_axis, and for each slice, it reverses the first sequence_lens[i] elements
 * along the time_axis.
 *
 * Inputs:
 * - input: Tensor of rank r >= 2
 * - sequence_lens: 1-D tensor specifying the length of each sequence
 *
 * Attributes:
 * - batch_axis: The axis along which to slice (default: 1)
 * - time_axis: The axis along which to reverse (default: 0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ReverseSequence"], frameworkName = "onnx")
class ReverseSequence : PreImportHook {

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
        val input = sd.getVariable(op.inputsToOp[0])
        val sequenceLens = sd.getVariable(op.inputsToOp[1])

        // Get attributes
        val batchAxis = (attributes.getOrDefault("batch_axis", 1L) as Number).toInt()
        val timeAxis = (attributes.getOrDefault("time_axis", 0L) as Number).toInt()

        // Use SameDiff's reverseSequence operation
        val output = sd.reverseSequence(outputNames[0], input, sequenceLens, timeAxis, batchAxis)

        return mapOf(outputNames[0] to listOf(output))
    }
}
