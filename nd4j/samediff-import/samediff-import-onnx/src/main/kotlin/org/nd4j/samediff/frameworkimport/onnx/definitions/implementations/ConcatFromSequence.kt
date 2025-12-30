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
 * Implementation of ONNX ConcatFromSequence operation.
 *
 * ONNX ConcatFromSequence spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConcatFromSequence
 *
 * Concatenates tensors from a sequence along a specified axis.
 *
 * Inputs:
 * - input_sequence: Sequence of tensors to concatenate
 *
 * Attributes:
 * - axis: Axis along which to concatenate (default: 0)
 * - new_axis: If 1, creates a new axis for stacking instead of concatenating (default: 0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["ConcatFromSequence"], frameworkName = "onnx")
class ConcatFromSequence : PreImportHook {

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
        
        // Get axis attribute
        val axis = (attributes.getOrDefault("axis", 0L) as Number).toInt()
        val newAxis = (attributes.getOrDefault("new_axis", 0L) as Number).toInt()
        
        // Get all inputs from the sequence
        val inputs = op.inputsToOp.map { sd.getVariable(it) }
        
        val output = if (newAxis == 1) {
            // Stack along new axis
            sd.stack(outputNames[0], axis, *inputs.toTypedArray())
        } else {
            // Concatenate along existing axis
            sd.concat(outputNames[0], axis, *inputs.toTypedArray())
        }
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
