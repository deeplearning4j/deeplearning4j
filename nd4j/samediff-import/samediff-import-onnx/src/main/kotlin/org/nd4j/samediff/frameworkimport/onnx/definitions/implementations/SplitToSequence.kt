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
 * Implementation of ONNX SplitToSequence operation.
 *
 * ONNX SplitToSequence spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#SplitToSequence
 *
 * Splits a tensor into a sequence of tensors along a specified axis.
 *
 * Inputs:
 * - input: Tensor to split
 * - split: Optional sizes for each output (if not provided, splits into equal parts)
 *
 * Attributes:
 * - axis: Axis along which to split (default: 0)
 * - keepdims: If 1, keep the dimension; if 0, squeeze it out (default: 1)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["SplitToSequence"], frameworkName = "onnx")
class SplitToSequence : PreImportHook {

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
        val split = if (op.inputsToOp.size > 1) sd.getVariable(op.inputsToOp[1]) else null
        
        // Get attributes
        val axis = (attributes.getOrDefault("axis", 0L) as Number).toInt()
        val keepdims = (attributes.getOrDefault("keepdims", 1L) as Number).toInt()
        
        // Perform split
        val outputs = if (split != null) {
            // Split with specified sizes - use splitV without names
            sd.splitV(input, split, outputNames.size, axis)
        } else {
            // Split into equal parts (one element per split)
            // For dynamic split, we need to determine numSplits at runtime
            // Use a reasonable default or extract from shape if possible
            val inputShape = input.shape
            val numSplits = if (inputShape != null && axis < inputShape.size) {
                inputShape[axis].toInt()
            } else {
                1 // fallback
            }
            sd.split(input, numSplits, axis)
        }
        
        // If keepdims is 0, squeeze the split dimension
        val finalOutputs = if (keepdims == 0) {
            outputs.mapIndexed { i, v ->
                sd.squeeze("${opName}_squeeze_$i", v, axis)
            }
        } else {
            outputs.toList()
        }
        
        // Create output map - sequence is represented as multiple outputs
        val resultMap = mutableMapOf<String, List<SDVariable>>()
        if (outputNames.isNotEmpty()) {
            resultMap[outputNames[0]] = finalOutputs
        }
        
        return resultMap
    }
}
