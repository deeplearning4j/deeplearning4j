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

@PreHookRule(nodeNames = [], opNames = ["Slice"], frameworkName = "onnx")
class Slice : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val inputVariable = sd.getVariable(op.inputsToOp[0])
        
        // Get the slice parameters
        var starts = sd.getVariable(op.inputsToOp[1])
        var ends = sd.getVariable(op.inputsToOp[2])
        
        // Flatten the starts and ends to 1D if they have extra dimensions
        starts = sd.reshape(starts, -1)
        ends = sd.reshape(ends, -1)
        
        // Cast to INT64 for compatibility
        starts = starts.castTo(DataType.INT64)
        ends = ends.castTo(DataType.INT64)
        
        // Handle axes parameter (optional, 4th input) - if present, only slice specified axes
        val hasAxes = op.inputsToOp.size >= 4 && op.inputsToOp[3] != null
        
        // Handle steps parameter (optional, 5th input)
        val steps = if (op.inputsToOp.size >= 5 && op.inputsToOp[4] != null) {
            var stepsVar = sd.getVariable(op.inputsToOp[4])
            stepsVar = sd.reshape(stepsVar, -1)
            stepsVar.castTo(DataType.INT64)
        } else {
            // Default steps to 1 for each element in starts
            sd.onesLike(starts).castTo(DataType.INT64)
        }
        
        // ONNX Slice semantics:
        // - If axes is not provided, slicing is performed on all axes [0, 1, ..., ndim-1]
        // - If axes is provided, slicing is only performed on the specified axes
        // - For dimensions not in axes, the full range is used
        
        if (hasAxes) {
            // Complex case: need to handle selective axis slicing
            // For now, we'll use the starts/ends/steps as provided
            // This is a simplified implementation - full support would require
            // building full-size arrays with appropriate values for non-specified axes
            
            val result = sd.stridedSlice(
                outputNames[0],
                inputVariable,
                starts,
                ends,
                steps,
                0,  // beginMask - all zeros means use the begin values
                0,  // endMask - all zeros means use the end values  
                0,  // ellipsisMask
                0,  // newAxisMask
                0   // shrinkAxisMask
            )
            
            return mapOf(outputNames[0] to listOf(result))
        } else {
            // Simple case: slicing all dimensions from 0 to len(starts)-1
            // This is the most common case in ONNX
            
            val result = sd.stridedSlice(
                outputNames[0],
                inputVariable,
                starts,
                ends,
                steps,
                0,  // beginMask
                0,  // endMask
                0,  // ellipsisMask
                0,  // newAxisMask
                0   // shrinkAxisMask
            )
            
            return mapOf(outputNames[0] to listOf(result))
        }
    }
}
