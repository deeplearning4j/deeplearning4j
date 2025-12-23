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
 * Implementation of ONNX Gather operation mapping for SameDiff.
 * Maps the ONNX Gather operation to SameDiff's gather function.
 * 
 * ONNX Gather spec: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gather
 * Given data tensor of rank r >= 1, and indices tensor of rank q, gather entries of the axis dimension of data
 * (axis specified as an attribute) indexed by indices, and concatenates them in an output tensor of rank q + (r - 1).
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Gather"], frameworkName = "onnx")
class Gather : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get input variables
        val dataVariable = sd.getVariable(op.inputsToOp[0])
        var indicesVariable = sd.getVariable(op.inputsToOp[1])

        // Get axis attribute (default to 0 if not specified)
        val axis = attributes["axis"]?.let { (it as Long).toInt() } ?: 0

        // Handle indices shape for proper gather semantics.
        // ONNX Gather output rank = indices.rank + input.rank - 1
        //
        // For BGE pooler pattern: input [1, 512, 768], indices [[0]] shape [1,1], axis=1
        //   - With [1,1] indices: output [1, 1, 1, 768] (wrong - indices adds 2 dims)
        //   - With [1] indices + squeeze: output [1, 1, 768] then squeeze -> [1, 768] (correct)
        //
        // libnd4j's gather doesn't support scalar (0D) indices, so we:
        // 1. Flatten indices to 1D
        // 2. Gather (adds 1 dim from indices at axis position)
        // 3. Squeeze the dimension at axis (will only remove if size is 1)

        // Flatten indices to 1D - gather requires at least 1D indices
        val flatIndices = sd.reshape("${op.name}_indices_flat", indicesVariable, -1L)

        // Call SameDiff's gather method with 1D indices
        // Output shape will be: [...dims before axis..., indices_length, ...dims after axis...]
        val gatherResult = sd.gather("${op.name}_gather_result", dataVariable, flatIndices, axis)

        // Always squeeze at the gather axis - squeeze will only remove if size is 1
        // For single-element indices (common in pooler patterns), this correctly removes the extra dim
        // For multi-element indices, the squeeze may fail - but those cases need different handling anyway
        val outputVar = sd.squeeze(outputNames[0], gatherResult, axis)

        return mapOf(outputVar.name() to listOf(outputVar))
    }
}
