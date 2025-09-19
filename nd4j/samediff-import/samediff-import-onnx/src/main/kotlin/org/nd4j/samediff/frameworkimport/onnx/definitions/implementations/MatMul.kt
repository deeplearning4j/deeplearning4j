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

import org.nd4j.autodiff.samediff.SDIndex
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

@PreHookRule(nodeNames = [], opNames = ["MatMul"], frameworkName = "onnx")
class MatMul : PreImportHook {

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
        val a = sd.getVariable(op.inputsToOp[0])
        val b = sd.getVariable(op.inputsToOp[1])
        
        // Get shapes and ranks
        val aShape = sd.shape(a).rename("${opName}_aShape")
        val bShape = sd.shape(b).rename("${opName}_bShape")
        val aRank = sd.rank(a).rename("${opName}_aRank")
        
        // Constants - use INT type to match rank type
        val three = sd.constant("${opName}_three", 3)
        
        // Check if we have the common 3D x 2D case
        val aIs3D = sd.eq("${opName}_aIs3D", aRank, three)
        
        // For the 3D case, we need to handle it specially
        // Get dimensions using gather instead of SDIndex to avoid issues
        val zero_idx = sd.constant("${opName}_zero_idx", 0L)
        val one_idx = sd.constant("${opName}_one_idx", 1L)
        val two_idx = sd.constant("${opName}_two_idx", 2L)
        
        val aDim0 = sd.gather("${opName}_aDim0", aShape, zero_idx, 0)
        val aDim1 = sd.gather("${opName}_aDim1", aShape, one_idx, 0)
        val aDim2 = sd.gather("${opName}_aDim2", aShape, two_idx, 0)
        
        // For B dimensions
        val bDim1 = sd.gather("${opName}_bDim1", bShape, one_idx, 0)
        
        // Calculate batch*seq for A
        val batchTimesSeq = sd.math.mul("${opName}_batchTimesSeq", aDim0, aDim1)
        
        // Create shape arrays as constants
        val reshapeShape = sd.stack("${opName}_reshapeShape", 0, batchTimesSeq, aDim2)
        
        // Reshape A from [batch, seq, hidden] to [batch*seq, hidden]
        val aReshaped = sd.reshape("${opName}_aReshaped", a, reshapeShape)
        
        // Do the 2D matmul
        val matmulResult = sd.linalg().mmul("${opName}_matmul", aReshaped, b, false, false, false)
        
        // Create output shape [batch, seq, output_dim]
        val outputShape = sd.stack("${opName}_outputShape", 0, aDim0, aDim1, bDim1)
        
        // Reshape back to 3D
        val output = sd.reshape("${opName}_output", matmulResult, outputShape)
        
        output.rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
