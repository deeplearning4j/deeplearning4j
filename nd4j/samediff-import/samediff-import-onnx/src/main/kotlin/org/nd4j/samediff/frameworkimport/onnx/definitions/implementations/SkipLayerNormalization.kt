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
 * Implementation of Microsoft ONNX SkipLayerNormalization operation.
 * 
 * SkipLayerNormalization combines residual connection (skip connection) with layer normalization:
 * 1. Add input and skip: sum = input + skip
 * 2. Apply layer normalization: output = LayerNorm(sum)
 * 
 * This is commonly used in transformer architectures where the residual connection
 * is combined with normalization in a single operation for efficiency.
 * 
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["SkipLayerNormalization"], frameworkName = "onnx")
class SkipLayerNormalization: PreImportHook {
    
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
        val skip = sd.getVariable(op.inputsToOp[1])
        val gamma = sd.getVariable(op.inputsToOp[2])
        
        // Optional beta (layer norm bias)
        var beta: SDVariable? = null
        if (op.inputsToOp.size > 3 && op.inputsToOp[3] != null) {
            beta = sd.getVariable(op.inputsToOp[3])
        }
        
        // Optional bias (bias added before skip connection)
        var bias: SDVariable? = null
        if (op.inputsToOp.size > 4 && op.inputsToOp[4] != null) {
            bias = sd.getVariable(op.inputsToOp[4])
        }
        
        val epsilon = attributes.getOrDefault("epsilon", 1e-12) as Number
        
        // Apply bias to input if present
        var processedInput = input
        if (bias != null) {
            processedInput = input.add(bias)
        }
        
        // Add skip connection
        val summed = processedInput.add(skip)
        
        // Apply layer normalization
        // LayerNorm normalizes over the last dimension (hidden_size)
        val result = if (beta != null) {
            sd.nn().layerNorm(outputNames[0], summed, gamma, beta, false, -1L)
        } else {
            // Create zero bias if not provided
            val zeroBias = sd.zerosLike(gamma)
            sd.nn().layerNorm(outputNames[0], summed, gamma, zeroBias, false, -1L)
        }
        
        // Some implementations also output the sum before layer norm for downstream use
        if (outputNames.size > 1) {
            summed.rename(outputNames[1])
            return mapOf(
                outputNames[0] to listOf(result.rename(outputNames[0])),
                outputNames[1] to listOf(summed.rename(outputNames[1]))
            )
        }
        
        return mapOf(outputNames[0] to listOf(result.rename(outputNames[0])))
    }
}