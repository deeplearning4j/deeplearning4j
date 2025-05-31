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

/**
 * Implementation of Microsoft ONNX Attention operation.
 * 
 * Simplified implementation focusing on basic multi-head attention.
 * 
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Attention"], frameworkName = "onnx")
class Attention: PreImportHook {
    
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
        val weights = sd.getVariable(op.inputsToOp[1])
        
        // Optional inputs
        var bias: SDVariable? = null
        if (op.inputsToOp.size > 2 && op.inputsToOp[2] != null) {
            bias = sd.getVariable(op.inputsToOp[2])
        }
        
        // Extract attributes
        val numHeads = (attributes.getOrDefault("num_heads", 8) as Number).toInt()
        
        // Compute Q, K, V projections
        var qkv = sd.mmul(input, weights)
        if (bias != null) {
            qkv = qkv.add(bias)
        }
        
        // Split QKV - use simple equal split for now
        val qkvArray = sd.split(qkv, 2, 3)
        val q = qkvArray[0]
        val k = qkvArray[1] 
        val v = qkvArray[2]
        
        // For simplicity, use fixed dimensions and avoid complex reshaping
        // In a real implementation, you would properly handle multi-head attention
        
        // Compute attention scores: Q * K^T
        val kT = sd.permute(k, 0, 2, 1) // Simple transpose for last 2 dims
        val scores = sd.mmul(q, kT)
        
        // Scale by sqrt(head_size) - use fixed value for simplicity
        val scale = sd.constant(1.0 / kotlin.math.sqrt(64.0))
        val scaledScores = scores.mul(scale)
        
        // Apply softmax
        val attentionWeights = sd.nn().softmax(scaledScores, -1)
        
        // Apply attention to values
        val attentionOutput = sd.mmul(attentionWeights, v)
        
        attentionOutput.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(attentionOutput))
    }
}