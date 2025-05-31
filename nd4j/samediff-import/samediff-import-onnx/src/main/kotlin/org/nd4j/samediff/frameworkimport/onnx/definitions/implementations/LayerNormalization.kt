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
 * Implementation of Microsoft ONNX LayerNormalization operation.
 * 
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["LayerNormalization"], frameworkName = "onnx")
class LayerNormalization: PreImportHook {
    
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
        val scale = sd.getVariable(op.inputsToOp[1])
        
        // Bias is optional
        var bias: SDVariable? = null
        if (op.inputsToOp.size > 2 && op.inputsToOp[2] != null) {
            bias = sd.getVariable(op.inputsToOp[2])
        }
        
        val axis = (attributes.getOrDefault("axis", -1) as Number).toInt()
        val epsilon = (attributes.getOrDefault("epsilon", 1e-5) as Number).toDouble()
        
        // Apply layer normalization using the correct API signature
        val result = if (bias != null) {
            sd.nn().layerNorm(outputNames[0], input, scale, bias, false, axis.toLong())
        } else {
            // Create zero bias if not provided
            val zeroBias = sd.zerosLike(scale)
            sd.nn().layerNorm(outputNames[0], input, scale, zeroBias, false, axis.toLong())
        }
        
        val outputs = mutableMapOf<String, List<SDVariable>>()
        outputs[outputNames[0]] = listOf(result)
        
        // If additional outputs are requested (mean and inverse std dev)
        if (outputNames.size > 1) {
            // Compute mean along the normalization axis - simplified version
            val actualAxis = if (axis < 0) -1L else axis.toLong()
            
            // Compute mean and variance manually for additional outputs
            val mean = sd.mean(input, false, actualAxis)
            mean.rename(outputNames[1])
            outputs[outputNames[1]] = listOf(mean)
            
            if (outputNames.size > 2) {
                // Compute inverse standard deviation
                val variance = sd.variance(input, false, actualAxis)
                val invStdDev = sd.math().pow(variance.add(sd.constant(epsilon)), sd.constant(-0.5))
                invStdDev.rename(outputNames[2])
                outputs[outputNames[2]] = listOf(invStdDev)
            }
        }
        
        return outputs
    }
}