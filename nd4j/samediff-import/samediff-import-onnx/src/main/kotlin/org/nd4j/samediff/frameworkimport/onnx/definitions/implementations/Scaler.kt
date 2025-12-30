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
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of ONNX Scaler operation (ML domain).
 *
 * ONNX Scaler spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Scaler
 *
 * Applies standard scaling: output = (input - offset) * scale
 * Used for feature preprocessing (e.g., StandardScaler, MinMaxScaler).
 *
 * Inputs:
 * - X: Input tensor of shape [N, C]
 *
 * Attributes:
 * - offset: Per-feature offset values
 * - scale: Per-feature scale values
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["Scaler"], frameworkName = "onnx")
class Scaler : PreImportHook {

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
        
        // Get offset and scale attributes
        val offsetArr = getFloatArray(attributes, "offset")
        val scaleArr = getFloatArray(attributes, "scale")
        
        // Apply transformation: output = (input - offset) * scale
        var result = input
        
        if (offsetArr != null && offsetArr.isNotEmpty()) {
            val offsetVar = sd.constant("${opName}_offset", Nd4j.createFromArray(*offsetArr))
            result = sd.math.sub("${opName}_centered", result, offsetVar)
        }
        
        if (scaleArr != null && scaleArr.isNotEmpty()) {
            val scaleVar = sd.constant("${opName}_scale", Nd4j.createFromArray(*scaleArr))
            result = sd.math.mul("${opName}_scaled", result, scaleVar)
        }
        
        val output = result.rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
    }
    
    private fun getFloatArray(attributes: Map<String, Any>, key: String): FloatArray? {
        val value = attributes[key] ?: return null
        return when (value) {
            is FloatArray -> value
            is DoubleArray -> value.map { it.toFloat() }.toFloatArray()
            is List<*> -> value.map { (it as Number).toFloat() }.toFloatArray()
            else -> null
        }
    }
}
