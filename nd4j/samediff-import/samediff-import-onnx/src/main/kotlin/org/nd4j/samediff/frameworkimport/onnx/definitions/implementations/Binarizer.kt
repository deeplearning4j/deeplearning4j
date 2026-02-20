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
 * Implementation of ONNX Binarizer operation (ML domain).
 *
 * ONNX Binarizer spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.Binarizer
 *
 * Maps input values to binary values (0 or 1) based on a threshold.
 * output = 1.0 if input > threshold else 0.0
 *
 * Inputs:
 * - X: Input tensor
 *
 * Attributes:
 * - threshold: Threshold for binarization (default: 0.0)
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["Binarizer"], frameworkName = "onnx")
class Binarizer : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Get input
        val input = sd.getVariable(op.inputsToOp[0])
        
        // Get threshold (default: 0.0)
        val threshold = (attributes.getOrDefault("threshold", 0.0) as Number).toDouble()
        
        // output = 1.0 if input > threshold else 0.0
        val comparison = sd.gt(input, threshold)
        val output = comparison.castTo(input.dataType()).rename(outputNames[0])
        
        return mapOf(outputNames[0] to listOf(output))
    }
}
