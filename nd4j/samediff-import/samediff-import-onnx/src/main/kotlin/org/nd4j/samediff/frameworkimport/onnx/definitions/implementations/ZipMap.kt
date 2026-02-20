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
 * Implementation of ONNX ZipMap operation (ML domain).
 *
 * ONNX ZipMap spec: https://github.com/onnx/onnx/blob/main/docs/Operators-ml.md#ai.onnx.ml.ZipMap
 *
 * Creates a map (dictionary) from the input tensor where each row becomes
 * a dictionary with class labels as keys and probabilities as values.
 *
 * Since SameDiff doesn't support dictionary/map output types, this implementation
 * passes through the input tensor unchanged, preserving the probability values.
 *
 * Inputs:
 * - X: Input tensor [N, C] where C is number of classes
 *
 * Attributes:
 * - classlabels_int64s: Integer class labels
 * - classlabels_strings: String class labels
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["ZipMap"], frameworkName = "onnx")
class ZipMap : PreImportHook {

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
        val x = sd.getVariable(op.inputsToOp[0])

        // ZipMap converts tensor to sequence of maps
        // Since SameDiff doesn't support map types, we pass through as identity
        // The consumer of this output should interpret it as class probabilities
        val output = sd.identity("${opName}_zipmap", x)

        val finalOutput = output.rename(outputNames[0])

        return mapOf(outputNames[0] to listOf(finalOutput))
    }
}
