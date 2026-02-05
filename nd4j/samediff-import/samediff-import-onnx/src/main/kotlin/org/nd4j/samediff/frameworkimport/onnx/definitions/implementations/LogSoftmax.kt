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
 * PreImportHook for ONNX LogSoftmax operation.
 *
 * ONNX LogSoftmax (opset 13+) defaults to axis=-1 (last dimension).
 * However, ND4J LogSoftMax defaults to dimension=1, which causes incorrect
 * results when the ONNX model doesn't explicitly set the axis attribute.
 *
 * @author Eclipse Deeplearning4j Contributors
 */
@PreHookRule(nodeNames = [], opNames = ["LogSoftmax"], frameworkName = "onnx")
class LogSoftmax : PreImportHook {

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

        // Get axis from attributes, default to -1 (last dimension) per ONNX opset 13+ spec
        val axis = when (val axisAttr = attributes["axis"]) {
            is Number -> axisAttr.toInt()
            else -> -1  // ONNX opset 13+ default
        }

        // Create log_softmax with the correct axis
        val result = sd.nn().logSoftmax(input, axis)

        // Rename to match expected output name
        val outputVar = result.rename(outputNames[0])

        return mapOf(outputVar.name() to listOf(outputVar))
    }
}
