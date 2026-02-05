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
 * Implementation of ONNX Optional operations.
 *
 * These ops handle optional values which can be present or absent.
 * Used in control flow and conditional execution.
 *
 * @author Eclipse Deeplearning4j Development Team
 */

/**
 * OptionalGetElement - Extracts the element from an optional value.
 * If the optional is empty, behavior is undefined (error in strict mode).
 */
@PreHookRule(nodeNames = [], opNames = ["OptionalGetElement"], frameworkName = "onnx")
class OptionalGetElement : PreImportHook {

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

        // In SameDiff, we treat optional as the tensor itself
        // This op just returns the input tensor (assuming it's present)
        val input = sd.getVariable(op.inputsToOp[0])
        val output = sd.identity("${opName}_output", input)

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * OptionalHasElement - Returns true if optional contains a value.
 */
@PreHookRule(nodeNames = [], opNames = ["OptionalHasElement"], frameworkName = "onnx")
class OptionalHasElement : PreImportHook {

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

        // Check if input exists and has elements
        if (op.inputsToOp.isEmpty() || op.inputsToOp[0] == null || op.inputsToOp[0].isEmpty()) {
            // No input - return false
            val output = sd.constant("${opName}_output", false)
            output.rename(outputNames[0])
            return mapOf(outputNames[0] to listOf(output))
        }

        // Input exists - return true (as scalar boolean)
        val output = sd.constant("${opName}_output", true)
        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * Optional - Creates an optional value from input.
 * If input is provided, wraps it as optional; otherwise creates empty optional.
 */
@PreHookRule(nodeNames = [], opNames = ["Optional"], frameworkName = "onnx")
class Optional : PreImportHook {

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

        // If input is provided, return it; otherwise create placeholder
        val output = if (op.inputsToOp.isNotEmpty() && op.inputsToOp[0] != null && op.inputsToOp[0].isNotEmpty()) {
            val input = sd.getVariable(op.inputsToOp[0])
            sd.identity("${opName}_output", input)
        } else {
            // Create empty tensor as placeholder
            sd.zerosLike("${opName}_output", sd.constant(0.0f))
        }

        output.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(output))
    }
}
