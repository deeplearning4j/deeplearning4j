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
 * Implementation of Microsoft ONNX FusedMatMul operation.
 *
 * FusedMatMul performs matrix multiplication with optional transpose and scaling:
 * output = alpha * (transA ? A^T : A) @ (transB ? B^T : B)
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#FusedMatMul
 *
 * Inputs:
 * - A: First input tensor
 * - B: Second input tensor
 *
 * Attributes:
 * - alpha: Scalar multiplier for the product (default: 1.0)
 * - transA: Whether to transpose A (default: 0)
 * - transB: Whether to transpose B (default: 0)
 * - transBatchA: Whether to transpose batch dims of A (default: 0)
 * - transBatchB: Whether to transpose batch dims of B (default: 0)
 *
 * @author Eclipse Deeplearning4j Development Team
 */
@PreHookRule(nodeNames = [], opNames = ["FusedMatMul"], frameworkName = "onnx")
class FusedMatMul : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        var a = sd.getVariable(op.inputsToOp[0])
        var b = sd.getVariable(op.inputsToOp[1])

        // Get attributes
        val alpha = (attributes.getOrDefault("alpha", 1.0) as Number).toDouble()
        val transA = (attributes.getOrDefault("transA", 0) as Number).toInt() != 0
        val transB = (attributes.getOrDefault("transB", 0) as Number).toInt() != 0
        val transBatchA = (attributes.getOrDefault("transBatchA", 0) as Number).toInt() != 0
        val transBatchB = (attributes.getOrDefault("transBatchB", 0) as Number).toInt() != 0

        // For FusedMatMul, we use mmul which has built-in transpose support via parameters
        // This avoids needing to know the rank at graph construction time

        // Note: transBatchA and transBatchB require knowing the rank to generate the permutation
        // For now, we skip these since they are rarely used in practice
        // A full implementation would need the shape info from dynamicVariables

        // Perform matrix multiplication with transpose flags
        var result = sd.mmul(a, b, transA, transB, false)

        // Apply alpha scaling if not 1.0
        if (alpha != 1.0) {
            result = sd.math.mul(result, alpha)
        }

        result.rename(outputNames[0])
        return mapOf(outputNames[0] to listOf(result))
    }
}
