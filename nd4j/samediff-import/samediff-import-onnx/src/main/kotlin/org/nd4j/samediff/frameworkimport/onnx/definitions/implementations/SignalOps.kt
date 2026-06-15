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
 * Signal processing operations for ONNX - commonly used in speech/audio models.
 * These implementations use native C++ ops for optimal performance.
 *
 * @author Adam Gibson
 */

/**
 * STFT - Short-Time Fourier Transform using native op
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#STFT
 */
@PreHookRule(nodeNames = [], opNames = ["STFT"], frameworkName = "onnx")
class STFT : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        // Inputs
        val signal = sd.getVariable(op.inputsToOp[0])
        val frameStep = if (op.inputsToOp.size > 1) sd.getVariable(op.inputsToOp[1]) else sd.constant(1L)
        val window = if (op.inputsToOp.size > 2 && op.inputsToOp[2].isNotEmpty()) sd.getVariable(op.inputsToOp[2]) else null
        val frameLength = if (op.inputsToOp.size > 3) sd.getVariable(op.inputsToOp[3]) else null

        // Attributes
        val onesided = (attributes.getOrDefault("onesided", 1L) as Number).toInt() != 0

        // Use native STFT op
        val output = sd.signal().stft(outputNames[0], signal, frameStep, window, frameLength, onesided)

        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * DFT - Discrete Fourier Transform using native op
 *
 * Reference: https://github.com/onnx/onnx/blob/main/docs/Operators.md#DFT
 */
@PreHookRule(nodeNames = [], opNames = ["DFT"], frameworkName = "onnx")
class DFT : PreImportHook {

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

        // Attributes
        val axis = (attributes.getOrDefault("axis", 1L) as Number).toInt()
        val inverse = (attributes.getOrDefault("inverse", 0L) as Number).toInt() != 0
        val onesided = (attributes.getOrDefault("onesided", 0L) as Number).toInt() != 0

        // Use native DFT op
        val output = sd.signal().dft(outputNames[0], input, axis, inverse, onesided)

        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * HannWindow - Generate Hann window using native op
 */
@PreHookRule(nodeNames = [], opNames = ["HannWindow"], frameworkName = "onnx")
class HannWindow : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val size = sd.getVariable(op.inputsToOp[0])
        val periodic = (attributes.getOrDefault("periodic", 1L) as Number).toInt() != 0

        // Use native hann_window op
        val output = sd.signal().hannWindow(outputNames[0], size, periodic)

        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * HammingWindow - Generate Hamming window using native op
 */
@PreHookRule(nodeNames = [], opNames = ["HammingWindow"], frameworkName = "onnx")
class HammingWindow : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val size = sd.getVariable(op.inputsToOp[0])
        val periodic = (attributes.getOrDefault("periodic", 1L) as Number).toInt() != 0

        // Use native hamming_window op
        val output = sd.signal().hammingWindow(outputNames[0], size, periodic)

        return mapOf(outputNames[0] to listOf(output))
    }
}

/**
 * BlackmanWindow - Generate Blackman window using native op
 */
@PreHookRule(nodeNames = [], opNames = ["BlackmanWindow"], frameworkName = "onnx")
class BlackmanWindow : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {
        val size = sd.getVariable(op.inputsToOp[0])
        val periodic = (attributes.getOrDefault("periodic", 1L) as Number).toInt() != 0

        // Use native blackman_window op
        val output = sd.signal().blackmanWindow(outputNames[0], size, periodic)

        return mapOf(outputNames[0] to listOf(output))
    }
}
