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
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention
import org.nd4j.samediff.frameworkimport.ImportGraph
import org.nd4j.samediff.frameworkimport.hooks.PreImportHook
import org.nd4j.samediff.frameworkimport.hooks.annotations.PreHookRule
import org.nd4j.samediff.frameworkimport.registry.OpMappingRegistry
import org.nd4j.shade.protobuf.GeneratedMessageV3
import org.nd4j.shade.protobuf.ProtocolMessageEnum

/**
 * Implementation of Microsoft ONNX MultiHeadAttention operation.
 *
 * Uses the native onnx_multi_head_attention op which handles pre-projected Q, K, V
 * tensors directly with proper attention computation.
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["MultiHeadAttention"], frameworkName = "onnx")
class MultiHeadAttention : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        var query = sd.getVariable(op.inputsToOp[0])
        var key = sd.getVariable(op.inputsToOp[1])
        var value = sd.getVariable(op.inputsToOp[2])

        // Optional inputs: ONNX often encodes missing optionals as empty-string placeholders.
        val biasInput = op.inputsToOp.getOrNull(3)
        val bias = if (!biasInput.isNullOrEmpty()) sd.getVariable(biasInput) else null

        val keyPaddingMaskInput = op.inputsToOp.getOrNull(4)
        val keyPaddingMask = if (!keyPaddingMaskInput.isNullOrEmpty()) sd.getVariable(keyPaddingMaskInput) else null

        val relativePosBiasInput = op.inputsToOp.getOrNull(5)
        val relativePosBias = if (!relativePosBiasInput.isNullOrEmpty()) sd.getVariable(relativePosBiasInput) else null

        val pastKeyInput = op.inputsToOp.getOrNull(6)
        val pastKey = if (!pastKeyInput.isNullOrEmpty()) sd.getVariable(pastKeyInput) else null

        val pastValueInput = op.inputsToOp.getOrNull(7)
        val pastValue = if (!pastValueInput.isNullOrEmpty()) sd.getVariable(pastValueInput) else null

        // Get attributes
        val numHeads = (attributes.getOrDefault("num_heads", 1) as Number).toInt()
        val scaleAttr = attributes["scale"]
        val scale = (scaleAttr as? Number)?.toDouble() ?: 0.0

        // ONNX MultiHeadAttention attribute. Defaults to non-causal when not present.
        var useCausalMask = ((attributes["unidirectional"] as? Number)?.toInt() ?: 0) != 0
        val isAttnMaskReformatBias = !relativePosBiasInput.isNullOrEmpty() &&
            relativePosBiasInput.contains("attn_mask_reformat")

        // Apply bias if present (ONNX bias is concatenated [Wq_bias, Wk_bias, Wv_bias])
        if (bias != null) {
            val splitBias = sd.split(bias, 3, 0)
            if (splitBias.size == 3) {
                query = sd.math.add("_q_bias", query, splitBias[0])
                key = sd.math.add("_k_bias", key, splitBias[1])
                value = sd.math.add("_v_bias", value, splitBias[2])
            }
        }

        // Combine attention bias sources: relativePosBias and keyPaddingMask
        // keyPaddingMask needs to be converted to additive bias format
        var attnBias: SDVariable? = relativePosBias
        if (keyPaddingMask != null) {
            val maskFilterValue = (attributes.getOrDefault("mask_filter_value", -3.4028235e+38) as Number).toDouble()
            // keyPaddingMask is [batch, seqKV], 1 = keep, 0 = mask
            // Convert to additive bias: (1 - mask) * maskFilterValue
            val one = sd.constant(1.0)
            val invMask = sd.math.sub(one, keyPaddingMask.castTo(query.dataType()))
            val paddingBias = sd.math.mul(invMask, maskFilterValue)
            // Expand to [batch, 1, 1, seqKV] for broadcasting
            val paddingBiasExpanded = sd.expandDims(sd.expandDims(paddingBias, 1), 2)
            
            attnBias = if (attnBias != null) {
                sd.math.add(attnBias, paddingBiasExpanded)
            } else {
                paddingBiasExpanded
            }
        }

        // SmolDocling exports a dynamic causal mask through attn_mask_reformat/Tile.
        // Keep this explicit mask during decode with KV cache: replacing it with a plain
        // causal flag loses the decode offset (past sequence length), which can collapse
        // generation quality after prefill.
        if (isAttnMaskReformatBias && keyPaddingMask == null) {
            useCausalMask = false
        }

        val hasPresentKeyOut = outputNames.size > 1 && outputNames[1].isNotEmpty()
        val hasPresentValueOut = outputNames.size > 2 && outputNames[2].isNotEmpty()
        val requestedOutputs = when {
            hasPresentValueOut -> 3
            hasPresentKeyOut -> 2
            else -> 1
        }

        // Use the native onnx_multi_head_attention op
        val outputs = OnnxMultiHeadAttention(
            sd, query, key, value,
            attnBias, pastKey, pastValue,
            numHeads, scale, useCausalMask, requestedOutputs
        ).outputVariables()

        // Rename outputs to match expected names
        outputs[0].rename(outputNames[0])
        val results = mutableMapOf(outputNames[0] to listOf(outputs[0]))

        // Output present key/value if requested
        if (hasPresentKeyOut && outputs.size > 1) {
            outputs[1].rename(outputNames[1])
            results[outputNames[1]] = listOf(outputs[1])
        }
        if (hasPresentValueOut && outputs.size > 2) {
            outputs[2].rename(outputNames[2])
            results[outputNames[2]] = listOf(outputs[2])
        }

        return results
    }
}
