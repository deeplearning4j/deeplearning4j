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
 * Implementation of Microsoft ONNX GroupQueryAttention (GQA) operation.
 *
 * GroupQueryAttention is an efficient variant of multi-head attention where
 * key-value heads are shared across multiple query heads. This is used in
 * models like LLaMA 2, Mistral, etc.
 *
 * Reference: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#GroupQueryAttention
 *
 * Inputs:
 * - query: Query tensor [batch, seq_len, num_heads * head_dim]
 * - key: Key tensor [batch, kv_seq_len, num_kv_heads * head_dim]
 * - value: Value tensor [batch, kv_seq_len, num_kv_heads * head_dim]
 * - past_key: Optional cached key
 * - past_value: Optional cached value
 * - seqlens_k: Optional sequence lengths for key
 * - total_sequence_length: Optional total sequence length
 * - cos_cache: Optional rotary embedding cos cache
 * - sin_cache: Optional rotary embedding sin cache
 *
 * Attributes:
 * - num_heads: Number of query heads
 * - kv_num_heads: Number of key-value heads (< num_heads for GQA)
 * - scale: Optional scaling factor
 * - local_window_size: Optional local attention window size
 *
 * @author Adam Gibson
 */
@PreHookRule(nodeNames = [], opNames = ["GroupQueryAttention"], frameworkName = "onnx")
class GroupQueryAttention : PreImportHook {

    override fun doImport(
        sd: SameDiff,
        attributes: Map<String, Any>,
        outputNames: List<String>,
        op: SameDiffOp,
        mappingRegistry: OpMappingRegistry<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum, GeneratedMessageV3, GeneratedMessageV3>,
        importGraph: ImportGraph<GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, GeneratedMessageV3, ProtocolMessageEnum>,
        dynamicVariables: Map<String, GeneratedMessageV3>
    ): Map<String, List<SDVariable>> {

        val query = sd.getVariable(op.inputsToOp[0])
        val key = sd.getVariable(op.inputsToOp[1])
        val value = sd.getVariable(op.inputsToOp[2])

        // Optional inputs — ONNX often uses empty strings for missing optionals
        val pastKeyInput = op.inputsToOp.getOrNull(3)
        val hasPastKey = !pastKeyInput.isNullOrEmpty()
        val pastKey = if (hasPastKey) sd.getVariable(pastKeyInput) else null

        val pastValueInput = op.inputsToOp.getOrNull(4)
        val hasPastValue = !pastValueInput.isNullOrEmpty()
        val pastValue = if (hasPastValue) sd.getVariable(pastValueInput) else null

        // Get attributes
        val numHeads = (attributes.getOrDefault("num_heads", 1) as Number).toInt()
        val scaleAttr = attributes["scale"]
        val scale = (scaleAttr as? Number)?.toDouble() ?: 0.0

        // Determine number of requested outputs
        val hasPresentKeyOut = outputNames.size > 1 && hasPastKey
        val hasPresentValueOut = outputNames.size > 2 && hasPastValue
        val requestedOutputs = when {
            hasPresentValueOut -> 3
            hasPresentKeyOut -> 2
            else -> 1
        }

        // Use the native onnx_multi_head_attention op which handles GQA natively.
        // Q is [batch, seq, numHeads * headDim], K/V are [batch, seq, kvNumHeads * headDim].
        // The C++ op derives numKvHeads = kvHidden / headDim and dispatches to
        // fusedGQADecodeCuda for GQA decode (seqQ=1), eliminating repeat_kv expansion.
        val outputs = OnnxMultiHeadAttention(
            sd, query, key, value,
            null, pastKey, pastValue,
            numHeads, scale, false, requestedOutputs
        ).outputVariables()

        // Rename outputs to match expected names
        outputs[0].rename(outputNames[0])
        val results = mutableMapOf(outputNames[0] to listOf(outputs[0]))

        // Output present key/value for caching if requested
        // The C++ op outputs present K/V in BHSD [batch, numKvHeads, totalSeq, headDim]
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
