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

package org.nd4j.ggml.architecture;

import lombok.Data;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Discovers what tensors exist per layer in a GGUF weight map and classifies
 * the attention and FFN patterns. This allows the graph builder to dispatch
 * to the correct method without hardcoding tensor name assumptions.
 *
 * <p>Supported attention patterns:</p>
 * <ul>
 *   <li><b>SEPARATE_QKV</b> — Standard LLaMA: separate attn_q, attn_k, attn_v weights</li>
 *   <li><b>GATED_ATTENTION</b> — Qwen3.5 full attention: separate Q/K/V + QK norms + output gating
 *       (Q weight is 2x because it includes both Q and a gate projection)</li>
 *   <li><b>GDN</b> — Gated Delta Network (linear attention): fused QKV + SSM components + causal conv1d</li>
 *   <li><b>FUSED_QKV</b> — Simple fused QKV (no SSM, no gate)</li>
 * </ul>
 */
public class LayerTensorDiscovery {

    public enum AttentionPattern {
        /** Separate Q, K, V weight matrices (standard LLaMA) */
        SEPARATE_QKV,
        /** Separate Q, K, V with QK norms and output gating (Qwen3.5 full attention layers) */
        GATED_ATTENTION,
        /** Gated Delta Network: fused QKV + SSM recurrence + causal conv (Qwen3.5 linear attention) */
        GDN,
        /** Simple fused QKV weight matrix */
        FUSED_QKV,
        /** No attention weights found */
        NONE
    }

    public enum FFNPattern {
        /** SwiGLU: gate + up + down (standard LLaMA) */
        SWIGLU,
        /** Mixture of Experts with router gate */
        MOE,
        /** GELU FFN: up + down, no gate (GPT-NeoX style) */
        GELU,
        /** No FFN weights found */
        NONE
    }

    /**
     * Name of the post-attention norm weight. Qwen3.5 uses "post_attention_norm"
     * instead of "ffn_norm". This field records which name was found.
     */
    @Data
    public static class LayerProfile {
        private final AttentionPattern attentionPattern;
        private final FFNPattern ffnPattern;
        private final int layerIdx;
        private final String postAttnNormKey;
    }

    /**
     * Scan the weight map for tensors belonging to the given layer and classify
     * the attention and FFN patterns.
     */
    public static LayerProfile discover(Map<String, INDArray> weights, int layerIdx) {
        String prefix = "blk." + layerIdx;

        AttentionPattern attnPattern = discoverAttentionPattern(weights, prefix);
        FFNPattern ffnPattern = discoverFFNPattern(weights, prefix);
        String postAttnNormKey = discoverPostAttnNormKey(weights, prefix);

        return new LayerProfile(attnPattern, ffnPattern, layerIdx, postAttnNormKey);
    }

    private static AttentionPattern discoverAttentionPattern(Map<String, INDArray> weights, String prefix) {
        boolean hasFusedQKV = weights.containsKey(prefix + ".attn_qkv.weight");
        boolean hasSeparateQ = weights.containsKey(prefix + ".attn_q.weight");
        boolean hasSSM = weights.containsKey(prefix + ".ssm_a");
        boolean hasQKNorms = weights.containsKey(prefix + ".attn_q_norm.weight");

        // GDN: fused QKV + SSM components
        if (hasFusedQKV && hasSSM) {
            return AttentionPattern.GDN;
        }
        // Fused QKV without SSM
        if (hasFusedQKV) {
            return AttentionPattern.FUSED_QKV;
        }
        // Gated attention: separate Q/K/V with QK norms (Qwen3.5 full attention)
        if (hasSeparateQ && hasQKNorms) {
            return AttentionPattern.GATED_ATTENTION;
        }
        // Standard separate Q/K/V
        if (hasSeparateQ) {
            return AttentionPattern.SEPARATE_QKV;
        }
        return AttentionPattern.NONE;
    }

    private static FFNPattern discoverFFNPattern(Map<String, INDArray> weights, String prefix) {
        boolean hasRouterGate = weights.containsKey(prefix + ".ffn_gate_inp.weight");
        boolean hasMoEExperts = weights.containsKey(prefix + ".ffn_gate.0.weight");
        boolean hasGate = weights.containsKey(prefix + ".ffn_gate.weight");
        boolean hasUp = weights.containsKey(prefix + ".ffn_up.weight");
        boolean hasDown = weights.containsKey(prefix + ".ffn_down.weight");

        if (hasRouterGate || hasMoEExperts) {
            return FFNPattern.MOE;
        }
        if (hasGate && hasUp && hasDown) {
            return FFNPattern.SWIGLU;
        }
        if (hasUp && hasDown && !hasGate) {
            return FFNPattern.GELU;
        }
        return FFNPattern.NONE;
    }

    /**
     * Discover which key name the post-attention norm uses.
     * Qwen3.5 uses "post_attention_norm" while standard LLaMA uses "ffn_norm".
     */
    private static String discoverPostAttnNormKey(Map<String, INDArray> weights, String prefix) {
        if (weights.containsKey(prefix + ".ffn_norm.weight")) {
            return prefix + ".ffn_norm";
        }
        if (weights.containsKey(prefix + ".post_attention_norm.weight")) {
            return prefix + ".post_attention_norm";
        }
        return prefix + ".ffn_norm"; // default
    }
}
