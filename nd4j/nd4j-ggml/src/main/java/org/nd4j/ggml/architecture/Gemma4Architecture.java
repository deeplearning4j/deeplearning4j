/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.nd4j.autodiff.samediff.SDIndex;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Dense Gemma 4 text trunk. GGUF matrix weights arrive in [out,in] order (or as
 * packed bytes with a .__q__ descriptor); embedding tables must arrive dense.
 * Runtime shapes are symbolic: activations and KV use [batch,sequence,heads,dim].
 * Each sharing layer publishes and caches its donor's current K/V under its own
 * names, preserving the existing GGUF prefill/decode ABI without cache alias races.
 *
 * <p>Norm arithmetic and linear accumulation are FP32, with explicit boundaries
 * back to the requested activation/cache dtype. Logits stay FLOAT through softcap.
 * No Gemma 1-3 normalization or attention assumptions are shared with this class.</p>
 */
public final class Gemma4Architecture implements ModelArchitecture {
    @Override public String getName() { return "gemma4"; }
    @Override public Set<String> getSupportedVariants() { return Set.of("gemma4"); }
    @Override public boolean canHandle(GGMLMetadata metadata) {
        return "gemma4".equalsIgnoreCase(metadata.getArchitecture());
    }
    @Override public String getDefaultChatTemplateType() { return "gemma"; }

    @Override
    public ArchitectureConfig getConfig(GGMLMetadata metadata) {
        Layout c = new Layout(metadata);
        List<String> types = new ArrayList<>();
        List<Integer> kvHeads = new ArrayList<>();
        for (int l = 0; l < c.layers; l++) {
            types.add(c.local[l] ? "sliding_attention" : "full_attention");
            kvHeads.add(c.kvHeads[l]);
        }
        // ArchitectureConfig has only a scalar headDim. Per-layer dimensions are
        // authoritative on the actual KV placeholders, not this summary field.
        return ArchitectureConfig.builder().numLayers(c.layers).hiddenSize(c.hidden)
                .intermediateSize(c.ffn[0]).numAttentionHeads(c.heads[0])
                .numKVHeads(c.kvHeads[0]).kvHeadsPerLayer(kvHeads)
                .headDim(c.globalDim).ropeDimensionCount(c.globalDim)
                .vocabSize(metadata.getVocabSize()).contextLength(metadata.getContextLength())
                .layerNormEpsilon((float)c.epsilon).ropeFreqBase((float)c.globalTheta)
                .layerTypes(types).useRmsNorm(true).useSwiGLU(false)
                .useRotaryEmbeddings(true).attentionMultiplier(1.0).decoderOnly(true).build();
    }

    @Override
    public SameDiff buildGraph(GGMLMetadata metadata, Map<String, INDArray> weights, ConversionOptions options) {
        Layout c = new Layout(metadata);
        DataType dtype = options.getTargetDataType();
        if (dtype != DataType.FLOAT && dtype != DataType.HALF && dtype != DataType.BFLOAT16) {
            throw new IllegalArgumentException("Gemma 4 requires FLOAT, HALF or BFLOAT16 activations");
        }
        validateWeights(weights, c);
        SameDiff sd = SameDiff.create();
        SDVariable ids = sd.placeHolder("input_ids", DataType.INT64, -1, -1);
        SDVariable position = sd.placeHolder("position_offset", DataType.INT64);
        SDVariable cachePosition = sd.placeHolder("cache_position", DataType.INT64);
        SDVariable actualSequenceLength = sd.placeHolder("actual_sequence_length", DataType.INT64);
        SDVariable mask = sd.placeHolder("_causal_mask", DataType.FLOAT, -1, -1, -1, -1);
        SDVariable[] keys = new SDVariable[c.layers];
        SDVariable[] values = new SDVariable[c.layers];
        List<String> outputs = new ArrayList<>();

        SDVariable hidden = sd.gather("embedded", weight(sd, weights, "token_embd.weight"), ids, 0);
        hidden = cast(hidden, dtype).mul("embed_scaled", Math.sqrt(c.hidden));
        SDVariable ple = null;
        if (c.ple > 0) {
            SDVariable tokenPle = cast(sd.gather("ple_embedded",
                    weight(sd, weights, "per_layer_token_embd.weight"), ids, 0), dtype).mul(Math.sqrt(c.ple));
            tokenPle = sd.reshape("ple_token", tokenPle, shape(sd, ids, c.layers, c.ple));
            SDVariable projected = linear(sd, weights, "ple_model_projection", hidden,
                    "per_layer_model_proj.weight", dtype).div(Math.sqrt(c.hidden));
            projected = sd.reshape("ple_projected", projected, shape(sd, ids, c.layers, c.ple));
            projected = rmsNorm(sd, "ple_projection_norm", projected,
                    weight(sd, weights, "per_layer_proj_norm.weight"), c.epsilon);
            ple = projected.add(tokenPle).mul("per_layer_inputs", 1.0 / Math.sqrt(2.0));
        }

        SDVariable localMask = slidingMask(sd, mask, cachePosition, sd.sizeAt(ids, 1), c.window);
        SDVariable positions = sd.range(sd.constant(0L), sd.sizeAt(ids, 1), sd.constant(1L), DataType.INT64)
                .add(position).castTo(DataType.FLOAT).reshape(1, -1, 1, 1);
        SDVariable[][] trig = new SDVariable[2][];
        for (int l = 0; l < c.layers; l++) {
            String p = "blk." + l + ".";
            int d = c.headDim(l);
            int type = c.local[l] ? 0 : 1;
            if (trig[type] == null) {
                float[] frequencies = new float[d / 2];
                for (int i = 0; i < frequencies.length; i++) {
                    frequencies[i] = (float)Math.pow(c.local[l] ? c.localTheta : c.globalTheta, -2.0 * i / d);
                }
                SDVariable inv = sd.constant("rope_inv_freq_" + type, Nd4j.createFromArray(frequencies));
                if (!c.local[l]) {
                    inv = inv.div(cast(weight(sd, weights, "rope_freqs.weight"), DataType.FLOAT));
                }
                SDVariable angle = positions.mul(inv.reshape(1, 1, 1, d / 2));
                trig[type] = new SDVariable[]{sd.math().cos(angle), sd.math().sin(angle)};
            }
            SDVariable normed = rmsNorm(sd, "attn_norm_" + l, hidden,
                    weight(sd, weights, p + "attn_norm.weight"), c.epsilon);
            SDVariable q = linear(sd, weights, "q_proj_" + l, normed, p + "attn_q.weight", dtype);
            q = sd.reshape("q_heads_" + l, q, shape(sd, ids, c.heads[l], d));
            q = rmsNorm(sd, "q_norm_" + l, q, weight(sd, weights, p + "attn_q_norm.weight"), c.epsilon);
            q = rope(sd, "q_rope_" + l, q, trig[type], d);
            if (c.donor[l] == l) {
                SDVariable k = linear(sd, weights, "k_proj_" + l, normed, p + "attn_k.weight", dtype);
                SDVariable v = linear(sd, weights, "v_proj_" + l, normed, p + "attn_v.weight", dtype);
                k = sd.reshape("k_heads_" + l, k, shape(sd, ids, c.kvHeads[l], d));
                v = sd.reshape("v_projected_heads_" + l, v, shape(sd, ids, c.kvHeads[l], d));
                k = rmsNorm(sd, "k_norm_" + l, k, weight(sd, weights, p + "attn_k_norm.weight"), c.epsilon);
                keys[l] = rope(sd, "k_rope_" + l, k, trig[type], d);
                values[l] = rmsNorm(sd, "v_heads_" + l, v, null, c.epsilon);
            } else {
                // Real identity nodes: merely listing output names does not create aliases.
                keys[l] = sd.identity("k_rope_" + l, keys[c.donor[l]]);
                values[l] = sd.identity("v_heads_" + l, values[c.donor[l]]);
            }
            SDVariable kc = sd.placeHolder("past_key_values." + l + ".key", dtype, -1, -1, c.kvHeads[l], d);
            SDVariable vc = sd.placeHolder("past_key_values." + l + ".value", dtype, -1, -1, c.kvHeads[l], d);
            SDVariable attn = sd.nn().dotProductAttentionV2("attn_out_" + l,
                    q, values[l], keys[l], null, null, kc, vc, cachePosition,
                    c.local[l] ? localMask : mask, 1.0, 0.0, false, false);
            attn = sd.reshape("attn_flat_" + l, attn, shape(sd, ids, c.heads[l] * d));
            attn = linear(sd, weights, "attn_projection_" + l, attn, p + "attn_output.weight", dtype);
            attn = rmsNorm(sd, "post_attention_norm_" + l, attn,
                    weight(sd, weights, p + "post_attention_norm.weight"), c.epsilon);
            hidden = hidden.add("post_attn_" + l, attn);
            SDVariable ffn = rmsNorm(sd, "ffn_norm_" + l, hidden,
                    weight(sd, weights, p + "ffn_norm.weight"), c.epsilon);
            SDVariable gate = gelu(sd, "ffn_gelu_" + l,
                    linear(sd, weights, "ffn_gate_" + l, ffn, p + "ffn_gate.weight", dtype));
            SDVariable up = linear(sd, weights, "ffn_up_" + l, ffn, p + "ffn_up.weight", dtype);
            ffn = linear(sd, weights, "ffn_down_" + l, gate.mul(up), p + "ffn_down.weight", dtype);
            ffn = rmsNorm(sd, "post_ffw_norm_" + l, ffn,
                    weight(sd, weights, p + "post_ffw_norm.weight"), c.epsilon);
            hidden = hidden.add("post_ffn_" + l, ffn);
            if (ple != null) {
                SDVariable layerPle = sd.gather(ple, new int[]{l}, 2);
                layerPle = sd.reshape("ple_input_" + l, layerPle, shape(sd, ids, c.ple));
                SDVariable pleGate = gelu(sd, "ple_gelu_" + l,
                        linear(sd, weights, "ple_gate_" + l, hidden, p + "inp_gate.weight", dtype));
                SDVariable correction = linear(sd, weights, "ple_back_projection_" + l,
                        pleGate.mul(layerPle), p + "proj.weight", dtype);
                correction = rmsNorm(sd, "post_norm_" + l, correction,
                        weight(sd, weights, p + "post_norm.weight"), c.epsilon);
                hidden = hidden.add("post_ple_" + l, correction);
            }
            hidden = sd.identity("layer_out_" + l, cast(cast(hidden, DataType.FLOAT).mul(
                    cast(weight(sd, weights, p + "layer_output_scale.weight"), DataType.FLOAT)), dtype));
            outputs.add(keys[l].name());
            outputs.add(values[l].name());
        }
        hidden = rmsNorm(sd, "final_norm", hidden, weight(sd, weights, "output_norm.weight"), c.epsilon);
        String head = weights.containsKey("output.weight") ? "output.weight" : "token_embd.weight";
        SDVariable logits = QuantizedLinear.matMulFloatOutput(sd, "logits_uncapped", hidden,
                weight(sd, weights, head), weights, head, dtype);
        softcap(sd, "logits", logits, c.softcap);
        outputs.add("logits");

        // Slice normalized hidden BEFORE the vocabulary projection. The runtime's
        // actual length excludes fixed-buffer right padding; sizeAt(hidden, 1) does not.
        SDVariable one = sd.constant(Nd4j.scalar(DataType.INT64, 1L));
        SDVariable zero = sd.constant(Nd4j.scalar(DataType.INT64, 0L));
        SDVariable begin = sd.stack("lm_last_begin", 0, zero, actualSequenceLength.sub(one), zero);
        SDVariable size = sd.stack("lm_last_size", 0, sd.sizeAt(hidden, 0), one, sd.sizeAt(hidden, 2));
        SDVariable hiddenLast = sd.slice("hidden_last", hidden, begin, size);
        SDVariable logitsLast = QuantizedLinear.matMulFloatOutput(sd, "logits_last_uncapped", hiddenLast,
                weight(sd, weights, head), weights, head, dtype);
        softcap(sd, "lm_logits_last", logitsLast, c.softcap);
        outputs.add("lm_logits_last");
        sd.setOutputs(outputs);
        return sd;
    }

    /** Both terminal projections keep FLOAT through the same owned-buffer softcap. */
    private static void softcap(SameDiff sd, String name, SDVariable logits, double softcap) {
        if (softcap > 0) {
            SDVariable cap = sd.constant(name + "_softcap", (float)softcap);
            sd.nn().fusedElementwiseChain(name, logits, new SDVariable[]{cap, cap},
                    new int[]{org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.OP_DIV,
                            org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.OP_TANH,
                            org.nd4j.linalg.api.ops.impl.transforms.custom.FusedElementwiseChain.OP_MUL});
        } else {
            sd.identity(name, logits);
        }
    }

    /** Direct gamma, not Gemma 1-3's gamma+1. Never square low-precision storage. */
    static SDVariable rmsNorm(SameDiff sd, String name, SDVariable x, SDVariable gamma, double epsilon) {
        SDVariable f = cast(x, DataType.FLOAT);
        // Keep the FP32 boundary, but avoid retaining square/scale intermediates
        // for every normalization during the first DSP prefill warmup.
        SDVariable normalized = new RmsNorm(sd, f,
                gamma == null ? null : cast(gamma, DataType.FLOAT), epsilon).outputVariable();
        return sd.identity(name, cast(normalized, x.dataType()));
    }

    /** PreciseGELU is the tanh formula in libnd4j; SDNN.gelu is x*sigmoid(1.702*x). */
    static SDVariable gelu(SameDiff sd, String name, SDVariable x) {
        return sd.identity(name, cast(sd.nn().preciseGelu(cast(x, DataType.FLOAT)), x.dataType()));
    }

    static SDVariable slidingMask(SameDiff sd, SDVariable causal, SDVariable cachePosition,
                                  SDVariable sequenceLength, int window) {
        SDVariable q = sd.range(sd.constant(0L), sequenceLength, sd.constant(1L), DataType.INT64)
                .add(cachePosition).reshape(-1, 1);
        SDVariable k = sd.range(sd.constant(0L), sd.sizeAt(causal, 3), sd.constant(1L), DataType.INT64)
                .reshape(1, -1);
        // k > q-window is allowed. Finite negative sentinel avoids 0 * -Infinity.
        // [1,1,Tq,K] broadcasts only batch/head, retaining every caller padding bias.
        SDVariable tooOld = sd.lte(k, q.sub(window)).castTo(DataType.FLOAT);
        SDVariable windowBias = tooOld.mul(-Float.MAX_VALUE);
        windowBias = sd.expandDims(sd.expandDims(windowBias, 0), 0);
        return causal.add("gemma4_local_mask", windowBias);
    }

    private static SDVariable rope(SameDiff sd, String name, SDVariable x, SDVariable[] trig, int dim) {
        SDVariable f = cast(x, DataType.FLOAT);
        SDVariable a = f.get(SDIndex.all(), SDIndex.all(), SDIndex.all(), SDIndex.interval(0, dim / 2));
        SDVariable b = f.get(SDIndex.all(), SDIndex.all(), SDIndex.all(), SDIndex.interval(dim / 2, dim));
        SDVariable rotated = sd.concat(3, a.mul(trig[0]).sub(b.mul(trig[1])),
                b.mul(trig[0]).add(a.mul(trig[1])));
        return sd.identity(name, cast(rotated, x.dataType()));
    }

    private static SDVariable shape(SameDiff sd, SDVariable ids, int... tail) {
        SDVariable[] dims = new SDVariable[2 + tail.length];
        dims[0] = sd.sizeAt(ids, 0);
        dims[1] = sd.sizeAt(ids, 1);
        for (int i = 0; i < tail.length; i++) dims[i + 2] = sd.constant((long)tail[i]);
        return sd.stack(0, dims);
    }

    private static SDVariable cast(SDVariable x, DataType dtype) {
        return x.dataType() == dtype ? x : x.castTo(dtype);
    }

    private static SDVariable linear(SameDiff sd, Map<String, INDArray> weights, String name,
                                     SDVariable x, String key, DataType dtype) {
        // Both dense and packed paths have the same activation storage boundary.
        // The packed weight itself is never cast or materialized as a dense matrix.
        return sd.identity(name, cast(QuantizedLinear.matMulFloatOutput(sd, name + "_fp32", x,
                weight(sd, weights, key), weights, key, dtype), dtype));
    }

    private static SDVariable weight(SameDiff sd, Map<String, INDArray> weights, String key) {
        SDVariable existing = sd.getVariable(key);
        return existing != null ? existing : sd.var(key, required(weights, key));
    }

    private static INDArray required(Map<String, INDArray> weights, String key) {
        INDArray value = weights.get(key);
        if (value == null) throw new IllegalArgumentException("Missing required Gemma 4 weight: " + key);
        return value;
    }

    private static void matrix(Map<String, INDArray> weights, String key, int out, int in) {
        INDArray w = required(weights, key);
        boolean packed = weights.containsKey(key + ".__q__");
        if ((!packed && w.rank() != 2) || (packed && w.dataType() != DataType.BYTE)) {
            throw new IllegalArgumentException("Invalid Gemma 4 matrix storage: " + key);
        }
        if (QuantizedLinear.logicalOutputDim(weights, key, w) != out
                || QuantizedLinear.logicalInputDim(weights, key, w) != in) {
            throw new IllegalArgumentException("Gemma 4 weight " + key + " must have logical [" + out + "," + in + "]");
        }
        if (!w.dataType().isFPType() && !(w.dataType() == DataType.BYTE && weights.containsKey(key + ".__q__"))) {
            throw new IllegalArgumentException("Gemma 4 weight lacks a valid dense/packed linear representation: " + key);
        }
    }

    private static void vector(Map<String, INDArray> weights, String key, int size) {
        INDArray w = required(weights, key);
        if (w.rank() != 1 || w.length() != size || !w.dataType().isFPType()) {
            throw new IllegalArgumentException("Gemma 4 weight " + key + " must be a floating vector of length " + size);
        }
    }

    private static void validateWeights(Map<String, INDArray> weights, Layout c) {
        INDArray token = required(weights, "token_embd.weight");
        if (token.rank() != 2 || !token.dataType().isFPType()) {
            throw new IllegalArgumentException("Gemma 4 token_embd.weight must be dequantized by the converter for gather");
        }
        int vocab = Math.toIntExact(token.size(0));
        matrix(weights, "token_embd.weight", vocab, c.hidden);
        vector(weights, "output_norm.weight", c.hidden);
        if (weights.containsKey("output.weight")) matrix(weights, "output.weight", vocab, c.hidden);
        boolean hasGlobal = false;
        for (boolean local : c.local) hasGlobal |= !local;
        if (hasGlobal) vector(weights, "rope_freqs.weight", c.globalDim / 2);
        if (c.ple > 0) {
            INDArray table = required(weights, "per_layer_token_embd.weight");
            if (table.rank() != 2 || !table.dataType().isFPType()) {
                throw new IllegalArgumentException("Gemma 4 per_layer_token_embd.weight must be dequantized by the converter for gather");
            }
            matrix(weights, "per_layer_token_embd.weight", vocab, Math.multiplyExact(c.layers, c.ple));
            matrix(weights, "per_layer_model_proj.weight", Math.multiplyExact(c.layers, c.ple), c.hidden);
            vector(weights, "per_layer_proj_norm.weight", c.ple);
        }
        for (int l = 0; l < c.layers; l++) {
            String p = "blk." + l + ".";
            if (weights.containsKey(p + "ffn_gate_inp.weight")) {
                throw new IllegalArgumentException("Gemma 4 MoE is not implemented by the dense text importer");
            }
            int d = c.headDim(l);
            matrix(weights, p + "attn_q.weight", c.heads[l] * d, c.hidden);
            matrix(weights, p + "attn_output.weight", c.hidden, c.heads[l] * d);
            vector(weights, p + "attn_q_norm.weight", d);
            if (c.donor[l] == l) {
                matrix(weights, p + "attn_k.weight", c.kvHeads[l] * d, c.hidden);
                matrix(weights, p + "attn_v.weight", c.kvHeads[l] * d, c.hidden);
                vector(weights, p + "attn_k_norm.weight", d);
            }
            for (String norm : new String[]{"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"}) {
                vector(weights, p + norm + ".weight", c.hidden);
            }
            matrix(weights, p + "ffn_gate.weight", c.ffn[l], c.hidden);
            matrix(weights, p + "ffn_up.weight", c.ffn[l], c.hidden);
            matrix(weights, p + "ffn_down.weight", c.hidden, c.ffn[l]);
            vector(weights, p + "layer_output_scale.weight", 1);
            if (c.ple > 0) {
                matrix(weights, p + "inp_gate.weight", c.ple, c.hidden);
                matrix(weights, p + "proj.weight", c.hidden, c.ple);
                vector(weights, p + "post_norm.weight", c.hidden);
            }
        }
    }

    @Override
    public Map<String, String> getTensorNamePatterns() {
        // Keep GGUF weight identities stable in the graph, including PLE and direct-gamma norms.
        Map<String, String> names = new LinkedHashMap<>();
        for (String key : new String[]{"token_embd", "output", "output_norm", "per_layer_token_embd",
                "per_layer_model_proj", "per_layer_proj_norm", "rope_freqs"}) {
            names.put(key + ".weight", key + ".weight");
        }
        for (String key : new String[]{"attn_q", "attn_k", "attn_v", "attn_output", "attn_q_norm",
                "attn_k_norm", "attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm",
                "ffn_gate", "ffn_up", "ffn_down", "inp_gate", "proj", "post_norm", "layer_output_scale"}) {
            names.put("blk.{layer}." + key + ".weight", "blk.{layer}." + key + ".weight");
        }
        return names;
    }

    /** Metadata-only layout, also used by small contract tests without constructing a model. */
    static final class Layout {
        final int layers, hidden, localDim, globalDim, window, ple;
        final double epsilon, localTheta, globalTheta, softcap;
        final boolean[] local;
        final int[] heads, kvHeads, ffn, donor;

        Layout(GGMLMetadata m) {
            if (!"gemma4".equalsIgnoreCase(m.getArchitecture())) {
                throw new IllegalArgumentException("Expected exact gemma4 architecture");
            }
            if (m.getExpertCount() > 0) throw new IllegalArgumentException("Gemma 4 MoE requires a separate importer");
            Map<String, Object> raw = m.getRawMetadata();
            layers = positive(m.getNumLayers(), "block_count");
            hidden = positive(m.getHiddenSize(), "embedding_length");
            localDim = integer(raw, "attention.key_length_swa");
            globalDim = integer(raw, "attention.key_length");
            if ((localDim & 1) != 0 || (globalDim & 1) != 0
                    || localDim != integer(raw, "attention.value_length_swa")
                    || globalDim != integer(raw, "attention.value_length")
                    || localDim != integer(raw, "rope.dimension_count_swa")
                    || globalDim != integer(raw, "rope.dimension_count")) {
                throw new IllegalArgumentException("Gemma 4 requires equal even K/V/RoPE dimensions per attention type");
            }
            window = integer(raw, "attention.sliding_window");
            ple = nonnegative(number(raw, "embedding_length_per_layer_input"), "embedding_length_per_layer_input");
            epsilon = number(raw, "attention.layer_norm_rms_epsilon").doubleValue();
            localTheta = number(raw, "rope.freq_base_swa").doubleValue();
            globalTheta = number(raw, "rope.freq_base").doubleValue();
            softcap = raw.containsKey("gemma4.final_logit_softcapping")
                    ? number(raw, "final_logit_softcapping").doubleValue() : 0;
            if (!(epsilon > 0) || !Double.isFinite(epsilon) || !(localTheta > 0) || !Double.isFinite(localTheta)
                    || !(globalTheta > 0) || !Double.isFinite(globalTheta) || softcap < 0 || !Double.isFinite(softcap)) {
                throw new IllegalArgumentException("Invalid Gemma 4 norm/RoPE/softcap metadata");
            }
            int shared = raw.containsKey("gemma4.attention.shared_kv_layers")
                    ? nonnegative(number(raw, "attention.shared_kv_layers"), "attention.shared_kv_layers") : 0;
            if (shared >= layers) throw new IllegalArgumentException("Gemma 4 shared KV must have preceding donor layers");
            local = new boolean[layers];
            heads = new int[layers]; kvHeads = new int[layers]; ffn = new int[layers]; donor = new int[layers];
            Object pattern = raw.get("gemma4.attention.sliding_window_pattern");
            int[] last = {-1, -1};
            for (int l = 0; l < layers; l++) {
                Object flag = element(pattern, l, layers, "attention.sliding_window_pattern");
                if (!(flag instanceof Boolean)) throw new IllegalArgumentException("Gemma 4 sliding pattern must contain booleans");
                local[l] = (Boolean) flag;
                heads[l] = layerInteger(raw, "attention.head_count", l, layers);
                kvHeads[l] = layerInteger(raw, "attention.head_count_kv", l, layers);
                ffn[l] = layerInteger(raw, "feed_forward_length", l, layers);
                if (heads[l] % kvHeads[l] != 0) throw new IllegalArgumentException("Gemma 4 query heads must divide into KV groups");
                int type = local[l] ? 0 : 1;
                if (l < layers - shared) {
                    donor[l] = l;
                    last[type] = l;
                } else {
                    donor[l] = last[type];
                    if (donor[l] < 0 || kvHeads[l] != kvHeads[donor[l]]) {
                        throw new IllegalArgumentException("Gemma 4 layer " + l + " has no compatible same-type KV donor");
                    }
                }
            }
        }
        int headDim(int layer) { return local[layer] ? localDim : globalDim; }
    }

    private static Number number(Map<String, Object> raw, String key) {
        Object value = raw.get("gemma4." + key);
        if (!(value instanceof Number)) throw new IllegalArgumentException("Missing/nonnumeric Gemma 4 metadata: " + key);
        return (Number)value;
    }
    private static int nonnegative(Number value, String key) {
        double d = value.doubleValue();
        if (!Double.isFinite(d) || d < 0 || d > Integer.MAX_VALUE || d != Math.rint(d)) {
            throw new IllegalArgumentException("Invalid Gemma 4 integer metadata: " + key);
        }
        return (int)d;
    }
    private static int positive(int value, String key) {
        if (value <= 0) throw new IllegalArgumentException("Gemma 4 requires positive " + key);
        return value;
    }
    private static int integer(Map<String, Object> raw, String key) {
        return positive(nonnegative(number(raw, key), key), key);
    }
    private static int layerInteger(Map<String, Object> raw, String key, int layer, int layers) {
        Object v = element(raw.get("gemma4." + key), layer, layers, key);
        if (!(v instanceof Number)) throw new IllegalArgumentException("Missing/nonnumeric Gemma 4 metadata: " + key);
        return positive(nonnegative((Number)v, key), key);
    }
    private static Object element(Object value, int layer, int layers, String key) {
        if (value instanceof Number || value instanceof Boolean) return value;
        // GGUFReader uses primitive arrays; programmatic metadata may also use lists.
        if (value instanceof int[] && ((int[])value).length == layers) return ((int[])value)[layer];
        if (value instanceof long[] && ((long[])value).length == layers) return ((long[])value)[layer];
        if (value instanceof boolean[] && ((boolean[])value).length == layers) return ((boolean[])value)[layer];
        if (value instanceof Object[] && ((Object[])value).length == layers) return ((Object[])value)[layer];
        if (value instanceof List && ((List<?>)value).size() == layers) return ((List<?>)value).get(layer);
        throw new IllegalArgumentException("Gemma 4 " + key + " must be scalar or have " + layers + " entries");
    }
}
