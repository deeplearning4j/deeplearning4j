/*
 * Copyright (c) Eclipse Deeplearning4j Contributors
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.ggml.architecture;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.ggml.format.GGMLMetadata;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** No model files, downloads, reflection, or execution-mode overrides. Run on the selected backend. */
class Gemma4ArchitectureContractTest {
    private static final int L = 4, H = 4, P = 2, VOCAB = 7, HEADS = 2;
    private static final double EPS = 1e-6;
    private static final boolean[] LOCAL = {true, false, true, false};

    @Test
    void e2bMetadataUsesTwoDonorsAndTwoHeadWidths() {
        GGMLMetadata m = metadata(35, 1536, 256, 512, 256, 20);
        boolean[] pattern = new boolean[35];
        int[] ffn = new int[35];
        for (int l = 0; l < 35; l++) {
            pattern[l] = l % 5 != 4;
            ffn[l] = l < 15 ? 6144 : 12288;
        }
        m.getRawMetadata().put("gemma4.attention.sliding_window_pattern", pattern);
        m.getRawMetadata().put("gemma4.feed_forward_length", ffn);
        m.getRawMetadata().put("gemma4.attention.head_count", 8);
        m.getRawMetadata().put("gemma4.attention.sliding_window", 512);
        Gemma4Architecture.Layout c = new Gemma4Architecture.Layout(m);
        for (int l = 0; l < 35; l++) {
            assertEquals(pattern[l] ? 256 : 512, c.headDim(l));
            assertEquals(l < 15 ? l : (pattern[l] ? 13 : 14), c.donor[l]);
            assertEquals(ffn[l], c.ffn[l]);
        }
        assertFalse(new GemmaArchitecture().getConfig(m).isUseSwiGLU());
        assertEquals(1.0, new GemmaArchitecture().getConfig(m).getAttentionMultiplier());
        // The new path must not absorb other Gemma variants.
        for (String arch : new String[]{"gemma", "gemma2", "gemma3", "gemma3n"}) {
            m.setArchitecture(arch);
            assertFalse(new Gemma4Architecture().canHandle(m));
            assertTrue(new GemmaArchitecture().getConfig(m).isUseSwiGLU());
        }
    }

    @Test
    void graphContractCreatesSharedAliasesAndConsumesEveryNorm() {
        Map<String, INDArray> w = weights(H);
        try (SameDiff sd = build(w, DataType.FLOAT)) {
            for (int l = 0; l < L; l++) {
                int d = LOCAL[l] ? 2 : 4;
                assertArrayEquals(new long[]{-1, -1, 1, d}, sd.getVariable("past_key_values." + l + ".key").getShape());
                assertArrayEquals(new long[]{-1, -1, 1, d}, sd.getVariable("past_key_values." + l + ".value").getShape());
                SameDiffOp attn = producer(sd, "attn_out_" + l);
                assertEquals("dot_product_attention_v2", attn.getOp().opName());
                assertEquals(Arrays.asList("q_rope_" + l, "v_heads_" + l, "k_rope_" + l),
                        attn.getInputsToOp().subList(0, 3));
                assertEquals("cache_position", attn.getInputsToOp().get(7));
                assertEquals(LOCAL[l] ? "gemma4_local_mask" : "_causal_mask", attn.getInputsToOp().get(8));
                assertEquals(1.0, ((DynamicCustomOp)attn.getOp()).tArgs()[0]);
                if (l >= 2) {
                    assertEquals(List.of("k_rope_" + (l - 2)), producer(sd, "k_rope_" + l).getInputsToOp());
                    assertEquals(List.of("v_heads_" + (l - 2)), producer(sd, "v_heads_" + l).getInputsToOp());
                    // GGUF retains these tensors, but reference shared layers must not project them.
                    assertNull(sd.getVariable("blk." + l + ".attn_k.weight"));
                    assertNull(sd.getVariable("blk." + l + ".attn_v.weight"));
                }
                for (String norm : new String[]{"attn_norm", "attn_q_norm", "post_attention_norm",
                        "ffn_norm", "post_ffw_norm", "post_norm"}) {
                    assertFalse(sd.getVariables().get("blk." + l + "." + norm + ".weight").getInputsForOp().isEmpty());
                }
                assertNotNull(sd.getVariable("ffn_gelu_" + l));
                assertNotNull(sd.getVariable("ple_gelu_" + l));
                assertNotNull(sd.getVariable("layer_out_" + l));
            }
            assertEquals(2 * L, sd.getOps().values().stream()
                    .filter(op -> "precise_gelu".equals(op.getOp().opName())).count());
            assertEquals(DataType.FLOAT, sd.getVariable("logits").dataType());
        }
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT", "HALF", "BFLOAT16"})
    void normAccumulatesInFloatAndUsesDirectGamma(DataType dtype) {
        try (SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", dtype, -1, 4);
            INDArray gamma = Nd4j.createFromArray(0.0f, 0.7f, -1.2f, 1.3f);
            Gemma4Architecture.rmsNorm(sd, "norm", x, sd.constant("gamma", gamma), EPS);
            Gemma4Architecture.rmsNorm(sd, "weightless", x, null, EPS);
            Gemma4Architecture.gelu(sd, "gelu", x);
            // Squaring these HALF values without promotion overflows.
            INDArray input = Nd4j.createFromArray(300f, -500f, 1000f, 0.1f, -2f, -0.5f, 0.5f, 2f)
                    .reshape(2, 4).castTo(dtype);
            Map<String, INDArray> result = sd.output(Map.of("x", input), "norm", "weightless", "gelu");
            double[] source = doubles(input);
            assertValues(norm(source, doubles(gamma), 4), result.get("norm"), dtype == DataType.FLOAT ? 2e-5 : 0.02);
            assertValues(norm(source, null, 4), result.get("weightless"), dtype == DataType.FLOAT ? 2e-5 : 0.02);
            assertValues(gelu(source), result.get("gelu"), dtype == DataType.FLOAT ? 2e-5 : 0.02);
            assertEquals(dtype, result.get("norm").dataType());
            assertEquals(dtype, result.get("weightless").dataType());
            assertEquals(0.0, result.get("norm").getDouble(0, 0));
        }
    }

    @Test
    void localMaskKeepsPaddingAndUsesCacheNotRopePosition() {
        try (SameDiff sd = SameDiff.create()) {
            SDVariable mask = sd.placeHolder("mask", DataType.FLOAT, -1, -1, -1, -1);
            SDVariable pos = sd.placeHolder("cache", DataType.INT64);
            Gemma4Architecture.slidingMask(sd, mask, pos, sd.sizeAt(mask, 2), 2);
            // Unequal batch/head/query sizes expose accidental axis broadcasting.
            for (int offset : new int[]{0, 3}) {
                INDArray bias = Nd4j.zeros(DataType.FLOAT, 2, 3, 2, 7);
                bias.putScalar(new long[]{1, 2, 1, 4}, -77.0);
                INDArray actual = sd.outputSingle(Map.of("mask", bias, "cache", Nd4j.scalar((long)offset)), "gemma4_local_mask");
                assertArrayEquals(bias.shape(), actual.shape());
                for (int b = 0; b < 2; b++) for (int h = 0; h < 3; h++)
                    for (int q = 0; q < 2; q++) for (int k = 0; k < 7; k++) {
                        double v = actual.getDouble(b, h, q, k);
                        if (k <= offset + q - 2) assertTrue(v < -1e30);
                        else assertEquals(bias.getDouble(b, h, q, k), v);
                    }
            }
        }
    }

    @Test
    void denseGraphMatchesIndependentReferenceAcrossDynamicShapes() {
        Map<String, INDArray> w = weights(H);
        try (SameDiff sd = build(w, DataType.FLOAT)) {
            int[][][] cases = {{{2, 1, 3}, {4, 2, 5}}, {{5, 3, 1, 6, 2}}};
            for (int[][] ids : cases) {
                // Position offset differs from cache position; proportional global factors are [2, 1e30].
                Map<String, double[]> expected = reference(ids, 7, w, true);
                Map<String, INDArray> feed = feed(ids, 7, 0, 8, true, true);
                Map<String, INDArray> actual = sd.output(feed, expected.keySet().toArray(new String[0]));
                expected.forEach((name, value) -> assertValues(value, actual.get(name), 5e-4));
                for (int l = 0; l < L; l++) {
                    int d = LOCAL[l] ? 2 : 4;
                    assertArrayEquals(new long[]{ids.length, ids[0].length, HEADS, d}, actual.get("q_rope_" + l).shape());
                    assertArrayEquals(new long[]{ids.length, ids[0].length, 1, d}, actual.get("k_rope_" + l).shape());
                }
                assertArrayEquals(new long[]{ids.length, ids[0].length, L, P}, actual.get("per_layer_inputs").shape());
            }
        }
    }

    @Test
    void finalSoftcapBoundsLargeUntiedLogits() {
        Map<String, INDArray> w = weights(H);
        w.put("output.weight", w.get("token_embd.weight").mul(1000));
        int[][] ids = {{2, 4, 5}};
        Map<String, double[]> expected = reference(ids, 0, w, false);
        assertTrue(Arrays.stream(expected.get("logits_uncapped")).anyMatch(v -> Math.abs(v) > 100));
        try (SameDiff sd = build(w, DataType.FLOAT)) {
            INDArray actual = sd.outputSingle(feed(ids, 0, 0, 8, false, true), "logits");
            assertValues(expected.get("logits"), actual, 5e-4);
            for (double v : doubles(actual)) assertTrue(Math.abs(v) <= 30);
        }
    }

    @Test
    void cachedDecodeMatchesFullReferencePastWindowBoundary() {
        Map<String, INDArray> w = weights(H);
        int[][] all = {{2, 3, 4, 1, 5, 6}};
        Map<String, double[]> full = reference(all, 0, w, false);
        try (SameDiff sd = build(w, DataType.FLOAT)) {
            Map<String, INDArray> feed = feed(new int[][]{{2, 3}}, 0, 0, 8, false, false);
            INDArray prefill = sd.outputSingle(feed, "logits");
            assertValues(Arrays.copyOfRange(full.get("logits"), 0, 2 * VOCAB), prefill, 5e-4);
            for (int pos = 2; pos < all[0].length; pos++) {
                Map<String, INDArray> next = feed(new int[][]{{all[0][pos]}}, pos, pos, 8, false, false);
                // Reuse the exact buffers written by the preceding invocation.
                for (int l = 0; l < L; l++) for (String kind : new String[]{"key", "value"}) {
                    String key = "past_key_values." + l + "." + kind;
                    next.put(key, feed.get(key));
                }
                INDArray actual = sd.outputSingle(next, "logits");
                assertValues(Arrays.copyOfRange(full.get("logits"), pos * VOCAB, (pos + 1) * VOCAB), actual, 5e-4);
                feed = next;
            }
        }
    }

    @Test
    void rejectsMissingRequiredWeightsAndMalformedLayout() {
        for (String missing : new String[]{"per_layer_model_proj.weight", "per_layer_token_embd.weight",
                "per_layer_proj_norm.weight", "rope_freqs.weight", "output_norm.weight", "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight", "blk.0.attn_v.weight", "blk.0.post_attention_norm.weight",
                "blk.0.post_ffw_norm.weight", "blk.0.post_norm.weight", "blk.0.proj.weight",
                "blk.0.layer_output_scale.weight"}) {
            Map<String, INDArray> w = weights(H);
            w.remove(missing);
            IllegalArgumentException failure = assertThrows(IllegalArgumentException.class, () -> build(w, DataType.FLOAT));
            assertTrue(failure.getMessage().contains(missing), failure.getMessage());
        }
        GGMLMetadata m = tinyMetadata(H);
        m.getRawMetadata().put("gemma4.attention.sliding_window_pattern", new boolean[]{true});
        assertThrows(IllegalArgumentException.class, () -> new Gemma4Architecture.Layout(m));
        m.getRawMetadata().put("gemma4.attention.sliding_window_pattern", new boolean[]{true, true, false, true});
        assertThrows(IllegalArgumentException.class, () -> new Gemma4Architecture.Layout(m), "global shared layer needs global donor");
    }

    @Test
    void packedQkvRemainPackedWithLogicalShapes() {
        Map<String, INDArray> w = weights(32);
        for (String projection : new String[]{"attn_q", "attn_k", "attn_v"}) {
            String key = "blk.0." + projection + ".weight";
            int out = projection.equals("attn_q") ? 4 : 2;
            // One Q8_0 block (fp16 scale + 32 signed bytes) per output row.
            w.put(key, Nd4j.createFromArray(new byte[out * 34]));
            w.put(key + ".__q__", Nd4j.createFromArray(8L, (long)out, 32L));
        }
        try (SameDiff sd = new GemmaArchitecture().buildGraph(tinyMetadata(32), w,
                ConversionOptions.builder().targetDataType(DataType.HALF).build())) {
            for (String prefix : new String[]{"q", "k", "v"}) {
                SameDiffOp op = producer(sd, prefix + "_proj_0_fp32");
                assertEquals("ggml_qmatmul", op.getOp().opName());
                SDVariable packed = sd.getVariable(op.getInputsToOp().get(1));
                assertEquals(DataType.BYTE, packed.dataType());
                assertEquals(DataType.HALF, sd.getVariable(prefix + "_proj_0").dataType());
            }
            assertArrayEquals(new long[]{-1, -1, 1, 2}, sd.getVariable("past_key_values.0.key").getShape());
            assertArrayEquals(new long[]{-1, -1, 1, 4}, sd.getVariable("past_key_values.1.key").getShape());
        }
    }

    private static SameDiffOp producer(SameDiff sd, String variable) {
        assertNotNull(sd.getVariable(variable), variable);
        return sd.getOps().get(sd.getVariables().get(variable).getOutputOfOp());
    }

    private static SameDiff build(Map<String, INDArray> w, DataType dtype) {
        return new GemmaArchitecture().buildGraph(tinyMetadata(H), w,
                ConversionOptions.builder().targetDataType(dtype).build());
    }

    private static GGMLMetadata tinyMetadata(int hidden) {
        GGMLMetadata m = metadata(L, hidden, 2, 4, P, 2);
        m.getRawMetadata().put("gemma4.attention.sliding_window_pattern", LOCAL.clone());
        m.getRawMetadata().put("gemma4.feed_forward_length", new int[]{6, 6, 8, 8});
        return m;
    }

    private static GGMLMetadata metadata(int layers, int hidden, int localDim, int globalDim, int ple, int shared) {
        Map<String, Object> raw = new HashMap<>();
        raw.put("gemma4.attention.key_length_swa", localDim);
        raw.put("gemma4.attention.value_length_swa", localDim);
        raw.put("gemma4.rope.dimension_count_swa", localDim);
        raw.put("gemma4.attention.key_length", globalDim);
        raw.put("gemma4.attention.value_length", globalDim);
        raw.put("gemma4.rope.dimension_count", globalDim);
        raw.put("gemma4.attention.head_count", HEADS);
        raw.put("gemma4.attention.head_count_kv", 1);
        raw.put("gemma4.attention.sliding_window", 2);
        raw.put("gemma4.embedding_length_per_layer_input", ple);
        raw.put("gemma4.attention.shared_kv_layers", shared);
        raw.put("gemma4.attention.layer_norm_rms_epsilon", EPS);
        raw.put("gemma4.rope.freq_base_swa", 10000.0);
        raw.put("gemma4.rope.freq_base", 1000000.0);
        raw.put("gemma4.final_logit_softcapping", 30.0);
        return GGMLMetadata.builder().architecture("gemma4").numLayers(layers).hiddenSize(hidden)
                .numAttentionHeads(HEADS).numKVHeads(1).vocabSize(VOCAB).contextLength(32)
                .attentionKeyLength(globalDim).layerNormEpsilon((float)EPS).rawMetadata(raw).build();
    }

    private static Map<String, INDArray> weights(int hidden) {
        Map<String, INDArray> w = new LinkedHashMap<>();
        put(w, "token_embd.weight", VOCAB, hidden);
        put(w, "per_layer_token_embd.weight", VOCAB, L * P);
        put(w, "per_layer_model_proj.weight", L * P, hidden);
        put(w, "per_layer_proj_norm.weight", P);
        put(w, "output_norm.weight", hidden);
        w.put("rope_freqs.weight", Nd4j.createFromArray(2f, 1e30f));
        for (int l = 0; l < L; l++) {
            String p = "blk." + l + ".";
            int d = LOCAL[l] ? 2 : 4, ffn = l < 2 ? 6 : 8;
            put(w, p + "attn_q.weight", HEADS * d, hidden);
            put(w, p + "attn_k.weight", d, hidden);
            put(w, p + "attn_v.weight", d, hidden);
            put(w, p + "attn_output.weight", hidden, HEADS * d);
            put(w, p + "attn_q_norm.weight", d);
            put(w, p + "attn_k_norm.weight", d);
            for (String norm : new String[]{"attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm", "post_norm"}) {
                put(w, p + norm + ".weight", hidden);
            }
            put(w, p + "ffn_gate.weight", ffn, hidden);
            put(w, p + "ffn_up.weight", ffn, hidden);
            put(w, p + "ffn_down.weight", hidden, ffn);
            put(w, p + "inp_gate.weight", P, hidden);
            put(w, p + "proj.weight", hidden, P);
            w.put(p + "layer_output_scale.weight", Nd4j.createFromArray(0.65f + 0.15f * l));
        }
        return w;
    }

    private static void put(Map<String, INDArray> w, String name, int... shape) {
        int n = 1;
        for (int d : shape) n *= d;
        float[] data = new float[n];
        int seed = Math.floorMod(name.hashCode(), 1000);
        for (int i = 0; i < n; i++) data[i] = (float)(Math.sin(seed + 0.73 * i + 0.019 * i * i) * 0.4 + (shape.length == 1 ? 0.8 : 0));
        long[] dims = Arrays.stream(shape).asLongStream().toArray();
        w.put(name, Nd4j.createFromArray(data).reshape(dims));
    }

    private static Map<String, INDArray> feed(int[][] ids, int offset, int cache, int capacity,
                                             boolean padding, boolean emptyCaches) {
        int b = ids.length, s = ids[0].length;
        long[] tokens = new long[b * s];
        for (int i = 0; i < b; i++) for (int j = 0; j < s; j++) tokens[i * s + j] = ids[i][j];
        Map<String, INDArray> f = new HashMap<>();
        f.put("input_ids", Nd4j.createFromArray(tokens).reshape(b, s));
        f.put("position_offset", Nd4j.scalar((long)offset));
        f.put("cache_position", Nd4j.scalar((long)cache));
        float[] bias = new float[b * s * capacity];
        for (int i = 0; i < b; i++) for (int q = 0; q < s; q++) for (int k = 0; k < capacity; k++) {
            if (k > cache + q || (padding && i == 0 && k == 1)) bias[(i * s + q) * capacity + k] = -1e9f;
        }
        f.put("_causal_mask", Nd4j.createFromArray(bias).reshape(b, 1, s, capacity));
        for (int l = 0; l < L; l++) for (String kind : new String[]{"key", "value"}) {
            f.put("past_key_values." + l + "." + kind,
                    Nd4j.zeros(DataType.FLOAT, b, emptyCaches ? 0 : capacity, 1, LOCAL[l] ? 2 : 4));
        }
        return f;
    }

    /** Straight loops on doubles; deliberately does not use importer helpers or SameDiff ops. */
    private static Map<String, double[]> reference(int[][] ids, int offset, Map<String, INDArray> arrays, boolean padding) {
        Map<String, double[]> w = new HashMap<>(), result = new LinkedHashMap<>();
        arrays.forEach((name, value) -> w.put(name, doubles(value)));
        int batch = ids.length, seq = ids[0].length, rows = batch * seq;
        double[] hidden = new double[rows * H], tokenPle = new double[rows * L * P];
        for (int b = 0; b < batch; b++) for (int s = 0; s < seq; s++) {
            for (int h = 0; h < H; h++) hidden[(b * seq + s) * H + h] = w.get("token_embd.weight")[ids[b][s] * H + h] * Math.sqrt(H);
            for (int p = 0; p < L * P; p++) tokenPle[(b * seq + s) * L * P + p] = w.get("per_layer_token_embd.weight")[ids[b][s] * L * P + p] * Math.sqrt(P);
        }
        double[] ple = scale(add(norm(scale(linear(hidden, w.get("per_layer_model_proj.weight"), H), 1 / Math.sqrt(H)),
                w.get("per_layer_proj_norm.weight"), P), tokenPle), 1 / Math.sqrt(2));
        result.put("per_layer_inputs", ple);
        List<double[]> keys = new ArrayList<>(), values = new ArrayList<>();
        for (int l = 0; l < L; l++) {
            String p = "blk." + l + ".";
            int d = LOCAL[l] ? 2 : 4;
            double[] n = norm(hidden, w.get(p + "attn_norm.weight"), H);
            result.put("attn_norm_" + l, n);
            double[] q = norm(linear(n, w.get(p + "attn_q.weight"), H), w.get(p + "attn_q_norm.weight"), d);
            result.put("q_norm_" + l, q);
            q = rotate(q, batch, seq, HEADS, d, offset, LOCAL[l], w.get("rope_freqs.weight"));
            double[] k, v;
            if (l < 2) {
                k = norm(linear(n, w.get(p + "attn_k.weight"), H), w.get(p + "attn_k_norm.weight"), d);
                k = rotate(k, batch, seq, 1, d, offset, LOCAL[l], w.get("rope_freqs.weight"));
                v = norm(linear(n, w.get(p + "attn_v.weight"), H), null, d);
            } else { k = keys.get(l - 2); v = values.get(l - 2); }
            keys.add(k); values.add(v);
            result.put("q_rope_" + l, q); result.put("k_rope_" + l, k); result.put("v_heads_" + l, v);
            double[] attn = attention(q, k, v, batch, seq, d, LOCAL[l], padding);
            attn = norm(linear(attn, w.get(p + "attn_output.weight"), HEADS * d), w.get(p + "post_attention_norm.weight"), H);
            result.put("post_attention_norm_" + l, attn);
            hidden = add(hidden, attn);
            n = norm(hidden, w.get(p + "ffn_norm.weight"), H);
            double[] gate = gelu(linear(n, w.get(p + "ffn_gate.weight"), H));
            result.put("ffn_gelu_" + l, gate);
            double[] ffn = linear(mul(gate, linear(n, w.get(p + "ffn_up.weight"), H)), w.get(p + "ffn_down.weight"), l < 2 ? 6 : 8);
            ffn = norm(ffn, w.get(p + "post_ffw_norm.weight"), H);
            result.put("post_ffw_norm_" + l, ffn);
            hidden = add(hidden, ffn);
            double[] layerPle = new double[rows * P];
            for (int r = 0; r < rows; r++) for (int j = 0; j < P; j++) layerPle[r * P + j] = ple[(r * L + l) * P + j];
            gate = gelu(linear(hidden, w.get(p + "inp_gate.weight"), H));
            result.put("ple_gelu_" + l, gate);
            double[] correction = norm(linear(mul(gate, layerPle), w.get(p + "proj.weight"), P), w.get(p + "post_norm.weight"), H);
            result.put("post_norm_" + l, correction);
            hidden = scale(add(hidden, correction), w.get(p + "layer_output_scale.weight")[0]);
            result.put("layer_out_" + l, hidden);
        }
        hidden = norm(hidden, w.get("output_norm.weight"), H);
        result.put("final_norm", hidden);
        double[] raw = linear(hidden, w.getOrDefault("output.weight", w.get("token_embd.weight")), H);
        result.put("logits_uncapped", raw);
        double[] logits = raw.clone();
        for (int i = 0; i < logits.length; i++) logits[i] = 30 * Math.tanh(logits[i] / 30);
        result.put("logits", logits);
        return result;
    }

    private static double[] attention(double[] q, double[] k, double[] v, int batch, int seq, int d,
                                      boolean local, boolean padding) {
        double[] out = new double[q.length];
        for (int b = 0; b < batch; b++) for (int s = 0; s < seq; s++) for (int h = 0; h < HEADS; h++) {
            double[] scores = new double[seq];
            double max = -Double.MAX_VALUE;
            for (int t = 0; t < seq; t++) {
                double dot = 0;
                for (int j = 0; j < d; j++) dot += q[((b * seq + s) * HEADS + h) * d + j] * k[(b * seq + t) * d + j];
                if (t > s || (local && t <= s - 2) || (padding && b == 0 && t == 1)) dot = -1e9;
                scores[t] = dot; max = Math.max(max, dot);
            }
            double sum = 0;
            for (int t = 0; t < seq; t++) { scores[t] = Math.exp(scores[t] - max); sum += scores[t]; }
            for (int j = 0; j < d; j++) for (int t = 0; t < seq; t++)
                out[((b * seq + s) * HEADS + h) * d + j] += scores[t] / sum * v[(b * seq + t) * d + j];
        }
        return out;
    }

    private static double[] rotate(double[] x, int batch, int seq, int heads, int d, int offset, boolean local, double[] factors) {
        double[] out = x.clone();
        for (int b = 0; b < batch; b++) for (int s = 0; s < seq; s++) for (int h = 0; h < heads; h++)
            for (int i = 0; i < d / 2; i++) {
                int a = ((b * seq + s) * heads + h) * d + i, z = a + d / 2;
                double angle = (offset + s) * Math.pow(local ? 10000 : 1000000, -2.0 * i / d) / (local ? 1 : factors[i]);
                out[a] = x[a] * Math.cos(angle) - x[z] * Math.sin(angle);
                out[z] = x[z] * Math.cos(angle) + x[a] * Math.sin(angle);
            }
        return out;
    }

    private static double[] linear(double[] x, double[] w, int in) {
        int width = w.length / in;
        double[] out = new double[x.length / in * width];
        for (int r = 0; r < x.length / in; r++) for (int o = 0; o < width; o++)
            for (int i = 0; i < in; i++) out[r * width + o] += x[r * in + i] * w[o * in + i];
        return out;
    }
    private static double[] norm(double[] x, double[] gamma, int width) {
        double[] out = new double[x.length];
        for (int r = 0; r < x.length / width; r++) {
            double sum = 0;
            for (int i = 0; i < width; i++) sum += x[r * width + i] * x[r * width + i];
            double scale = 1 / Math.sqrt(sum / width + EPS);
            for (int i = 0; i < width; i++) out[r * width + i] = x[r * width + i] * scale * (gamma == null ? 1 : gamma[i]);
        }
        return out;
    }
    private static double[] gelu(double[] x) {
        double[] out = x.clone();
        for (int i = 0; i < x.length; i++) out[i] = 0.5 * x[i] * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x[i] + 0.044715 * x[i] * x[i] * x[i])));
        return out;
    }
    private static double[] add(double[] a, double[] b) {
        double[] out = a.clone();
        for (int i = 0; i < out.length; i++) out[i] += b[i];
        return out;
    }
    private static double[] mul(double[] a, double[] b) {
        double[] out = a.clone();
        for (int i = 0; i < out.length; i++) out[i] *= b[i];
        return out;
    }
    private static double[] scale(double[] a, double factor) {
        double[] out = a.clone();
        for (int i = 0; i < out.length; i++) out[i] *= factor;
        return out;
    }
    private static double[] doubles(INDArray array) {
        float[] values = array.castTo(DataType.FLOAT).dup('c').data().asFloat();
        double[] out = new double[values.length];
        for (int i = 0; i < out.length; i++) out[i] = values[i];
        return out;
    }
    private static void assertValues(double[] expected, INDArray actual, double tolerance) {
        assertNotNull(actual);
        double[] observed = doubles(actual);
        assertEquals(expected.length, observed.length);
        for (int i = 0; i < expected.length; i++)
            assertEquals(expected[i], observed[i], tolerance * Math.max(1, Math.abs(expected[i])), "element " + i);
    }
}
