/*
 * SPDX-License-Identifier: Apache-2.0
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0.
 * See the NOTICE file distributed with this work for copyright ownership.
 */
package org.eclipse.deeplearning4j.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.BytePointer;
import org.eclipse.deeplearning4j.safetensors.ModelOptQwenConfig;
import org.eclipse.deeplearning4j.llm.tokenizer.HuggingFaceTokenizer;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsHeader;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsHeader.TensorInfo;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.LLaMAArchitecture;
import org.nd4j.ggml.architecture.QuantizedLinear;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.blas.params.MMulTranspose;
import org.nd4j.linalg.api.ops.impl.reduce.Mmul;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.Reader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/**
 * Cache-only first-layer isolation; NOT a whole-model causality test.
 * The original RMS/dense-gate probe is retained; the separate full-block test below
 * invokes the shared architecture builders and observes the recurrence boundary.
 * proc-028 (ad16b002-b950-45fb-aa08-928b1efcbf31) first diverged at token 110,
 * but layer-zero GDN state already differed after the fully accepted first W=5.
 * Its five input IDs are used here without loading the 2.54 GB embedding tensor.
 *
 * Scope: the optimized native RMS -> dense alpha/beta branch only. The production
 * ModelOptQwenImporter adapts gamma to FLOAT (1 + stored BF16), enables
 * multiplyNormInFloat, and retains dense projection weights in BF16.
 * LLaMAArchitecture.buildGatedDeltaNet uses QuantizedLinear then BF16 sigmoid
 * for beta; alpha is cast back to FLOAT AFTER its BF16 projection store.
 * QuantizedLinear's actual pre-store *_accum node is observed, not a second GEMM.
 * RmsNorm is the fused normalization seen in proc-028, not a reimplementation
 * of the private architecture builder or a fabricated one-layer model config.
 *
 * QKV/FP8, convolution, Q/K L2 normalization, gate-decay and recurrent state are
 * deliberately outside this narrow slice. A pass proves only this slice for
 * these rows/shapes; it does not clear the complete frontend or explain token 110.
 * Observing intermediate outputs can alter fusion boundaries. DSP remains enabled
 * with its normal lifecycle; this does not claim the full model's replay topology.
 */
@Slf4j
@EnabledIfSystemProperty(named = "qwen.nvfp4.windowParity", matches = "true")
public class TestQwenNvfp4WindowParity {
    private static final String REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
    private static final String PREFIX = "nvidia-Qwen3.6-27B-NVFP4-" + REVISION + "-";
    private static final String TEXT = "model.language_model.";
    private static final String LAYER = TEXT + "layers.0.";
    private static final int[] IDS = {579, 264, 7047, 1817, 25};
    private static final String[] PROJECTIONS = {
            "alpha_accum", "alpha", "alpha_compute",
            "beta_accum", "beta", "beta_sigmoid", "beta_compute"
    };

    @Test
    void firstLayerRmsAndDenseGatesWindowVersusScalar() throws Exception {
        File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        JsonObject textConfig = json(cached(cache, "config.json")).getAsJsonObject("text_config");
        JsonObject index = json(cached(cache, "model.safetensors.index.json")).getAsJsonObject("weight_map");
        int hidden = textConfig.get("hidden_size").getAsInt();
        int heads = textConfig.get("linear_num_value_heads").getAsInt();
        double epsilon = textConfig.get("rms_norm_eps").getAsDouble();
        assertEquals(5120, hidden, "Pinned checkpoint hidden size");
        assertEquals(48, heads, "Pinned checkpoint value heads");
        assertTrue(Double.isFinite(epsilon) && epsilon > 0);
        List<String> failures = new ArrayList<>();
        try (INDArray rows = embeddingRows(cache, index, hidden);
             INDArray rawGamma = smallTensor(cache, index, LAYER + "input_layernorm.weight", hidden);
             INDArray alpha = smallTensor(cache, index, LAYER + "linear_attn.in_proj_a.weight", heads, hidden);
             INDArray beta = smallTensor(cache, index, LAYER + "linear_attn.in_proj_b.weight", heads, hidden);
             INDArray gamma = oneCenteredGamma(rawGamma);
             INDArray normalized = Nd4j.create(DataType.BFLOAT16, 1, IDS.length, hidden)) {
            log.info("WINDOW_PARITY source=nvidia/Qwen3.6-27B-NVFP4@{} ids={} epsilon={} "
                    + "scope=first-layer-RMS-dense-alpha-beta", REVISION, Arrays.toString(IDS), epsilon);
            Snapshot original = snapshot(rows);
            Nd4j.exec(new RmsNorm(rows, gamma, normalized, epsilon));
            Snapshot nativeWindow = snapshot(normalized);
            List<Snapshot> nativeScalars = new ArrayList<>();
            for (int i = 0; i < IDS.length; i++) {
                try (INDArray row = rowCopy(rows, i);
                     INDArray out = Nd4j.exec(new RmsNorm(row, gamma, epsilon))[0]) {
                    nativeScalars.add(snapshot(out));
                }
            }
            compare("native/rms W5-vs-5xW1", nativeWindow, join(nativeScalars), failures);
            // Same cached rows feed both shapes. No recurrent/conv state or position is involved.
            exercise("frontend", rows, gamma, alpha, beta, epsilon, true, failures);
            // Fix the *entire* projection input, independently of any RMS cross-shape drift.
            // Every scalar input is a bit-checked copy of this one normalized W5 result.
            exercise("isolated-projections/common-native-RMS", normalized, gamma, alpha, beta,
                    epsilon, false, failures);
            compare("embedding/input-not-mutated", original, snapshot(rows), failures);
        }
        assertTrue(failures.isEmpty(), () -> "First upstream drift in diagnostic order: "
                + failures.get(0) + "\nAll observed mismatches:\n" + String.join("\n", failures));
        log.info("WINDOW_PARITY exact parity at BF16 rounding boundaries and downstream gates; "
                + "FLOAT accumulators satisfy independent forward-error bounds. "
                + "No claim about FP8 QKV, conv, L2, GDN state, or full-model generation");
    }

    /**
     * Full shared layer-zero attention, not a synthetic checkpoint or copied block math.
     * Outputs deliberately expose fusion boundaries: a pass does not certify the full-model
     * optimizer/replay topology or locate the later token-117 divergence by itself.
     */
    @Test
    void firstLayerFullGdnAttentionWindowVersusScalar() throws Exception {
        fullBlockParity(false);
    }

    @Test
    void optimizedFirstLayerGdnWindowVersusScalar() throws Exception {
        fullBlockParity(true);
    }

    private void fullBlockParity(boolean optimized) throws Exception {
        File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        ModelOptQwenConfig config = ModelOptQwenConfig.read(cached(cache, "config.json"),
                cached(cache, "hf_quant_config.json"), cached(cache, "generation_config.json"));
        assertEquals(5120, config.getArchitecture().getHiddenSize());
        assertEquals(16, config.getKeyHeads());
        assertEquals(48, config.getValueHeads());
        assertEquals(128, config.getKeyHeadDim());
        assertEquals(128, config.getValueHeadDim());
        assertEquals(10240, config.getConvChannels());
        assertEquals(4, config.getConvKernel()); // state is [1,10240,3], weight [10240,4]
        JsonObject index = json(cached(cache, "model.safetensors.index.json")).getAsJsonObject("weight_map");
        List<String> failures = new ArrayList<>();
        try (BlockWeights weights = new BlockWeights(cache, index, config);
             INDArray rows = embeddingRows(cache, index, config.getArchitecture().getHiddenSize());
             INDArray initial = Nd4j.zeros(DataType.FLOAT, 1, 48, 128, 128);
             INDArray convInitial = Nd4j.zeros(DataType.BFLOAT16, 1, 10240, 3);
             SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", DataType.BFLOAT16, 1, -1, 5120);
            SDVariable state = sd.placeHolder("state", DataType.FLOAT, 1, 48, 128, 128);
            SDVariable conv = sd.placeHolder("conv", DataType.BFLOAT16, 1, 10240, 3);
            SDVariable length = sd.placeHolder("actual_sequence_length", DataType.INT64);
            SDVariable attention = new FirstGdnBlock(config).build(sd, x, weights.values, state, conv, length);
            // Match the importer's constant ownership, without rebuilding or copying weights.
            List<SDVariable> constants = new ArrayList<>();
            for (SDVariable variable : sd.variables()) {
                if (variable.getVariableType() == VariableType.VARIABLE) constants.add(variable);
            }
            sd.convertToConstants(constants);
            Map<String, String> stages = new LinkedHashMap<>();
            stages.put("input", x.name());
            String[] names = {"block_rms", "gdn_qkv_0", "gdn_conv_0", "gdn_q_0", "gdn_k_0", "gdn_v_0",
                    "gdn_q_grouped_0", "gdn_k_grouped_0", "gdn_q_normsq_0", "gdn_k_normsq_0",
                    "gdn_q_norm_cast_0", "gdn_k_norm_cast_0", "gdn_q_l2norm_0", "gdn_k_l2norm_0",
                    "gdn_beta_proj_0_accum", "gdn_beta_proj_0", "gdn_beta_0",
                    "gdn_alpha_proj_0_accum", "gdn_alpha_proj_0", "gdn_a_plus_bias_0", "gdn_softplus_0"};
            for (String name : names) {
                assertNotNull(sd.getVariable(name), "Shared builder stage " + name);
                stages.put(name, name);
            }
            // Discover the actual op arguments, including unnamed/renamed casts. Do not
            // reconstruct Q/K/V/beta/g or guess names at the recurrence boundary.
            GatedDeltaRule recurrence = null;
            for (SameDiffOp op : sd.getOps().values()) {
                if (op.getOp() instanceof GatedDeltaRule) {
                    assertNull(recurrence, "Expected exactly one recurrent block");
                    recurrence = (GatedDeltaRule) op.getOp();
                }
            }
            assertNotNull(recurrence);
            String[] roles = {"recurrence/Q", "recurrence/K", "recurrence/V", "recurrence/beta", "recurrence/g"};
            assertEquals(7, recurrence.args().length);
            for (int i = 0; i < roles.length; i++) stages.put(roles[i], recurrence.arg(i).name());
            String[] downstream = {"gdn_out_0", "model.layers.0.gdn.ssm_norm_0",
                    "gdn_gate_proj_0", "gdn_z_reshaped_0", "gdn_gate_input_0", "gdn_gate_act_0",
                    "gdn_gated_0", "gdn_output_cast_0", "gdn_flat_0"};
            for (String name : downstream) {
                assertNotNull(sd.getVariable(name), "Shared builder stage " + name);
                stages.put(name, name);
            }
            stages.put("attention", attention.name());
            stages.put("state", "gdn_state_out_0");
            stages.put("convState", "conv_state_out_0");
            if (optimized) {
                // Preserve only the block's externally observable outputs, exactly as
                // production optimization does; do not pin diagnostic intermediates.
                stages.clear();
                stages.put("attention", attention.name());
                stages.put("state", "gdn_state_out_0");
                stages.put("convState", "conv_state_out_0");
            }
            String[] outputs = stages.values().stream().distinct().toArray(String[]::new);
            sd.setOutputs(outputs);
            SameDiff execution = optimized ? GraphOptimizer.optimize(sd, Arrays.asList(outputs)) : sd;
            try {
            assertTrue(execution.isDspAutoCompileEnabled(), "DSP must remain enabled");
            log.info("BLOCK_PARITY shared=buildRMSNorm+buildGDNAttention source={} DSP={} native={} mode={} stages={}",
                    REVISION, sd.isDspAutoCompileEnabled(), sd.isDspNativeAutoCompileEnabled(),
                    sd.getGraphExecutionMode(), stages);
            Snapshot initialSnapshot = snapshot(initial), convSnapshot = snapshot(convInitial);
            Map<String, Snapshot> first = null;
            for (int repeat = 0; repeat < 3; repeat++) {
                Map<String, Snapshot> window = blockOutput(execution, rows, initialSnapshot, convSnapshot, stages, outputs);
                Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
                for (String stage : stages.keySet()) scalar.put(stage, new ArrayList<>());
                Snapshot nextState = initialSnapshot, nextConv = convSnapshot;
                for (int t = 0; t < IDS.length; t++) {
                    try (INDArray row = rowCopy(rows, t)) {
                        Map<String, Snapshot> step = blockOutput(execution, row, nextState, nextConv, stages, outputs);
                        for (String stage : stages.keySet()) scalar.get(stage).add(step.get(stage));
                        nextState = step.get("state");
                        nextConv = step.get("convState");
                    }
                }
                for (String stage : stages.keySet()) {
                    boolean finalState = stage.equals("state") || stage.equals("convState");
                    Snapshot chained = finalState ? scalar.get(stage).get(IDS.length - 1) : join(scalar.get(stage));
                    boolean accumulator = stage.equals("gdn_alpha_proj_0_accum") || stage.equals("gdn_beta_proj_0_accum");
                    String context = "block/repeat=" + repeat + "/" + stage;
                    compare(context, window.get(stage), chained, failures, !accumulator);
                    if (accumulator) {
                        Snapshot weight = snapshot(weights.values.get(stage.contains("alpha")
                                ? "blk.0.ssm_alpha.weight" : "blk.0.ssm_beta.weight"));
                        validateAccumulation(context + "/W5", window.get(stage), window.get("block_rms"), weight);
                        validateAccumulation(context + "/W1", chained, join(scalar.get("block_rms")), weight);
                    }
                    if (first != null) compare(context + "/repeat", first.get(stage), window.get(stage), failures, !accumulator);
                }
                // Fix all five recurrence inputs to this W5 observation: distinguish
                // upstream drift from recurrence determinism without changing the graph.
                if (!optimized) directRecurrence(window, initialSnapshot, failures);
                if (first == null) first = window;
            }
            } finally {
                if (execution != sd) execution.close();
            }
        }
        assertTrue(failures.isEmpty(), () -> "First observable drift in stage order (optimized=" + optimized + "): " + failures.get(0)
                + "\nAll observed mismatches:\n" + String.join("\n", failures));
        log.info("BLOCK_PARITY shared first attention stored stages exact; dense FLOAT accumulators bounded. "
                + "Observability changes fusion; no full-model or token-117 causality claim.");
    }

    @Test
    void diagnosticConnectedRealPrefixWindowVersusScalar() throws Exception {
        connectedParity(false);
    }

    @Test
    void optimizedConnectedRealPrefixWindowVersusScalar() throws Exception {
        connectedParity(true);
    }

    /**
     * proc-058 maps GDN ext 2196/2198 to past_gdn_state.0/.1 and conv ext
     * 2257/2255 to past_conv_state.0/.1 (EXT_INPUT_DISCOVER + PRE_EXEC_STATE_FP).
     * Pair indices are lexicographic within kind, NOT arbitrary numeric layer indices.
     * Build the actual connected prefix, preserving both residuals, both norms and MLP.
     * This is a bounded graph observation, not a replay-topology or token-169 proof.
     */
    @Test
    void diagnosticConnectedRealRejectedPrefix() throws Exception {
        connectedParity(false, true);
    }

    @Test
    void optimizedConnectedRealRejectedPrefix() throws Exception {
        connectedParity(true, true);
    }

    private void connectedParity(boolean optimized) throws Exception {
        connectedParity(optimized, false);
    }

    private void connectedParity(boolean optimized, boolean rejectedPrefix) throws Exception {
        File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        ModelOptQwenConfig config = ModelOptQwenConfig.read(cached(cache, "config.json"),
                cached(cache, "hf_quant_config.json"), cached(cache, "generation_config.json"));
        ArchitectureConfig architecture = config.getArchitecture();
        assertEquals(List.of("linear_attention", "linear_attention"), architecture.getLayerTypes().subList(0, 2));
        assertEquals(5120, architecture.getHiddenSize());
        int[] promptIds = connectedPromptIds(cache);
        // GenerationPipeline.prefillWarmupAndFreeze right-pads token IDs with zero,
        // irrespective of tokenizer pad metadata. proc-058 uses width 128, actual 36.
        int[] paddedPromptIds = Arrays.copyOf(promptIds, 128);
        int[] windowIds = rejectedPrefix ? new int[]{579, 264, 7047, 92961, 188489} : IDS;
        JsonObject index = json(cached(cache, "model.safetensors.index.json")).getAsJsonObject("weight_map");
        List<String> failures = new ArrayList<>();
        try (BlockWeights first = new BlockWeights(cache, index, config, 0, true);
             BlockWeights second = new BlockWeights(cache, index, config, 1, true);
             INDArray prompt = embeddingRows(cache, index, architecture.getHiddenSize(), paddedPromptIds);
             INDArray generated = embeddingRows(cache, index, architecture.getHiddenSize(), new int[]{8160});
             INDArray windowRows = embeddingRows(cache, index, architecture.getHiddenSize(), windowIds);
             INDArray zeroState = Nd4j.zeros(DataType.FLOAT, 1, config.getValueHeads(), config.getKeyHeadDim(), config.getValueHeadDim());
             INDArray zeroConv = Nd4j.zeros(DataType.BFLOAT16, 1, config.getConvChannels(), config.getConvKernel() - 1);
             SameDiff source = SameDiff.create()) {
            long payload = first.bytes + second.bytes;
            assertTrue(payload < 1024L * 1024 * 1024, "Connected weights must remain below 1 GiB");
            Map<String, INDArray> weights = new LinkedHashMap<>(first.values);
            weights.putAll(second.values);
            SDVariable hidden = source.placeHolder("embedded", DataType.BFLOAT16, 1, -1, architecture.getHiddenSize());
            SDVariable length = source.placeHolder("actual_sequence_length", DataType.INT64);
            ConnectedBlocks builder = new ConnectedBlocks(config);
            Map<String, String> stages = new LinkedHashMap<>();
            // The rejected-prefix diagnostic requests live produced stages, not an
            // external placeholder output (which cannot be exposed by the DSP compiler).
            if (!optimized && !rejectedPrefix) stages.put(hidden.name(), hidden.name());
            for (int layer = 0; layer < 2; layer++) {
                SDVariable state = source.placeHolder("past_gdn_state." + layer, DataType.FLOAT,
                        1, config.getValueHeads(), config.getKeyHeadDim(), config.getValueHeadDim());
                SDVariable conv = source.placeHolder("past_conv_state." + layer, DataType.BFLOAT16,
                        1, config.getConvChannels(), config.getConvKernel() - 1);
                hidden = builder.block(source, hidden, layer, weights, state, conv, length);
                if (!optimized) {
                    String[] ordered = {"model.layers." + layer + ".input_layernorm", "gdn_qkv_" + layer,
                            "gdn_conv_" + layer, "gdn_q_l2norm_" + layer, "gdn_k_l2norm_" + layer,
                            "gdn_beta_proj_" + layer, "gdn_beta_" + layer, "gdn_alpha_proj_" + layer,
                            "gdn_softplus_" + layer, "gdn_out_" + layer,
                            "model.layers." + layer + ".gdn.ssm_norm_" + layer,
                            "gdn_gate_proj_" + layer, "gdn_gate_act_" + layer, "gdn_gated_" + layer,
                            "gdn_output_cast_" + layer, "gdn_proj_" + layer, "post_attn_" + layer,
                            "model.layers." + layer + ".post_attention_layernorm",
                            "gate_" + layer, "up_" + layer,
                            "swiglu_" + layer, "down_" + layer, "layer_out_" + layer};
                    for (String name : ordered) {
                        assertNotNull(source.getVariable(name), "Production connected stage " + name);
                        stages.put(name, name);
                    }
                }
                stages.put("gdn_state_out_" + layer, "gdn_state_out_" + layer);
                stages.put("conv_state_out_" + layer, "conv_state_out_" + layer);
            }
            List<SDVariable> constants = new ArrayList<>();
            for (SDVariable variable : source.variables()) {
                if (variable.getVariableType() == VariableType.VARIABLE) constants.add(variable);
            }
            source.convertToConstants(constants);
            List<String> observed = new ArrayList<>(stages.values());
            String[] alphaOperands = null;
            if (!optimized && !rejectedPrefix) {
                String accum = "gdn_alpha_proj_1_accum";
                SameDiffOp gemm = source.getOps().get(source.getVariables().get(accum).getOutputOfOp());
                assertEquals("matmul", gemm.getOp().opName());
                alphaOperands = gemm.getInputsToOp().toArray(new String[0]);
                assertEquals(2, alphaOperands.length);
                observed.add(accum);
                Collections.addAll(observed, alphaOperands);
                log.info("CONNECTED_NUMERIC observing actual FLOAT GEMM operands={} and {}; "
                        + "extra outputs alter fusion boundaries, NOT full-model topology", Arrays.toString(alphaOperands), accum);
            }
            String[] outputs = observed.stream().distinct().toArray(String[]::new);
            source.setOutputs(outputs);
            for (String output : outputs) {
                String opName = source.getVariables().get(output).getOutputOfOp();
                SameDiffOp op = opName == null ? null : source.getOps().get(opName);
                log.info("CONNECTED_SOURCE output={} dtype={} producer={} op={} inputs={}", output,
                        source.getVariable(output).dataType(), opName, op == null ? "placeholder" : op.getOp().opName(),
                        op == null ? Collections.emptyList() : op.getInputsToOp());
            }
            SameDiff execution = optimized ? GraphOptimizer.optimize(source, Arrays.asList(outputs)) : source;
            try {
                assertTrue(execution.isDspAutoCompileEnabled(), "DSP lifecycle must remain enabled");
                log.info("CONNECTED_PARITY optimized={} layers=[0,1] payloadBytes={} promptIds={} seedToken=8160 "
                                + "seedPosition=37 paddedPrefillWidth=128 actualPrefillLength=36 windowIds={} "
                                + "source=buildTransformerBlock DSP={} native={} mode={}",
                        optimized, payload, Arrays.toString(promptIds), Arrays.toString(windowIds),
                        execution.isDspAutoCompileEnabled(), execution.isDspNativeAutoCompileEnabled(), execution.getGraphExecutionMode());
                Map<String, Snapshot> zeros = new LinkedHashMap<>();
                for (int layer = 0; layer < 2; layer++) {
                    zeros.put("gdn_state_out_" + layer, snapshot(zeroState));
                    zeros.put("conv_state_out_" + layer, snapshot(zeroConv));
                }
                // Real prompt in one prefill, then the actual first generated token. No
                // zero-state or unrelated embedding substitution at layer 1's boundary.
                Map<String, Snapshot> prefill = connectedOutput(execution, prompt, zeros, outputs, promptIds.length);
                Map<String, Snapshot> seed = connectedOutput(execution, generated, prefill, outputs);
                if (rejectedPrefix) {
                    try (INDArray alternative = embeddingRows(cache, index, architecture.getHiddenSize(), IDS)) {
                        rejectedPrefixParity(execution, windowRows, alternative, seed, outputs, optimized, failures);
                    }
                } else {
                Map<String, Snapshot> firstWindow = null;
                for (int repeat = 0; repeat < 3; repeat++) {
                    Map<String, Snapshot> window = connectedOutput(execution, windowRows, seed, outputs);
                    Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
                    for (String output : outputs) scalar.put(output, new ArrayList<>());
                    Map<String, Snapshot> next = seed;
                    for (int t = 0; t < IDS.length; t++) {
                        try (INDArray row = rowCopy(windowRows, t)) {
                            next = connectedOutput(execution, row, next, outputs);
                            for (String output : outputs) scalar.get(output).add(next.get(output));
                        }
                    }
                    if (!optimized) {
                        connectedAlphaProbe("connected/" + repeat, window, scalar, alphaOperands, failures);
                        if (repeat == 0) capturedAlphaDspProbe(window, scalar, alphaOperands, failures);
                    }
                    for (String output : stages.values()) {
                        boolean state = output.startsWith("gdn_state_out_") || output.startsWith("conv_state_out_");
                        Snapshot expected = state ? next.get(output) : join(scalar.get(output));
                        compare("connected/optimized=" + optimized + "/repeat=" + repeat + "/" + output,
                                window.get(output), expected, failures);
                        if (firstWindow != null) compare("connected/repeat-stability/" + output,
                                firstWindow.get(output), window.get(output), failures);
                    }
                    if (firstWindow == null) firstWindow = window;
                }
                }
                log.info("CONNECTED_PARITY result optimized={} firstObservedDrift={}", optimized,
                        failures.isEmpty() ? "NONE in requested stored stages" : failures.get(0));
            } finally {
                if (execution != source) execution.close();
            }
        }
        assertTrue(failures.isEmpty(), () -> "Connected real-prefix drift; optimized=" + optimized + "\n"
                + String.join("\n", failures));
    }

    /**
     * proc-069 step 0: base 37, four drafts, two accepted; consume 579/264/7047
     * and commit at 40. No predictor is involved in this bounded two-block probe.
     * Diagnostic outputs alter fusion; optimized outputs remain state-only.
     */
    private static void rejectedPrefixParity(SameDiff sd, INDArray rows, INDArray alternative,
                                             Map<String, Snapshot> seed, String[] outputs,
                                             boolean optimized, List<String> failures) {
        final int live = 3;
        assertEquals(5, rows.size(1));
        compare("rejected-prefix/common-live-input", livePrefix(snapshot(rows), live),
                livePrefix(snapshot(alternative), live), failures);
        Map<String, Snapshot> first = null;
        for (int repeat = 0; repeat < 3; repeat++) {
            Map<String, Snapshot> rerun = connectedOutput(sd, rows, seed, outputs, live, true);
            var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor, "Rejected-prefix graph must actually compile with DSP enabled");
            assertNotNull(executor.getNativePlanHandle(), "Rejected-prefix native DSP plan required");
            Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
            for (String output : outputs) scalar.put(output, new ArrayList<>());
            Map<String, Snapshot> next = seed;
            for (int t = 0; t < live; t++) {
                try (INDArray row = rowCopy(rows, t)) {
                    next = connectedOutput(sd, row, next, outputs);
                    for (String output : outputs) scalar.get(output).add(next.get(output));
                }
            }
            // Same accepted tokens, different real embedding suffix. Run full W5
            // first here too; neither rejected verification may contaminate its rerun.
            Map<String, Snapshot> other = connectedOutput(sd, alternative, seed, outputs, live, true);
            for (String output : outputs) {
                boolean state = output.startsWith("gdn_state_out_") || output.startsWith("conv_state_out_");
                Snapshot actual = state ? rerun.get(output) : livePrefix(rerun.get(output), live);
                Snapshot expected = state ? next.get(output) : join(scalar.get(output));
                String label = "rejected-prefix/optimized=" + optimized + "/repeat=" + repeat + "/" + output;
                compare(label + "/W5actual3-vs-3xW1", actual, expected, failures);
                compare(label + "/suffix-invariance", actual,
                        state ? other.get(output) : livePrefix(other.get(output), live), failures);
                if (first != null) compare(label + "/repeat-stability", actual,
                        state ? first.get(output) : livePrefix(first.get(output), live), failures);
            }
            if (first == null) first = rerun;
        }
        log.info("REJECTED_PREFIX result optimized={} base=37 commit=40 liveIds=[579,264,7047] "
                        + "rejectedSuffix=[92961,188489] alternateSuffix=[1817,25] firstObservedDrift={} "
                        + "scope=connected-layers-0-1-not-full-model-or-predictor",
                optimized, failures.isEmpty() ? "NONE" : failures.get(0));
    }

    private static Snapshot livePrefix(Snapshot value, int live) {
        assertEquals(1, value.shape[0]);
        assertEquals(5, value.shape[1], "Only window outputs may be sliced; states are compared in full");
        long[] shape = value.shape.clone();
        shape[1] = live;
        return new Snapshot(value.dtype, shape, Arrays.copyOf(value.values, value.values.length / 5 * live));
    }

    /**
     * Separate, bounded arithmetic-only DSP observation of the captured connected operands.
     * The fully observed connected graph requests external input 'embedded', which the
     * compiler cannot expose as an output. Do not remove that observation to disguise the
     * limitation. This small graph proves only GEMM/cast behavior, not connected fusion.
     */
    private static void capturedAlphaDspProbe(Map<String, Snapshot> window,
                                              Map<String, List<Snapshot>> scalar, String[] operands,
                                              List<String> failures) {
        String accum = "gdn_alpha_proj_1_accum", stored = "gdn_alpha_proj_1";
        String[] outputs = {accum, stored};
        try (SameDiff probe = SameDiff.create()) {
            SDVariable a = probe.placeHolder(operands[0], DataType.FLOAT, 1, -1, 5120);
            SDVariable b = probe.placeHolder(operands[1], DataType.FLOAT, 5120, 48);
            new Mmul(probe, a, b, MMulTranspose.allFalse(), Mmul.Arithmetic.SERIAL_FMA)
                    .outputVariable().rename(accum).castTo(DataType.BFLOAT16).rename(stored);
            probe.setOutputs(outputs);
            assertTrue(probe.isDspAutoCompileEnabled());
            log.info("CONNECTED_NUMERIC captured-only DSP probe: same FLOAT operands and production SERIAL_FMA mmul/cast; "
                    + "no connected fusion claim; connected observations remain unchanged");
            for (int repeat = 0; repeat < 3; repeat++) {
                Map<String, Snapshot> w = new LinkedHashMap<>(window);
                w.putAll(capturedAlphaOutput(probe, window.get(operands[0]), window.get(operands[1]), operands, outputs));
                Map<String, List<Snapshot>> s = new LinkedHashMap<>(scalar);
                s.put(accum, new ArrayList<>());
                s.put(stored, new ArrayList<>());
                for (int row = 0; row < IDS.length; row++) {
                    Map<String, Snapshot> out = capturedAlphaOutput(probe, scalar.get(operands[0]).get(row),
                            scalar.get(operands[1]).get(row), operands, outputs);
                    for (String output : outputs) s.get(output).add(out.get(output));
                }
                connectedAlphaProbe("captured-DSP/" + repeat, w, s, operands, failures);
                compare("captured-DSP/" + repeat + "/stored-W5-vs-W1", w.get(stored), join(s.get(stored)), failures);
                compare("captured-DSP/" + repeat + "/stored-vs-connected-W5", window.get(stored), w.get(stored), failures);
            }
        }
    }

    private static Map<String, Snapshot> capturedAlphaOutput(SameDiff probe, Snapshot a, Snapshot b,
                                                              String[] operands, String[] outputs) {
        try (INDArray x = restore(a); INDArray weight = restore(b)) {
            logConnectedLifecycle(probe, "captured-only/before", a.shape[1]);
            Map<String, INDArray> out = probe.output(Map.of(operands[0], x, operands[1], weight), outputs);
            logConnectedLifecycle(probe, "captured-only/after", a.shape[1]);
            var executor = probe.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor, "Captured arithmetic must actually compile; no inferred DSP coverage");
            assertNotNull(executor.getNativePlanHandle(), "Captured arithmetic requires a native DSP plan");
            Map<String, Snapshot> result = new LinkedHashMap<>();
            for (String output : outputs) result.put(output, snapshot(out.get(output)));
            return result;
        }
    }

    private static void logConnectedLifecycle(SameDiff sd, String phase, long width) {
        var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        log.info("CONNECTED_NUMERIC lifecycle phase={} width={} plan={} nativeHandle={} frozen={} mode={}",
                phase, width, executor == null || executor.getCurrentPlan() == null ? null
                        : Integer.toHexString(System.identityHashCode(executor.getCurrentPlan())),
                executor == null ? null : executor.getNativePlanHandle(), sd.isDspShapesFrozen(), sd.getGraphExecutionMode());
    }

    /** Host DOUBLE oracle only: never feeds a value back into the connected graph. */
    private static void connectedAlphaProbe(String repeat, Map<String, Snapshot> window,
                                            Map<String, List<Snapshot>> scalar, String[] operands,
                                            List<String> failures) {
        Snapshot a = window.get(operands[0]), b = window.get(operands[1]);
        Snapshot result = window.get("gdn_alpha_proj_1_accum");
        Snapshot stored = window.get("gdn_alpha_proj_1");
        assertEquals(DataType.FLOAT, a.dtype);
        assertEquals(DataType.FLOAT, b.dtype);
        assertEquals(DataType.FLOAT, result.dtype);
        assertArrayEquals(new long[]{5120, 48}, b.shape);
        int k = 5120, n = 48;
        assertEquals(IDS.length * k, a.values.length);
        int reported = 0;
        for (int row = 0; row < IDS.length; row++) {
            Snapshot sa = scalar.get(operands[0]).get(row), sb = scalar.get(operands[1]).get(row);
            Snapshot sr = scalar.get("gdn_alpha_proj_1_accum").get(row);
            Snapshot ss = scalar.get("gdn_alpha_proj_1").get(row);
            assertEquals(k, sa.values.length);
            assertArrayEquals(b.shape, sb.shape);
            int inputBits = 0, weightBits = 0, nonBf16 = 0;
            for (int i = 0; i < k; i++) {
                if (Float.floatToRawIntBits(a.values[row * k + i]) != Float.floatToRawIntBits(sa.values[i])) inputBits++;
                if ((Float.floatToRawIntBits(a.values[row * k + i]) & 0xffff) != 0
                        || (Float.floatToRawIntBits(sa.values[i]) & 0xffff) != 0) nonBf16++;
            }
            for (int i = 0; i < b.values.length; i++) {
                if (Float.floatToRawIntBits(b.values[i]) != Float.floatToRawIntBits(sb.values[i])) weightBits++;
                if ((Float.floatToRawIntBits(b.values[i]) & 0xffff) != 0
                        || (Float.floatToRawIntBits(sb.values[i]) & 0xffff) != 0) nonBf16++;
            }
            log.info("CONNECTED_NUMERIC repeat={} token={} A_bitDifferences={} B_bitDifferences={} nonBF16Operands={} "
                    + "(BF16-exact operands are also TF32-exact; these values alone cannot identify a TF32 kernel)",
                    repeat, IDS[row], inputBits, weightBits, nonBf16);
            for (int head = 0; head < n; head++) {
                double dot = 0, scalarDot = 0, abs = 0, scalarAbs = 0;
                float sequential = 0, reverse = 0;
                for (int i = 0; i < k; i++) {
                    double p = (double) a.values[row * k + i] * b.values[i * n + head];
                    double sp = (double) sa.values[i] * sb.values[i * n + head];
                    dot += p;
                    scalarDot += sp;
                    abs += Math.abs(p);
                    scalarAbs += Math.abs(sp);
                    sequential += (float) p;
                    reverse += a.values[row * k + k - 1 - i] * b.values[(k - 1 - i) * n + head];
                }
                float w = result.values[row * n + head], s = sr.values[head];
                float ws = stored.values[row * n + head], sc = ss.values[head];
                double u = Math.scalb(1.0, -24), ud = Math.scalb(1.0, -53);
                double bound = k * u / (1 - k * u) * abs + k * (double) Float.MIN_VALUE;
                double scalarBound = k * u / (1 - k * u) * scalarAbs + k * (double) Float.MIN_VALUE;
                double oracleBound = k * ud / (1 - k * ud) * abs;
                if (!Double.isFinite(dot) || !Double.isFinite(scalarDot) || !Float.isFinite(w) || !Float.isFinite(s)
                        || Math.abs(w - dot) > bound || Math.abs(s - scalarDot) > scalarBound) {
                    failures.add("connected alpha FLOAT oracle bound repeat=" + repeat + " row=" + row + " head=" + head);
                }
                // Bracket the DOUBLE value directly; do not round DOUBLE->FLOAT->BF16.
                int truncated = Float.floatToRawIntBits((float) dot) & 0xffff0000;
                double base = Float.intBitsToFloat(truncated);
                double neighbor = Float.intBitsToFloat(truncated + 0x10000);
                double lower = Math.min(base, neighbor), upper = Math.max(base, neighbor);
                // FLOAT conversion can cross an exact BF16 endpoint; correct the bracket.
                if (dot < lower) {
                    upper = lower;
                    int bits = Float.floatToRawIntBits((float) lower);
                    lower = Float.intBitsToFloat(bits == 0 ? 0x80010000
                            : bits + (lower > 0 ? -0x10000 : 0x10000));
                } else if (dot > upper) {
                    lower = upper;
                    int bits = Float.floatToRawIntBits((float) upper);
                    upper = Float.intBitsToFloat(bits + (upper >= 0 ? 0x10000 : -0x10000));
                }
                double midpoint = (lower + upper) * 0.5;
                double rounded = dot < midpoint ? lower : dot > midpoint ? upper
                        : ((Float.floatToRawIntBits((float) lower) >>> 16) & 1) == 0 ? lower : upper;
                if (head == 18 || (ws != sc && reported++ < 8)) {
                    log.info("CONNECTED_NUMERIC repeat={} token={} head={} W5raw={} W1raw={} W5hex={} W1hex={} "
                                    + "doubleDot={} scalarDoubleDot={} sumAbs={} doubleErrorBound={} "
                                    + "W5error={} W1error={} fp32Bound={} scalarFp32Bound={} "
                                    + "BF16lower={} upper={} midpoint={} oracleMinusMid={} W5minusMid={} W1minusMid={} "
                                    + "oracleBF16={} W5stored={} W1stored={} sequentialFloat={} reverseFloat={}",
                            repeat, IDS[row], head, w, s, Float.toHexString(w), Float.toHexString(s),
                            dot, scalarDot, abs, oracleBound, w - dot, s - scalarDot, bound, scalarBound,
                            lower, upper, midpoint, dot - midpoint, w - midpoint, s - midpoint,
                            rounded, ws, sc, sequential, reverse);
                }
            }
        }
    }

    private static int[] connectedPromptIds(File cache) throws IOException {
        String tokenizerJson = Files.readString(cached(cache, "tokenizer.json").toPath(), StandardCharsets.UTF_8);
        JsonObject tokenizerConfig = json(cached(cache, "tokenizer_config.json"));
        String template = Files.readString(cached(cache, "chat_template.jinja").toPath(), StandardCharsets.UTF_8);
        assertFalse(template.isBlank());
        if (tokenizerConfig.has("chat_template") && !tokenizerConfig.get("chat_template").isJsonNull()) {
            assertEquals(template.trim(), tokenizerConfig.get("chat_template").getAsString().trim());
        }
        tokenizerConfig.addProperty("chat_template", template);
        try (HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.fromJson(tokenizerJson, new Gson().toJson(tokenizerConfig))) {
            assertEquals(248046, tokenizer.getEosTokenId());
            int[] ids = tokenizer.encodePrompt("What is 2 + 2? Reply briefly with only a JSON object whose key is answer and value is the integer result.").getIds();
            assertEquals(36, ids.length, "proc-058 actual prefill length; tokenizer/template must match");
            return ids;
        }
    }

    private static final class ConnectedBlocks extends LLaMAArchitecture {
        private final ModelOptQwenConfig config;
        ConnectedBlocks(ModelOptQwenConfig config) { this.config = config; }
        @Override protected int getGdnKeyHeads(int valueHeads) {
            assertEquals(config.getValueHeads(), valueHeads);
            return config.getKeyHeads();
        }
        @Override protected boolean multiplyNormInFloat() { return true; }
        SDVariable block(SameDiff sd, SDVariable input, int layer, Map<String, INDArray> weights,
                         SDVariable state, SDVariable conv, SDVariable length) {
            return buildTransformerBlock(sd, input, layer, config.getArchitecture(), weights, DataType.BFLOAT16,
                    null, null, length, null, null, null, state, conv);
        }
    }

    private static Map<String, Snapshot> connectedOutput(SameDiff sd, INDArray input,
                                                        Map<String, Snapshot> state, String[] outputs) {
        return connectedOutput(sd, input, state, outputs, Math.toIntExact(input.size(1)));
    }

    private static Map<String, Snapshot> connectedOutput(SameDiff sd, INDArray input,
                                                        Map<String, Snapshot> state, String[] outputs, int actualLength) {
        return connectedOutput(sd, input, state, outputs, actualLength, false);
    }

    private static Map<String, Snapshot> connectedOutput(SameDiff sd, INDArray input,
                                                        Map<String, Snapshot> state, String[] outputs,
                                                        int actualLength, boolean rejectedFirst) {
        assertTrue(actualLength > 0 && actualLength <= input.size(1));
        List<INDArray> owned = new ArrayList<>();
        try {
            Map<String, INDArray> feeds = new LinkedHashMap<>();
            INDArray x = input.dup('c');
            owned.add(x);
            feeds.put("embedded", x);
            INDArray length = Nd4j.scalar(DataType.INT64, actualLength);
            owned.add(length);
            feeds.put("actual_sequence_length", length);
            for (int layer = 0; layer < 2; layer++) {
                for (String kind : new String[]{"gdn", "conv"}) {
                    Snapshot original = state.get(kind + "_state_out_" + layer);
                    INDArray copy = restore(original);
                    owned.add(copy);
                    // Full-array bit check, not four sampled words. Every invocation owns
                    // new state storage, so neither W5 nor scalar can mutate the seed.
                    List<String> errors = new ArrayList<>();
                    compare("connected/independent-input/" + kind + "/" + layer, original, snapshot(copy), errors);
                    assertTrue(errors.isEmpty(), () -> String.join("\n", errors));
                    feeds.put("past_" + kind + "_state." + layer, copy);
                }
            }
            if (rejectedFirst) {
                // autoregressive_decode.cu SPEC_STATE_RERUN: no state commit or
                // restoration between full verification and the accepted-prefix pass.
                // Reuse the SAME buffers and graph; change only actual length.
                length.assign(input.size(1));
                sd.output(feeds, outputs);
                List<String> unchanged = new ArrayList<>();
                for (int layer = 0; layer < 2; layer++) {
                    for (String kind : new String[]{"gdn", "conv"}) {
                        compare("rejected-prefix/pre-rerun-uncommitted/" + kind + "/" + layer,
                                state.get(kind + "_state_out_" + layer),
                                snapshot(feeds.get("past_" + kind + "_state." + layer)), unchanged);
                    }
                }
                compare("rejected-prefix/pre-rerun-input", snapshot(input), snapshot(x), unchanged);
                assertTrue(unchanged.isEmpty(), () -> String.join("\n", unchanged));
                length.assign(actualLength);
                log.info("REJECTED_PREFIX rerun width={} actual={} sameFeeds=true stateRestored=false",
                        input.size(1), actualLength);
            }
            logConnectedLifecycle(sd, "before", input.size(1));
            Map<String, INDArray> result = sd.output(feeds, outputs);
            logConnectedLifecycle(sd, "after", input.size(1));
            Map<String, Snapshot> snapshots = new LinkedHashMap<>();
            for (String output : outputs) {
                INDArray array = result.get(output);
                assertNotNull(array, output);
                assertEquals(sd.getVariable(output).dataType(), array.dataType(), output);
                if (output.startsWith("gdn_alpha_proj_1") && array.dataType() == DataType.FLOAT) {
                    log.info("CONNECTED_NUMERIC layout name={} shape={} stride={} order={}", output,
                            Arrays.toString(array.shape()), Arrays.toString(array.stride()), array.ordering());
                }
                snapshots.put(output, snapshot(array));
            }
            return snapshots;
        } finally {
            for (INDArray array : owned) array.close();
        }
    }

    @Test
    void firstActualFullAttentionWindowVersusScalar() throws Exception {
        fullAttentionParity(false);
    }

    @Test
    void optimizedFirstActualFullAttentionWindowVersusScalar() throws Exception {
        fullAttentionParity(true);
    }

    /**
     * Actual layer-3 weights and unmodified shared attention math. Embedding rows are
     * controlled local block inputs, NOT the hidden states produced by layers 0..2.
     * Seed KV is produced by this same block at positions 0..4; compare positions 5..9.
     * This proves/localizes block parity only, not the later full-model token drift.
     */
    private void fullAttentionParity(boolean optimized) throws Exception {
        File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        ModelOptQwenConfig config = ModelOptQwenConfig.read(cached(cache, "config.json"),
                cached(cache, "hf_quant_config.json"), cached(cache, "generation_config.json"));
        ArchitectureConfig c = config.getArchitecture();
        int layer = c.getLayerTypes().indexOf("full_attention");
        assertEquals(3, layer);
        assertEquals(24, c.getNumAttentionHeads());
        assertEquals(4, c.getNumKVHeads());
        assertEquals(256, c.getHeadDimension());
        assertEquals(5120, c.getHiddenSize());
        JsonObject index = json(cached(cache, "model.safetensors.index.json")).getAsJsonObject("weight_map");
        List<String> failures = new ArrayList<>();
        try (BlockWeights weights = new BlockWeights(cache, index, config, layer);
             INDArray rows = embeddingRows(cache, index, c.getHiddenSize());
             INDArray zeros = Nd4j.zeros(DataType.BFLOAT16, 1, 640, 4, 256);
             SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", DataType.BFLOAT16, 1, -1, 5120);
            SDVariable key = sd.placeHolder("keys", DataType.BFLOAT16, 1, 640, 4, 256);
            SDVariable value = sd.placeHolder("values", DataType.BFLOAT16, 1, 640, 4, 256);
            SDVariable position = sd.placeHolder("position_offset", DataType.INT64);
            SDVariable cachePosition = sd.placeHolder("cache_position", DataType.INT64);
            SDVariable bias = sd.placeHolder("_causal_mask", DataType.FLOAT, 1, 1, -1, 640);
            SDVariable attention = new FirstFullAttentionBlock().build(sd, x, layer, c,
                    weights.values, position, cachePosition, bias, key, value);
            List<SDVariable> constants = new ArrayList<>();
            for (SDVariable variable : sd.variables()) {
                if (variable.getVariableType() == VariableType.VARIABLE) constants.add(variable);
            }
            sd.convertToConstants(constants);
            List<String> stages = new ArrayList<>();
            if (!optimized) {
                Collections.addAll(stages, "full_block_rms", "q_full_3", "k_3", "v_3", "q_3",
                        "attn_gate_3", "k_heads_3", "v_heads_3", "model.layers.3.self_attn.q_norm_3",
                        "model.layers.3.self_attn.k_norm_3", "q_rope_3", "k_rope_3", "attn_out_3",
                        "attn_flat_3", "gate_sigmoid_3", "gated_attn_3");
            }
            stages.add(attention.name());
            String[] outputs = stages.toArray(new String[0]);
            for (String stage : stages) assertNotNull(sd.getVariable(stage), stage);
            sd.setOutputs(outputs);
            SameDiff execution = optimized ? GraphOptimizer.optimize(sd, stages) : sd;
            try {
                assertTrue(execution.isDspAutoCompileEnabled(), "DSP remains enabled with its normal lifecycle");
                log.info("FULL_ATTENTION_PARITY source={} layer={} optimized={} outputs={} KV=[1,640,4,256] Q=24x256",
                        REVISION, layer, optimized, stages);
                Snapshot empty = snapshot(zeros);
                int history = Integer.getInteger("qwen.nvfp4.attentionHistory", IDS.length);
                assertTrue(history >= IDS.length && history <= 640 - IDS.length
                        && history % IDS.length == 0, "History must fit KV capacity in complete input windows");
                Snapshot initialK = empty, initialV = empty;
                for (int start = 0; start < history; start += IDS.length) {
                    Map<String, Snapshot> seed = fullAttentionOutput(execution, rows, initialK, initialV,
                            start, outputs, failures);
                    initialK = seed.get("keys");
                    initialV = seed.get("values");
                }
                log.info("FULL_ATTENTION_PARITY history={} optimized={}", history, optimized);
                Map<String, Snapshot> first = null;
                for (int repeat = 0; repeat < 3; repeat++) {
                    Map<String, Snapshot> window = fullAttentionOutput(execution, rows, initialK, initialV,
                            history, outputs, failures);
                    Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
                    for (String stage : stages) scalar.put(stage, new ArrayList<>());
                    Snapshot nextK = initialK, nextV = initialV;
                    for (int t = 0; t < IDS.length; t++) {
                        try (INDArray row = rowCopy(rows, t)) {
                            Map<String, Snapshot> step = fullAttentionOutput(execution, row, nextK, nextV,
                                    history + t, outputs, failures);
                            for (String stage : stages) scalar.get(stage).add(step.get(stage));
                            nextK = step.get("keys");
                            nextV = step.get("values");
                        }
                    }
                    String label = "full-attention/optimized=" + optimized + "/repeat=" + repeat + "/";
                    for (String stage : stages) {
                        compare(label + stage, window.get(stage), join(scalar.get(stage)), failures);
                        if (first != null) compare(label + stage + "/repeat", first.get(stage), window.get(stage), failures);
                    }
                    compare(label + "KVcommit/keys", window.get("keys"), nextK, failures);
                    compare(label + "KVcommit/values", window.get("values"), nextV, failures);
                    if (first != null) {
                        compare(label + "keys/repeat", first.get("keys"), window.get("keys"), failures);
                        compare(label + "values/repeat", first.get("values"), window.get("values"), failures);
                    }
                    if (first == null) first = window;
                }
            } finally {
                if (execution != sd) execution.close();
            }
        }
        assertTrue(failures.isEmpty(), () -> "First actual full-attention drift (optimized=" + optimized + "): "
                + failures.get(0) + "\n" + String.join("\n", failures));
        log.info("FULL_ATTENTION_PARITY PASS optimized={} stored outputs and KV bit-exact; local block only; "
                + "no layer0..2 hidden states, full-model generation, token or long-run replay proof", optimized);
    }

    @Test
    void firstLayerPackedMlpWindowVersusScalar() throws Exception {
        mlpParity(false);
    }

    @Test
    void optimizedFirstLayerPackedMlpWindowVersusScalar() throws Exception {
        mlpParity(true);
    }

    /**
     * Real layer-zero NVFP4 MLP, with embedding rows as controlled local inputs.
     * These are NOT post-attention hidden states and cannot establish deep-state or
     * token-169 causality. Diagnostic outputs change fusion boundaries; the second
     * variant preserves only the final output during production graph optimization.
     */
    private void mlpParity(boolean optimized) throws Exception {
        File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        ModelOptQwenConfig config = ModelOptQwenConfig.read(cached(cache, "config.json"),
                cached(cache, "hf_quant_config.json"), cached(cache, "generation_config.json"));
        ArchitectureConfig c = config.getArchitecture();
        assertEquals(5120, c.getHiddenSize());
        assertEquals(17408, c.getIntermediateSize());
        JsonObject index = json(cached(cache, "model.safetensors.index.json")).getAsJsonObject("weight_map");
        List<String> failures = new ArrayList<>();
        try (BlockWeights weights = new BlockWeights(cache, index, c);
             INDArray rows = embeddingRows(cache, index, c.getHiddenSize());
             SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", DataType.BFLOAT16, 1, -1, c.getHiddenSize());
            SDVariable mlp = new FirstMlpBlock().build(sd, x, c, weights.values);
            List<SDVariable> constants = new ArrayList<>();
            for (SDVariable variable : sd.variables()) {
                if (variable.getVariableType() == VariableType.VARIABLE) constants.add(variable);
            }
            sd.convertToConstants(constants);
            List<String> stages = new ArrayList<>();
            if (!optimized) {
                Collections.addAll(stages, "mlp_rms", "gate_0", "up_0");
                // Read the actual shared builder's SiLU operand, not a second activation.
                SameDiffOp multiply = sd.getOps().get(sd.getVariables().get("swiglu_0").getOutputOfOp());
                stages.add(multiply.getInputsToOp().get(0));
                stages.add("swiglu_0");
            }
            stages.add(mlp.name());
            String[] outputs = stages.toArray(new String[0]);
            for (String stage : stages) assertNotNull(sd.getVariable(stage), stage);
            assertEquals(3, sd.getOps().values().stream()
                    .filter(op -> "modelopt_nvfp4_linear".equals(op.getOp().opName())).count(),
                    "All three projections must use the production packed operator");
            sd.setOutputs(outputs);
            SameDiff execution = optimized ? GraphOptimizer.optimize(sd, stages) : sd;
            try {
                assertTrue(execution.isDspAutoCompileEnabled(), "DSP remains enabled with normal lifecycle");
                log.info("MLP_PARITY source={} layer=0 optimized={} outputs={} ids={} controlled=embedding-rows "
                                + "shared=buildRMSNorm+buildSwiGLUFFN packedBytes={} mode={}",
                        REVISION, optimized, stages, Arrays.toString(IDS), weights.bytes, execution.getGraphExecutionMode());
                Snapshot original = snapshot(rows);
                Map<String, Snapshot> first = null;
                for (int repeat = 0; repeat < 3; repeat++) {
                    Map<String, Snapshot> window = mlpOutput(execution, rows, c, outputs, failures);
                    Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
                    for (String stage : stages) scalar.put(stage, new ArrayList<>());
                    List<Snapshot> inputRows = new ArrayList<>();
                    for (int t = 0; t < IDS.length; t++) {
                        try (INDArray row = rowCopy(rows, t)) {
                            inputRows.add(snapshot(row));
                            Map<String, Snapshot> step = mlpOutput(execution, row, c, outputs, failures);
                            for (String stage : stages) scalar.get(stage).add(step.get(stage));
                        }
                    }
                    String label = "mlp/optimized=" + optimized + "/repeat=" + repeat + "/";
                    compare(label + "identical-scalar-inputs", original, join(inputRows), failures);
                    for (String stage : stages) {
                        compare(label + stage + "/W5-vs-5xW1", window.get(stage), join(scalar.get(stage)), failures);
                        if (first != null) compare(label + stage + "/repeat", first.get(stage), window.get(stage), failures);
                    }
                    compare(label + "input-not-mutated", original, snapshot(rows), failures);
                    if (first == null) first = window;
                }
            } finally {
                if (execution != sd) execution.close();
            }
        }
        assertTrue(failures.isEmpty(), () -> "First actual packed MLP drift (optimized=" + optimized + "): "
                + failures.get(0) + "\n" + String.join("\n", failures));
        log.info("MLP_PARITY PASS optimized={} all observed BF16 stores bit-exact. Controlled local inputs only; "
                + "no full-model, token parity, deep-state causality, or long-run replay claim", optimized);
    }

    private static final class FirstMlpBlock extends LLaMAArchitecture {
        @Override protected boolean multiplyNormInFloat() { return true; }
        SDVariable build(SameDiff sd, SDVariable x, ArchitectureConfig config, Map<String, INDArray> weights) {
            SDVariable rms = buildRMSNorm(sd, x, "mlp_rms", "blk.0.post_attention_norm",
                    weights, config, DataType.BFLOAT16);
            return buildSwiGLUFFN(sd, rms, 0, config, weights, DataType.BFLOAT16);
        }
    }

    private static Map<String, Snapshot> mlpOutput(SameDiff sd, INDArray input, ArchitectureConfig config,
                                                  String[] outputs, List<String> failures) {
        Snapshot original = snapshot(input);
        Map<String, INDArray> result = sd.output(Collections.singletonMap("x", input), outputs);
        Map<String, Snapshot> snapshots = new LinkedHashMap<>();
        for (String name : outputs) {
            INDArray array = result.get(name);
            assertNotNull(array, name);
            assertEquals(DataType.BFLOAT16, array.dataType(), name + " stored boundary");
            int width = name.equals("mlp_rms") || name.equals("down_0")
                    ? config.getHiddenSize() : config.getIntermediateSize();
            assertArrayEquals(new long[]{1, input.size(1), width}, array.shape(), name);
            snapshots.put(name, snapshot(array));
        }
        compare("mlp/call-input-not-mutated", original, snapshot(input), failures);
        return snapshots;
    }

    private static final class FirstFullAttentionBlock extends LLaMAArchitecture {
        @Override protected boolean multiplyNormInFloat() { return true; }
        SDVariable build(SameDiff sd, SDVariable x, int layer, ArchitectureConfig config,
                         Map<String, INDArray> weights, SDVariable position, SDVariable cachePosition,
                         SDVariable mask, SDVariable keys, SDVariable values) {
            SDVariable rms = buildRMSNorm(sd, x, "full_block_rms", "blk." + layer + ".attn_norm",
                    weights, config, DataType.BFLOAT16);
            return buildGatedAttention(sd, rms, layer, config, weights, DataType.BFLOAT16,
                    position, cachePosition, mask, keys, values);
        }
    }

    private static Map<String, Snapshot> fullAttentionOutput(SameDiff sd, INDArray input,
            Snapshot keys, Snapshot values, int start, String[] outputs, List<String> failures) {
        int width = Math.toIntExact(input.size(1));
        assertTrue(start >= 0 && start + width <= 640);
        // Java snapshots own all cross-call state. Every invocation gets new independent
        // caller-owned cache buffers; the native attention op must commit writes into them.
        try (INDArray k = restore(keys); INDArray v = restore(values);
             INDArray x = input.dup('c'); INDArray position = Nd4j.scalar(DataType.INT64, start);
             INDArray cachePosition = Nd4j.scalar(DataType.INT64, start);
             INDArray mask = Nd4j.create(DataType.FLOAT, 1, 1, width, 640)) {
            float[] bias = new float[width * 640];
            for (int t = 0; t < width; t++) {
                Arrays.fill(bias, t * 640 + start + t + 1, (t + 1) * 640, Float.NEGATIVE_INFINITY);
            }
            try (INDArray b = Nd4j.createFromArray(bias)) { mask.assign(b.reshape(mask.shape())); }
            Snapshot original = snapshot(x);
            compare("full-attention/independent-key-copy", keys, snapshot(k), failures);
            compare("full-attention/independent-value-copy", values, snapshot(v), failures);
            Map<String, INDArray> feeds = new LinkedHashMap<>();
            feeds.put("x", x); feeds.put("keys", k); feeds.put("values", v);
            feeds.put("position_offset", position); feeds.put("cache_position", cachePosition);
            feeds.put("_causal_mask", mask);
            Map<String, INDArray> result = sd.output(feeds, outputs);
            Map<String, Snapshot> snapshots = new LinkedHashMap<>();
            for (String name : outputs) {
                INDArray array = result.get(name);
                assertNotNull(array, name);
                assertEquals(sd.getVariable(name).dataType(), array.dataType(), name);
                assertEquals(DataType.BFLOAT16, array.dataType(), name + " stored boundary");
                assertEquals(1, array.size(0), name);
                assertEquals(width, array.size(1), name);
                if (name.equals("attn_proj_3")) assertArrayEquals(new long[]{1, width, 5120}, array.shape());
                snapshots.put(name, snapshot(array));
            }
            Snapshot committedK = snapshot(k), committedV = snapshot(v);
            checkCacheCommit("keys", keys, committedK, snapshots.get("k_rope_3"), start, width, failures);
            checkCacheCommit("values", values, committedV, snapshots.get("v_heads_3"), start, width, failures);
            snapshots.put("keys", committedK); snapshots.put("values", committedV);
            compare("full-attention/input-not-mutated", original, snapshot(x), failures);
            return snapshots;
        }
    }

    private static void checkCacheCommit(String label, Snapshot before, Snapshot after, Snapshot projected,
                                         int start, int width, List<String> failures) {
        assertEquals(DataType.BFLOAT16, after.dtype);
        assertArrayEquals(new long[]{1, 640, 4, 256}, after.shape);
        int from = start * 1024, to = (start + width) * 1024;
        boolean changed = false;
        for (int i = 0; i < after.values.length; i++) {
            if (i >= from && i < to) {
                changed |= Float.floatToRawIntBits(before.values[i]) != Float.floatToRawIntBits(after.values[i]);
            } else {
                assertEquals(Float.floatToRawIntBits(before.values[i]), Float.floatToRawIntBits(after.values[i]),
                        label + " changed outside committed interval at " + i);
            }
        }
        assertTrue(changed, label + " must commit new projected tokens, not merely compare unchanged caches");
        Snapshot written = new Snapshot(after.dtype, new long[]{1, width, 4, 256},
                Arrays.copyOfRange(after.values, from, to));
        if (projected != null) compare("full-attention/" + label + "/projection-commit", projected, written, failures);
    }

    private static final class FirstGdnBlock extends LLaMAArchitecture {
        private final ModelOptQwenConfig config;
        FirstGdnBlock(ModelOptQwenConfig config) { this.config = config; }
        @Override protected int getGdnKeyHeads(int valueHeads) {
            assertEquals(config.getValueHeads(), valueHeads);
            return config.getKeyHeads();
        }
        @Override protected boolean multiplyNormInFloat() { return true; }
        SDVariable build(SameDiff sd, SDVariable x, Map<String, INDArray> weights,
                         SDVariable state, SDVariable conv, SDVariable length) {
            ArchitectureConfig architecture = config.getArchitecture();
            SDVariable rms = buildRMSNorm(sd, x, "block_rms", "blk.0.attn_norm", weights,
                    architecture, DataType.BFLOAT16);
            return buildGDNAttention(sd, rms, 0, architecture, weights, DataType.BFLOAT16, state, conv, length);
        }
    }

    private static Map<String, Snapshot> blockOutput(SameDiff sd, INDArray input, Snapshot state,
            Snapshot conv, Map<String, String> stages, String[] outputs) {
        // Rehydrate independent snapshots then dup: no borrowed graph output ever feeds
        // the next call, and no W5 state can alias the independent W1 chain.
        try (INDArray stateCopy = restore(state); INDArray convCopy = restore(conv);
             INDArray stateInput = stateCopy.dup('c'); INDArray convInput = convCopy.dup('c');
             INDArray length = Nd4j.scalar(DataType.INT64, input.size(1))) {
            Map<String, INDArray> feeds = new LinkedHashMap<>();
            feeds.put("x", input);
            feeds.put("state", stateInput);
            feeds.put("conv", convInput);
            feeds.put("actual_sequence_length", length);
            Map<String, INDArray> result = sd.output(feeds, outputs);
            Map<String, Snapshot> snapshots = new LinkedHashMap<>();
            for (Map.Entry<String, String> stage : stages.entrySet()) {
                INDArray value = result.get(stage.getValue());
                assertNotNull(value, stage.getKey());
                assertEquals(sd.getVariable(stage.getValue()).dataType(), value.dataType(), stage.getKey());
                assertEquals(1, value.size(0), stage.getKey());
                if (stage.getKey().equals("state")) {
                    assertEquals(DataType.FLOAT, value.dataType());
                    assertArrayEquals(new long[]{1, 48, 128, 128}, value.shape());
                } else if (stage.getKey().equals("convState")) {
                    assertEquals(DataType.BFLOAT16, value.dataType());
                    assertArrayEquals(new long[]{1, 10240, 3}, value.shape());
                } else {
                    assertEquals(input.size(1), value.size(1), stage.getKey());
                    if (stage.getKey().startsWith("recurrence/")) {
                        assertEquals(DataType.FLOAT, value.dataType(), stage.getKey());
                        boolean vector = stage.getKey().equals("recurrence/Q")
                                || stage.getKey().equals("recurrence/K") || stage.getKey().equals("recurrence/V");
                        assertArrayEquals(vector ? new long[]{1, input.size(1), 48, 128}
                                : new long[]{1, input.size(1), 48}, value.shape(), stage.getKey());
                    }
                    if (stage.getKey().equals("attention")) {
                        assertEquals(DataType.BFLOAT16, value.dataType());
                        assertArrayEquals(new long[]{1, input.size(1), 5120}, value.shape());
                    }
                }
                snapshots.put(stage.getKey(), snapshot(value));
            }
            return snapshots;
        }
    }

    private static INDArray restore(Snapshot snapshot) {
        INDArray result = Nd4j.create(snapshot.dtype, snapshot.shape);
        try (INDArray floats = Nd4j.createFromArray(snapshot.values)) {
            result.assign(floats.reshape(snapshot.shape));
        }
        return result;
    }

    private static INDArray timeCopy(INDArray array, int t) {
        return array.rank() == 4 ? array.get(all(), interval(t, t + 1), all(), all()).dup('c')
                : array.get(all(), interval(t, t + 1), all()).dup('c');
    }

    private static void directRecurrence(Map<String, Snapshot> window, Snapshot initial, List<String> failures) {
        try (INDArray q = restore(window.get("recurrence/Q")); INDArray k = restore(window.get("recurrence/K"));
             INDArray v = restore(window.get("recurrence/V")); INDArray beta = restore(window.get("recurrence/beta"));
             INDArray g = restore(window.get("recurrence/g")); INDArray state = restore(initial);
             INDArray length = Nd4j.scalar(DataType.INT64, IDS.length);
             INDArray one = Nd4j.scalar(DataType.INT64, 1)) {
            INDArray[] full = Nd4j.exec(new GatedDeltaRule(q, k, v, beta, g, state, length));
            try (INDArray fullOut = full[0]; INDArray fullState = full[1]) {
                Snapshot fullOutputSnapshot = snapshot(fullOut), fullStateSnapshot = snapshot(fullState);
                Snapshot next = initial;
                List<Snapshot> scalar = new ArrayList<>();
                for (int t = 0; t < IDS.length; t++) {
                    try (INDArray qt = timeCopy(q, t); INDArray kt = timeCopy(k, t); INDArray vt = timeCopy(v, t);
                         INDArray bt = timeCopy(beta, t); INDArray gt = timeCopy(g, t); INDArray previous = restore(next)) {
                        INDArray[] step = Nd4j.exec(new GatedDeltaRule(qt, kt, vt, bt, gt, previous, one));
                        try (INDArray out = step[0]; INDArray stateOut = step[1]; INDArray detached = stateOut.dup('c')) {
                            scalar.add(snapshot(out));
                            next = snapshot(detached);
                        }
                    }
                }
                compare("direct-GDN/common-W5-inputs/output", fullOutputSnapshot, join(scalar), failures);
                compare("direct-GDN/common-W5-inputs/state", fullStateSnapshot, next, failures);
                compare("direct-GDN/vs-shared-block/output", window.get("gdn_out_0"), fullOutputSnapshot, failures);
                compare("direct-GDN/vs-shared-block/state", window.get("state"), fullStateSnapshot, failures);
            }
        }
    }

    /** First-layer only, same transforms as ModelOptQwenImporter.tensorSpecs/adapt. */
    private static final class BlockWeights implements AutoCloseable {
        final Map<String, INDArray> values = new LinkedHashMap<>();
        final List<INDArray> owned = new ArrayList<>();
        final File cache;
        final JsonObject index;
        long bytes;
        long payloadBudget = 128L * 1024 * 1024;
        String sourceLayer = LAYER;
        String destinationLayer = "blk.0.";
        BlockWeights(File cache, JsonObject index, ModelOptQwenConfig config) throws IOException {
            this(cache, index, config, 0, false);
        }
        BlockWeights(File cache, JsonObject index, ModelOptQwenConfig config,
                     int layer, boolean connected) throws IOException {
            this.cache = cache;
            this.index = index;
            sourceLayer = TEXT + "layers." + layer + ".";
            destinationLayer = "blk." + layer + ".";
            if (connected) payloadBudget = 480L * 1024 * 1024;
            try {
                int hidden = config.getArchitecture().getHiddenSize();
                INDArray rawNorm = load("input_layernorm.weight", "BF16", hidden);
                values.put(destinationLayer + "attn_norm.weight", own(oneCenteredGamma(rawNorm)));
                int valueWidth = config.getValueHeads() * config.getValueHeadDim();
                packed("in_proj_qkv", "attn_qkv", config.getConvChannels(), hidden);
                packed("in_proj_z", "attn_gate", valueWidth, hidden);
                packed("out_proj", "ssm_out", hidden, valueWidth);
                values.put(destinationLayer + "ssm_alpha.weight", load("linear_attn.in_proj_a.weight", "BF16", 48, hidden));
                values.put(destinationLayer + "ssm_beta.weight", load("linear_attn.in_proj_b.weight", "BF16", 48, hidden));
                INDArray conv = load("linear_attn.conv1d.weight", "BF16", config.getConvChannels(), 1, config.getConvKernel());
                values.put(destinationLayer + "ssm_conv1d.weight", conv.reshape(config.getConvChannels(), config.getConvKernel()));
                INDArray rawA = load("linear_attn.A_log", "F32|BF16", 48);
                float[] a = snapshot(rawA).values;
                for (int i = 0; i < a.length; i++) {
                    a[i] = (float) -Math.exp(a[i]);
                    assertTrue(Float.isFinite(a[i]));
                }
                values.put(destinationLayer + "ssm_a", own(Nd4j.createFromArray(a)));
                values.put(destinationLayer + "ssm_dt.bias", load("linear_attn.dt_bias", "F32|BF16", 48));
                // Unlike input_layernorm, the gated per-head norm is NOT offset by 1.
                values.put(destinationLayer + "ssm_norm.weight", load("linear_attn.norm.weight", "BF16", 128));
                if (connected) {
                    values.put(destinationLayer + "post_attention_norm.weight", own(oneCenteredGamma(
                            load("post_attention_layernorm.weight", "BF16", hidden))));
                    int intermediate = config.getArchitecture().getIntermediateSize();
                    packedMlp("gate", intermediate, hidden);
                    packedMlp("up", intermediate, hidden);
                    packedMlp("down", hidden, intermediate);
                }
                log.info("BLOCK_PARITY layer={} connected={} retained packed payload={} bytes", layer, connected, bytes);
            } catch (IOException | RuntimeException | Error failure) {
                try { close(); } catch (RuntimeException cleanup) { failure.addSuppressed(cleanup); }
                throw failure;
            }
        }
        BlockWeights(File cache, JsonObject index, ModelOptQwenConfig config, int layer) throws IOException {
            this.cache = cache;
            this.index = index;
            sourceLayer = TEXT + "layers." + layer + ".";
            String dst = "blk." + layer + ".";
            try {
                ArchitectureConfig c = config.getArchitecture();
                values.put(dst + "attn_norm.weight", own(oneCenteredGamma(
                        load("input_layernorm.weight", "BF16", c.getHiddenSize()))));
                for (String role : new String[]{"q", "k"}) {
                    values.put(dst + "attn_" + role + "_norm.weight", own(oneCenteredGamma(
                            load("self_attn." + role + "_norm.weight", "BF16", c.getHeadDimension()))));
                }
                int width = c.getNumAttentionHeads() * c.getHeadDimension();
                int kvWidth = c.getNumKVHeads() * c.getHeadDimension();
                packedProjection("self_attn.q_proj", dst + "attn_q.weight", 2 * width, c.getHiddenSize());
                packedProjection("self_attn.k_proj", dst + "attn_k.weight", kvWidth, c.getHiddenSize());
                packedProjection("self_attn.v_proj", dst + "attn_v.weight", kvWidth, c.getHiddenSize());
                packedProjection("self_attn.o_proj", dst + "attn_output.weight", c.getHiddenSize(), width);
                log.info("FULL_ATTENTION_PARITY layer={} retained bytes={}; no MLP/embedding/full model", layer, bytes);
            } catch (IOException | RuntimeException | Error failure) {
                try { close(); } catch (RuntimeException cleanup) { failure.addSuppressed(cleanup); }
                throw failure;
            }
        }
        BlockWeights(File cache, JsonObject index, ArchitectureConfig config) throws IOException {
            this.cache = cache;
            this.index = index;
            int hidden = config.getHiddenSize(), intermediate = config.getIntermediateSize();
            // Three packed matrices + E4M3 block scales + BF16 norm + three F32 global scales.
            payloadBudget = 3L * hidden * intermediate * 9 / 16 + 2L * hidden + 12;
            try {
                values.put("blk.0.post_attention_norm.weight", own(oneCenteredGamma(
                        load("post_attention_layernorm.weight", "BF16", hidden))));
                packedMlp("gate", intermediate, hidden);
                packedMlp("up", intermediate, hidden);
                packedMlp("down", hidden, intermediate);
                assertEquals(payloadBudget, bytes, "Exact pinned MLP payload budget");
            } catch (IOException | RuntimeException | Error failure) {
                try { close(); } catch (RuntimeException cleanup) { failure.addSuppressed(cleanup); }
                throw failure;
            }
        }
        private void packedMlp(String role, int n, int k) throws IOException {
            String source = "mlp." + role + "_proj", key = destinationLayer + "ffn_" + role + ".weight";
            JsonObject quant = json(cached(cache, "hf_quant_config.json")).getAsJsonObject("quantization")
                    .getAsJsonObject("quantized_layers").getAsJsonObject(sourceLayer + source);
            assertNotNull(quant, source);
            assertEquals("W4A16_NVFP4", quant.get("quant_algo").getAsString(), source);
            assertEquals(16, quant.get("group_size").getAsInt(), source);
            assertEquals(0, k % 16);
            values.put(key, load(source + ".weight", "U8", n, k / 2));
            values.put(key + QuantizedLinear.MODELOPT_BLOCK_SCALE,
                    load(source + ".weight_scale", "F8_E4M3", n, k / 16));
            values.put(key + QuantizedLinear.MODELOPT_GLOBAL_SCALE, scale(source + ".weight_scale_2"));
            // ModelOpt W4A16 does not use the exported input_scale as an activation quantizer.
        }
        private INDArray own(INDArray array) { owned.add(array); return array; }
        private INDArray load(String suffix, String dtype, long... shape) throws IOException {
            String name = sourceLayer + suffix;
            try (SafeTensorsReader reader = SafeTensorsReader.open(shard(cache, index, name))) {
                TensorInfo info = reader.getTensorInfo(name);
                assertNotNull(info, name);
                assertTrue(Arrays.asList(dtype.split("\\|")).contains(info.getDtype()), name + " dtype=" + info.getDtype());
                if (shape.length == 0) {
                    long count = 1;
                    for (long dimension : info.getShape()) count = Math.multiplyExact(count, dimension);
                    assertEquals(1, count, name);
                } else assertArrayEquals(shape, info.getShape(), name);
                bytes = Math.addExact(bytes, info.getDataLength());
                assertTrue(info.getDataLength() <= 64L * 1024 * 1024 && bytes <= payloadBudget,
                        "First-layer payload budget exceeded: " + name);
                log.info("BLOCK_PARITY tensor={} dtype={} shape={} bytes={} shard={}", name,
                        info.getDtype(), Arrays.toString(info.getShape()), info.getDataLength(), reader.getFile());
                return own(reader.readTensor(name));
            }
        }
        private void packed(String hf, String canonical, int n, int k) throws IOException {
            packedProjection("linear_attn." + hf, destinationLayer + canonical + ".weight", n, k);
        }
        private void packedProjection(String source, String key, int n, int k) throws IOException {
            JsonObject quant = json(cached(cache, "hf_quant_config.json")).getAsJsonObject("quantization")
                    .getAsJsonObject("quantized_layers").getAsJsonObject(sourceLayer + source);
            assertNotNull(quant, source);
            assertEquals("FP8", quant.get("quant_algo").getAsString(), source);
            values.put(key, load(source + ".weight", "F8_E4M3", n, k));
            values.put(key + QuantizedLinear.MODELOPT_FP8_SCALE, scale(source + ".weight_scale"));
            values.put(key + QuantizedLinear.MODELOPT_INPUT_SCALE, scale(source + ".input_scale"));
        }
        private INDArray scale(String source) throws IOException {
            INDArray raw = load(source, "F32");
            assertTrue(Float.isFinite(raw.getFloat(0)) && raw.getFloat(0) > 0, source);
            return raw.rank() == 0 ? raw : raw.reshape(new long[0]);
        }
        @Override public void close() {
            for (INDArray array : owned) {
                if (array.wasClosed() || array.data().wasClosed()) continue;
                array.data().setConstant(false);
                array.setCloseable(true);
                array.close();
            }
        }
    }

    private static void exercise(String label, INDArray rows, INDArray gamma, INDArray alpha,
                                 INDArray beta, double epsilon, boolean includeRms,
                                 List<String> failures) {
        List<String> stages = new ArrayList<>();
        stages.add("input_echo");
        if (includeRms) stages.add("rms");
        stages.addAll(Arrays.asList(PROJECTIONS));
        String[] outputs = stages.toArray(new String[0]);
        // Inputs live longer than the graph, including both shape-keyed plans. Never close a
        // borrowed output; snapshot it before the next call can reuse its backing storage.
        try (INDArray windowInput = rows.dup('c');
             INDArray scalarInput = Nd4j.create(DataType.BFLOAT16, 1, 1, rows.size(2));
             SameDiff sd = SameDiff.create()) {
            SDVariable x = sd.placeHolder("x", DataType.BFLOAT16, 1, -1, rows.size(2));
            sd.identity("input_echo", x);
            SDVariable input = x;
            if (includeRms) {
                input = new RmsNorm(sd, x, sd.var("gamma", gamma.dup('c')), epsilon).outputVariable();
                input = sd.updateVariableNameAndReference(input, "rms");
            }
            SDVariable a = QuantizedLinear.matMul(sd, "alpha", input, sd.var("alpha_weight", alpha.dup('c')),
                    Collections.emptyMap(), "blk.0.ssm_alpha.weight", DataType.BFLOAT16);
            sd.updateVariableNameAndReference(a.castTo(DataType.FLOAT), "alpha_compute");
            SDVariable b = QuantizedLinear.matMul(sd, "beta", input, sd.var("beta_weight", beta.dup('c')),
                    Collections.emptyMap(), "blk.0.ssm_beta.weight", DataType.BFLOAT16);
            SDVariable sigmoid = sd.nn().sigmoid("beta_sigmoid", b);
            sd.updateVariableNameAndReference(sigmoid.castTo(DataType.FLOAT), "beta_compute");
            for (String output : outputs) assertNotNull(sd.getVariable(output), output);
            sd.setOutputs(outputs);
            assertTrue(sd.isDspAutoCompileEnabled(), "DSP must remain enabled");
            log.info("WINDOW_PARITY {} DSP autoCompile={} nativeAutoCompile={} mode={}", label,
                    sd.isDspAutoCompileEnabled(), sd.isDspNativeAutoCompileEnabled(), sd.getGraphExecutionMode());
            Snapshot expectedInput = snapshot(rows);
            Map<String, Snapshot> firstWindow = null;
            // Observe cold and repeated shape-keyed execution, without clearing/freezing/forcing plans.
            for (int repeat = 0; repeat < 3; repeat++) {
                Map<String, Snapshot> window = evaluate(sd, windowInput, outputs);
                compare(label + "/repeat=" + repeat + "/W5-input-echo", expectedInput,
                        window.get("input_echo"), failures);
                Map<String, List<Snapshot>> scalar = new LinkedHashMap<>();
                for (String stage : stages) scalar.put(stage, new ArrayList<>());
                for (int row = 0; row < IDS.length; row++) {
                    try (INDArray source = rowCopy(rows, row)) {
                        scalarInput.assign(source);
                        Snapshot expectedRow = snapshot(source);
                        String context = label + "/repeat=" + repeat + "/row=" + row + "/tokenId=" + IDS[row];
                        compare(context + "/input-copy", expectedRow, snapshot(scalarInput), failures);
                        Map<String, Snapshot> result = evaluate(sd, scalarInput, outputs);
                        compare(context + "/input-echo", expectedRow, result.get("input_echo"), failures);
                        for (String stage : stages) scalar.get(stage).add(result.get(stage));
                    }
                }
                for (String stage : stages) {
                    String context = label + "/repeat=" + repeat + "/" + stage;
                    Snapshot scalarStage = join(scalar.get(stage));
                    boolean accumulator = stage.endsWith("_accum");
                    compare(context + " W5-vs-5xW1", window.get(stage), scalarStage, failures, !accumulator);
                    if (accumulator) {
                        // FP32 GEMM reduction order may vary with M. Check both against
                        // a DOUBLE oracle and its FP32 forward-error bound; the BF16
                        // rounding boundary and every downstream stage remain bit-exact.
                        Snapshot weight = snapshot(stage.startsWith("alpha") ? alpha : beta);
                        String inputStage = includeRms ? "rms" : "input_echo";
                        validateAccumulation(context + "/W5", window.get(stage), window.get(inputStage), weight);
                        validateAccumulation(context + "/W1", scalarStage, join(scalar.get(inputStage)), weight);
                    }
                    // Warmup and compiled GEMM may use different FP32 reduction orders
                    // too. Each accumulator was checked against the DOUBLE oracle above;
                    // all stored BF16 and downstream outputs must still be bit-identical.
                    if (firstWindow != null) compare(context + " W5-repeat", firstWindow.get(stage),
                            window.get(stage), failures, !accumulator);
                }
                if (firstWindow == null) firstWindow = window;
                compare(label + "/input-not-mutated", expectedInput, snapshot(windowInput), failures);
            }
        }
    }

    private static Map<String, Snapshot> evaluate(SameDiff sd, INDArray input, String[] outputs) {
        Map<String, INDArray> result = sd.output(Collections.singletonMap("x", input), outputs);
        Map<String, Snapshot> snapshots = new LinkedHashMap<>();
        for (String name : outputs) {
            INDArray array = result.get(name);
            assertNotNull(array, name);
            DataType expected = name.endsWith("_accum") || name.endsWith("_compute")
                    ? DataType.FLOAT : DataType.BFLOAT16;
            assertEquals(expected, array.dataType(), name);
            assertEquals(3, array.rank(), name);
            assertEquals(1, array.size(0), name);
            assertEquals(input.size(1), array.size(1), name);
            assertEquals(name.equals("input_echo") || name.equals("rms") ? input.size(2) : 48,
                    array.size(2), name);
            snapshots.put(name, snapshot(array));
        }
        return snapshots;
    }

    private static INDArray rowCopy(INDArray rows, int row) {
        return rows.get(all(), interval(row, row + 1), all()).dup('c');
    }

    private static final class Snapshot {
        final DataType dtype;
        final long[] shape;
        final float[] values;
        Snapshot(DataType dtype, long[] shape, float[] values) {
            this.dtype = dtype;
            this.shape = shape;
            this.values = values;
        }
    }

    private static Snapshot snapshot(INDArray array) {
        // Widening finite BF16 to FLOAT is exact (including signed zero). Compare raw
        // float bits, not an epsilon; duplicate views before the bulk host read.
        try (INDArray copy = array.dup('c')) {
            if (copy.dataType() == DataType.FLOAT) {
                return new Snapshot(array.dataType(), array.shape().clone(), copy.data().asFloat());
            }
            assertEquals(DataType.BFLOAT16, copy.dataType());
            try (INDArray wide = copy.castTo(DataType.FLOAT)) {
                return new Snapshot(array.dataType(), array.shape().clone(), wide.data().asFloat());
            }
        }
    }

    private static Snapshot join(List<Snapshot> rows) {
        Snapshot first = rows.get(0);
        long[] shape = first.shape.clone();
        assertEquals(1, shape[1]);
        shape[1] = rows.size();
        float[] values = new float[Math.multiplyExact(first.values.length, rows.size())];
        for (int i = 0; i < rows.size(); i++) {
            assertEquals(first.dtype, rows.get(i).dtype);
            assertArrayEquals(first.shape, rows.get(i).shape);
            System.arraycopy(rows.get(i).values, 0, values, i * first.values.length, first.values.length);
        }
        return new Snapshot(first.dtype, shape, values);
    }

    private static void validateAccumulation(String stage, Snapshot output, Snapshot input, Snapshot weight) {
        int k = Math.toIntExact(input.shape[2]);
        int n = Math.toIntExact(weight.shape[0]);
        double u = Math.scalb(1.0, -24);
        double gamma = k * u / (1.0 - k * u);
        for (int row = 0; row < output.values.length / n; row++) {
            for (int col = 0; col < n; col++) {
                double sum = 0, absoluteSum = 0;
                for (int inner = 0; inner < k; inner++) {
                    double product = (double) input.values[row * k + inner] * weight.values[col * k + inner];
                    sum += product;
                    absoluteSum += Math.abs(product);
                }
                double bound = gamma * absoluteSum + k * (double) Float.MIN_VALUE;
                assertEquals(sum, output.values[row * n + col], bound,
                        stage + " FP32 accumulation error at row=" + row + " column=" + col);
            }
        }
    }

    private static void compare(String stage, Snapshot window, Snapshot scalar, List<String> failures) {
        compare(stage, window, scalar, failures, true);
    }

    private static void compare(String stage, Snapshot window, Snapshot scalar, List<String> failures,
                                boolean requireBitExact) {
        assertEquals(window.dtype, scalar.dtype, stage);
        assertArrayEquals(window.shape, scalar.shape, stage);
        assertEquals(window.values.length, scalar.values.length, stage);
        int firstBit = -1, firstNumeric = -1, count = 0, nonFinite = 0;
        double max = 0;
        for (int i = 0; i < window.values.length; i++) {
            float a = window.values[i], b = scalar.values[i];
            if (!Float.isFinite(a) || !Float.isFinite(b)) nonFinite++;
            if (Float.floatToRawIntBits(a) != Float.floatToRawIntBits(b)) {
                if (firstBit < 0) firstBit = i;
                count++;
            }
            if (a != b && firstNumeric < 0) firstNumeric = i;
            max = Math.max(max, Math.abs((double) a - b));
        }
        String detail = stage + " dtype=" + window.dtype + " shape=" + Arrays.toString(window.shape)
                + " bitMismatches=" + count + " firstBit=" + firstBit + " firstNumeric=" + firstNumeric
                + " maxAbsDiff=" + max + " nonFinite=" + nonFinite;
        if (firstBit >= 0) {
            long rowWidth = 1;
            for (int d = 2; d < window.shape.length; d++) rowWidth *= window.shape[d];
            int row = Math.toIntExact(firstBit / rowWidth);
            detail += " axis1=" + row
                    + (window.shape[1] == IDS.length && row < IDS.length ? " tokenId=" + IDS[row] : "")
                    + " innerOffset=" + firstBit % rowWidth
                    + " W5=" + window.values[firstBit] + " scalar=" + scalar.values[firstBit]
                    + " W5bits=0x" + Integer.toHexString(Float.floatToRawIntBits(window.values[firstBit]))
                    + " scalarBits=0x" + Integer.toHexString(Float.floatToRawIntBits(scalar.values[firstBit]));
        }
        log.info("WINDOW_PARITY {}", detail);
        if ((requireBitExact && count != 0) || nonFinite != 0) failures.add(detail);
    }

    private static INDArray oneCenteredGamma(INDArray raw) {
        float[] values = snapshot(raw).values;
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0f + values[i]; // Same FLOAT adaptation as ModelOptQwenImporter.adapt.
            assertTrue(Float.isFinite(values[i]), "Nonfinite gamma at " + i);
        }
        return Nd4j.createFromArray(values);
    }

    private static INDArray smallTensor(File cache, JsonObject index, String name, long... shape) throws IOException {
        try (SafeTensorsReader reader = SafeTensorsReader.open(shard(cache, index, name))) {
            TensorInfo info = reader.getTensorInfo(name);
            assertNotNull(info, name);
            assertArrayEquals(shape, info.getShape(), name);
            assertEquals("BF16", info.getDtype(), name);
            assertTrue(info.getDataLength() <= 1024 * 1024, "Only small dense tensors may be loaded: " + name);
            log.info("WINDOW_PARITY tensor={} shard={} dtype={} shape={} bytes={}", name,
                    reader.getFile(), info.getDtype(), Arrays.toString(info.getShape()), info.getDataLength());
            return reader.readTensor(name);
        }
    }

    private static INDArray embeddingRows(File cache, JsonObject index, int hidden) throws IOException {
        return embeddingRows(cache, index, hidden, IDS);
    }

    private static INDArray embeddingRows(File cache, JsonObject index, int hidden, int[] ids) throws IOException {
        assertTrue(ids.length > 0 && ids.length <= 128, "Bounded row-only embedding read");
        String name = TEXT + "embed_tokens.weight";
        File file = shard(cache, index, name);
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            SafeTensorsHeader header = SafeTensorsHeader.fromRandomAccessFile(raf);
            TensorInfo info = header.getTensorInfo(name);
            assertNotNull(info, name);
            assertEquals("BF16", info.getDtype());
            assertArrayEquals(new long[]{248320, hidden}, info.getShape());
            long rowBytes = Math.multiplyExact((long) hidden, 2L);
            assertEquals(Math.multiplyExact(info.getShape()[0], rowBytes), info.getDataLength());
            assertTrue(info.getDataStart() >= 0);
            long start = Math.addExact(header.getDataOffset(), info.getDataStart());
            assertTrue(Math.addExact(start, info.getDataLength()) <= raf.length(), "Truncated embedding payload");
            byte[] bytes = new byte[Math.toIntExact(Math.multiplyExact(rowBytes, ids.length))];
            float[] expected = new float[Math.multiplyExact(hidden, ids.length)];
            for (int row = 0; row < ids.length; row++) {
                assertTrue(ids[row] >= 0 && ids[row] < info.getShape()[0]);
                long offset = Math.addExact(start, Math.multiplyExact((long) ids[row], rowBytes));
                raf.seek(offset);
                raf.readFully(bytes, Math.toIntExact(row * rowBytes), Math.toIntExact(rowBytes));
                log.info("WINDOW_PARITY embedding token={} shard={} absoluteOffset={} bytes={}",
                        ids[row], file, offset, rowBytes);
            }
            ByteBuffer little = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < expected.length; i++) {
                expected[i] = Float.intBitsToFloat(Short.toUnsignedInt(little.getShort()) << 16);
                assertTrue(Float.isFinite(expected[i]), "Nonfinite cached embedding at " + i);
            }
            if (ByteOrder.nativeOrder() != ByteOrder.LITTLE_ENDIAN) {
                for (int i = 0; i < bytes.length; i += 2) {
                    byte b = bytes[i]; bytes[i] = bytes[i + 1]; bytes[i + 1] = b;
                }
            }
            INDArray result = Nd4j.createUninitialized(DataType.BFLOAT16, new long[]{1, ids.length, hidden}, 'c');
            try {
                // Borrow host pointer exactly as SafeTensorsReader does. Do not free this alias.
                new BytePointer(result.data().pointer()).capacity(bytes.length).put(bytes);
                Nd4j.getAffinityManager().tagLocation(result, AffinityManager.Location.HOST);
                List<String> errors = new ArrayList<>();
                compare("embedding/raw-row-storage", new Snapshot(DataType.BFLOAT16,
                        result.shape().clone(), expected), snapshot(result), errors);
                assertTrue(errors.isEmpty(), () -> String.join("\n", errors));
                return result;
            } catch (RuntimeException | Error failure) {
                result.close();
                throw failure;
            }
        }
    }

    private static File shard(File cache, JsonObject index, String tensor) throws IOException {
        assertTrue(index.has(tensor), "Missing pinned index entry " + tensor);
        String shard = index.get(tensor).getAsString();
        assertTrue(shard.matches("model-[0-9]{5}-of-00003\\.safetensors"), shard);
        return cached(cache, shard);
    }

    private static File cached(File cache, String suffix) throws IOException {
        File file = new File(cache, PREFIX + suffix);
        if (!file.isFile()) throw new IOException("Required pinned cache file missing (downloads forbidden): " + file);
        return file;
    }

    private static JsonObject json(File file) throws IOException {
        try (Reader reader = Files.newBufferedReader(file.toPath(), StandardCharsets.UTF_8)) {
            return new Gson().fromJson(reader, JsonObject.class);
        }
    }
}
