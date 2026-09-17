/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.safetensors;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.LLaMAArchitecture;
import org.nd4j.ggml.architecture.QuantizedLinear;
import org.nd4j.ggml.convert.ConversionOptions;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.eclipse.deeplearning4j.safetensors.ModelOptQwenConfig.require;

/**
 * Executable text-only ModelOpt importer. Evidence: {@link ModelOptQwenConfig#EVIDENCE_BASE}
 * config.json and hf_quant_config.json; layout follows Transformers 5.7.0 Qwen3_5.
 *
 * <p>Accepts actual shard files, including revision-prefixed download cache names. Reads each
 * tensor with the reader's bounded staging buffer; retained memory is packed model storage,
 * not a dense dequantized model. Vision is not imported. The checkpoint's dense NextN/MTP
 * predictor is imported with its own cache branch and the shared packed output head. KV caches
 * use the graph's floating cache ABI, not the exporter's optional FP8 KV-cache compression.
 * NVFP4 projections are W4A16 (input_scale is not an activation FP4 quantizer); FP8 projections
 * use the checkpoint's static input_scale in the native ModelOpt op.</p>
 */
public final class ModelOptQwenImporter {
    private static final String TEXT = "model.language_model.";

    private ModelOptQwenImporter() { }

    /** Caller owns the returned model and must close it after all generation sessions finish. */
    public static ImportedModel importTextOnly(File configFile, File quantFile, List<File> shards) throws IOException {
        return importTextOnly(configFile, quantFile, shards, DataType.BFLOAT16, true);
    }

    /**
     * Packed storage is invariant; computeType controls activations and dense auxiliary projections only.
     * Reads sibling generation_config.json (retaining a revision prefix when present).
     * MTP always retains full target logits for verification, in addition to last-position prefill logits.
     */
    public static ImportedModel importTextOnly(File configFile, File quantFile, List<File> shards,
            DataType computeType, boolean lastPositionLogitsOnly) throws IOException {
        return importTextOnly(ModelOptQwenConfig.read(configFile, quantFile), shards, computeType, lastPositionLogitsOnly);
    }

    public static ImportedModel importTextOnly(File configFile, File quantFile, File generationConfigFile,
            List<File> shards, DataType computeType, boolean lastPositionLogitsOnly) throws IOException {
        return importTextOnly(ModelOptQwenConfig.read(configFile, quantFile, generationConfigFile),
                shards, computeType, lastPositionLogitsOnly);
    }

    private static ImportedModel importTextOnly(ModelOptQwenConfig config, List<File> shards,
            DataType computeType, boolean lastPositionLogitsOnly) throws IOException {
        require(computeType == DataType.FLOAT || computeType == DataType.HALF || computeType == DataType.BFLOAT16,
                "ModelOpt compute requires FLOAT, HALF or BFLOAT16");
        Map<String, TensorSpec> specs = tensorSpecs(config);
        Map<String, File> sources = new LinkedHashMap<>();
        // Header preflight precedes any model-sized allocations.
        for (File shard : shards) {
            SafeTensorsHeader header = SafeTensorsHeader.fromFile(shard);
            for (String name : header.getTensorNames()) {
                if (sources.putIfAbsent(name, shard) != null) {
                    throw new IOException("Duplicate tensor across shards: " + name);
                }
                TensorSpec spec = specs.get(name);
                require(spec != null || !name.startsWith("mtp."), "Unmapped predictor tensor: " + name);
                if (spec != null) spec.validate(header.getTensorInfo(name));
            }
        }
        for (String name : specs.keySet()) {
            if (!sources.containsKey(name)) throw new IOException("Missing required text tensor: " + name);
        }

        List<INDArray> owned = new ArrayList<>();
        Map<String, INDArray> weights = new LinkedHashMap<>();
        SameDiff graph = SameDiff.create();
        try (MemoryWorkspace ignored = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
            for (File shard : shards) {
                try (SafeTensorsReader reader = SafeTensorsReader.open(shard)) {
                    for (String name : reader.getTensorNames()) {
                        TensorSpec spec = specs.get(name);
                        if (spec == null) continue;
                        INDArray raw = reader.readTensor(name);
                        owned.add(raw);
                        INDArray mapped = adapt(raw, spec, computeType, owned);
                        weights.put(spec.canonical, mapped);
                    }
                }
            }
            LLaMAArchitecture architecture = new LLaMAArchitecture() {
                @Override
                protected int getGdnKeyHeads(int valueHeads) {
                    require(valueHeads == config.getValueHeads(), "GDN value-head count disagrees with HF config");
                    return config.getKeyHeads();
                }
                @Override
                protected boolean multiplyNormInFloat() { return true; }
            };
            ConversionOptions options = ConversionOptions.builder().targetDataType(computeType)
                    .lastPositionLogitsOnly(lastPositionLogitsOnly).build();
            architecture.buildGraph(graph, config.getArchitecture(), weights, options);
            List<SDVariable> constants = new ArrayList<>();
            for (SDVariable variable : graph.variables()) {
                if (variable.getVariableType() == VariableType.VARIABLE) constants.add(variable);
            }
            graph.convertToConstants(constants);
            return new ImportedModel(graph, config, computeType, owned);
        } catch (IOException | RuntimeException | Error failure) {
            try { graph.close(); } catch (RuntimeException | Error cleanup) { failure.addSuppressed(cleanup); }
            closeArrays(owned, failure);
            throw failure;
        }
    }

    /**
     * Build all required tensor contracts before loading. Canonical names are logical graph keys,
     * not a claim that HF tensors have GGUF packing, dimension order, or transformed values.
     */
    private static Map<String, TensorSpec> tensorSpecs(ModelOptQwenConfig config) {
        Map<String, TensorSpec> specs = new LinkedHashMap<>();
        ArchitectureConfig c = config.getArchitecture();
        add(specs, TEXT + "embed_tokens.weight", "token_embd.weight", "BF16", Transform.NONE,
                c.getVocabSize(), c.getHiddenSize());
        add(specs, TEXT + "norm.weight", "output_norm.weight", "BF16", Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        linear(specs, config, "lm_head", "output.weight", c.getVocabSize(), c.getHiddenSize());
        for (int layer = 0; layer < c.getNumLayers(); layer++) {
            String hf = TEXT + "layers." + layer + ".";
            String dst = "blk." + layer + ".";
            add(specs, hf + "input_layernorm.weight", dst + "attn_norm.weight", "BF16",
                    Transform.ONE_CENTERED_NORM, c.getHiddenSize());
            add(specs, hf + "post_attention_layernorm.weight", dst + "post_attention_norm.weight", "BF16",
                    Transform.ONE_CENTERED_NORM, c.getHiddenSize());
            linear(specs, config, hf + "mlp.gate_proj", dst + "ffn_gate.weight", c.getIntermediateSize(), c.getHiddenSize());
            linear(specs, config, hf + "mlp.up_proj", dst + "ffn_up.weight", c.getIntermediateSize(), c.getHiddenSize());
            linear(specs, config, hf + "mlp.down_proj", dst + "ffn_down.weight", c.getHiddenSize(), c.getIntermediateSize());
            if ("full_attention".equals(c.getLayerTypes().get(layer))) {
                long width = (long) c.getNumAttentionHeads() * c.getHeadDimension();
                long kvWidth = (long) c.getNumKVHeads() * c.getHeadDimension();
                // HF [head0 Q | head0 gate | head1 Q | head1 gate ...], consumed unchanged.
                linear(specs, config, hf + "self_attn.q_proj", dst + "attn_q.weight", 2 * width, c.getHiddenSize());
                linear(specs, config, hf + "self_attn.k_proj", dst + "attn_k.weight", kvWidth, c.getHiddenSize());
                linear(specs, config, hf + "self_attn.v_proj", dst + "attn_v.weight", kvWidth, c.getHiddenSize());
                linear(specs, config, hf + "self_attn.o_proj", dst + "attn_output.weight", c.getHiddenSize(), width);
                add(specs, hf + "self_attn.q_norm.weight", dst + "attn_q_norm.weight", "BF16",
                        Transform.ONE_CENTERED_NORM, c.getHeadDimension());
                add(specs, hf + "self_attn.k_norm.weight", dst + "attn_k_norm.weight", "BF16",
                        Transform.ONE_CENTERED_NORM, c.getHeadDimension());
            } else {
                long valueWidth = (long) config.getValueHeads() * config.getValueHeadDim();
                linear(specs, config, hf + "linear_attn.in_proj_qkv", dst + "attn_qkv.weight",
                        config.getConvChannels(), c.getHiddenSize());
                linear(specs, config, hf + "linear_attn.in_proj_z", dst + "attn_gate.weight", valueWidth, c.getHiddenSize());
                linear(specs, config, hf + "linear_attn.out_proj", dst + "ssm_out.weight", c.getHiddenSize(), valueWidth);
                denseLinear(specs, config, hf + "linear_attn.in_proj_a", dst + "ssm_alpha.weight",
                        config.getValueHeads(), c.getHiddenSize());
                denseLinear(specs, config, hf + "linear_attn.in_proj_b", dst + "ssm_beta.weight",
                        config.getValueHeads(), c.getHiddenSize());
                add(specs, hf + "linear_attn.conv1d.weight", dst + "ssm_conv1d.weight", "BF16", Transform.CONV,
                        config.getConvChannels(), 1, config.getConvKernel());
                add(specs, hf + "linear_attn.A_log", dst + "ssm_a", "F32|BF16", Transform.NEGATIVE_EXP, config.getValueHeads());
                add(specs, hf + "linear_attn.dt_bias", dst + "ssm_dt.bias", "F32|BF16", Transform.NONE, config.getValueHeads());
                // Gated RMS norm is already one-centered; unlike QK/trunk norms do NOT add 1.
                add(specs, hf + "linear_attn.norm.weight", dst + "ssm_norm.weight", "BF16", Transform.NONE, config.getValueHeadDim());
            }
        }
        // ModelOpt excludes mtp* from quantization, not from checkpoint storage.
        // Upstream Qwen3_5MultiTokenPredictor: norm(embed), norm(hidden), concat, fc,
        // one full-attention decoder layer, norm, shared lm_head.
        String mtp = "blk." + c.getNumLayers() + ".";
        denseLinear(specs, config, "mtp.fc", mtp + "nextn.eh_proj.weight", c.getHiddenSize(), 2L * c.getHiddenSize());
        add(specs, "mtp.pre_fc_norm_embedding.weight", mtp + "nextn.enorm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        add(specs, "mtp.pre_fc_norm_hidden.weight", mtp + "nextn.hnorm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        add(specs, "mtp.norm.weight", mtp + "nextn.shared_head_norm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        String hfMtp = "mtp.layers.0.";
        add(specs, hfMtp + "input_layernorm.weight", mtp + "attn_norm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        add(specs, hfMtp + "post_attention_layernorm.weight", mtp + "post_attention_norm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHiddenSize());
        long mtpWidth = (long) c.getNumAttentionHeads() * c.getHeadDimension();
        long mtpKvWidth = (long) c.getNumKVHeads() * c.getHeadDimension();
        denseLinear(specs, config, hfMtp + "self_attn.q_proj", mtp + "attn_q.weight", 2 * mtpWidth, c.getHiddenSize());
        denseLinear(specs, config, hfMtp + "self_attn.k_proj", mtp + "attn_k.weight", mtpKvWidth, c.getHiddenSize());
        denseLinear(specs, config, hfMtp + "self_attn.v_proj", mtp + "attn_v.weight", mtpKvWidth, c.getHiddenSize());
        denseLinear(specs, config, hfMtp + "self_attn.o_proj", mtp + "attn_output.weight", c.getHiddenSize(), mtpWidth);
        add(specs, hfMtp + "self_attn.q_norm.weight", mtp + "attn_q_norm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHeadDimension());
        add(specs, hfMtp + "self_attn.k_norm.weight", mtp + "attn_k_norm.weight", "BF16",
                Transform.ONE_CENTERED_NORM, c.getHeadDimension());
        denseLinear(specs, config, hfMtp + "mlp.gate_proj", mtp + "ffn_gate.weight", c.getIntermediateSize(), c.getHiddenSize());
        denseLinear(specs, config, hfMtp + "mlp.up_proj", mtp + "ffn_up.weight", c.getIntermediateSize(), c.getHiddenSize());
        denseLinear(specs, config, hfMtp + "mlp.down_proj", mtp + "ffn_down.weight", c.getHiddenSize(), c.getIntermediateSize());
        for (Map.Entry<String, JsonElement> entry : config.quantizedLayers().entrySet()) {
            String name = entry.getKey();
            require(!name.startsWith("mtp."), "Expected dense checkpoint predictor: " + name);
            if ((name.startsWith(TEXT) || name.equals("lm_head")) && !specs.containsKey(name + ".weight")) {
                throw new IllegalArgumentException("Unmapped quantized text projection: " + name);
            }
        }
        return specs;
    }

    private static void denseLinear(Map<String, TensorSpec> specs, ModelOptQwenConfig config,
            String name, String canonical, long n, long k) {
        require(!config.quantizedLayers().has(name), "Expected dense auxiliary projection: " + name);
        add(specs, name + ".weight", canonical, "BF16", Transform.DENSE, n, k);
    }

    private static void linear(Map<String, TensorSpec> specs, ModelOptQwenConfig config,
            String name, String canonical, long n, long k) {
        JsonObject description = config.quantizedLayers().getAsJsonObject(name);
        require(description != null, "Missing quantization policy: " + name);
        String algorithm = description.get("quant_algo").getAsString();
        if ("W4A16_NVFP4".equals(algorithm)) {
            require(description.get("group_size").getAsInt() == 16 && k % 16 == 0, "Invalid NVFP4 block width: " + name);
            add(specs, name + ".weight", canonical, "U8", Transform.NONE, n, k / 2);
            add(specs, name + ".weight_scale", canonical + QuantizedLinear.MODELOPT_BLOCK_SCALE,
                    "F8_E4M3", Transform.NONE, n, k / 16);
            add(specs, name + ".weight_scale_2", canonical + QuantizedLinear.MODELOPT_GLOBAL_SCALE,
                    "F32", Transform.SCALAR);
            // The export contains input_scale but W4A16 explicitly does not quantize activations.
        } else if ("FP8".equals(algorithm)) {
            add(specs, name + ".weight", canonical, "F8_E4M3", Transform.NONE, n, k);
            add(specs, name + ".weight_scale", canonical + QuantizedLinear.MODELOPT_FP8_SCALE,
                    "F32", Transform.SCALAR);
            add(specs, name + ".input_scale", canonical + QuantizedLinear.MODELOPT_INPUT_SCALE,
                    "F32", Transform.SCALAR);
        } else {
            throw new IllegalArgumentException("Unsupported ModelOpt algorithm " + algorithm + " for " + name);
        }
    }

    private static void add(Map<String, TensorSpec> specs, String hf, String canonical, String dtype,
            Transform transform, long... shape) {
        require(specs.put(hf, new TensorSpec(canonical, dtype, transform, shape)) == null, "Duplicate spec: " + hf);
    }

    private enum Transform { NONE, DENSE, CONV, SCALAR, ONE_CENTERED_NORM, NEGATIVE_EXP }

    private static final class TensorSpec {
        final String canonical;
        final String dtype;
        final Transform transform;
        final long[] shape;
        TensorSpec(String canonical, String dtype, Transform transform, long[] shape) {
            this.canonical = canonical;
            this.dtype = dtype;
            this.transform = transform;
            this.shape = shape;
        }
        void validate(SafeTensorsHeader.TensorInfo info) {
            boolean validDtype = dtype.equals(info.getDtype())
                    || ("F32|BF16".equals(dtype) && ("F32".equals(info.getDtype()) || "BF16".equals(info.getDtype())));
            require(validDtype, "Wrong storage dtype for " + info.getName() + ": " + info.getDtype());
            if (transform == Transform.SCALAR) {
                long count = 1;
                for (long dim : info.getShape()) count = Math.multiplyExact(count, dim);
                require(count == 1, "Expected per-tensor scalar scale: " + info.getName());
            } else {
                require(Arrays.equals(shape, info.getShape()), "Wrong shape for " + info.getName()
                        + ": " + Arrays.toString(info.getShape()) + ", expected " + Arrays.toString(shape));
            }
        }
    }

    private static INDArray adapt(INDArray raw, TensorSpec spec, DataType computeType, List<INDArray> owned) {
        if (spec.transform == Transform.SCALAR) {
            float scale = raw.getFloat(0);
            require(Float.isFinite(scale) && scale > 0, "Invalid scale " + spec.canonical);
            return raw.rank() == 0 ? raw : raw.reshape(new long[0]);
        }
        if (spec.transform == Transform.ONE_CENTERED_NORM || spec.transform == Transform.NEGATIVE_EXP) {
            // Only small norm/coefficient vectors are transformed. Packed projections/scales never enter here.
            float[] values = new float[Math.toIntExact(raw.length())];
            for (int i = 0; i < values.length; i++) {
                float value = raw.getFloat(i);
                values[i] = spec.transform == Transform.ONE_CENTERED_NORM
                        ? 1.0f + value : (float) -Math.exp(value);
                require(Float.isFinite(values[i]), "Non-finite coefficient: " + spec.canonical);
            }
            INDArray mapped = Nd4j.createFromArray(values);
            owned.add(mapped);
            raw.close();
            return mapped;
        }
        INDArray result = raw;
        if ((spec.transform == Transform.CONV || spec.transform == Transform.DENSE) && raw.dataType() != computeType) {
            result = raw.castTo(computeType);
            owned.add(result);
            raw.close();
        }
        // HF [channels,1,kernel] -> native [channels,kernel], with no transpose or channel reordering.
        return spec.transform == Transform.CONV ? result.reshape(spec.shape[0], spec.shape[2]) : result;
    }

    private static void closeArrays(List<INDArray> arrays, Throwable failure) {
        for (INDArray array : arrays) {
            try {
                if (array.wasClosed() || array.data().wasClosed()) continue;
                array.data().setConstant(false);
                array.setCloseable(true);
                array.close();
            } catch (RuntimeException | Error cleanup) {
                failure.addSuppressed(cleanup);
            }
        }
    }

    /** Graph and arrays have one lifetime. Do not close the graph while a generation pipeline uses it. */
    public static final class ImportedModel implements AutoCloseable {
        private final SameDiff graph;
        private final ModelOptQwenConfig config;
        private final DataType computeType;
        private final List<INDArray> owned;
        private boolean closed;
        private ImportedModel(SameDiff graph, ModelOptQwenConfig config, DataType computeType, List<INDArray> owned) {
            this.graph = graph;
            this.config = config;
            this.computeType = computeType;
            this.owned = owned;
        }
        public SameDiff getGraph() { return graph; }
        /**
         * Import-owned generation metadata. Pass BOS/EOS/pad and the complete stopTokenIds set
         * to the generation consumer; the primary EOS alone is not the checkpoint's stop policy.
         */
        public ModelOptQwenConfig getConfig() { return config; }
        public DataType getComputeType() { return computeType; }
        @Override
        public void close() {
            if (closed) return;
            closed = true;
            RuntimeException failure = new IllegalStateException("Failed to close imported ModelOpt model");
            try { graph.close(); } catch (RuntimeException | Error cleanup) { failure.addSuppressed(cleanup); }
            closeArrays(owned, failure);
            owned.clear();
            if (failure.getSuppressed().length > 0) throw failure;
        }
    }
}
