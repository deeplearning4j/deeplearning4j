/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.safetensors;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import org.nd4j.ggml.architecture.ArchitectureConfig;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.HashSet;
import java.util.Set;

/** Explicit HF text configuration adapter; no GGUF dimension or RoPE conventions are inferred. */
public final class ModelOptQwenConfig {
    public static final String REPOSITORY = "nvidia/Qwen3.6-27B-NVFP4";
    public static final String REVISION = "0893e1606ff3d5f97a441f405d5fc541a6bdf404";
    public static final String EVIDENCE_BASE = "https://huggingface.co/" + REPOSITORY + "/resolve/" + REVISION + "/";

    private final ArchitectureConfig architecture;
    private final JsonObject quantizedLayers;
    private final int keyHeads;
    private final int valueHeads;
    private final int keyHeadDim;
    private final int valueHeadDim;
    private final int convKernel;
    private final int eosTokenId;
    private final int bosTokenId;
    private final int padTokenId;
    private final Set<Integer> stopTokenIds;

    private ModelOptQwenConfig(JsonObject root, JsonObject quantRoot, JsonObject generation) {
        require("qwen3_5".equals(root.get("model_type").getAsString()), "Expected qwen3_5 model_type");
        JsonObject text = root.getAsJsonObject("text_config");
        JsonObject quant = quantRoot.getAsJsonObject("quantization");
        require("modelopt".equals(quantRoot.getAsJsonObject("producer").get("name").getAsString()),
                "Expected ModelOpt producer");
        require("MIXED_PRECISION".equals(quant.get("quant_algo").getAsString()), "Expected mixed precision");
        quantizedLayers = quant.getAsJsonObject("quantized_layers");
        require(quantizedLayers.equals(root.getAsJsonObject("quantization_config").get("quantized_layers")),
                "config.json and hf_quant_config.json quantized_layers disagree");
        validateGroups(root.getAsJsonObject("quantization_config").getAsJsonObject("config_groups"));
        require(!text.get("attention_bias").getAsBoolean() && text.get("attn_output_gate").getAsBoolean(),
                "Only bias-free gated Qwen attention is supported");
        require("silu".equals(text.get("hidden_act").getAsString()), "Expected SiLU MLP/GDN activation");
        require(!root.get("tie_word_embeddings").getAsBoolean(), "Expected independent lm_head");
        require("bfloat16".equals(text.get("dtype").getAsString()), "Expected BF16 checkpoint compute");
        int layers = positive(text, "num_hidden_layers");
        List<String> layerTypes = new ArrayList<>();
        for (JsonElement value : text.getAsJsonArray("layer_types")) {
            String type = value.getAsString();
            require("linear_attention".equals(type) || "full_attention".equals(type), "Unknown layer type: " + type);
            layerTypes.add(type);
        }
        require(layerTypes.size() == layers, "layer_types must describe every target layer");
        keyHeads = positive(text, "linear_num_key_heads");
        valueHeads = positive(text, "linear_num_value_heads");
        keyHeadDim = positive(text, "linear_key_head_dim");
        valueHeadDim = positive(text, "linear_value_head_dim");
        convKernel = positive(text, "linear_conv_kernel_dim");
        require(valueHeads % keyHeads == 0, "GDN value heads must be a multiple of key heads");
        int headDim = positive(text, "head_dim");
        JsonObject rope = text.getAsJsonObject("rope_parameters");
        require("default".equals(rope.get("rope_type").getAsString()), "Unsupported RoPE scaling");
        double partial = rope.get("partial_rotary_factor").getAsDouble();
        require(partial == text.get("partial_rotary_factor").getAsDouble(), "Conflicting partial rotary factors");
        int rotaryDims = (int) (headDim * partial);
        require(rotaryDims > 0 && rotaryDims <= headDim && rotaryDims % 2 == 0, "Invalid rotary width");
        int heads = positive(text, "num_attention_heads");
        int kvHeads = positive(text, "num_key_value_heads");
        require(heads % kvHeads == 0, "Invalid grouped attention heads");
        int mtpLayers = positive(text, "mtp_num_hidden_layers");
        require(mtpLayers == 1, "Only a single Qwen MTP predictor layer is supported");
        require(!text.get("mtp_use_dedicated_embeddings").getAsBoolean(),
                "Expected MTP to share the target embedding and output head");
        // Generation metadata is authoritative, not the text config's training EOS.
        JsonElement eos = generation.get("eos_token_id");
        require(eos != null && !eos.isJsonNull(), "generation_config.json must define eos_token_id");
        Set<Integer> stops = new LinkedHashSet<>();
        if (eos.isJsonArray()) {
            for (JsonElement id : eos.getAsJsonArray()) stops.add(id.getAsInt());
        } else {
            stops.add(eos.getAsInt());
        }
        require(!stops.isEmpty(), "Generation EOS list is empty");
        int vocab = positive(text, "vocab_size");
        require(stops.stream().allMatch(id -> id >= 0 && id < vocab), "Generation EOS outside vocabulary");
        eosTokenId = stops.iterator().next();
        stopTokenIds = Collections.unmodifiableSet(stops);
        bosTokenId = generation.get("bos_token_id").getAsInt();
        padTokenId = generation.get("pad_token_id").getAsInt();
        require(bosTokenId >= 0 && bosTokenId < vocab && padTokenId >= 0 && padTokenId < vocab,
                "Generation BOS/pad outside vocabulary");
        architecture = ArchitectureConfig.builder()
                .numLayers(layers).numMtpLayers(mtpLayers)
                .hiddenSize(positive(text, "hidden_size"))
                .intermediateSize(positive(text, "intermediate_size"))
                .vocabSize(positive(text, "vocab_size"))
                .contextLength(positive(text, "max_position_embeddings"))
                .numAttentionHeads(heads).numKVHeads(kvHeads).headDim(headDim)
                .layerNormEpsilon(text.get("rms_norm_eps").getAsFloat())
                .layerTypes(layerTypes).fullAttentionInterval(positive(text, "full_attention_interval"))
                .ropeFreqBase(rope.get("rope_theta").getAsFloat()).ropeDimensionCount(rotaryDims)
                // HF rotate_half pairs the two halves. Native fusedRoPE type 0 implements this.
                // Text positions have identical T/H/W coordinates, so interleaved MRoPE reduces to this.
                .ropeType(0).build();
    }

    private void validateGroups(JsonObject groups) {
        require(groups != null, "Missing config_groups (static activation policy is required)");
        Set<String> seen = new HashSet<>();
        for (Map.Entry<String, JsonElement> groupEntry : groups.entrySet()) {
            JsonObject group = groupEntry.getValue().getAsJsonObject();
            JsonObject weight = group.getAsJsonObject("weights");
            require("float".equals(weight.get("type").getAsString()) && !weight.get("dynamic").getAsBoolean(),
                    "Expected static floating weight quantization");
            int bits = weight.get("num_bits").getAsInt();
            require(bits == 4 || bits == 8, "Only NVFP4/FP8 weight groups are supported");
            JsonObject activation = group.getAsJsonObject("input_activations");
            if (bits == 4) {
                require(weight.get("group_size").getAsInt() == 16 && activation == null,
                        "NVFP4 must be W4A16 with 16-element blocks, not activation-FP4");
            } else {
                require(activation != null && "float".equals(activation.get("type").getAsString())
                                && activation.get("num_bits").getAsInt() == 8
                                && !activation.get("dynamic").getAsBoolean(),
                        "FP8 requires static FP8 input activation quantization");
            }
            for (JsonElement target : group.getAsJsonArray("targets")) {
                String name = target.getAsString();
                require(seen.add(name), "Duplicate quantization target " + name);
                JsonObject layer = quantizedLayers.getAsJsonObject(name);
                require(layer != null && (bits == 4 ? "W4A16_NVFP4" : "FP8")
                        .equals(layer.get("quant_algo").getAsString()), "Conflicting quantization group for " + name);
            }
        }
        Set<String> required = new HashSet<>();
        for (Map.Entry<String, JsonElement> layer : quantizedLayers.entrySet()) required.add(layer.getKey());
        require(seen.equals(required), "Quantization groups must cover quantized_layers exactly");
    }

    public static ModelOptQwenConfig read(File config, File hfQuantConfig) throws IOException {
        String name = config.getName();
        require(name.endsWith("config.json"), "Pass generation_config.json explicitly for nonstandard config names");
        File generation = new File(config.getAbsoluteFile().getParentFile(),
                name.substring(0, name.length() - "config.json".length()) + "generation_config.json");
        return read(config, hfQuantConfig, generation);
    }

    /** Explicit resources support revision-prefixed caches without guessing metadata contents. */
    public static ModelOptQwenConfig read(File config, File hfQuantConfig, File generationConfig) throws IOException {
        return new ModelOptQwenConfig(json(config), json(hfQuantConfig), json(generationConfig));
    }

    static JsonObject json(File file) throws IOException {
        try (Reader reader = Files.newBufferedReader(file.toPath(), StandardCharsets.UTF_8)) {
            return new Gson().fromJson(reader, JsonObject.class);
        }
    }

    static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException(message);
    }

    private static int positive(JsonObject object, String key) {
        int value = object.get(key).getAsInt();
        require(value > 0, "Expected positive " + key);
        return value;
    }

    public ArchitectureConfig getArchitecture() { return architecture; }
    public int getKeyHeads() { return keyHeads; }
    public int getValueHeads() { return valueHeads; }
    public int getKeyHeadDim() { return keyHeadDim; }
    public int getValueHeadDim() { return valueHeadDim; }
    public int getConvKernel() { return convKernel; }
    public int getEosTokenId() { return eosTokenId; }
    public int getBosTokenId() { return bosTokenId; }
    public int getPadTokenId() { return padTokenId; }
    public Set<Integer> getStopTokenIds() { return stopTokenIds; }
    public int getConvChannels() { return 2 * keyHeads * keyHeadDim + valueHeads * valueHeadDim; }
    JsonObject quantizedLayers() { return quantizedLayers; }
}
