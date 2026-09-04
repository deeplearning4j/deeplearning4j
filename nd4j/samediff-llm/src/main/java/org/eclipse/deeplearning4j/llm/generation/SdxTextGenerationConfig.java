/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.databind.node.ObjectNode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.TreeMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Derives the native SDX text-generation contract from the SameDiff graph that
 * will actually be packaged.
 *
 * <p>This belongs beside the graph importer and generation pipeline rather than
 * in an Android application. In particular, hybrid models such as Qwen3.5 must
 * emit the v2 recurrent-state contract; treating them as the older KV-only v1
 * profile fails before any mobile-specific accelerator behavior is reached.</p>
 */
public final class SdxTextGenerationConfig {

    public static final int KV_ONLY_FORMAT_VERSION = 1;
    public static final int RECURRENT_STATE_FORMAT_VERSION = 2;
    public static final String KV_ONLY_PROFILE = "causal-lm-in-graph-kv-v1";
    public static final String RECURRENT_STATE_PROFILE = "causal-lm-in-graph-state-v2";

    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Pattern KV_INPUT =
            Pattern.compile("^past_key_values\\.([0-9]+)\\.(key|value)$");
    private static final Pattern TRAILING_LAYER =
            Pattern.compile(".*(?:\\.|_)([0-9]+)(?:\\.(?:key|value))?$");

    private SdxTextGenerationConfig() {
    }

    public static ObjectNode derive(SameDiff graph, Options options) throws IOException {
        Objects.requireNonNull(graph, "graph");
        Objects.requireNonNull(options, "options");
        options.validate();

        ModelIOConfig ioConfig = ModelIOConfig.discover(graph);
        String inputIds = requireGraphVariable(graph, ioConfig.getInputIdsName(), "input IDs");
        String causalMask = requireGraphVariable(graph, ioConfig.getCausalMaskName(), "causal mask");
        String positionOffset =
                requireGraphVariable(graph, ioConfig.getPositionOffsetName(), "position offset");
        String cachePosition =
                requireGraphVariable(graph, ioConfig.getCachePositionName(), "cache position");
        String actualSequenceLength = findActualSequenceLength(graph);

        KvContract kv = deriveKvContract(graph, ioConfig);
        String kvDtype = deriveKvDtype(graph, kv);
        List<ModelIOConfig.RecurrentStatePair> recurrent =
                new ArrayList<>(ModelIOConfig.findRecurrentStatePairs(graph, ioConfig));
        recurrent.sort(Comparator.comparing(
                pair -> pair.inputName, SdxTextGenerationConfig::compareLayerNames));

        final boolean stateful = !recurrent.isEmpty();
        ObjectNode root = JSON.createObjectNode();
        root.put("formatVersion",
                stateful ? RECURRENT_STATE_FORMAT_VERSION : KV_ONLY_FORMAT_VERSION);
        root.put("profile", stateful ? RECURRENT_STATE_PROFILE : KV_ONLY_PROFILE);

        ObjectNode io = root.putObject("io");
        io.put("inputIds", inputIds);
        io.put("causalMask", causalMask);
        io.put("positionOffset", positionOffset);
        io.put("cachePosition", cachePosition);
        io.put("actualSequenceLength", actualSequenceLength);

        LogitsContract logits = deriveLogitsContract(graph, ioConfig);
        io.put("logits", logits.decode);
        if (!logits.prefill.equals(logits.decode)) {
            io.put("prefillLogits", logits.prefill);
        }
        putStrings(io, "kvKeyInputs", kv.keyInputs);
        putStrings(io, "kvValueInputs", kv.valueInputs);
        putKvShapeTemplates(graph, io, "kvKeyShapes", kv.keyInputs);
        putKvShapeTemplates(graph, io, "kvValueShapes", kv.valueInputs);
        putStrings(io, "prefillKeyOutputs", kv.keyOutputs);
        putStrings(io, "prefillValueOutputs", kv.valueOutputs);

        if (stateful) {
            ArrayNode states = io.putArray("recurrentStates");
            for (ModelIOConfig.RecurrentStatePair pair : recurrent) {
                ObjectNode state = states.addObject();
                state.put("input", requireGraphVariable(graph, pair.inputName, "recurrent input"));
                state.put("output", requireGraphOutput(graph, pair.outputName, "recurrent output"));
                if (pair.isGdn()) {
                    state.put("kind", "GDN");
                } else if (pair.isConv()) {
                    state.put("kind", "CONV");
                } else {
                    throw new IOException("Unsupported recurrent state op '" + pair.opType
                            + "' for " + pair.inputName);
                }
                state.put("dataType", sdxDtype(
                        graph.getVariable(pair.inputName), true, "recurrent state"));
                long[] shape = recurrentShape(graph, pair.inputName);
                ArrayNode shapeNode = state.putArray("shape");
                for (long dimension : shape) {
                    shapeNode.add(dimension);
                }
            }
        }

        ObjectNode execution = root.putObject("execution");
        execution.put("kvLayout", "BSHD");
        execution.put("kvDtype", kvDtype);
        execution.put("maskDtype", sdxDtype(
                graph.getVariable(causalMask), false, "causal mask"));
        execution.put("planOwnsKvScatter", true);

        ObjectNode tokens = root.putObject("tokens");
        if (options.bosId != null) {
            tokens.put("bosId", options.bosId);
        }
        tokens.put("padId", options.padId);
        ArrayNode eos = tokens.putArray("eosIds");
        for (Integer token : new LinkedHashSet<>(options.eosIds)) {
            eos.add(token);
        }

        ObjectNode limits = root.putObject("limits");
        limits.put("contextLength", options.contextLength);
        limits.put("maxPrefillLength", options.maxPrefillLength);
        limits.put("maxBatchSize", 1);

        ObjectNode sampling = root.putObject("samplingDefaults");
        sampling.put("maxNewTokens", options.maxNewTokens);
        sampling.put("minNewTokens", options.minNewTokens);
        sampling.put("temperature", options.temperature);
        sampling.put("topK", options.topK);
        sampling.put("topP", options.topP);
        sampling.put("repetitionPenalty", options.repetitionPenalty);
        sampling.put("seed", options.seed);
        return root;
    }

    public static Path write(SameDiff graph, Options options, Path output) throws IOException {
        Objects.requireNonNull(output, "output");
        JSON.writerWithDefaultPrettyPrinter().writeValue(output.toFile(), derive(graph, options));
        return output;
    }

    private static KvContract deriveKvContract(
            SameDiff graph, ModelIOConfig ioConfig) throws IOException {
        ModelIOConfig.KVCacheNames inputs = ModelIOConfig.findKVCacheInputNames(graph);
        if (inputs == null || inputs.keyNames.isEmpty() || inputs.valueNames.isEmpty()) {
            throw new IOException("SameDiff graph has no canonical past_key_values KV inputs");
        }

        Map<Integer, String> keys = indexKvInputs(inputs.keyNames, "key");
        Map<Integer, String> values = indexKvInputs(inputs.valueNames, "value");
        if (!keys.keySet().equals(values.keySet())) {
            throw new IOException("KV key/value inputs do not cover the same layers: keys="
                    + keys.keySet() + ", values=" + values.keySet());
        }

        Set<String> graphOutputs = new LinkedHashSet<>(graph.outputs());
        List<String> keyInputs = new ArrayList<>();
        List<String> valueInputs = new ArrayList<>();
        List<String> keyOutputs = new ArrayList<>();
        List<String> valueOutputs = new ArrayList<>();
        for (Integer layer : keys.keySet()) {
            String keyInput = keys.get(layer);
            String valueInput = values.get(layer);
            keyInputs.add(keyInput);
            valueInputs.add(valueInput);
            keyOutputs.add(resolveKvOutput(
                    graphOutputs, ioConfig.inputToPresentName(keyInput), "k_rope_" + layer,
                    "key", layer));
            valueOutputs.add(resolveKvOutput(
                    graphOutputs, ioConfig.inputToPresentName(valueInput), "v_heads_" + layer,
                    "value", layer));
        }
        return new KvContract(keyInputs, valueInputs, keyOutputs, valueOutputs);
    }

    private static String deriveKvDtype(SameDiff graph, KvContract kv) throws IOException {
        List<String> tensors = new ArrayList<>();
        tensors.addAll(kv.keyInputs);
        tensors.addAll(kv.valueInputs);
        tensors.addAll(kv.keyOutputs);
        tensors.addAll(kv.valueOutputs);

        String expectedDtype = null;
        String expectedTensor = null;
        for (String tensor : tensors) {
            String dtype = sdxDtype(
                    graph.getVariable(tensor), true, "KV tensor '" + tensor + "'");
            if (expectedDtype == null) {
                expectedDtype = dtype;
                expectedTensor = tensor;
            } else if (!expectedDtype.equals(dtype)) {
                throw new IOException("KV dtype contract mismatch: '" + expectedTensor + "' is "
                        + expectedDtype + " but '" + tensor + "' is " + dtype);
            }
        }
        if (expectedDtype == null) {
            throw new IOException("SameDiff graph has no KV tensors");
        }
        return expectedDtype;
    }

    private static Map<Integer, String> indexKvInputs(
            List<String> names, String expectedKind) throws IOException {
        Map<Integer, String> result = new TreeMap<>();
        for (String name : names) {
            Matcher matcher = KV_INPUT.matcher(name);
            if (!matcher.matches() || !expectedKind.equals(matcher.group(2))) {
                throw new IOException("Unsupported KV input binding: " + name);
            }
            int layer = Integer.parseInt(matcher.group(1));
            if (result.put(layer, name) != null) {
                throw new IOException("Duplicate KV " + expectedKind + " layer " + layer);
            }
        }
        return result;
    }

    private static String resolveKvOutput(
            Set<String> graphOutputs,
            String presentConvention,
            String ggufConvention,
            String kind,
            int layer) throws IOException {
        if (graphOutputs.contains(ggufConvention)) {
            return ggufConvention;
        }
        if (presentConvention != null && graphOutputs.contains(presentConvention)) {
            return presentConvention;
        }
        throw new IOException("No registered prefill " + kind + " output for KV layer "
                + layer + "; expected '" + ggufConvention + "' or '"
                + presentConvention + "'");
    }

    private static LogitsContract deriveLogitsContract(
            SameDiff graph, ModelIOConfig ioConfig) throws IOException {
        Set<String> outputs = new LinkedHashSet<>(graph.outputs());
        if (outputs.contains("lm_logits_last")) {
            // Native generation consumes only the final prompt/decode position. Prefer the
            // dedicated [B,1,V] projection even when the graph also exposes full [B,S,V]
            // logits, and support generation-only mobile graphs that omit the full branch.
            return new LogitsContract("lm_logits_last", "lm_logits_last");
        }
        String discovered = requireGraphOutput(
                graph, ioConfig.getLogitsOutputName(), "logits output");
        return new LogitsContract(discovered, discovered);
    }

    private static long[] recurrentShape(SameDiff graph, String inputName)
            throws IOException {
        long[] shape = GenerationPipeline.deriveRecurrentStateShape(graph, inputName);
        if (!positiveShape(shape)) {
            SDVariable input = graph.getVariable(inputName);
            shape = input == null ? null : input.getShape();
        }
        if (!positiveShape(shape)) {
            throw new IOException("Could not derive a concrete recurrent-state shape for "
                    + inputName);
        }
        return Arrays.copyOf(shape, shape.length);
    }

    private static void putKvShapeTemplates(
            SameDiff graph, ObjectNode io, String field, List<String> inputNames)
            throws IOException {
        ArrayNode shapes = io.putArray(field);
        for (String inputName : inputNames) {
            SDVariable variable = graph.getVariable(inputName);
            long[] shape = variable == null ? null : variable.getShape();
            if (shape == null || shape.length != 4 || shape[2] <= 0 || shape[3] <= 0) {
                throw new IOException("Could not derive a BSHD KV shape template for "
                        + inputName);
            }
            ArrayNode shapeNode = shapes.addArray();
            shapeNode.add(1L);
            shapeNode.add(-1L);
            shapeNode.add(shape[2]);
            shapeNode.add(shape[3]);
        }
    }

    private static boolean positiveShape(long[] shape) {
        if (shape == null || shape.length == 0) {
            return false;
        }
        for (long dimension : shape) {
            if (dimension <= 0) {
                return false;
            }
        }
        return true;
    }

    private static String findActualSequenceLength(SameDiff graph) throws IOException {
        for (String input : graph.inputs()) {
            String normalized = input.toLowerCase(Locale.ROOT);
            if (normalized.contains("actual_sequence_length")
                    || normalized.contains("actual_seq_len")
                    || normalized.equals("sequence_length")
                    || normalized.equals("seq_len")) {
                return requireGraphVariable(graph, input, "actual sequence length");
            }
        }
        throw new IOException("SameDiff graph is missing actual_sequence_length");
    }

    private static String requireGraphVariable(
            SameDiff graph, String name, String label) throws IOException {
        if (name == null || name.trim().isEmpty() || !graph.hasVariable(name)) {
            throw new IOException("SameDiff graph is missing " + label + " binding");
        }
        return name;
    }

    private static String requireGraphOutput(
            SameDiff graph, String name, String label) throws IOException {
        requireGraphVariable(graph, name, label);
        if (!graph.outputs().contains(name)) {
            throw new IOException("SameDiff " + label + " is not a registered graph output: "
                    + name);
        }
        return name;
    }

    private static String sdxDtype(
            SDVariable variable, boolean allowInt8, String label) throws IOException {
        if (variable == null || variable.dataType() == null) {
            throw new IOException("Unable to determine " + label + " data type");
        }
        switch (variable.dataType()) {
            case FLOAT:
                return "FLOAT32";
            case HALF:
                return "FLOAT16";
            case BFLOAT16:
                return "BFLOAT16";
            case BYTE:
                if (allowInt8) {
                    return "INT8";
                }
                break;
            default:
                break;
        }
        throw new IOException("Unsupported " + label + " data type: "
                + variable.dataType());
    }

    private static void putStrings(ObjectNode parent, String name, List<String> values) {
        ArrayNode array = parent.putArray(name);
        values.forEach(array::add);
    }

    private static int compareLayerNames(String left, String right) {
        Integer leftLayer = trailingLayer(left);
        Integer rightLayer = trailingLayer(right);
        if (leftLayer != null && rightLayer != null) {
            int comparison = Integer.compare(leftLayer, rightLayer);
            if (comparison != 0) {
                return comparison;
            }
        }
        return left.compareTo(right);
    }

    private static Integer trailingLayer(String name) {
        Matcher matcher = TRAILING_LAYER.matcher(name);
        return matcher.matches() ? Integer.valueOf(matcher.group(1)) : null;
    }

    private static final class KvContract {
        private final List<String> keyInputs;
        private final List<String> valueInputs;
        private final List<String> keyOutputs;
        private final List<String> valueOutputs;

        private KvContract(
                List<String> keyInputs,
                List<String> valueInputs,
                List<String> keyOutputs,
                List<String> valueOutputs) {
            this.keyInputs = keyInputs;
            this.valueInputs = valueInputs;
            this.keyOutputs = keyOutputs;
            this.valueOutputs = valueOutputs;
        }
    }

    private static final class LogitsContract {
        private final String prefill;
        private final String decode;

        private LogitsContract(String prefill, String decode) {
            this.prefill = prefill;
            this.decode = decode;
        }
    }

    public static final class Options {
        private final int contextLength;
        private final int maxPrefillLength;
        private final Integer bosId;
        private final int padId;
        private final List<Integer> eosIds;
        private final int maxNewTokens;
        private final int minNewTokens;
        private final double temperature;
        private final int topK;
        private final double topP;
        private final double repetitionPenalty;
        private final long seed;

        private Options(Builder builder) {
            contextLength = builder.contextLength;
            maxPrefillLength = builder.maxPrefillLength;
            bosId = builder.bosId;
            padId = builder.padId;
            eosIds = Collections.unmodifiableList(new ArrayList<>(builder.eosIds));
            maxNewTokens = builder.maxNewTokens;
            minNewTokens = builder.minNewTokens;
            temperature = builder.temperature;
            topK = builder.topK;
            topP = builder.topP;
            repetitionPenalty = builder.repetitionPenalty;
            seed = builder.seed;
        }

        public static Builder builder() {
            return new Builder();
        }

        private void validate() {
            if (contextLength < 2) {
                throw new IllegalArgumentException("contextLength must be at least 2");
            }
            if (maxPrefillLength < 1 || maxPrefillLength >= contextLength) {
                throw new IllegalArgumentException(
                        "maxPrefillLength must be in [1, contextLength)");
            }
            if (padId < 0 || eosIds.isEmpty() || eosIds.stream().anyMatch(id -> id == null || id < 0)) {
                throw new IllegalArgumentException("padId and eosIds must be non-negative");
            }
            if (maxNewTokens < 1 || maxNewTokens >= contextLength) {
                throw new IllegalArgumentException("maxNewTokens is outside the context envelope");
            }
            if (minNewTokens < 0 || minNewTokens > maxNewTokens) {
                throw new IllegalArgumentException("minNewTokens must be in [0, maxNewTokens]");
            }
            if (temperature < 0.0 || topK < 0 || topP < 0.0 || topP > 1.0
                    || repetitionPenalty <= 0.0) {
                throw new IllegalArgumentException("Invalid sampling defaults");
            }
        }

        public static final class Builder {
            private int contextLength;
            private int maxPrefillLength;
            private Integer bosId;
            private int padId = -1;
            private List<Integer> eosIds = Collections.emptyList();
            private int maxNewTokens = 128;
            private int minNewTokens;
            private double temperature;
            private int topK;
            private double topP = 1.0;
            private double repetitionPenalty = 1.0;
            private long seed;

            public Builder contextLength(int value) {
                contextLength = value;
                return this;
            }

            public Builder maxPrefillLength(int value) {
                maxPrefillLength = value;
                return this;
            }

            public Builder bosId(Integer value) {
                bosId = value;
                return this;
            }

            public Builder padId(int value) {
                padId = value;
                return this;
            }

            public Builder eosIds(List<Integer> value) {
                eosIds = Objects.requireNonNull(value, "eosIds");
                return this;
            }

            public Builder maxNewTokens(int value) {
                maxNewTokens = value;
                return this;
            }

            public Builder minNewTokens(int value) {
                minNewTokens = value;
                return this;
            }

            public Builder temperature(double value) {
                temperature = value;
                return this;
            }

            public Builder topK(int value) {
                topK = value;
                return this;
            }

            public Builder topP(double value) {
                topP = value;
                return this;
            }

            public Builder repetitionPenalty(double value) {
                repetitionPenalty = value;
                return this;
            }

            public Builder seed(long value) {
                seed = value;
                return this;
            }

            public Options build() {
                Options result = new Options(this);
                result.validate();
                return result;
            }
        }
    }
}
