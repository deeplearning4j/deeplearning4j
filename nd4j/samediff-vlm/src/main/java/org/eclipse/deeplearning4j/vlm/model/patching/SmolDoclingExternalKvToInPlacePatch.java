/*
 *  ******************************************************************************
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  ******************************************************************************
 */

package org.eclipse.deeplearning4j.vlm.model.patching;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ops.impl.transforms.custom.OnnxMultiHeadAttention;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Canonicalizes SmolDocling's exported external concat/repeat KV graph to the existing
 * {@code onnx_multi_head_attention} in-place cache contract.
 *
 * <p>The ONNX export concatenates every past/current K/V pair, repeats GQA heads, and publishes
 * sixty {@code present.N.key/value} graph outputs. That makes the physical attention length part
 * of every decode shape. This patch instead gives each MHA its canonical BHSD cache placeholders
 * plus one shared logical {@code cache_position}. The full cache capacity can then stay fixed while
 * the active prefix advances through EOS, allowing one padded DSP/Triton plan to replay.</p>
 *
 * <p>The rewrite is deliberately fail-closed: all 30 layers and both K/V paths are validated before
 * the first mutation. A graph that only partially resembles the supported SmolDocling export is
 * rejected rather than left with a mixture of external and in-place cache semantics.</p>
 */
@Slf4j
public final class SmolDoclingExternalKvToInPlacePatch implements SameDiffGraphPatch {

    public static final String ENABLED_PROPERTY = "vlm.smoldocling.inplace-kv.enabled";
    public static final String CACHE_POSITION = "cache_position";
    public static final String CAUSAL_MASK = "causal_mask";
    public static final String SCHEMA_VERSION = "smoldocling-inplace-kv-v2";

    private static final int LAYER_COUNT = 30;
    private static final Pattern PRESENT_OUTPUT = Pattern.compile("present\\.[0-9]+\\.(key|value)");

    @Override
    public String name() {
        return "smoldocling-external-kv-to-inplace";
    }

    @Override
    public String description() {
        return "Replace SmolDocling external concat/repeat KV with fixed-capacity BHSD caches and cache_position";
    }

    /** Apply the patch when enabled and when the graph is the supported SmolDocling decoder. */
    public static boolean applyIfEnabled(SameDiff graph) {
        String configured = System.getProperty(ENABLED_PROPERTY);
        if (configured != null && "false".equalsIgnoreCase(configured.trim())) {
            return false;
        }
        SmolDoclingExternalKvToInPlacePatch patch = new SmolDoclingExternalKvToInPlacePatch();
        if (patch.isAlreadyCanonical(graph)) {
            return false;
        }
        if (!patch.appliesTo(graph)) {
            return false;
        }
        if (!patch.apply(graph)) {
            throw new IllegalStateException("SmolDocling in-place KV rewrite matched but was not applied");
        }
        return true;
    }

    @Override
    public boolean appliesTo(SameDiff graph) {
        return graph != null
                && graph.getVariable("inputs_embeds") != null
                && graph.getVariable("past_key_values.0.key") != null
                && graph.getVariable("present.0.key") != null
                && countMhaOps(graph) == LAYER_COUNT;
    }

    @Override
    public boolean apply(SameDiff graph) {
        List<LayerRewrite> rewrites = discoverAndValidate(graph);

        SDVariable cachePosition = graph.getVariable(CACHE_POSITION);
        if (cachePosition != null) {
            require(cachePosition.getVariableType() == VariableType.PLACEHOLDER,
                    "cache_position exists but is not a placeholder");
            require(cachePosition.dataType() == DataType.INT64,
                    "cache_position must be INT64, got " + cachePosition.dataType());
            require(cachePosition.getShape() != null && cachePosition.getShape().length == 1
                            && cachePosition.getShape()[0] == 1,
                    "cache_position must have shape [1]");
        }

        SDVariable causalMask = graph.getVariable(CAUSAL_MASK);
        if (causalMask != null) {
            require(causalMask.getVariableType() == VariableType.PLACEHOLDER,
                    "causal_mask exists but is not a placeholder");
            require(causalMask.dataType().isFPType(),
                    "causal_mask must be floating point, got " + causalMask.dataType());
            require(causalMask.getShape() != null && causalMask.getShape().length == 4,
                    "causal_mask must be rank 4");
        }

        if (cachePosition == null) cachePosition = graph.placeHolder(CACHE_POSITION, DataType.INT64, 1);
        if (causalMask == null) causalMask = graph.placeHolder(CAUSAL_MASK, DataType.FLOAT, 1, 1, -1, -1);

        for (LayerRewrite rewrite : rewrites) {
            DifferentialFunction function = rewrite.op.getOp();
            graph.replaceArgsFor(function,
                    rewrite.query, rewrite.currentKey, rewrite.currentValue, causalMask,
                    rewrite.pastKey, rewrite.pastValue, cachePosition);

            if (function instanceof OnnxMultiHeadAttention) {
                Map<String, Object> properties = new LinkedHashMap<>(function.propertiesForFunction());
                properties.put("numOutputs", 1);
                function.setPropertiesForFunction(properties);
            }
        }

        List<String> outputs = new ArrayList<>(graph.outputs());
        outputs.removeIf(name -> PRESENT_OUTPUT.matcher(name).matches());
        graph.setOutputs(outputs);

        log.info("Applied {} to {} MHA layers; fixed cache buffers now advance through shared {}",
                SCHEMA_VERSION, rewrites.size(), CACHE_POSITION);
        return true;
    }

    private List<LayerRewrite> discoverAndValidate(SameDiff graph) {
        List<LayerRewrite> rewrites = new ArrayList<>(LAYER_COUNT);
        Set<String> seenMhaOps = new HashSet<>();

        for (int layer = 0; layer < LAYER_COUNT; layer++) {
            String pastKeyName = "past_key_values." + layer + ".key";
            String pastValueName = "past_key_values." + layer + ".value";
            String presentKeyName = "present." + layer + ".key";
            String presentValueName = "present." + layer + ".value";
            String currentKeyName = "/model/layers." + layer
                    + "/attn/k_rotary/RotaryEmbedding/output_0";
            String currentValueName = "/model/layers." + layer
                    + "/attn/v_proj/MatMul/output_0";

            SDVariable pastKey = requireVariable(graph, pastKeyName);
            SDVariable pastValue = requireVariable(graph, pastValueName);
            SDVariable currentKey = requireVariable(graph, currentKeyName);
            SDVariable currentValue = requireVariable(graph, currentValueName);
            require(pastKey.getVariableType() == VariableType.PLACEHOLDER,
                    pastKeyName + " must be a placeholder");
            require(pastValue.getVariableType() == VariableType.PLACEHOLDER,
                    pastValueName + " must be a placeholder");
            require(currentKey.dataType() == pastKey.dataType(),
                    "Layer " + layer + " K/cache dtype mismatch: "
                            + currentKey.dataType() + " vs " + pastKey.dataType());
            require(currentValue.dataType() == pastValue.dataType(),
                    "Layer " + layer + " V/cache dtype mismatch: "
                            + currentValue.dataType() + " vs " + pastValue.dataType());

            SameDiffOp presentKeyConcat = requireProducer(graph, presentKeyName, "concat");
            SameDiffOp presentValueConcat = requireProducer(graph, presentValueName, "concat");
            validatePresentConcat(graph, presentKeyConcat, pastKeyName, currentKeyName, presentKeyName);
            validatePresentConcat(graph, presentValueConcat, pastValueName, currentValueName, presentValueName);

            SameDiffOp mha = findLayerMha(graph, layer);
            require(seenMhaOps.add(mha.getName()), "MHA op reused by multiple layers: " + mha.getName());
            List<String> inputs = mha.getInputsToOp();
            require(inputs != null && inputs.size() == 6,
                    "Layer " + layer + " MHA must have six external-KV inputs, got "
                            + (inputs == null ? 0 : inputs.size()));
            require(mha.getOutputsOfOp() != null && mha.getOutputsOfOp().size() == 1,
                    "Layer " + layer + " MHA must have exactly one output");
            require(dependsOn(graph, inputs.get(1), presentKeyName, 10, new HashSet<>()),
                    "Layer " + layer + " MHA key does not depend on " + presentKeyName);
            require(dependsOn(graph, inputs.get(2), presentValueName, 10, new HashSet<>()),
                    "Layer " + layer + " MHA value does not depend on " + presentValueName);

            rewrites.add(new LayerRewrite(mha,
                    requireVariable(graph, inputs.get(0)), currentKey, currentValue,
                    pastKey, pastValue));
        }

        require(rewrites.size() == LAYER_COUNT,
                "Expected " + LAYER_COUNT + " SmolDocling MHA layers, found " + rewrites.size());
        return rewrites;
    }

    private static void validatePresentConcat(SameDiff graph, SameDiffOp concat, String pastName,
                                              String currentName, String presentName) {
        List<String> inputs = concat.getInputsToOp();
        require(inputs != null && inputs.size() == 2,
                presentName + " must be produced by a two-input concat");
        require(inputs.get(0).equals(pastName),
                presentName + " concat must consume " + pastName + " first, got " + inputs);
        require(dependsOn(graph, inputs.get(1), currentName, 4, new HashSet<>()),
                presentName + " current branch does not depend on " + currentName);
    }

    private static boolean dependsOn(SameDiff graph, String variableName, String target,
                                     int remainingDepth, Set<String> visited) {
        if (variableName.equals(target)) {
            return true;
        }
        if (remainingDepth <= 0 || !visited.add(variableName)) {
            return false;
        }
        org.nd4j.autodiff.samediff.internal.Variable variable = graph.getVariables().get(variableName);
        if (variable == null || variable.getOutputOfOp() == null) {
            return false;
        }
        SameDiffOp producer = graph.getOps().get(variable.getOutputOfOp());
        if (producer == null || producer.getInputsToOp() == null) {
            return false;
        }
        for (String input : producer.getInputsToOp()) {
            if (dependsOn(graph, input, target, remainingDepth - 1, visited)) {
                return true;
            }
        }
        return false;
    }

    private static SameDiffOp requireProducer(SameDiff graph, String variableName, String opName) {
        org.nd4j.autodiff.samediff.internal.Variable variable = graph.getVariables().get(variableName);
        require(variable != null && variable.getOutputOfOp() != null,
                variableName + " has no producer");
        SameDiffOp producer = graph.getOps().get(variable.getOutputOfOp());
        require(producer != null && producer.getOp() != null
                        && opName.equals(producer.getOp().opName()),
                variableName + " must be produced by " + opName);
        return producer;
    }

    private static SameDiffOp findLayerMha(SameDiff graph, int layer) {
        String marker = "layers." + layer + "/";
        SameDiffOp found = null;
        for (SameDiffOp candidate : graph.getOps().values()) {
            if (candidate.getOp() == null
                    || !"onnx_multi_head_attention".equals(candidate.getOp().opName())
                    || candidate.getName() == null || !candidate.getName().contains(marker)) {
                continue;
            }
            require(found == null, "Multiple MHA ops found for layer " + layer);
            found = candidate;
        }
        require(found != null, "No MHA op found for layer " + layer);
        return found;
    }

    private static SDVariable requireVariable(SameDiff graph, String name) {
        SDVariable variable = graph.getVariable(name);
        require(variable != null, "Required SmolDocling variable is missing: " + name);
        return variable;
    }

    private boolean isAlreadyCanonical(SameDiff graph) {
        if (graph == null || graph.getVariable(CACHE_POSITION) == null || countMhaOps(graph) != LAYER_COUNT) {
            return false;
        }
        for (SameDiffOp op : graph.getOps().values()) {
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())) {
                if (op.getInputsToOp() == null || op.getInputsToOp().size() != 7
                        || !CAUSAL_MASK.equals(op.getInputsToOp().get(3))
                        || !CACHE_POSITION.equals(op.getInputsToOp().get(6))) {
                    return false;
                }
            }
        }
        return graph.outputs().stream().noneMatch(name -> PRESENT_OUTPUT.matcher(name).matches());
    }

    private static int countMhaOps(SameDiff graph) {
        int count = 0;
        for (SameDiffOp op : graph.getOps().values()) {
            if (op.getOp() != null && "onnx_multi_head_attention".equals(op.getOp().opName())) {
                count++;
            }
        }
        return count;
    }

    private static void require(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static final class LayerRewrite {
        private final SameDiffOp op;
        private final SDVariable query;
        private final SDVariable currentKey;
        private final SDVariable currentValue;
        private final SDVariable pastKey;
        private final SDVariable pastValue;

        private LayerRewrite(SameDiffOp op, SDVariable query, SDVariable currentKey,
                             SDVariable currentValue, SDVariable pastKey, SDVariable pastValue) {
            this.op = op;
            this.query = query;
            this.currentKey = currentKey;
            this.currentValue = currentValue;
            this.pastKey = pastKey;
            this.pastValue = pastValue;
        }
    }
}
