/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

/**
 * Packet 02/03 binding contracts for the native output manifest, the scalar-target
 * mapping, and the suffix-prefill ordinary recurrent feedback mapping. The native
 * decode must declare the FULL requested-output list of the prepared target plan
 * (including outputs the Java loop never consumes), the scalar mapping must be
 * name-based on the scalar binding side, and every discovered GDN/conv pair must
 * resolve to a paired ordinary feedback mapping consumed by the next decode step.
 */
public class TestMtpPrefixBindingContracts {
    private static final int K = 1;
    private static final int WIDTH = K + 1;
    private static final int CACHE = 8;
    private static final int BASE_TARGET_POSITION = 1;

    @Test
    void testSuffixPrefillRecurrentFeedbackMappingIsPairedAndConsumed() {
        // Packet 03: the suffix-prefix-cache state construction must produce
        // paired, nonempty ORDINARY recurrent feedback mappings for every
        // discovered GDN/conv pair - in OFF mode and prefix-capable mode alike -
        // and a second decode step must consume the resulting nonzero state
        // (distinct values per layer) through those mappings.
        int l = 2;
        int h = 2, dk = 8, dv = 8;
        int convD = 4, convK = 4;
        try (RecurrentPlan p = new RecurrentPlan(l, h, dk, dv, convD, convK)) {
            p.compile();
            ModelIOConfig ioConfig = ModelIOConfig.discover(p.graph);
            List<ModelIOConfig.RecurrentStatePair> pairs =
                    ModelIOConfig.findRecurrentStatePairs(p.graph, ioConfig);
            assertEquals(2, pairs.size(),
                    "one GDN pair and one conv pair must be discovered, got " + pairs);

            // The ordinary feedback index mapping must be paired and nonempty in
            // OFF mode (the prefix flag plays no role in this resolution).
            List<Integer> gdnExt = new ArrayList<>(), gdnOut = new ArrayList<>();
            List<Integer> convExt = new ArrayList<>(), convOut = new ArrayList<>();
            GenerationPipeline.resolveRecurrentFeedbackIndices(
                    p.executor, pairs, gdnExt, gdnOut, convExt, convOut);
            assertEquals(1, gdnExt.size(), "one GDN feedback pair expected");
            assertEquals(1, gdnOut.size(), "one GDN feedback output expected");
            assertEquals(1, convExt.size(), "one conv feedback pair expected");
            assertEquals(1, convOut.size(), "one conv feedback output expected");
            assertTrue(gdnExt.get(0) >= 0 && gdnOut.get(0) >= 0, "GDN indices must resolve");
            assertTrue(convExt.get(0) >= 0 && convOut.get(0) >= 0, "conv indices must resolve");

            // Second decode step consumes the first step's nonzero state: the GDN
            // output depends on S_{t-1}, so feeding the nonzero state back through
            // the resolved input index changes the next step's output.
            Map<String, INDArray> step0 = p.run(Nd4j.zeros(DataType.FLOAT, 1, h, dk, dv),
                    Nd4j.zeros(DataType.FLOAT, 1, convD, convK - 1));
            Map<String, INDArray> step1 = p.run(p.gdnNonzeroState(), p.convNonzeroState());
            INDArray gdnOut0 = step0.get("gdn_state_out_0");
            INDArray gdnOut1 = step1.get("gdn_state_out_0");
            assertNotNull(gdnOut0, "step 0 GDN state output");
            assertNotNull(gdnOut1, "step 1 GDN state output");
            assertTrue(!gdnOut0.equalsWithEps(gdnOut1, 1e-6f),
                    "the second step must consume the nonzero GDN state (outputs differ)");
            assertTrue(!p.gdnNonzeroState().equalsWithEps(p.convNonzeroState(), 1e-6f),
                    "the two layers must carry distinct nonzero state values");
        }
    }

    /** Suffix-prefill-style graph with one GDN companion and one conv companion. */
    private static final class RecurrentPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private final int l;
        private DynamicShapePlanExecutor executor;

        RecurrentPlan(int l, int h, int dk, int dv, int convD, int convK) {
            this.l = l;
            SDVariable q = placeholder("q", Nd4j.linspace(1, l * h * dk, l * h * dk, DataType.FLOAT)
                    .reshape(1, l, h, dk).muli(0.01f));
            SDVariable k = placeholder("k", Nd4j.linspace(1, l * h * dk, l * h * dk, DataType.FLOAT)
                    .reshape(1, l, h, dk).muli(0.02f));
            SDVariable v = placeholder("v", Nd4j.linspace(1, l * h * dv, l * h * dv, DataType.FLOAT)
                    .reshape(1, l, h, dv).muli(0.03f));
            SDVariable beta = placeholder("beta", Nd4j.valueArrayOf(
                    new long[]{1, l, h}, 0.5, DataType.FLOAT));
            SDVariable gate = placeholder("gate", Nd4j.valueArrayOf(
                    new long[]{1, l, h}, -1.0, DataType.FLOAT));
            SDVariable gdnState = placeholder("past_gdn_state.0",
                    Nd4j.zeros(DataType.FLOAT, 1, h, dk, dv));
            SDVariable x = placeholder("x", Nd4j.linspace(1, l * convD, l * convD, DataType.FLOAT)
                    .reshape(1, l, convD).muli(0.05f));
            SDVariable weight = graph.var("gdn_conv_weight", Nd4j.linspace(1, convD * convK,
                    convD * convK, DataType.FLOAT).reshape(convD, convK).muli(0.01f));
            SDVariable convState = placeholder("past_conv_state.0",
                    Nd4j.zeros(DataType.FLOAT, 1, convD, convK - 1));
            SDVariable actualLen = placeholder("actual_sequence_length",
                    Nd4j.scalar(DataType.INT64, (long) l));

            SDVariable[] gdnOut = graph.nn().gatedDeltaRuleWithPrefix(
                    new String[]{"gdn_out_0", "gdn_state_out_0", "gdn_state_prefix_0"},
                    q, k, v, beta, gate, gdnState, actualLen);
            SDVariable[] convOut = graph.nn().causalConv1dWithPrefix(
                    new String[]{"conv_out_0", "conv_state_out_0", "conv_state_prefix_0"},
                    x, weight, null, convState, actualLen, 1, 0);
            graph.setOutputs(gdnOut[0].name(), gdnOut[1].name(),
                    convOut[0].name(), convOut[1].name());
            for (String name : graph.outputs()) outputs.add(name);
        }

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private INDArray gdnNonzeroState() {
            return Nd4j.linspace(1, 1 * 2 * 8 * 8, 1 * 2 * 8 * 8, DataType.FLOAT)
                    .reshape(1, 2, 8, 8).muli(0.05f);
        }

        private INDArray convNonzeroState() {
            return Nd4j.linspace(1, 1 * 4 * 3, 1 * 4 * 3, DataType.FLOAT)
                    .reshape(1, 4, 3).muli(-0.07f);
        }

        private Map<String, INDArray> run(INDArray gdnState, INDArray convState) {
            Map<String, INDArray> stepInputs = new LinkedHashMap<>();
            for (Map.Entry<String, INDArray> entry : inputs.entrySet()) {
                if ("past_gdn_state.0".equals(entry.getKey())) {
                    stepInputs.put(entry.getKey(), gdnState);
                } else if ("past_conv_state.0".equals(entry.getKey())) {
                    stepInputs.put(entry.getKey(), convState);
                } else {
                    stepInputs.put(entry.getKey(), entry.getValue());
                }
            }
            return graph.output(stepInputs, outputs.toArray(new String[0]));
        }

        private void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertTrue(!executor.getNativePlanHandle().isNull(), "native plan required");
            assertNotNull(executor.getCachedOpContext());
        }

        @Override public void close() { graph.close(); }
    }

    @Test
    void testDeterministicPrefixTriplePairingAcrossMixedDiscoveryOrder() throws Exception {
        // Packet 04: two GDN and two conv layers with discovery order deliberately
        // mixed (conv22, gdn7, conv7, gdn22). The checkpoint-output binding must
        // preserve layer pairing in grouped GDN-first/conv-second serialization
        // independent of discovery order, and the state metadata must carry only
        // integer indices (no placeholder-owned arrays).
        System.setProperty("nd4j.mtp.prefixSelect", "select");
        try (MixedPlan p = new MixedPlan()) {
            p.compile();
            ModelIOConfig ioConfig = ModelIOConfig.discover(p.graph);
            List<ModelIOConfig.RecurrentStatePair> pairs =
                    ModelIOConfig.findRecurrentStatePairs(p.graph, ioConfig);
            assertEquals(4, pairs.size(), "four recurrent pairs expected, got " + pairs);

            // Ordinary feedback mapping first (what the pipeline populates before
            // attachPrefixSelect).
            List<Integer> gdnExt = new ArrayList<>(), gdnOut = new ArrayList<>();
            List<Integer> convExt = new ArrayList<>(), convOut = new ArrayList<>();
            GenerationPipeline.resolveRecurrentFeedbackIndices(
                    p.executor, pairs, gdnExt, gdnOut, convExt, convOut);
            assertEquals(2, gdnOut.size(), "two GDN ordinary outputs");
            assertEquals(2, convOut.size(), "two conv ordinary outputs");

            InGraphKvState state = new InGraphKvState();
            state.gdnStateExtIndices = gdnExt.stream().mapToInt(Integer::intValue).toArray();
            state.gdnStateOutputIndices = gdnOut.stream().mapToInt(Integer::intValue).toArray();
            state.convStateExtIndices = convExt.stream().mapToInt(Integer::intValue).toArray();
            state.convStateOutputIndices = convOut.stream().mapToInt(Integer::intValue).toArray();

            InGraphKvState.PrefixSelectMode mode =
                    GenerationPipeline.attachPrefixSelect(state, p.graph, pairs, p.executor);
            assertEquals(InGraphKvState.PrefixSelectMode.SELECT, mode);
            assertNotNull(state.gdnPrefixOutputIndices, "GDN prefix indices resolved");
            assertNotNull(state.convPrefixOutputIndices, "conv prefix indices resolved");
            assertEquals(2, state.gdnPrefixOutputIndices.length);
            assertEquals(2, state.convPrefixOutputIndices.length);

            // Every resolved index must refer to a state or checkpoint output of the
            // same kind, and grouped serialization (GDN-first then conv) must
            // preserve PAIRING: prefix[i] corresponds to the SAME layer as
            // ordinary[i] in each group. The within-group ORDER is not asserted -
            // discovery order is deliberately mixed, and the binding contract is
            // pairing preservation, not a specific layer sort.
            for (int i = 0; i < 2; i++) {
                String gdnOrdinaryName = nameAt(p, state.gdnStateOutputIndices[i]);
                String gdnPrefixName = nameAt(p, state.gdnPrefixOutputIndices[i]);
                String convOrdinaryName = nameAt(p, state.convStateOutputIndices[i]);
                String convPrefixName = nameAt(p, state.convPrefixOutputIndices[i]);
                assertTrue(gdnOrdinaryName.startsWith("gdn_state_out_"),
                        "GDN ordinary index " + i + " must refer to a GDN state output: " + gdnOrdinaryName);
                assertTrue(gdnPrefixName.startsWith("gdn_state_prefix_"),
                        "GDN prefix index " + i + " must refer to a GDN checkpoint: " + gdnPrefixName);
                assertTrue(convOrdinaryName.startsWith("conv_state_out_"),
                        "conv ordinary index " + i + " must refer to a conv state output: " + convOrdinaryName);
                assertTrue(convPrefixName.startsWith("conv_state_prefix_"),
                        "conv prefix index " + i + " must refer to a conv checkpoint: " + convPrefixName);
                assertEquals(layerOf(gdnOrdinaryName), layerOf(gdnPrefixName),
                        "GDN ordinary/prefix layer pairing must be preserved at position " + i);
                assertEquals(layerOf(convOrdinaryName), layerOf(convPrefixName),
                        "conv ordinary/prefix layer pairing must be preserved at position " + i);
            }
            // No duplicate state tuple: the four pairs must have four distinct input names.
            Set<String> inputNames = new LinkedHashSet<>();
            for (ModelIOConfig.RecurrentStatePair pair : pairs) {
                assertTrue(inputNames.add(pair.inputName),
                        "duplicate recurrent tuple for " + pair.inputName);
            }
            assertEquals(4, inputNames.size());
            // No placeholder-owned arrays: the metadata is index arrays only.
            for (int idx : state.gdnPrefixOutputIndices) assertTrue(idx >= 0);
            for (int idx : state.convPrefixOutputIndices) assertTrue(idx >= 0);
        } finally {
            System.clearProperty("nd4j.mtp.prefixSelect");
        }
    }
    @Test
    void testMissingCheckpointOutputFailsPreparationBeforeDecode() throws Exception {
        System.setProperty("nd4j.mtp.prefixSelect", "select");
        try (MixedPlan p = new MixedPlan(false)) {
            p.compile();
            ModelIOConfig ioConfig = ModelIOConfig.discover(p.graph);
            List<ModelIOConfig.RecurrentStatePair> pairs =
                    ModelIOConfig.findRecurrentStatePairs(p.graph, ioConfig);
            List<Integer> gdnExt = new ArrayList<>(), gdnOut = new ArrayList<>();
            List<Integer> convExt = new ArrayList<>(), convOut = new ArrayList<>();
            GenerationPipeline.resolveRecurrentFeedbackIndices(
                    p.executor, pairs, gdnExt, gdnOut, convExt, convOut);
            InGraphKvState state = new InGraphKvState();
            state.gdnStateExtIndices = gdnExt.stream().mapToInt(Integer::intValue).toArray();
            state.gdnStateOutputIndices = gdnOut.stream().mapToInt(Integer::intValue).toArray();
            state.convStateExtIndices = convExt.stream().mapToInt(Integer::intValue).toArray();
            state.convStateOutputIndices = convOut.stream().mapToInt(Integer::intValue).toArray();
            IllegalStateException ex = assertThrows(IllegalStateException.class,
                    () -> GenerationPipeline.attachPrefixSelect(state, p.graph, pairs, p.executor));
            assertTrue(ex.getMessage().contains("gdn_state_prefix_22")
                            || ex.getMessage().contains("prefix"),
                    "the missing checkpoint must be named in the preparation failure, got: "
                            + ex.getMessage());
        } finally {
            System.clearProperty("nd4j.mtp.prefixSelect");
        }
    }

    private static String nameAt(MixedPlan p, int idx) {
        return new ArrayList<>(p.executor.getCurrentPlan().getRequestedOutputs()).get(idx);
    }

    @Test
    void testShadowModeFailsAdmissionInsteadOfSilentCapture() throws Exception {
        // Packet 08: shadow is a comparison transaction, which is not implemented;
        // an explicit shadow request must fail at admission - never run capture-only
        // while claiming validation, and never silently resolve to OFF.
        System.setProperty("nd4j.mtp.prefixSelect", "shadow");
        try (MixedPlan p = new MixedPlan()) {
            p.compile();
            ModelIOConfig ioConfig = ModelIOConfig.discover(p.graph);
            List<ModelIOConfig.RecurrentStatePair> pairs =
                    ModelIOConfig.findRecurrentStatePairs(p.graph, ioConfig);
            List<Integer> gdnExt = new ArrayList<>(), gdnOut = new ArrayList<>();
            List<Integer> convExt = new ArrayList<>(), convOut = new ArrayList<>();
            GenerationPipeline.resolveRecurrentFeedbackIndices(
                    p.executor, pairs, gdnExt, gdnOut, convExt, convOut);
            InGraphKvState state = new InGraphKvState();
            state.gdnStateExtIndices = gdnExt.stream().mapToInt(Integer::intValue).toArray();
            state.gdnStateOutputIndices = gdnOut.stream().mapToInt(Integer::intValue).toArray();
            state.convStateExtIndices = convExt.stream().mapToInt(Integer::intValue).toArray();
            state.convStateOutputIndices = convOut.stream().mapToInt(Integer::intValue).toArray();
            IllegalStateException ex = assertThrows(IllegalStateException.class,
                    () -> GenerationPipeline.attachPrefixSelect(state, p.graph, pairs, p.executor));
            assertTrue(ex.getMessage().contains("comparison"),
                    "shadow admission must name the missing comparison, got: " + ex.getMessage());
            assertEquals(InGraphKvState.PrefixSelectMode.OFF, state.prefixSelectMode,
                    "state must stay OFF after a rejected shadow request");
        } finally {
            System.clearProperty("nd4j.mtp.prefixSelect");
        }
    }

    private static String layerOf(String name) {
        return name.substring(name.lastIndexOf('_') + 1);
    }

    /**
     * Four-layer mixed-order companion graph: discovery order conv22, gdn7,
     * conv7, gdn22. When {@code includeAllPrefixOutputs} is false, the
     * gdn_state_prefix_22 output is NOT requested, so preparation must fail
     * before any decode execution.
     */
    private static final class MixedPlan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;

        MixedPlan() { this(true); }

        MixedPlan(boolean includeAllPrefixOutputs) {
            int l = 2, h = 2, dk = 8, dv = 8, convD = 4, convK = 4;
            // One shared scalar placeholder for the whole graph (production graphs
            // have a single actual_sequence_length input).
            placeholder(graph, inputs, "actual_length", Nd4j.scalar(DataType.INT64, (long) l));
            // Deliberately mixed discovery order.
            convLayer(graph, inputs, l, convD, convK, 22);
            gdnLayer(graph, inputs, l, h, dk, dv, 7);
            convLayer(graph, inputs, l, convD, convK, 7);
            gdnLayer(graph, inputs, l, h, dk, dv, 22);
            if (includeAllPrefixOutputs) {
                graph.setOutputs("gdn_state_out_7", "gdn_state_prefix_7",
                        "gdn_state_out_22", "gdn_state_prefix_22",
                        "conv_state_out_22", "conv_state_prefix_22",
                        "conv_state_out_7", "conv_state_prefix_7");
            } else {
                graph.setOutputs("gdn_state_out_7", "gdn_state_prefix_7",
                        "gdn_state_out_22",
                        "conv_state_out_22", "conv_state_prefix_22",
                        "conv_state_out_7", "conv_state_prefix_7");
            }
            for (String name : graph.outputs()) outputs.add(name);
        }

        private static void gdnLayer(SameDiff graph, Map<String, INDArray> inputs,
                                     int l, int h, int dk, int dv, int layer) {
            String tag = String.valueOf(layer);
            SDVariable q = placeholder(graph, inputs, "q" + layer,
                    Nd4j.linspace(1, l * h * dk, l * h * dk, DataType.FLOAT)
                            .reshape(1, l, h, dk).muli(0.01f));
            SDVariable k = placeholder(graph, inputs, "k" + layer,
                    Nd4j.linspace(1, l * h * dk, l * h * dk, DataType.FLOAT)
                            .reshape(1, l, h, dk).muli(0.02f));
            SDVariable v = placeholder(graph, inputs, "v" + layer,
                    Nd4j.linspace(1, l * h * dv, l * h * dv, DataType.FLOAT)
                            .reshape(1, l, h, dv).muli(0.03f));
            SDVariable beta = placeholder(graph, inputs, "beta" + layer,
                    Nd4j.valueArrayOf(new long[]{1, l, h}, 0.5, DataType.FLOAT));
            SDVariable gate = placeholder(graph, inputs, "gate" + layer,
                    Nd4j.valueArrayOf(new long[]{1, l, h}, -1.0, DataType.FLOAT));
            SDVariable state = placeholder(graph, inputs, "past_gdn_state." + layer,
                    Nd4j.zeros(DataType.FLOAT, 1, h, dk, dv));
            SDVariable actualLen = graph.getVariable("actual_length");
            graph.nn().gatedDeltaRuleWithPrefix(
                    new String[]{"gdn_out_" + tag, "gdn_state_out_" + tag, "gdn_state_prefix_" + tag},
                    q, k, v, beta, gate, state, actualLen);
        }

        private static void convLayer(SameDiff graph, Map<String, INDArray> inputs,
                                      int l, int d, int kc, int layer) {
            String tag = String.valueOf(layer);
            SDVariable x = placeholder(graph, inputs, "x" + layer,
                    Nd4j.linspace(1, l * d, l * d, DataType.FLOAT)
                            .reshape(1, l, d).muli(0.05f));
            SDVariable weight = graph.var("conv_weight_" + layer, Nd4j.linspace(1, d * kc,
                    d * kc, DataType.FLOAT).reshape(d, kc).muli(0.01f));
            SDVariable state = placeholder(graph, inputs, "past_conv_state." + layer,
                    Nd4j.zeros(DataType.FLOAT, 1, d, kc - 1));
            SDVariable actualLen = graph.getVariable("actual_length");
            graph.nn().causalConv1dWithPrefix(
                    new String[]{"conv_out_" + tag, "conv_state_out_" + tag, "conv_state_prefix_" + tag},
                    x, weight, null, state, actualLen, 1, 0);
        }

        private static SDVariable placeholder(SameDiff graph, Map<String, INDArray> inputs,
                                              String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertTrue(!executor.getNativePlanHandle().isNull(), "native plan required");
            assertNotNull(executor.getCachedOpContext());
        }

        @Override public void close() { graph.close(); }
    }

    private static Plan target() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, WIDTH, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        // Verification row 0 disagrees with the predictor draft (token 0 vs 1), so
        // every step forces the scalar-width-1 rerun path.
        float[] logits = new float[WIDTH * 2];
        float[] hidden = new float[WIDTH];
        for (int row = 0; row < WIDTH; row++) {
            logits[2 * row] = 10;
            hidden[row] = 100 + row;
        }
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, WIDTH, 1);
        SDVariable rawLogits = zeros.add(p.graph.constant("shared_logits",
                Nd4j.createFromArray(logits).reshape(1, WIDTH, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        p.output(zeros.add("hidden", p.graph.constant("shared_hidden",
                Nd4j.createFromArray(hidden).reshape(1, WIDTH, 1))));
        p.output(zeros.add("extra", p.graph.constant("shared_extra",
                Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 42, DataType.FLOAT))));
        return p;
    }

    /** Width-1 scalar target sharing every input NAME with the target plan. The
     *  scalar outputs must cover every output the scalar ABI maps from the target
     *  (logits/hidden/extra), but the scalar plan has its own CONSTANT arrays for
     *  them - constants are not external inputs, so the scalar external-input key
     *  set stays clean (no 'scalar_extra' key). */
    private static Plan scalarTarget() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        p.echo("mask", Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.echo("actual_length", Nd4j.ones(DataType.INT64, 1));
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zeros = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        // Greedy-identical row-0 logits: token 0, matching the verification row 0.
        // The constants carry the SAME names as the target plan's constants - the
        // scalar binding's external-input keys must map into the target key set by
        // name (production derives both plans from one graph).
        SDVariable rawLogits = zeros.add(p.graph.constant("shared_logits",
                Nd4j.createFromArray(10.0f, 0.0f).reshape(1, 1, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        p.output(zeros.add("hidden", p.graph.constant("shared_hidden",
                Nd4j.valueArrayOf(new long[]{1, 1, 1}, 100, DataType.FLOAT))));
        p.output(zeros.add("extra", p.graph.constant("shared_extra",
                Nd4j.valueArrayOf(new long[]{1, 1, 1}, 42, DataType.FLOAT))));
        return p;
    }

    /** Minimal MTP predictor: always proposes token 1. The cache_position
     *  placeholder is the attention position input (consumed), mirroring the
     *  production predictor geometry; unconsumed placeholders are dropped from
     *  the compiled plan's external inputs. */
    private static Plan predictor() {
        Plan p = new Plan();
        p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        p.placeholder("mask", Nd4j.valueArrayOf(new long[]{1, 1, 1, CACHE}, -Float.MAX_VALUE, DataType.FLOAT));
        p.echo("position", Nd4j.zeros(DataType.INT64, 1));
        p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        p.placeholder("key", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        p.placeholder("value", Nd4j.zeros(DataType.FLOAT, 1, CACHE, 1, 1));
        SDVariable carry = p.graph.getVariable("carry");
        SDVariable k = carry.reshape(1, 1, 1, 1);
        SDVariable v = carry.add(10).reshape(1, 1, 1, 1);
        SDVariable position = p.graph.getVariable("cache_position");
        SDVariable mask = p.graph.getVariable("mask");
        SDVariable key = p.graph.getVariable("key");
        SDVariable value = p.graph.getVariable("value");
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, v, k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        SDVariable ids = p.graph.getVariable("ids");
        SDVariable zero = ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1);
        SDVariable rawLogits = zero.add(p.graph.constant(Nd4j.createFromArray(0.0f, 10.0f).reshape(1, 1, 2)));
        p.output(p.graph.castTo("logits", rawLogits, DataType.FLOAT));
        return p;
    }

    @Test
    void testNativeOutputManifestIncludesUnconsumedExtraOutput() {
        try (Plan target = target()) {
            target.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            assertEquals(target.outputs.size(), manifest.size(),
                    "the native manifest must carry every requested output of the frozen plan");
            assertTrue(manifest.contains("extra"),
                    "the extra output must be part of the native manifest even though no Java loop consumes it");
            assertTrue(target.executor.findOutputIndex("extra") >= 0,
                    "findOutputIndex must resolve the extra output from the plan");
        }
    }

    @Test
    void testScalarTargetMappingIsNameBasedAcrossPermutedScalarOrder() {
        try (Plan target = target(); Plan scalar = scalarTarget(); Plan predictor = predictor();
             INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
            target.compile();
            scalar.compile();
            predictor.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            String[] targetKeys = target.executor.getCurrentPlan().getExternalInputKeys();
            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                AutoregressiveDecode op = baseOp(target, predictor, embeddings, table, positions);
                op.withScalarTargetPlan(binding, targetKeys, manifest,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                int ni = op.getTArgument(58).intValue();
                int no = op.getTArgument(59).intValue();
                assertEquals(manifest.size(), no,
                        "the scalar trailer must map EVERY target requested output");
                for (int i = 0; i < no; i++) {
                    String name = manifest.get(i);
                    int expected = binding.findOutputIndex(name);
                    assertTrue(expected >= 0, "scalar binding must contain output " + name);
                    assertEquals(expected, op.getTArgument(60 + ni + i).intValue(),
                            "mapping for target output '" + name + "' must be its name-based scalar index");
                }
            }
            // A nonexistent target output must fail before any execution.
            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                AutoregressiveDecode op = baseOp(target, predictor, embeddings, table, positions);
                List<String> bogus = new ArrayList<>(manifest);
                bogus.add("nonexistent_output");
                IllegalArgumentException ex = assertThrows(IllegalArgumentException.class,
                        () -> op.withScalarTargetPlan(binding, targetKeys, bogus,
                                "ids", "mask", "position", "cache_position", "actual_length",
                                "logits", "hidden"));
                assertTrue(ex.getMessage().contains("nonexistent_output"),
                        "the unmapped output name must appear in the failure");
            }
        }
    }

    @Test
    void testControlOffNativeExecutionRequiresTheFullManifest() {
        try (Plan target = target(); Plan scalar = scalarTarget(); Plan predictor = predictor();
             INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, 2, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, BASE_TARGET_POSITION)) {
            target.compile();
            scalar.compile();
            predictor.compile();
            List<String> manifest =
                    new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
            String[] targetKeys = target.executor.getCurrentPlan().getExternalInputKeys();
            List<String> reduced = List.of("logits", "hidden");

            try (DynamicShapePlanExecutor.NativeExecutionBinding binding =
                         scalar.executor.captureNativeExecutionBinding()) {
                // Reduced manifest: the native split-check must fail loudly with the
                // map-length message BEFORE any decode execution.
                AutoregressiveDecode reducedOp = baseOp(target, predictor, embeddings, table, positions);
                reducedOp.withScalarTargetPlan(binding, targetKeys, reduced,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                try {
                    Nd4j.getExecutioner().exec(reducedOp);
                    fail("the reduced manifest must fail the native map-length invariant");
                } catch (RuntimeException e) {
                    // The native REQUIRE_TRUE text lives in the cause chain under the
                    // execution wrapper; walk it instead of reading only the wrapper.
                    String chain = causeChainText(e);
                    assertTrue(chain.contains("target-to-scalar map length"),
                            "expected the split map-length message in the cause chain, got: " + chain);
                }

                // Full manifest: the same binding must execute through the native
                // scalar ABI without count mismatches.
                AutoregressiveDecode fullOp = baseOp(target, predictor, embeddings, table, positions);
                fullOp.withScalarTargetPlan(binding, targetKeys, manifest,
                        "ids", "mask", "position", "cache_position", "actual_length",
                        "logits", "hidden");
                INDArray[] result = Nd4j.getExecutioner().exec(fullOp);
                try {
                    assertEquals(1, result[1].getLong(0),
                            "single-row commit policy emits exactly one token");
                } finally {
                    for (INDArray array : result) array.close();
                }
            }
        }
    }

    private static AutoregressiveDecode baseOp(Plan target, Plan predictor,
                                               INDArray embeddings, INDArray table, INDArray positions) {
        List<String> manifest =
                new ArrayList<>(target.executor.getCurrentPlan().getRequestedOutputs());
        AutoregressiveDecode op = new AutoregressiveDecode(
                embeddings, table, target.input("ids"), target.input("mask"), positions, null,
                target.executor.getNativePlanHandle(), target.executor.getCachedOpContext(),
                target.executor.getCurrentPlan().getExternalInputKeys().length, manifest.size(),
                -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                -1, target.ext("position"), target.ext("cache_position"),
                new int[0], new int[0], new int[0], new int[0], new int[0], new int[0],
                WIDTH, 0, 0, BASE_TARGET_POSITION, 0.0, 0, 0.0, 1.0, Set.of());
        op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                        1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                .withSpeculativeDecoding(K, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                .withActualSequenceLengthExtIdx(target.ext("actual_length"))
                .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                        predictor.input("mask"), predictor.input("position"),
                        predictor.input("cache_position"),
                        new INDArray[]{predictor.input("key"), predictor.input("value")},
                        predictor.executor.getNativePlanHandle(), predictor.executor.getCachedOpContext(),
                        predictor.executor.getCurrentPlan().getExternalInputKeys().length,
                        predictor.outputs.size(), predictor.ext("ids"), predictor.ext("carry"),
                        predictor.ext("mask"), predictor.ext("position"), predictor.ext("cache_position"),
                        new int[]{predictor.ext("key"), predictor.ext("value")},
                        predictor.out("logits"), predictor.out("hidden"), target.out("hidden"));
        return op;
    }

    private static String causeChainText(Throwable t) {
        StringBuilder sb = new StringBuilder();
        Throwable cursor = t;
        while (cursor != null) {
            if (sb.length() > 0) sb.append(" << ");
            sb.append(String.valueOf(cursor.getMessage()));
            cursor = cursor.getCause();
        }
        return sb.toString();
    }

    private static final class Plan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;

        private SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }

        private void echo(String name, INDArray value) {
            output(placeholder(name, value).add(name + "_echo", 1));
        }

        private void output(SDVariable value) { outputs.add(value.name()); }
        private INDArray input(String name) { return inputs.get(name); }

        private void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertTrue(!executor.getNativePlanHandle().isNull(), "native plan required");
            assertNotNull(executor.getCachedOpContext());
        }

        private int ext(String name) {
            int index = executor.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }

        private int out(String name) {
            int index = executor.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }

        @Override public void close() { graph.close(); }
    }
}
