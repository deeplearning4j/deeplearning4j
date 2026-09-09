/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.nd4j.dsp.runtime;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SameDiffOp;
import org.nd4j.autodiff.samediff.serde.SDZSerializer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Opt-in exact-artifact CPU feasibility gate, not a generation benchmark or a
 * cross-backend qualification. No import, graph rewrite, export or text session.
 * Both schedules execute the same native model with independent inference contexts.
 */
@EnabledIfSystemProperty(named = "sdx.qwen.prefillParity", matches = "true")
class Qwen35PrefillParityTest {
    private static final Path ROOT = Paths.get("/tmp/kompile-a2-bf16-device-snapshot");
    private static final Path MODEL = ROOT.resolve("model.sdz");
    private static final Path METADATA = ROOT.resolve("cache-metadata/objects/"
            + "27d3ad110c9e590bb5b7c6c7d011ae53f06bce02b0a9d4198a7f69a64967b470"
            + "/bundle/metadata/text-generation.json");
    private static final String SHA256 =
            "5b362ffbb583858dc2d410e11974767fed69ec46b37f31518ac9f9bd58539d94";
    private static final long MODEL_BYTES = 1505820715L;
    // Qwen35DesktopExecutionGateTest's exact Android smoke prompt.
    private static final long[] TOKENS = {248045, 846, 198, 20206, 440, 279, 3074,
            3299, 25, 5354, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271};
    private static final int CAPACITY = TOKENS.length + 1;
    // A small smoke-gate budget, fixed before observing results: four BF16
    // relative spacings (4 * 2^-7), plus one spacing at unit scale near zero.
    // FP32 GDN storage still receives BF16 projections, so it uses the same
    // end-to-end budget, NOT an inappropriate FP32-only arithmetic tolerance.
    private static final double ATOL = 1.0 / 128.0;
    private static final double RTOL = 4.0 / 128.0;

    @Test
    void exactImmutableGraphBatched19MatchesRolling1AndTransferredDecode() throws Exception {
        fingerprint("before");
        byte[] metadataBefore = Files.readAllBytes(METADATA);
        byte[] manifestBefore = Files.readAllBytes(ROOT.resolve("manifest.json"));
        JsonNode manifest = new ObjectMapper().readTree(manifestBefore);
        assertEquals(MODEL.toRealPath(), ROOT.resolve(manifest.required("modelPath").asText()).toRealPath(),
                "The native manifest must load the fingerprinted model");
        assertEquals(METADATA.toRealPath(), ROOT.resolve(manifest.required("textGeneration")
                .required("configPath").asText()).toRealPath(), "The native metadata must match inspection");
        try {
            Nd4j.getEnvironment();
            assertTrue(Nd4j.getBackend().getClass().getName().contains(".cpu."),
                    "This host-buffer feasibility gate requires the CPU ND4J backend");
            Contract c = inspect(new ObjectMapper().readTree(metadataBefore));
            List<String> mismatches = new ArrayList<>();
            System.out.println("QWEN_PREFILL BEGIN native-model-load " + MODEL);
            try (SdxRuntime runtime = SdxRuntime.create();
                 SdxRuntime.SdxModel model = runtime.loadModel(ROOT.toString(),
                         new SdxRuntime.ModelOptions().backend(SdxRuntime.SDX_BACKEND_AUTO)
                                 .strictBackend(true).allowRuntimeJit(true));
                 Frame batch = new Frame(model, c, TOKENS.length);
                 Frame rolling = new Frame(model, c, 1);
                 Frame batchNext = new Frame(model, c, 1);
                 Frame rollingNext = new Frame(model, c, 1)) {
                batch.run("batch19", TOKENS, 0);
                Map<String, float[]> history = new LinkedHashMap<>();
                for (String output : c.kv.values()) {
                    history.put(output, new float[TOKENS.length * 2 * 256]);
                }
                for (int p = 0; p < TOKENS.length; p++) {
                    rolling.run("rolling1/" + p, new long[]{TOKENS[p]}, p);
                    for (String output : c.kv.values()) {
                        float[] current = values(rolling.outputs.get(output));
                        System.arraycopy(current, 0, history.get(output), p * current.length, current.length);
                    }
                    rolling.advanceRecurrent();
                }
                for (String output : c.kv.values()) {
                    compare("current-KV/" + output, values(batch.outputs.get(output)),
                            history.get(output), ATOL, RTOL, mismatches);
                }
                compareFrames("prefill", batch, rolling, mismatches);
                expectArgmax("batch19", batch.outputs.get(c.logits), 2232, mismatches);
                expectArgmax("rolling19", rolling.outputs.get(c.logits), 2232, mismatches);

                // assign into the destination's existing buffers: never replace a
                // bound INDArray with an output/alias from another context.
                batchNext.transferFrom(batch);
                rollingNext.transferFrom(rolling);
                batchNext.run("batch-state/decode2232", new long[]{2232}, TOKENS.length);
                rollingNext.run("rolling-state/decode2232", new long[]{2232}, TOKENS.length);
                // A non-transferred continuation is the independent transfer control.
                rolling.run("original-state/decode2232", new long[]{2232}, TOKENS.length);
                compareFrames("transfer-control", rollingNext, rolling, mismatches);
                compareFrames("transferred-decode", batchNext, rollingNext, mismatches);
                for (String output : c.kv.values()) {
                    compare("decode-current-KV/" + output, values(batchNext.outputs.get(output)),
                            values(rollingNext.outputs.get(output)), ATOL, RTOL, mismatches);
                }
                expectArgmax("batch-state/decode", batchNext.outputs.get(c.logits), 248046, mismatches);
                expectArgmax("rolling-state/decode", rollingNext.outputs.get(c.logits), 248046, mismatches);
            }
            assertTrue(mismatches.isEmpty(), "Native prefill parity rejected:\n" + String.join("\n", mismatches));
        } finally {
            // Also execute after native admission/numerical failure.
            fingerprint("after");
            assertArrayEquals(metadataBefore, Files.readAllBytes(METADATA), "Metadata was mutated");
            assertArrayEquals(manifestBefore, Files.readAllBytes(ROOT.resolve("manifest.json")), "Manifest was mutated");
        }
    }

    /**
     * Diagnostic only: materialize a dependency cut ending at the first causal
     * convolution, not logits or the remaining layers. Enable both class and
     * method properties. Different GEMM/GEMV reduction orders need not be bitwise
     * identical; this reports the first differing checkpoint, not a new budget.
     * Requested intermediates may change fusion/liveness decisions: this cut
     * localizes arithmetic differences, it does not qualify full-model replay.
     */
    @Test
    @EnabledIfSystemProperty(named = "sdx.qwen.firstLayerDiagnostic", matches = "true")
    void exactImmutableFirstLayerBatched19VersusChained1Diagnostic() throws Exception {
        fingerprint("first-layer/before");
        byte[] metadataBefore = Files.readAllBytes(METADATA);
        byte[] manifestBefore = Files.readAllBytes(ROOT.resolve("manifest.json"));
        try {
            JsonNode manifest = new ObjectMapper().readTree(manifestBefore);
            assertEquals(MODEL.toRealPath(), ROOT.resolve(manifest.required("modelPath").asText()).toRealPath());
            assertEquals(METADATA.toRealPath(), ROOT.resolve(manifest.required("textGeneration")
                    .required("configPath").asText()).toRealPath());
            Nd4j.getEnvironment();
            assertTrue(Nd4j.getBackend().getClass().getName().contains(".cpu."), "CPU ND4J required");
            Prefix p;
            try (SameDiff graph = SDZSerializer.load(MODEL.toFile(), false)) {
                p = discoverPrefix(graph, new ObjectMapper().readTree(metadataBefore));
            }
            try (SdxRuntime runtime = SdxRuntime.create();
                 SdxRuntime.SdxModel model = runtime.loadModel(ROOT.toString(),
                         new SdxRuntime.ModelOptions().backend(SdxRuntime.SDX_BACKEND_AUTO)
                                 .strictBackend(true).allowRuntimeJit(true))) {
                // AUTO denotes the linked backend, not proof of CPU selection.
                String maps = Files.readString(Paths.get("/proc/self/maps"));
                assertTrue(maps.contains("libnd4jcpu.so"), "CPU native library must be mapped");
                assertFalse(maps.contains("libnd4jcuda.so"), "Mixed CPU/CUDA process is not this diagnostic");
                String cpuShim = "/cpu-parity-shim/libjnisdx.so";
                assertTrue(maps.contains(cpuShim), "Use the explicitly CPU-linked parity JNI shim");
                assertTrue(maps.lines().filter(line -> line.contains("/libjnisdx.so"))
                        .allMatch(line -> line.contains(cpuShim)), "A second SDX JNI shim was loaded");
                System.out.println("QWEN_PREFIX CPU_SHIM " + maps.lines().filter(line -> line.contains(cpuShim))
                        .findFirst().orElseThrow());
                try (PrefixFrame batch = new PrefixFrame(model, p, TOKENS.length);
                     PrefixFrame single = new PrefixFrame(model, p, 1)) {
                    batch.run(TOKENS);
                    Map<String, float[]> chained = new LinkedHashMap<>();
                    for (String name : p.stages) chained.put(name,
                            new float[Math.toIntExact(batch.out.get(name).length())]);
                    for (int t = 0; t < TOKENS.length; t++) {
                        // Embedding, normalization and projection have no recurrent
                        // dependencies; only convolution consumes the previous state.
                        single.run(new long[]{TOKENS[t]});
                        for (String name : p.stages) {
                            float[] row = values(single.out.get(name));
                            System.arraycopy(row, 0, chained.get(name), t * row.length, row.length);
                        }
                        single.state.assign(single.out.get(p.stateOut));
                    }
                    String first = null;
                    for (String name : p.stages) {
                        int differing = diagnosticDiff(name, values(batch.out.get(name)), chained.get(name));
                        if (first == null && differing >= 0) first = name;
                    }
                    diagnosticDiff("final-state/" + p.stateOut, values(batch.out.get(p.stateOut)),
                            values(single.out.get(p.stateOut)));
                    System.out.println("QWEN_PREFIX FIRST_NONBITIDENTICAL_CHECKPOINT=" + (first == null ? "NONE" : first)
                            + " (observed checkpoints only; numerical differences are not a bit-parity requirement)");
                    projectionReference(p, batch.out, chained);
                }
            }
        } finally {
            fingerprint("first-layer/after");
            assertArrayEquals(metadataBefore, Files.readAllBytes(METADATA), "Metadata was mutated");
            assertArrayEquals(manifestBefore, Files.readAllBytes(ROOT.resolve("manifest.json")), "Manifest was mutated");
        }
    }

    private static final class Prefix {
        String ids, length, stateIn, stateOut, projectionInput, projectionRaw, qkv;
        long[] lengthShape, stateShape;
        int hidden, channels;
        float[] weight; // Logical C-order [hidden, channels], independent of model ownership.
        final List<String> stages = new ArrayList<>();
        final Map<String, DataType> types = new LinkedHashMap<>();
        final Map<String, long[]> shapes = new LinkedHashMap<>();
        String[] outputs() {
            List<String> names = new ArrayList<>(stages);
            names.add(stateOut);
            return names.toArray(new String[0]);
        }
    }

    private static SameDiffOp producer(SameDiff graph, String name) {
        requiredVariable(graph, name);
        String op = graph.getVariables().get(name).getOutputOfOp();
        return op == null ? null : graph.getOps().get(op);
    }

    // Postorder gives dependency order, without guessing generated variable names.
    private static void ancestors(SameDiff graph, String name, Map<String, SameDiffOp> found) {
        SameDiffOp op = producer(graph, name);
        if (op == null || found.containsKey(op.getName())) return;
        for (String input : op.getInputsToOp()) ancestors(graph, input, found);
        found.put(op.getName(), op);
    }

    private static Prefix discoverPrefix(SameDiff graph, JsonNode metadata) {
        Prefix p = new Prefix();
        JsonNode io = metadata.required("io");
        p.ids = io.required("inputIds").asText();
        p.length = io.required("actualSequenceLength").asText();
        assertEquals(DataType.INT64, requiredVariable(graph, p.ids).dataType());
        long[] idsShape = requiredVariable(graph, p.ids).getShape();
        assertNotNull(idsShape);
        assertEquals(2, idsShape.length);
        assertTrue(idsShape[0] == 1 || idsShape[0] == -1,
                "Serialized batch dimension must admit one sequence");
        assertEquals(-1, idsShape[1], "Serialized sequence dimension must be dynamic");
        assertEquals(DataType.INT64, requiredVariable(graph, p.length).dataType());
        p.lengthShape = requiredVariable(graph, p.length).getShape();
        if (p.lengthShape == null) p.lengthShape = new long[0];
        assertTrue(p.lengthShape.length == 0 || Arrays.equals(p.lengthShape, new long[]{1}));
        List<JsonNode> candidates = new ArrayList<>();
        for (JsonNode state : io.required("recurrentStates")) {
            if (!"CONV".equals(state.required("kind").asText())) continue;
            var conv = producer(graph, state.required("output").asText());
            assertNotNull(conv);
            assertEquals("causal_conv1d", conv.getOp().opName());
            Map<String, SameDiffOp> upstream = new LinkedHashMap<>();
            ancestors(graph, conv.getInputsToOp().get(0), upstream);
            // The first layer has no earlier causal convolution in its data ancestry.
            if (upstream.values().stream().noneMatch(o -> "causal_conv1d".equals(o.getOp().opName()))) {
                candidates.add(state);
            }
        }
        assertEquals(1, candidates.size(), "Require an unambiguous first-layer dependency cut");
        JsonNode state = candidates.get(0);
        p.stateIn = state.required("input").asText();
        p.stateOut = state.required("output").asText();
        p.stateShape = shape(state.required("shape"));
        assertEquals(3, p.stateShape.length);
        assertEquals(1, p.stateShape[0]);
        assertTrue(p.stateShape[2] > 0 && p.stateShape[2] < TOKENS.length);
        p.channels = Math.toIntExact(p.stateShape[1]);
        assertTrue(p.channels > 0);
        var conv = producer(graph, p.stateOut);
        assertEquals(Arrays.asList(conv.getInputsToOp().get(0), conv.getInputsToOp().get(1),
                p.stateIn, p.length), conv.getInputsToOp(), "No bias or hidden state dependencies in this cut");
        assertEquals(p.stateOut, conv.getOutputsOfOp().get(1));
        assertArrayEquals(new long[]{1, 0}, ((DynamicCustomOp) conv.getOp()).iArgs());
        assertArrayEquals(new long[]{p.channels, p.stateShape[2] + 1},
                requiredVariable(graph, conv.getInputsToOp().get(1)).getShape());
        p.qkv = conv.getInputsToOp().get(0);
        Map<String, SameDiffOp> upstream = new LinkedHashMap<>();
        ancestors(graph, p.qkv, upstream);
        var projections = upstream.values().stream().filter(o -> "matmul".equals(o.getOp().opName()))
                .collect(java.util.stream.Collectors.toList());
        var embeddings = upstream.values().stream().filter(o -> "gather".equals(o.getOp().opName())
                && o.getInputsToOp().contains(p.ids)).collect(java.util.stream.Collectors.toList());
        assertEquals(1, projections.size(), "Only the first dense QKV projection may be in the cut");
        assertEquals(1, embeddings.size(), "Embedding must consume metadata inputIds");
        assertTrue(upstream.values().stream().anyMatch(o -> "mean".equals(o.getOp().opName())
                || "reduce_mean".equals(o.getOp().opName()) || "rms_norm".equals(o.getOp().opName())),
                "Require RMS normalization ancestry");
        var mm = projections.get(0);
        long[] flags = ((DynamicCustomOp) mm.getOp()).iArgs();
        for (long flag : flags) assertEquals(0, flag, "Reference admits non-transposed dense matmul only");
        p.projectionInput = mm.getInputsToOp().get(0);
        p.projectionRaw = mm.getOutputsOfOp().get(0);
        String projected = p.qkv;
        while (!projected.equals(p.projectionRaw)) {
            var cast = producer(graph, projected);
            assertNotNull(cast);
            assertEquals("cast", cast.getOp().opName(), "QKV may only narrow the dense projection");
            projected = cast.getInputsToOp().get(0);
        }
        var embedding = embeddings.get(0);
        long[] table = requiredVariable(graph, embedding.getInputsToOp().get(0)).getShape();
        assertNotNull(table);
        assertEquals(2, table.length);
        p.hidden = Math.toIntExact(table[1]);
        for (long token : TOKENS) assertTrue(token >= 0 && token < table[0]);
        try (INDArray w = constantOperand(graph, mm.getInputsToOp().get(1))) {
            assertArrayEquals(new long[]{p.hidden, p.channels}, w.shape());
            p.weight = values(w);
        }
        String norm = p.projectionInput;
        while (producer(graph, norm) != null && "cast".equals(producer(graph, norm).getOp().opName())) {
            norm = producer(graph, norm).getInputsToOp().get(0);
        }
        // Materialize checkpoints in dependency order, including pre-narrowing
        // matmul output to separate accumulation drift from BF16 rounding.
        for (String name : new String[]{embedding.getOutputsOfOp().get(0), norm, p.projectionInput,
                p.projectionRaw, p.qkv, conv.getOutputsOfOp().get(0)}) {
            if (p.stages.contains(name)) continue;
            p.stages.add(name);
            int features = name.equals(p.projectionRaw) || name.equals(p.qkv)
                    || name.equals(conv.getOutputsOfOp().get(0)) ? p.channels : p.hidden;
            p.shapes.put(name, new long[]{1, -1, features});
        }
        p.shapes.put(p.stateOut, p.stateShape.clone());
        p.shapes.put(p.stateIn, p.stateShape.clone());
        for (String name : p.shapes.keySet()) {
            p.types.put(name, requiredVariable(graph, name).dataType());
            assertTrue(p.types.get(name) == DataType.BFLOAT16 || p.types.get(name) == DataType.FLOAT,
                    name + " unexpected storage dtype");
        }
        assertEquals(dtype(state.required("dataType").asText()), p.types.get(p.stateIn));
        assertEquals(p.types.get(p.stateIn), p.types.get(p.stateOut));
        assertEquals(p.types.get(p.qkv), p.types.get(p.stateOut));
        long[] declaredState = requiredVariable(graph, p.stateIn).getShape();
        assertNotNull(declaredState);
        assertEquals(p.stateShape.length, declaredState.length);
        for (int d = 0; d < declaredState.length; d++) {
            assertTrue(declaredState[d] == -1 || declaredState[d] == p.stateShape[d],
                    "State metadata must fit serialized dimension " + d);
        }
        upstream.values().forEach(o -> System.out.println("QWEN_PREFIX DEP " + o.getOp().opName()
                + " " + o.getInputsToOp() + " -> " + o.getOutputsOfOp()));
        p.stages.forEach(n -> System.out.println("QWEN_PREFIX CHECKPOINT " + n + " dtype=" + p.types.get(n)
                + " shape=" + Arrays.toString(p.shapes.get(n))));
        return p;
    }

    // Evaluate only parameter-side cast/transpose views, never run the Java graph.
    // Every returned buffer is owned, dense C order; serialized weights are not changed.
    private static INDArray constantOperand(SameDiff graph, String name) {
        SDVariable v = requiredVariable(graph, name);
        if (v.getVariableType() == VariableType.CONSTANT || v.getVariableType() == VariableType.VARIABLE) {
            assertNotNull(v.getArr(), name);
            assertTrue(v.dataType().isFPType(), "Packed weights require a different reference: " + name);
            return v.getArr().dup('c');
        }
        var op = producer(graph, name);
        assertNotNull(op, "Weight must not depend on a placeholder: " + name);
        try (INDArray input = constantOperand(graph, op.getInputsToOp().get(0))) {
            if ("cast".equals(op.getOp().opName())) {
                if (input.dataType() == v.dataType()) return input.dup('c');
                try (INDArray cast = input.castTo(v.dataType())) { return cast.dup('c'); }
            }
            assertEquals("permute", op.getOp().opName(), "Unsupported parameter transformation: " + name);
            long[] axes = ((DynamicCustomOp) op.getOp()).iArgs();
            assertArrayEquals(new long[]{1, 0}, axes);
            return input.permute(1, 0).dup('c');
        }
    }

    private static final class PrefixFrame implements AutoCloseable {
        final Prefix p;
        final SdxRuntime.SdxContext context;
        INDArray ids, length, state;
        final Map<String, INDArray> out = new LinkedHashMap<>();
        PrefixFrame(SdxRuntime.SdxModel model, Prefix p, int width) {
            this.p = p;
            context = model.createInferenceContext(p.outputs());
            try {
                ids = Nd4j.create(DataType.INT64, new long[]{1, width}, 'c').assign(0);
                length = Nd4j.create(DataType.INT64, p.lengthShape, 'c').assign(width);
                state = Nd4j.create(p.types.get(p.stateIn), p.stateShape, 'c').assign(0);
                assertEquals(new java.util.HashSet<>(Arrays.asList(p.ids, p.length, p.stateIn)),
                        new java.util.HashSet<>(Arrays.asList(context.inputNames())),
                        "Cut must not execute later layers or depend on their caches");
                assertEquals(3, context.numInputs(), "Exactly ids, actual length and the first CONV state");
                assertArrayEquals(p.outputs(), context.outputNames());
                for (String name : p.outputs()) {
                    long[] s = p.shapes.get(name).clone();
                    if (!name.equals(p.stateOut)) s[1] = width;
                    out.put(name, Nd4j.create(p.types.get(name), s, 'c').assign(Double.NaN));
                }
            } catch (RuntimeException | Error e) { close(); throw e; }
        }
        void run(long[] tokens) {
            assertEquals(ids.length(), tokens.length);
            for (int i = 0; i < tokens.length; i++) ids.putScalar(i, tokens[i]);
            float[] previous = values(state);
            out.values().forEach(a -> a.assign(Double.NaN));
            Object[] inputs = Arrays.stream(context.inputNames()).map(n -> n.equals(p.ids) ? ids
                    : n.equals(p.length) ? length : state).toArray();
            long start = System.nanoTime();
            context.runNd4j(inputs, Arrays.stream(p.outputs()).map(out::get).toArray());
            var report = context.executionReport();
            assertEquals(SdxRuntime.SDX_STATUS_OK, report.status_code);
            assertEquals(SdxRuntime.SDX_BACKEND_AUTO, report.requested_backend);
            assertEquals(SdxRuntime.SDX_BACKEND_AUTO, report.applied_backend);
            assertEquals(0, report.used_fallback);
            assertTrue(report.execution_count > 0);
            assertNotEquals(3, report.plan_phase, "Replay blocked");
            for (int i = 0; i < context.numOutputs(); i++) {
                String name = context.outputName(i);
                var actual = context.outputTensor(i);
                assertEquals(out.get(name).dataType().toInt(), actual.dtype, name);
                assertArrayEquals(out.get(name).shape(), actual.shapeValues(), name);
            }
            // A state is exactly the last K-1 input QKV rows (or previous
            // history when L < K-1), transposed to [B,D,K-1], NOT conv output.
            float[] qkv = values(out.get(p.qkv)), next = values(out.get(p.stateOut));
            int history = Math.toIntExact(p.stateShape[2]);
            float[] expected = new float[next.length];
            for (int d = 0; d < p.channels; d++) for (int h = 0; h < history; h++) {
                int t = tokens.length - history + h;
                expected[d * history + h] = t < 0 ? previous[d * history + tokens.length + h]
                        : qkv[t * p.channels + d];
            }
            assertArrayEquals(expected, next, "Exact causal state suffix invariant");
            assertArrayEquals(previous, values(state), "State input must not be overwritten");
            System.out.printf("QWEN_PREFIX RUN width=%d elapsed_ms=%.3f phase=%d%n",
                    tokens.length, (System.nanoTime() - start) / 1e6, report.plan_phase);
        }
        @Override public void close() {
            try { context.close(); }
            finally {
                out.values().forEach(INDArray::close);
                if (state != null) state.close();
                if (length != null) length.close();
                if (ids != null) ids.close();
            }
        }
    }

    private static int diagnosticDiff(String name, float[] a, float[] b) {
        assertEquals(a.length, b.length, name);
        int first = -1, count = 0;
        double max = 0, squared = 0;
        for (int i = 0; i < a.length; i++) {
            assertTrue(Float.isFinite(a[i]) && Float.isFinite(b[i]), name + " nonfinite at " + i);
            if (Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                if (first < 0) first = i;
                count++;
            }
            double error = (double) a[i] - b[i];
            max = Math.max(max, Math.abs(error)); squared += error * error;
        }
        System.out.printf("QWEN_PREFIX DIFF %s n=%d nonbitidentical=%d first=%d max_abs=%.9g rmse=%.9g%n",
                name, a.length, count, first, max, Math.sqrt(squared / a.length));
        return first;
    }

    private static void projectionReference(Prefix p, Map<String, INDArray> batch, Map<String, float[]> single) {
        float[] a = values(batch.get(p.projectionInput)), b = single.get(p.projectionInput);
        float[] raw = values(batch.get(p.projectionRaw)), rawSingle = single.get(p.projectionRaw);
        float[] q = values(batch.get(p.qkv)), qSingle = single.get(p.qkv);
        java.util.Set<Integer> samples = new java.util.LinkedHashSet<>();
        for (int i = 0; i < 16; i++) samples.add(i * (q.length - 1) / 15);
        for (int i = 0; i < q.length && samples.size() < 48; i++) {
            if (Float.floatToRawIntBits(q[i]) != Float.floatToRawIntBits(qSingle[i])) samples.add(i);
        }
        for (int i : samples) {
            int row = i / p.channels, col = i % p.channels;
            float dot = 0, dotSingle = 0;
            double wide = 0;
            for (int k = 0; k < p.hidden; k++) {
                float w = p.weight[k * p.channels + col];
                assertTrue(Float.isFinite(w), "Nonfinite projection weight");
                // Deliberately separate FP32 products/adds, not BLAS or native matmul.
                dot += a[row * p.hidden + k] * w;
                dotSingle += b[row * p.hidden + k] * w;
                wide += (double) a[row * p.hidden + k] * w;
            }
            assertTrue(Float.isFinite(dot) && Float.isFinite(dotSingle) && Double.isFinite(wide));
            int bits = Float.floatToRawIntBits(dot);
            float bf16Rne = Float.intBitsToFloat((bits + 0x7fff + ((bits >>> 16) & 1)) & 0xffff0000);
            System.out.printf("QWEN_PREFIX DOT_SAMPLE index=%d raw_batch=%.9g raw_single=%.9g "
                            + "qkv_batch=%.9g qkv_single=%.9g fp32_serial=%.9g fp32_single=%.9g "
                            + "fp64=%.17g bf16_rne=%.9g fp32_low16=0x%04x "
                            + "raw_batch_low16=0x%04x raw_single_low16=0x%04x fp64_minus_qkv_midpoint=%.17g%n",
                    i, raw[i], rawSingle[i], q[i], qSingle[i], dot, dotSingle, wide, bf16Rne, bits & 0xffff,
                    Float.floatToRawIntBits(raw[i]) & 0xffff, Float.floatToRawIntBits(rawSingle[i]) & 0xffff,
                    wide - ((double) q[i] + qSingle[i]) / 2);
        }
        System.out.println("QWEN_PREFIX DOT_REFERENCE sampled=" + samples.size()
                + " (serial FP32 and FP64 diagnostics, not a BLAS reduction-order oracle; RNE tie low16=0x8000)");
    }

    private static final class Contract {
        String ids, mask, offset, position, length, logits;
        int vocabulary;
        final Map<String, DataType> types = new LinkedHashMap<>();
        final Map<String, long[]> shapes = new LinkedHashMap<>();
        final Map<String, String> kv = new LinkedHashMap<>();
        final Map<String, String> recurrent = new LinkedHashMap<>();
        String[] outputNames() {
            List<String> names = new ArrayList<>();
            names.add(logits);
            names.addAll(kv.values());
            names.addAll(recurrent.values());
            return names.toArray(new String[0]);
        }
    }

    private static Contract inspect(JsonNode metadata) throws Exception {
        long start = System.nanoTime();
        System.out.println("QWEN_PREFILL BEGIN graph-admission");
        assertEquals(2, metadata.path("formatVersion").asInt());
        JsonNode io = metadata.required("io"), execution = metadata.required("execution");
        assertEquals("BSHD", execution.required("kvLayout").asText());
        assertTrue(execution.required("planOwnsKvScatter").asBoolean());
        assertTrue(metadata.required("limits").required("maxPrefillLength").asInt() >= TOKENS.length);
        assertTrue(metadata.required("limits").required("contextLength").asInt() >= CAPACITY);
        assertTrue(metadata.required("limits").required("maxBatchSize").asInt() >= 1);
        Contract c = new Contract();
        c.ids = io.required("inputIds").asText();
        c.mask = io.required("causalMask").asText();
        c.offset = io.required("positionOffset").asText();
        c.position = io.required("cachePosition").asText();
        c.length = io.required("actualSequenceLength").asText();
        c.logits = io.required("logits").asText();
        // Inspection only. Close the Java graph before loading the native model.
        try (SameDiff graph = SDZSerializer.load(MODEL.toFile(), false)) {
            admit(graph, c, c.ids, DataType.INT64, new long[]{1, -1});
            for (String scalar : new String[]{c.offset, c.position, c.length}) {
                SDVariable variable = requiredVariable(graph, scalar);
                assertEquals(DataType.INT64, variable.dataType(), scalar);
                long[] shape = variable.getShape();
                // SameDiffSerializer leaves rank-zero FlatVariable shapes null.
                if (shape == null) shape = new long[0];
                assertTrue(shape.length == 0 || Arrays.equals(shape, new long[]{1}), scalar);
                admit(graph, c, scalar, DataType.INT64, shape);
            }
            admit(graph, c, c.mask, dtype(execution.required("maskDtype").asText()),
                    new long[]{1, 1, -1, -1});
            DataType kvType = dtype(execution.required("kvDtype").asText());
            assertEquals(DataType.BFLOAT16, kvType, "Exact snapshot KV contract");
            for (String kind : new String[]{"Key", "Value"}) {
                JsonNode inputs = io.required("kv" + kind + "Inputs");
                JsonNode shapes = io.required("kv" + kind + "Shapes");
                JsonNode outputs = io.required("prefill" + kind + "Outputs");
                assertEquals(6, inputs.size(), "Every attention layer must be covered");
                assertEquals(inputs.size(), outputs.size());
                assertEquals(inputs.size(), shapes.size());
                for (int i = 0; i < inputs.size(); i++) {
                    String input = inputs.get(i).asText(), output = outputs.get(i).asText();
                    long[] shape = shape(shapes.get(i));
                    assertArrayEquals(new long[]{1, -1, 2, 256}, shape, input);
                    admit(graph, c, input, kvType, shape);
                    DataType producedType = requiredVariable(graph, output).dataType();
                    assertTrue(producedType.isFPType(), output + " must expose floating current K/V");
                    admit(graph, c, output, producedType, shape);
                    c.kv.put(input, output);
                }
            }
            assertEquals(36, io.required("recurrentStates").size());
            for (JsonNode state : io.required("recurrentStates")) {
                String input = state.required("input").asText(), output = state.required("output").asText();
                DataType type = dtype(state.required("dataType").asText());
                boolean conv = "CONV".equals(state.required("kind").asText());
                assertTrue(conv || "GDN".equals(state.required("kind").asText()), input);
                assertEquals(conv ? DataType.BFLOAT16 : DataType.FLOAT, type, input);
                long[] shape = shape(state.required("shape"));
                assertArrayEquals(conv ? new long[]{1, 6144, 3} : new long[]{1, 16, 128, 128}, shape, input);
                admit(graph, c, input, type, shape);
                admit(graph, c, output, type, shape);
                c.recurrent.put(input, output);
            }
            assertEquals(12, c.kv.size(), "KV names must be unique");
            assertEquals(36, c.recurrent.size(), "Recurrent state names must be unique");
            SDVariable logits = requiredVariable(graph, c.logits);
            // ARRAY variables deliberately return null from SDVariable.getShape().
            // LLaMAArchitecture's last-position head preserves [B,1,V]; derive V
            // from the serialized embedding table, then check native output shape.
            long[] embeddingShape = requiredVariable(graph, "model.embed_tokens.weight").getShape();
            assertNotNull(embeddingShape, "Serialized embedding shape");
            assertEquals(2, embeddingShape.length, "Exact BF16 embedding must be a dense table");
            assertTrue(embeddingShape[0] > 248069, "Exact tokenizer vocabulary must fit logits");
            c.vocabulary = Math.toIntExact(embeddingShape[0]);
            assertTrue(logits.dataType().isFPType(), "Logits must have floating storage");
            admit(graph, c, c.logits, logits.dataType(), new long[]{1, 1, c.vocabulary});
        }
        System.out.printf("QWEN_PREFILL END graph-admission elapsed_ms=%.3f states=%d kv=%d%n",
                (System.nanoTime() - start) / 1e6, c.recurrent.size(), c.kv.size());
        return c;
    }

    private static SDVariable requiredVariable(SameDiff graph, String name) {
        SDVariable v = graph.getVariable(name);
        assertNotNull(v, "Missing exact-graph variable: " + name);
        return v;
    }

    private static void admit(SameDiff graph, Contract c, String name, DataType type, long[] expected) {
        SDVariable v = requiredVariable(graph, name);
        assertEquals(type, v.dataType(), name + ": graph/metadata dtype mismatch; do not substitute "
                + "float KV for quantized storage or discard inline scales through the public tensor API");
        if (v.getVariableType() != VariableType.ARRAY) {
            long[] actual = v.getShape();
            if (actual == null && expected.length == 0) actual = new long[0];
            assertNotNull(actual, name + " must have a declared input shape");
            assertEquals(expected.length, actual.length, name + " rank");
            for (int d = 0; d < actual.length; d++) {
                assertTrue(actual[d] == -1 || expected[d] == -1 || actual[d] == expected[d],
                        name + " graph=" + Arrays.toString(actual) + " metadata=" + Arrays.toString(expected));
                if (expected[d] == -1) assertEquals(-1, actual[d], name + " must admit variable-length prefill");
            }
        } // Intermediate shapes are checked against native outputs after every run.
        c.types.put(name, type);
        c.shapes.put(name, expected.clone());
    }

    private static final class Frame implements AutoCloseable {
        final Contract c;
        final SdxRuntime.SdxContext context;
        final Map<String, INDArray> inputs = new LinkedHashMap<>();
        final Map<String, INDArray> outputs = new LinkedHashMap<>();
        final int width;

        Frame(SdxRuntime.SdxModel model, Contract c, int width) {
            this.c = c;
            this.width = width;
            long start = System.nanoTime();
            System.out.println("QWEN_PREFILL BEGIN context-create width=" + width);
            context = model.createInferenceContext(c.outputNames());
            try {
                for (String name : new String[]{c.ids, c.mask, c.offset, c.position, c.length}) {
                    long[] shape = c.shapes.get(name).clone();
                    if (name.equals(c.ids)) shape = new long[]{1, width};
                    if (name.equals(c.mask)) shape = new long[]{1, 1, width, CAPACITY};
                    inputs.put(name, Nd4j.create(c.types.get(name), shape, 'c').assign(0));
                }
                for (String name : c.kv.keySet()) {
                    inputs.put(name, Nd4j.create(c.types.get(name), new long[]{1, CAPACITY, 2, 256}, 'c').assign(0));
                }
                for (String name : c.recurrent.keySet()) {
                    inputs.put(name, Nd4j.create(c.types.get(name), c.shapes.get(name), 'c').assign(0));
                }
                for (String name : c.outputNames()) {
                    long[] shape = c.shapes.get(name).clone();
                    if (c.kv.containsValue(name)) shape[1] = width;
                    outputs.put(name, Nd4j.create(c.types.get(name), shape, 'c').assign(0));
                }
                assertEquals(inputs.keySet(), new java.util.HashSet<>(Arrays.asList(context.inputNames())),
                        "Native context has unexpected/missing inputs: " + Arrays.toString(context.inputNames()));
                assertArrayEquals(c.outputNames(), context.outputNames());
                System.out.printf("QWEN_PREFILL END context-create width=%d elapsed_ms=%.3f%n",
                        width, (System.nanoTime() - start) / 1e6);
            } catch (RuntimeException | Error e) {
                close();
                throw e;
            }
        }

        void run(String stage, long[] tokens, int position) {
            assertEquals(width, tokens.length);
            INDArray ids = inputs.get(c.ids), mask = inputs.get(c.mask);
            for (int i = 0; i < width; i++) ids.putScalar(i, tokens[i]);
            inputs.get(c.offset).assign(position);
            inputs.get(c.position).assign(position);
            inputs.get(c.length).assign(width);
            // Same additive-mask convention as native SdxGenerationSession.
            mask.assign(-1.0e9);
            for (int q = 0; q < width; q++) {
                for (int k = 0; k <= position + q; k++) mask.putScalar((long) q * CAPACITY + k, 0.0);
            }
            for (INDArray output : outputs.values()) output.assign(Double.NaN);
            Object[] in = Arrays.stream(context.inputNames()).map(inputs::get).toArray();
            Object[] out = Arrays.stream(context.outputNames()).map(outputs::get).toArray();
            long start = System.nanoTime();
            System.out.printf("QWEN_PREFILL BEGIN %s width=%d position=%d%n", stage, width, position);
            context.runNd4j(in, out);
            SdxRuntime.ExecutionReport report = context.executionReport();
            System.out.printf("QWEN_PREFILL END %s elapsed_ms=%.3f phase=%d executions=%d requested=%d applied=%d fallback=%d%n",
                    stage, (System.nanoTime() - start) / 1e6, report.plan_phase, report.execution_count,
                    report.requested_backend, report.applied_backend, report.used_fallback);
            assertEquals(SdxRuntime.SDX_STATUS_OK, report.status_code, stage);
            assertEquals(SdxRuntime.SDX_BACKEND_AUTO, report.requested_backend, stage);
            assertEquals(SdxRuntime.SDX_BACKEND_AUTO, report.applied_backend, stage);
            assertEquals(0, report.used_fallback, stage);
            assertTrue(report.execution_count > 0, stage);
            // Not a replay qualification: first execution may legitimately be warmup.
            assertNotEquals(3, report.plan_phase, stage + " replay blocked");
            // Verify actual in-place cache writeback, not merely two equally
            // unchanged buffers. Current K/V and their BF16 cache rows must agree
            // exactly; all still-unwritten rows must remain zero.
            c.kv.forEach((input, output) -> {
                float[] cache = values(inputs.get(input));
                // RoPE may promote its current output. Scatter narrows to the
                // metadata-declared KV storage dtype, which must be reflected
                // in the writeback reference rather than assuming equal dtypes.
                float[] current = storageValues(outputs.get(output), inputs.get(input).dataType());
                int row = 2 * 256;
                assertArrayEquals(current, Arrays.copyOfRange(cache, position * row,
                        (position + width) * row), stage + "/scatter/" + input);
                assertArrayEquals(new float[(CAPACITY - position - width) * row],
                        Arrays.copyOfRange(cache, (position + width) * row, cache.length),
                        stage + "/unwritten-cache/" + input);
            });
            for (int i = 0; i < context.numOutputs(); i++) {
                String name = context.outputName(i);
                SdxRuntime.TensorView actual = context.outputTensor(i); // borrowed; context owns it
                assertEquals(outputs.get(name).dataType().toInt(), actual.dtype, stage + "/" + name);
                assertArrayEquals(outputs.get(name).shape(), actual.shapeValues(), stage + "/" + name);
            }
        }

        void advanceRecurrent() {
            c.recurrent.forEach((input, output) -> inputs.get(input).assign(outputs.get(output)));
        }

        void transferFrom(Frame source) {
            c.kv.forEach((input, output) -> copyPreserving(input, source.inputs.get(input)));
            c.recurrent.forEach((input, output) -> copyPreserving(input, source.outputs.get(output)));
        }

        void copyPreserving(String name, INDArray source) {
            INDArray destination = inputs.get(name);
            assertNotSame(source, destination, name + " destination must be independent");
            long address = destination.data().addressPointer().address();
            assertNotEquals(source.data().addressPointer().address(), address, name + " aliases source");
            destination.assign(source);
            assertSame(destination, inputs.get(name));
            assertEquals(address, destination.data().addressPointer().address(), name + " destination replaced");
            assertArrayEquals(values(source), values(destination), name + " transfer must be lossless");
        }

        @Override public void close() {
            try { context.close(); }
            finally {
                outputs.values().forEach(INDArray::close);
                inputs.values().forEach(INDArray::close);
            }
        }
    }

    private static void compareFrames(String stage, Frame a, Frame b, List<String> failures) {
        for (String input : a.c.kv.keySet()) {
            compare(stage + "/retained-KV/" + input, values(a.inputs.get(input)), values(b.inputs.get(input)),
                    ATOL, RTOL, failures);
        }
        for (String output : a.c.recurrent.values()) {
            compare(stage + "/" + output, values(a.outputs.get(output)), values(b.outputs.get(output)),
                    ATOL, RTOL, failures);
        }
        compare(stage + "/logits", values(a.outputs.get(a.c.logits)), values(b.outputs.get(b.c.logits)),
                ATOL, RTOL, failures);
    }

    private static float[] storageValues(INDArray array, DataType storage) {
        if (array.dataType() == storage) return values(array);
        try (INDArray narrowed = array.castTo(storage)) {
            return values(narrowed);
        }
    }

    private static float[] values(INDArray array) {
        // Owned C-contiguous CPU buffers; one bulk read, not per-element syncs.
        return array.data().asFloat();
    }

    private static void compare(String name, float[] a, float[] b, double atol, double rtol,
                                List<String> failures) {
        assertEquals(a.length, b.length, name);
        double maxAbs = 0, squareError = 0;
        int bad = 0, first = -1;
        for (int i = 0; i < a.length; i++) {
            double error = Math.abs((double) a[i] - b[i]);
            boolean finite = Float.isFinite(a[i]) && Float.isFinite(b[i]);
            if (!finite || error > atol + rtol * Math.max(Math.abs(a[i]), Math.abs(b[i]))) {
                bad++;
                if (first < 0) first = i;
            }
            maxAbs = Math.max(maxAbs, error);
            squareError += error * error;
        }
        String evidence = String.format("%s n=%d max_abs=%.9g rmse=%.9g atol=%.9g rtol=%.9g mismatches=%d%s",
                name, a.length, maxAbs, Math.sqrt(squareError / a.length), atol, rtol, bad,
                first < 0 ? "" : " first=" + first + " batch=" + a[first] + " rolling=" + b[first]);
        System.out.println("QWEN_PREFILL DIFF " + evidence);
        if (bad > 0) failures.add(evidence);
    }

    private static void expectArgmax(String stage, INDArray logits, int expected, List<String> failures) {
        float[] data = values(logits);
        int best = 0;
        for (int i = 0; i < data.length; i++) {
            if (!Float.isFinite(data[i])) {
                failures.add(stage + " nonfinite logit at " + i);
                break;
            }
            if (data[i] > data[best]) best = i;
        }
        System.out.printf("QWEN_PREFILL ARGMAX %s actual=%d expected=%d%n", stage, best, expected);
        if (best != expected) failures.add(stage + " argmax=" + best + " expected=" + expected);
    }

    private static DataType dtype(String name) {
        return "FLOAT32".equals(name) ? DataType.FLOAT : DataType.valueOf(name);
    }

    private static long[] shape(JsonNode node) {
        long[] result = new long[node.size()];
        for (int i = 0; i < result.length; i++) result[i] = node.get(i).asLong();
        return result;
    }

    private static void fingerprint(String stage) throws Exception {
        long start = System.nanoTime();
        System.out.println("QWEN_PREFILL BEGIN sha256/" + stage);
        assertEquals(MODEL_BYTES, Files.size(MODEL), "Exact immutable SDZ size");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(MODEL)) {
            byte[] buffer = new byte[1024 * 1024];
            int n;
            while ((n = input.read(buffer)) != -1) digest.update(buffer, 0, n);
        }
        StringBuilder hash = new StringBuilder();
        for (byte value : digest.digest()) hash.append(String.format("%02x", value & 255));
        System.out.printf("QWEN_PREFILL END sha256/%s sha=%s elapsed_ms=%.3f%n",
                stage, hash, (System.nanoTime() - start) / 1e6);
        assertEquals(SHA256, hash.toString(), "Exact immutable SDZ SHA-256 " + stage);
    }
}
