/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueNDArray;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import java.util.LinkedHashMap;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.ggml.GGMLModelImport;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/** Replays native predictor artifacts with the original GGUF weights and independent
 * input storage. This isolates execution drift, not checkpoint/model math fidelity.
 * No target generation, invented hidden values, mode forcing, or cache clearing.
 * Required properties: mtp.reference.gguf and qwen.mtp.snapshotPrefix.
 */
@Slf4j
public class TestGgufMtpCapturedReplay {
    /** Exact small-integer oracle for layout preservation through fused casts/matmul. */
    @Test
    void fusedCastMatmulPreservesWeightLayout() {
        BenchmarkConfig config = BenchmarkConfig.optimal();
        config.tritonIncludeTypes(config.getTritonIncludeTypes() + ",MATMUL");
        BenchmarkConfigApplier.apply(config);
        List<String> failures = new ArrayList<>();
        for (char order : new char[]{'c', 'f'}) {
            DspDiagnostics.clear(); // Diagnostic events only; never execution caches.
            SameDiff graph = SameDiff.create();
            try (INDArray weights = Nd4j.create(DataType.HALF, new long[]{3, 2}, order);
                 INDArray input = Nd4j.create(DataType.HALF, 1, 3)) {
                try {
                for (int row = 0; row < 3; row++) {
                    for (int col = 0; col < 2; col++) weights.putScalar(new long[]{row, col}, 2 * row + col + 1);
                }
                var x = graph.placeHolder("x", DataType.HALF, 1, 3);
                var w = graph.constant("w", weights);
                x.castTo(DataType.FLOAT).mmul(w.castTo(DataType.FLOAT)).rename("out");
                graph.setOutputs("out");
                for (int step = 1; step <= 8; step++) {
                    for (int col = 0; col < 3; col++) input.putScalar(new long[]{0, col}, step * (col + 1));
                    INDArray out = graph.output(Map.of("x", input), "out").get("out");
                    float[] values = floats(out);
                    log.info("CAST_MATMUL_LAYOUT order={} step={} actual={} expected=[{},{}]", order, step,
                            Arrays.toString(values), 22 * step, 28 * step);
                    if (values.length != 2 || values[0] != 22 * step || values[1] != 28 * step) {
                        failures.add("order=" + order + " step=" + step + " actual=" + Arrays.toString(values));
                    }
                }
                var executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
                Pointer handle = executor.getNativePlanHandle();
                NativeOps ops = Nd4j.getNativeOps();
                String audit = ops.getPlanSegmentCompilationAudit(handle, 0);
                log.info("CAST_MATMUL_COVERAGE order={} audit={}", order, audit);
                String diagnostics = DspDiagnostics.getJsonReport();
                assertTrue(diagnostics != null && diagnostics.contains("range[0-2] DONE status=OK"),
                        "Require successful compilation of the combined cast/matmul range (COMPILE diagnostics required): " + audit);
                assertEquals("Triton GPU", ops.getPlanSegmentCompiledBackend(handle, 0), audit);
                assertEquals(0, ops.getPlanSegmentGapSlotCount(handle, 0),
                        "No native gap may substitute for fused matmul: " + audit);
                } finally {
                    graph.close();
                }
            }
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    @Test
    void fusedIndependentCastsPreserveUnequalMultiBlockDomains() {
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        long[][] shapes = {{1, 257}, {17, 33}, {1, 4097}};
        String[] outputs = {"y0", "y1", "y2"};
        Map<String, INDArray> feeds = new LinkedHashMap<>();
        SameDiff graph = SameDiff.create();
        try {
            for (int i = 0; i < shapes.length; i++) {
                feeds.put("x" + i, Nd4j.create(DataType.HALF, shapes[i], i == 1 ? 'f' : 'c'));
                graph.placeHolder("x" + i, DataType.HALF, shapes[i]).castTo(DataType.FLOAT).rename(outputs[i]);
            }
            graph.setOutputs(outputs);
            for (int step = 1; step <= 8; step++) {
                Map<String, float[]> expected = new LinkedHashMap<>();
                for (int i = 0; i < shapes.length; i++) {
                    INDArray input = feeds.get("x" + i);
                    float[] values = new float[Math.toIntExact(input.length())];
                    for (int j = 0; j < values.length; j++) values[j] = j % 31 - 15 + step;
                    expected.put(outputs[i], values);
                    try (INDArray source = Nd4j.createFromArray(values).reshape(shapes[i])) {
                        input.assign(source);
                    }
                }
                Map<String, INDArray> actual = graph.output(feeds, outputs);
                for (String name : outputs) {
                    assertArrayEquals(expected.get(name), floats(actual.get(name)),
                            "independent cast " + name + " step=" + step);
                }
            }
        } finally {
            graph.close();
            for (INDArray input : feeds.values()) if (!input.wasClosed()) input.close();
        }
    }

    @Test
    void fusedSharedInputsPreserveBroadcastDomains() {
        assertBroadcastDomains(false);
    }

    @Test
    void fusedSharedProducerPreservesOwnAndBroadcastDomains() {
        assertBroadcastDomains(true);
    }

    private static void assertBroadcastDomains(boolean internalProducer) {
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        try (INDArray input = Nd4j.create(DataType.HALF, 2, 1);
             SameDiff graph = SameDiff.create()) {
            var x = graph.placeHolder("x", DataType.HALF, 2, 1);
            var u = internalProducer ? x.castTo(DataType.FLOAT).add(1.0) : x;
            graph.identity("own", u);
            u.add(graph.constant("leftBias", Nd4j.zeros(u.dataType(), 2, 3))).rename("left");
            u.add(graph.constant("rightBias", Nd4j.zeros(u.dataType(), 3, 2, 1))).rename("right");
            graph.setOutputs("own", "left", "right");
            List<String> failures = new ArrayList<>();
            for (int step = 1; step <= 8; step++) {
                input.putScalar(new long[]{0, 0}, 10 * step);
                input.putScalar(new long[]{1, 0}, 20 * step);
                float a = 10 * step + (internalProducer ? 1 : 0);
                float b = 20 * step + (internalProducer ? 1 : 0);
                Map<String, float[]> expected = Map.of("own", new float[]{a, b},
                        "left", new float[]{a, a, a, b, b, b},
                        "right", new float[]{a, b, a, b, a, b});
                Map<String, INDArray> out = graph.output(Map.of("x", input), "own", "left", "right");
                for (String name : expected.keySet()) {
                    float[] observed = floats(out.get(name));
                    log.info("BROADCAST_DOMAIN internal={} step={} output={} values={}",
                            internalProducer, step, name, Arrays.toString(observed));
                    if (!Arrays.equals(expected.get(name), observed)) failures.add("internal=" + internalProducer
                            + " step=" + step + " output=" + name + " actual=" + Arrays.toString(observed));
                }
            }
            assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
        }
    }

    /** Precompute before pipeline initialization so references cannot repair shared
     * execution state between production predictor calls. */
    public static final class PreparedReference {
        private final List<DspTensorSnapshot> captures = new ArrayList<>();
        private final List<Map<String, float[]>> references = new ArrayList<>();

        public PreparedReference(String file, String prefix) throws Exception {
            assertNotNull(file);
            assertNotNull(prefix);
            for (int call = 1; call <= 3; call++) {
                DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
                captures.add(capture);
                references.add(freshReference(file, capture));
            }
        }

        public void verifyAfterPrefill(String file, Map<String, INDArray> productionPrefill) throws Exception {
            Map<String, INDArray> prefill = new LinkedHashMap<>();
            Map<String, INDArray> scalar = captures.get(0).reconstructInputs();
            SameDiff raw = null, optimized = null;
            try {
                if (Boolean.getBoolean("mtp.reference.scalarOutputSwitch")) {
                    prefill.putAll(captures.get(0).reconstructInputs());
                    log.info("MTP_PREPARED isolation: identical scalar input shapes, K/V outputs then logits/hidden");
                } else {
                    for (Map.Entry<String, INDArray> entry : productionPrefill.entrySet()) {
                        prefill.put(entry.getKey(), entry.getValue().dup());
                    }
                }
                raw = GGMLModelImport.importModel(file);
                optimized = GraphOptimizer.optimize(raw, new ArrayList<>(raw.outputs()), GraphOptimizer.defaultOptimizations());
                if (!Boolean.getBoolean("mtp.reference.scalarOnlyAfterPipeline")) {
                    optimized.output(prefill, "mtp_key_states", "mtp_value_states");
                }
                log.info("MTP_PREPARED scalarOnlyAfterPipeline={}", Boolean.getBoolean("mtp.reference.scalarOnlyAfterPipeline"));
                Map<String, INDArray> warm = optimized.output(scalar, "mtp_logits", "mtp_hidden_states");
                warm.get("mtp_logits").getDouble(0);
                var executor = optimized.getOrCreateSession().getDynamicShapePlanExecutor();
                executor.setMaxKvCacheLength((int) scalar.get("mtp_past_key_values.0.key").size(1));
                executor.configureMaxAllocationForKvCache(warm);
                executor.setShapesFrozen(true);
                log.info("MTP_PREPARED independent graph: predictor prefill then scalar output-set switch");
                try (var binding = executor.captureNativeExecutionBinding()) {
                    verify(binding);
                }
            } finally {
                if (optimized != null && optimized != raw) optimized.close();
                if (raw != null) raw.close();
                for (INDArray a : scalar.values()) if (!a.wasClosed()) a.close();
                for (INDArray a : prefill.values()) if (!a.wasClosed()) a.close();
            }
        }

        public void verify(NativeExecutionBinding binding) {
            Map<String, INDArray> retained = new LinkedHashMap<>();
            INDArray[] borrowed = binding.getExternalInputsSnapshot();
            for (String name : DspTensorSnapshot.MTP_INPUTS) {
                int index = binding.findExternalInputIndex(name);
                assertTrue(index >= 0, name);
                retained.put(name, borrowed[index]);
                binding.getBackendOwner().nativeOps().markPlanExternalInputVariable(binding.getPlanHandle(), index);
            }
            List<String> failures = new ArrayList<>();
            for (int call = 0; call < captures.size(); call++) {
                restore(captures.get(call), retained);
                Map<String, float[]> actual = executeBound(binding);
                for (String name : DspTensorSnapshot.MTP_INPUTS) {
                    if (name.contains("past_key_values")) actual.put(name, floats(retained.get(name)));
                }
                Map<String, float[]> expected = references.get(call);
                log.info("MTP_PREPARED call={} freshDraft={} boundDraft={}", call + 1,
                        argmax(expected.get("mtp_logits")), argmax(actual.get("mtp_logits")));
                for (String name : expected.keySet()) {
                    int first = Arrays.mismatch(expected.get(name), actual.get(name));
                    log.info("MTP_PREPARED call={} tensor={} firstMismatch={}", call + 1, name, first);
                    if (first >= 0) failures.add("call=" + (call + 1) + "/" + name + " firstMismatch=" + first);
                }
            }
            assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
            // All inputs belong to the prepared session, never this verifier.
        }
    }

    @Test
    void consecutiveBoundCallsMatchPrecomputedReferences() throws Exception {
        String file = System.getProperty("mtp.reference.gguf");
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        assertNotNull(file);
        assertNotNull(prefix);
        List<DspTensorSnapshot> captures = new ArrayList<>();
        List<Map<String, float[]>> references = new ArrayList<>();
        for (int call = 1; call <= 3; call++) {
            DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
            captures.add(capture);
            references.add(freshReference(file, capture));
        }
        if (Boolean.getBoolean("mtp.reference.optimal")) {
            BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        }
        if (Boolean.getBoolean("mtp.reference.fusionOnly")) {
            Nd4j.getEnvironment().setTritonSectionFusion(true);
            log.info("MTP_CONFIG changed sectionFusion only to true");
        }
        if (Boolean.getBoolean("mtp.reference.compileAll")) {
            Nd4j.getEnvironment().setTritonCompileAll(true);
            log.info("MTP_CONFIG changed compileAll to true");
        }
        if (Boolean.getBoolean("mtp.reference.unscoredFusion")) {
            Nd4j.getEnvironment().setTritonFusionScoring(false);
            log.info("MTP_CONFIG fusion scoring=false (OPTIMAL aggressive fusion policy)");
        }
        // No reference graph execution or destruction between retained calls.
        Map<String, INDArray> retained = captures.get(0).reconstructInputs();
        SameDiff raw = null, reused = null;
        NativeExecutionBinding binding = null;
        List<String> failures = new ArrayList<>();
        try {
            raw = GGMLModelImport.importModel(file);
            reused = GraphOptimizer.optimize(raw, new ArrayList<>(raw.outputs()), GraphOptimizer.defaultOptimizations());
            Map<String, INDArray> warm = reused.output(retained, "mtp_logits", "mtp_hidden_states");
            warm.get("mtp_logits").getDouble(0);
            var executor = reused.getOrCreateSession().getDynamicShapePlanExecutor();
            executor.setMaxKvCacheLength((int) retained.get("mtp_past_key_values.0.key").size(1));
            executor.configureMaxAllocationForKvCache(warm);
            executor.setShapesFrozen(true);
            binding = executor.captureNativeExecutionBinding();
            INDArray[] borrowed = binding.getExternalInputsSnapshot();
            for (String name : DspTensorSnapshot.MTP_INPUTS) {
                int index = binding.findExternalInputIndex(name);
                assertTrue(index >= 0, name);
                assertSame(retained.get(name), borrowed[index], name);
                binding.getBackendOwner().nativeOps().markPlanExternalInputVariable(binding.getPlanHandle(), index);
            }
            for (int call = 0; call < captures.size(); call++) {
                restore(captures.get(call), retained);
                Map<String, float[]> actual = executeBound(binding);
                for (String name : DspTensorSnapshot.MTP_INPUTS) {
                    if (name.contains("past_key_values")) actual.put(name, floats(retained.get(name)));
                }
                Map<String, float[]> expected = references.get(call);
                log.info("MTP_CONSECUTIVE call={} freshDraft={} boundDraft={}", call + 1,
                        argmax(expected.get("mtp_logits")), argmax(actual.get("mtp_logits")));
                for (String name : expected.keySet()) {
                    int first = Arrays.mismatch(expected.get(name), actual.get(name));
                    double maxAbs = 0;
                    float[] a = expected.get(name), b = actual.get(name);
                    assertEquals(a.length, b.length, name);
                    for (int i = 0; i < a.length; i++) maxAbs = Math.max(maxAbs, Math.abs((double) a[i] - b[i]));
                    log.info("MTP_CONSECUTIVE call={} tensor={} firstMismatch={} maxAbs={}", call + 1, name, first, maxAbs);
                    if (first >= 0) failures.add("call=" + (call + 1) + "/" + name + " firstMismatch=" + first);
                }
            }
        } finally {
            if (binding != null) binding.close();
            if (reused != null && reused != raw) reused.close();
            if (raw != null) raw.close();
            for (INDArray a : retained.values()) if (!a.wasClosed()) a.close();
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    private static Map<String, float[]> freshReference(String file, DspTensorSnapshot capture) throws Exception {
        Map<String, INDArray> inputs = capture.reconstructInputs();
        SameDiff raw = null, optimized = null;
        try {
            raw = GGMLModelImport.importModel(file);
            optimized = GraphOptimizer.optimize(raw, new ArrayList<>(raw.outputs()), GraphOptimizer.defaultOptimizations());
            Map<String, INDArray> output = optimized.output(inputs, "mtp_logits", "mtp_hidden_states");
            Map<String, float[]> result = new LinkedHashMap<>();
            for (String name : output.keySet()) result.put(name, floats(output.get(name)));
            for (String name : DspTensorSnapshot.MTP_INPUTS) {
                if (name.contains("past_key_values")) result.put(name, floats(inputs.get(name)));
            }
            return result;
        } finally {
            if (optimized != null && optimized != raw) optimized.close();
            if (raw != null) raw.close();
            for (INDArray a : inputs.values()) if (!a.wasClosed()) a.close();
        }
    }

    /** Artifact-only check: no model execution or recapture. Post-execution staging
     * of read-only inputs must still equal their pre-execution live values. */
    @Test
    void capturedReadOnlyStagingMatchesLiveInputs() throws Exception {
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        assertNotNull(prefix, "Provide qwen.mtp.snapshotPrefix");
        List<String> failures = new ArrayList<>();
        int checked = 0;
        for (int call = 1; call <= 3; call++) {
            DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
            for (String name : DspTensorSnapshot.MTP_INPUTS) {
                if (name.contains("past_key_values")) continue; // KV is intentionally mutable.
                DspTensorSnapshot.Tensor staging = capture.tensors.get("staging/" + name);
                if (staging == null) {
                    log.info("MTP_STAGING call={} input={} no captured staging", call, name);
                    continue;
                }
                DspTensorSnapshot.Tensor live = capture.tensors.get(name);
                assertEquals(live.dtype, staging.dtype, name);
                assertArrayEquals(live.shape, staging.shape, name);
                int first = Arrays.mismatch(live.raw, staging.raw);
                log.info("MTP_STAGING call={} input={} firstDifferentByte={}", call, name, first);
                checked++;
                if (first >= 0) failures.add("call=" + call + "/" + name + " firstDifferentByte=" + first);
            }
        }
        assertTrue(checked > 0, "No staging tensors captured; cannot establish staging parity");
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    @Test
    void advancingInputsMatchFreshPredictor() throws Exception {
        String file = System.getProperty("mtp.reference.gguf");
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        assertNotNull(file);
        assertNotNull(prefix);
        DspTensorSnapshot first = DspTensorSnapshot.read(Path.of(prefix + ".tensor-1.dspt"));
        Map<String, INDArray> retained = first.reconstructInputs();
        List<String> failures = new ArrayList<>();
        SameDiff raw = null, reused = null;
        NativeExecutionBinding binding = null;
        try {
            raw = GGMLModelImport.importModel(file);
            reused = GraphOptimizer.optimize(raw, new ArrayList<>(raw.outputs()), GraphOptimizer.defaultOptimizations());
            boolean productionHandoff = Boolean.getBoolean("mtp.reference.productionHandoff");
            for (int warmup = 0; warmup < (productionHandoff ? 1 : 6); warmup++) {
                restore(first, retained);
                Map<String, INDArray> warmOutputs = reused.output(retained, "mtp_logits", "mtp_hidden_states");
                if (productionHandoff) {
                    // Reproduce GenerationPipeline.prepareBundledMtp's explicit
                    // post-scalar-warmup handoff; do not force an execution mode.
                    warmOutputs.get("mtp_logits").getDouble(0);
                    var executor = reused.getOrCreateSession().getDynamicShapePlanExecutor();
                    executor.setMaxKvCacheLength((int) retained.get("mtp_past_key_values.0.key").size(1));
                    executor.configureMaxAllocationForKvCache(warmOutputs);
                    executor.setShapesFrozen(true);
                }
            }
            if (Boolean.getBoolean("mtp.reference.directBinding")) {
                binding = reused.getOrCreateSession().getDynamicShapePlanExecutor().captureNativeExecutionBinding();
                INDArray[] borrowed = binding.getExternalInputsSnapshot();
                for (String name : DspTensorSnapshot.MTP_INPUTS) {
                    int index = binding.findExternalInputIndex(name);
                    assertTrue(index >= 0, name);
                    assertSame(retained.get(name), borrowed[index], "Update the binding's actual retained input: " + name);
                    if (Boolean.getBoolean("mtp.reference.markVariable")) {
                        binding.getBackendOwner().nativeOps().markPlanExternalInputVariable(binding.getPlanHandle(), index);
                    }
                }
            }
            for (int call = 1; call <= 3; call++) {
                DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
                Map<String, INDArray> independent = capture.reconstructInputs();
                SameDiff freshRaw = null, fresh = null;
                try {
                    freshRaw = GGMLModelImport.importModel(file);
                    fresh = GraphOptimizer.optimize(freshRaw, new ArrayList<>(freshRaw.outputs()), GraphOptimizer.defaultOptimizations());
                    Map<String, INDArray> expected = fresh.output(independent, "mtp_logits", "mtp_hidden_states");
                    float[] logits = floats(expected.get("mtp_logits"));
                    float[] hidden = floats(expected.get("mtp_hidden_states"));
                    restore(capture, retained);
                    float[] actualLogits, actualHidden;
                    if (binding != null) {
                        Map<String, float[]> copied = executeBound(binding);
                        actualLogits = copied.get("mtp_logits");
                        actualHidden = copied.get("mtp_hidden_states");
                        log.info("MTP_ADVANCE call={} entry=retainedNativeBinding", call);
                    } else {
                    Map<String, INDArray> actual;
                    if (Boolean.getBoolean("mtp.reference.nativeSteady")) {
                        var executor = reused.getOrCreateSession().getDynamicShapePlanExecutor();
                        actual = executor.executeSteadyState(executor.getCurrentPlan(), retained);
                        log.info("MTP_ADVANCE call={} entry=executeSteadyState", call);
                    } else {
                        actual = reused.output(retained, "mtp_logits", "mtp_hidden_states");
                    }
                    actualLogits = floats(actual.get("mtp_logits"));
                    actualHidden = floats(actual.get("mtp_hidden_states"));
                    }
                    int logitMismatch = Arrays.mismatch(logits, actualLogits);
                    int hiddenMismatch = Arrays.mismatch(hidden, actualHidden);
                    log.info("MTP_ADVANCE call={} freshDraft={} reusedDraft={} firstLogit={} firstHidden={}",
                            call, argmax(logits), argmax(actualLogits), logitMismatch, hiddenMismatch);
                    if (logitMismatch >= 0 || hiddenMismatch >= 0) failures.add("call=" + call + " advanced-input drift");
                    for (String name : DspTensorSnapshot.MTP_INPUTS) {
                        if (!name.contains("past_key_values")) continue;
                        int mismatch = Arrays.mismatch(floats(independent.get(name)), floats(retained.get(name)));
                        log.info("MTP_ADVANCE call={} cache={} firstMismatch={}", call, name, mismatch);
                        if (mismatch >= 0) failures.add("call=" + call + "/" + name + " drift");
                    }
                } finally {
                    if (fresh != null && fresh != freshRaw) fresh.close();
                    if (freshRaw != null) freshRaw.close();
                    for (INDArray a : independent.values()) if (!a.wasClosed()) a.close();
                }
            }
        } finally {
            if (binding != null) binding.close();
            if (reused != null && reused != raw) reused.close();
            if (raw != null) raw.close();
            for (INDArray a : retained.values()) if (!a.wasClosed()) a.close();
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    /** Same owned-output readback pattern as DspMultiPlanShapeSwitchTest.
     * Completion here is the test's readback/lease boundary, not a production fix. */
    private static Map<String, float[]> executeBound(NativeExecutionBinding binding) {
        NativeOps ops = binding.getBackendOwner().nativeOps();
        Pointer stream = ops.dspGetExecutionStream(binding.getPlanHandle());
        Map<String, float[]> copied = new LinkedHashMap<>();
        binding.beginNativeUse();
        try {
            int status = ops.executeSteadyStatePlan(binding.getPlanHandle(), binding.getContextHandle(), stream);
            ops.streamSynchronize(stream);
            assertEquals(0, ops.lastErrorCode(), ops.lastErrorMessage());
            assertEquals(0, status, ops.lastErrorMessage());
            for (String name : binding.getRequestedOutputs()) {
                OpaqueNDArray output = ops.getOutputArrayNative(binding.getContextHandle(), binding.findOutputIndex(name));
                assertNotNull(output);
                assertFalse(output.isNull());
                output.attachOwner(binding.getBackendOwner());
                long[] info = output.shapeInfo();
                DataType dtype = ArrayOptionsHelper.dataType(Shape.extras(info));
                Pointer special = ops.getOpaqueNDArraySpecialBuffer(output);
                Pointer primary = special == null || special.isNull() ? ops.getOpaqueNDArrayBuffer(output) : null;
                OpaqueDataBuffer source = ops.dbCreateExternalDataBuffer(output.length(), dtype.toInt(), primary, special);
                assertNotNull(source);
                assertFalse(source.isNull());
                try (INDArray copy = Nd4j.createUninitialized(dtype, Shape.shape(info), Shape.stride(info), Shape.order(info))) {
                    try {
                        ops.copyBuffer(copy.data().opaqueBuffer(), output.length(), source, 0, 0);
                        Nd4j.getExecutioner().commit();
                        copied.put(name, floats(copy));
                    } finally {
                        Nd4j.getExecutioner().commit();
                        ops.deleteDataBuffer(source);
                    }
                }
                // output is borrowed from the context; never delete it.
            }
        } finally {
            ops.streamSynchronize(stream);
            assertEquals(0, ops.lastErrorCode(), ops.lastErrorMessage());
            Nd4j.getExecutioner().commit();
            binding.completeNativeUse();
        }
        return copied;
    }

    private static void restore(DspTensorSnapshot capture, Map<String, INDArray> inputs) {
        for (String name : DspTensorSnapshot.MTP_INPUTS) {
            if (Boolean.getBoolean("mtp.reference.deviceAssign")) {
                try (INDArray source = capture.tensors.get(name).reconstruct(capture.payloadOrder)) {
                    inputs.get(name).assign(source);
                }
            } else {
                capture.tensors.get(name).restore(inputs.get(name), capture.payloadOrder);
            }
        }
    }

    @Test
    void capturedInputsMatchFreshPredictor() throws Exception {
        String file = System.getProperty("mtp.reference.gguf");
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        assertNotNull(file, "Provide the exact captured model via mtp.reference.gguf");
        assertNotNull(prefix, "Provide qwen.mtp.snapshotPrefix");
        assertTrue(Files.isRegularFile(Path.of(file)), file);
        List<String> failures = new ArrayList<>();
        for (int call = 1; call <= 3; call++) {
            DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
            assertEquals(call, capture.callIndex);
            Map<String, INDArray> inputs = capture.reconstructInputs();
            SameDiff raw = null;
            SameDiff optimized = null;
            try {
                raw = GGMLModelImport.importModel(file);
                List<String> outputs = new ArrayList<>(raw.outputs());
                optimized = GraphOptimizer.optimize(raw, outputs, GraphOptimizer.defaultOptimizations());
                Map<String, INDArray> observed = optimized.output(inputs, "mtp_logits", "mtp_hidden_states");
                String label = "call=" + call + "/position=" + capture.sourcePosition;
                compare(capture, DspTensorSnapshot.LOGITS, observed.get("mtp_logits"), label, failures);
                compare(capture, DspTensorSnapshot.HIDDEN, observed.get("mtp_hidden_states"), label, failures);
                if (Boolean.getBoolean("mtp.reference.repeat")) {
                    float[] freshLogits = floats(observed.get("mtp_logits"));
                    float[] freshHidden = floats(observed.get("mtp_hidden_states"));
                    // Identical complete inputs at stable addresses through the natural
                    // freeze/replay lifecycle isolate plan reuse from target/controller work.
                    for (int repeat = 1; repeat <= 6; repeat++) {
                        for (String name : DspTensorSnapshot.MTP_INPUTS) {
                            capture.tensors.get(name).restore(inputs.get(name), capture.payloadOrder);
                        }
                        observed = optimized.output(inputs, "mtp_logits", "mtp_hidden_states");
                        float[] repeatedLogits = floats(observed.get("mtp_logits"));
                        float[] repeatedHidden = floats(observed.get("mtp_hidden_states"));
                        int logitMismatch = Arrays.mismatch(freshLogits, repeatedLogits);
                        int hiddenMismatch = Arrays.mismatch(freshHidden, repeatedHidden);
                        log.info("MTP_REPEAT {} repeat={} freshDraft={} repeatedDraft={} firstLogit={} firstHidden={}",
                                label, repeat, argmax(freshLogits), argmax(repeatedLogits), logitMismatch, hiddenMismatch);
                        if (logitMismatch >= 0 || hiddenMismatch >= 0) failures.add(label
                                + " repeat=" + repeat + " changed identical-input result");
                    }
                }
                for (String name : DspTensorSnapshot.MTP_INPUTS) {
                    String expectedName = name.contains("past_key_values") ? "postexec/" + name : name;
                    compare(capture, expectedName, inputs.get(name), label, failures);
                }
                try (INDArray draft = capture.tensors.get(DspTensorSnapshot.DRAFT).reconstruct(capture.payloadOrder)) {
                    int freshToken = argmax(floats(observed.get("mtp_logits")));
                    log.info("MTP_GGUF_REFERENCE {} nativeDraft={} freshDraft={}", label, draft.getLong(0), freshToken);
                    if (draft.getLong(0) != freshToken) failures.add(label + " draft mismatch native="
                            + draft.getLong(0) + " fresh=" + freshToken);
                }
            } finally {
                // Outputs belong to the session; close graphs before their independent feeds.
                if (optimized != null && optimized != raw) optimized.close();
                if (raw != null) raw.close();
                for (INDArray input : inputs.values()) if (!input.wasClosed()) input.close();
            }
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    private static void compare(DspTensorSnapshot capture, String name, INDArray actual,
                                String label, List<String> failures) {
        assertNotNull(actual, name);
        assertNotNull(capture.tensors.get(name), "Missing captured tensor " + name);
        try (INDArray expected = capture.tensors.get(name).reconstruct(capture.payloadOrder)) {
            assertEquals(expected.dataType(), actual.dataType(), label + "/" + name);
            assertArrayEquals(expected.shape(), actual.shape(), label + "/" + name);
            float[] a = floats(expected), b = floats(actual);
            int first = -1, mismatches = 0;
            double maxAbs = 0;
            // Same weights and storage dtype: exact comparison is a drift discriminator.
            // Numerical differences are reported, never silently accepted as a quality fix.
            for (int i = 0; i < a.length; i++) {
                if (!Float.isFinite(a[i]) || !Float.isFinite(b[i])
                        || Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                    if (first < 0) first = i;
                    mismatches++;
                    maxAbs = Math.max(maxAbs, Math.abs((double) a[i] - b[i]));
                }
            }
            log.info("MTP_GGUF_REFERENCE {} tensor={} dtype={} mismatches={}/{} first={} maxAbs={}",
                    label, name, expected.dataType(), mismatches, a.length, first, maxAbs);
            if (first >= 0) failures.add(label + "/" + name + " first=" + first
                    + " native=" + a[first] + " fresh=" + b[first] + " maxAbs=" + maxAbs);
        }
    }

    private static float[] floats(INDArray array) {
        try (INDArray copy = array.dup('c')) {
            if (copy.dataType() == DataType.FLOAT) return copy.data().asFloat().clone();
            try (INDArray converted = copy.castTo(DataType.FLOAT)) {
                return converted.data().asFloat().clone();
            }
        }
    }

    private static int argmax(float[] values) {
        assertTrue(values.length > 0);
        int best = 0;
        for (int i = 0; i < values.length; i++) {
            assertTrue(Float.isFinite(values[i]), "Nonfinite fresh logits at " + i);
            if (values[i] > values[best]) best = i;
        }
        return best;
    }
}
