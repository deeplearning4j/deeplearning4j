/* SPDX-License-Identifier: Apache-2.0 */
package org.eclipse.deeplearning4j.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;
import org.eclipse.deeplearning4j.safetensors.ModelOptQwenConfig;
import org.eclipse.deeplearning4j.safetensors.SafeTensorsReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.ggml.architecture.ArchitectureConfig;
import org.nd4j.ggml.architecture.LLaMAArchitecture;
import org.nd4j.ggml.architecture.QuantizedLinear;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

/** Cached-predictor steady-state probes for the native MTP path (no target layers, no downloads).
 * Reference values are a known-good production run's first warmup — not a math oracle. */
@Slf4j
public class TestQwenMtpPredictorLifecycle {
    private static final String PREFIX = "nvidia-Qwen3.6-27B-NVFP4-0893e1606ff3d5f97a441f405d5fc541a6bdf404-";
    private static final String HIDDEN = "mtp_hidden_states";
    private static final String KEY = "mtp_past_key_values.0.key", VALUE = "mtp_past_key_values.0.value";
    private static final int CAPACITY = 16, CALLS = 12;

    /** Fail before loading weights if the installed backend cannot enter the native MTP path.
     * An inherited default is unsupported; older defaults silently called ordinary execution.
     * This is a binding prerequisite only, not an actual-predictor numerical regression.
     */
    @Test void actualPredictorSteadyStateBindingPrerequisite() throws Exception {
        assertTrue(Nd4j.backends().isCudaAvailable(), "CUDA binding prerequisite");
        NativeOps nativeOps = Nd4j.getNativeOps();
        var method = nativeOps.getClass().getMethod("executeSteadyStatePlan",
                Pointer.class, OpaqueContext.class, Pointer.class);
        log.info("MTP_STEADY_BINDING backend={} declaration={} default={}",
                nativeOps.getClass().getName(), method.getDeclaringClass().getName(), method.isDefault());
        assertFalse(method.isDefault(),
                "BINDING BLOCKER: " + nativeOps.getClass().getName()
                        + " inherits " + method.getDeclaringClass().getName()
                        + ".executeSteadyStatePlan rather than a real native override. "
                        + "Expose a native wrapper calling NativeDynamicShapePlan::executeSteadyState "
                        + "with mapped context inputs/requested outputs and the supplied stream; "
                        + "ordinary execute is not evidence for the actual MTP fast path.");
        assertTrue(Modifier.isNative(method.getModifiers()), "JavaCPP native override required");
        try {
            assertEquals(1, nativeOps.executeSteadyStatePlan(null, null, null),
                    "Resolve the actual JNI symbol before loading checkpoint weights");
            assertTrue(nativeOps.lastErrorMessage().contains("null plan handle"));
        } finally {
            nativeOps.clearLastError();
        }
    }

    @Test void actualPredictorNaturalLifecycle() throws Exception {
        assertTrue(Nd4j.backends().isCudaAvailable(), "CUDA probe, not a skipped CPU result");
        List<String> failures = new ArrayList<>();
        int mask = Nd4j.getNativeOps().dspDiagGetEnabledMask();
        int level = Nd4j.getNativeOps().dspDiagGetLevel();
        List<SameDiff> graphs = new ArrayList<>();
        try (Weights weights = new Weights()) {
            DspDiagnostics.setCategories(DspDiagnostics.GRAPH_REPLAY | DspDiagnostics.VERIFY);
            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
            try {
                for (boolean diagnostic : new boolean[]{false, true}) run(weights, diagnostic, false, failures, graphs);
                weights.loadHead();
                run(weights, false, true, failures, graphs);
            } finally {
                for (int i = graphs.size() - 1; i >= 0; i--) graphs.get(i).close();
            }
        } finally {
            DspDiagnostics.setCategories(mask);
            DspDiagnostics.setLevel(level);
        }
        assertTrue(failures.isEmpty(), () -> "First drift: " + failures.get(0) + "\n" + String.join("\n", failures));
    }

    /** Cache-only actual-input replay. Run with qwen.nvfp4.windowParity=true and
     * qwen.mtp.snapshotPrefix=/path/to/capture-prefix. This compares the same two
     * production outputs, plus in-place KV commits, against a fresh graph for EACH
     * captured call. It does not infer actual hidden state from chain-probe samples.
     */
    @Test
    @EnabledIfSystemProperty(named = "qwen.mtp.snapshotPrefix", matches = ".+")
    void capturedPredictorInputsAgainstFreshGraph() throws Exception {
        assertTrue(Nd4j.backends().isCudaAvailable(), "CUDA replay probe");
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        List<DspTensorSnapshot> captures = new ArrayList<>();
        long total = 0;
        for (int call = 1; call <= 3; call++) {
            DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
            assertEquals(call, capture.callIndex);
            assertTrue(capture.source.equals("executeMtpCuda/pre-executeSteadyState")
                    || capture.source.startsWith("executeMtpCuda/pre-executeSteadyState;stream="), capture.source);
            for (DspTensorSnapshot.Tensor t : capture.tensors.values()) total += t.raw.length;
            assertTrue(total <= DspTensorSnapshot.BUDGET, "Whole capture budget");
            captures.add(capture);
        }
        List<String> failures = new ArrayList<>();
        List<SameDiff> graphs = new ArrayList<>();
        try (Weights weights = new Weights()) {
            weights.loadHead(); // Actual cached packed head, never a fabricated head.
            try {
                SameDiff probe = graph(weights, false, true, graphs);
                String[] outputs = {"mtp_logits", HIDDEN};
                try (Inputs actual = new Inputs(captures.get(0))) {
                    // Natural lifecycle; no forced mode, frozen flag, or cache clearing.
                    // Repeat identical full inputs until the bounded replay gate.
                    for (int i = 0; i < 6; i++) {
                        actual.restore(captures.get(0));
                        evaluate(probe, actual, outputs, "artifact/warmup/" + i);
                    }
                    lifecycleGate(probe, "artifact/warmup", failures);
                    auditProduction(probe, "artifact/warmup", failures);
                    if (Boolean.parseBoolean(System.getProperty("qwen.mtp.skipProbe", "false"))) {
                        // Skip-probe: production warms ONCE (prepareBundledMtp) then advances
                        // positions 37->38->39 through DIFFERENT prefixes. Every earlier probe
                        // warmed by replaying capture-1 repeatedly — numerically identical to a
                        // warmup-pinned constant. Here, warmed on pos 37 only, feed pos 39
                        // DIRECTLY (never pos 38): wrong output = a warmup-pinned quantity
                        // breaks when the prefix advances past the warmup length.
                        DspTensorSnapshot capture = captures.get(2);
                        SameDiff fresh = graph(weights, false, true, graphs);
                        try (Inputs independent = new Inputs(capture)) {
                            actual.restore(capture);
                            Map<String, float[]> expected = evaluate(fresh, independent, outputs, "skip/fresh");
                            Map<String, float[]> observed = evaluate(probe, actual, outputs, "skip/compiled");
                            float[] exp = expected.get("mtp_logits"), obs = observed.get("mtp_logits");
                            int eTok = 0, oTok = 0;
                            for (int j = 1; j < exp.length; j++) {
                                if (exp[j] > exp[eTok]) eTok = j;
                                if (obs[j] > obs[oTok]) oTok = j;
                            }
                            log.info("MTP_SKIP_PROBE pos={} freshArgmax={} warmedArgmax={}",
                                    capture.sourcePosition, eTok, oTok);
                            if (eTok != oTok)
                                failures.add("skip-probe fresh=" + eTok + " warmed=" + oTok
                                        + " — warmup-pinned plan constant breaks on prefix advance");
                        } finally { release(fresh, graphs); }
                    }
                    for (int captureIndex = 0; captureIndex < captures.size(); captureIndex++) {
                        DspTensorSnapshot capture = captures.get(captureIndex);
                        SameDiff fresh = graph(weights, false, true, graphs);
                        try (Inputs independent = new Inputs(capture)) {
                            actual.restore(capture);
                            assertNotEquals(independent.hidden.data().address(), actual.hidden.data().address());
                            assertNotEquals(independent.keys.data().address(), actual.keys.data().address());
                            assertEquals(capture.sourcePosition, actual.feeds.get("mtp_position_offset").getLong(0));
                            assertEquals(capture.sourcePosition, actual.feeds.get("mtp_cache_position").getLong(0));
                            String label = "artifact/call=" + capture.callIndex + "/position=" + capture.sourcePosition;
                            Map<String, float[]> expected = evaluate(fresh, independent, outputs, label + "/fresh");
                            assertEquals(0, DspPlanAssertions.getTotalGraphReplays(fresh));
                            assertTrue(DspPlanAssertions.getPlanPhase(fresh) < 2);
                            Map<String, float[]> observed = evaluate(probe, actual, outputs, label + "/compiled");
                            compareNativeCapture(capture, label, expected, observed, failures);
                            compare(label, expected, observed, failures);
                            float[] logits = expected.get("mtp_logits");
                            int token = 0;
                            for (int j = 1; j < logits.length; j++)
                                if (logits[j] > logits[token]) token = j;
                            log.info("MTP_CAPTURE_REPLAY {} freshArgmax={}", label, token);
                            if (captureIndex + 1 < captures.size()) {
                                try (Inputs next = new Inputs(captures.get(captureIndex + 1))) {
                                    compare(label + "/native-carry-to-next-call",
                                            Map.of(HIDDEN, expected.get(HIDDEN)),
                                            Map.of(HIDDEN, floats(next.hidden)), failures);
                                    long capturedToken = next.feeds.get("mtp_input_ids").getLong(0);
                                    if (capturedToken != token)
                                        failures.add(label + " native token=" + capturedToken + " fresh token=" + token);
                                }
                            }
                            lifecycleGate(probe, label, failures);
                        } finally { release(fresh, graphs); }
                    }
                    release(probe, graphs);
                }
            } finally {
                for (int i = graphs.size() - 1; i >= 0; i--) graphs.get(i).close();
            }
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    /** Replays the three complete native captures with production's scalar-warmup/freeze
     * handoff. A second pass requires the native FAST diagnostic on every call: the first
     * three calls may still be admitted to build phases by executeSteadyState itself.
     * No target layers, invented head, extra requested tensors or ordinary API substitution.
     */
    @Test
    @EnabledIfSystemProperty(named = "qwen.mtp.snapshotPrefix", matches = ".+")
    void capturedPredictorSteadyStateAgainstFreshGraph() throws Exception {
        actualPredictorSteadyStateBindingPrerequisite();
        String prefix = System.getProperty("qwen.mtp.snapshotPrefix");
        List<DspTensorSnapshot> captures = new ArrayList<>();
        long total = 0;
        for (int call = 1; call <= 3; call++) {
            DspTensorSnapshot capture = DspTensorSnapshot.read(Path.of(prefix + ".tensor-" + call + ".dspt"));
            assertEquals(call, capture.callIndex);
            assertTrue(capture.source.equals("executeMtpCuda/pre-executeSteadyState")
                    || capture.source.startsWith("executeMtpCuda/pre-executeSteadyState;stream="), capture.source);
            for (DspTensorSnapshot.Tensor tensor : capture.tensors.values()) total += tensor.raw.length;
            assertTrue(total <= DspTensorSnapshot.BUDGET, "Whole capture budget");
            captures.add(capture);
        }
        NativeOps nativeOps = Nd4j.getNativeOps();
        int mask = nativeOps.dspDiagGetEnabledMask(), level = nativeOps.dspDiagGetLevel();
        List<String> failures = new ArrayList<>();
        List<SameDiff> graphs = new ArrayList<>();
        try (Weights weights = new Weights()) {
            // EXECUTE records the existing native FAST/FALLBACK gate, without VERIFY's
            // op-sanity policy changing which execution path is selected.
            DspDiagnostics.setCategories(DspDiagnostics.EXECUTE);
            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_FULL);
            weights.loadHead();
            try {
                SameDiff probe = graph(weights, false, true, graphs);
                String[] outputs = {"mtp_logits", HIDDEN};
                try (Inputs actual = new Inputs(captures.get(0))) {
                    Map<String, INDArray> warmup = probe.output(actual.feeds, outputs);
                    snapshot(warmup, actual, outputs);
                    lifecycle(probe, "steady/scalar-warmup");
                    var executor = probe.getOrCreateSession().getDynamicShapePlanExecutor();
                    assertNotNull(executor);
                    executor.setMaxKvCacheLength((int) actual.keys.size(1));
                    executor.configureMaxAllocationForKvCache(warmup);
                    executor.setShapesFrozen(true); // Same handoff as GenerationPipeline.prepareBundledMtp.
                    long handle = executor.getNativePlanHandle().address();
                    assertEquals(outputs.length, executor.getCurrentPlan().getRequestedOutputs().size());
                    assertTrue(executor.getCurrentPlan().getRequestedOutputs().containsAll(Arrays.asList(outputs)));
                    assertTrue(Arrays.asList(executor.getCurrentPlan().getExternalInputKeys()).containsAll(actual.feeds.keySet()));
                    assertNotNull(executor.getCachedOpContext());
                    for (int pass = 0; pass < 2; pass++) {
                        if (pass == 1) {
                            // Replay admission is lifecycle-owned, not a fixed count of calls.
                            // Verify every additional warmup call rather than forcing a phase.
                            SameDiff reference = graph(weights, false, true, graphs);
                            try (Inputs independent = new Inputs(captures.get(0))) {
                                Map<String, float[]> expected = evaluate(reference, independent, outputs,
                                        "steady/admission-reference");
                                for (int attempt = 0; attempt < 16 && DspPlanAssertions.getPlanPhase(probe) < 2; attempt++) {
                                    actual.restore(captures.get(0));
                                    Map<String, float[]> observed = evaluate(probe, actual, outputs,
                                            "steady/admission/" + attempt, true, false);
                                    compare("steady/admission/" + attempt, expected, observed, failures);
                                }
                                lifecycleGate(probe, "steady/admission", failures);
                                assertEquals(2, DspPlanAssertions.getPlanPhase(probe),
                                        "Native lifecycle must admit replay within the bounded warmup");
                            } finally { release(reference, graphs); }
                        }
                        for (int i = 0; i < captures.size(); i++) {
                            DspTensorSnapshot capture = captures.get(i);
                            String label = "steady/" + (pass == 0 ? "production-handoff" : "fast-required")
                                    + "/call=" + capture.callIndex + "/position=" + capture.sourcePosition;
                            SameDiff fresh = graph(weights, false, true, graphs);
                            try (Inputs independent = new Inputs(capture)) {
                                actual.restore(capture);
                                // In-window clobber probe: AFTER the restore (staging of correct bytes),
                                // BEFORE the plan reads. Production has no restore at all between the
                                // last correct staging and the read — the target plan's writes land in
                                // exactly this window. Design notes: a UNIFORM shift to all K rows is
                                // attention-invariant (Q·δ constant across rows, softmax unchanged) —
                                // corrupt a SUBSET instead, and also corrupt V (V has no such
                                // invariance at all). Zeroing V rows [5,15) must change the output if
                                // the plan reads the live buffer on replay; unchanged output proves a
                                // frozen/disconnected KV substrate.
                                if (capture.callIndex == 3 && pass == 0 && Boolean.parseBoolean(
                                        System.getProperty("qwen.mtp.clobberProbe", "false"))) {
                                    int pos = (int) capture.sourcePosition;
                                    try (INDArray deadV = actual.values.get(
                                            all(), interval(5, Math.min(15, pos)),
                                            all(), all());
                                         INDArray noisyK = actual.keys.get(
                                            all(), interval(0, pos),
                                            all(), all())) {
                                        deadV.assign(0f);
                                        noisyK.addi(Nd4j.randn(DataType.FLOAT, noisyK.shape()).muli(0.7f));
                                    }
                                    Nd4j.getExecutioner().commit();
                                    log.info("MTP_CLOBBER_PROBE subset-V-zero + nonuniform-K-noise inside call=3 window position={}", pos);
                                }
                                assertNotEquals(independent.hidden.data().address(), actual.hidden.data().address());
                                assertNotEquals(independent.keys.data().address(), actual.keys.data().address());
                                Map<String, float[]> expected = evaluate(fresh, independent, outputs, label + "/fresh");
                                assertEquals(0, DspPlanAssertions.getTotalGraphReplays(fresh));
                                Map<String, float[]> observed = evaluate(probe, actual, outputs, label, true, pass == 1);
                                assertEquals(handle, executor.getNativePlanHandle().address(), "Retain the warmed native plan");
                                compareNativeCapture(capture, label, expected, observed, failures);
                                compare(label, expected, observed, failures);
                                float[] expectedLogits = expected.get("mtp_logits"), observedLogits = observed.get("mtp_logits");
                                int expectedToken = 0, observedToken = 0;
                                for (int j = 1; j < expectedLogits.length; j++) {
                                    if (expectedLogits[j] > expectedLogits[expectedToken]) expectedToken = j;
                                    if (observedLogits[j] > observedLogits[observedToken]) observedToken = j;
                                }
                                log.info("MTP_CAPTURE_STEADY {} freshArgmax={} replayArgmax={}", label, expectedToken, observedToken);
                                if (expectedToken != observedToken)
                                    failures.add(label + " argmax fresh=" + expectedToken + " steady=" + observedToken);
                                if (i + 1 < captures.size()) {
                                    try (Inputs next = new Inputs(captures.get(i + 1))) {
                                        compare(label + "/captured-full-carry", Map.of(HIDDEN, floats(next.hidden)),
                                                Map.of(HIDDEN, observed.get(HIDDEN)), failures);
                                        if (next.feeds.get("mtp_input_ids").getLong(0) != observedToken)
                                            failures.add(label + " token differs from captured next input");
                                    }
                                    // Production-only interleaving probe: between predictor calls the
                                    // real decode loop executes the TARGET verification forward (cast
                                    // workspaces + KV writes + allocator churn) against the same device
                                    // arena. Reproduce that pressure without touching the predictor's
                                    // own ext buffers: allocate, fill with a distinct pattern, and free
                                    // predictor-sized scratch. If the NEXT call's replay diverges, the
                                    // cross-plan arena/alias clobber is caught in isolation.
                                    int upcomingPos = (int) captures.get(i + 1).sourcePosition;
                                    try (var churnK = Nd4j.create(DataType.FLOAT, 1, upcomingPos + 1, 4, 256).muli(0.017f);
                                         var churnV = Nd4j.create(DataType.FLOAT, 1, upcomingPos + 1, 4, 256).muli(-0.019f);
                                         var churnQ = Nd4j.create(DataType.FLOAT, 1, 1, 16, 256).muli(0.023f)) {
                                        // touch on device then let GC scope release them
                                        Nd4j.getExecutioner().commit();
                                    }
                                    log.info("MTP_INTERLEAVE_CHURN after call={} position={} next={}",
                                            capture.callIndex, capture.sourcePosition, upcomingPos);
                                }
                            } finally { release(fresh, graphs); }
                        }
                    }
                    lifecycleGate(probe, "steady/final", failures);
                    auditProduction(probe, "steady/final", failures);
                    release(probe, graphs);
                }
            } finally {
                for (int i = graphs.size() - 1; i >= 0; i--) graphs.get(i).close();
            }
        } finally {
            DspDiagnostics.setCategories(mask);
            DspDiagnostics.setLevel(level);
        }
        assertTrue(failures.isEmpty(), () -> String.join("\n", failures));
    }

    private static void run(Weights weights, boolean diagnostic, boolean production, List<String> failures, List<SameDiff> graphs) {
        String label = production ? "production-logits-hidden" : diagnostic ? "diagnostic" : "final-only";
        try (Inputs seed = new Inputs(); Inputs repeated = new Inputs(); Inputs carry = new Inputs()) {
            SameDiff reference = graph(weights, diagnostic, production, graphs);
            String[] outputs = reference.outputs().toArray(new String[0]);
            Map<String, float[]> expected = evaluate(reference, seed, outputs, label + "/fresh-warmup");
            assertEquals(0, DspPlanAssertions.getTotalGraphReplays(reference), "Reference must not replay");
            assertTrue(DspPlanAssertions.getPlanPhase(reference) < 2, "Reference must be a natural first warmup");
            SameDiff probe = graph(weights, diagnostic, production, graphs);
            boolean sawUnfrozen = false;
            for (int call = 0; call < (production ? 6 : CALLS); call++) {
                repeated.reset(); // Intentional identical-input/state experiment; cache writes are real.
                boolean frozenBefore = probe.isDspShapesFrozen();
                Map<String, float[]> actual = evaluate(probe, repeated, outputs, label + "/reset/" + call);
                sawUnfrozen |= !frozenBefore;
                compare(label + "/reset/" + call, expected, actual, failures);
            }
            assertTrue(sawUnfrozen, "Must start with natural warmup");
            // Retain every lifecycle assertion as a final test failure, while collecting the
            // independent diagnostic experiment too. This does not waive the replay gate.
            lifecycleGate(probe, label, failures);
            if (production) auditProduction(probe, label, failures);
            release(reference, graphs);
            // Carry remains on device: attention updates K/V in place, hidden.assign consumes the
            // returned device array. Each local reference receives independent copies BEFORE call.
            for (int step = 0; step < 4; step++) {
                carry.position(2 + step);
                SameDiff fresh = graph(weights, diagnostic, production, graphs);
                try (Inputs independent = new Inputs()) {
                    independent.copyFrom(carry);
                    Map<String, float[]> localExpected = evaluate(fresh, independent, outputs,
                            label + "/carry-reference/" + step);
                    assertEquals(0, DspPlanAssertions.getTotalGraphReplays(fresh));
                    assertTrue(DspPlanAssertions.getPlanPhase(fresh) < 2);
                    Map<String, INDArray> out = probe.output(carry.feeds, outputs);
                    // D2D copy BEFORE host observation; no host-roundtrip carry injection.
                    carry.hidden.assign(out.get(HIDDEN));
                    Map<String, float[]> actual = snapshot(out, carry, outputs);
                    lifecycle(probe, label + "/carry/" + step);
                    compare(label + "/carry/" + step, localExpected, actual, failures);
                    assertArrayEquals(actual.get(HIDDEN), floats(carry.hidden),
                            "Device carry copy must preserve predictor output exactly");
                } finally {
                    release(fresh, graphs);
                }
            }
            lifecycleGate(probe, label + "/carry", failures);
            log.info("MTP_PROBE variant={} finished; observed failures={} (not a full-MTP fix claim)", label, failures.size());
            release(probe, graphs);
        }
    }

    private static void auditProduction(SameDiff sd, String label, List<String> failures) {
        var handle = DspPlanAssertions.getPlanHandleForQuery(sd);
        int segments = Nd4j.getNativeOps().getPlanSegmentCount(handle);
        boolean tritonReplayed = false;
        for (int i = 0; i < segments; i++) {
            String backend = DspPlanAssertions.getSegmentCompiledBackend(sd, i);
            int replays = DspPlanAssertions.getSegmentReplayCount(sd, i);
            log.info("MTP_PROBE audit={} backend={} {}", label, backend,
                    DspPlanAssertions.snapshotSegmentState(sd, i));
            tritonReplayed |= backend != null && backend.contains("Triton") && replays > 0;
        }
        if (!tritonReplayed) failures.add(label + " no observed Triton segment replay within bounded calls");
    }

    private static void release(SameDiff execution, List<SameDiff> graphs) {
        // GraphOptimizer deep-copies constants. Bound retained checkpoint copies to two
        // execution graphs; raw builders share Weights and are closed only at the end.
        execution.close();
        graphs.remove(execution);
    }

    private static void lifecycleGate(SameDiff sd, String label, List<String> failures) {
        for (Runnable check : Arrays.<Runnable>asList(
                () -> DspPlanAssertions.assertFullyReplaying(sd, label),
                () -> DspPlanAssertions.assertTotalGraphReplaysAtLeast(sd, 1, label),
                () -> DspPlanAssertions.assertNoPhaseContractViolations(sd, label))) {
            try { check.run(); }
            catch (AssertionError error) {
                log.error("MTP_PROBE lifecycle failure {}", error.getMessage());
                failures.add(label + " lifecycle: " + error.getMessage());
            }
        }
    }

    private static SameDiff graph(Weights weights, boolean diagnostic, boolean production, List<SameDiff> owned) {
        SameDiff sd = SameDiff.create();
        owned.add(sd);
        new Predictor().build(sd, weights, production);
        List<SDVariable> constants = new ArrayList<>();
        for (SDVariable v : sd.variables()) if (v.getVariableType() == VariableType.VARIABLE) constants.add(v);
        sd.convertToConstants(constants);
        List<String> names = new ArrayList<>();
        if (diagnostic) {
            // Real production variable names in upstream-to-downstream order, not copied math.
            names.addAll(Arrays.asList("mtp_embedded_storage", "mtp.enorm", "mtp.hnorm",
                    "mtp_embedding_hidden_concat", "mtp_eh_projected", "model.layers.64.input_layernorm",
                    "q_rope_64", "k_rope_64", "v_heads_64", "attn_out_64", "gated_attn_64",
                    "attn_proj_64", "post_attn_64", "model.layers.64.post_attention_layernorm",
                    "gate_64", "up_64", "swiglu_64", "down_64"));
        }
        // GenerationPipeline.prepare native MTP requests only these two outputs, in this order.
        // K/V are committed in place; requesting their diagnostic outputs changes fusion.
        names.addAll(production ? Arrays.asList("mtp_logits", HIDDEN)
                : Arrays.asList(HIDDEN, "mtp_key_states", "mtp_value_states"));
        for (String name : names) assertNotNull(sd.getVariable(name), "Production boundary " + name);
        sd.setOutputs(names.toArray(new String[0]));
        SameDiff optimized = GraphOptimizer.optimize(sd, names);
        // Fusion may reorder the graph-level outputs; keep diagnostic comparison order explicit.
        optimized.setOutputs(names.toArray(new String[0]));
        assertNotSame(sd, optimized, "Execution must own its optimizer-cloned constants independently");
        owned.add(optimized);
        assertTrue(optimized.isDspAutoCompileEnabled());
        assertTrue(optimized.isDspNativeAutoCompileEnabled());
        assertEquals(GraphExecutionMode.AUTO, optimized.getGraphExecutionMode());
        log.info("MTP_PROBE graph variant={} builder=LLaMAArchitecture.buildMtpBranch outputs={}", diagnostic, names);
        return optimized;
    }

    private static final class Predictor extends LLaMAArchitecture {
        @Override protected boolean multiplyNormInFloat() { return true; }
        void build(SameDiff sd, Weights w, boolean production) {
            SDVariable embedding = sd.var("token_embd.weight", w.values.get("token_embd.weight"));
            // Head-free variants leave the unrequested boundary unbound; production uses
            // the actual packed shared head. No fake head/table or remapped token IDs.
            SDVariable head = production ? sd.var("lm_head.weight", w.values.get("output.weight"))
                    : sd.placeHolder("unrequested_lm_head", DataType.BFLOAT16, 248320, 5120);
            buildMtpBranch(sd, w.config, w.values, DataType.BFLOAT16, embedding, head, new ArrayList<>());
        }
    }

    private static Map<String, float[]> evaluate(SameDiff sd, Inputs in, String[] outputs, String label) {
        return evaluate(sd, in, outputs, label, false, false);
    }

    private static Map<String, float[]> evaluate(SameDiff sd, Inputs in, String[] outputs, String label,
                                                 boolean steadyState, boolean requireFast) {
        float[] beforeK = floats(in.keys), beforeV = floats(in.values), beforeHidden = floats(in.hidden);
        Map<String, INDArray> out;
        if (steadyState) {
            var executor = sd.getOrCreateSession().getDynamicShapePlanExecutor();
            DspDiagnostics.clear(); // Only diagnostic events, never the live plan/cache.
            out = executor.executeSteadyState(executor.getCurrentPlan(), in.feeds);
            String report = DspDiagnostics.getJsonReport();
            boolean fast = report != null && report.contains("[DSP_GATE] FAST executeSteadyState()");
            log.info("MTP_STEADY_ENTRY {} nativeFast={} executeCount={}", label, fast,
                    Nd4j.getNativeOps().getPlanExecuteCount(executor.getNativePlanHandle()));
            if (requireFast) assertTrue(fast,
                    label + " must enter native FAST executeSteadyState, not ordinary execution: " + report);
        } else {
            out = sd.output(in.feeds, outputs);
        }
        lifecycle(sd, label);
        Map<String, float[]> result = snapshot(out, in, outputs);
        int offset = in.feeds.get("mtp_cache_position").getInt(0) * 1024;
        if (result.containsKey("mtp_key_states")) {
            System.arraycopy(result.get("mtp_key_states"), 0, beforeK, offset, 1024);
            System.arraycopy(result.get("mtp_value_states"), 0, beforeV, offset, 1024);
            assertArrayEquals(beforeK, result.get("committed-key-cache"), label + " exact K commit + untouched prefix/tail");
            assertArrayEquals(beforeV, result.get("committed-value-cache"), label + " exact V commit + untouched prefix/tail");
        } else {
            // Full cache parity is checked against independent fresh execution. Here also
            // verify every prefix/tail value outside the one committed position is untouched.
            for (int i = 0; i < beforeK.length; i++) if (i < offset || i >= offset + 1024) {
                assertEquals(beforeK[i], result.get("committed-key-cache")[i], label + " K untouched " + i);
                assertEquals(beforeV[i], result.get("committed-value-cache")[i], label + " V untouched " + i);
            }
        }
        assertArrayEquals(beforeHidden, floats(in.hidden), label + " hidden input must not mutate");
        return result;
    }
    private static void lifecycle(SameDiff sd, String label) {
        log.info("MTP_PROBE call={} plan={} phase={} frozen={} replays={} frozenExec={} mode={}", label,
                DspPlanAssertions.getPlanHandleForQuery(sd).address(), DspPlanAssertions.getPlanPhase(sd),
                sd.isDspShapesFrozen(), DspPlanAssertions.getTotalGraphReplays(sd),
                DspPlanAssertions.getFrozenExecCount(sd), sd.getGraphExecutionMode());
    }
    private static Map<String, float[]> snapshot(Map<String, INDArray> out, Inputs in, String[] outputs) {
        Map<String, float[]> result = new LinkedHashMap<>();
        for (String name : outputs) {
            INDArray a = out.get(name);
            assertNotNull(a, name);
            result.put(name, floats(a));
        }
        assertArrayEquals(new long[]{1, 1, 5120}, out.get(HIDDEN).shape());
        assertEquals(DataType.BFLOAT16, out.get(HIDDEN).dataType());
        if (out.containsKey("mtp_logits")) {
            assertArrayEquals(new long[]{1, 1, 248320}, out.get("mtp_logits").shape());
            assertEquals(DataType.FLOAT, out.get("mtp_logits").dataType());
        }
        result.put("committed-key-cache", floats(in.keys));
        result.put("committed-value-cache", floats(in.values));
        return result;
    }
    private static float[] floats(INDArray a) {
        try (INDArray c = a.dup('c')) {
            if (c.dataType() == DataType.FLOAT) return c.data().asFloat().clone();
            try (INDArray f = c.castTo(DataType.FLOAT)) { return f.data().asFloat().clone(); }
        }
    }
    private static void compareNativeCapture(DspTensorSnapshot capture, String label,
                                             Map<String, float[]> fresh, Map<String, float[]> compiled,
                                             List<String> failures) {
        if (!capture.tensors.containsKey(DspTensorSnapshot.LOGITS)
                && !capture.tensors.containsKey(DspTensorSnapshot.HIDDEN)) {
            assertFalse(capture.tensors.containsKey(DspTensorSnapshot.DRAFT), "Draft without native outputs");
            log.info("MTP_CAPTURE {} legacy input-only artifact: native outputs unavailable", label);
            return;
        }
        assertNotNull(capture.tensors.get(DspTensorSnapshot.LOGITS), "Missing captured native logits");
        assertNotNull(capture.tensors.get(DspTensorSnapshot.HIDDEN), "Missing captured native hidden");
        Map<String, float[]> nativeOutputs = new LinkedHashMap<>();
        for (String name : new String[]{DspTensorSnapshot.LOGITS, DspTensorSnapshot.HIDDEN}) {
            try (INDArray tensor = capture.tensors.get(name).reconstruct(capture.payloadOrder)) {
                String output = name.equals(DspTensorSnapshot.LOGITS) ? "mtp_logits" : HIDDEN;
                assertEquals(output.equals(HIDDEN) ? DataType.BFLOAT16 : DataType.FLOAT, tensor.dataType(), name);
                assertArrayEquals(output.equals(HIDDEN) ? new long[]{1, 1, 5120} : new long[]{1, 1, 248320},
                        tensor.shape(), name);
                nativeOutputs.put(output, floats(tensor));
            }
        }
        compare(label + "/captured-native-vs-fresh", nativeOutputs, fresh, failures);
        compare(label + "/captured-native-vs-compiled", nativeOutputs, compiled, failures);
        int nativeLogitsToken = argmax(nativeOutputs.get("mtp_logits"));
        int freshToken = argmax(fresh.get("mtp_logits")), compiledToken = argmax(compiled.get("mtp_logits"));
        if (nativeLogitsToken != freshToken || nativeLogitsToken != compiledToken)
            failures.add(label + " logits argmax captured=" + nativeLogitsToken + " fresh=" + freshToken
                    + " compiled=" + compiledToken);
        if (capture.tensors.containsKey(DspTensorSnapshot.DRAFT)) {
            try (INDArray draft = capture.tensors.get(DspTensorSnapshot.DRAFT).reconstruct(capture.payloadOrder)) {
                assertEquals(DataType.INT64, draft.dataType());
                assertEquals(0, draft.rank(), "Captured draft scalar");
                long nativeDraft = draft.getLong(0);
                log.info("MTP_CAPTURE {} capturedLogitsArgmax={} capturedNativeDraft={} freshArgmax={} compiledArgmax={}",
                        label, nativeLogitsToken, nativeDraft, freshToken, compiledToken);
                if (nativeDraft != nativeLogitsToken)
                    failures.add(label + " native argmax stage: captured logits=" + nativeLogitsToken
                            + " captured draft=" + nativeDraft);
                if (nativeDraft != freshToken || nativeDraft != compiledToken)
                    failures.add(label + " selected token: captured native=" + nativeDraft + " fresh=" + freshToken
                            + " compiled=" + compiledToken);
            }
        } else {
            log.info("MTP_CAPTURE {} nine-tensor artifact: selected native draft unavailable", label);
        }
    }

    private static int argmax(float[] logits) {
        int best = 0;
        for (int i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static void compare(String label, Map<String, float[]> expected, Map<String, float[]> actual,
                                List<String> failures) {
        for (String name : expected.keySet()) {
            float[] a = expected.get(name), b = actual.get(name);
            assertEquals(a.length, b.length, name);
            int first = -1, count = 0;
            double max = 0;
            for (int i = 0; i < a.length; i++) {
                if (!Float.isFinite(a[i]) || !Float.isFinite(b[i]) || Float.floatToRawIntBits(a[i]) != Float.floatToRawIntBits(b[i])) {
                    if (first < 0) first = i;
                    count++;
                    max = Math.max(max, Math.abs((double) a[i] - b[i]));
                }
            }
            log.info("MTP_PROBE compare={} stage={} mismatch={}/{} first={} maxAbs={} firstExpected={} firstActual={}",
                    label, name, count, a.length, first, max, first < 0 ? 0 : a[first], first < 0 ? 0 : b[first]);
            if (first >= 0) failures.add(label + " stage=" + name + " first=" + first + " expected=" + a[first]
                    + " actual=" + b[first] + " mismatches=" + count + " maxAbs=" + max);
        }
    }

    private static final class Inputs implements AutoCloseable {
        final Map<String, INDArray> feeds = new LinkedHashMap<>();
        final INDArray hidden, keys, values;
        Inputs(DspTensorSnapshot capture) {
            // Expected native outputs are evidence, never graph feeds (legacy seven-input files still work).
            feeds.putAll(capture.reconstructInputs());
            hidden = feeds.get("mtp_target_hidden_states"); keys = feeds.get(KEY); values = feeds.get(VALUE);
        }
        Inputs() {
            hidden = Nd4j.create(DataType.BFLOAT16, 1, 1, 5120);
            keys = Nd4j.create(DataType.BFLOAT16, 1, CAPACITY, 4, 256);
            values = Nd4j.create(DataType.BFLOAT16, 1, CAPACITY, 4, 256);
            feeds.put("mtp_input_ids", Nd4j.createFromArray(new long[][]{{579}}));
            feeds.put("mtp_target_hidden_states", hidden);
            feeds.put(KEY, keys); feeds.put(VALUE, values);
            feeds.put("mtp_position_offset", Nd4j.scalar(DataType.INT64, 2));
            feeds.put("mtp_cache_position", Nd4j.scalar(DataType.INT64, 2));
            feeds.put("mtp_causal_mask", Nd4j.create(DataType.FLOAT, 1, 1, 1, CAPACITY));
            reset();
        }
        void restore(DspTensorSnapshot capture) {
            assertTrue(capture.tensors.keySet().containsAll(feeds.keySet()));
            for (String name : feeds.keySet())
                capture.tensors.get(name).restore(feeds.get(name), capture.payloadOrder);
        }
        void reset() {
            deterministic(hidden, 7);
            keys.assign(0); values.assign(0);
            try (INDArray k = keys.get(all(), interval(0, 2), all(), all());
                 INDArray v = values.get(all(), interval(0, 2), all(), all())) {
                deterministic(k, 3); deterministic(v, 11);
            }
            position(2);
        }
        void position(int p) {
            feeds.get("mtp_position_offset").assign(p);
            feeds.get("mtp_cache_position").assign(p);
            float[] bias = new float[CAPACITY];
            Arrays.fill(bias, p + 1, CAPACITY, Float.NEGATIVE_INFINITY);
            try (INDArray b = Nd4j.createFromArray(bias)) {
                feeds.get("mtp_causal_mask").assign(b.reshape(1, 1, 1, CAPACITY));
            }
        }
        void copyFrom(Inputs other) {
            for (String name : feeds.keySet()) feeds.get(name).assign(other.feeds.get(name));
            assertNotEquals(keys.data().address(), other.keys.data().address(), "Independent cache storage");
        }
        @Override public void close() { for (INDArray a : feeds.values()) if (!a.wasClosed()) a.close(); }
    }
    private static void deterministic(INDArray a, int seed) {
        float[] data = new float[Math.toIntExact(a.length())];
        for (int i = 0; i < data.length; i++) data[i] = ((i * seed % 127) - 63) / 256.0f;
        try (INDArray source = Nd4j.createFromArray(data).reshape(a.shape())) { a.assign(source); }
    }

    private static final class Weights implements AutoCloseable {
        final File cache = new File(System.getProperty("user.home"), ".cache/dl4j-llm-models");
        final Map<String, INDArray> values = new LinkedHashMap<>();
        final ArchitectureConfig config;
        final JsonObject index;
        long bytes;
        Weights() throws IOException {
            ModelOptQwenConfig model = ModelOptQwenConfig.read(cached("config.json"), cached("hf_quant_config.json"), cached("generation_config.json"));
            config = model.getArchitecture();
            assertEquals(64, config.getNumLayers()); assertEquals(1, config.getNumMtpLayers());
            assertEquals(5120, config.getHiddenSize()); assertEquals(17408, config.getIntermediateSize());
            assertEquals(24, config.getNumAttentionHeads()); assertEquals(4, config.getNumKVHeads());
            assertEquals(256, config.getHeadDimension());
            try (Reader r = Files.newBufferedReader(cached("model.safetensors.index.json").toPath())) {
                index = new Gson().fromJson(r, JsonObject.class).getAsJsonObject("weight_map");
            }
            try {
                load("model.language_model.embed_tokens.weight", "token_embd.weight", false, 248320, 5120);
                load("mtp.fc.weight", "blk.64.nextn.eh_proj.weight", false, 5120, 10240);
                load("mtp.pre_fc_norm_embedding.weight", "blk.64.nextn.enorm.weight", true, 5120);
                load("mtp.pre_fc_norm_hidden.weight", "blk.64.nextn.hnorm.weight", true, 5120);
                load("mtp.norm.weight", "blk.64.nextn.shared_head_norm.weight", true, 5120);
                load("mtp.layers.0.input_layernorm.weight", "blk.64.attn_norm.weight", true, 5120);
                load("mtp.layers.0.post_attention_layernorm.weight", "blk.64.post_attention_norm.weight", true, 5120);
                for (String role : new String[]{"q", "k"}) load("mtp.layers.0.self_attn." + role + "_norm.weight", "blk.64.attn_" + role + "_norm.weight", true, 256);
                load("mtp.layers.0.self_attn.q_proj.weight", "blk.64.attn_q.weight", false, 12288, 5120);
                load("mtp.layers.0.self_attn.k_proj.weight", "blk.64.attn_k.weight", false, 1024, 5120);
                load("mtp.layers.0.self_attn.v_proj.weight", "blk.64.attn_v.weight", false, 1024, 5120);
                load("mtp.layers.0.self_attn.o_proj.weight", "blk.64.attn_output.weight", false, 5120, 6144);
                load("mtp.layers.0.mlp.gate_proj.weight", "blk.64.ffn_gate.weight", false, 17408, 5120);
                load("mtp.layers.0.mlp.up_proj.weight", "blk.64.ffn_up.weight", false, 17408, 5120);
                load("mtp.layers.0.mlp.down_proj.weight", "blk.64.ffn_down.weight", false, 5120, 17408);
                long count = index.entrySet().stream().filter(e -> e.getKey().startsWith("mtp.")).count();
                assertEquals(15, count, "Exact predictor coverage; unknown tensors must not be ignored");
                log.info("MTP_PROBE cache={} loadedBytes={} predictorTensors={} configLayers=64 executedTargetLayers=0", cache, bytes, count);
            } catch (IOException | RuntimeException | Error e) { close(); throw e; }
        }
        void loadHead() throws IOException {
            // ModelOptQwenImporter.linear W4A16 mapping: packed U8, E4M3 block scales,
            // positive scalar global scale. No activation quantization or dense expansion.
            loadPacked("lm_head.weight", "output.weight", "U8", 248320, 2560);
            loadPacked("lm_head.weight_scale", "output.weight" + QuantizedLinear.MODELOPT_BLOCK_SCALE,
                    "F8_E4M3", 248320, 320);
            loadPacked("lm_head.weight_scale_2", "output.weight" + QuantizedLinear.MODELOPT_GLOBAL_SCALE, "F32");
            log.info("MTP_PROBE complete predictor payload bytes={}", bytes);
        }
        void loadPacked(String source, String destination, String dtype, long... shape) throws IOException {
            assertTrue(index.has(source), source);
            try (SafeTensorsReader reader = SafeTensorsReader.open(cached(index.get(source).getAsString()))) {
                var info = reader.getTensorInfo(source);
                assertEquals(dtype, info.getDtype(), source);
                if (shape.length != 0) assertArrayEquals(shape, info.getShape(), source);
                INDArray raw = reader.readTensor(source);
                values.put(destination, raw);
                if (shape.length == 0) {
                    assertEquals(1, raw.length(), source);
                    assertTrue(Float.isFinite(raw.getFloat(0)) && raw.getFloat(0) > 0, source);
                    values.put(destination, raw.reshape(new long[0]));
                }
                bytes += info.getDataLength();
                assertTrue(bytes < 5L * 1024 * 1024 * 1024, "Bounded real packed head + embedding + predictor");
                log.info("MTP_PROBE packed tensor={} dtype={} shape={} bytes={}", source, dtype,
                        Arrays.toString(info.getShape()), info.getDataLength());
            }
        }
        File cached(String suffix) throws IOException {
            File f = new File(cache, PREFIX + suffix);
            if (!f.isFile()) throw new IOException("Cache-only file absent: " + f);
            return f;
        }
        void load(String source, String destination, boolean norm, long... shape) throws IOException {
            assertTrue(index.has(source), source);
            try (SafeTensorsReader reader = SafeTensorsReader.open(cached(index.get(source).getAsString()))) {
                var info = reader.getTensorInfo(source);
                assertNotNull(info, source); assertEquals("BF16", info.getDtype(), source);
                assertArrayEquals(shape, info.getShape(), source);
                bytes += info.getDataLength();
                assertTrue(bytes < 4L * 1024 * 1024 * 1024, "Bounded predictor + embedding payload");
                log.info("MTP_PROBE tensor={} canonical={} shard={} bytes={} dtype={} shape={}", source,
                        destination, reader.getFile(), info.getDataLength(), info.getDtype(), Arrays.toString(shape));
                INDArray raw = reader.readTensor(source);
                if (norm) {
                    float[] gamma = floats(raw);
                    for (int i = 0; i < gamma.length; i++) gamma[i] = 1.0f + gamma[i];
                    raw.close();
                    raw = Nd4j.createFromArray(gamma); // Exactly ModelOptQwenImporter.ONE_CENTERED_NORM.
                }
                values.put(destination, raw);
            }
        }
        @Override public void close() {
            for (INDArray a : values.values()) if (!a.wasClosed() && !a.data().wasClosed()) {
                a.data().setConstant(false); a.setCloseable(true); a.close();
            }
        }
    }
}
