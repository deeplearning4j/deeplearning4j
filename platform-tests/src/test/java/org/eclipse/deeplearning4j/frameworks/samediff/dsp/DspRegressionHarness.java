/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.eclipse.deeplearning4j.frameworks.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Shared base class for DSP regression tests that target gaps in the consolidated DSP
 * test framework — in particular, perf-floor assertions and lifecycle/invariant coverage
 * that the existing 53 tests do not exercise.
 *
 * Design goals:
 *  - Every subclass builds its own {@link #buildFixture()} graph (typically a small
 *    synthetic decode-loop — NOT full VLM — so tests are cheap enough to run in CI).
 *  - The harness snapshots + restores all DSP/Triton environment knobs around every
 *    test using the same pattern as DspLifecycleValidationTest.EnvSnapshot so one
 *    misbehaving test cannot leak state into the next.
 *  - Shared assertion helpers ({@link #assertTokPerSecFloor}) live here so every
 *    regression test uses the same timing/warmup protocol.
 */
@Slf4j
public abstract class DspRegressionHarness {

    /** The graph under test. Subclasses return a freshly-built SameDiff instance. */
    protected abstract SameDiff buildFixture();

    /** Default tok/s floor — can be overridden with the {@code dsp.perf.minTokPerSec} system property. */
    protected static double defaultTokPerSecFloor() {
        String s = System.getProperty("dsp.perf.minTokPerSec");
        if (s == null || s.isEmpty()) return 70.0;
        try { return Double.parseDouble(s); } catch (NumberFormatException e) { return 70.0; }
    }

    /** Whether CUDA is available. Used by @Disabled conditions and skip logic. */
    protected static boolean isCudaAvailable() {
        try {
            String name = Nd4j.getBackend().getClass().getName().toLowerCase();
            // JCublasBackend is the CUDA backend — check for both "cuda" and "cublas"
            return name.contains("cuda") || name.contains("cublas") || name.contains("jcublas");
        } catch (Throwable t) {
            return false;
        }
    }

    private EnvSnapshot beforeSnapshot;

    @BeforeEach
    public void snapshotEnv() {
        this.beforeSnapshot = EnvSnapshot.capture();
    }

    @AfterEach
    public void restoreEnv() {
        if (beforeSnapshot != null) {
            beforeSnapshot.restore();
        }
    }

    /**
     * Run {@code warmup} iterations (not timed) then {@code iterations} timed iterations
     * and assert that the achieved tok/s meets the floor. Throws AssertionError with a
     * human-readable message on regression.
     */
    protected void assertTokPerSecFloor(SameDiff sd,
                                        Map<String, INDArray> inputs,
                                        String outputName,
                                        int warmup,
                                        int iterations,
                                        double minTokPerSec) {
        // Warmup — exercise JIT, Triton compile cache, CUDA graph capture, etc.
        for (int i = 0; i < warmup; i++) {
            sd.output(inputs, outputName);
        }

        long startNs = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            sd.output(inputs, outputName);
        }
        long elapsedNs = System.nanoTime() - startNs;
        double seconds = elapsedNs / 1.0e9;
        double tokPerSec = iterations / seconds;

        log.info("assertTokPerSecFloor: {} iters in {} ms -> {} tok/s (floor={})",
                iterations, elapsedNs / 1_000_000L, String.format("%.2f", tokPerSec),
                String.format("%.2f", minTokPerSec));

        assertTrue(tokPerSec >= minTokPerSec,
                String.format("DSP perf regression: achieved %.2f tok/s, floor=%.2f tok/s. " +
                                "This would have caught the current batch-zero regression.",
                        tokPerSec, minTokPerSec));
    }

    /**
     * Assert that every segment output slot is zero before first write.
     *
     * TODO(native-instrumentation): this currently skips with a clear message because
     * it needs access to the libnd4j DspDiagnostics ring buffer via a JNI method that
     * does not yet exist. Follow-up: add {@code NativeOps::dspRingBufferSnapshot()}
     * returning an array of event records, then this method can iterate and assert
     * that every SEGMENT_SLOT_WRITE event was preceded by a SEGMENT_SLOT_ZERO event
     * with matching slot id. See {@code libnd4j/include/graph/DspDiagnostics.h}.
     */
    protected void assertSlotZeroBeforeWrite(SameDiff sd, Map<String, INDArray> inputs) {
        // Execute once so the ring buffer has events, even though we can't read them yet.
        sd.output(inputs, sd.outputs().get(0));
        assumeTrue(false,
                "assertSlotZeroBeforeWrite: needs JNI access to DspDiagnostics ring buffer. " +
                "Skipping until NativeOps::dspRingBufferSnapshot() is added (see TODO in harness).");
    }

    // ────────────────────────────────────────────────────────────────────────
    // EnvSnapshot — copied from DspLifecycleValidationTest.EnvSnapshot.
    // Kept here (not imported) because the original is package-private.
    // ────────────────────────────────────────────────────────────────────────
    protected static final class EnvSnapshot {
        boolean dspBatchZero;
        boolean dspBatchZeroKernel;
        boolean dspBatchedGemm;
        boolean dspCastSinkMatmul;
        boolean dspCastElimination;
        boolean dspFp16Compute;
        boolean dspFreezeMergeSegments;
        boolean dspFreezeRecompile;
        boolean tritonGraphCapture;
        boolean tritonSectionFusion;
        boolean tritonConsolidatedArgTable;
        boolean tritonArgDirtyTracking;
        boolean tritonSkipKernels;
        boolean tritonVerifyKernels;
        boolean tritonForceRecapture;
        boolean tritonCompileAll;
        boolean tritonCooperativeLaunch;
        boolean tritonVerbose;
        boolean tritonDumpSections;
        boolean tritonAllowFallbackCapture;
        String  tritonIncludeTypes;
        String  tritonExcludeOps;
        boolean cublasTf32;
        boolean tritonTf32;

        static EnvSnapshot capture() {
            Environment env = Nd4j.getEnvironment();
            EnvSnapshot s = new EnvSnapshot();
            try {
                s.dspBatchZero             = env.dspBatchZero();
                s.dspBatchZeroKernel       = env.dspBatchZeroKernel();
                s.dspBatchedGemm           = env.dspBatchedGemm();
                s.dspCastSinkMatmul        = env.dspCastSinkMatmul();
                s.dspCastElimination       = env.dspCastElimination();
                s.dspFp16Compute           = env.dspFp16Compute();
                s.dspFreezeMergeSegments   = env.dspFreezeMergeSegments();
                s.dspFreezeRecompile       = env.dspFreezeRecompile();
                s.tritonGraphCapture       = env.tritonGraphCapture();
                s.tritonSectionFusion      = env.tritonSectionFusion();
                s.tritonConsolidatedArgTable = env.tritonConsolidatedArgTable();
                s.tritonArgDirtyTracking   = env.tritonArgDirtyTracking();
                s.tritonSkipKernels        = env.tritonSkipKernels();
                s.tritonVerifyKernels      = env.tritonVerifyKernels();
                s.tritonForceRecapture     = env.tritonForceRecapture();
                s.tritonCompileAll         = env.tritonCompileAll();
                s.tritonCooperativeLaunch  = env.tritonCooperativeLaunch();
                s.tritonVerbose            = env.tritonVerbose();
                s.tritonDumpSections       = env.tritonDumpSections();
                s.tritonAllowFallbackCapture = env.tritonAllowFallbackCapture();
                s.tritonIncludeTypes       = env.tritonIncludeTypes();
                s.tritonExcludeOps         = env.tritonExcludeOps();
                s.cublasTf32               = env.cublasTf32Enabled();
                s.tritonTf32               = env.tritonTf32Enabled();
            } catch (Throwable t) {
                log.warn("EnvSnapshot.capture: partial snapshot due to {}", t.toString());
            }
            return s;
        }

        void restore() {
            Environment env = Nd4j.getEnvironment();
            try {
                env.setDspBatchZero(dspBatchZero);
                env.setDspBatchZeroKernel(dspBatchZeroKernel);
                env.setDspBatchedGemm(dspBatchedGemm);
                env.setDspCastSinkMatmul(dspCastSinkMatmul);
                env.setDspCastElimination(dspCastElimination);
                env.setDspFp16Compute(dspFp16Compute);
                env.setDspFreezeMergeSegments(dspFreezeMergeSegments);
                env.setDspFreezeRecompile(dspFreezeRecompile);
                env.setTritonGraphCapture(tritonGraphCapture);
                env.setTritonSectionFusion(tritonSectionFusion);
                env.setTritonConsolidatedArgTable(tritonConsolidatedArgTable);
                env.setTritonArgDirtyTracking(tritonArgDirtyTracking);
                env.setTritonSkipKernels(tritonSkipKernels);
                env.setTritonVerifyKernels(tritonVerifyKernels);
                env.setTritonForceRecapture(tritonForceRecapture);
                env.setTritonCompileAll(tritonCompileAll);
                env.setTritonCooperativeLaunch(tritonCooperativeLaunch);
                env.setTritonVerbose(tritonVerbose);
                env.setTritonDumpSections(tritonDumpSections);
                env.setTritonAllowFallbackCapture(tritonAllowFallbackCapture);
                if (tritonIncludeTypes != null) env.setTritonIncludeTypes(tritonIncludeTypes);
                if (tritonExcludeOps   != null) env.setTritonExcludeOps(tritonExcludeOps);
                env.setCublasTf32Enabled(cublasTf32);
                env.setTritonTf32Enabled(tritonTf32);
            } catch (Throwable t) {
                log.warn("EnvSnapshot.restore: partial restore due to {}", t.toString());
            }
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    // Shared tiny decode-loop fixture helper (used by most subclasses).
    // ────────────────────────────────────────────────────────────────────────

    /**
     * Build a small synthetic decode-loop-ish graph: one matmul followed by a softmax,
     * shaped so it can be driven for N steps cheaply. Subclasses that don't need
     * anything exotic can return this from {@link #buildFixture()}.
     */
    protected SameDiff buildTinyDecodeFixture(int hidden, int vocab) {
        SameDiff sd = SameDiff.create();
        INDArray wArr = Nd4j.randn(new long[]{hidden, vocab}).castTo(Nd4j.defaultFloatingPointType());
        sd.constant("W", wArr);
        sd.placeHolder("x", Nd4j.defaultFloatingPointType(), 1, hidden);
        sd.nn().softmax("logits", sd.mmul(sd.getVariable("x"), sd.getVariable("W")), 1);
        return sd;
    }
}
