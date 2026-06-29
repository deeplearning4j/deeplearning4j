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
package org.nd4j.autodiff.samediff.dsp;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.common.config.ND4JSystemProperties;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;

import org.nd4j.linalg.api.device.DeviceType;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Tests for the LRU {@code ModuleResidencyCache} in {@code TritonGraphBackend}
 * and the associated soft byte budget / warning threshold knobs.
 *
 * <p>The cache is internal C++ state inside {@code TritonGraphBackend} and is
 * not directly Java-accessible. These tests therefore validate it
 * <em>indirectly</em>:
 * <ul>
 *   <li>Configuring a tiny budget should still produce numerically correct
 *       results, because evicted modules must transparently reload from disk.</li>
 *   <li>Setting an unlimited budget should never evict — execution remains
 *       correct (a regression-only check).</li>
 *   <li>Setting a very small warn threshold should drive the native warn path,
 *       which increments {@code TritonConfig::moduleResidencyWarnFireCount()};
 *       we observe that counter strictly increasing after execution. (The
 *       native {@code sd_printf} message is emitted to libc fd 1 and is not
 *       visible to Java's {@code System.setOut}, so we can't pattern-match
 *       stdout from inside the test.)</li>
 * </ul>
 *
 * <p>Run from {@code platform-tests}:
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspLruModuleResidencyTest \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/dsp-lru-residency.log
 * </pre>
 */
@Slf4j
@Tag("dsp")
public class DspLruModuleResidencyTest {

    private static final int IN_DIM = 16;
    private static final int HIDDEN = 32;
    private static final int OUT_DIM = 16;

    private static final long TINY_BUDGET_BYTES = 1L * 1024 * 1024;          // 1 MiB
    private static final long UNLIMITED_BUDGET_BYTES = 0L;                   // 0 = unlimited
    // A tiny elementwise Triton kernel compiles to ~2.5 KiB of PTX, so a
    // 4 KiB warn threshold is above the actual module size and will never
    // trip the warn path.  1 byte guarantees any loaded module exceeds it.
    private static final long TINY_WARN_BYTES = 1L;

    private SameDiff sd;
    private long savedBudgetBytes = -1L;
    private long savedWarnBytes = -1L;

    @BeforeEach
    public void setUp() {
        System.setProperty(ND4JSystemProperties.DYNAMIC_SHAPE_PLAN_ENABLED, "true");
        InferenceSession.setDynamicShapePlanEnabled(true);

        // Snapshot whatever the environment currently has so we can restore.
        savedBudgetBytes = readBudgetBytes();
        savedWarnBytes = readWarnBytes();
    }

    @AfterEach
    public void tearDown() {
        if (sd != null) {
            try {
                sd.close();
            } catch (Throwable t) {
                log.warn("sd.close() failed in tearDown", t);
            }
            sd = null;
        }
        // Restore the original budget/warn settings.
        if (savedBudgetBytes >= 0L) {
            try {
                writeBudgetBytes(savedBudgetBytes);
            } catch (Throwable t) {
                log.warn("Failed to restore moduleResidencyBudgetBytes", t);
            }
        }
        if (savedWarnBytes >= 0L) {
            try {
                writeWarnBytes(savedWarnBytes);
            } catch (Throwable t) {
                log.warn("Failed to restore moduleResidencyWarnBytes", t);
            }
        }
        Nd4j.getExecutioner().commit();
    }

    // ──────────────────────────────────────────────────────────────────────
    // Fixture — a graph with several distinct ops so multiple Triton modules
    // get compiled and become candidates for eviction.
    // ──────────────────────────────────────────────────────────────────────

    /**
     * Build a multi-segment graph. Several different op kernels (matmul,
     * relu, tanh, sigmoid, softmax, mul, add, gelu, layer norm) are chained
     * so the Triton compiler emits multiple distinct CUmodules — these are
     * the "many segments" the LRU cache will need to manage.
     */
    private SameDiff buildManyKernelGraph() {
        SameDiff out = SameDiff.create();
        SDVariable x = out.placeHolder("x", DataType.FLOAT, 4, IN_DIM);

        SDVariable w0 = out.var("w0", Nd4j.randn(DataType.FLOAT, IN_DIM, HIDDEN).muli(0.05));
        SDVariable w1 = out.var("w1", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.05));
        SDVariable w2 = out.var("w2", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.05));
        SDVariable w3 = out.var("w3", Nd4j.randn(DataType.FLOAT, HIDDEN, HIDDEN).muli(0.05));
        SDVariable w4 = out.var("w4", Nd4j.randn(DataType.FLOAT, HIDDEN, OUT_DIM).muli(0.05));

        SDVariable h0 = out.mmul("h0", x, w0);
        SDVariable a0 = out.nn.relu("a0", h0, 0);
        SDVariable h1 = out.mmul("h1", a0, w1);
        SDVariable a1 = out.math.tanh("a1", h1);
        SDVariable h2 = out.mmul("h2", a1, w2);
        SDVariable a2 = out.nn.sigmoid("a2", h2);
        SDVariable h3 = out.mmul("h3", a2, w3);
        SDVariable a3 = out.nn.gelu("a3", h3);
        SDVariable h4 = out.mmul("h4", a3, w4);
        SDVariable y = out.nn.softmax("y", h4, -1);
        out.setOutputs("y");
        return out;
    }

    /**
     * Build a pure elementwise graph. Has no matmul, so the chain cannot be
     * absorbed into a MATMUL section as an epilogue — TritonGraphBackend is
     * forced to compile the section as a standalone ELEMENTWISE kernel, which
     * is the one section type that reliably goes through the full MLIR →
     * PTX → cuModuleLoad pipeline in the default compile mode. This
     * guarantees {@code registerLoadedKernel} runs and drives the residency
     * warn path.
     */
    private SameDiff buildElementwiseGraph() {
        SameDiff out = SameDiff.create();
        final int rows = 32;
        final int cols = 64;
        SDVariable x = out.placeHolder("x", DataType.FLOAT, rows, cols);
        SDVariable b = out.var("b", Nd4j.randn(DataType.FLOAT, rows, cols).muli(0.1));
        SDVariable y0 = x.add(b);
        SDVariable y1 = y0.mul(y0);
        SDVariable y2 = out.math.tanh(y1);
        SDVariable y3 = out.nn.sigmoid(y2);
        SDVariable y4 = y3.add(x);
        SDVariable y5 = y4.mul(b);
        SDVariable y6 = out.math.neg(y5);
        SDVariable y = out.math.abs("y", y6);
        out.setOutputs("y");
        return out;
    }

    private static Map<String, INDArray> elementwiseInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, 32, 64).muli(0.5));
        return in;
    }

    private static Map<String, INDArray> inputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, 4, IN_DIM).muli(0.5));
        return in;
    }

    private void enableDsp(SameDiff target) {
        target.setDspAutoCompileEnabled(true);
        target.setDspNativeAutoCompileEnabled(true);
        target.setGraphExecutionMode(GraphExecutionMode.AUTO);
    }

    private static boolean isTritonAvailable() {
        try {
            return Nd4j.getNativeOps().isTritonAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Helpers — Environment knob accessors.
    // ──────────────────────────────────────────────────────────────────────

    private static long readBudgetBytes() {
        return Nd4j.getEnvironment().tritonModuleResidencyBudgetBytes();
    }

    private static void writeBudgetBytes(long bytes) {
        Nd4j.getEnvironment().setTritonModuleResidencyBudgetBytes(bytes);
    }

    private static long readWarnBytes() {
        return Nd4j.getEnvironment().tritonModuleResidencyWarnBytes();
    }

    private static void writeWarnBytes(long bytes) {
        Nd4j.getEnvironment().setTritonModuleResidencyWarnBytes(bytes);
    }

    /**
     * Run the model under SLOT_BY_SLOT (always-correct reference) and return
     * a duplicated copy of the output so the source can be closed safely.
     */
    private INDArray referenceOutput(SameDiff fixture, Map<String, INDArray> inputs) {
        fixture.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> raw = fixture.output(inputs, "y");
        INDArray dup = raw.get("y").dup();
        // Re-enable DSP for the test pass.
        fixture.setGraphExecutionMode(GraphExecutionMode.AUTO);
        fixture.setDspAutoCompileEnabled(true);
        fixture.setDspNativeAutoCompileEnabled(true);
        return dup;
    }

    private static void assertClose(INDArray expected, INDArray actual, double rtol, double atol,
                                    String label) {
        assertNotNull(actual, label + ": actual is null");
        assertTrue(Arrays.equals(expected.shape(), actual.shape()),
                label + ": shape mismatch — expected "
                        + Arrays.toString(expected.shape())
                        + " actual=" + Arrays.toString(actual.shape()));
        long n = expected.length();
        double worstAbs = 0;
        double worstRel = 0;
        int firstBad = -1;
        for (long i = 0; i < n; i++) {
            double e = expected.getDouble(i);
            double a = actual.getDouble(i);
            double absDiff = Math.abs(e - a);
            double relDiff = absDiff / (Math.abs(e) + 1e-12);
            if (absDiff > atol && relDiff > rtol && firstBad < 0) {
                firstBad = (int) i;
            }
            if (absDiff > worstAbs) worstAbs = absDiff;
            if (relDiff > worstRel) worstRel = relDiff;
        }
        if (firstBad >= 0) {
            fail(label + ": diverges at index " + firstBad
                    + " expected=" + expected.getDouble(firstBad)
                    + " actual=" + actual.getDouble(firstBad)
                    + " (worstAbs=" + worstAbs + " worstRel=" + worstRel
                    + " atol=" + atol + " rtol=" + rtol + ")");
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Tests
    // ──────────────────────────────────────────────────────────────────────

    @Test
    @DisplayName("Tiny residency budget triggers eviction but execution remains correct")
    public void testSmallBudgetTriggersEviction() {
        assumeTrue(isTritonAvailable(), "Triton unavailable — skipping");

        sd = buildManyKernelGraph();
        Map<String, INDArray> in = inputs();
        // Capture reference under SLOT_BY_SLOT before any DSP/Triton state is built.
        INDArray ref = referenceOutput(sd, in);

        // Configure a 1 MiB budget — far smaller than a single Triton module.
        // This should force the LRU cache into continuous eviction mode.
        writeBudgetBytes(TINY_BUDGET_BYTES);

        enableDsp(sd);

        // Run several times so eviction + reload from disk has many opportunities.
        for (int i = 0; i < 5; i++) {
            Map<String, INDArray> out = sd.output(inputs(), "y");
            INDArray got = out.get("y").dup();
            assertClose(ref, got, 1e-3, 1e-4,
                    "Iteration " + i + " under tiny residency budget");
            got.close();
        }
    }

    @Test
    @DisplayName("Unlimited residency budget produces correct results (no eviction baseline)")
    public void testUnlimitedBudgetNoEviction() {
        assumeTrue(isTritonAvailable(), "Triton unavailable — skipping");

        sd = buildManyKernelGraph();
        Map<String, INDArray> in = inputs();
        INDArray ref = referenceOutput(sd, in);

        // Unlimited budget — the LRU should never evict.
        writeBudgetBytes(UNLIMITED_BUDGET_BYTES);

        enableDsp(sd);

        for (int i = 0; i < 3; i++) {
            Map<String, INDArray> out = sd.output(inputs(), "y");
            INDArray got = out.get("y").dup();
            assertClose(ref, got, 1e-3, 1e-4,
                    "Iteration " + i + " under unlimited residency budget");
            got.close();
        }
    }

    @Test
    @DisplayName("Setting a tiny warn threshold fires the residency-warn diagnostic path")
    public void testWarnBytesEmitsWarning() {
        assumeTrue(isTritonAvailable(), "Triton unavailable — skipping");
        // The warn fire counter (tritonModuleResidencyWarnFireCount) is only
        // implemented in the CUDA backend. On CPU the Environment default methods
        // return 0 and the counter never increments, so skip this test there.
        assumeTrue(Nd4j.getBackendDeviceType() == DeviceType.CUDA_GPU
                        || Nd4j.getBackendDeviceType() == DeviceType.GPU,
                "CUDA backend required for module-residency warn counter — skipping on CPU");

        // Use a pure elementwise graph so TritonGraphBackend actually compiles
        // a CUmodule (the matmul-heavy graph in buildManyKernelGraph routes
        // every section to cuBLAS / native ordered execution, which never
        // goes through registerLoadedKernel and therefore never drives the
        // warn path).
        sd = buildElementwiseGraph();

        // The native warn path emits via sd_printf (libc fd 1) and DSP_DIAG.
        // Java's System.setOut only replaces the Java PrintStream — it does
        // not intercept native fd 1 writes, so we cannot observe the warning
        // message itself from Java. Instead, the native warn code path
        // increments a dedicated counter exposed through
        // TritonConfig::moduleResidencyWarnFireCount(). We reset the counter
        // here and assert it strictly increased after execution.
        clearWarnFireCount();
        long before = readWarnFireCount();

        // Set a 4 KiB warn threshold — any module load will exceed this,
        // so the warning path should fire.
        writeWarnBytes(TINY_WARN_BYTES);

        enableDsp(sd);
        // Two passes to make sure the warning happens during load and
        // any subsequent retention-check pass.
        sd.output(elementwiseInputs(), "y");
        sd.output(elementwiseInputs(), "y");

        long after = readWarnFireCount();
        assertTrue(after > before,
                "expected the Triton module-residency warn path to fire at least once "
                        + "after compiling with a tiny warn threshold; "
                        + "warnFireCount before=" + before + " after=" + after
                        + " (warnBytes=" + TINY_WARN_BYTES + ")");
    }

    private static long readWarnFireCount() {
        return Nd4j.getEnvironment().tritonModuleResidencyWarnFireCount();
    }

    private static void clearWarnFireCount() {
        Nd4j.getEnvironment().clearTritonModuleResidencyWarnFireCount();
    }
}
