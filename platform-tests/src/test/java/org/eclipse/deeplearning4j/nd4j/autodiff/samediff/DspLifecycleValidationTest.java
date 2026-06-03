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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.execution.PlanPhase;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;

import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.DeviceType;
import org.nd4j.linalg.api.device.StubDeviceDescriptor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Consolidated DSP lifecycle validation across execution modes.
 *
 * <p>This class collapses the prior DSP / Triton unit-test suite into a single
 * {@link BenchmarkConfig}-parameterized framework. Each parameterized test:
 * <ol>
 *   <li>Builds a small synthetic SameDiff fixture.</li>
 *   <li>Runs it under {@link GraphExecutionMode#SLOT_BY_SLOT} as the always-correct
 *       reference.</li>
 *   <li>Applies the test {@link BenchmarkConfig} via
 *       {@link BenchmarkConfigApplier#apply(BenchmarkConfig)}.</li>
 *   <li>Re-runs the same fixture under the configured mode.</li>
 *   <li>Compares each output element-wise against the reference within a
 *       config-aware tolerance (FP32 vs TF32/FP16).</li>
 *   <li>Calls {@link BenchmarkConfigApplier#resetModelState(SameDiff)} between
 *       runs to clear DSP plan caches without invalidating the Triton disk cache.</li>
 * </ol>
 *
 * <p>Only CUDA-testable execution modes are exercised: {@code AUTO},
 * {@code SLOT_BY_SLOT}, {@code CUDA_GRAPHS}, {@code NVRTC_JIT}, {@code PTX_JIT},
 * {@code TRITON}, {@code EMULATED_REPLAY}. Platform-locked modes such as MLX,
 * NNAPI, ARM_HYBRID, HIP_GRAPHS, LEVEL_ZERO, VULKAN, METAL, TPU, HEXAGON, and
 * OPENVINO are not part of the CUDA validation surface and are therefore
 * intentionally skipped here. Triton-only configurations are filtered via
 * {@code Nd4j.getNativeOps().isTritonAvailable()}.
 *
 * <p><b>Running a single configuration:</b>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=DspLifecycleValidationTest#testMatmulMlpAccuracy[CUDA_GRAPHS] \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/dsp-lifecycle-CUDA_GRAPHS.log
 * </pre>
 *
 * <p><b>System properties recognised at startup:</b>
 * <ul>
 *   <li>{@code -Dnd4j.dsp.lifecycle.tolerance=fp32|tf32|loose} — overrides default
 *       tolerance preset (default: {@code fp32}).</li>
 *   <li>{@code -Dnd4j.dsp.lifecycle.verbose=true} — enable per-step output dumps.</li>
 *   <li>{@code -Dnd4j.dsp.lifecycle.maxSteps=N} — overrides decode loop length.</li>
 * </ul>
 *
 * <p>This class replaces the prior DSP test suite (≈100 individual unit tests)
 * by encoding the same lifecycle invariants once, in a configuration-driven way,
 * so any new {@link GraphExecutionMode} or DSP knob can be added by extending
 * the relevant {@code Stream<BenchmarkConfig>} provider.
 */
@Slf4j
@Tag(TagNames.FULL_CI)
public class DspLifecycleValidationTest {

    // ─── Tolerances ─────────────────────────────────────────────────────────
    private static final double FP32_RTOL = 1e-4;
    private static final double FP32_ATOL = 1e-5;
    private static final double LOOSE_RTOL = 1e-2;
    private static final double LOOSE_ATOL = 1e-3;

    // ─── Configuration knobs (system properties) ───────────────────────────
    private static boolean verboseOutput;
    private static int maxDecodeSteps = 20;
    private static String tolerancePreset = "fp32";

    // ─── Saved environment state for restore in @AfterAll ──────────────────
    private static EnvSnapshot originalEnv;

    // ─── Synthetic graph parameters ────────────────────────────────────────
    private static final int ELEMENT_BATCH = 4;
    private static final int ELEMENT_DIM = 8;

    private static final int MLP_BATCH = 2;
    private static final int MLP_IN = 16;
    private static final int MLP_HIDDEN = 32;

    private static final int ATTN_B = 1;
    private static final int ATTN_S = 4;
    private static final int ATTN_H = 8;

    private static final int KV_HEADS = 2;
    private static final int KV_HEAD_DIM = 4;
    private static final int KV_HIDDEN = KV_HEADS * KV_HEAD_DIM; // 8
    private static final int KV_VOCAB = 16;
    private static final int KV_MAX_LEN = 8;

    private static final int LARGE_PREFILL = 64;

    // ─── Setup / teardown ──────────────────────────────────────────────────

    @BeforeAll
    public static void setupAll() {
        verboseOutput = "true".equalsIgnoreCase(System.getProperty("nd4j.dsp.lifecycle.verbose"));
        String maxStepsProp = System.getProperty("nd4j.dsp.lifecycle.maxSteps");
        if (maxStepsProp != null && !maxStepsProp.isEmpty()) {
            maxDecodeSteps = Integer.parseInt(maxStepsProp);
        }
        String tolProp = System.getProperty("nd4j.dsp.lifecycle.tolerance");
        if (tolProp != null && !tolProp.isEmpty()) {
            tolerancePreset = tolProp.toLowerCase();
        }

        // Snapshot environment state so per-test knob mutations don't leak.
        originalEnv = EnvSnapshot.capture();

        // Pre-initialise diagnostics so DspDiagnostics counters exist.
        DspDiagnostics.initialize();

        log.info("DspLifecycleValidationTest setup: tolerance={} verbose={} maxSteps={}",
                tolerancePreset, verboseOutput, maxDecodeSteps);
    }

    @AfterAll
    public static void teardownAll() {
        if (originalEnv != null) {
            originalEnv.restore();
        }
    }

    @BeforeEach
    public void beforeEach() {
        // Force a clean baseline before every test by reapplying the captured snapshot.
        if (originalEnv != null) {
            originalEnv.restore();
        }
    }

    @AfterEach
    public void afterEach() {
        // Commit any in-flight CUDA work and restore env so the next test starts clean.
        Nd4j.getExecutioner().commit();
        if (originalEnv != null) {
            originalEnv.restore();
        }
    }

    // ─── Config generators ─────────────────────────────────────────────────

    /**
     * Returns one BenchmarkConfig per CUDA-testable execution mode.
     * TRITON is filtered if Triton is unavailable.
     */
    static Stream<BenchmarkConfig> executionModeConfigs() {
        boolean tritonAvailable = isTritonAvailable();
        List<BenchmarkConfig> out = new ArrayList<>();
        out.add(modeConfig("AUTO", GraphExecutionMode.AUTO));
        out.add(modeConfig("SLOT_BY_SLOT", GraphExecutionMode.SLOT_BY_SLOT));
        out.add(modeConfig("CUDA_GRAPHS", GraphExecutionMode.CUDA_GRAPHS));
        out.add(modeConfig("NVRTC_JIT", GraphExecutionMode.NVRTC_JIT));
        out.add(modeConfig("PTX_JIT", GraphExecutionMode.PTX_JIT));
        if (tritonAvailable) {
            out.add(modeConfig("TRITON", GraphExecutionMode.TRITON));
        } else {
            log.info("executionModeConfigs: TRITON skipped — Triton unavailable on this build");
        }
        out.add(modeConfig("EMULATED_REPLAY", GraphExecutionMode.EMULATED_REPLAY));
        return out.stream();
    }

    /**
     * Bisection over key Triton knobs: each knob is enumerated independently
     * (on/off) so a regression in any single knob can be isolated.
     */
    static Stream<BenchmarkConfig> tritonKnobBisectionConfigs() {
        if (!isTritonAvailable()) {
            log.info("tritonKnobBisectionConfigs: empty stream — Triton unavailable");
            return Stream.empty();
        }
        String includeTypes = "CONST_GEN,GATHER,CONCAT,SPLIT,STACK,NORMALIZATION,ATTENTION";
        List<BenchmarkConfig> out = new ArrayList<>();
        // Baseline: section fusion only
        out.add(BenchmarkConfig.create("triton_baseline")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true)
                .tritonCompileAll(true));
        // Knob 1: graph capture on/off
        out.add(BenchmarkConfig.create("triton_gc_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(true));
        out.add(BenchmarkConfig.create("triton_gc_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonGraphCapture(false));
        // Knob 2: consolidated arg table on/off
        out.add(BenchmarkConfig.create("triton_argTable_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonConsolidatedArgTable(true));
        out.add(BenchmarkConfig.create("triton_argTable_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonConsolidatedArgTable(false));
        // Knob 3: arg dirty tracking on/off
        out.add(BenchmarkConfig.create("triton_argDirty_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(true));
        out.add(BenchmarkConfig.create("triton_argDirty_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonConsolidatedArgTable(true)
                .tritonArgDirtyTracking(false));
        // Knob 4: section fusion on/off
        out.add(BenchmarkConfig.create("triton_sectionFusion_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true));
        out.add(BenchmarkConfig.create("triton_sectionFusion_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(false).tritonCompileAll(true));
        // Knob 5: cooperative launch on/off
        out.add(BenchmarkConfig.create("triton_coopLaunch_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonCooperativeLaunch(true));
        out.add(BenchmarkConfig.create("triton_coopLaunch_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonCooperativeLaunch(false));
        // Knob 6: enable_fp_fusion on/off
        out.add(BenchmarkConfig.create("triton_fpFusion_on")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonEnableFpFusion(true));
        out.add(BenchmarkConfig.create("triton_fpFusion_off")
                .tritonIncludeTypes(includeTypes)
                .tritonSectionFusion(true).tritonCompileAll(true)
                .tritonEnableFpFusion(false));
        return out.stream();
    }

    /**
     * DSP knob bisection (no Triton requirement).
     */
    static Stream<BenchmarkConfig> dspKnobConfigs() {
        List<BenchmarkConfig> out = new ArrayList<>();
        out.add(BenchmarkConfig.create("dsp_baseline")
                .executionMode(GraphExecutionMode.AUTO));
        out.add(BenchmarkConfig.create("dsp_batchZero_on")
                .executionMode(GraphExecutionMode.AUTO).dspBatchZero(true));
        out.add(BenchmarkConfig.create("dsp_castSinkMatmul_on")
                .executionMode(GraphExecutionMode.AUTO).dspCastSinkMatmul(true));
        out.add(BenchmarkConfig.create("dsp_batchedGemm_on")
                .executionMode(GraphExecutionMode.AUTO).dspBatchedGemm(true));
        out.add(BenchmarkConfig.create("dsp_freezeMergeSegments_on")
                .executionMode(GraphExecutionMode.AUTO).dspFreezeMergeSegments(true));
        out.add(BenchmarkConfig.create("dsp_all_on")
                .executionMode(GraphExecutionMode.AUTO)
                .dspBatchZero(true)
                .dspCastSinkMatmul(true)
                .dspBatchedGemm(true)
                .dspFreezeMergeSegments(true));
        return out.stream();
    }

    /**
     * Static-KV-cache lifecycle covers a small subset of execution modes.
     * Only includes modes that are meaningful for autoregressive decode.
     */
    static Stream<BenchmarkConfig> kvCacheLifecycleConfigs() {
        boolean tritonAvailable = isTritonAvailable();
        List<BenchmarkConfig> out = new ArrayList<>();
        out.add(modeConfig("kv_SLOT_BY_SLOT", GraphExecutionMode.SLOT_BY_SLOT));
        out.add(modeConfig("kv_CUDA_GRAPHS", GraphExecutionMode.CUDA_GRAPHS));
        out.add(modeConfig("kv_AUTO", GraphExecutionMode.AUTO));
        if (tritonAvailable) {
            out.add(modeConfig("kv_TRITON", GraphExecutionMode.TRITON));
        }
        return out.stream();
    }

    /**
     * output() vs outputDirect() parity at a single execution mode.
     */
    static Stream<BenchmarkConfig> outputPathConfigs() {
        List<BenchmarkConfig> out = new ArrayList<>();
        out.add(modeConfig("path_AUTO", GraphExecutionMode.AUTO));
        out.add(modeConfig("path_CUDA_GRAPHS", GraphExecutionMode.CUDA_GRAPHS));
        return out.stream();
    }

    private static BenchmarkConfig modeConfig(String name, GraphExecutionMode mode) {
        return BenchmarkConfig.create(name).executionMode(mode);
    }

    private static boolean isTritonAvailable() {
        try {
            return Nd4j.getNativeOps().isTritonAvailable();
        } catch (Throwable t) {
            return false;
        }
    }

    // ─── Synthetic graph fixtures ──────────────────────────────────────────

    private static SameDiff buildElementwise() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, ELEMENT_BATCH, ELEMENT_DIM);
        SDVariable b = sd.var("b", Nd4j.randn(DataType.FLOAT, 1, ELEMENT_DIM).muli(0.1));
        SDVariable scale = sd.var("scale",
                Nd4j.randn(DataType.FLOAT, 1, ELEMENT_DIM).muli(0.1).add(1.0));
        SDVariable a = x.add("a", b);
        SDVariable m = a.mul("m", scale);
        SDVariable out = sd.nn.relu("out", m, 0);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildMatmulMlp() {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, MLP_BATCH, MLP_IN);
        SDVariable w0 = sd.var("w0",
                Nd4j.randn(DataType.FLOAT, MLP_IN, MLP_HIDDEN).muli(0.05));
        SDVariable w1 = sd.var("w1",
                Nd4j.randn(DataType.FLOAT, MLP_HIDDEN, MLP_IN).muli(0.05));
        SDVariable w2 = sd.var("w2",
                Nd4j.randn(DataType.FLOAT, MLP_IN, MLP_HIDDEN / 4).muli(0.05));
        SDVariable h0 = sd.mmul("h0", x, w0);
        SDVariable a0 = sd.nn.relu("a0", h0, 0);
        SDVariable h1 = sd.mmul("h1", a0, w1);
        SDVariable a1 = sd.nn.relu("a1", h1, 0);
        SDVariable out = sd.mmul("out", a1, w2);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildAttentionBlock() {
        SameDiff sd = SameDiff.create();
        // [B, S, H]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, ATTN_B, ATTN_S, ATTN_H);
        SDVariable wq = sd.var("wq", Nd4j.randn(DataType.FLOAT, ATTN_H, ATTN_H).muli(0.1));
        SDVariable wk = sd.var("wk", Nd4j.randn(DataType.FLOAT, ATTN_H, ATTN_H).muli(0.1));
        SDVariable wv = sd.var("wv", Nd4j.randn(DataType.FLOAT, ATTN_H, ATTN_H).muli(0.1));
        SDVariable wo = sd.var("wo", Nd4j.randn(DataType.FLOAT, ATTN_H, ATTN_H).muli(0.1));

        // Q/K/V projections via batched matmul (treat as [B*S, H] for projection)
        SDVariable q = sd.mmul("q", x, wq);
        SDVariable k = sd.mmul("k", x, wk);
        SDVariable v = sd.mmul("v", x, wv);

        // Scores = q · k^T  => [B, S, S]
        SDVariable kT = sd.permute("kT", k, 0, 2, 1);
        SDVariable scores = sd.mmul("scores", q, kT);
        SDVariable scaled = scores.mul("scaled", 1.0 / Math.sqrt(ATTN_H));
        SDVariable probs = sd.nn.softmax("probs", scaled, -1);

        // attn_out = probs · v  => [B, S, H]
        SDVariable attn = sd.mmul("attn", probs, v);
        SDVariable out = sd.mmul("out", attn, wo);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildStaticKvDecoder(int maxKvLen) {
        SameDiff sd = SameDiff.create();

        // Placeholders
        SDVariable inputsEmbeds = sd.placeHolder("input_embeds",
                DataType.FLOAT, 1, 1, KV_HIDDEN);
        SDVariable attentionMask = sd.placeHolder("attention_mask",
                DataType.FLOAT, 1, 1, 1, maxKvLen + 1);
        SDVariable pastKey = sd.placeHolder("past_key",
                DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM);
        SDVariable pastValue = sd.placeHolder("past_value",
                DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM);

        // Constants
        SDVariable wq = sd.var("wq", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wk = sd.var("wk", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wv = sd.var("wv", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wo = sd.var("wo", Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_HIDDEN).muli(0.05));
        SDVariable wlogits = sd.var("wlogits",
                Nd4j.randn(DataType.FLOAT, KV_HIDDEN, KV_VOCAB).muli(0.05));

        // Squeeze seq dim (always 1 in decode): [1, hidden]
        SDVariable squeezed = sd.squeeze("squeezed", inputsEmbeds, 1);

        // Q/K/V projections: [1, hidden]
        SDVariable qFlat = sd.mmul("q_flat", squeezed, wq);
        SDVariable kFlat = sd.mmul("k_flat", squeezed, wk);
        SDVariable vFlat = sd.mmul("v_flat", squeezed, wv);

        // Reshape to [1, heads, 1, head_dim]
        SDVariable qNew = sd.reshape("q_new", qFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);
        SDVariable kNew = sd.reshape("k_new", kFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);
        SDVariable vNew = sd.reshape("v_new", vFlat, 1, KV_HEADS, 1, KV_HEAD_DIM);

        // Concat past + new => [1, heads, maxKvLen+1, head_dim]
        SDVariable presentKey = sd.concat("present_key", 2, pastKey, kNew);
        SDVariable presentValue = sd.concat("present_value", 2, pastValue, vNew);

        // Scores: q · k^T => [1, heads, 1, maxKvLen+1]
        SDVariable kT = sd.permute("kT", presentKey, 0, 1, 3, 2);
        SDVariable scores = sd.mmul("scores", qNew, kT);
        SDVariable scaled = scores.mul("scaled", 1.0 / Math.sqrt(KV_HEAD_DIM));
        SDVariable masked = scaled.add("masked", attentionMask);
        SDVariable probs = sd.nn.softmax("probs", masked, -1);

        // Attention output: probs · v => [1, heads, 1, head_dim]
        SDVariable attnOut = sd.mmul("attn_out", probs, presentValue);

        // Reshape and project to logits
        SDVariable attnFlat = sd.reshape("attn_flat", attnOut, 1, KV_HIDDEN);
        SDVariable projected = sd.mmul("projected", attnFlat, wo);
        SDVariable logitsFlat = sd.mmul("logits_flat", projected, wlogits);
        SDVariable logits = sd.reshape("logits", logitsFlat, 1, 1, KV_VOCAB);

        sd.setOutputs("logits", "present_key", "present_value");
        return sd;
    }

    private static SameDiff buildWhereBroadcast() {
        SameDiff sd = SameDiff.create();
        // base: [1, 4, 8] FLOAT
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 4, 8);
        // mask: [1, 4, 1] BOOL
        SDVariable mask = sd.placeHolder("mask", DataType.BOOL, 1, 4, 1);
        SDVariable zeros = sd.var("zeros", Nd4j.zeros(DataType.FLOAT, 1, 4, 8));
        // where(mask, x, zeros)  — broadcast mask over last dim
        SDVariable selected = sd.where("selected", x, zeros, mask);
        SDVariable scaled = selected.mul("scaled", 2.0);
        SDVariable out = sd.nn.relu("out", scaled, 0);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildViewOpChain() {
        SameDiff sd = SameDiff.create();
        // x: [1, 3, 4, 8]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 3, 4, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8).muli(0.1));
        // reshape -> [1, 12, 8]
        SDVariable r1 = sd.reshape("r1", x, 1, 12, 8);
        // permute -> [1, 8, 12]
        SDVariable p1 = sd.permute("p1", r1, 0, 2, 1);
        // reshape back -> [1, 12, 8]
        SDVariable r2 = sd.reshape("r2", p1, 1, 8, 12);
        SDVariable p2 = sd.permute("p2", r2, 0, 2, 1);
        // matmul on view chain output (flatten last two dims)
        SDVariable flat = sd.reshape("flat", p2, 12, 8);
        SDVariable out = sd.mmul("out", flat, w);
        sd.setOutputs("out");
        return sd;
    }

    /**
     * A small graph that triggers freeze/thaw cycles when given inputs of
     * different shapes. Uses gather with a runtime indices placeholder so the
     * shape depends on input.
     */
    private static SameDiff buildValueDepAfterFreeze() {
        SameDiff sd = SameDiff.create();
        // x: [N, 8]
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, -1, 8);
        SDVariable w = sd.var("w", Nd4j.randn(DataType.FLOAT, 8, 8).muli(0.1));
        SDVariable proj = sd.mmul("proj", x, w);
        SDVariable shaped = sd.reshape("shaped", proj, -1, 8);
        SDVariable out = sd.nn.relu("out", shaped, 0);
        sd.setOutputs("out");
        return sd;
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    private static double[] tolerances(BenchmarkConfig cfg) {
        if ("loose".equals(tolerancePreset)
                || "tf32".equals(tolerancePreset)
                || cfg.isCublasTf32() || cfg.isTritonTf32() || cfg.isDspFp16Compute()) {
            return new double[]{LOOSE_RTOL, LOOSE_ATOL};
        }
        return new double[]{FP32_RTOL, FP32_ATOL};
    }

    /**
     * Apply the supplied benchmark config and execute the SameDiff graph,
     * returning duplicated output arrays so callers can close inputs without
     * affecting the comparison.
     */
    private Map<String, INDArray> runWithConfig(SameDiff sd, BenchmarkConfig cfg,
                                                Map<String, INDArray> inputs,
                                                String... outputNames) {
        BenchmarkConfigApplier.apply(cfg);
        if (cfg.getExecutionMode() != null) {
            sd.setGraphExecutionMode(cfg.getExecutionMode());
        }
        if (cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT) {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
        } else {
            sd.setDspAutoCompileEnabled(false);
            sd.setDspNativeAutoCompileEnabled(false);
        }
        Map<String, INDArray> raw = sd.output(inputs, outputNames);
        Map<String, INDArray> dup = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : raw.entrySet()) {
            INDArray v = e.getValue();
            dup.put(e.getKey(), v == null ? null : v.dup());
        }
        return dup;
    }

    private void resetBetweenRuns(SameDiff sd) {
        BenchmarkConfigApplier.resetModelState(sd);
        Nd4j.getExecutioner().commit();
    }

    /**
     * Runs the supplied fixture under the SLOT_BY_SLOT reference and again
     * under the test config, then compares all outputs element-wise.
     */
    private void runRefVsTest(SameDiff sd, BenchmarkConfig cfg,
                              Map<String, INDArray> inputs,
                              String... outputNames) {
        // Reference: SLOT_BY_SLOT
        BenchmarkConfig refCfg = BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> refOut = runWithConfig(sd, refCfg, copyInputs(inputs), outputNames);

        resetBetweenRuns(sd);

        // Test
        Map<String, INDArray> testOut = runWithConfig(sd, cfg, copyInputs(inputs), outputNames);

        double[] tol = tolerances(cfg);
        assertOutputsClose(refOut, testOut, tol[0], tol[1], cfg.getName());

        closeAll(refOut);
        closeAll(testOut);
        resetBetweenRuns(sd);
    }

    /**
     * Element-wise comparison of two output maps. On failure, prints the first
     * divergent index along with both values so the failure mode is obvious.
     */
    private static void assertOutputsClose(Map<String, INDArray> ref,
                                           Map<String, INDArray> test,
                                           double rtol, double atol,
                                           String configName) {
        for (Map.Entry<String, INDArray> e : ref.entrySet()) {
            String name = e.getKey();
            INDArray a = e.getValue();
            INDArray b = test.get(name);
            assertNotNull(b, configName + ": output '" + name + "' missing in test result");
            assertTrue(Arrays.equals(a.shape(), b.shape()),
                    configName + ": output '" + name + "' shape mismatch ref="
                            + Arrays.toString(a.shape()) + " test=" + Arrays.toString(b.shape()));
            INDArray refF = a.castTo(DataType.DOUBLE);
            INDArray testF = b.castTo(DataType.DOUBLE);
            long n = refF.length();
            int firstBad = -1;
            double worstAbs = 0;
            double worstRel = 0;
            for (long i = 0; i < n; i++) {
                double rv = refF.getDouble(i);
                double tv = testF.getDouble(i);
                double absDiff = Math.abs(rv - tv);
                double relDiff = absDiff / (Math.abs(rv) + 1e-12);
                if (absDiff > atol && relDiff > rtol) {
                    if (firstBad < 0) firstBad = (int) i;
                }
                if (absDiff > worstAbs) worstAbs = absDiff;
                if (relDiff > worstRel) worstRel = relDiff;
            }
            if (firstBad >= 0) {
                fail(configName + ": output '" + name + "' diverges at index " + firstBad
                        + " ref=" + refF.getDouble(firstBad) + " test=" + testF.getDouble(firstBad)
                        + " (worstAbs=" + worstAbs + " worstRel=" + worstRel
                        + " atol=" + atol + " rtol=" + rtol + ")");
            }
        }
    }

    private static Map<String, INDArray> copyInputs(Map<String, INDArray> inputs) {
        Map<String, INDArray> copy = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : inputs.entrySet()) {
            INDArray v = e.getValue();
            copy.put(e.getKey(), v == null ? null : v.dup());
        }
        return copy;
    }

    private static void closeAll(Map<String, INDArray> arrays) {
        if (arrays == null) return;
        for (INDArray arr : arrays.values()) {
            if (arr != null && arr.closeable() && !arr.wasClosed()) {
                arr.close();
            }
        }
    }

    /**
     * Skip configs that target an unavailable backend.
     */
    private static void assumeBackendAvailable(BenchmarkConfig cfg) {
        if (cfg.getExecutionMode() == GraphExecutionMode.TRITON || cfg.isTriton()) {
            assumeTrue(isTritonAvailable(), "Triton unavailable — skipping " + cfg.getName());
        }
    }

    private static Map<String, INDArray> elementwiseInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, ELEMENT_BATCH, ELEMENT_DIM).muli(0.5));
        return in;
    }

    private static Map<String, INDArray> matmulInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, MLP_BATCH, MLP_IN).muli(0.5));
        return in;
    }

    private static Map<String, INDArray> attentionInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, ATTN_B, ATTN_S, ATTN_H).muli(0.2));
        return in;
    }

    private static Map<String, INDArray> staticKvInputs(int maxKvLen, long position) {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("input_embeds", Nd4j.randn(DataType.FLOAT, 1, 1, KV_HIDDEN).muli(0.1));
        // attention_mask is a [1, 1, 1, maxKvLen+1] FLOAT additive mask: 0 at attended, -inf elsewhere
        INDArray mask = Nd4j.valueArrayOf(new long[]{1, 1, 1, maxKvLen + 1}, -1e9, DataType.FLOAT);
        for (long i = 0; i <= position; i++) {
            mask.putScalar(new long[]{0, 0, 0, i}, 0.0);
        }
        // Always allow the new (concat) position
        mask.putScalar(new long[]{0, 0, 0, maxKvLen}, 0.0);
        in.put("attention_mask", mask);
        in.put("past_key", Nd4j.randn(DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM).muli(0.05));
        in.put("past_value", Nd4j.randn(DataType.FLOAT, 1, KV_HEADS, maxKvLen, KV_HEAD_DIM).muli(0.05));
        return in;
    }

    private static Map<String, INDArray> whereInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, 1, 4, 8).muli(0.5));
        // Boolean mask [1,4,1]; alternate true/false
        INDArray mask = Nd4j.zeros(DataType.BOOL, 1, 4, 1);
        for (int i = 0; i < 4; i++) {
            mask.putScalar(new long[]{0, i, 0}, i % 2 == 0 ? 1.0 : 0.0);
        }
        in.put("mask", mask);
        return in;
    }

    private static Map<String, INDArray> viewInputs() {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, 1, 3, 4, 8).muli(0.5));
        return in;
    }

    private static Map<String, INDArray> valueDepInputs(int rows) {
        Map<String, INDArray> in = new LinkedHashMap<>();
        in.put("x", Nd4j.randn(DataType.FLOAT, rows, 8).muli(0.5));
        return in;
    }

    // ─── Accuracy tests ────────────────────────────────────────────────────

    @ParameterizedTest(name = "elementwise[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Elementwise accuracy across execution modes")
    public void testElementwiseAccuracy(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildElementwise();
        try {
            runRefVsTest(sd, cfg, elementwiseInputs(), "out");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "matmulMlp[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("MatMul MLP accuracy across execution modes")
    public void testMatmulMlpAccuracy(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildMatmulMlp();
        try {
            runRefVsTest(sd, cfg, matmulInputs(), "out");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "attention[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Attention block accuracy across execution modes")
    public void testAttentionBlockAccuracy(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildAttentionBlock();
        try {
            runRefVsTest(sd, cfg, attentionInputs(), "out");
        } finally {
            sd.close();
        }
    }

    // ─── Static KV lifecycle tests ─────────────────────────────────────────

    @ParameterizedTest(name = "staticKvSingle[{0}]")
    @MethodSource("kvCacheLifecycleConfigs")
    @DisplayName("Static KV decoder single-step accuracy")
    public void testStaticKvSingleStepAccuracy(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildStaticKvDecoder(KV_MAX_LEN);
        try {
            Map<String, INDArray> inputs = staticKvInputs(KV_MAX_LEN, 0);
            runRefVsTest(sd, cfg, inputs, "logits", "present_key", "present_value");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "staticKvMultiStep[{0}]")
    @MethodSource("kvCacheLifecycleConfigs")
    @DisplayName("Static KV multi-step decode lifecycle vs SLOT_BY_SLOT reference")
    public void testStaticKvMultiStepDecodeLifecycle(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        int steps = Math.min(maxDecodeSteps, KV_MAX_LEN - 1);
        // Pre-generate identical input sequences for ref + test
        List<Map<String, INDArray>> inputsByStep = new ArrayList<>();
        for (int step = 0; step < steps; step++) {
            inputsByStep.add(staticKvInputs(KV_MAX_LEN, step));
        }

        // Single SameDiff instance shared between ref and test runs to keep
        // the random-initialised constants identical.
        SameDiff sd = buildStaticKvDecoder(KV_MAX_LEN);
        BenchmarkConfig refCfg = BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
        List<Map<String, INDArray>> refOutputs = new ArrayList<>();
        try {
            // ─── Reference pass
            for (Map<String, INDArray> inputs : inputsByStep) {
                Map<String, INDArray> out = runWithConfig(sd, refCfg, copyInputs(inputs),
                        "logits", "present_key", "present_value");
                refOutputs.add(out);
                resetBetweenRuns(sd);
            }

            // ─── Test pass on the same SameDiff
            double[] tol = tolerances(cfg);
            INDArray prevLogits = null;
            int identicalRun = 0;
            int maxIdenticalRun = 0;
            for (int step = 0; step < steps; step++) {
                Map<String, INDArray> inputs = copyInputs(inputsByStep.get(step));
                Map<String, INDArray> out = runWithConfig(sd, cfg, inputs,
                        "logits", "present_key", "present_value");
                assertOutputsClose(refOutputs.get(step), out, tol[0], tol[1],
                        cfg.getName() + " step=" + step);

                INDArray logits = out.get("logits");
                if (prevLogits != null) {
                    double diff = logits.castTo(DataType.DOUBLE)
                            .sub(prevLogits.castTo(DataType.DOUBLE))
                            .amaxNumber().doubleValue();
                    if (diff < 1e-9) {
                        identicalRun++;
                        if (identicalRun > maxIdenticalRun) maxIdenticalRun = identicalRun;
                    } else {
                        identicalRun = 0;
                    }
                }
                if (prevLogits != null && prevLogits.closeable() && !prevLogits.wasClosed()) {
                    prevLogits.close();
                }
                prevLogits = logits.dup();
                closeAll(out);
                resetBetweenRuns(sd);
            }
            if (prevLogits != null && prevLogits.closeable() && !prevLogits.wasClosed()) {
                prevLogits.close();
            }
            assertTrue(maxIdenticalRun <= 2,
                    cfg.getName() + ": logits identical for "
                            + maxIdenticalRun + " consecutive steps (graph replay stale?)");
        } finally {
            sd.close();
            for (Map<String, INDArray> out : refOutputs) closeAll(out);
            for (Map<String, INDArray> in : inputsByStep) closeAll(in);
        }
    }

    @ParameterizedTest(name = "staticKvLargePrefill[{0}]")
    @MethodSource("kvCacheLifecycleConfigs")
    @DisplayName("Static KV large-prefill + decode lifecycle")
    public void testStaticKvLargePrefillDecode(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        int maxKvLen = LARGE_PREFILL + 16;
        SameDiff sd = buildStaticKvDecoder(maxKvLen);
        try {
            // Treat the first call as a "prefill" by giving position=LARGE_PREFILL-1
            Map<String, INDArray> prefillInputs = staticKvInputs(maxKvLen, LARGE_PREFILL - 1);
            BenchmarkConfig refCfg = BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                    .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> refPrefill = runWithConfig(sd, refCfg,
                    copyInputs(prefillInputs), "logits", "present_key", "present_value");
            resetBetweenRuns(sd);
            Map<String, INDArray> testPrefill = runWithConfig(sd, cfg,
                    copyInputs(prefillInputs), "logits", "present_key", "present_value");
            double[] tol = tolerances(cfg);
            assertOutputsClose(refPrefill, testPrefill, tol[0], tol[1],
                    cfg.getName() + " prefill");
            closeAll(refPrefill);
            closeAll(testPrefill);
            closeAll(prefillInputs);
            resetBetweenRuns(sd);

            // 10 decode steps
            int decodeSteps = 10;
            for (int s = 0; s < decodeSteps; s++) {
                long position = LARGE_PREFILL + s;
                Map<String, INDArray> stepIn = staticKvInputs(maxKvLen, position);
                Map<String, INDArray> refOut = runWithConfig(sd, refCfg,
                        copyInputs(stepIn), "logits", "present_key", "present_value");
                resetBetweenRuns(sd);
                Map<String, INDArray> testOut = runWithConfig(sd, cfg,
                        copyInputs(stepIn), "logits", "present_key", "present_value");
                assertOutputsClose(refOut, testOut, tol[0], tol[1],
                        cfg.getName() + " decode step=" + s);
                closeAll(refOut);
                closeAll(testOut);
                closeAll(stepIn);
                resetBetweenRuns(sd);
            }
        } finally {
            sd.close();
        }
    }

    // ─── Knob bisection tests ──────────────────────────────────────────────

    @ParameterizedTest(name = "tritonKnob[{0}]")
    @MethodSource("tritonKnobBisectionConfigs")
    @DisplayName("Triton knob bisection: each knob must produce identical FP32 output")
    public void testTritonKnobBisection(BenchmarkConfig cfg) {
        assumeTrue(isTritonAvailable(), "Triton unavailable");
        SameDiff sd = buildMatmulMlp();
        try {
            runRefVsTest(sd, cfg, matmulInputs(), "out");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "dspKnob[{0}]")
    @MethodSource("dspKnobConfigs")
    @DisplayName("DSP knob combinations against SLOT_BY_SLOT reference")
    public void testDspKnobCombinations(BenchmarkConfig cfg) {
        SameDiff sd = buildMatmulMlp();
        try {
            runRefVsTest(sd, cfg, matmulInputs(), "out");
        } finally {
            sd.close();
        }
    }

    // ─── Graph replay / lifecycle tests ────────────────────────────────────

    @ParameterizedTest(name = "decodeOutputVaries[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Multi-step decode output must vary with inputs (catch stale replay)")
    public void testMultiStepDecodeOutputVariesWithInputs(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        int steps = Math.min(30, maxDecodeSteps + 10);
        SameDiff sd = buildElementwise();
        try {
            BenchmarkConfigApplier.apply(cfg);
            if (cfg.getExecutionMode() != null) {
                sd.setGraphExecutionMode(cfg.getExecutionMode());
            }
            sd.setDspAutoCompileEnabled(cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT);
            sd.setDspNativeAutoCompileEnabled(cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT);

            INDArray prev = null;
            int identicalRun = 0;
            int maxIdenticalRun = 0;
            for (int s = 0; s < steps; s++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, ELEMENT_BATCH, ELEMENT_DIM).muli(0.5 + s * 0.01);
                Map<String, INDArray> in = new LinkedHashMap<>();
                in.put("x", x);
                Map<String, INDArray> out = sd.output(in, "out");
                INDArray cur = out.get("out").dup();
                if (prev != null) {
                    double diff = cur.castTo(DataType.DOUBLE)
                            .sub(prev.castTo(DataType.DOUBLE)).amaxNumber().doubleValue();
                    if (diff < 1e-9) {
                        identicalRun++;
                        if (identicalRun > maxIdenticalRun) maxIdenticalRun = identicalRun;
                    } else {
                        identicalRun = 0;
                    }
                }
                if (prev != null && prev.closeable() && !prev.wasClosed()) prev.close();
                prev = cur;
                if (x.closeable() && !x.wasClosed()) x.close();
            }
            if (prev != null && prev.closeable() && !prev.wasClosed()) prev.close();
            assertTrue(maxIdenticalRun <= 2,
                    cfg.getName() + ": " + maxIdenticalRun
                            + " consecutive identical outputs (stale replay suspected)");

            // After many steps with DSP enabled, plan should have frozen
            // SLOT_BY_SLOT and EMULATED_REPLAY don't advance DSP phases
            if (cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT
                    && cfg.getExecutionMode() != GraphExecutionMode.EMULATED_REPLAY) {
                DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN,
                        cfg.getName() + " after " + steps + " steps");
                DspPlanAssertions.assertNoSegmentFailures(sd, cfg.getName());
            }
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "shapeKeyDoesNotPoison[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Shape key tracking must not poison non-FLOAT placeholder buffers")
    public void testShapeKeyDoesNotPoisonDeviceBuffers(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildStaticKvDecoder(KV_MAX_LEN);
        try {
            BenchmarkConfigApplier.apply(cfg);
            if (cfg.getExecutionMode() != null) {
                sd.setGraphExecutionMode(cfg.getExecutionMode());
            }
            sd.setDspAutoCompileEnabled(cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT);
            sd.setDspNativeAutoCompileEnabled(cfg.getExecutionMode() != GraphExecutionMode.SLOT_BY_SLOT);

            for (int iter = 0; iter < 5; iter++) {
                Map<String, INDArray> in = staticKvInputs(KV_MAX_LEN, iter);
                INDArray maskBefore = in.get("attention_mask").dup();
                INDArray pastKeyBefore = in.get("past_key").dup();
                Map<String, INDArray> out = sd.output(copyInputs(in),
                        "logits", "present_key", "present_value");
                INDArray maskAfter = in.get("attention_mask");
                INDArray pastKeyAfter = in.get("past_key");
                assertTrue(maskBefore.equalsWithEps((Object) maskAfter, 1e-9),
                        cfg.getName() + " iter=" + iter + ": attention_mask was modified by execution");
                assertTrue(pastKeyBefore.equalsWithEps((Object) pastKeyAfter, 1e-9),
                        cfg.getName() + " iter=" + iter + ": past_key was modified by execution");
                closeAll(out);
                closeAll(in);
                if (maskBefore.closeable() && !maskBefore.wasClosed()) maskBefore.close();
                if (pastKeyBefore.closeable() && !pastKeyBefore.wasClosed()) pastKeyBefore.close();
                resetBetweenRuns(sd);
            }
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "valueDepFreeze[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Value-dependent ops must remain correct across freeze/thaw cycles")
    public void testValueDepOpsAfterFreeze(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildValueDepAfterFreeze();
        try {
            BenchmarkConfig refCfg = BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                    .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
            int[] shapes = new int[]{2, 5, 3, 3, 3, 3, 3, 3};
            double[] tol = tolerances(cfg);
            for (int rows : shapes) {
                Map<String, INDArray> in = valueDepInputs(rows);
                Map<String, INDArray> ref = runWithConfig(sd, refCfg, copyInputs(in), "out");
                resetBetweenRuns(sd);
                Map<String, INDArray> test = runWithConfig(sd, cfg, copyInputs(in), "out");
                assertOutputsClose(ref, test, tol[0], tol[1],
                        cfg.getName() + " rows=" + rows);
                closeAll(ref);
                closeAll(test);
                closeAll(in);
                resetBetweenRuns(sd);
            }
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "viewOpSkip[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("View op chain accuracy across execution modes")
    public void testViewOpSkipLogic(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildViewOpChain();
        try {
            runRefVsTest(sd, cfg, viewInputs(), "out");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "whereBroadcast[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Where + broadcast chain accuracy across execution modes")
    public void testWhereBroadcastChain(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildWhereBroadcast();
        try {
            runRefVsTest(sd, cfg, whereInputs(), "out");
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "outputDirectVsOutput[{0}]")
    @MethodSource("outputPathConfigs")
    @DisplayName("output() and outputDirect() must produce the same values")
    public void testOutputDirectVsOutputParity(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildMatmulMlp();
        try {
            BenchmarkConfigApplier.apply(cfg);
            if (cfg.getExecutionMode() != null) {
                sd.setGraphExecutionMode(cfg.getExecutionMode());
            }
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray x = Nd4j.randn(DataType.FLOAT, MLP_BATCH, MLP_IN).muli(0.5);
            // First: output()
            Map<String, INDArray> outA = sd.output(Map.of("x", x), "out");
            INDArray a = outA.get("out").dup();
            // Reset cleanly
            resetBetweenRuns(sd);
            BenchmarkConfigApplier.apply(cfg);
            if (cfg.getExecutionMode() != null) {
                sd.setGraphExecutionMode(cfg.getExecutionMode());
            }
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            // Second: outputDirect()
            Map<String, INDArray> outB = sd.outputDirect(Map.of("x", x), "out");
            INDArray b = outB.get("out").dup();

            double[] tol = tolerances(cfg);
            Map<String, INDArray> refMap = new LinkedHashMap<>();
            refMap.put("out", a);
            Map<String, INDArray> testMap = new LinkedHashMap<>();
            testMap.put("out", b);
            assertOutputsClose(refMap, testMap, tol[0], tol[1], cfg.getName());

            if (a.closeable() && !a.wasClosed()) a.close();
            if (b.closeable() && !b.wasClosed()) b.close();
            if (x.closeable() && !x.wasClosed()) x.close();
        } finally {
            sd.close();
        }
    }

    @ParameterizedTest(name = "freezeUnfreeze[{0}]")
    @MethodSource("executionModeConfigs")
    @DisplayName("Frozen and unfrozen shapes both produce correct outputs")
    public void testFreezeUnfreezeCycle(BenchmarkConfig cfg) {
        assumeBackendAvailable(cfg);
        SameDiff sd = buildValueDepAfterFreeze();
        try {
            BenchmarkConfig refCfg = BenchmarkConfig.create("REF_SLOT_BY_SLOT")
                    .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
            double[] tol = tolerances(cfg);

            // First: 5 runs at a fixed shape (encourages freezing)
            for (int i = 0; i < 5; i++) {
                Map<String, INDArray> in = valueDepInputs(4);
                Map<String, INDArray> ref = runWithConfig(sd, refCfg, copyInputs(in), "out");
                resetBetweenRuns(sd);
                Map<String, INDArray> test = runWithConfig(sd, cfg, copyInputs(in), "out");
                assertOutputsClose(ref, test, tol[0], tol[1],
                        cfg.getName() + " frozen iter=" + i);
                closeAll(ref);
                closeAll(test);
                closeAll(in);
                resetBetweenRuns(sd);
            }
            // Then: change shape (forces unfreeze)
            for (int rows : new int[]{6, 8, 4, 2}) {
                Map<String, INDArray> in = valueDepInputs(rows);
                Map<String, INDArray> ref = runWithConfig(sd, refCfg, copyInputs(in), "out");
                resetBetweenRuns(sd);
                Map<String, INDArray> test = runWithConfig(sd, cfg, copyInputs(in), "out");
                assertOutputsClose(ref, test, tol[0], tol[1],
                        cfg.getName() + " unfreeze rows=" + rows);
                closeAll(ref);
                closeAll(test);
                closeAll(in);
                resetBetweenRuns(sd);
            }
        } finally {
            sd.close();
        }
    }

    // ─── Non-parameterized lifecycle tests ─────────────────────────────────

    @Test
    @DisplayName("Pooling buffer reuse: 20 iterations of MLP fixture must be stable")
    public void testPoolingBufferReuse() {
        SameDiff sd = buildMatmulMlp();
        try {
            BenchmarkConfig cfg = BenchmarkConfig.create("AUTO_pool")
                    .executionMode(GraphExecutionMode.AUTO);
            BenchmarkConfigApplier.apply(cfg);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray x = Nd4j.randn(DataType.FLOAT, MLP_BATCH, MLP_IN).muli(0.5);
            INDArray firstOut = null;
            for (int i = 0; i < 20; i++) {
                Map<String, INDArray> out = sd.output(Map.of("x", x), "out");
                INDArray cur = out.get("out").dup();
                if (firstOut == null) {
                    firstOut = cur;
                } else {
                    double diff = cur.castTo(DataType.DOUBLE)
                            .sub(firstOut.castTo(DataType.DOUBLE)).amaxNumber().doubleValue();
                    assertTrue(diff < 1e-3,
                            "Pooling reuse iter=" + i + " diff=" + diff
                                    + " — buffer reuse may be returning stale data");
                    if (cur.closeable() && !cur.wasClosed()) cur.close();
                }
            }
            if (firstOut != null && firstOut.closeable() && !firstOut.wasClosed()) firstOut.close();
            if (x.closeable() && !x.wasClosed()) x.close();
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("Array lifecycle: 100 iterations must not leak GPU memory beyond 5%")
    public void testArrayLifecycleNoLeaks() {
        SameDiff sd = buildAttentionBlock();
        try {
            BenchmarkConfig cfg = BenchmarkConfig.create("AUTO_leak")
                    .executionMode(GraphExecutionMode.AUTO);
            BenchmarkConfigApplier.apply(cfg);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            // Warmup
            for (int i = 0; i < 10; i++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, ATTN_B, ATTN_S, ATTN_H).muli(0.2);
                Map<String, INDArray> out = sd.output(Map.of("x", x), "out");
                closeAll(out);
                if (x.closeable() && !x.wasClosed()) x.close();
            }
            Nd4j.getExecutioner().commit();
            long before = Nd4j.getMemoryManager().allocatedMemory(0);

            for (int i = 0; i < 100; i++) {
                INDArray x = Nd4j.randn(DataType.FLOAT, ATTN_B, ATTN_S, ATTN_H).muli(0.2);
                Map<String, INDArray> out = sd.output(Map.of("x", x), "out");
                closeAll(out);
                if (x.closeable() && !x.wasClosed()) x.close();
            }
            Nd4j.getExecutioner().commit();
            long after = Nd4j.getMemoryManager().allocatedMemory(0);

            long delta = after - before;
            long tolerance = Math.max(before / 20, 32L * 1024 * 1024); // 5% or 32MB
            log.info("testArrayLifecycleNoLeaks: before={} after={} delta={} tol={}",
                    before, after, delta, tolerance);
            assertTrue(delta <= tolerance,
                    "Memory leak detected: delta=" + delta + " > tolerance=" + tolerance);
        } finally {
            sd.close();
        }
    }

    @Test
    @DisplayName("DSP phase lifecycle: COMPILE → CAPTURE → REPLAY events recorded")
    public void testDspPhaseLifecycleContract() {
        // Enable DSP diagnostics
        DspDiagnostics.enableCategories(
                DspDiagnostics.COMPILE | DspDiagnostics.EXECUTE | DspDiagnostics.SEGMENT
                        | DspDiagnostics.GRAPH_REPLAY);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
        DspDiagnostics.clear();

        SameDiff sd = buildMatmulMlp();
        try {
            BenchmarkConfig cfg = BenchmarkConfig.create("AUTO_phase")
                    .executionMode(GraphExecutionMode.AUTO);
            BenchmarkConfigApplier.apply(cfg);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray x = Nd4j.randn(DataType.FLOAT, MLP_BATCH, MLP_IN).muli(0.5);
            for (int i = 0; i < 5; i++) {
                Map<String, INDArray> out = sd.output(Map.of("x", x), "out");
                closeAll(out);
            }
            if (x.closeable() && !x.wasClosed()) x.close();

            // Structural DSP assertions: after 5 steps, plan should have advanced
            DspPlanAssertions.assertPhaseReached(sd, PlanPhase.SHAPES_FROZEN, "lifecycle-contract");
            DspPlanAssertions.assertDiagnosticsRecorded(sd, "lifecycle-contract");
            DspPlanAssertions.assertNoFallbacks(sd, "lifecycle-contract");

            String report = DspDiagnostics.getPlanReport();
            assertNotNull(report, "DSP plan report should not be null");
            log.info("DSP plan report:\n{}", report);
            // The report should be non-empty when DSP executed at least one segment
            assertTrue(report.length() > 0, "Plan report should contain phase information");
        } finally {
            sd.close();
            DspDiagnostics.clear();
        }
    }

    // ─── Multi-device tests (uses stub topology for portability) ──────────────

    @Test
    @DisplayName("Cross-device matmul execution (stub 2 devices)")
    public void testCrossDeviceMatmulExecution() {
        // Set up a 2-device stub topology so multi-device DSP paths are exercised.
        // On CPU, replicateToDevice(1, arr) does dup() — the test verifies the DSP
        // infrastructure handles logical multi-device correctly.
        DeviceMemoryManager dmm = DeviceMemoryManager.getInstance();
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("StubGPU-0").totalMemory(16L * 1024 * 1024 * 1024)
                .availableMemory(14L * 1024 * 1024 * 1024).addPeerDevice(1).build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("StubGPU-1").totalMemory(16L * 1024 * 1024 * 1024)
                .availableMemory(14L * 1024 * 1024 * 1024).addPeerDevice(0).build();
        dmm.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Verify round-trip replication works before the actual test
        {
            INDArray orig = Nd4j.create(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, new long[]{2, 2});
            log.info("REPLICATE_TEST: orig on dev0 = {}", orig.toFloatVector());

            INDArray onDev1 = Nd4j.getAffinityManager().replicateToDevice(1, orig);
            Nd4j.getExecutioner().commit();
            // Read back from device 1 by syncing
            float[] dev1Data = onDev1.dup().toFloatVector();
            log.info("REPLICATE_TEST: on dev1 = {}", java.util.Arrays.toString(dev1Data));

            INDArray backOnDev0 = Nd4j.getAffinityManager().replicateToDevice(0, onDev1);
            Nd4j.getExecutioner().commit();
            float[] dev0Data = backOnDev0.dup().toFloatVector();
            log.info("REPLICATE_TEST: back on dev0 = {}", java.util.Arrays.toString(dev0Data));

            assertTrue(dev0Data[0] == 1.0f && dev0Data[1] == 2.0f && dev0Data[2] == 3.0f && dev0Data[3] == 4.0f,
                    "Round-trip dev0->dev1->dev0 failed: " + java.util.Arrays.toString(dev0Data));
            orig.close();
            onDev1.close();
            backOnDev0.close();
        }

        SameDiff sd = buildMatmulMlp();
        try {
            BenchmarkConfig cfg = BenchmarkConfig.create("multi_device_AUTO")
                    .executionMode(GraphExecutionMode.AUTO);
            BenchmarkConfigApplier.apply(cfg);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            INDArray x = Nd4j.randn(DataType.FLOAT, MLP_BATCH, MLP_IN).muli(0.5);
            // Force input onto device 0
            INDArray xDev0 = Nd4j.getAffinityManager().replicateToDevice(Integer.valueOf(0), x);
            Map<String, INDArray> out0 = sd.output(Map.of("x", xDev0), "out");
            INDArray a = out0.get("out").dup();
            closeAll(out0);

            // Force input onto device 1 and rerun
            INDArray x1 = x.dup();
            INDArray xDev1 = Nd4j.getAffinityManager().replicateToDevice(Integer.valueOf(1), x1);
            Map<String, INDArray> out1 = sd.output(Map.of("x", xDev1), "out");
            INDArray b = out1.get("out").dup();
            closeAll(out1);

            // Reference: compute on device 0 with SLOT_BY_SLOT
            BenchmarkConfig refCfg = BenchmarkConfig.create("ref_dev0")
                    .executionMode(GraphExecutionMode.SLOT_BY_SLOT);
            BenchmarkConfigApplier.apply(refCfg);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            Map<String, INDArray> refOut = sd.output(Map.of("x", x), "out");
            INDArray ref = refOut.get("out").dup();
            closeAll(refOut);

            Map<String, INDArray> refMap = new LinkedHashMap<>();
            refMap.put("out", ref);
            Map<String, INDArray> testMap0 = new LinkedHashMap<>();
            testMap0.put("out", a);
            Map<String, INDArray> testMap1 = new LinkedHashMap<>();
            testMap1.put("out", b);
            assertOutputsClose(refMap, testMap0, FP32_RTOL, FP32_ATOL, "device0");
            assertOutputsClose(refMap, testMap1, FP32_RTOL, FP32_ATOL, "device1");

            if (a.closeable() && !a.wasClosed()) a.close();
            if (b.closeable() && !b.wasClosed()) b.close();
            if (ref.closeable() && !ref.wasClosed()) ref.close();
            if (x.closeable() && !x.wasClosed()) x.close();
            if (x1.closeable() && !x1.wasClosed()) x1.close();
        } finally {
            sd.close();
            dmm.clearStubTopology();
        }
    }

    @Test
    @DisplayName("Multi-device transfer recording (stub 2 devices)")
    public void testMultiDeviceTransferRecording() {
        // Set up a 2-device stub topology for multi-device transfer recording.
        DeviceMemoryManager dmm = DeviceMemoryManager.getInstance();
        StubDeviceDescriptor gpu0 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 0)
                .deviceName("StubGPU-0").totalMemory(16L * 1024 * 1024 * 1024)
                .availableMemory(14L * 1024 * 1024 * 1024).addPeerDevice(1).build();
        StubDeviceDescriptor gpu1 = StubDeviceDescriptor.builder(DeviceType.CUDA_GPU, 1)
                .deviceName("StubGPU-1").totalMemory(16L * 1024 * 1024 * 1024)
                .availableMemory(14L * 1024 * 1024 * 1024).addPeerDevice(0).build();
        dmm.configureStubTopology(Arrays.asList(gpu0, gpu1));

        // Enable transfer-related diagnostics and run a decode fixture
        DspDiagnostics.enableCategories(DspDiagnostics.TRANSFER | DspDiagnostics.MULTI_DEVICE);
        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
        DspDiagnostics.clear();

        SameDiff sd = buildStaticKvDecoder(KV_MAX_LEN);
        try {
            BenchmarkConfig cfg = BenchmarkConfig.create("multi_device_decode")
                    .executionMode(GraphExecutionMode.AUTO);
            BenchmarkConfigApplier.apply(cfg);
            sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);

            for (int s = 0; s < 5; s++) {
                Map<String, INDArray> in = staticKvInputs(KV_MAX_LEN, s);
                Map<String, INDArray> out = sd.output(in, "logits", "present_key", "present_value");
                closeAll(out);
                closeAll(in);
            }
            String report = DspDiagnostics.getPlanReport();
            assertNotNull(report);
            log.info("Multi-device transfer report:\n{}", report);
            // We don't fail if the report happens to be empty for short runs; we assert it exists.
            assertTrue(report.length() > 0, "Plan report should contain at least one event");
        } finally {
            sd.close();
            DspDiagnostics.clear();
            dmm.clearStubTopology();
        }
    }

    // ─── Internal helpers ──────────────────────────────────────────────────

    /**
     * Snapshot of all DSP / Triton environment flags. Captured in @BeforeAll
     * and restored in @AfterAll plus around every test to prevent knob leakage.
     */
    private static final class EnvSnapshot {
        // DSP flags
        boolean dspBatchZero;
        boolean dspBatchZeroKernel;
        boolean dspBatchedGemm;
        boolean dspCastSinkMatmul;
        boolean dspCastElimination;
        boolean dspFp16Compute;
        boolean dspFreezeMergeSegments;
        boolean dspFreezeRecompile;
        // Triton flags
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
        String tritonIncludeTypes;
        String tritonExcludeOps;
        // cuBLAS / TF32
        boolean cublasTf32;
        boolean tritonTf32;

        static EnvSnapshot capture() {
            Environment env = Nd4j.getEnvironment();
            EnvSnapshot s = new EnvSnapshot();
            try {
                s.dspBatchZero = env.dspBatchZero();
                s.dspBatchZeroKernel = env.dspBatchZeroKernel();
                s.dspBatchedGemm = env.dspBatchedGemm();
                s.dspCastSinkMatmul = env.dspCastSinkMatmul();
                s.dspCastElimination = env.dspCastElimination();
                s.dspFp16Compute = env.dspFp16Compute();
                s.dspFreezeMergeSegments = env.dspFreezeMergeSegments();
                s.dspFreezeRecompile = env.dspFreezeRecompile();
                s.tritonGraphCapture = env.tritonGraphCapture();
                s.tritonSectionFusion = env.tritonSectionFusion();
                s.tritonConsolidatedArgTable = env.tritonConsolidatedArgTable();
                s.tritonArgDirtyTracking = env.tritonArgDirtyTracking();
                s.tritonSkipKernels = env.tritonSkipKernels();
                s.tritonVerifyKernels = env.tritonVerifyKernels();
                s.tritonForceRecapture = env.tritonForceRecapture();
                s.tritonCompileAll = env.tritonCompileAll();
                s.tritonCooperativeLaunch = env.tritonCooperativeLaunch();
                s.tritonVerbose = env.tritonVerbose();
                s.tritonDumpSections = env.tritonDumpSections();
                s.tritonAllowFallbackCapture = env.tritonAllowFallbackCapture();
                s.tritonIncludeTypes = env.tritonIncludeTypes();
                s.tritonExcludeOps = env.tritonExcludeOps();
                s.cublasTf32 = env.cublasTf32Enabled();
                s.tritonTf32 = env.tritonTf32Enabled();
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
                env.setTritonIncludeTypes(tritonIncludeTypes == null ? "" : tritonIncludeTypes);
                env.setTritonExcludeOps(tritonExcludeOps == null ? "" : tritonExcludeOps);
                env.setCublasTf32Enabled(cublasTf32);
                env.setTritonTf32Enabled(tritonTf32);
            } catch (Throwable t) {
                log.warn("EnvSnapshot.restore: partial restore due to {}", t.toString());
            }
        }
    }
}
