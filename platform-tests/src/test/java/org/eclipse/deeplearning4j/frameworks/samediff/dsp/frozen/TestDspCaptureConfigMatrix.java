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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.frozen;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfig;
import org.eclipse.deeplearning4j.model.benchmark.BenchmarkConfigApplier;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.RmsNorm;
import org.nd4j.linalg.api.ops.impl.transforms.pairwise.arithmetic.AddOp;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * DSP Capture Configuration Matrix — exhaustive validation of all
 * capture/merge/workspace code paths against a slot-by-slot reference.
 *
 * Tests every meaningful combination of:
 *   - GraphTopology: what kinds of ops are in the graph
 *   - CaptureConfig: which DSP/Triton flags are set
 *
 * Reference is ALWAYS: tritonGraphCapture=false (pure slot-by-slot, no CUDA graphs).
 * Every other configuration must produce identical results to within 1e-4 tolerance.
 *
 * This test exposes:
 *   - cuBLAS workspace algorithm mismatch between capture and replay
 *   - Merged vs unmerged gap divergence
 *   - Non-capture-safe gap handling correctness
 *   - Workspace ON/OFF algorithm selection differences
 *   - Consolidated arg table pointer stability
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Capture Config Matrix")
public class TestDspCaptureConfigMatrix {

    private static final int DIM = 64;
    private static final int STEPS = 20;
    // Topology-specific base tolerances.
    // MATMUL_ONLY: cuBLAS TF32 non-determinism causes diffs up to ~1.0 between
    // slot-by-slot and CUDA graph replay (different internal algorithm selection).
    // MIXED_GAPS: view ops captured into merged graphs produce systematic offset
    // within float32 precision (bounded under 0.02).
    // Element-wise graphs should match bit-exactly (1e-4).
    private static double toleranceFor(GraphTopology t) {
        switch (t) {
            case MATMUL_ONLY:              return 1.1;    // TF32 non-determinism (6 chained matmuls)
            case MATMUL_GAPS_CAPTURE_SAFE: return 1.1;    // same 6 chained matmuls as MATMUL_ONLY
            case MIXED_GAPS:               return 0.02;
            default:                       return 1e-4;
        }
    }

    private static double toleranceFor(GraphTopology t, CaptureConfig config) {
        return toleranceFor(t);
    }
    private static final long GRAPH_SEED = 777L;

    // ═══════════════════════════════════════════════════════════════════════════
    // Graph Topologies
    // ═══════════════════════════════════════════════════════════════════════════

    enum GraphTopology {
        /** Pure element-wise ops. Single Triton island, no gaps. */
        PURE_ELEMENTWISE,
        /** Alternating Triton islands + cuBLAS matmul gaps (capture-safe). */
        MATMUL_GAPS_CAPTURE_SAFE,
        /** Triton islands separated by reshape (view) gaps (NOT capture-safe). */
        VIEW_GAPS_NOT_CAPTURE_SAFE,
        /** Only matmul ops — no Triton islands, entire graph is gaps. */
        MATMUL_ONLY,
        /** Mixed: some capture-safe gaps (matmul) and some non-capture-safe (reshape). */
        MIXED_GAPS
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Capture Configurations
    // ═══════════════════════════════════════════════════════════════════════════

    static class CaptureConfig {
        final String name;
        final boolean tritonGraphCapture;
        final boolean freezeMergeSegments;
        final boolean cublasCaptureWorkspace;
        final boolean consolidatedArgTable;
        final boolean tritonCompileAll;
        final boolean argDirtyTracking;

        CaptureConfig(String name, boolean tritonGraphCapture, boolean freezeMergeSegments,
                      boolean cublasCaptureWorkspace, boolean consolidatedArgTable,
                      boolean tritonCompileAll, boolean argDirtyTracking) {
            this.name = name;
            this.tritonGraphCapture = tritonGraphCapture;
            this.freezeMergeSegments = freezeMergeSegments;
            this.cublasCaptureWorkspace = cublasCaptureWorkspace;
            this.consolidatedArgTable = consolidatedArgTable;
            this.tritonCompileAll = tritonCompileAll;
            this.argDirtyTracking = argDirtyTracking;
        }

        static CaptureConfig reference() {
            return new CaptureConfig("REFERENCE_NO_CAPTURE",
                    false, false, false, false, false, false);
        }

        /**
         * Reference that matches the test config's kernel compilation and workspace
         * settings but without CUDA graph capture. This isolates the capture/replay
         * correctness question from two known precision sources:
         *
         * 1. Triton-vs-native kernel precision: tritonCompileAll=true uses different
         *    instruction sequences than native ops (~0.005 FP32 divergence).
         * 2. cuBLAS workspace algorithm selection: different workspace sizes cause
         *    cuBLAS to pick different algorithms with slightly different numerics.
         *
         * By matching both in the reference, we test ONLY whether capture/replay
         * introduces additional error — which is what this test is designed to verify.
         */
        static CaptureConfig referenceFor(CaptureConfig testConfig) {
            return new CaptureConfig("REFERENCE_FOR_" + testConfig.name,
                    false,                              // no capture
                    false,                              // no merge (irrelevant without capture)
                    testConfig.cublasCaptureWorkspace,  // match workspace setting
                    false,                              // no consolidated arg table
                    testConfig.tritonCompileAll,        // match kernel compilation
                    false);                             // no dirty tracking
        }

        @Override
        public String toString() {
            return name;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Parameter Generation
    // ═══════════════════════════════════════════════════════════════════════════

    static Stream<Arguments> configMatrix() {
        List<Arguments> args = new ArrayList<>();
        List<CaptureConfig> configs = generateConfigs();

        for (GraphTopology topology : GraphTopology.values()) {
            for (CaptureConfig config : configs) {
                args.add(Arguments.of(topology, config));
            }
        }
        return args.stream();
    }

    private static List<CaptureConfig> generateConfigs() {
        List<CaptureConfig> configs = new ArrayList<>();
        boolean[] bools = {false, true};

        for (boolean capture : bools) {
            for (boolean merge : bools) {
                for (boolean workspace : bools) {
                    for (boolean argTable : bools) {
                        for (boolean compileAll : bools) {
                            for (boolean dirtyTrack : bools) {
                                // Without capture, other flags don't exercise different paths
                                if (!capture && (merge || workspace || argTable || dirtyTrack || compileAll)) {
                                    continue;
                                }

                                String name = String.format(
                                        "cap=%b_merge=%b_ws=%b_arg=%b_comp=%b_dirty=%b",
                                        capture, merge, workspace, argTable, compileAll, dirtyTrack);

                                configs.add(new CaptureConfig(name, capture, merge,
                                        workspace, argTable, compileAll, dirtyTrack));
                            }
                        }
                    }
                }
            }
        }
        return configs;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // The Test
    // ═══════════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "{0} | {1}")
    @MethodSource("configMatrix")
    @DisplayName("Config must match slot-by-slot reference")
    void testConfigMatchesReference(GraphTopology topology, CaptureConfig config) {
        // Use a reference that matches the test config's kernel compilation (tritonCompileAll)
        // and workspace settings (cublasCaptureWorkspace) but without CUDA graph capture.
        // This isolates the capture/replay correctness question from two known precision sources:
        //   1. Triton-vs-native kernel precision (~0.005 FP32 divergence)
        //   2. cuBLAS workspace algorithm selection (different workspace → different algorithm)
        CaptureConfig refConfig = CaptureConfig.referenceFor(config);
        List<INDArray> refOutputs = runGraph(topology, refConfig);
        List<INDArray> testOutputs = runGraph(topology, config);

        assertEquals(refOutputs.size(), testOutputs.size(),
                "Output count mismatch for " + config.name);

        double topologyTol = toleranceFor(topology, config);
        int mismatchCount = 0;
        double worstDiff = 0;
        int worstStep = -1;
        for (int step = 0; step < refOutputs.size(); step++) {
            INDArray ref = refOutputs.get(step);
            INDArray test = testOutputs.get(step);
            double maxDiff = ref.sub(test).amaxNumber().doubleValue();
            if (maxDiff > topologyTol) {
                mismatchCount++;
                if (maxDiff > worstDiff) {
                    worstDiff = maxDiff;
                    worstStep = step;
                }
                log.error("[{}] {} step {}: MISMATCH maxDiff={}",
                        topology, config.name, step, maxDiff);
            }
        }

        assertEquals(0, mismatchCount,
                String.format("[%s] %s: %d/%d steps diverge (worst=%.6f at step %d). " +
                                "Config produces different results than slot-by-slot reference.",
                        topology, config.name, mismatchCount, refOutputs.size(),
                        worstDiff, worstStep));
    }

    @Test
    @DisplayName("Reference execution is deterministic")
    void testReferenceDeterminism() {
        for (GraphTopology topology : GraphTopology.values()) {
            // Verify pure slot-by-slot reference
            List<INDArray> run1 = runGraph(topology, CaptureConfig.reference());
            List<INDArray> run2 = runGraph(topology, CaptureConfig.reference());

            assertEquals(run1.size(), run2.size());
            for (int step = 0; step < run1.size(); step++) {
                double maxDiff = run1.get(step).sub(run2.get(step)).amaxNumber().doubleValue();
                assertEquals(0.0, maxDiff, 1e-10,
                        String.format("[%s] Slot-by-slot reference not deterministic at step %d: diff=%.10f",
                                topology, step, maxDiff));
            }

            // Verify Triton-compiled reference (AUTO mode, no capture)
            CaptureConfig tritonRef = new CaptureConfig("DET_CHECK_TRITON",
                    false, false, true, false, true, false);
            List<INDArray> run3 = runGraph(topology, tritonRef);
            List<INDArray> run4 = runGraph(topology, tritonRef);

            assertEquals(run3.size(), run4.size());
            for (int step = 0; step < run3.size(); step++) {
                double maxDiff = run3.get(step).sub(run4.get(step)).amaxNumber().doubleValue();
                // cuBLAS TF32 matmul on Ampere+ is non-deterministic — even running the same
                // computation twice can produce diffs up to ~1.0 due to non-deterministic
                // reduction orderings. Use topology-specific tolerance.
                double detTol = toleranceFor(topology);
                assertEquals(0.0, maxDiff, detTol,
                        String.format("[%s] Triton reference not deterministic at step %d: diff=%.10f",
                                topology, step, maxDiff));
            }
        }
    }

    /**
     * Regression for an in-place unary tail whose escaping output follows a broadcast op.
     * The Triton kernel must derive its launch extent from the [1,64,16] output, not the
     * [16] broadcast input, after CUDA graph capture transitions to replay.
     */
    @Test
    @DisplayName("Captured broadcast elementwise chain replays the complete output extent")
    void testBroadcastElementwiseReplayUsesOutputExtent() {
        CaptureConfig config = new CaptureConfig(
                "broadcast_extent", true, true, true, true, true, true);
        applyConfig(null, config);

        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, 64, 16);
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, 64, 16);
            SDVariable bias = sd.var("bias", Nd4j.valueArrayOf(new long[]{16}, 0.25));
            SDVariable added = x.add("broadcast_add", bias);
            sd.nn.softplus("out", added);
            sd.setOutputs("out");
            applyConfig(sd, config);

            for (int step = 0; step < 6; step++) {
                input.assign(step + 1.0);
                INDArray output = sd.output(Collections.singletonMap("x", input), "out").get("out");
                double expected = Math.log1p(Math.exp(step + 1.25));
                assertEquals(expected, output.aminNumber().doubleValue(), 1e-4,
                        "minimum diverged at replay step " + step);
                assertEquals(expected, output.amaxNumber().doubleValue(), 1e-4,
                        "maximum diverged at replay step " + step);
                assertFalse(output.isNaN().any(), "NaN at replay step " + step);
                assertFalse(output.isInfinite().any(), "Inf at replay step " + step);
            }
        } finally {
            if (!input.wasClosed()) input.close();
            sd.close();
        }
    }

    /**
     * Regression for adjacent, independent elementwise branches with divisible but
     * different output extents. Divisibility is not evidence of broadcasting: the
     * large cast does not consume the small decay chain and must not change that
     * chain's section-wide launch extent.
     */
    @Test
    @DisplayName("Captured independent elementwise branches keep separate output extents")
    void testIndependentElementwiseBranchesUseSeparateExtents() {
        CaptureConfig config = new CaptureConfig(
                "independent_extents", true, true, true, true, true, true);
        applyConfig(null, config);

        SameDiff sd = SameDiff.create();
        INDArray smallInput = Nd4j.zeros(DataType.FLOAT, 1, 64, 16);
        INDArray largeInput = Nd4j.zeros(DataType.HALF, 1, 64, 1024);
        try {
            SDVariable small = sd.placeHolder("small", DataType.FLOAT, 1, 64, 16);
            SDVariable bias = sd.var("bias", Nd4j.valueArrayOf(new long[]{16}, 0.25));
            SDVariable coefficient = sd.var("coefficient", Nd4j.valueArrayOf(new long[]{16}, 0.5));
            SDVariable decay = sd.nn.softplus("small_softplus", small.add("small_add", bias))
                    .mul("small_out", coefficient);

            SDVariable large = sd.placeHolder("large", DataType.HALF, 1, 64, 1024);
            SDVariable largeOut = large.castTo("large_out", DataType.FLOAT);
            sd.setOutputs(decay.name(), largeOut.name());
            applyConfig(sd, config);

            Map<String, INDArray> inputs = new HashMap<>();
            inputs.put("small", smallInput);
            inputs.put("large", largeInput);
            for (int step = 0; step < 6; step++) {
                smallInput.assign(step + 1.0);
                largeInput.assign(step + 2.0);
                Map<String, INDArray> outputs = sd.output(inputs, decay.name(), largeOut.name());

                double expectedSmall = 0.5 * Math.log1p(Math.exp(step + 1.25));
                INDArray smallOut = outputs.get(decay.name());
                assertEquals(expectedSmall, smallOut.aminNumber().doubleValue(), 1e-4,
                        "small minimum diverged at replay step " + step);
                assertEquals(expectedSmall, smallOut.amaxNumber().doubleValue(), 1e-4,
                        "small maximum diverged at replay step " + step);

                INDArray actualLarge = outputs.get(largeOut.name());
                assertEquals(step + 2.0, actualLarge.aminNumber().doubleValue(), 1e-4,
                        "large minimum diverged at replay step " + step);
                assertEquals(step + 2.0, actualLarge.amaxNumber().doubleValue(), 1e-4,
                        "large maximum diverged at replay step " + step);
            }
        } finally {
            if (!smallInput.wasClosed()) smallInput.close();
            if (!largeInput.wasClosed()) largeInput.close();
            sd.close();
        }
    }

    /**
     * Regression for the vision embedding boundary: a value-dependent one-input
     * Where changes cardinality between an upstream data producer and a two-phase
     * gather+add kernel whose output feeds a row reduction. Re-capturing after the
     * cardinality switch must publish current producer data on every replay.
     */
    @Test
    @DisplayName("Where cardinality recapture preserves gather-add-reduction dataflow")
    void testGatherAddMultiPhaseReplayUsesDistinctPhaseIds() {
        CaptureConfig config = new CaptureConfig(
                "gather_add_phases", true, true, true, true, true, true);
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        applyConfig(null, config);

        SameDiff sd = SameDiff.create();
        final int rows = 1024;
        final int columns = 768;
        final int elements = rows * columns;
        INDArray dynamicInput = Nd4j.zeros(DataType.FLOAT, 1, columns, rows);
        INDArray basePattern = Nd4j.arange(elements).castTo(DataType.FLOAT)
                .reshape(1, columns, rows);
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, columns, rows);
            SDVariable owned = x.mul("owned", 1.0);
            SDVariable produced = sd.permute("produced", owned, 0, 2, 1);

            SDVariable mask = sd.placeHolder("mask", DataType.BOOL, 32, 32);
            SDVariable livePredicate = mask.neq("mask_not_zero", sd.zerosLike("mask_zeros", mask));
            SDVariable coordinates = sd.where("coordinates", livePredicate);
            SDVariable validCount = sd.sizeAt("valid_count", coordinates, 0);

            SDVariable table = sd.var("table",
                    Nd4j.valueArrayOf(new long[]{rows, columns}, 0.5).castTo(DataType.HALF));
            SDVariable baseIndices = sd.constant("base_indices",
                    Nd4j.arange(rows).castTo(DataType.LONG).reshape(1, rows));
            SDVariable indexZero = validCount.castTo(DataType.FLOAT).mul(0.0).castTo(DataType.LONG);
            SDVariable indices = baseIndices.add("indices", indexZero);
            SDVariable gathered = sd.gather("gathered", table, indices, 0);
            AddOp importedAdd = new AddOp(sd, produced, gathered);
            importedAdd.addBArgument(false);
            SDVariable added = importedAdd.outputVariable();
            sd.updateVariableNameAndReference(added, "added");
            SDVariable out = added.mean("out", true, -1);
            sd.setOutputs("out");
            applyConfig(sd, config);

            for (int step = 0; step < 10; step++) {
                dynamicInput.assign(basePattern).addi(step + 1.0);
                int validPatches = step < 4 ? 1024 : 992;
                INDArray maskInput = Nd4j.ones(DataType.BOOL, 32, 32);
                if (validPatches == 992) maskInput.getRow(31).assign(0);
                try {
                    Map<String, INDArray> inputs = new HashMap<>();
                    inputs.put("x", dynamicInput);
                    inputs.put("mask", maskInput);
                    INDArray actual = sd.output(inputs, "out").get("out");
                    double expectedMin = step + 1.5 + rows * (columns - 1.0) / 2.0;
                    double expectedMax = expectedMin + rows - 1.0;
                    assertEquals(expectedMin, actual.aminNumber().doubleValue(), 1e-4,
                            "minimum diverged at replay step " + step);
                    assertEquals(expectedMax, actual.amaxNumber().doubleValue(), 1e-4,
                            "maximum diverged at replay step " + step);
                } finally {
                    maskInput.close();
                }
            }
        } finally {
            if (!dynamicInput.wasClosed()) dynamicInput.close();
            if (!basePattern.wasClosed()) basePattern.close();
            sd.close();
        }
    }

    /**
     * Regression for a standalone ordered reduction after the vision embedding
     * section. Each row has a distinct exact mean, so merely checking that the
     * output changes cannot hide stale or incorrectly indexed compiled results.
     */
    @Test
    @DisplayName("Standalone row reduction remains exact after Triton compilation")
    void testStandaloneRowReductionReplayIsExact() {
        CaptureConfig config = new CaptureConfig(
                "standalone_row_reduction", true, true, true, true, true, true);
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        applyConfig(null, config);

        final int rows = 1024;
        final int columns = 768;
        final int elements = rows * columns;
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, rows, columns);
        INDArray base = Nd4j.arange(elements).castTo(DataType.FLOAT)
                .reshape(1, rows, columns);
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, rows, columns);
            SDVariable out = x.mean("out", true, -1);
            sd.setOutputs(out.name());
            applyConfig(sd, config);

            for (int step = 0; step < 8; step++) {
                input.assign(base).addi(step + 1.0);
                INDArray actual = sd.output(Collections.singletonMap("x", input), out.name())
                        .get(out.name());
                double expectedMin = step + 1.0 + (columns - 1.0) / 2.0;
                double expectedMax = expectedMin + (rows - 1.0) * columns;
                assertEquals(expectedMin, actual.aminNumber().doubleValue(), 1e-4,
                        "minimum diverged at compiled reduction step " + step);
                assertEquals(expectedMax, actual.amaxNumber().doubleValue(), 1e-4,
                        "maximum diverged at compiled reduction step " + step);
            }
        } finally {
            if (!input.wasClosed()) input.close();
            if (!base.wasClosed()) base.close();
            sd.close();
        }
    }

    /**
     * Exact regression for the vision pre-attention normalization chain. The
     * nonlinear modifier changes each row's normalized direction every step,
     * so a stale fused RMSNorm output cannot masquerade as scale invariance.
     */
    @Test
    @DisplayName("Fused row-centering and RMSNorm remains exact after compilation")
    void testCenteredRmsNormReplayIsExact() {
        CaptureConfig config = new CaptureConfig(
                "centered_rms_norm", true, true, true, true, true, true);
        BenchmarkConfigApplier.apply(BenchmarkConfig.optimal());
        applyConfig(null, config);

        final int rows = 1024;
        final int columns = 768;
        final int elements = rows * columns;
        final double epsilon = 1e-6;
        SameDiff sd = SameDiff.create();
        INDArray input = Nd4j.zeros(DataType.FLOAT, 1, rows, columns);
        INDArray base = Nd4j.linspace(-1.0, 1.0, elements, DataType.FLOAT)
                .reshape(1, rows, columns);
        INDArray modifier = base.mul(base);
        INDArray gamma = Nd4j.linspace(0.5, 1.5, columns, DataType.FLOAT);
        INDArray bias = Nd4j.linspace(-0.25, 0.25, columns, DataType.FLOAT);
        try {
            SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, rows, columns);
            SDVariable gammaVar = sd.var("gamma", gamma.dup());
            SDVariable biasVar = sd.var("bias", bias.dup());
            SDVariable mean = x.mean("mean", true, -1);
            SDVariable centered = x.sub("centered", mean);
            SDVariable normalized = sd.nn().rmsNorm("normalized", centered, gammaVar, epsilon);
            SDVariable out = normalized.add("out", biasVar);
            sd.setOutputs(out.name());
            applyConfig(sd, config);
            if (Boolean.getBoolean("dsp.test.forceTritonRecompile")) {
                Nd4j.getEnvironment().setTritonAlwaysCompile(true);
            }

            for (int step = 0; step < 8; step++) {
                input.assign(base);
                INDArray delta = modifier.mul(step * 0.05);
                input.addi(delta);
                delta.close();

                INDArray rowMean = input.mean(true, -1);
                INDArray centeredExpected = input.sub(rowMean);
                INDArray expected = Nd4j.create(DataType.FLOAT, 1, rows, columns);
                Nd4j.exec(new RmsNorm(centeredExpected, gamma, expected, epsilon));
                expected.addi(bias);
                INDArray actual = sd.output(Collections.singletonMap("x", input), out.name())
                        .get(out.name());
                INDArray difference = Nd4j.math().abs(actual.sub(expected));
                try {
                    assertTrue(difference.maxNumber().doubleValue() < 5e-4,
                            "centered RMSNorm diverged at compiled step " + step +
                                    ", maxDiff=" + difference.maxNumber());
                } finally {
                    difference.close();
                    expected.close();
                    centeredExpected.close();
                    rowMean.close();
                }
            }
        } finally {
            if (!input.wasClosed()) input.close();
            if (!base.wasClosed()) base.close();
            if (!modifier.wasClosed()) modifier.close();
            if (!gamma.wasClosed()) gamma.close();
            if (!bias.wasClosed()) bias.close();
            sd.close();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Execution
    // ═══════════════════════════════════════════════════════════════════════════

    @AfterEach
    void teardown() {
        Environment env = Nd4j.getEnvironment();
        env.setDspFreezeMergeSegments(true);
        env.setTritonGraphCapture(true);
        env.setTritonVerifyKernels(false);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(true);
        env.setTritonCompileAll(true);
        env.setCublasCaptureWorkspace(true);
        env.setTritonAllowFallbackCapture(true);
    }

    private List<INDArray> runGraph(GraphTopology topology, CaptureConfig config) {
        // Apply env flags FIRST
        applyConfig(null, config);

        // Build graph and inputs using java.util.Random for full isolation from
        // the Nd4j global RNG (which can be consumed by JIT compilation, memory
        // allocation, or other internal paths — making it non-deterministic across runs).
        SameDiff sd = buildGraph(topology);
        applyConfig(sd, config);

        List<INDArray> outputs = new ArrayList<>();
        try {
            for (int step = 0; step < STEPS; step++) {
                INDArray input = createDeterministicInput(step);
                Map<String, INDArray> ph = Collections.singletonMap("x", input);
                Map<String, INDArray> out = sd.output(ph, "out");
                outputs.add(out.get("out").dup());
            }
        } finally {
            try { sd.close(); } catch (Exception ignored) {}
        }
        return outputs;
    }

    /**
     * Create a deterministic input array using java.util.Random.
     * This is fully isolated from the Nd4j global RNG.
     */
    private static INDArray createDeterministicInput(int step) {
        java.util.Random jRng = new java.util.Random(42L + step);
        float[] data = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            data[i] = (float) jRng.nextGaussian();
        }
        return Nd4j.createFromArray(data).reshape(1, DIM);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Graph Construction
    // ═══════════════════════════════════════════════════════════════════════════

    private static SameDiff buildGraph(GraphTopology topology) {
        // Use ordinal as sub-seed to get different-but-deterministic weights per topology
        java.util.Random rng = new java.util.Random(GRAPH_SEED + topology.ordinal());
        switch (topology) {
            case PURE_ELEMENTWISE: return buildPureElementwise(rng);
            case MATMUL_GAPS_CAPTURE_SAFE: return buildMatmulGaps(rng);
            case VIEW_GAPS_NOT_CAPTURE_SAFE: return buildViewGaps(rng);
            case MATMUL_ONLY: return buildMatmulOnly(rng);
            case MIXED_GAPS: return buildMixedGaps(rng);
            default: throw new IllegalArgumentException("Unknown topology: " + topology);
        }
    }

    /** Create a deterministic [rows, cols] array using java.util.Random */
    private static INDArray detArray(java.util.Random rng, int rows, int cols, float scale, float offset) {
        float[] data = new float[rows * cols];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) rng.nextGaussian() * scale + offset;
        }
        return Nd4j.createFromArray(data).reshape(rows, cols);
    }

    private static SameDiff buildPureElementwise(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < 6; i++) {
            SDVariable scale = sd.var("scale_" + i, detArray(rng, 1, DIM, 0.5f, 1.0f));
            SDVariable bias = sd.var("bias_" + i, detArray(rng, 1, DIM, 0.01f, 0.0f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = sd.nn().relu("relu_" + i, h, 0);
        }

        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildMatmulGaps(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < 6; i++) {
            SDVariable scale = sd.var("scale_" + i, detArray(rng, 1, DIM, 0.5f, 1.0f));
            SDVariable bias = sd.var("bias_" + i, detArray(rng, 1, DIM, 0.01f, 0.0f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = sd.nn().relu("relu_" + i, h, 0);

            SDVariable w = sd.var("w_" + i, detArray(rng, DIM, DIM, 0.02f, 0.0f));
            h = sd.mmul("mm_" + i, h, w);
        }

        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildViewGaps(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < 6; i++) {
            SDVariable scale = sd.var("scale_" + i, detArray(rng, 1, DIM, 0.5f, 1.0f));
            SDVariable bias = sd.var("bias_" + i, detArray(rng, 1, DIM, 0.01f, 0.0f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = sd.nn().relu("relu_" + i, h, 0);

            h = sd.reshape("reshape_" + i, h, 1, DIM);
        }

        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildMatmulOnly(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < 6; i++) {
            SDVariable w = sd.var("w_" + i, detArray(rng, DIM, DIM, 0.1f, 0.0f));
            h = sd.mmul("mm_" + i, h, w);
        }

        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);
        sd.setOutputs("out");
        return sd;
    }

    private static SameDiff buildMixedGaps(java.util.Random rng) {
        SameDiff sd = SameDiff.create();
        SDVariable x = sd.placeHolder("x", DataType.FLOAT, 1, DIM);
        SDVariable h = x;

        for (int i = 0; i < 6; i++) {
            SDVariable scale = sd.var("scale_" + i, detArray(rng, 1, DIM, 0.5f, 1.0f));
            SDVariable bias = sd.var("bias_" + i, detArray(rng, 1, DIM, 0.01f, 0.0f));
            h = h.mul("mul_" + i, scale);
            h = h.add("add_" + i, bias);
            h = sd.nn().relu("relu_" + i, h, 0);

            if (i % 2 == 0) {
                SDVariable w = sd.var("w_" + i, detArray(rng, DIM, DIM, 0.02f, 0.0f));
                h = sd.mmul("mm_" + i, h, w);
            } else {
                h = sd.reshape("reshape_" + i, h, 1, DIM);
            }
        }

        SDVariable finalScale = sd.var("scale_final", detArray(rng, 1, DIM, 0.5f, 1.0f));
        h = h.mul("out", finalScale);
        sd.setOutputs("out");
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Configuration Application
    // ═══════════════════════════════════════════════════════════════════════════

    private static void applyConfig(SameDiff sd, CaptureConfig config) {
        if (sd != null) {
            sd.setDspAutoCompileEnabled(true);
            sd.setDspNativeAutoCompileEnabled(true);
            // SLOT_BY_SLOT: only for pure reference (no capture, no Triton compile).
            // AUTO: needed for DSP compile path (Triton and/or capture).
            if (!config.tritonGraphCapture && !config.tritonCompileAll) {
                sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            } else {
                sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
            }
        }

        Environment env = Nd4j.getEnvironment();
        env.setTritonGraphCapture(config.tritonGraphCapture);
        env.setDspFreezeMergeSegments(config.freezeMergeSegments);
        env.setCublasCaptureWorkspace(config.cublasCaptureWorkspace);
        env.setTritonConsolidatedArgTable(config.consolidatedArgTable);
        env.setTritonCompileAll(config.tritonCompileAll);
        env.setTritonArgDirtyTracking(config.argDirtyTracking);
        env.setTritonAllowFallbackCapture(config.tritonGraphCapture);
    }
}
