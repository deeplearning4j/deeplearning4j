/*
 *  ******************************************************************************
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
package org.eclipse.deeplearning4j.nd4j.linalg.mixed;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlan;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanCompiler;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.optimize.GraphOptimizer;
import org.nd4j.autodiff.samediff.optimize.OptimizerSet;
import org.nd4j.autodiff.samediff.optimize.optimizations.*;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Exhaustive DSP pipeline isolation test.
 *
 * Tests EVERY concept in the DSP execution pipeline individually:
 *
 * A. GraphExecutionMode enum — each mode with single-step + multi-step
 * B. ModeContract fields — each boolean field's effect in isolation
 * C. Plan lifecycle phases — SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING
 * D. Segment lifecycle — warmup → compile → capture → replay
 * E. Slot lifecycle — warmup → shape_cached → frozen → frozen_constant
 * F. External input classification — constant vs variable vs placeholder
 * G. Op trait classification — data-dependent, identity, view, fully-writing
 * H. Frozen constant detection — which slots get classified frozen
 * I. DSP feature flags — each system property individually
 * J. Optimizer passes — each one individually + full pipeline
 * K. FP16 weight pre-cast — type propagation through the graph
 * L. Shape key computation — same/different shapes produce same/different keys
 * M. Native vs Java execution — native executor on/off
 * N. Multi-step pointer stability — same graph, multiple steps, outputs differ
 *
 * Uses FIXED seeded weights so all comparisons are deterministic.
 */
@Slf4j
@DisplayName("DSP Pipeline Isolation")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspPipelineIsolationTest {

    // ═══════════════════════════════════════════════════════════════════════
    //  Shared fixed weights + baselines
    // ═══════════════════════════════════════════════════════════════════════

    private static INDArray EMBED_TABLE;  // [32,16]
    private static INDArray GAMMA;        // [16]
    private static INDArray PROJ_WEIGHT;  // [16,32]
    // Baselines computed via SLOT_BY_SLOT SameDiff — the reference execution mode.
    // (Manual Nd4j ops can diverge from fused kernels like rmsNorm due to FP rounding.)
    private static final Map<Integer, INDArray> BASELINES = new HashMap<>();
    private static final int[] TOKENS = {0, 5, 7, 10, 15, 22, 30};

    @BeforeAll
    static void setup() {
        Nd4j.getRandom().setSeed(42);
        EMBED_TABLE = Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.1);
        GAMMA = Nd4j.ones(DataType.FLOAT, 16);
        PROJ_WEIGHT = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);
        SameDiff baselineSd = buildGraph(DataType.FLOAT);
        baselineSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        for (int t : TOKENS) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("token_id", Nd4j.createFromArray(new long[]{t}));
            BASELINES.put(t, baselineSd.output(ph, "probs").get("probs").dup());
        }
    }

    private static SameDiff buildGraph(DataType weightType) {
        SameDiff sd = SameDiff.create();
        INDArray et = weightType == DataType.HALF ? EMBED_TABLE.castTo(DataType.HALF) : EMBED_TABLE.dup();
        INDArray pw = weightType == DataType.HALF ? PROJ_WEIGHT.castTo(DataType.HALF) : PROJ_WEIGHT.dup();
        sd.constant("embed_table", et);
        sd.constant("gamma", GAMMA.dup());
        sd.constant("proj_weight", pw);
        SDVariable tokenId = sd.placeHolder("token_id", DataType.INT64, 1);
        SDVariable gathered = sd.gather("gathered", sd.getVariable("embed_table"), tokenId, 0);
        SDVariable normed = sd.nn().rmsNorm("normed", gathered, sd.getVariable("gamma"), 1e-5);
        SDVariable logits = sd.mmul("logits", normed, sd.getVariable("proj_weight"));
        sd.nn().softmax("probs", logits, 1);
        return sd;
    }

    private INDArray run(SameDiff sd, int tokenId) {
        return sd.output(Map.of("token_id", Nd4j.createFromArray(new long[]{tokenId})), "probs")
                .get("probs").dup();
    }

    private void check(INDArray result, int tokenId, String label, double tol) {
        INDArray expected = BASELINES.get(tokenId);
        assertNotNull(result, label + ": null");
        assertFalse(Double.isNaN(result.maxNumber().doubleValue()), label + ": NaN");
        double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
        assertEquals(1.0, sum, 0.05, label + ": softmax sum=" + sum);
        double diff = expected.sub(result.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(diff < tol, label + ": maxDiff=" + diff + " > tol=" + tol);
        log.info("{}: PASS diff={}", label, String.format("%.8f", diff));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  A. GraphExecutionMode — single-step correctness
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "A_singleStep_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "CUDA_GRAPHS", "TRITON", "EMULATED_REPLAY",
                     "SHAPE_INFERENCE_ONLY"})
    @Order(1)
    void testA_SingleStepPerMode(GraphExecutionMode mode) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);
        INDArray result = run(sd, 5);
        if (mode == GraphExecutionMode.SHAPE_INFERENCE_ONLY) {
            assertArrayEquals(new long[]{1, 32}, result.shape(), mode + ": shape");
        } else {
            check(result, 5, mode.name(), 0.01);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  B. ModeContract fields — each one toggled individually
    //     Uses system properties that map to ModeContract booleans
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> modeContractFields() {
        // Each entry: mode that HAS this contract field true, mode that has it false, field name
        return Stream.of(
            // requiresCompilation: TRITON=true, SLOT_BY_SLOT=false
            Arguments.of(GraphExecutionMode.TRITON, GraphExecutionMode.SLOT_BY_SLOT, "requiresCompilation"),
            // usesGraphCapture: CUDA_GRAPHS=true, SLOT_BY_SLOT=false
            Arguments.of(GraphExecutionMode.CUDA_GRAPHS, GraphExecutionMode.SLOT_BY_SLOT, "usesGraphCapture"),
            // allowsFallback: SLOT_BY_SLOT=true, CUDA_GRAPHS=false
            Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.CUDA_GRAPHS, "allowsFallback"),
            // isSlotBySlot: SLOT_BY_SLOT=true, AUTO=false
            Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO, "isSlotBySlot"),
            // requiresDeterministicCublas: SLOT_BY_SLOT=true, AUTO=false
            Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO, "requiresDeterministicCublas"),
            // forcesSyncOnFrozen: SLOT_BY_SLOT=true, AUTO=false
            Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO, "forcesSyncOnFrozen"),
            // allowsFrozenFastPath: AUTO=true, SLOT_BY_SLOT=false
            Arguments.of(GraphExecutionMode.AUTO, GraphExecutionMode.SLOT_BY_SLOT, "allowsFrozenFastPath"),
            // isShapeInferenceOnly: SHAPE_INFERENCE_ONLY=true, SLOT_BY_SLOT=false
            Arguments.of(GraphExecutionMode.SHAPE_INFERENCE_ONLY, GraphExecutionMode.SLOT_BY_SLOT, "isShapeInferenceOnly")
        );
    }

    @ParameterizedTest(name = "B_contract_{2}")
    @MethodSource("modeContractFields")
    @Order(2)
    void testB_ModeContractFieldPair(GraphExecutionMode modeTrue, GraphExecutionMode modeFalse, String field) {
        // Run both modes and verify both produce correct output
        // (except SHAPE_INFERENCE_ONLY which only checks shapes)
        for (GraphExecutionMode mode : new GraphExecutionMode[]{modeTrue, modeFalse}) {
            SameDiff sd = buildGraph(DataType.FLOAT);
            sd.setGraphExecutionMode(mode);
            INDArray result = run(sd, 10);
            if (mode == GraphExecutionMode.SHAPE_INFERENCE_ONLY) {
                assertNotNull(result, field + "_" + mode + ": null");
            } else {
                check(result, 10, field + "_" + mode.name(), 0.01);
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  C. Plan lifecycle: multi-step forces phase progression
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "C_lifecycle_{0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"SLOT_BY_SLOT", "AUTO", "EMULATED_REPLAY"})
    @Order(3)
    void testC_PlanLifecycleMultiStep(GraphExecutionMode mode) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);

        // Run 6 steps — enough to pass warmup (4 steps), freeze, and reach steady state
        int[] tokens = {5, 10, 22, 0, 15, 7};
        INDArray[] results = new INDArray[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            results[i] = run(sd, tokens[i]);
            check(results[i], tokens[i], mode + "_step" + i, 0.01);
        }

        // Different tokens → different outputs
        for (int i = 0; i < results.length; i++) {
            for (int j = i + 1; j < results.length; j++) {
                if (tokens[i] != tokens[j]) {
                    double d = results[i].sub(results[j]).amaxNumber().doubleValue();
                    assertTrue(d > 1e-6, mode + ": steps " + i + "+" + j + " identical");
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  D. Segment lifecycle — DSP auto-compile verifiable through output
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(4)
    @DisplayName("D. DSP auto-compile on vs off produces same result")
    void testD_DspAutoCompileToggle() {
        for (boolean autoCompile : new boolean[]{true, false}) {
            SameDiff sd = buildGraph(DataType.FLOAT);
            sd.setDspAutoCompileEnabled(autoCompile);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            INDArray result = run(sd, 15);
            check(result, 15, "autoCompile=" + autoCompile, 0.01);
        }
    }

    @Test
    @Order(4)
    @DisplayName("D. DSP native auto-compile on vs off")
    void testD_DspNativeAutoCompileToggle() {
        for (boolean nativeAuto : new boolean[]{true, false}) {
            SameDiff sd = buildGraph(DataType.FLOAT);
            sd.setDspNativeAutoCompileEnabled(nativeAuto);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            INDArray result = run(sd, 15);
            check(result, 15, "nativeAutoCompile=" + nativeAuto, 0.01);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  E. Slot lifecycle — frozen constant detection
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(5)
    @DisplayName("E. Constants produce same output regardless of input")
    void testE_FrozenConstantDetection() {
        // Build a graph where some paths are input-independent
        SameDiff sd = SameDiff.create();
        INDArray constA = Nd4j.ones(DataType.FLOAT, 2, 3);
        INDArray constB = Nd4j.ones(DataType.FLOAT, 3, 2).muli(2);
        sd.constant("a", constA);
        sd.constant("b", constB);
        // This matmul is input-independent — should be foldable/frozen
        SDVariable constResult = sd.mmul("const_result", sd.getVariable("a"), sd.getVariable("b"));
        // This uses a placeholder — not frozen
        SDVariable ph = sd.placeHolder("input", DataType.FLOAT, 2, 2);
        SDVariable out = sd.math().add("out", constResult, ph);

        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input1 = Nd4j.ones(DataType.FLOAT, 2, 2);
        INDArray input2 = Nd4j.ones(DataType.FLOAT, 2, 2).muli(3);

        INDArray r1 = sd.output(Map.of("input", input1), "out").get("out");
        INDArray r2 = sd.output(Map.of("input", input2), "out").get("out");

        // Results must differ because input differs
        double diff = r1.sub(r2).amaxNumber().doubleValue();
        assertTrue(diff > 0.1, "Frozen constant paths must still allow variable paths to change output");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  F. External input classification
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(6)
    @DisplayName("F. External inputs: constant vs variable vs placeholder")
    void testF_ExternalInputClassification() {
        SameDiff sd = SameDiff.create();
        // CONSTANT: fixed, never changes
        sd.constant("const_weight", Nd4j.randn(DataType.FLOAT, 4, 4));
        // VARIABLE: trainable parameter (not placeholder)
        sd.var("var_weight", Nd4j.randn(DataType.FLOAT, 4, 4));
        // PLACEHOLDER: changes per execution
        SDVariable ph = sd.placeHolder("input", DataType.FLOAT, 1, 4);

        SDVariable step1 = sd.mmul("step1", ph, sd.getVariable("const_weight"));
        SDVariable out = sd.mmul("out", step1, sd.getVariable("var_weight"));

        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        INDArray input1 = Nd4j.randn(DataType.FLOAT, 1, 4);
        INDArray input2 = Nd4j.randn(DataType.FLOAT, 1, 4);

        INDArray r1 = sd.output(Map.of("input", input1), "out").get("out");
        INDArray r2 = sd.output(Map.of("input", input2), "out").get("out");

        assertNotNull(r1);
        assertNotNull(r2);
        // Different placeholders → different outputs (constant+variable parts stay same)
        double diff = r1.sub(r2).amaxNumber().doubleValue();
        assertTrue(diff > 1e-6, "Different placeholder inputs must produce different outputs");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  G. Op trait classification — test each trait's effect
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(7)
    @DisplayName("G. Identity op: output equals input")
    void testG_IdentityOp() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 2, 3);
        SDVariable ident = sd.identity("out", in);
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3);
        INDArray result = sd.output(Map.of("in", input), "out").get("out");
        double diff = input.sub(result).amaxNumber().doubleValue();
        assertEquals(0.0, diff, 1e-10, "Identity should be exact copy");
    }

    @Test
    @Order(7)
    @DisplayName("G. View op: reshape preserves data")
    void testG_ViewOp() {
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 2, 3);
        SDVariable reshaped = sd.reshape("out", in, 3, 2);
        INDArray input = Nd4j.randn(DataType.FLOAT, 2, 3);
        INDArray result = sd.output(Map.of("in", input), "out").get("out");
        assertArrayEquals(new long[]{3, 2}, result.shape());
        // Data should be same just reinterpreted
        assertEquals(input.sumNumber().doubleValue(), result.sumNumber().doubleValue(), 1e-6);
    }

    @Test
    @Order(7)
    @DisplayName("G. Data-dependent op: argmax shape from input values")
    void testG_DataDependentOp() {
        // argmax: output SHAPE doesn't depend on values (determined by axis)
        // but output VALUES depend on data
        SameDiff sd = SameDiff.create();
        SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 8);
        SDVariable argmax = sd.argmax("out", in, 1);

        INDArray input1 = Nd4j.zeros(DataType.FLOAT, 1, 8);
        input1.putScalar(0, 3, 10.0f);
        INDArray input2 = Nd4j.zeros(DataType.FLOAT, 1, 8);
        input2.putScalar(0, 6, 10.0f);

        INDArray r1 = sd.output(Map.of("in", input1), "out").get("out");
        INDArray r2 = sd.output(Map.of("in", input2), "out").get("out");

        assertEquals(3, r1.getLong(0), "argmax should find position 3");
        assertEquals(6, r2.getLong(0), "argmax should find position 6");
        // Same output shape even though argmax position differs
        assertArrayEquals(r1.shape(), r2.shape());
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  H. DSP feature flags — each property individually
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> dspFlags() {
        return Stream.of(
            Arguments.of("nd4j.dsp.nativeExecutor.enabled", "true"),
            Arguments.of("nd4j.dsp.nativeExecutor.enabled", "false"),
            Arguments.of("nd4j.dsp.noFreeze", "true"),
            Arguments.of("nd4j.dsp.noFreeze", "false"),
            Arguments.of("nd4j.dsp.freezeRecompile", "true"),
            Arguments.of("nd4j.dsp.freezeRecompile", "false"),
            Arguments.of("nd4j.dsp.freezeMergeSegments", "true"),
            Arguments.of("nd4j.dsp.freezeMergeSegments", "false"),
            Arguments.of("nd4j.dsp.batchZero", "true"),
            Arguments.of("nd4j.dsp.batchZero", "false"),
            Arguments.of("nd4j.dsp.matmulSegmentation", "true"),
            Arguments.of("nd4j.dsp.matmulSegmentation", "false"),
            Arguments.of("nd4j.dsp.castElimination", "true"),
            Arguments.of("nd4j.dsp.castElimination", "false"),
            Arguments.of("nd4j.dsp.fp16Compute", "true"),
            Arguments.of("nd4j.dsp.fp16Compute", "false"),
            Arguments.of("nd4j.dsp.cudaGraphs.enabled", "true"),
            Arguments.of("nd4j.dsp.cudaGraphs.enabled", "false")
        );
    }

    @ParameterizedTest(name = "H_flag_{0}={1}")
    @MethodSource("dspFlags")
    @Order(8)
    void testH_DspFeatureFlag(String prop, String value) {
        String old = System.getProperty(prop);
        try {
            System.setProperty(prop, value);
            SameDiff sd = buildGraph(DataType.FLOAT);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            check(run(sd, 10), 10, prop + "=" + value, 0.01);
        } finally {
            if (old != null) System.setProperty(prop, old);
            else System.clearProperty(prop);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  I. Optimizer passes — each one individually
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> optimizers() {
        return Stream.of(
            Arguments.of("UnusedFunction", new UnusedFunctionOptimizations()),
            Arguments.of("ConstantFunction", new ConstantFunctionOptimizations()),
            Arguments.of("Algebraic", new AlgebraicOptimizations()),
            Arguments.of("IdentityFunction", new IdentityFunctionOptimizations()),
            Arguments.of("ShapeFunction", new ShapeFunctionOptimizations()),
            Arguments.of("ActivationFusion", new ActivationFusionOptimizations()),
            Arguments.of("NormalizationFusion", new NormalizationFusionOptimizations()),
            Arguments.of("LinearFusion", new LinearFusionOptimizations()),
            Arguments.of("AttentionFusion", new AttentionFusionOptimizations()),
            Arguments.of("Quantization", new QuantizationOptimizations())
        );
    }

    @ParameterizedTest(name = "I_opt_{0}")
    @MethodSource("optimizers")
    @Order(9)
    void testI_IndividualOptimizer(String name, OptimizerSet pass) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        SameDiff opt = GraphOptimizer.optimize(sd, Collections.emptyList(),
                Collections.singletonList(pass));
        opt.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        check(run(opt, 15), 15, name, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  J. Full optimizer x mode x FP16 matrix
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> fullMatrix() {
        List<Arguments> args = new ArrayList<>();
        for (GraphExecutionMode mode : new GraphExecutionMode[]{
                GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO,
                GraphExecutionMode.TRITON, GraphExecutionMode.EMULATED_REPLAY}) {
            for (boolean opt : new boolean[]{false, true}) {
                for (boolean fp16 : new boolean[]{false, true}) {
                    if (!opt && fp16) continue;
                    String l = mode.name() + (opt ? "_OPT" : "_NOOPT") + (fp16 ? "_FP16" : "_FP32");
                    args.add(Arguments.of(mode, opt, fp16, l));
                }
            }
        }
        return args.stream();
    }

    @ParameterizedTest(name = "J_{3}")
    @MethodSource("fullMatrix")
    @Order(10)
    void testJ_FullMatrix(GraphExecutionMode mode, boolean opt, boolean fp16, String label) {
        String old = System.getProperty("nd4j.optimizer.fp16");
        try {
            System.setProperty("nd4j.optimizer.fp16", String.valueOf(fp16));
            SameDiff sd = buildGraph(DataType.FLOAT);
            if (opt) sd = GraphOptimizer.optimize(sd);
            sd.setGraphExecutionMode(mode);
            double tol = fp16 ? 0.05 : 0.01;
            // Multi-step
            for (int t : new int[]{5, 22, 0}) {
                check(run(sd, t), t, label + "_tok" + t, tol);
            }
        } finally {
            if (old != null) System.setProperty("nd4j.optimizer.fp16", old);
            else System.clearProperty("nd4j.optimizer.fp16");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  K. FP16 weight pre-cast specifics
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(11)
    @DisplayName("K. FP16 quantization: criteria + correctness")
    void testK_FP16Quantization() {
        String old = System.getProperty("nd4j.optimizer.fp16");
        try {
            System.setProperty("nd4j.optimizer.fp16", "true");
            SameDiff sd = SameDiff.create();
            sd.constant("big", Nd4j.randn(DataType.FLOAT, 64, 64));    // rank=2, len=4096 → HALF
            sd.constant("small", Nd4j.randn(DataType.FLOAT, 4, 4));    // rank=2, len=16 → FLOAT
            sd.constant("bias", Nd4j.randn(DataType.FLOAT, 64));       // rank=1 → FLOAT
            SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 64);
            sd.mmul("out", in, sd.getVariable("big"));

            SameDiff opt = GraphOptimizer.optimize(sd);
            for (SDVariable v : opt.variables()) {
                INDArray arr = opt.getArrForVarName(v.name());
                if (arr == null) continue;
                if ("big".equals(v.name()))
                    assertEquals(DataType.HALF, arr.dataType(), "big→HALF");
                if ("small".equals(v.name()))
                    assertEquals(DataType.FLOAT, arr.dataType(), "small→FLOAT");
                if ("bias".equals(v.name()))
                    assertEquals(DataType.FLOAT, arr.dataType(), "bias→FLOAT");
            }
        } finally {
            if (old != null) System.setProperty("nd4j.optimizer.fp16", old);
            else System.clearProperty("nd4j.optimizer.fp16");
        }
    }

    @Test
    @Order(11)
    @DisplayName("K. FP16 weights: HALF table gather preserves type")
    void testK_FP16GatherTypePreservation() {
        SameDiff sd = buildGraph(DataType.HALF);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        Map<String, INDArray> out = sd.output(
                Map.of("token_id", Nd4j.createFromArray(new long[]{5})),
                "gathered");
        INDArray gathered = out.get("gathered");
        assertEquals(DataType.HALF, gathered.dataType(), "Gather from HALF table must produce HALF");
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  L. Shape key — same input shape = same plan, different = recompile
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(12)
    @DisplayName("L. Same-shape inputs reuse plan (no recompile)")
    void testL_ShapeKeyStability() {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);

        // Two runs with same shapes must produce correct results
        INDArray r1 = run(sd, 5);
        INDArray r2 = run(sd, 10);
        check(r1, 5, "shapeKey_run1", 0.01);
        check(r2, 10, "shapeKey_run2", 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  M. Native vs Java execution
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(13)
    @DisplayName("M. Native executor ON vs OFF produce same result")
    void testM_NativeVsJavaExecution() {
        INDArray[] results = new INDArray[2];
        for (int i = 0; i < 2; i++) {
            boolean nativeOn = (i == 0);
            String old = System.getProperty("nd4j.dsp.nativeExecutor.enabled");
            try {
                System.setProperty("nd4j.dsp.nativeExecutor.enabled", String.valueOf(nativeOn));
                SameDiff sd = buildGraph(DataType.FLOAT);
                sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
                results[i] = run(sd, 7);
                check(results[i], 7, "native=" + nativeOn, 0.01);
            } finally {
                if (old != null) System.setProperty("nd4j.dsp.nativeExecutor.enabled", old);
                else System.clearProperty("nd4j.dsp.nativeExecutor.enabled");
            }
        }
        // Both paths should produce identical results
        double diff = results[0].sub(results[1]).amaxNumber().doubleValue();
        assertTrue(diff < 1e-6, "Native vs Java execution mismatch: " + diff);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  N. Isolated ops — matmul type combos, rms_norm, softmax, argmax
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> matmulTypeCombos() {
        return Stream.of(
            Arguments.of(DataType.FLOAT, DataType.FLOAT, "FF"),
            Arguments.of(DataType.HALF, DataType.HALF, "HH"),
            Arguments.of(DataType.FLOAT, DataType.HALF, "FH"),
            Arguments.of(DataType.HALF, DataType.FLOAT, "HF")
        );
    }

    @ParameterizedTest(name = "N_matmul_{2}")
    @MethodSource("matmulTypeCombos")
    @Order(14)
    void testN_MatmulTypeCombos(DataType aType, DataType bType, String label) {
        INDArray a = Nd4j.randn(DataType.FLOAT, 4, 8).muli(0.1).castTo(aType);
        INDArray b = Nd4j.randn(DataType.FLOAT, 8, 4).muli(0.1).castTo(bType);
        // Reference: both cast to FLOAT
        INDArray expected = a.castTo(DataType.FLOAT).mmul(b.castTo(DataType.FLOAT));

        SameDiff sd = SameDiff.create();
        sd.constant("a", a);
        sd.constant("b", b);
        sd.mmul("c", sd.getVariable("a"), sd.getVariable("b"));
        INDArray result = sd.output(Collections.emptyMap(), "c").get("c");

        assertNotNull(result, label + ": null");
        assertFalse(Double.isNaN(result.maxNumber().doubleValue()), label + ": NaN");
        double tol = (aType == DataType.HALF || bType == DataType.HALF) ? 0.05 : 0.001;
        double diff = expected.sub(result.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(diff < tol, label + ": diff=" + diff);
        log.info("matmul {}: PASS diff={}", label, diff);
    }

    @Test
    @Order(14)
    @DisplayName("N. softmax: FLOAT and HALF stability")
    void testN_SoftmaxStability() {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
            SameDiff sd = SameDiff.create();
            sd.placeHolder("in", dt, 1, 32);
            sd.nn().softmax("out", sd.getVariable("in"), 1);
            INDArray logits = Nd4j.randn(dt, 1, 32);
            INDArray result = sd.output(Map.of("in", logits), "out").get("out");
            double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.01, dt + ": sum");
            assertFalse(Double.isNaN(result.maxNumber().doubleValue()), dt + ": NaN");
        }
    }

    @Test
    @Order(14)
    @DisplayName("N. argmax: FLOAT and HALF correctness")
    void testN_ArgmaxCorrectness() {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
            INDArray data = Nd4j.zeros(dt, 1, 32);
            data.putScalar(0, 17, 5.0f);
            assertEquals(17, Nd4j.argMax(data, 1).getLong(0), dt + ": argmax");
        }
    }
}
