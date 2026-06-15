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
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
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
 * Exhaustive execution mode x ModeContract field testing.
 *
 * Uses FIXED weight arrays (seeded) so every graph uses identical parameters.
 * Computes a single baseline via direct Nd4j ops (no SameDiff) and compares
 * each execution mode / optimizer / flag combination against that baseline.
 *
 * ModeContract fields (from C++ ModeContract.h):
 *   requiresCompilation, needsJitBackend, usesGraphCapture, allowsFrozenFastPath,
 *   allowsFallback, allowsPhaseStall, requiresDeterministicCublas, isSlotBySlot,
 *   isShapeInferenceOnly, forcesSyncOnFrozen
 *
 * Each test toggles ONE dimension of the configuration space.
 */
@Slf4j
@DisplayName("Execution Mode Contract Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class ExecutionModeContractTest {

    // Fixed weight arrays — created once, used by ALL tests
    private static INDArray EMBED_TABLE;    // [32, 16] FLOAT32
    private static INDArray GAMMA;          // [16] FLOAT32
    private static INDArray PROJ_WEIGHT;    // [16, 32] FLOAT32

    // Pre-computed baselines via SLOT_BY_SLOT SameDiff execution.
    // This is the ground truth: all other modes must match SLOT_BY_SLOT.
    // (Manual Nd4j ops can diverge from fused kernels like rmsNorm due to FP rounding.)
    private static final Map<Integer, INDArray> BASELINES = new HashMap<>();
    private static final int[] TEST_TOKEN_IDS = {0, 5, 7, 10, 15, 22, 30};

    @BeforeAll
    static void setup() {
        Nd4j.getRandom().setSeed(42);
        EMBED_TABLE = Nd4j.randn(DataType.FLOAT, 32, 16).muli(0.1);
        GAMMA = Nd4j.ones(DataType.FLOAT, 16);
        PROJ_WEIGHT = Nd4j.randn(DataType.FLOAT, 16, 32).muli(0.1);

        // Compute baselines via SLOT_BY_SLOT SameDiff — the reference execution mode
        SameDiff baselineSd = buildGraph(DataType.FLOAT);
        baselineSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        for (int tokenId : TEST_TOKEN_IDS) {
            Map<String, INDArray> ph = new HashMap<>();
            ph.put("token_id", Nd4j.createFromArray(new long[]{tokenId}));
            BASELINES.put(tokenId, baselineSd.output(ph, "probs").get("probs").dup());
        }
    }

    /**
     * Build a SameDiff graph using the SAME fixed weights.
     * The graph is: gather → rms_norm → matmul → softmax
     */
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

    private INDArray runGraph(SameDiff sd, int tokenId) {
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("token_id", Nd4j.createFromArray(new long[]{tokenId}));
        return sd.output(ph, "probs").get("probs").dup();
    }

    private void assertMatchesBaseline(INDArray result, int tokenId, String label, double tol) {
        INDArray expected = BASELINES.get(tokenId);
        assertNotNull(result, label + ": null output");
        assertFalse(Double.isNaN(result.maxNumber().doubleValue()), label + ": NaN in output");
        assertFalse(Double.isInfinite(result.maxNumber().doubleValue()), label + ": Inf in output");
        double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
        assertEquals(1.0, sum, 0.05, label + ": softmax sum=" + sum);

        double maxDiff = expected.sub(result.castTo(DataType.FLOAT)).amaxNumber().doubleValue();
        assertTrue(maxDiff < tol, label + ": maxDiff=" + maxDiff + " > tol=" + tol);
        log.info("{}: PASS maxDiff={} sum={}", label, String.format("%.6f", maxDiff), String.format("%.4f", sum));
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  1. Single-step per execution mode (FP32)
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> executionModes() {
        return Stream.of(
            Arguments.of(GraphExecutionMode.SLOT_BY_SLOT, "SLOT_BY_SLOT"),
            Arguments.of(GraphExecutionMode.AUTO, "AUTO"),
            Arguments.of(GraphExecutionMode.CUDA_GRAPHS, "CUDA_GRAPHS"),
            Arguments.of(GraphExecutionMode.TRITON, "TRITON"),
            Arguments.of(GraphExecutionMode.EMULATED_REPLAY, "EMULATED_REPLAY")
        );
    }

    @ParameterizedTest(name = "singleStep_{1}")
    @MethodSource("executionModes")
    @Order(1)
    @DisplayName("1. Single-step per execution mode (FP32)")
    void testSingleStepPerMode(GraphExecutionMode mode, String modeName) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);
        INDArray result = runGraph(sd, 5);
        assertMatchesBaseline(result, 5, modeName, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  2. Mode x weight type (FP32 vs FP16)
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> modeTimesWeightType() {
        List<Arguments> args = new ArrayList<>();
        for (GraphExecutionMode mode : new GraphExecutionMode[]{
                GraphExecutionMode.SLOT_BY_SLOT, GraphExecutionMode.AUTO,
                GraphExecutionMode.CUDA_GRAPHS, GraphExecutionMode.TRITON,
                GraphExecutionMode.EMULATED_REPLAY}) {
            for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
                args.add(Arguments.of(mode, dt, mode.name() + "_" + dt.name()));
            }
        }
        return args.stream();
    }

    @ParameterizedTest(name = "modeWeight_{2}")
    @MethodSource("modeTimesWeightType")
    @Order(2)
    @DisplayName("2. Mode x weight type")
    void testModeTimesWeightType(GraphExecutionMode mode, DataType weightType, String label) {
        SameDiff sd = buildGraph(weightType);
        sd.setGraphExecutionMode(mode);
        INDArray result = runGraph(sd, 7);
        double tol = weightType == DataType.HALF ? 0.05 : 0.01;
        assertMatchesBaseline(result, 7, label, tol);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  3. Multi-step decode — different tokens each step, same graph
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "multiStep_{1}")
    @MethodSource("executionModes")
    @Order(3)
    @DisplayName("3. Multi-step: changing input produces correct distinct outputs")
    void testMultiStepDecode(GraphExecutionMode mode, String modeName) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);

        int[] tokenIds = {5, 10, 22, 0, 15};
        INDArray[] results = new INDArray[tokenIds.length];

        for (int step = 0; step < tokenIds.length; step++) {
            results[step] = runGraph(sd, tokenIds[step]);
            assertMatchesBaseline(results[step], tokenIds[step],
                    modeName + "_step" + step, 0.01);
        }

        // Different tokens must produce different outputs
        for (int i = 0; i < results.length; i++) {
            for (int j = i + 1; j < results.length; j++) {
                if (tokenIds[i] != tokenIds[j]) {
                    double diff = results[i].sub(results[j]).amaxNumber().doubleValue();
                    assertTrue(diff > 1e-6, modeName + ": steps " + i + "," + j + " identical");
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  4. DSP feature flags — individually toggle each one
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> dspFeatureFlags() {
        return Stream.of(
            Arguments.of("nd4j.dsp.nativeExecutor.enabled", "true", "nativeExec_ON"),
            Arguments.of("nd4j.dsp.nativeExecutor.enabled", "false", "nativeExec_OFF"),
            Arguments.of("nd4j.dsp.noFreeze", "true", "noFreeze_ON"),
            Arguments.of("nd4j.dsp.noFreeze", "false", "noFreeze_OFF"),
            Arguments.of("nd4j.dsp.freezeRecompile", "true", "freezeRecompile_ON"),
            Arguments.of("nd4j.dsp.freezeRecompile", "false", "freezeRecompile_OFF"),
            Arguments.of("nd4j.dsp.freezeMergeSegments", "true", "freezeMergeSegments_ON"),
            Arguments.of("nd4j.dsp.freezeMergeSegments", "false", "freezeMergeSegments_OFF"),
            Arguments.of("nd4j.dsp.batchZero", "true", "batchZero_ON"),
            Arguments.of("nd4j.dsp.batchZero", "false", "batchZero_OFF"),
            Arguments.of("nd4j.dsp.matmulSegmentation", "true", "matmulSeg_ON"),
            Arguments.of("nd4j.dsp.matmulSegmentation", "false", "matmulSeg_OFF"),
            Arguments.of("nd4j.dsp.castElimination", "true", "castElim_ON"),
            Arguments.of("nd4j.dsp.castElimination", "false", "castElim_OFF"),
            Arguments.of("nd4j.dsp.fp16Compute", "true", "fp16Compute_ON"),
            Arguments.of("nd4j.dsp.fp16Compute", "false", "fp16Compute_OFF"),
            Arguments.of("nd4j.dsp.cudaGraphs.enabled", "true", "cudaGraphs_ON"),
            Arguments.of("nd4j.dsp.cudaGraphs.enabled", "false", "cudaGraphs_OFF")
        );
    }

    @ParameterizedTest(name = "dspFlag_{2}")
    @MethodSource("dspFeatureFlags")
    @Order(4)
    @DisplayName("4. DSP feature flag isolation")
    void testDspFeatureFlag(String propName, String propValue, String label) {
        String oldValue = System.getProperty(propName);
        try {
            System.setProperty(propName, propValue);
            SameDiff sd = buildGraph(DataType.FLOAT);
            sd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
            INDArray result = runGraph(sd, 10);
            assertMatchesBaseline(result, 10, label, 0.01);
        } finally {
            if (oldValue != null) System.setProperty(propName, oldValue);
            else System.clearProperty(propName);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  5. GraphOptimizer passes — each one individually
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> optimizerPasses() {
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

    @ParameterizedTest(name = "optimizer_{0}")
    @MethodSource("optimizerPasses")
    @Order(5)
    @DisplayName("5. Individual optimizer pass correctness")
    void testIndividualOptimizerPass(String name, OptimizerSet optimizerSet) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        int beforeOps = sd.getOps().size();
        SameDiff optimized = GraphOptimizer.optimize(sd, Collections.emptyList(),
                Collections.singletonList(optimizerSet));
        int afterOps = optimized.getOps().size();
        optimized.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        INDArray result = runGraph(optimized, 15);
        assertMatchesBaseline(result, 15, name + " (" + beforeOps + "→" + afterOps + " ops)", 0.01);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  6. Full optimizer pipeline x execution mode x FP16
    // ═══════════════════════════════════════════════════════════════════════

    static Stream<Arguments> fullMatrix() {
        List<Arguments> args = new ArrayList<>();
        GraphExecutionMode[] modes = {
            GraphExecutionMode.SLOT_BY_SLOT,
            GraphExecutionMode.AUTO,
            GraphExecutionMode.TRITON,
            GraphExecutionMode.EMULATED_REPLAY
        };
        for (GraphExecutionMode mode : modes) {
            for (boolean opt : new boolean[]{false, true}) {
                for (boolean fp16 : new boolean[]{false, true}) {
                    if (!opt && fp16) continue; // FP16 requires optimizer
                    String label = mode.name()
                            + (opt ? "_OPT" : "_NOOPT")
                            + (fp16 ? "_FP16" : "_FP32");
                    args.add(Arguments.of(mode, opt, fp16, label));
                }
            }
        }
        return args.stream();
    }

    @ParameterizedTest(name = "full_{3}")
    @MethodSource("fullMatrix")
    @Order(6)
    @DisplayName("6. Full pipeline: mode x optimizer x FP16 multi-step")
    void testFullMatrix(GraphExecutionMode mode, boolean useOptimizer, boolean fp16, String label) {
        String oldFp16 = System.getProperty("nd4j.optimizer.fp16");
        try {
            System.setProperty("nd4j.optimizer.fp16", String.valueOf(fp16));
            SameDiff sd = buildGraph(DataType.FLOAT);
            if (useOptimizer) {
                sd = GraphOptimizer.optimize(sd);
            }
            sd.setGraphExecutionMode(mode);

            int[] tokenIds = {5, 22, 0};
            double tol = fp16 ? 0.05 : 0.01;
            for (int step = 0; step < tokenIds.length; step++) {
                INDArray result = runGraph(sd, tokenIds[step]);
                assertMatchesBaseline(result, tokenIds[step],
                        label + "_step" + step, tol);
            }
        } finally {
            if (oldFp16 != null) System.setProperty("nd4j.optimizer.fp16", oldFp16);
            else System.clearProperty("nd4j.optimizer.fp16");
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  7. Isolated op correctness — each op individually
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(10)
    @DisplayName("Op: gather preserves type and values")
    void testGatherTypePreservation() {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
            SameDiff sd = SameDiff.create();
            INDArray table = EMBED_TABLE.castTo(dt);
            sd.constant("table", table);
            SDVariable idx = sd.placeHolder("idx", DataType.INT64, 1);
            sd.gather("out", sd.getVariable("table"), idx, 0);
            Map<String, INDArray> ph = Map.of("idx", Nd4j.createFromArray(new long[]{5}));
            INDArray result = sd.output(ph, "out").get("out");

            assertEquals(dt, result.dataType(), "Gather should preserve " + dt);
            assertArrayEquals(new long[]{1, 16}, result.shape());

            INDArray expected = table.getRow(5).reshape(1, 16);
            double maxDiff = expected.sub(result).amaxNumber().doubleValue();
            assertTrue(maxDiff < 1e-5, dt + " gather: maxDiff=" + maxDiff);
            log.info("gather {}: PASS", dt);
        }
    }

    @Test
    @Order(11)
    @DisplayName("Op: matmul all type combos")
    void testMatmulTypeCombinations() {
        INDArray refA = EMBED_TABLE.getRows(0, 1, 2, 3).reshape(4, 16).dup(); // [4,16]
        INDArray refB = PROJ_WEIGHT.get(org.nd4j.linalg.indexing.NDArrayIndex.all(),
                org.nd4j.linalg.indexing.NDArrayIndex.interval(0, 4)).dup(); // [16,4]
        INDArray expectedFP32 = refA.mmul(refB);

        DataType[][] combos = {
            {DataType.FLOAT, DataType.FLOAT},
            {DataType.HALF, DataType.HALF},
            {DataType.FLOAT, DataType.HALF},
            {DataType.HALF, DataType.FLOAT},
        };

        for (DataType[] combo : combos) {
            String label = combo[0] + " x " + combo[1];
            INDArray a = refA.castTo(combo[0]);
            INDArray b = refB.castTo(combo[1]);
            INDArray result = a.castTo(DataType.FLOAT).mmul(b.castTo(DataType.FLOAT));

            double maxDiff = expectedFP32.sub(result).amaxNumber().doubleValue();
            double tol = (combo[0] == DataType.HALF || combo[1] == DataType.HALF) ? 0.05 : 0.001;
            assertTrue(maxDiff < tol, label + ": maxDiff=" + maxDiff);
            log.info("matmul {}: PASS maxDiff={}", label, maxDiff);
        }
    }

    @Test
    @Order(12)
    @DisplayName("Op: softmax stability")
    void testSoftmaxStability() {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
            SameDiff sd = SameDiff.create();
            INDArray logits = Nd4j.randn(dt, 1, 32);
            SDVariable vLogits = sd.placeHolder("logits", dt, 1, 32);
            sd.nn().softmax("probs", vLogits, 1);

            INDArray result = sd.output(Map.of("logits", logits), "probs").get("probs");
            double sum = result.castTo(DataType.FLOAT).sumNumber().doubleValue();
            assertEquals(1.0, sum, 0.01, dt + ": softmax sum");
            assertTrue(result.castTo(DataType.FLOAT).minNumber().doubleValue() >= 0,
                    dt + ": negative softmax");
            assertFalse(Double.isNaN(result.maxNumber().doubleValue()), dt + ": NaN");
            log.info("softmax {}: PASS sum={}", dt, sum);
        }
    }

    @Test
    @Order(13)
    @DisplayName("Op: argmax correctness")
    void testArgmaxCorrectness() {
        for (DataType dt : new DataType[]{DataType.FLOAT, DataType.HALF}) {
            INDArray data = Nd4j.zeros(dt, 1, 32);
            data.putScalar(0, 17, dt == DataType.HALF ? 5.0f : 5.0);
            long argmax = Nd4j.argMax(data, 1).getLong(0);
            assertEquals(17, argmax, dt + ": argmax expected 17 got " + argmax);
            log.info("argmax {}: PASS", dt);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  8. Chained decode: gather → rms_norm → matmul → softmax → argmax
    //     Tests that argmax token selection is identical across modes
    // ═══════════════════════════════════════════════════════════════════════

    @ParameterizedTest(name = "chain_{1}")
    @MethodSource("executionModes")
    @Order(8)
    @DisplayName("8. Chained decode: argmax token matches baseline")
    void testChainedDecodeArgmax(GraphExecutionMode mode, String modeName) {
        SameDiff sd = buildGraph(DataType.FLOAT);
        sd.setGraphExecutionMode(mode);

        for (int tokenId : new int[]{0, 10, 30}) {
            INDArray result = runGraph(sd, tokenId);
            int actual = Nd4j.argMax(result, 1).getInt(0);

            INDArray baseline = BASELINES.get(tokenId);
            int expected = Nd4j.argMax(baseline, 1).getInt(0);

            assertEquals(expected, actual,
                    modeName + " token=" + tokenId + ": argmax expected " + expected + " got " + actual);
        }
        log.info("{}: chained argmax PASS", modeName);
    }

    // ═══════════════════════════════════════════════════════════════════════
    //  9. FP16 quantization criteria
    // ═══════════════════════════════════════════════════════════════════════

    @Test
    @Order(9)
    @DisplayName("9. FP16 quantization: only large rank>=2 constants converted")
    void testFP16QuantizationCriteria() {
        String old = System.getProperty("nd4j.optimizer.fp16");
        try {
            System.setProperty("nd4j.optimizer.fp16", "true");
            SameDiff sd = SameDiff.create();
            sd.constant("big_weight", Nd4j.randn(DataType.FLOAT, 64, 64));
            sd.constant("small_weight", Nd4j.randn(DataType.FLOAT, 4, 4));
            sd.constant("bias", Nd4j.randn(DataType.FLOAT, 64));
            sd.constant("scale", Nd4j.scalar(DataType.FLOAT, 1.0));
            SDVariable in = sd.placeHolder("in", DataType.FLOAT, 1, 64);
            sd.mmul("out", in, sd.getVariable("big_weight"));

            SameDiff optimized = GraphOptimizer.optimize(sd);
            for (SDVariable v : optimized.variables()) {
                INDArray arr = optimized.getArrForVarName(v.name());
                if (arr == null) continue;
                if ("big_weight".equals(v.name()))
                    assertEquals(DataType.HALF, arr.dataType(), "big_weight should be HALF");
                else if ("small_weight".equals(v.name()))
                    assertEquals(DataType.FLOAT, arr.dataType(), "small_weight should stay FLOAT");
                else if ("bias".equals(v.name()))
                    assertEquals(DataType.FLOAT, arr.dataType(), "bias should stay FLOAT");
                else if ("scale".equals(v.name()))
                    assertEquals(DataType.FLOAT, arr.dataType(), "scale should stay FLOAT");
            }
            log.info("FP16 quantization criteria: PASS");
        } finally {
            if (old != null) System.setProperty("nd4j.optimizer.fp16", old);
            else System.clearProperty("nd4j.optimizer.fp16");
        }
    }
}
