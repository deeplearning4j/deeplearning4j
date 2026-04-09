package org.eclipse.deeplearning4j.vlm;

import lombok.extern.slf4j.Slf4j;
import org.eclipse.deeplearning4j.model.benchmark.*;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.junit.jupiter.api.*;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Benchmark-subgraph correctness tests using small synthetic SameDiff models.
 *
 * Each test builds a minimal SameDiff graph with the op-type signature of
 * a target SmolDocling decode subgraph, validates that the subgraph produces
 * correct outputs under Triton section fusion and the full execution mode matrix.
 *
 * <pre>
 *   cd platform-tests && mvn test \
 *     -Dtest=TestBenchmarkSubgraphCorrectness \
 *     -Dbackend.artifactId=nd4j-cuda-12.9
 * </pre>
 */
@Slf4j
public class TestBenchmarkSubgraphCorrectness {

    private static final ValidationConfig TOLERANCE = ValidationConfig.tf32Tolerant();

    // ========================================================================
    // Test 1: Early Gather Ladder (GATHER, CONST_GEN, SHAPE_MANIP)
    // ========================================================================
    @Test
    @DisplayName("1. Early gather ladder: GATHER + CONST_GEN + SHAPE_MANIP")
    public void testEarlyGatherLadder() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 32, 16));
        SDVariable idx = sd.placeHolder("idx", DataType.INT64, 4);
        SDVariable gathered = sd.gather("gather_out", table, idx, 0);
        SDVariable reshaped = sd.reshape("reshaped", gathered, 4, 4, 4);
        SDVariable constGen = sd.constant("const_val", Nd4j.ones(DataType.FLOAT, 4, 4, 4).mul(0.1));
        SDVariable output = sd.math.add("output", reshaped, constGen);
        sd.setOutputs("output");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx", Nd4j.createFromArray(new long[]{0, 1, 2, 3}));

        runSubgraphMatrix(sd, inputs, "early_gather_ladder");
        sd.close();
    }

    // ========================================================================
    // Test 2: Normalization Tail (GATHER, CONST_GEN, NORMALIZATION via SQUARE+REDUCE)
    // ========================================================================
    @Test
    @DisplayName("2. Normalization tail: GATHER + CONST_GEN + REDUCTION")
    public void testNormalizationTail() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 32, 16));
        SDVariable idx = sd.placeHolder("idx", DataType.INT64, 4);
        SDVariable gathered = sd.gather("gather_out", table, idx, 0);
        SDVariable squared = sd.math.square("squared", gathered);
        SDVariable constGen = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 4, 16).mul(1e-5f));
        SDVariable output = sd.math.add("output", squared, constGen);
        sd.setOutputs("output");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx", Nd4j.createFromArray(new long[]{5, 10, 15, 20}));

        runSubgraphMatrix(sd, inputs, "normalization_tail");
        sd.close();
    }

    // ========================================================================
    // Test 3: Mask Reformat (CONST_GEN, GATHER, ELEMENTWISE, CONCAT, TILE, SHAPE_MANIP)
    // ========================================================================
    @Test
    @DisplayName("3. Mask reformat: CONST_GEN + GATHER + ELEMENTWISE + CONCAT + TILE")
    public void testMaskReformatSliceTile() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 16, 8));
        SDVariable idx = sd.placeHolder("idx", DataType.INT64, 4);
        SDVariable gathered = sd.gather("gather_out", table, idx, 0);
        SDVariable reshaped = sd.reshape("reshaped", gathered, 4, 2, 4);
        SDVariable tiled = sd.tile("tiled", reshaped, new int[]{1, 2, 1});
        SDVariable elemOut = sd.math.mul("elem_out", tiled, tiled);
        SDVariable elemTiled = sd.tile("elem_tiled", elemOut, new int[]{1, 1, 2});
        SDVariable constGen = sd.constant("const_mask", Nd4j.ones(DataType.FLOAT, 4, 4, 4));
        SDVariable concatOut = sd.concat("concat_out", 2, tiled, constGen);
        SDVariable output = sd.math.add("mask_out", concatOut, elemTiled);
        sd.setOutputs("mask_out", "concat_out");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx", Nd4j.createFromArray(new long[]{0, 2, 4, 6}));

        runSubgraphMatrix(sd, inputs, "mask_reformat");
        sd.close();
    }

    // ========================================================================
    // Test 4: simple_const_gather (CONST_GEN, GATHER, SHAPE_MANIP, ELEMENTWISE)
    // ========================================================================
    @Test
    @DisplayName("4. simple_const_gather: CONST_GEN + GATHER + SHAPE_MANIP + ELEMENTWISE")
    public void testSimpleConstGather() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 32, 16));
        SDVariable idx = sd.placeHolder("idx", DataType.INT64, 4);
        SDVariable gathered = sd.gather("gather_out", table, idx, 0);
        SDVariable reshaped = sd.reshape("reshaped", gathered, 2, 2, 16);
        SDVariable constGen = sd.constant("bias", Nd4j.ones(DataType.FLOAT, 2, 2, 16).mul(0.5));
        SDVariable added = sd.math.add("added", reshaped, constGen);
        SDVariable squared = sd.math.square("squared", added);
        SDVariable output = sd.math.mul("output", squared, reshaped);
        sd.setOutputs("output", "squared");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx", Nd4j.createFromArray(new long[]{0, 1, 2, 3}));

        runSubgraphMatrix(sd, inputs, "simple_const_gather");
        sd.close();
    }

    // ========================================================================
    // Test 5: concat_ladder+gather_ladder (CONCAT, GATHER, SHAPE_MANIP, CONST_GEN)
    // ========================================================================
    @Test
    @DisplayName("5. concat_ladder+gather_ladder: CONCAT + GATHER + SHAPE_MANIP + CONST_GEN")
    public void testConcatLadderGatherLadder() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 32, 8));
        SDVariable idx1 = sd.placeHolder("idx1", DataType.INT64, 4);
        SDVariable idx2 = sd.placeHolder("idx2", DataType.INT64, 4);
        SDVariable g1 = sd.gather("gather1", table, idx1, 0);
        SDVariable g2 = sd.gather("gather2", table, idx2, 0);
        SDVariable concatOut = sd.concat("concat_out", 1, g1, g2);
        SDVariable reshaped = sd.reshape("reshaped", concatOut, 4, 16);
        SDVariable constGen = sd.constant("scale", Nd4j.ones(DataType.FLOAT, 4, 16).mul(0.25));
        SDVariable output = sd.math.add("output", reshaped, constGen);
        sd.setOutputs("output", "concat_out");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx1", Nd4j.createFromArray(new long[]{0, 1, 2, 3}));
        inputs.put("idx2", Nd4j.createFromArray(new long[]{4, 5, 6, 7}));

        runSubgraphMatrix(sd, inputs, "concat_gather_ladder");
        sd.close();
    }

    // ========================================================================
    // Test 6: Short Attention Tail (CONST_GEN, GATHER, SHAPE_MANIP, ELEMENTWISE, ATTENTION via mmul+softmax)
    // ========================================================================
    @Test
    @DisplayName("6. Short attention tail: mmul + softmax + ELEMENTWISE")
    public void testShortAttentionTail() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();
        SDVariable query = sd.placeHolder("query", DataType.FLOAT, 2, 4, 8);
        SDVariable key = sd.placeHolder("key", DataType.FLOAT, 2, 4, 8);
        SDVariable value = sd.placeHolder("value", DataType.FLOAT, 2, 4, 8);
        SDVariable qScaled = sd.math.mul("q_scaled", query, sd.constant(0.35355339f));
        SDVariable qk = sd.mmul("qk", qScaled, key, false, true, false);
        SDVariable attnScores = sd.nn.softmax("attn_scores", qk, -1);
        SDVariable attnOut = sd.mmul("attn_out", attnScores, value, false, false, false);
        SDVariable reshaped = sd.reshape("output", attnOut, 2, 32);
        sd.setOutputs("output", "attn_out", "attn_scores");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("query", Nd4j.rand(DataType.FLOAT, 2, 4, 8));
        inputs.put("key", Nd4j.rand(DataType.FLOAT, 2, 4, 8));
        inputs.put("value", Nd4j.rand(DataType.FLOAT, 2, 4, 8));

        runSubgraphMatrix(sd, inputs, "attention_tail");
        sd.close();
    }

    // ========================================================================
    // Test 7: Full Attention-Prep Tail
    // (SHAPE_MANIP, CONCAT, CONST_GEN, GATHER, STACK, ELEMENTWISE, ATTENTION)
    // ========================================================================
    @Test
    @DisplayName("7. Full attention-prep tail: full prep + scaled dot-product attention")
    public void testFullAttentionPrepTail() {
        if (!Nd4j.getNativeOps().isTritonAvailable()) return;
        SameDiff sd = SameDiff.create();

        // Phase 1: GATHER (embedding lookup)
        SDVariable table = sd.var("table", Nd4j.rand(DataType.FLOAT, 32, 24));
        SDVariable idx = sd.placeHolder("idx", DataType.INT64, 4);
        SDVariable embeddings = sd.gather("embeddings", table, idx, 0);
        SDVariable reshaped = sd.reshape("reshaped", embeddings, 4, 4, 6);

        // Phase 2: CONST_GEN + ELEMENTWISE (position bias)
        SDVariable bias = sd.constant("pos_bias", Nd4j.rand(DataType.FLOAT, 4, 4, 6));
        SDVariable withBias = sd.math.add("with_bias", reshaped, bias);

        // Phase 3: SPLIT into Q, K, V via strided slice (each [4,4,2])
        SDVariable q = sd.stridedSlice("query", withBias,
                new long[]{0, 0, 0}, new long[]{4, 4, 2}, new long[]{1, 1, 1}, 0, 0, 0, 0, 0);
        SDVariable k = sd.stridedSlice("key", withBias,
                new long[]{0, 0, 2}, new long[]{4, 4, 4}, new long[]{1, 1, 1}, 0, 0, 0, 0, 0);
        SDVariable v = sd.stridedSlice("value", withBias,
                new long[]{0, 0, 4}, new long[]{4, 4, 6}, new long[]{1, 1, 1}, 0, 0, 0, 0, 0);

        // Phase 4: ATTENTION (scaled dot-product, flatten to 2D for mmul)
        SDVariable qFlat = sd.reshape("q_flat", q, 16, 2);
        SDVariable kFlat = sd.reshape("k_flat", k, 16, 2);
        SDVariable vFlat = sd.reshape("v_flat", v, 16, 2);
        SDVariable qScaled = sd.math.mul("q_scaled", qFlat, sd.constant(0.5f));
        SDVariable qk = sd.mmul("qk", qScaled, kFlat, false, true, false);
        SDVariable attnScores = sd.nn.softmax("attn_scores", qk, -1);
        SDVariable attnOut = sd.mmul("attn_out", attnScores, vFlat, false, false, false);

        SDVariable output = sd.reshape("output", attnOut, 16, 2);
        sd.setOutputs("output", "attn_out", "attn_scores");

        Map<String, INDArray> inputs = new LinkedHashMap<>();
        inputs.put("idx", Nd4j.createFromArray(new long[]{0, 1, 2, 3}));

        runSubgraphMatrix(sd, inputs, "attention_prep_tail");
        sd.close();
    }

    // ========================================================================
    // Core validation: run one subgraph across the execution mode matrix
    // ========================================================================

    private void runSubgraphMatrix(SameDiff sd, Map<String, INDArray> inputs, String label) {
        // Reference: standard output() (no DSP)
        Map<String, INDArray> refOutputs = sd.output(inputs, sd.outputs().toArray(new String[0]));
        Map<String, INDArray> refDuped = dupAll(refOutputs);

        // Test configs
        BenchmarkConfig[] configs = {
            BenchmarkConfig.create(label + "_triton")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ELEMENTWISE,SHAPE_MANIPULATION,ATTENTION,TILE,STRIDED_SLICE,REDUCTION,NORMALIZATION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonNumWarps(4)
                    .tritonNumStages(1)
                    .maxTokens(1).minDiversityPct(0),

            BenchmarkConfig.create(label + "_optimal")
                    .tritonIncludeTypes("CONST_GEN,GATHER,CONCAT,SPLIT,STACK,ELEMENTWISE,SHAPE_MANIPULATION,ATTENTION,TILE,STRIDED_SLICE,REDUCTION,NORMALIZATION")
                    .tritonSectionFusion(true)
                    .tritonCompileAll(true)
                    .tritonNumWarps(4)
                    .tritonNumStages(1)
                    .tritonConsolidatedArgTable(true)
                    .tritonArgDirtyTracking(true)
                    .cublasTf32(true)
                    .tritonTf32(true)
                    .dspBatchedGemm(true)
                    .maxTokens(1).minDiversityPct(0),
        };

        for (BenchmarkConfig config : configs) {
            Map<String, INDArray> testOutputs = runWithConfig(sd, config, inputs);
            compareSubgraphOutputs(label + "/" + config.getName(), refDuped, testOutputs);
            closeAll(testOutputs);
        }

        closeAll(refDuped);
    }

    // ========================================================================
    // Execution helpers
    // ========================================================================

    private Map<String, INDArray> runWithConfig(SameDiff sd, BenchmarkConfig config,
                                                  Map<String, INDArray> inputs) {
        sd.resetSession();
        sd.clearDynamicShapePlanCache();
        BenchmarkConfigApplier.apply(config);
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.compileNativeDynamicShapePlan(sd.outputs(), config.getExecutionMode(), true);
        Map<String, INDArray> result = sd.outputDirect(inputs, sd.outputs().toArray(new String[0]));
        return dupAll(result);
    }

    private Map<String, INDArray> dupAll(Map<String, INDArray> outputs) {
        Map<String, INDArray> duped = new LinkedHashMap<>();
        for (Map.Entry<String, INDArray> e : outputs.entrySet()) {
            duped.put(e.getKey(), e.getValue().dup());
        }
        return duped;
    }

    private void closeAll(Map<String, INDArray> outputs) {
        for (INDArray arr : outputs.values()) {
            if (arr != null && !arr.wasClosed()) { arr.setCloseable(true); arr.close(); }
        }
    }

    private void compareSubgraphOutputs(String label,
                                          Map<String, INDArray> reference,
                                          Map<String, INDArray> test) {
        for (Map.Entry<String, INDArray> entry : reference.entrySet()) {
            String varName = entry.getKey();
            INDArray refArr = entry.getValue();
            INDArray testArr = test.get(varName);

            assertNotNull(testArr, label + ": missing output " + varName);
            assertArrayEquals(refArr.shape(), testArr.shape(),
                    label + ": shape mismatch for " + varName);

            if (refArr.dataType().isFPType()) {
                INDArray refF = refArr.dataType() == DataType.FLOAT ? refArr : refArr.castTo(DataType.FLOAT);
                INDArray testF = testArr.dataType() == DataType.FLOAT ? testArr : testArr.castTo(DataType.FLOAT);
                INDArray diff = org.nd4j.linalg.ops.transforms.Transforms.abs(refF.sub(testF));
                double maxDiff = diff.maxNumber().doubleValue();
                double tol = TOLERANCE.getDefaultAbsTol();

                if (refArr != refF && refArr.closeable()) refF.close();
                if (testArr != testF && testArr.closeable()) testF.close();
                diff.close();

                log.info("[{}] {} maxDiff={:.6e} (tol={:.2e})", label, varName, maxDiff, tol);
                assertTrue(maxDiff <= tol,
                        String.format("%s: %s maxDiff=%.6e exceeds tol=%.6e",
                                label, varName, maxDiff, tol));
            }
        }
    }
}
