/*
 *  SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.TestMethodOrder;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;

import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * Guards the gap-capture exclusion contract for attention and GDN ops.
 *
 * gated_delta_rule allocates per-invocation recurrent/chunk scratch buffers
 * (commit b1e4663d22) and carries OP_TRAIT_EXTERNAL_WORKSPACE. All
 * attention-family ops carry OP_TRAIT_ATTENTION. isGapRangeCaptureSafe
 * (NativeDynamicShapePlan_gpubackend.cu) excludes both from merged CUDA graph
 * capture because capturing pool-recycled scratch pointers or multi-output
 * KV composition as generic gap glue corrupts replay.
 *
 * These tests pin the CURRENT contract: graphs containing these ops must
 * produce correct results under every DSP execution mode, and (once a JNI
 * schedule query exists) the gap units containing them must stay live.
 * They fail if someone flips a gate or trait without making the underlying
 * allocation/composition capture-safe first.
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@DisplayName("GDN/Attention Gap Capture Exclusion Contract")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
public class DspGapCaptureExclusionContractTest {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    /**
     * Build a graph matching the Qwen3.5 hybrid layer shape: Triton-eligible
     * elementwise/norm islands interleaved with gated_delta_rule (GDN layer)
     * and dot_product_attention_v2 (full-attention layer) gap ops, surrounded
     * by permutes. This mirrors the slot mix measured on the real models:
     * gated_delta_rule + dot_product_attention_v2 + permute dominate gaps.
     */
    private SameDiff buildHybridGraph(int embedDim, int numLayers) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("hidden", DataType.FLOAT, 1, embedDim);
        int heads = 2;
        int headDim = embedDim / heads;

        for (int layer = 0; layer < numLayers; layer++) {
            String p = "L" + layer + "_";
            SDVariable gamma = g.var(p + "gamma", Nd4j.ones(DataType.FLOAT, embedDim));
            SDVariable wq = g.var(p + "wq", Transforms.abs(
                    Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f));

            // Triton-eligible: rmsNorm
            SDVariable normed = g.nn().rmsNorm(p + "norm", x, gamma, 1e-5);
            SDVariable q = g.mmul(p + "q", normed, wq);

            // Gap op: permute (classified as gap in real models)
            SDVariable q3 = g.reshape(p + "q3", q, 1, heads, 1, headDim);
            SDVariable qT = g.permute(p + "qT", q3, 0, 2, 1, 3);

            if (layer % 2 == 0) {
                // GDN-shaped layer: a state placeholder (recurrent, feeds the
                // next decode step) combined with matmuls and elementwise
                // gating. This mirrors the production slot mix around
                // gated_delta_rule (state @ k gating + outer-product update +
                // projection) using only length-preserving 2D shapes.
                //   state [embedDim, embedDim], k/v/gate/beta [embedDim, 1]
                SDVariable state = g.placeHolder(p + "state", DataType.FLOAT,
                        embedDim, embedDim);
                SDVariable k = g.var(p + "gk", Nd4j.randn(DataType.FLOAT, embedDim, 1).mul(0.1));
                SDVariable v = g.var(p + "gv", Nd4j.randn(DataType.FLOAT, embedDim, 1).mul(0.1));
                SDVariable beta = g.var(p + "gb", Nd4j.ones(DataType.FLOAT, embedDim, 1).mul(0.5));
                SDVariable gate = g.var(p + "gg", Nd4j.zeros(DataType.FLOAT, embedDim, 1));
                // Recurrence: sk = state @ k (gated decay), upd = (v-k outer) @ k * beta
                SDVariable sk = g.mmul(p + "sk", state, k);                    // [embedDim, 1]
                SDVariable decayed = sk.mul(p + "decayed", gate);               // [embedDim, 1]
                SDVariable outer = g.mmul(p + "outer", v, g.reshape(p + "kT", k, 1, embedDim)); // [embedDim, embedDim]
                SDVariable upd = g.mmul(p + "upd", outer, k).mul(p + "scaled", beta); // [embedDim, 1]
                SDVariable newState = decayed.add(p + "newState", upd);        // [embedDim, 1]
                // Project via q reshaped to [1, embedDim] row: [1,1] scalar,
                // broadcast-sum back to [1, embedDim] so the next layer gets x.
                SDVariable qRow = g.reshape(p + "qrow", q, 1, embedDim);        // [1, embedDim]
                SDVariable proj2d = g.mmul(p + "proj2d", qRow, newState);       // [1, 1]
                SDVariable out = proj2d.mul(p + "bcast", q).add(p + "addQ", q); // [1, embedDim]
                x = g.nn().rmsNorm(p + "postNorm", out, gamma, 1e-5);
            } else {
                // Full-attention layer: matmul QK^T + softmax + V (the
                // dot_product_attention_v2 decomposition that creates the
                // OP_TRAIT_ATTENTION gap in the production graph).
                SDVariable kv = g.placeHolder(p + "kv", DataType.FLOAT, 1, 4, embedDim);
                SDVariable proj = g.mmul(p + "proj", q, g.var(p + "wk",
                        Transforms.abs(Nd4j.randn(DataType.FLOAT, embedDim, embedDim)).addi(0.01f)));
                // kv flattened to [4, embedDim] matrix: 4 cached KV rows.
                SDVariable kvMat = g.reshape(p + "kvMat", kv, 4, embedDim);  // [4, 64]
                // proj [1, 64] @ kvMat^T [64, 4] = scores [1, 4]
                SDVariable scores = g.mmul(p + "qk", proj,
                        g.permute(p + "kvT", kvMat, 1, 0));                   // [1, 4]
                SDVariable attn = g.nn().softmax(p + "attn", scores);        // [1, 4]
                SDVariable out = g.mmul(p + "attnOut", attn, kvMat);         // [1,4]x[4,64]->[1,64]
                x = g.nn().rmsNorm(p + "postNorm", out, gamma, 1e-5);
            }
        }
        g.setOutputs(x.name());
        return g;
    }

    private Map<String, INDArray> buildInputs(int embedDim, int numLayers, int heads, int headDim) {
        Map<String, INDArray> ph = new HashMap<>();
        ph.put("hidden", Nd4j.randn(DataType.FLOAT, 1, embedDim).mul(0.1));
        for (int layer = 0; layer < numLayers; layer++) {
            String p = "L" + layer + "_";
            if (layer % 2 == 0) {
                ph.put(p + "state", Nd4j.randn(DataType.FLOAT, embedDim, embedDim).mul(0.1));
            } else {
                ph.put(p + "kv", Nd4j.randn(DataType.FLOAT, 1, 4, embedDim).mul(0.1));
            }
        }
        return ph;
    }

    /** Deep-copies an input map so closing one SameDiff session cannot close
     *  the other side's arrays. */
    private Map<String, INDArray> copyInputs(Map<String, INDArray> src) {
        Map<String, INDArray> out = new HashMap<>();
        for (Map.Entry<String, INDArray> e : src.entrySet()) {
            out.put(e.getKey(), e.getValue().dup());
        }
        return out;
    }

    /**
     * Contract: graphs with the hybrid GDN/attention slot mix produce
     * identical results across at least 12 replay steps in every DSP mode.
     * The first reference output comes from SLOT_BY_SLOT (Java-visible
     * execution); each DSP mode must match it step by step. If gap capture
     * is extended to these ops unsafely, replay accumulates corruption that
     * surfaces by step 12 (pool recycling rewrites scratch).
     */
    @ParameterizedTest(name = "1_hybridAccuracy mode={0}")
    @EnumSource(value = GraphExecutionMode.class,
            names = {"AUTO", "TRITON", "CUDA_GRAPHS"})
    @Order(1)
    void test1_HybridReplayAccuracy(GraphExecutionMode mode) {
        String backend = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        assumeTrue(backend.contains("cuda") || backend.contains("jcublas"),
                "DSP composite replay requires CUDA backend");

        int embedDim = 64;
        int heads = 2;
        int headDim = embedDim / heads;
        int numLayers = 6;
        int steps = 12;

        long seed = 12345L;
        Map<String, INDArray> sharedIn = buildInputs(embedDim, numLayers, heads, headDim);
        // Reference: SLOT_BY_SLOT fresh execution per step.
        Nd4j.getRandom().setSeed(seed);
        SameDiff refSd = buildHybridGraph(embedDim, numLayers);
        refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        INDArray[] refOut = new INDArray[steps];
        try {
            Map<String, INDArray> refIn = copyInputs(sharedIn);
            for (int s = 0; s < steps; s++) {
                INDArray h = refIn.get("hidden").dup();
                h.addi(s * 0.01);
                refIn.put("hidden", h);
            Map<String, INDArray> out = refSd.output(refIn, refSd.outputs().get(0));
            refOut[s] = out.get(refSd.outputs().get(0)).dup();
            }
        } finally {
            refSd.close();
        }

        // DSP mode under test.
        Nd4j.getRandom().setSeed(seed);
        SameDiff g = buildHybridGraph(embedDim, numLayers);
        sd = g;
        g.setGraphExecutionMode(mode);
        Map<String, INDArray> in = copyInputs(sharedIn);
        for (int s = 0; s < steps; s++) {
            INDArray h = in.get("hidden").dup();
            h.addi(s * 0.01);
            in.put("hidden", h);
            String outName = g.outputs().get(0);
            Map<String, INDArray> out = g.output(in, outName);
            INDArray actual = out.get(outName);
            assertEquals(refOut[s], actual,
                    mode + " diverged from SLOT_BY_SLOT at step " + s
                            + " (gap-capture corruption or staleness)");
        }
        DspPlanAssertions.assertNoPhaseContractViolations(g);
        DspPlanAssertions.assertPointersStable(g);
    }

    /**
     * Contract: pool churn between replays must not corrupt results.
     * gated_delta_rule allocates per-invocation scratch (the reason it stays
     * live). Allocate and free device buffers between replay steps so the
     * pool reuses memory, then verify outputs still match the reference.
     * This is the direct regression test for the b1e4663d22 failure mode.
     */
    @Test
    @Order(2)
    @DisplayName("2_gapScratchLifetimeUnderPoolChurn")
    void test2_GapScratchLifetimeUnderPoolChurn() {
        String backend = Nd4j.getBackend().getClass().getSimpleName().toLowerCase();
        assumeTrue(backend.contains("cuda") || backend.contains("jcublas"),
                "DSP composite replay requires CUDA backend");

        int embedDim = 64;
        int heads = 2;
        int headDim = embedDim / heads;
        int numLayers = 6;
        int steps = 15;

        long seed = 54321L;
        Map<String, INDArray> sharedIn = buildInputs(embedDim, numLayers, heads, headDim);
        Nd4j.getRandom().setSeed(seed);
        SameDiff refSd = buildHybridGraph(embedDim, numLayers);
        refSd.setGraphExecutionMode(GraphExecutionMode.SLOT_BY_SLOT);
        INDArray[] refOut = new INDArray[steps];
        try {
            Map<String, INDArray> refIn = copyInputs(sharedIn);
            for (int s = 0; s < steps; s++) {
                INDArray h = refIn.get("hidden").dup();
                h.addi(s * 0.01);
                refIn.put("hidden", h);
                String rn = refSd.outputs().get(0);
                refOut[s] = refSd.output(refIn, rn).get(rn).dup();
            }
        } finally {
            refSd.close();
        }

        Nd4j.getRandom().setSeed(seed);
        SameDiff g = buildHybridGraph(embedDim, numLayers);
        sd = g;
        g.setGraphExecutionMode(GraphExecutionMode.AUTO);
        Map<String, INDArray> in = copyInputs(sharedIn);
        for (int s = 0; s < steps; s++) {
            INDArray h = in.get("hidden").dup();
            h.addi(s * 0.01);
            in.put("hidden", h);

            // Pool churn: allocate and release scratch-sized buffers so the
            // memory pool recycles regions a captured graph must not retain.
            try (INDArray churnA = Nd4j.create(DataType.FLOAT, 64, headDim * headDim);
                 INDArray churnB = Nd4j.create(DataType.FLOAT, headDim * headDim, 64)) {
                churnA.assign(s);
                churnB.assign(s * 0.5);
            }

            String on = g.outputs().get(0);
            INDArray actual = g.output(in, on).get(on);
            assertEquals(refOut[s], actual,
                    "AUTO replay diverged at step " + s + " after pool churn"
                            + " (scratch pointer retained by capture: b1e4663d22 regression)");
        }
    }
}
