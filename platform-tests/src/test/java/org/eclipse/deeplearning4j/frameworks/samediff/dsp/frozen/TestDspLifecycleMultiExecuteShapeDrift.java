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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.frozen;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Standalone, self-contained regression tests for DSP lifecycle stability across
 * multiple executions — specifically targeting the failure modes seen in
 * production models (bge-base-en-v1.5 slot 93 "stale ownership", qwen slot 211
 * "in-place-fused pointer replacement").
 *
 * <p>The goal is to exercise the DSP AUTO path end-to-end through
 * {@code SLOT_BY_SLOT → SHAPES_FROZEN → REPLAYING} with:</p>
 *
 * <ol>
 *   <li>A gather-based embedding lookup — the canonical op whose VALDEP
 *       classification was misclassified by the pre-fix {@code hasIntLongInputs}
 *       heuristic and which the new {@code NativeOps.getOpTraits("gather")}
 *       consult keeps classified as non-VALDEP.</li>
 *   <li>A fusion-eligible chain (biasAdd → relu) that triggers
 *       {@code FusionPass::applyFusions} to set {@code inPlaceFused=true} on
 *       the relu slot. Repeated execution must keep the aliased input buffer's
 *       pointer stable across executions — a rotation manifests as
 *       "in-place-fused pointer replacement".</li>
 *   <li>Same-shape repeated execution — shape stable, buffers must be reused
 *       from the slot cache via the {@code cached-reuse} / {@code view-op-reuse}
 *       paths in {@code NativeDynamicShapePlan_slotexec.cpp}. Any pointer
 *       rotation here is a regression of the lifecycle invariant "after freeze,
 *       slot buffers do not rotate when shapes are stable".</li>
 * </ol>
 *
 * <p>None of these tests depend on an external model. Failures indicate a
 * regression in the DSP allocator, FusionPass, or slot ownership bookkeeping.</p>
 *
 * <p>Run from platform-tests:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=TestDspLifecycleMultiExecuteShapeDrift \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       -Dnd4j.dsp.diagnostics=ALL -Dnd4j.dsp.diagnostics.level=full \
 *       2&gt;&amp;1 | tee /tmp/dsp-lifecycle-shape-drift.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Lifecycle — multi-execute shape-drift regression")
public class TestDspLifecycleMultiExecuteShapeDrift {

    private static final int VOCAB_SIZE = 64;
    private static final int HIDDEN = 32;
    private static final int OUT_DIM = 16;
    private static final int REPEATS = 8;

    private SameDiff sd;

    @AfterEach
    public void teardown() {
        Nd4j.getExecutioner().commit();
        if (sd != null) {
            try {
                sd.close();
            } catch (Exception ignored) {
            }
            sd = null;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 1. Gather + matmul, STABLE shape — regression for slot-93-style failure
    //    (bge embedding lookup then projection, same seq length each step)
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("gather(embed, indices) → matmul executes cleanly under repeated stable-shape calls")
    public void testGatherMatmulStableShapeRepeatedExecution() {
        sd = buildGatherMatmul();
        enableDsp(sd);

        int seqLen = 7;
        INDArray firstOut = null;
        for (int i = 0; i < REPEATS; i++) {
            Map<String, INDArray> ph = indicesInput(seqLen);
            final int step = i;
            Map<String, INDArray> outMap = assertDoesNotThrow(
                    () -> sd.output(ph, "out"),
                    "Step " + step + ": DSP execution must not throw a lifecycle error "
                            + "for stable-shape gather→matmul");
            INDArray out = outMap.get("out");
            assertNotNull(out, "Step " + i + ": output should not be null");
            assertEquals(2, out.rank(), "Step " + i + ": output should be rank-2 [seqLen, OUT_DIM]");
            assertEquals(seqLen, out.size(0));
            assertEquals(OUT_DIM, out.size(1));
            if (firstOut == null) firstOut = out.dup();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 2. Gather + matmul, DRIFTING shape — exercises the post-VALDEP-fix path
    //    where gather's output length legitimately varies between executions.
    //    This was the original symptom that fired `LIFECYCLE_ERROR:
    //    value-dependent output shape changed at slot N (gather) during
    //    SHAPES_FROZEN phase` before the trait-table consultation fix.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("gather(embed, indices) tolerates seq-length drift across executions")
    public void testGatherMatmulShapeDriftAcrossExecutions() {
        sd = buildGatherMatmul();
        enableDsp(sd);

        int[] seqLens = {4, 6, 4, 9, 6, 4};
        for (int i = 0; i < seqLens.length; i++) {
            final int seqLen = seqLens[i];
            final int step = i;
            Map<String, INDArray> ph = indicesInput(seqLen);
            Map<String, INDArray> outMap = assertDoesNotThrow(
                    () -> sd.output(ph, "out"),
                    "Step " + step + " (seqLen=" + seqLen + "): drifting-shape gather must not "
                            + "trigger LIFECYCLE_ERROR — its output shape is derivable from "
                            + "input shapes alone and OpTraitTable marks it non-VALDEP");
            INDArray out = outMap.get("out");
            assertNotNull(out);
            assertEquals(seqLen, out.size(0),
                    "Step " + i + ": output rows must track the indices length");
            assertEquals(OUT_DIM, out.size(1));
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // 3. biasAdd → relu, STABLE shape — fusion-eligible chain regression for
    //    slot-211-style failure. FusionPass should fold relu in-place over the
    //    biasAdd output. Once the slot is marked inPlaceFused, the output slot
    //    aliases the input buffer and repeated execution must keep that alias
    //    pointer identical — any rotation surfaces as an in-place-fused
    //    pointer-replacement lifecycle error at writeOutputSlot's frozen guard.
    // ═══════════════════════════════════════════════════════════════════════════

    @Test
    @DisplayName("biasAdd → relu fused chain stays pointer-stable across executions")
    public void testBiasAddReluFusedChainStable() {
        sd = buildBiasAddReluChain();
        enableDsp(sd);

        Map<String, INDArray> ph = matMulInput();
        INDArray firstOut = null;
        for (int i = 0; i < REPEATS; i++) {
            final int step = i;
            Map<String, INDArray> outMap = assertDoesNotThrow(
                    () -> sd.output(ph, "out"),
                    "Step " + step + ": in-place-fused biasAdd+relu chain must not trigger "
                            + "'in-place-fused pointer replacement' lifecycle error");
            INDArray out = outMap.get("out");
            assertNotNull(out);
            assertTrue(out.minNumber().doubleValue() >= 0.0,
                    "Step " + i + ": relu output must be non-negative");
            if (firstOut == null) firstOut = out.dup();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Graph builders
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Embedding lookup + linear projection:
     *   indices (INT32, dynamic) → gather(embed [VOCAB, HIDDEN]) → [L, HIDDEN]
     *                                                            → mmul(W) → [L, OUT]
     */
    private static SameDiff buildGatherMatmul() {
        SameDiff sd = SameDiff.create();
        SDVariable indices = sd.placeHolder("indices", DataType.INT32, -1);
        SDVariable embed   = sd.var("embed", Nd4j.randn(DataType.FLOAT, VOCAB_SIZE, HIDDEN).muli(0.05));
        SDVariable w       = sd.var("w",     Nd4j.randn(DataType.FLOAT, HIDDEN, OUT_DIM).muli(0.05));
        SDVariable gathered = sd.gather("gathered", embed, indices, 0);
        SDVariable out      = sd.mmul("out", gathered, w);
        sd.setOutputs("out");
        return sd;
    }

    /**
     * biasAdd → relu chain. FusionPass typically fuses activation after
     * biasAdd so the relu slot is stamped {@code inPlaceFused=true} with
     * {@code inPlaceFusedInputIdx} pointing at the biasAdd output.
     */
    private static SameDiff buildBiasAddReluChain() {
        SameDiff sd = SameDiff.create();
        SDVariable x    = sd.placeHolder("x", DataType.FLOAT, -1, HIDDEN);
        SDVariable w    = sd.var("w",    Nd4j.randn(DataType.FLOAT, HIDDEN, OUT_DIM).muli(0.05));
        SDVariable b    = sd.var("b",    Nd4j.zeros(DataType.FLOAT, OUT_DIM));
        SDVariable h    = sd.mmul("h", x, w);
        SDVariable hb   = sd.nn.biasAdd("hb", h, b, true);
        SDVariable out  = sd.nn.relu("out", hb, 0);
        sd.setOutputs("out");
        return sd;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Inputs + DSP configuration helpers
    // ═══════════════════════════════════════════════════════════════════════════

    private static Map<String, INDArray> indicesInput(int seqLen) {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        int[] idx = new int[seqLen];
        for (int i = 0; i < seqLen; i++) idx[i] = i % VOCAB_SIZE;
        ph.put("indices", Nd4j.createFromArray(idx).castTo(DataType.INT32));
        return ph;
    }

    private static Map<String, INDArray> matMulInput() {
        Map<String, INDArray> ph = new LinkedHashMap<>();
        ph.put("x", Nd4j.randn(DataType.FLOAT, 3, HIDDEN));
        return ph;
    }

    private static void enableDsp(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        sd.setGraphExecutionMode(GraphExecutionMode.AUTO);
    }
}
