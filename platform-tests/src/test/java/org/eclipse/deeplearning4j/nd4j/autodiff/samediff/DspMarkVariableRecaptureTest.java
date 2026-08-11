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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.*;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.execution.DspPlanAssertions;
import org.nd4j.autodiff.samediff.execution.GraphExecutionMode;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.LinkedHashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Focused reproduction of the native-decode "stuck token" hypothesis at the
 * SameDiff/DSP level — without needing the full SmolDocling model.
 *
 * <h3>Hypothesis under test</h3>
 * The autoregressive_decode C++ op warms+freezes the decoder plan, then calls
 * {@code markExternalInputVariable(...)} on its per-step inputs. On a plan that
 * is already frozen, that calls {@code invalidateSegmentCaptures} which resets
 * the affected segments to BUILDING:WARMUP. The GPU_COMPILER capture path then
 * reaches its seal guard ({@code if (!segPhase.isSealed())}) — which is TRUE for
 * a WARMUP segment — and calls {@code markCaptured}, sealing a segment that was
 * never actually captured. Replay then reads stale (capture-time) data and the
 * decode freezes on one token (native matches steps 0-4, then repeats step 4).
 *
 * <h3>What this test does</h3>
 * Builds a small mixed graph (cuBLAS matmul gaps + Triton-eligible add/rmsNorm
 * islands → composite GPU_COMPILER capture, the same path as the decoder),
 * warms it to steady-state replay, then triggers the exact decode invalidation
 * via {@link org.nd4j.autodiff.samediff.execution.DspHandle#markVariable(String)}.
 * After that it keeps executing with changing inputs and asserts the replay
 * output still tracks the input (matches a slot-by-slot reference) and that the
 * capture state machine recorded no capture failures or phase-contract
 * violations.
 *
 * Pre-fix (markCaptured-from-WARMUP): the replay is stale → output diverges from
 * the SBS reference and/or a phase-contract violation is recorded → test fails.
 * Post-fix: the WARMUP segment re-captures correctly → output tracks input.
 */
@Slf4j
@Tag(TagNames.FULL_CI)
@TestInstance(TestInstance.Lifecycle.PER_METHOD)
public class DspMarkVariableRecaptureTest {

    private SameDiff sd;

    @AfterEach
    void cleanup() {
        if (sd != null) {
            sd.close();
            sd = null;
        }
    }

    /**
     * out = matmul( rmsNorm( matmul(x,W)+b , gamma ) ..., wOut )
     * matmul = cuBLAS gap op; add + rmsNorm = Triton islands. The mix forces the
     * composite (GPU_COMPILER) capture path whose seal guard is under test.
     * Weights are graph variables (stable device addresses); only "x" changes.
     */
    private SameDiff buildMixedGraph(int dim, int layers) {
        SameDiff g = SameDiff.create();
        SDVariable x = g.placeHolder("x", DataType.FLOAT, 1, dim);
        SDVariable h = x;
        for (int l = 0; l < layers; l++) {
            String p = "l" + l + "_";
            SDVariable w = g.var(p + "w", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.05f));
            SDVariable b = g.var(p + "b", Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.05f));
            SDVariable gamma = g.var(p + "g", Nd4j.ones(DataType.FLOAT, dim));
            SDVariable mm = g.mmul(p + "mm", h, w);            // cuBLAS gap
            SDVariable add = mm.add(p + "add", b);             // Triton island
            h = g.nn().rmsNorm(p + "norm", add, gamma, 1e-5);  // Triton island
        }
        g.mmul("out", h, g.var("wOut", Nd4j.randn(DataType.FLOAT, dim, dim).muli(0.05f)));
        return g;
    }

    private void configureMode(SameDiff g, GraphExecutionMode mode) {
        g.getSessions().clear();
        g.setGraphExecutionMode(mode);
        g.setDspAutoCompileEnabled(true);
        g.setDspNativeAutoCompileEnabled(true);
    }

    // ════════════════════════════════════════════════════════════════════════
    // TEST 1 (baseline): captured replay must track a changing input — no stale
    // replay even before any markVariable invalidation.
    // ════════════════════════════════════════════════════════════════════════
    @Test
    void test1_capturedReplayTracksChangingInput() {
        final int dim = 64, layers = 6, steps = 16;
        Nd4j.getRandom().setSeed(7);
        SameDiff g = buildMixedGraph(dim, layers);
        sd = g;

        // Precompute slot-by-slot references for fixed inputs.
        INDArray[] inputs = new INDArray[steps];
        INDArray[] refs = new INDArray[steps];
        configureMode(g, GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        for (int i = 0; i < steps; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1f);
            ph.put("x", inputs[i]);
            refs[i] = g.output(ph, "out").get("out").dup();
        }

        // Run the SAME inputs through DSP (capture/replay) and compare.
        configureMode(g, GraphExecutionMode.AUTO);
        for (int i = 0; i < steps; i++) {
            ph.put("x", inputs[i]);
            INDArray dspOut = g.output(ph, "out").get("out");
            assertFalse(dspOut.isNaN().any(), "step " + i + ": NaN in DSP output");
            double maxDiff = Transforms.abs(dspOut.sub(refs[i])).maxNumber().doubleValue();
            assertTrue(maxDiff < 1e-3,
                    "step " + i + ": DSP replay diverged from slot-by-slot reference "
                            + "(stale/stuck replay) maxDiff=" + maxDiff);
        }

        int replays = DspPlanAssertions.getTotalGraphReplays(g);
        log.info("test1 tracks-input: totalReplays={} planPhase={}", replays,
                DspPlanAssertions.getPlanPhase(g));
        assertTrue(replays > 0, "expected graph replays > 0, got " + replays);
        DspPlanAssertions.assertNoCaptureFailures(g, "tracks-changing-input");
        DspPlanAssertions.assertNoPhaseContractViolations(g, "tracks-changing-input");
    }

    // ════════════════════════════════════════════════════════════════════════
    // TEST 2 (the decode bug): after the plan is frozen+replaying, marking an
    // input variable (what autoregressive_decode does) invalidates segments to
    // WARMUP. The plan must RE-CAPTURE correctly — not seal an uncaptured
    // segment (markCaptured-from-WARMUP) and replay stale data.
    // ════════════════════════════════════════════════════════════════════════
    @Test
    void test2_replaySurvivesMarkVariableInvalidation() {
        final int dim = 64, layers = 6, steps = 16;
        Nd4j.getRandom().setSeed(11);
        SameDiff g = buildMixedGraph(dim, layers);
        sd = g;

        // Slot-by-slot references for fixed inputs.
        INDArray[] inputs = new INDArray[steps];
        INDArray[] refs = new INDArray[steps];
        configureMode(g, GraphExecutionMode.SLOT_BY_SLOT);
        Map<String, INDArray> ph = new LinkedHashMap<>();
        for (int i = 0; i < steps; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1f);
            ph.put("x", inputs[i]);
            refs[i] = g.output(ph, "out").get("out").dup();
        }

        // Warm DSP to steady-state replay.
        configureMode(g, GraphExecutionMode.AUTO);
        for (int i = 0; i < 8; i++) {
            ph.put("x", inputs[i % steps]);
            g.output(ph, "out");
        }
        assertTrue(DspPlanAssertions.getTotalGraphReplays(g) > 0,
                "expected replay before markVariable");

        // ── The decode trigger: mark the per-step input variable on a frozen
        //    plan → invalidateSegmentCaptures → segments reset to BUILDING:WARMUP.
        g.dsp().markVariable("x");
        log.info("test2: called markVariable('x') — segments invalidated to WARMUP");

        // Continue with changing inputs; replay must re-capture correctly.
        for (int i = 0; i < steps; i++) {
            ph.put("x", inputs[i]);
            INDArray dspOut = g.output(ph, "out").get("out");
            assertFalse(dspOut.isNaN().any(), "post-markVariable step " + i + ": NaN");
            double maxDiff = Transforms.abs(dspOut.sub(refs[i])).maxNumber().doubleValue();
            assertTrue(maxDiff < 1e-3,
                    "post-markVariable step " + i + ": DSP replay diverged from slot-by-slot "
                            + "reference — markCaptured-from-WARMUP sealed an uncaptured segment "
                            + "→ stale replay (the stuck-token bug). maxDiff=" + maxDiff);
        }

        log.info("test2 post-markVariable: totalReplays={} planPhase={}",
                DspPlanAssertions.getTotalGraphReplays(g), DspPlanAssertions.getPlanPhase(g));
        // The capture state machine must not have sealed an uncaptured segment.
        DspPlanAssertions.assertNoCaptureFailures(g, "after markVariable re-capture");
        DspPlanAssertions.assertNoPhaseContractViolations(g, "after markVariable re-capture");
    }

    /**
     * Reproduces the native autoregressive handoff more closely than test2:
     * one execution auto-seals the plan, then the same address-stable input buffer is
     * marked variable and overwritten in place on every step. The native decode loop
     * does exactly this for input_ids, positions, masks, and recurrent state.
     */
    @Test
    void test3_fixedAddressDeviceWrittenInputTracksImmediatelyAfterAutoSeal() {
        final int dim = 64, layers = 6, steps = 16;
        Nd4j.getRandom().setSeed(19);
        SameDiff g = buildMixedGraph(dim, layers);
        sd = g;

        INDArray[] inputs = new INDArray[steps];
        INDArray[] refs = new INDArray[steps];
        Map<String, INDArray> ph = new LinkedHashMap<>();
        configureMode(g, GraphExecutionMode.SLOT_BY_SLOT);
        for (int i = 0; i < steps; i++) {
            inputs[i] = Nd4j.randn(DataType.FLOAT, 1, dim).muli(0.1f);
            ph.put("x", inputs[i]);
            refs[i] = g.output(ph, "out").get("out").dup();
        }

        configureMode(g, GraphExecutionMode.AUTO);
        INDArray fixedInput = Nd4j.create(DataType.FLOAT, 1, dim);
        Object fixedDataBuffer = fixedInput.data();
        ph.put("x", fixedInput);

        fixedInput.assign(inputs[0]);
        Nd4j.getExecutioner().commit();
        INDArray first = g.output(ph, "out").get("out");
        double firstDiff = Transforms.abs(first.sub(refs[0])).maxNumber().doubleValue();
        assertTrue(firstDiff < 1e-3, "first auto-seal execution diverged: maxDiff=" + firstDiff);
        assertEquals(1, DspPlanAssertions.getPlanPhase(g),
                "one AUTO execution must leave this regression at SHAPES_FROZEN");

        g.dsp().markVariable("x");
        log.info("test3: marked fixed-address input variable immediately after auto-seal");

        for (int i = 1; i < steps; i++) {
            fixedInput.assign(inputs[i]);
            Nd4j.getExecutioner().commit();
            assertSame(fixedDataBuffer, fixedInput.data(),
                    "fixed decode input DataBuffer identity changed at step " + i);

            INDArray dspOut = g.output(ph, "out").get("out");
            assertFalse(dspOut.isNaN().any(), "fixed-address step " + i + ": NaN");
            double maxDiff = Transforms.abs(dspOut.sub(refs[i])).maxNumber().doubleValue();
            assertTrue(maxDiff < 1e-3,
                    "fixed-address device-written input was stale at step " + i
                            + " immediately after auto-seal: maxDiff=" + maxDiff);
        }

        assertTrue(DspPlanAssertions.getTotalGraphReplays(g) > 0,
                "fixed-address plan never reached replay after auto-seal invalidation");
        DspPlanAssertions.assertNoCaptureFailures(g, "fixed-address post-auto-seal recapture");
        DspPlanAssertions.assertNoPhaseContractViolations(g, "fixed-address post-auto-seal recapture");
    }
}
