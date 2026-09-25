/*
 * SPDX-License-Identifier: Apache-2.0
 */
package org.eclipse.deeplearning4j.llm.generation;

import org.bytedeco.javacpp.Pointer;
import org.junit.jupiter.api.Test;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor;
import org.nd4j.autodiff.samediff.execution.DynamicShapePlanExecutor.NativeExecutionBinding;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.AutoregressiveDecode;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1d;
import org.nd4j.linalg.api.ops.impl.transforms.custom.CausalConv1dWithPrefix;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRule;
import org.nd4j.linalg.api.ops.impl.transforms.custom.GatedDeltaRuleWithPrefix;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.OpaqueContext;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Model-free native MTP transaction: physical W5 verify -> restored accepted-prefix
 * rerun, with real causal_conv1d and gated_delta_rule state feedback, target KV,
 * target hidden carry and predictor KV repair. EOS terminates the FIRST commit;
 * a budget of five permits all four proposals even when only one is accepted.
 *
 * <p>All arithmetic is FLOAT32. One channel/head is intentional: convolution
 * weights [1/4, 1/2, 1], q=k=1, beta=1/2, gate=0 give the sequential recurrence
 * s[t] = (s[t-1] + conv[t])/2, NOT a summed-token proxy. A Java scalar loop is
 * the independent numerical oracle. Class-dependent hidden contributions make
 * every active-row logit observable, while large fixed biases force acceptance.
 * A zero attention query gives an independently calculable mean of visible KV
 * values; hidden = recurrent output + attention, so KV visibility affects logits.
 *
 * <p>This is not a Qwen reproduction or proof of CUDA graph capture. It does not
 * exercise scalar-plan handoff, quantized weights, or disagreement-triggered
 * double rollback within ONE transaction. W4/W2/W1 are separate transactions
 * on the same plans/buffers, each restored from the same saved input snapshot.
 * Rejected KV rows are scratch (attention physically writes W5); assert every
 * retained row and the terminal mask, not a promise that scratch is erased.
 */
public class TestMtpPrefixRerunState {
    private static final int WIDTH = 5;
    private static final int CACHE = 16;
    private static final int VOCAB = 5;
    private static final int START = 1;
    private static final int BASE = 1;
    private static final int DRAFT = 4;
    private static final int EOS = 0;
    // Ordinary (non-terminal) correction token for the nonterminal continuation fixture.
    private static final int CORRECTION = 2;
    private static final float SENTINEL = -7f;
    // FP32 attention divides by up to five, then predictor repair scales carry by eight.
    private static final double EPS = 5e-5;

    @Test
    public void testAcceptedPrefixesRepeatFromSameSnapshot() {
        try (Plan target = target(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);
            try (Snapshot targetBefore = new Snapshot(target);
                 Snapshot predictorBefore = new Snapshot(predictor)) {
                // Same handles, same allocation addresses, including across shrinking prefixes.
                long handle = target.executor.getNativePlanHandle().address();
                for (int consumed : new int[]{4, 4, 2, 2, 1, 1}) {
                    targetBefore.restore(target);
                    predictorBefore.restore(predictor);
                    target.input("accepted").assign(consumed - 1);
                    target.binding.beginNativeUse();
                    try {
                        predictor.binding.beginNativeUse();
                        try {
                            runTransaction(target, predictor, consumed, false, true, WIDTH, START);
                        } finally {
                            Nd4j.getExecutioner().commit();
                            predictor.binding.completeNativeUse();
                        }
                    } finally {
                        target.binding.completeNativeUse();
                    }
                    assertEquals(handle, target.executor.getNativePlanHandle().address(),
                            "must retain the physical-W5 plan across transactions");
                }
            }
        }
    }

    /**
     * Packets 07/09: forced partial acceptance through the SELECT fast path.
     * Identical fixture to the reference-rerun oracle, but the target uses the
     * COMPANION capture ops and the decode attaches prefix-select metadata. The
     * controller must commit checkpoint[consumed-1] directly.
     *
     * <p>THE SELECT PROOF IS COUNTER-BASED, NOT NUMERICS-ONLY: legacy recovery
     * produces correct results too, so a numerics-only pass proves nothing. The
     * current invocation's MTP_P0_CUDA summary is read from the native DSP
     * diagnostics ring (DspDiagnostics.getJsonReport) for exactly this one call:
     * targetVerify=1, reruns=0, shortened=0, checkpointSelectCommits=1,
     * checkpointSelectFallbacks=0. Exactly one SPEC_STATE_SELECT event, zero
     * SPEC_STATE_RERUN, zero SELECT_INELIGIBLE. A missing or ambiguous summary is
     * a FAILURE, never a pass. Requires -Dnd4j.mtp.multiRowCommit=1.</p>
     */
    @Test
    public void testForcedSelectPartialAcceptanceCommitsCheckpoint() {
        // Explicit precondition: this is a MULTI-ROW commit fixture (consumed=2).
        // Without numeric 1 the native commitCap is 1 and every expectation below
        // fails for policy reasons, not correctness.
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This SELECT proof requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");

        try (Plan target = selectTarget(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);
            try (Snapshot targetBefore = new Snapshot(target);
                 Snapshot predictorBefore = new Snapshot(predictor)) {
                // ONE transaction: consumed=2 (accepted=1 draft + EOS bonus).
                // Output budget stays FIVE so all four proposals execute; EOS ends
                // the commit after two consumed inputs.
                final int consumed = 2;
                targetBefore.restore(target);
                predictorBefore.restore(predictor);
                target.input("accepted").assign(consumed - 1);

                // ── Observation interval: save config, configure, clear, invoke ONCE ──
                NativeOps nativeOps = Nd4j.getNativeOps();
                int savedMask = nativeOps.dspDiagGetEnabledMask();
                int savedLevel = nativeOps.dspDiagGetLevel();
                String report;
                target.binding.beginNativeUse();
                try {
                    predictor.binding.beginNativeUse();
                    try {
                        DspDiagnostics.initialize();
                        DspDiagnostics.setCategories(DspDiagnostics.KV_CACHE);
                        DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
                        DspDiagnostics.clear();
                        runTransaction(target, predictor, consumed, true, true, WIDTH, START);
                        // Read BEFORE any other native invocation or teardown.
                        report = DspDiagnostics.getJsonReport();
                    } finally {
                        Nd4j.getExecutioner().commit();
                        predictor.binding.completeNativeUse();
                    }
                } finally {
                    target.binding.completeNativeUse();
                    // Restore prior diagnostic configuration even on assertion path.
                    nativeOps.dspDiagSetCategories(savedMask);
                    nativeOps.dspDiagSetLevel(savedLevel);
                }

                // ── Assert the current invocation's real counters ──
                SelectSummary summary = SelectSummary.parse(report);
                // Expected bytes derived from the fixture's actual state tensors:
                // GDN [1,1,1,1] FLOAT = 4 bytes + conv [1,1,2] FLOAT = 8 bytes.
                summary.assertSelectContract(consumed,
                        target.input("gdn").length() * 4L
                                + target.input("conv").length() * 4L,
                        2);
            }
        }
    }

    /**
     * Diagnostics-off numerical counterpart of the forced-SELECT proof: the SAME
     * fixture and invocation, but no diagnostic category is enabled for it and its
     * assertions never consult a diagnostic report. Passing it under
     * command-line diagnostics=none therefore proves the production path is
     * numerically correct without observability instrumentation. Diagnostic
     * configuration is saved and restored through the native mask + wrapper cache
     * so the two never disagree.
     */
    @Test
    public void testForcedSelectPartialAcceptanceNumericsWithoutDiagnostics() {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This SELECT proof requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");

        try (Plan target = selectTarget(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);
            final int consumed = 2;
            target.input("accepted").assign(consumed - 1);

            NativeOps nativeOps = Nd4j.getNativeOps();
            int savedMask = nativeOps.dspDiagGetEnabledMask();
            int savedLevel = nativeOps.dspDiagGetLevel();
            try {
                // Force the native mask to NONE through the wrapper (refreshes the
                // Java cached mask together with the native mask).
                DspDiagnostics.enableCategories(DspDiagnostics.NONE);
                assertFalse(DspDiagnostics.isEnabled(DspDiagnostics.KV_CACHE),
                        "precondition: no diagnostic category may be enabled for this invocation");
                target.binding.beginNativeUse();
                try {
                    predictor.binding.beginNativeUse();
                    try {
                        runTransaction(target, predictor, consumed, true, true, WIDTH, START);
                    } finally {
                        Nd4j.getExecutioner().commit();
                        predictor.binding.completeNativeUse();
                    }
                } finally {
                    target.binding.completeNativeUse();
                }
            } finally {
                // Restore the native mask through the wrapper so the cached Java
                // mask and the native mask stay consistent.
                DspDiagnostics.enableCategories(savedMask);
                nativeOps.dspDiagSetLevel(savedLevel);
            }
        }
    }

    /**
     * Continuation from the selected state. Transaction 1 forces the SELECT commit
     * (checkpoint[1] -> GDN 2.890625, conv [1,4]); transaction 2 then runs on the
     * SAME plans/buffers WITHOUT restoring the snapshot. If the committed state were
     * stale (still the initial 2.0), transaction 2's recurrence diverges immediately
     * (it would end at GDN 1.841796875 instead of 4.28515625).
     *
     * <p>Transaction 2 asserts only the PURE RECURRENCE state (GDN/conv) - the
     * quantity SELECT actually commits. The attention-derived hidden/logit values
     * depend on a visible KV set that spans BOTH transactions, which the
     * single-transaction oracle does not model, so they are deliberately not
     * asserted here (transaction 1 still asserts the full oracle).
     */
    @Test
    public void testContinuationFromSelectedState() {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This continuation proof requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");
        try (Plan target = selectTarget(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);

            // Transaction 1: forced SELECT, consumed=2, from the reset initial state;
            // full independent oracle applies (single transaction, from reset).
            target.input("accepted").assign(1);
            target.binding.beginNativeUse();
            try {
                predictor.binding.beginNativeUse();
                try {
                    runTransaction(target, predictor, 2, true, true, WIDTH, START);
                } finally {
                    Nd4j.getExecutioner().commit();
                    predictor.binding.completeNativeUse();
                }
            } finally {
                target.binding.completeNativeUse();
            }
            assertEquals(2.890625, target.input("gdn").getDouble(0), EPS,
                    "transaction 1 must leave the SELECT-committed GDN state, not the initial 2.0");

            // Pure-recurrence continuation of transaction 2 from (2.890625, [1,4]):
            //   row0 token=1: conv=0.25*1+0.5*4+1=3.25; state=(2.890625+3.25)/2=3.0703125
            //   row1 token=4: conv=0.25*4+0.5*1+4=5.5;  state=(3.0703125+5.5)/2=4.28515625
            // A rolled-back (stale initial 2.0) state would end at 1.841796875.

            // Transaction 2: a NEW non-EOS pending token, same forced acceptance
            // (consumed=2), from the CONTINUED state and advanced positions.
            target.input("ids").assign(BASE);
            target.input("accepted").assign(1);

            NativeOps nativeOps = Nd4j.getNativeOps();
            int savedMask = nativeOps.dspDiagGetEnabledMask();
            int savedLevel = nativeOps.dspDiagGetLevel();
            String report;
            target.binding.beginNativeUse();
            try {
                predictor.binding.beginNativeUse();
                try {
                    DspDiagnostics.initialize();
                    DspDiagnostics.setCategories(DspDiagnostics.KV_CACHE);
                    DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
                    DspDiagnostics.clear();
                    runTransaction(target, predictor, 2, true, false, WIDTH, START);
                    report = DspDiagnostics.getJsonReport();
                } finally {
                    Nd4j.getExecutioner().commit();
                    predictor.binding.completeNativeUse();
                }
            } finally {
                target.binding.completeNativeUse();
                nativeOps.dspDiagSetCategories(savedMask);
                nativeOps.dspDiagSetLevel(savedLevel);
            }

            // The committed state CONTINUED: the recurrence advanced from
            // 2.890625, not from the initial 2.0.
            assertEquals(4.28515625, target.input("gdn").getDouble(0), EPS,
                    "transaction 2 must advance the SELECT-committed GDN state (continued, not rolled back)");
            assertEquals(1.0, target.input("conv").getDouble(0), EPS, "transaction 2 conv older");
            assertEquals(4.0, target.input("conv").getDouble(1), EPS, "transaction 2 conv newest");

            // Transaction 2 committed via SELECT again, on the continued state.
            SelectSummary summary = SelectSummary.parse(report);
            summary.assertSelectContract(2,
                    target.input("gdn").length() * 4L + target.input("conv").length() * 4L,
                    2);
        }
    }

    /**
     * PATCH B: REAL nonterminal continuation. Two budget-limited native
     * invocations on the same live bindings, no snapshot restore, no position
     * reset, no re-prefill: the second call resumes from the actual advanced
     * position and the actual final emitted (non-EOS) pending token left by the
     * first.
     *
     * <p>Two independent chains start from identical fresh snapshots:
     * SELECT call 1 -> SELECT call 2 vs OFF call 1 -> OFF call 2. The chains
     * must produce identical retained recurrent state, retained KV, positions,
     * masks and pending pair. The SELECT chain must actually select on its
     * eligible partial steps (commits &gt; 0, zero fallbacks), which the OFF
     * chain does not have.
     *
     * <p>Per-invocation budget 3, physical W5, configured K4: the first step of
     * each call proposes at most remainingOutput-1 = 2 tokens, so the
     * EOS-fixture four-proposal assertion does not apply. The recurrence consumes
     * [entryPending, emitted[0..m-2]] and the last emitted token stays pending
     * for the next call. Requires -Dnd4j.mtp.multiRowCommit=1.
     */
    @Test
    public void testNonterminalContinuationSelectMatchesOff() {
        assertEquals("1", System.getenv("SD_MTP_MULTI_ROW_COMMIT"),
                "This continuation proof requires multi-row commit. Run with -Dnd4j.mtp.multiRowCommit=1.");
        ChainEvidence select = runNonterminalChain(true);
        ChainEvidence off = runNonterminalChain(false);

        // The SELECT chain must have actually committed selected state.
        assertTrue(select.copiedBytes > 0,
                "the SELECT chain must have copied selected-state bytes");
        assertEquals(0, off.copiedBytes,
                "the OFF chain must not copy selected-state bytes");

        // Cross-chain equality: both chains start from identical fresh snapshots
        // and must end in identical retained recurrent state, KV, positions,
        // masks and pending pair. This is the core continuation-correctness proof.
        assertEquals(select.gdn1, off.gdn1, EPS, "GDN after call 1");
        assertEquals(select.conv0_1, off.conv0_1, EPS, "conv older after call 1");
        assertEquals(select.conv1_1, off.conv1_1, EPS, "conv newest after call 1");
        assertEquals(select.gdn2, off.gdn2, EPS, "GDN after call 2");
        assertEquals(select.conv0_2, off.conv0_2, EPS, "conv older after call 2");
        assertEquals(select.conv1_2, off.conv1_2, EPS, "conv newest after call 2");
        assertEquals(select.pending1, off.pending1, 0L, "pending after call 1");
        assertEquals(select.pending2, off.pending2, 0L, "pending after call 2");
        assertEquals(select.targetPos, off.targetPos, "target position after 2 calls");
        assertEquals(select.carry, off.carry, EPS, "predictor carry");
        assertTrue(Arrays.equals(select.key, off.key), "retained target key");
        assertTrue(Arrays.equals(select.value, off.value), "retained target value");
        assertTrue(Arrays.equals(select.pkey, off.pkey), "retained predictor key");
        assertTrue(Arrays.equals(select.pvalue, off.pvalue), "retained predictor value");
        assertTrue(Arrays.equals(select.mask, off.mask), "retained target mask");
    }

    /**
     * Runs two budget-limited invocations of the nonterminal fixture on the same
     * live bindings (no snapshot restore, no position reset, no re-prefill) and
     * returns the retained evidence. selectMode=true drives the SELECT fast path
     * and records its copied bytes; false drives the OFF (legacy recovery) path.
     */
    private static ChainEvidence runNonterminalChain(boolean selectMode) {
        ChainEvidence e = new ChainEvidence();
        try (Plan target = selectTargetNonterminal(); Plan predictor = predictor()) {
            target.compile();
            predictor.compile();
            reset(target, predictor);
            long handle = target.executor.getNativePlanHandle().address();
            long entry = START;
            for (int call = 1; call <= 2; call++) {
                target.input("accepted").assign(1);
                String report = null;
                target.binding.beginNativeUse();
                try {
                    predictor.binding.beginNativeUse();
                    try {
                        if (selectMode) {
                            DspDiagnostics.initialize();
                            DspDiagnostics.setCategories(DspDiagnostics.KV_CACHE);
                            DspDiagnostics.setLevel(DspDiagnostics.LEVEL_DETAILED);
                            DspDiagnostics.clear();
                        }
                        // consumed=0: the nonterminal call does not terminate, so the
                        // EOS-fixture "consumed" count is not meaningful; the second
                        // call resumes from the actual advanced position (entry) and
                        // the actual pending token left by the first.
                        long[] emitted = runTransaction(target, predictor, 0, selectMode, false, 3, (int) entry);
                        if (selectMode) report = DspDiagnostics.getJsonReport();
                        entry += emitted[0];
                        if (call == 1) {
                            e.gdn1 = target.input("gdn").getDouble(0);
                            e.conv0_1 = target.input("conv").getDouble(0);
                            e.conv1_1 = target.input("conv").getDouble(1);
                            e.pending1 = emitted[emitted.length - 1];
                        } else {
                            e.gdn2 = target.input("gdn").getDouble(0);
                            e.conv0_2 = target.input("conv").getDouble(0);
                            e.conv1_2 = target.input("conv").getDouble(1);
                            e.pending2 = emitted[emitted.length - 1];
                            e.targetPos = target.input("position").getLong(0);
                            e.carry = predictor.input("carry").getDouble(0);
                            captureRetained(e, target, predictor);
                        }
                    } finally {
                        Nd4j.getExecutioner().commit();
                        predictor.binding.completeNativeUse();
                    }
                } finally {
                    target.binding.completeNativeUse();
                }
                if (selectMode) {
                    SelectSummary s = SelectSummary.parse(report);
                    assertEquals(1, s.p0Summaries, "call " + call + ": one P0 summary");
                    assertTrue(s.selectCommits >= 1,
                            "call " + call + ": the SELECT chain must commit; " + s.summaryMessage);
                    assertEquals(0, s.selectFallbacks,
                            "call " + call + ": no SELECT fallback; " + s.ineligibleMessages);
                    e.copiedBytes += Math.max(0, s.selectBytes);
                }
                assertEquals(handle, target.executor.getNativePlanHandle().address(),
                        "call " + call + ": same physical plan retained");
            }
        }
        return e;
    }

    /** Retained evidence for one two-call nonterminal chain. */
    private static final class ChainEvidence {
        double gdn1, conv0_1, conv1_1, gdn2, conv0_2, conv1_2;
        long pending1, pending2, targetPos;
        double carry, copiedBytes;
        double[] key, value, pkey, pvalue, mask;
    }

    private static void captureRetained(ChainEvidence e, Plan target, Plan predictor) {
        e.key = new double[CACHE];
        e.value = new double[CACHE];
        e.pkey = new double[CACHE];
        e.pvalue = new double[CACHE];
        e.mask = new double[WIDTH * CACHE];
        for (int c = 0; c < CACHE; c++) {
            e.key[c] = target.input("key").getDouble(0, c, 0, 0);
            e.value[c] = target.input("value").getDouble(0, c, 0, 0);
            e.pkey[c] = predictor.input("key").getDouble(0, c, 0, 0);
            e.pvalue[c] = predictor.input("value").getDouble(0, c, 0, 0);
        }
        for (int r = 0; r < WIDTH; r++)
            for (int c = 0; c < CACHE; c++)
                e.mask[r * CACHE + c] = target.input("mask").getDouble(0, 0, r, c);
    }

    /**
     * Nonterminal variant of the SELECT target: identical arithmetic to
     * {@link #selectTarget()} but the per-row correction is an ordinary token
     * (CORRECTION), not EOS. Row r &lt; accepted targets DRAFT (matching the
     * predictor draft -> accepted); row r &gt;= accepted targets CORRECTION, an
     * ordinary (non-EOS) correction token, so that draft row is rejected (partial
     * acceptance) and CORRECTION stays pending for a continuation. Because
     * CORRECTION &lt;&gt; EOS, a budget-limited call ends by budget, not by stop.
     */
    private static Plan selectTargetNonterminal() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        SDVariable mask = p.placeholder("mask", mask(WIDTH));
        p.echo("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable position = p.placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable length = p.placeholder("actual_length", Nd4j.scalar(DataType.INT64, WIDTH));
        SDVariable accepted = p.placeholder("accepted", Nd4j.valueArrayOf(new long[]{1}, 3, DataType.INT64));
        SDVariable convState = p.placeholder("conv", Nd4j.createFromArray(0.25f, -0.5f).reshape(1, 1, 2));
        SDVariable gdnState = p.placeholder("gdn", Nd4j.valueArrayOf(new long[]{1, 1, 1, 1}, 2, DataType.FLOAT));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable x = ids.castTo(DataType.FLOAT).reshape(1, WIDTH, 1);
        SDVariable[] conv = new CausalConv1dWithPrefix(p.graph, x,
                p.graph.constant(Nd4j.createFromArray(0.25f, 0.5f, 1f).reshape(1, 3)),
                null, convState, length, 0, 0).outputVariables();
        p.namedOutput(conv[1], "conv_next");
        p.namedOutput(conv[2], "conv_prefix");
        SDVariable qk = p.graph.constant(Nd4j.ones(DataType.FLOAT, 1, WIDTH, 1, 1));
        SDVariable[] gdn = new GatedDeltaRuleWithPrefix(p.graph, qk, qk,
                conv[0].reshape(1, WIDTH, 1, 1),
                p.graph.constant(Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 0.5, DataType.FLOAT)),
                p.graph.constant(Nd4j.zeros(DataType.FLOAT, 1, WIDTH, 1)), gdnState, length)
                .outputVariables();
        p.namedOutput(gdn[1], "gdn_next");
        p.namedOutput(gdn[2], "gdn_prefix");
        SDVariable attention = p.graph.nn().dotProductAttentionV2("attention", qk.mul(0),
                conv[0].reshape(1, WIDTH, 1, 1), gdn[0], null, null,
                key, value, position, mask, 0.0, 0.0, false, false);
        SDVariable hidden = gdn[0].add(attention).reshape(1, WIDTH, 1);
        p.namedOutput(hidden, "hidden");
        SDVariable rows = p.graph.constant(Nd4j.createFromArray(0L, 1L, 2L, 3L, 4L).reshape(1, WIDTH, 1));
        SDVariable match = rows.lt(accepted.reshape(1, 1, 1)).castTo(DataType.FLOAT);
        SDVariable draftBias = p.graph.constant(Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB));
        // CORRECTION = 2 -> bias at index 2.
        SDVariable corrBias = p.graph.constant(Nd4j.createFromArray(0f, 0f, 40f, 0f, 0f).reshape(1, 1, VOCAB));
        SDVariable slope = p.graph.constant(Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f).reshape(1, 1, VOCAB)).div(16);
        p.namedOutput(match.mul(draftBias).add(match.rsub(1).mul(corrBias)).add(hidden.mul(slope)), "logits");
        return p;
    }

    private static long[] runTransaction(Plan target, Plan predictor, int consumed, boolean selectMode,
                                       boolean fullNumerics, int budget, int entryPosition) {
        target.assertInputBinding("conv");
        target.assertInputBinding("gdn");
        try (INDArray embeddings = Nd4j.zeros(DataType.FLOAT, 1, 1, 1);
             INDArray table = Nd4j.ones(DataType.FLOAT, VOCAB, 1);
             INDArray positions = Nd4j.valueArrayOf(new long[]{1, 1}, entryPosition, DataType.INT64)) {
            AutoregressiveDecode op = new AutoregressiveDecode(
                    embeddings, table, target.input("ids"), target.input("mask"), positions,
                    new INDArray[]{target.input("key"), target.input("value")},
                    target.binding.getPlanHandle(), target.binding.getContextHandle(),
                    target.binding.getInputCount(), target.binding.getOutputCount(),
                    -1, -1, target.ext("mask"), -1, target.ext("ids"), target.out("logits"),
                    -1, target.ext("position"), target.ext("cache_position"),
                    new int[]{target.ext("key"), target.ext("value")}, new int[0],
                    new int[]{target.ext("gdn")}, new int[]{target.out("gdn_next")},
                    new int[]{target.ext("conv")}, new int[]{target.out("conv_next")},
                    budget, EOS, 1, entryPosition, 0.0, 0, 0.0, 1.0, Set.of());
            op.withDecodePolicy(AutoregressiveDecode.DECODE_STRATEGY_SPECULATIVE,
                            1, WIDTH, 1, 1, -1, 1, 1.0, 0.0, 0)
                    .withActualSequenceLengthExtIdx(target.ext("actual_length"))
                    .withSpeculativeDecoding(WIDTH - 1, AutoregressiveDecode.SPECULATOR_TYPE_MTP)
                    .withMtpPlan(predictor.input("ids"), predictor.input("carry"),
                            predictor.input("mask"), predictor.input("position"),
                            predictor.input("cache_position"),
                            new INDArray[]{predictor.input("key"), predictor.input("value")},
                            predictor.binding.getPlanHandle(), predictor.binding.getContextHandle(),
                            predictor.binding.getInputCount(),
                            predictor.binding.getOutputCount(), predictor.ext("ids"), predictor.ext("carry"),
                            predictor.ext("mask"), predictor.ext("position"), predictor.ext("cache_position"),
                            new int[]{predictor.ext("key"), predictor.ext("value")},
                            predictor.out("logits"), predictor.out("hidden"), target.out("hidden"));
            if (selectMode) {
                // Packet 07 SELECT: both companions captured; prefix output indices
                // resolved by name against this binding. SELECT mode is the only
                // supported non-off mode (Packet 08 rejects shadow at admission).
                op.withMtpPrefixSelect(2,
                        new int[]{target.ext("gdn")}, new int[]{target.out("gdn_next")},
                        new int[]{target.ext("conv")}, new int[]{target.out("conv_next")},
                        new int[]{target.out("gdn_prefix"), target.out("conv_prefix")});
            }
            INDArray[] result = Nd4j.getExecutioner().exec(op);
            long[] out;
            try {
                String label = "consumed=" + consumed + (selectMode ? " select" : " rerun");
                assertEquals(WIDTH, target.input("ids").size(1), "physical width must remain five");
                if (fullNumerics) {
                    // The EOS-fixture detailed oracle: every fullNumerics caller runs from
                    // the reset state with the EOS-terminated token pattern. Multi-call
                    // continuations pass fullNumerics=false and drive their own oracle
                    // from the actual emitted tokens.
                    assertEquals(consumed, result[1].getLong(0), label + ": one EOS-terminated transaction");
                    assertEquals(budget - 1, result[2].getFloat(7), 0f, label + ": proposals require budget-1 verification");
                    assertEquals(consumed - 1, result[2].getFloat(8), 0f, label + ": accepted drafts");
                    assertEquals(1, result[2].getFloat(9), 0f, label + ": exactly one speculative transaction");
                    for (int row = 0; row < consumed; row++) {
                        assertEquals(row == consumed - 1 ? EOS : DRAFT, result[0].getLong(row),
                                label + ": emitted token " + row);
                    }
                    assertEquals(consumed, target.input("actual_length").getLong(0),
                            label + ": authoritative rerun length, not full verification length");
                    checkNumerics(target, predictor, consumed, label, selectMode);
                    assertEquals(entryPosition + consumed, positions.getLong(0), label + ": published position");
                }
                long cnt = result[1].getLong(0);
                assertTrue(cnt >= 1, label + ": at least one token must be emitted");
                out = new long[(int) cnt + 1];
                out[0] = cnt;
                for (int i = 0; i < cnt; i++) out[i + 1] = result[0].getLong(i);
                return out;
            } finally {
                for (INDArray array : result) array.close();
            }
        }
    }

    private static void checkNumerics(Plan target, Plan predictor, int consumed, String label,
                                      boolean selectMode) {
        // Independent row-by-row reference; no ND4J op or window graph is used here.
        double state = 2.0;
        double older = 0.25;
        double previous = -0.5;
        double carry = 7.0;
        double visibleValueSum = SENTINEL; // one pre-existing target KV row
        try (INDArray logits = target.copyOutput("logits");
             INDArray hidden = target.copyOutput("hidden")) {
            assertArrayEquals(new long[]{1, WIDTH, VOCAB}, logits.shape());
            assertArrayEquals(new long[]{1, WIDTH, 1}, hidden.shape());
            for (int row = 0; row < consumed; row++) {
                int token = row == 0 ? BASE : DRAFT; // EOS is pending, never consumed.
                double convolution = 0.25 * older + 0.5 * previous + token;
                state = 0.5 * state + 0.5 * convolution;
                older = previous;
                previous = token;
                visibleValueSum += convolution;
                double expectedHidden = state + visibleValueSum / (START + row + 1);
                assertEquals(expectedHidden, hidden.getDouble(0, row, 0), EPS, label + ": hidden " + row);
                for (int cls = 0; cls < VOCAB; cls++) {
                    int winner = row < consumed - 1 ? DRAFT : EOS;
                    double expected = (cls == winner ? 40.0 : 0.0) + expectedHidden * (cls + 1) / 16.0;
                    assertEquals(expected, logits.getDouble(0, row, cls), EPS,
                            label + ": logit " + row + "/" + cls);
                }
                assertEquals(state, target.input("key").getDouble(0, START + row, 0, 0), EPS,
                        label + ": retained target key " + row);
                assertEquals(convolution, target.input("value").getDouble(0, START + row, 0, 0), EPS,
                        label + ": retained target value " + row);
                int predictorRow = START - 1 + row;
                double predictorKey = 64.0 * predictorRow + 8.0 * carry + token;
                assertEquals(predictorKey, predictor.input("key").getDouble(0, predictorRow, 0, 0), EPS,
                        label + ": repaired predictor key " + row);
                assertEquals(predictorKey + 0.5, predictor.input("value").getDouble(0, predictorRow, 0, 0), EPS,
                        label + ": repaired predictor value " + row);
                carry = expectedHidden;
            }
        }
        target.assertInputBinding("conv");
        target.assertInputBinding("gdn");
        String evidence = "";
        if (selectMode) {
            // The diagnostics ring still holds THIS invocation's events: the clear
            // ran before the op and no other native invocation has happened. Read
            // it right here so the numerical failure carries its evidence.
            SelectSummary s = SelectSummary.parse(DspDiagnostics.getJsonReport());
            evidence = " [P0 summary=" + s.summaryMessage
                    + " selectEvents=" + s.selectEvents + " rerunEvents=" + s.rerunEvents
                    + " ineligible=" + s.ineligibleMessages
                    + " trail=" + s.selectTrail + "]";
        }
        assertEquals(state, target.input("gdn").getDouble(0), EPS,
                label + ": full GDN state" + evidence);
        assertEquals(older, target.input("conv").getDouble(0), EPS,
                label + ": conv older" + evidence);
        assertEquals(previous, target.input("conv").getDouble(1), EPS,
                label + ": conv newest" + evidence);
        assertEquals(carry, predictor.input("carry").getDouble(0), EPS, label + ": pending carry");
        assertEquals(EOS, target.input("ids").getLong(0), label + ": target pending token");
        assertEquals(EOS, predictor.input("ids").getLong(0), label + ": predictor pending token");
        for (String position : new String[]{"position", "cache_position"}) {
            assertEquals(START + consumed, target.input(position).getLong(0), label + ": target " + position);
            assertEquals(START + consumed - 1, predictor.input(position).getLong(0),
                    label + ": predictor " + position);
        }
        for (int row = 0; row < WIDTH; row++) {
            for (int col = 0; col < CACHE; col++) {
                double bias = target.input("mask").getDouble(0, 0, row, col);
                if (col < START + consumed) assertEquals(0, bias, EPS, label + ": visible target KV");
                else assertTrue(bias <= -1e9, label + ": rejected target KV must be masked");
            }
        }
        for (int col = 0; col < CACHE; col++) {
            double bias = predictor.input("mask").getDouble(col);
            if (col < START + consumed - 1) assertEquals(0, bias, EPS, label + ": visible predictor KV");
            else assertTrue(bias <= -1e9, label + ": pending/rejected predictor KV must be masked at " + col);
        }
        // Existing prefix and rows outside the physical write window must not change.
        for (String kv : new String[]{"key", "value"}) {
            assertEquals(SENTINEL, target.input(kv).getFloat(0), 0f, label + ": pre-existing " + kv);
            for (int row = START + WIDTH - 2; row < CACHE; row++) {
                assertEquals(SENTINEL, predictor.input(kv).getFloat(0, row, 0, 0), 0f,
                        label + ": untouched predictor " + kv + " " + row);
            }
            for (int row = START + WIDTH; row < CACHE; row++) {
                assertEquals(SENTINEL, target.input(kv).getFloat(0, row, 0, 0), 0f,
                        label + ": untouched target " + kv + " " + row);
            }
        }
    }

    private static Plan target() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        SDVariable mask = p.placeholder("mask", mask(WIDTH));
        p.echo("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable position = p.placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable length = p.placeholder("actual_length", Nd4j.scalar(DataType.INT64, WIDTH));
        SDVariable accepted = p.placeholder("accepted", Nd4j.valueArrayOf(new long[]{1}, 3, DataType.INT64));
        SDVariable convState = p.placeholder("conv", Nd4j.createFromArray(0.25f, -0.5f).reshape(1, 1, 2));
        SDVariable gdnState = p.placeholder("gdn", Nd4j.valueArrayOf(new long[]{1, 1, 1, 1}, 2, DataType.FLOAT));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable x = ids.castTo(DataType.FLOAT).reshape(1, WIDTH, 1);
        SDVariable[] conv = new CausalConv1d(p.graph, x,
                p.graph.constant(Nd4j.createFromArray(0.25f, 0.5f, 1f).reshape(1, 3)),
                null, convState, length, 0, 0).outputVariables();
        p.namedOutput(conv[1], "conv_next");
        SDVariable qk = p.graph.constant(Nd4j.ones(DataType.FLOAT, 1, WIDTH, 1, 1));
        SDVariable[] gdn = new GatedDeltaRule(p.graph, qk, qk, conv[0].reshape(1, WIDTH, 1, 1),
                p.graph.constant(Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 0.5, DataType.FLOAT)),
                p.graph.constant(Nd4j.zeros(DataType.FLOAT, 1, WIDTH, 1)), gdnState, length).outputVariables();
        p.namedOutput(gdn[1], "gdn_next");
        SDVariable attention = p.graph.nn().dotProductAttentionV2("attention", qk.mul(0),
                conv[0].reshape(1, WIDTH, 1, 1), gdn[0], null, null,
                key, value, position, mask, 0.0, 0.0, false, false);
        SDVariable hidden = gdn[0].add(attention).reshape(1, WIDTH, 1);
        p.namedOutput(hidden, "hidden");
        SDVariable rows = p.graph.constant(Nd4j.createFromArray(0L, 1L, 2L, 3L, 4L).reshape(1, WIDTH, 1));
        SDVariable match = rows.lt(accepted.reshape(1, 1, 1)).castTo(DataType.FLOAT);
        SDVariable draftBias = p.graph.constant(Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB));
        SDVariable eosBias = p.graph.constant(Nd4j.createFromArray(40f, 0f, 0f, 0f, 0f).reshape(1, 1, VOCAB));
        SDVariable slope = p.graph.constant(Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f).reshape(1, 1, VOCAB)).div(16);
        p.namedOutput(match.mul(draftBias).add(match.rsub(1).mul(eosBias)).add(hidden.mul(slope)), "logits");
        return p;
    }

    /**
     * Packet 07 SELECT target: identical arithmetic to {@link #target()} but the
     * recurrent ops are the COMPANION capture variants, and the checkpoint
     * outputs are requested. The companion ops produce byte-identical ordinary
     * outputs (activation and final state), so the same independent scalar
     * oracle applies to both fixtures; the only difference is the extra
     * time-leading checkpoint tensor each companion emits.
     */
    private static Plan selectTarget() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, WIDTH));
        SDVariable mask = p.placeholder("mask", mask(WIDTH));
        p.echo("position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable position = p.placeholder("cache_position", Nd4j.valueArrayOf(new long[]{1}, START, DataType.INT64));
        SDVariable length = p.placeholder("actual_length", Nd4j.scalar(DataType.INT64, WIDTH));
        SDVariable accepted = p.placeholder("accepted", Nd4j.valueArrayOf(new long[]{1}, 3, DataType.INT64));
        SDVariable convState = p.placeholder("conv", Nd4j.createFromArray(0.25f, -0.5f).reshape(1, 1, 2));
        SDVariable gdnState = p.placeholder("gdn", Nd4j.valueArrayOf(new long[]{1, 1, 1, 1}, 2, DataType.FLOAT));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable x = ids.castTo(DataType.FLOAT).reshape(1, WIDTH, 1);
        SDVariable[] conv = new CausalConv1dWithPrefix(p.graph, x,
                p.graph.constant(Nd4j.createFromArray(0.25f, 0.5f, 1f).reshape(1, 3)),
                null, convState, length, 0, 0).outputVariables();
        p.namedOutput(conv[1], "conv_next");
        p.namedOutput(conv[2], "conv_prefix");
        SDVariable qk = p.graph.constant(Nd4j.ones(DataType.FLOAT, 1, WIDTH, 1, 1));
        SDVariable[] gdn = new GatedDeltaRuleWithPrefix(p.graph, qk, qk,
                conv[0].reshape(1, WIDTH, 1, 1),
                p.graph.constant(Nd4j.valueArrayOf(new long[]{1, WIDTH, 1}, 0.5, DataType.FLOAT)),
                p.graph.constant(Nd4j.zeros(DataType.FLOAT, 1, WIDTH, 1)), gdnState, length)
                .outputVariables();
        p.namedOutput(gdn[1], "gdn_next");
        p.namedOutput(gdn[2], "gdn_prefix");
        SDVariable attention = p.graph.nn().dotProductAttentionV2("attention", qk.mul(0),
                conv[0].reshape(1, WIDTH, 1, 1), gdn[0], null, null,
                key, value, position, mask, 0.0, 0.0, false, false);
        SDVariable hidden = gdn[0].add(attention).reshape(1, WIDTH, 1);
        p.namedOutput(hidden, "hidden");
        SDVariable rows = p.graph.constant(Nd4j.createFromArray(0L, 1L, 2L, 3L, 4L).reshape(1, WIDTH, 1));
        SDVariable match = rows.lt(accepted.reshape(1, 1, 1)).castTo(DataType.FLOAT);
        SDVariable draftBias = p.graph.constant(Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB));
        SDVariable eosBias = p.graph.constant(Nd4j.createFromArray(40f, 0f, 0f, 0f, 0f).reshape(1, 1, VOCAB));
        SDVariable slope = p.graph.constant(Nd4j.createFromArray(1f, 2f, 3f, 4f, 5f).reshape(1, 1, VOCAB)).div(16);
        p.namedOutput(match.mul(draftBias).add(match.rsub(1).mul(eosBias)).add(hidden.mul(slope)), "logits");
        return p;
    }

    private static Plan predictor() {
        Plan p = new Plan();
        SDVariable ids = p.placeholder("ids", Nd4j.ones(DataType.INT64, 1, 1));
        SDVariable carry = p.placeholder("carry", Nd4j.valueArrayOf(new long[]{1, 1, 1}, 7, DataType.FLOAT));
        SDVariable mask = p.placeholder("mask", mask(1));
        SDVariable rope = p.placeholder("position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable position = p.placeholder("cache_position", Nd4j.zeros(DataType.INT64, 1));
        SDVariable key = p.placeholder("key", cache());
        SDVariable value = p.placeholder("value", cache());
        SDVariable k = carry.reshape(1, 1, 1, 1).mul(8)
                .add(ids.castTo(DataType.FLOAT).reshape(1, 1, 1, 1))
                .add(rope.castTo(DataType.FLOAT).reshape(1, 1, 1, 1).mul(64));
        p.output(p.graph.nn().dotProductAttentionV2("attention", k, k.add(0.5), k, null, null,
                key, value, position, mask, 0.0, 0.0, false, false));
        p.output(carry.add("hidden", 1));
        p.namedOutput(ids.castTo(DataType.FLOAT).mul(0).reshape(1, 1, 1).add(p.graph.constant(
                Nd4j.createFromArray(0f, 0f, 0f, 0f, 40f).reshape(1, 1, VOCAB))), "logits");
        return p;
    }

    private static INDArray cache() {
        return Nd4j.valueArrayOf(new long[]{1, CACHE, 1, 1}, SENTINEL, DataType.FLOAT);
    }

    private static INDArray mask(int rows) {
        return Nd4j.valueArrayOf(new long[]{1, 1, rows, CACHE}, -Float.MAX_VALUE, DataType.FLOAT);
    }

    private static void reset(Plan target, Plan predictor) {
        target.input("ids").assign(BASE);
        target.input("actual_length").assign(WIDTH);
        target.input("gdn").assign(2);
        target.input("conv").putScalar(0, 0.25f);
        target.input("conv").putScalar(1, -0.5f);
        target.input("position").assign(START);
        target.input("cache_position").assign(START);
        predictor.input("ids").assign(BASE);
        predictor.input("carry").assign(7);
        predictor.input("position").assign(START - 1);
        predictor.input("cache_position").assign(START - 1);
        for (Plan p : new Plan[]{target, predictor}) {
            p.input("key").assign(SENTINEL);
            p.input("value").assign(SENTINEL);
            p.input("mask").assign(-Float.MAX_VALUE);
        }
        for (int row = 0; row < WIDTH; row++) target.input("mask").putScalar(new long[]{0, 0, row, 0}, 0);
    }

    private static final class Snapshot implements AutoCloseable {
        private final Map<String, INDArray> saved = new LinkedHashMap<>();
        Snapshot(Plan p) { p.inputs.forEach((name, array) -> saved.put(name, array.dup())); }
        void restore(Plan p) { saved.forEach((name, array) -> p.input(name).assign(array)); }
        @Override public void close() { saved.values().forEach(INDArray::close); }
    }

    private static final class Plan implements AutoCloseable {
        private final SameDiff graph = SameDiff.create();
        private final Map<String, INDArray> inputs = new LinkedHashMap<>();
        private final List<String> outputs = new ArrayList<>();
        private DynamicShapePlanExecutor executor;
        private NativeExecutionBinding binding;
        SDVariable placeholder(String name, INDArray value) {
            inputs.put(name, value);
            return graph.placeHolder(name, value.dataType(), value.shape());
        }
        void output(SDVariable v) { outputs.add(v.name()); }
        void namedOutput(SDVariable v, String name) {
            graph.updateVariableNameAndReference(v, name);
            output(v);
        }
        void echo(String name, INDArray value) { output(placeholder(name, value).add(name + "_echo", 1)); }
        INDArray input(String name) { return inputs.get(name); }
        void compile() {
            graph.setDspAutoCompileEnabled(true);
            graph.setDspNativeAutoCompileEnabled(true);
            for (int i = 0; i < 8; i++) graph.output(inputs, outputs.toArray(new String[0]));
            executor = graph.getOrCreateSession().getDynamicShapePlanExecutor();
            assertNotNull(executor);
            assertNotNull(executor.getNativePlanHandle());
            assertFalse(executor.getNativePlanHandle().isNull());
            assertNotNull(executor.getCachedOpContext());
            binding = executor.captureNativeExecutionBinding();
            String[] keys = binding.getExternalInputKeysSnapshot();
            INDArray[] boundInputs = binding.getExternalInputsSnapshot();
            for (int i = 0; i < keys.length; i++) {
                if (inputs.containsKey(keys[i])) inputs.put(keys[i], boundInputs[i]);
            }
        }
        int ext(String name) {
            int index = binding.findExternalInputIndex(name);
            assertTrue(index >= 0, "missing input " + name);
            return index;
        }
        int out(String name) {
            int index = binding.findOutputIndex(name);
            assertTrue(index >= 0, "missing output " + name);
            return index;
        }
        void assertInputBinding(String name) {
            OpaqueContext context = binding.getContextHandle();
            NativeOps ops = context.backendOwner().nativeOps();
            OpaqueNDArray bound = ops.getInputArrayNative(context, ext(name));
            assertNotNull(bound, "missing bound input " + name);
            Pointer nativeBuffer = ops.getOpaqueNDArraySpecialBuffer(bound);
            Pointer javaBuffer = ops.dbSpecialBuffer(input(name).data().opaqueBuffer());
            assertNotNull(nativeBuffer);
            assertNotNull(javaBuffer);
            assertEquals(javaBuffer.address(), nativeBuffer.address(),
                    "native input binding must reference fixture storage: " + name);
        }

        /** Copy native outputs WITHOUT another graph execution or taking ownership of plan buffers. */
        INDArray copyOutput(String name) {
            OpaqueContext context = binding.getContextHandle();
            NativeOps ops = context.backendOwner().nativeOps();
            Integer slot = executor.getCurrentPlan().getOutputNameToSlotIndex().get(name);
            assertNotNull(slot, "missing native output slot " + name);
            OpaqueNDArray output = ops.getPlanSlotOutputArray(executor.getNativePlanHandle(), slot);
            assertNotNull(output);
            assertFalse(output.isNull());
            output.attachOwner(context.backendOwner());
            long[] info = OpaqueNDArray.getOpaqueNDArrayShapeInfo(output);
            assertEquals(DataType.FLOAT, ArrayOptionsHelper.dataType(info), "native output dtype " + name);
            long length = OpaqueNDArray.getOpaqueNDArrayLength(output);
            INDArray copy = Nd4j.createUninitialized(DataType.FLOAT, Shape.shape(info), Shape.stride(info), Shape.order(info));
            Pointer special = ops.getOpaqueNDArraySpecialBuffer(output);
            Pointer primary = special == null || special.isNull() ? ops.getOpaqueNDArrayBuffer(output) : null;
            OpaqueDataBuffer source = ops.dbCreateExternalDataBuffer(length, DataType.FLOAT.toInt(), primary, special);
            try {
                assertNotNull(source);
                assertFalse(source.isNull());
                ops.copyBuffer(copy.data().opaqueBuffer(), length, source, 0, 0);
                Nd4j.getExecutioner().commit();
            } catch (RuntimeException | Error e) {
                copy.close();
                throw e;
            } finally {
                if (source != null && !source.isNull()) ops.deleteDataBuffer(source);
            }
            return copy;
        }
        @Override public void close() {
            if (binding != null) binding.close();
            graph.close();
        }
    }

    /**
     * Parses the SINGLE current-call MTP_P0_CUDA summary plus event tags out of
     * the DspDiagnostics JSON report captured for exactly one native invocation.
     *
     * <p>The P0 summary is the MTP_P0_CUDA event emitted at the end of every
     * decode call carrying key=value counters. This parser extracts its message
     * field and the required counters; a missing, duplicated, or ambiguous
     * summary is itself a failure, never substituted with zeros.</p>
     */
    static final class SelectSummary {
        final String summaryMessage;
        final int selectCommits;
        final int selectFallbacks;
        final int selectBytes;
        final List<Integer> selectLayerCounts;
        final int rerunEvents;
        final int selectEvents;
        final int ineligibleEvents;
        final List<String> ineligibleMessages;
        final int p0Summaries;
        /** Every SELECT_COPY / SPEC_STATE_SELECT message in ring order. */
        final List<String> selectTrail;

        private SelectSummary(String summaryMessage, int selectCommits, int selectFallbacks,
                              int selectBytes, List<Integer> selectLayerCounts,
                              int rerunEvents, int selectEvents, int ineligibleEvents,
                              List<String> ineligibleMessages, int p0Summaries,
                              List<String> selectTrail) {
            this.summaryMessage = summaryMessage;
            this.selectCommits = selectCommits;
            this.selectFallbacks = selectFallbacks;
            this.selectBytes = selectBytes;
            this.selectLayerCounts = selectLayerCounts;
            this.rerunEvents = rerunEvents;
            this.selectEvents = selectEvents;
            this.ineligibleEvents = ineligibleEvents;
            this.ineligibleMessages = ineligibleMessages;
            this.p0Summaries = p0Summaries;
            this.selectTrail = selectTrail;
        }

        /**
         * Extract event messages from the serialized report. The JSON shape is the
         * C++ DspDiagnostics serialization: an events array whose entries carry a
         * message string field. Messages are matched on the MTP_P0_CUDA /
         * SPEC_STATE_SELECT / SPEC_STATE_RERUN / SELECT_INELIGIBLE event TAGS -
         * prefix matches on the message text, not arbitrary substring hits.
         */
        static SelectSummary parse(String jsonReport) {
            List<String> messages = extractEventMessages(jsonReport);
            String summary = null;
            int p0Count = 0;
            int selectCommits = -1;
            int selectFallbacks = -1;
            int selectBytes = -1;
            List<Integer> selectLayerCounts = new ArrayList<>();
            int rerun = 0;
            int select = 0;
            int ineligible = 0;
            List<String> ineligibleMsgs = new ArrayList<>();
            List<String> trail = new ArrayList<>();
            for (String message : messages) {
                if (message == null) continue;
                if (message.startsWith("MTP_P0_CUDA ")) {
                    p0Count++;
                    summary = message;
                    selectCommits = extractCounter(message, "checkpointSelectCommits");
                    selectFallbacks = extractCounter(message, "checkpointSelectFallbacks");
                    selectBytes = extractCounter(message, "checkpointSelectBytes");
                } else if (message.startsWith("SPEC_STATE_SELECT")) {
                    select++;
                    selectLayerCounts.add(extractCounter(message, "layers"));
                    trail.add(message);
                } else if (message.startsWith("SPEC_STATE_RERUN")) {
                    rerun++;
                } else if (message.startsWith("SELECT_INELIGIBLE")) {
                    ineligible++;
                    ineligibleMsgs.add(message);
                } else if (message.startsWith("SELECT_COPY")) {
                    trail.add(message);
                }
            }
            return new SelectSummary(summary, selectCommits, selectFallbacks, selectBytes,
                    selectLayerCounts, rerun, select, ineligible, ineligibleMsgs, p0Count, trail);
        }

        /** Pull event "message" fields from the report's events array. */
        private static List<String> extractEventMessages(String json) {
            List<String> out = new ArrayList<>();
            if (json == null || json.isEmpty()) return out;
            // The events array entries are flat JSON objects with a "message" key.
            // Scan message occurrences and decode the JSON string payload.
            int idx = 0;
            while (true) {
                int keyAt = json.indexOf("\"message\"", idx);
                if (keyAt < 0) break;
                int colonAt = json.indexOf(':', keyAt + "\"message\"".length());
                if (colonAt < 0) break;
                int quoteAt = json.indexOf('"', colonAt + 1);
                if (quoteAt < 0) break;
                StringBuilder sb = new StringBuilder();
                int cursor = quoteAt + 1;
                boolean closed = false;
                while (cursor < json.length()) {
                    char c = json.charAt(cursor);
                    if (c == '\\' && cursor + 1 < json.length()) {
                        char next = json.charAt(cursor + 1);
                        switch (next) {
                            case 'n': sb.append('\n'); break;
                            case 't': sb.append('\t'); break;
                            case '"': sb.append('"'); break;
                            case '\\': sb.append('\\'); break;
                            default: sb.append(next);
                        }
                        cursor += 2;
                        continue;
                    }
                    if (c == '"') {
                        closed = true;
                        cursor++;
                        break;
                    }
                    sb.append(c);
                    cursor++;
                }
                if (!closed) break;
                out.add(sb.toString());
                idx = cursor;
            }
            return out;
        }

        /** Extract an integer key=value counter from a P0 summary message; -1 if absent. */
        private static int extractCounter(String message, String key) {
            String needle = key + "=";
            int at = message.indexOf(needle);
            if (at < 0) return -1;
            int start = at + needle.length();
            int end = start;
            while (end < message.length() && (Character.isDigit(message.charAt(end))
                    || (end == start && message.charAt(end) == '-'))) {
                end++;
            }
            try {
                return Integer.parseInt(message.substring(start, end));
            } catch (NumberFormatException e) {
                return -1;
            }
        }

        /**
         * The SELECT contract: exactly one summary, one commit, zero fallbacks/reruns,
         * the exact selected-state byte count, and the layer count reported by the
         * SPEC_STATE_SELECT event. Expected bytes are derived from the fixture's
         * actual state tensors (GDN [1,1,1,1] FLOAT = 4; conv [1,1,2] FLOAT = 8),
         * not from a general runtime constant.
         */
        void assertSelectContract(int consumed, long expectedSelectBytes, int expectedLayers) {
            String context = "consumed=" + consumed + " ";
            String evidence = context + "summary=" + summaryMessage
                    + " events: SPEC_STATE_SELECT=" + selectEvents
                    + " SPEC_STATE_RERUN=" + rerunEvents
                    + " SELECT_INELIGIBLE=" + ineligibleEvents
                    + (ineligibleMessages.isEmpty() ? "" : " records=" + ineligibleMessages);
            assertEquals(1, p0Summaries,
                    context + "expected EXACTLY ONE MTP_P0_CUDA summary for the observed invocation; "
                    + (p0Summaries == 0
                        ? "the diagnostics ring carried none - observability failure, not a pass"
                        : "ambiguous: multiple decode calls inside the observation interval")
                    + "; " + evidence);
            String summaryText = summaryMessage == null ? "<no summary>" : summaryMessage;
            assertTrue(selectCommits >= 0,
                    context + "checkpointSelectCommits MISSING from summary: " + summaryText);
            assertTrue(selectFallbacks >= 0,
                    context + "checkpointSelectFallbacks MISSING from summary: " + summaryText);
            assertTrue(selectBytes >= 0,
                    context + "checkpointSelectBytes MISSING from summary: " + summaryText);
            assertEquals(1, selectCommits,
                    context + "exactly one checkpoint-select commit required; " + evidence);
            assertEquals(0, selectFallbacks,
                    context + "zero checkpoint-select fallbacks required; " + evidence);
            assertEquals(expectedSelectBytes, selectBytes,
                    context + "checkpoint-select bytes must equal the admitted state byte total; "
                            + evidence);
            assertEquals(1, selectEvents,
                    context + "exactly one SPEC_STATE_SELECT event required; " + evidence);
            assertEquals(1, selectLayerCounts.size(),
                    context + "exactly one SPEC_STATE_SELECT layers field required; " + evidence);
            assertTrue(selectLayerCounts.get(0) >= 0,
                    context + "layers MISSING from SPEC_STATE_SELECT: " + selectTrail.get(0));
            assertEquals(expectedLayers, selectLayerCounts.get(0),
                    context + "SPEC_STATE_SELECT must report the admitted layer count; " + evidence);
            assertEquals(0, rerunEvents,
                    context + "zero SPEC_STATE_RERUN events required (a fallback-only pass is a failure); "
                            + evidence);
            assertEquals(0, ineligibleEvents,
                    context + "zero SELECT_INELIGIBLE events required; " + evidence);
        }
    }
}
