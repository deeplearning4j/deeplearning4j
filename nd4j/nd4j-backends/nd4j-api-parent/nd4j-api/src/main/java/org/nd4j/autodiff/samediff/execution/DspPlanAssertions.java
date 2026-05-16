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

package org.nd4j.autodiff.samediff.execution;

import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Assertion helpers that let tests pin down WHICH DSP phase failed instead of
 * only checking final output tensors.
 *
 * <p>Every method reads the live {@link DspDebugger.PlanReport} /
 * {@link DspDebugger.GraphReplayReport} and throws an {@link AssertionError}
 * with a detailed plan-state dump on failure. The failure message names the
 * phase, segment, and reason — so a regression localizes to a phase without
 * requiring log-diving.</p>
 *
 * <p>Intended use in platform-tests:</p>
 * <pre>
 *   sd.output(placeholders, "out");           // run once to compile
 *   DspPlanAssertions.assertPhaseReached(sd, PlanPhase.REPLAYING);
 *
 *   for (int i = 0; i &lt; 5; i++) sd.output(placeholders, "out");
 *   DspPlanAssertions.assertFullyReplaying(sd);
 *   DspPlanAssertions.assertNoCaptureFailures(sd);
 *   DspPlanAssertions.assertNoPhaseContractViolations(sd);
 * </pre>
 *
 * <p>All methods accept an optional trailing context string that is included
 * in the failure message — use it to record which test configuration triggered
 * the failure (e.g., {@code "cuda_graphs + frozen constants"}).</p>
 */
public final class DspPlanAssertions {

    private DspPlanAssertions() {}

    // ═══════════════════════════════════════════════════════════════════════
    // Plan phase assertions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the plan has reached at least the given phase.
     * A plan at REPLAYING satisfies {@code assertPhaseReached(SHAPES_FROZEN)}.
     */
    public static void assertPhaseReached(SameDiff sd, PlanPhase expected) {
        assertPhaseReached(sd, expected, null);
    }

    public static void assertPhaseReached(SameDiff sd, PlanPhase expected, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        if (report.planPhase == null || !report.planPhase.isAtLeast(expected)) {
            fail("assertPhaseReached", context,
                    "expected plan phase >= " + expected + " but was " + report.planPhase,
                    report);
        }
    }

    /** Assert the plan phase is exactly the given value. */
    public static void assertPhaseExact(SameDiff sd, PlanPhase expected) {
        assertPhaseExact(sd, expected, null);
    }

    public static void assertPhaseExact(SameDiff sd, PlanPhase expected, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        if (report.planPhase != expected) {
            fail("assertPhaseExact", context,
                    "expected plan phase == " + expected + " but was " + report.planPhase,
                    report);
        }
    }

    /** Assert the plan has reached REPLAYING and every replay-eligible segment is in REPLAYING. */
    public static void assertFullyReplaying(SameDiff sd) {
        assertFullyReplaying(sd, null);
    }

    public static void assertFullyReplaying(SameDiff sd, String context) {
        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.PlanReport report = debugger.analyzePlan();
        requirePlanAvailable(report, context);

        if (report.planPhase != PlanPhase.REPLAYING) {
            fail("assertFullyReplaying", context,
                    "plan phase is " + report.planPhase + ", expected REPLAYING",
                    report);
        }

        List<String> stuck = new ArrayList<>();
        for (DspDebugger.SegmentReport seg : report.segments) {
            if (seg.capturable && !seg.captureFailed && seg.phase != ExecutionPhase.REPLAYING) {
                stuck.add("seg[" + seg.index + "] phase=" + seg.phase
                        + " execCount=" + seg.executionCount);
            }
        }
        if (!stuck.isEmpty()) {
            fail("assertFullyReplaying", context,
                    stuck.size() + " capturable segment(s) are not REPLAYING: " + stuck,
                    report);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Unified GraphNodePhase assertions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the plan's unified lifecycle phase matches.
     * Prefer this over assertPhaseExact(PlanPhase) for new code.
     */
    public static void assertGraphNodePhase(SameDiff sd, GraphNodePhase expected) {
        assertGraphNodePhase(sd, expected, null);
    }

    public static void assertGraphNodePhase(SameDiff sd, GraphNodePhase expected, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        if (report.graphNodePhase != expected) {
            fail("assertGraphNodePhase", context,
                    "expected unified phase " + expected + " but was " + report.graphNodePhase
                            + " (planPhase=" + report.planPhase + ")",
                    report);
        }
    }

    /**
     * Assert the plan is SEALED (steady-state replay).
     * Equivalent to assertPhaseExact(PlanPhase.REPLAYING) but uses unified terminology.
     */
    public static void assertSealed(SameDiff sd) {
        assertGraphNodePhase(sd, GraphNodePhase.SEALED);
    }

    public static void assertSealed(SameDiff sd, String context) {
        assertGraphNodePhase(sd, GraphNodePhase.SEALED, context);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Segment assertions
    // ═══════════════════════════════════════════════════════════════════════

    /** Assert the plan has exactly the expected number of segments. */
    public static void assertSegmentCount(SameDiff sd, int expected) {
        assertSegmentCount(sd, expected, null);
    }

    public static void assertSegmentCount(SameDiff sd, int expected, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        if (report.numSegments != expected) {
            fail("assertSegmentCount", context,
                    "expected " + expected + " segments but found " + report.numSegments,
                    report);
        }
    }

    /** Assert no capturable segment recorded a capture failure. */
    public static void assertNoCaptureFailures(SameDiff sd) {
        assertNoCaptureFailures(sd, null);
    }

    public static void assertNoCaptureFailures(SameDiff sd, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);

        List<String> failed = new ArrayList<>();
        for (DspDebugger.SegmentReport seg : report.segments) {
            if (seg.captureFailed) {
                failed.add("seg[" + seg.index + "] execCount=" + seg.executionCount
                        + " phase=" + seg.phase);
            }
        }
        if (!failed.isEmpty()) {
            fail("assertNoCaptureFailures", context,
                    failed.size() + " segment(s) have captureFailed=true: " + failed,
                    report);
        }
    }

    /** Assert a specific segment has reached (at least) the given execution phase. */
    public static void assertSegmentReachedPhase(SameDiff sd, int segmentIndex,
                                                 ExecutionPhase expected) {
        assertSegmentReachedPhase(sd, segmentIndex, expected, null);
    }

    public static void assertSegmentReachedPhase(SameDiff sd, int segmentIndex,
                                                 ExecutionPhase expected, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        if (segmentIndex < 0 || segmentIndex >= report.segments.size()) {
            fail("assertSegmentReachedPhase", context,
                    "segment index " + segmentIndex + " out of range [0," + report.segments.size() + ")",
                    report);
        }
        DspDebugger.SegmentReport seg = report.segments.get(segmentIndex);
        if (seg.phase != expected) {
            fail("assertSegmentReachedPhase", context,
                    "seg[" + segmentIndex + "] expected phase " + expected
                            + " but was " + seg.phase + " (execCount=" + seg.executionCount + ")",
                    report);
        }
    }

    /** Assert every capturable segment has reached (exactly) the given execution phase. */
    public static void assertAllCapturableSegmentsReachedPhase(SameDiff sd,
                                                               ExecutionPhase expected) {
        assertAllCapturableSegmentsReachedPhase(sd, expected, null);
    }

    public static void assertAllCapturableSegmentsReachedPhase(SameDiff sd,
                                                               ExecutionPhase expected,
                                                               String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        List<String> mismatches = new ArrayList<>();
        for (DspDebugger.SegmentReport seg : report.segments) {
            if (seg.capturable && !seg.captureFailed && seg.phase != expected) {
                mismatches.add("seg[" + seg.index + "] phase=" + seg.phase
                        + " execCount=" + seg.executionCount);
            }
        }
        if (!mismatches.isEmpty()) {
            fail("assertAllCapturableSegmentsReachedPhase", context,
                    "expected " + expected + " but " + mismatches.size()
                            + " segment(s) differ: " + mismatches,
                    report);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Phase contract assertions
    // ═══════════════════════════════════════════════════════════════════════

    /** Assert the phase contract (see {@link DspDebugger#validatePhaseContract()}) holds. */
    public static void assertNoPhaseContractViolations(SameDiff sd) {
        assertNoPhaseContractViolations(sd, null);
    }

    public static void assertNoPhaseContractViolations(SameDiff sd, String context) {
        DspDebugger debugger = DspDebugger.attach(sd);
        DspDebugger.PhaseContractReport contract = debugger.validatePhaseContract();
        if (contract.hasViolations()) {
            DspDebugger.PlanReport report = debugger.analyzePlan();
            StringBuilder sb = new StringBuilder();
            sb.append(contract.getViolations().size())
                    .append(" phase contract violation(s):\n");
            for (DspDebugger.PhaseViolation v : contract.getViolations()) {
                sb.append("  - ").append(v).append("\n");
            }
            fail("assertNoPhaseContractViolations", context, sb.toString(), report);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Slot / op assertions
    // ═══════════════════════════════════════════════════════════════════════

    /** Assert the plan contains an op with the given name (case-sensitive match on SlotInfo.opName). */
    public static void assertOpCompiled(SameDiff sd, String opName) {
        assertOpCompiled(sd, opName, null);
    }

    public static void assertOpCompiled(SameDiff sd, String opName, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        for (DspDebugger.SlotInfo slot : report.slots) {
            if (opName.equals(slot.opName)) return;
        }
        fail("assertOpCompiled", context,
                "no slot with opName='" + opName + "' found in compiled plan",
                report);
    }

    /** Assert no slot is flagged as risky for graph capture (data-dependent, view-capable, etc.). */
    public static void assertNoRiskyOps(SameDiff sd) {
        assertNoRiskyOps(sd, null);
    }

    public static void assertNoRiskyOps(SameDiff sd, String context) {
        DspDebugger.PlanReport report = DspDebugger.attach(sd).analyzePlan();
        requirePlanAvailable(report, context);
        List<DspDebugger.SlotInfo> risky = report.getRiskyOps();
        if (!risky.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append(risky.size()).append(" risky op(s) in plan:\n");
            for (DspDebugger.SlotInfo s : risky) {
                sb.append("  - ").append(s).append("\n");
            }
            fail("assertNoRiskyOps", context, sb.toString(), report);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Segment Replay Detail Assertions
    // ═══════════════════════════════════════════════════════════════════════

    /** Replay mode constants returned by {@code getPlanSegmentReplayMode}. */
    public static final int REPLAY_MODE_NONE = 0;
    public static final int REPLAY_MODE_MONOLITHIC = 1;
    public static final int REPLAY_MODE_COMPOSITE = 2;

    /**
     * Assert that a segment has the expected number of gap units in its
     * composite replay schedule. Fails with the actual count if mismatched.
     */
    public static void assertSegmentGapUnitCount(SameDiff sd, int segmentIndex,
                                                  int expectedCount) {
        assertSegmentGapUnitCount(sd, segmentIndex, expectedCount, null);
    }

    public static void assertSegmentGapUnitCount(SameDiff sd, int segmentIndex,
                                                  int expectedCount, String context) {
        int actual = getNativeOps().getPlanSegmentGapUnitCount(getPlanHandle(sd), segmentIndex);
        if (actual != expectedCount) {
            fail("assertSegmentGapUnitCount", context,
                    "segment " + segmentIndex + ": expected " + expectedCount +
                    " gap units but was " + actual, null);
        }
    }

    /**
     * Assert that a segment has the expected number of island units.
     */
    public static void assertSegmentIslandUnitCount(SameDiff sd, int segmentIndex,
                                                     int expectedCount) {
        assertSegmentIslandUnitCount(sd, segmentIndex, expectedCount, null);
    }

    public static void assertSegmentIslandUnitCount(SameDiff sd, int segmentIndex,
                                                     int expectedCount, String context) {
        int actual = getNativeOps().getPlanSegmentIslandUnitCount(getPlanHandle(sd), segmentIndex);
        if (actual != expectedCount) {
            fail("assertSegmentIslandUnitCount", context,
                    "segment " + segmentIndex + ": expected " + expectedCount +
                    " island units but was " + actual, null);
        }
    }

    /**
     * Assert that gap slots cover at least the expected number of slots.
     */
    public static void assertSegmentGapSlotCount(SameDiff sd, int segmentIndex,
                                                  int minExpected) {
        assertSegmentGapSlotCount(sd, segmentIndex, minExpected, null);
    }

    public static void assertSegmentGapSlotCount(SameDiff sd, int segmentIndex,
                                                  int minExpected, String context) {
        int actual = getNativeOps().getPlanSegmentGapSlotCount(getPlanHandle(sd), segmentIndex);
        if (actual < minExpected) {
            fail("assertSegmentGapSlotCount", context,
                    "segment " + segmentIndex + ": expected >= " + minExpected +
                    " gap slots but was " + actual, null);
        }
    }

    /**
     * Assert that a specific island's replay handle is ready (captured and instantiated).
     */
    public static void assertIslandHandleReady(SameDiff sd, int segmentIndex, int islandIndex) {
        assertIslandHandleReady(sd, segmentIndex, islandIndex, null);
    }

    public static void assertIslandHandleReady(SameDiff sd, int segmentIndex, int islandIndex,
                                                String context) {
        int ready = getNativeOps().getPlanSegmentIslandHandleReady(
                getPlanHandle(sd), segmentIndex, islandIndex);
        if (ready != 1) {
            fail("assertIslandHandleReady", context,
                    "segment " + segmentIndex + " island " + islandIndex +
                    ": handle not ready (result=" + ready + ")", null);
        }
    }

    /**
     * Assert the replay mode for a segment.
     * @param expectedMode one of REPLAY_MODE_NONE, REPLAY_MODE_MONOLITHIC, REPLAY_MODE_COMPOSITE
     */
    public static void assertSegmentReplayMode(SameDiff sd, int segmentIndex, int expectedMode) {
        assertSegmentReplayMode(sd, segmentIndex, expectedMode, null);
    }

    public static void assertSegmentReplayMode(SameDiff sd, int segmentIndex, int expectedMode,
                                                String context) {
        int actual = getNativeOps().getPlanSegmentReplayMode(getPlanHandle(sd), segmentIndex);
        if (actual != expectedMode) {
            String expectedName = replayModeName(expectedMode);
            String actualName = replayModeName(actual);
            fail("assertSegmentReplayMode", context,
                    "segment " + segmentIndex + ": expected " + expectedName +
                    " but was " + actualName, null);
        }
    }

    /**
     * Assert the monolithic replay handle is ready for a segment.
     */
    public static void assertMonolithicHandleReady(SameDiff sd, int segmentIndex) {
        assertMonolithicHandleReady(sd, segmentIndex, null);
    }

    public static void assertMonolithicHandleReady(SameDiff sd, int segmentIndex, String context) {
        int ready = getNativeOps().getPlanSegmentMonolithicHandleReady(
                getPlanHandle(sd), segmentIndex);
        if (ready != 1) {
            fail("assertMonolithicHandleReady", context,
                    "segment " + segmentIndex + ": monolithic handle not ready (result=" + ready + ")", null);
        }
    }

    /**
     * Assert the segment execution count is at least the expected value.
     */
    public static void assertSegmentExecCountAtLeast(SameDiff sd, int segmentIndex,
                                                      int minCount) {
        assertSegmentExecCountAtLeast(sd, segmentIndex, minCount, null);
    }

    public static void assertSegmentExecCountAtLeast(SameDiff sd, int segmentIndex,
                                                      int minCount, String context) {
        int actual = getNativeOps().getSegmentExecutionCount(getPlanHandle(sd), segmentIndex);
        if (actual < minCount) {
            fail("assertSegmentExecCountAtLeast", context,
                    "segment " + segmentIndex + ": expected execCount >= " + minCount +
                    " but was " + actual, null);
        }
    }

    /**
     * Assert the segment execution count is exactly the expected value.
     */
    public static void assertSegmentExecCountExact(SameDiff sd, int segmentIndex,
                                                    int expectedCount) {
        assertSegmentExecCountExact(sd, segmentIndex, expectedCount, null);
    }

    public static void assertSegmentExecCountExact(SameDiff sd, int segmentIndex,
                                                    int expectedCount, String context) {
        int actual = getNativeOps().getSegmentExecutionCount(getPlanHandle(sd), segmentIndex);
        if (actual != expectedCount) {
            fail("assertSegmentExecCountExact", context,
                    "segment " + segmentIndex + ": expected execCount == " + expectedCount +
                    " but was " + actual, null);
        }
    }

    /**
     * Get the gap unit count for a segment (non-asserting query).
     */
    public static int getSegmentGapUnitCount(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentGapUnitCount(getPlanHandle(sd), segmentIndex);
    }

    /**
     * Get the island unit count for a segment (non-asserting query).
     */
    public static int getSegmentIslandUnitCount(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentIslandUnitCount(getPlanHandle(sd), segmentIndex);
    }

    /**
     * Get the total gap slot count for a segment (non-asserting query).
     */
    public static int getSegmentGapSlotCount(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentGapSlotCount(getPlanHandle(sd), segmentIndex);
    }

    /**
     * Get the replay mode for a segment (non-asserting query).
     * @return one of REPLAY_MODE_NONE, REPLAY_MODE_MONOLITHIC, REPLAY_MODE_COMPOSITE
     */
    public static int getSegmentReplayMode(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentReplayMode(getPlanHandle(sd), segmentIndex);
    }

    /**
     * Get the segment execution count (non-asserting query).
     */
    public static int getSegmentExecCount(SameDiff sd, int segmentIndex) {
        return getNativeOps().getSegmentExecutionCount(getPlanHandle(sd), segmentIndex);
    }

    private static String replayModeName(int mode) {
        switch (mode) {
            case REPLAY_MODE_NONE: return "NONE";
            case REPLAY_MODE_MONOLITHIC: return "MONOLITHIC";
            case REPLAY_MODE_COMPOSITE: return "COMPOSITE";
            default: return "UNKNOWN(" + mode + ")";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Plan-Level Replay + Pointer Assertions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert total CUDA graph replays across all segments reached at least N.
     */
    public static void assertTotalGraphReplaysAtLeast(SameDiff sd, int minReplays) {
        assertTotalGraphReplaysAtLeast(sd, minReplays, null);
    }

    public static void assertTotalGraphReplaysAtLeast(SameDiff sd, int minReplays, String context) {
        int actual = getNativeOps().getPlanTotalGraphReplays(getPlanHandle(sd));
        if (actual < minReplays) {
            fail("assertTotalGraphReplaysAtLeast", context,
                    "expected >= " + minReplays + " total graph replays but was " + actual, null);
        }
    }

    /**
     * Assert buffer pointers are stable (same addresses across executions).
     */
    public static void assertPointersStable(SameDiff sd) {
        assertPointersStable(sd, null);
    }

    public static void assertPointersStable(SameDiff sd, String context) {
        int stable = getNativeOps().getPlanPointersStable(getPlanHandle(sd));
        if (stable != 1) {
            fail("assertPointersStable", context,
                    "buffer pointers are NOT stable (result=" + stable + ")", null);
        }
    }

    /**
     * Assert the frozen execution count has reached at least N.
     */
    public static void assertFrozenExecCountAtLeast(SameDiff sd, int minCount) {
        assertFrozenExecCountAtLeast(sd, minCount, null);
    }

    public static void assertFrozenExecCountAtLeast(SameDiff sd, int minCount, String context) {
        int actual = getNativeOps().getPlanFrozenExecutionCount(getPlanHandle(sd));
        if (actual < minCount) {
            fail("assertFrozenExecCountAtLeast", context,
                    "expected frozen exec count >= " + minCount + " but was " + actual
                            + " (-1 means not frozen)", null);
        }
    }

    /**
     * Assert the plan's compilation has been sealed (no more recompiles allowed).
     */
    public static void assertCompilationSealed(SameDiff sd) {
        assertCompilationSealed(sd, null);
    }

    public static void assertCompilationSealed(SameDiff sd, String context) {
        int sealed = getNativeOps().isPlanCompilationSealed(getPlanHandle(sd));
        if (sealed != 1) {
            fail("assertCompilationSealed", context,
                    "compilation is NOT sealed (result=" + sealed + ")", null);
        }
    }

    /**
     * Assert zero mid-execution recompilations happened after seal.
     * Any non-zero value means the plan was recompiled when it shouldn't have been.
     */
    public static void assertNoMidExecutionRecompiles(SameDiff sd) {
        assertNoMidExecutionRecompiles(sd, null);
    }

    public static void assertNoMidExecutionRecompiles(SameDiff sd, String context) {
        long recompiles = getNativeOps().getPlanMidExecutionCompileCount(getPlanHandle(sd));
        if (recompiles > 0) {
            fail("assertNoMidExecutionRecompiles", context,
                    recompiles + " mid-execution recompile(s) detected after seal — "
                            + "this indicates the plan was invalidated and rebuilt during replay", null);
        }
    }

    /**
     * Assert the number of segments with captured CUDA graphs.
     */
    public static void assertCapturedGraphSegments(SameDiff sd, int expectedCount) {
        assertCapturedGraphSegments(sd, expectedCount, null);
    }

    public static void assertCapturedGraphSegments(SameDiff sd, int expectedCount, String context) {
        int actual = getNativeOps().getPlanNumCapturedGraphSegments(getPlanHandle(sd));
        if (actual != expectedCount) {
            fail("assertCapturedGraphSegments", context,
                    "expected " + expectedCount + " captured graph segments but was " + actual, null);
        }
    }

    /**
     * Assert at least N segments have captured graphs.
     */
    public static void assertCapturedGraphSegmentsAtLeast(SameDiff sd, int minCount) {
        assertCapturedGraphSegmentsAtLeast(sd, minCount, null);
    }

    public static void assertCapturedGraphSegmentsAtLeast(SameDiff sd, int minCount, String context) {
        int actual = getNativeOps().getPlanNumCapturedGraphSegments(getPlanHandle(sd));
        if (actual < minCount) {
            fail("assertCapturedGraphSegmentsAtLeast", context,
                    "expected >= " + minCount + " captured graph segments but was " + actual, null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Per-Segment Replay Detail Assertions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the segment's replay count (number of graph replays) is at least N.
     * This is separate from execution count — a segment can execute via slot-by-slot
     * without replaying. Replay count only increments when a captured graph is replayed.
     */
    public static void assertSegmentReplayCountAtLeast(SameDiff sd, int segmentIndex,
                                                        int minCount) {
        assertSegmentReplayCountAtLeast(sd, segmentIndex, minCount, null);
    }

    public static void assertSegmentReplayCountAtLeast(SameDiff sd, int segmentIndex,
                                                        int minCount, String context) {
        int actual = getNativeOps().getPlanSegmentReplayCount(getPlanHandle(sd), segmentIndex);
        if (actual < minCount) {
            fail("assertSegmentReplayCountAtLeast", context,
                    "segment " + segmentIndex + ": expected replay count >= " + minCount
                            + " but was " + actual, null);
        }
    }

    /**
     * Assert the segment replay state.
     * States: 0=EMPTY, 1=CAPTURING, 2=CAPTURED, 3=READY, 4=ERROR
     */
    public static final int REPLAY_STATE_EMPTY = 0;
    public static final int REPLAY_STATE_CAPTURING = 1;
    public static final int REPLAY_STATE_CAPTURED = 2;
    public static final int REPLAY_STATE_READY = 3;
    public static final int REPLAY_STATE_ERROR = 4;

    public static void assertSegmentReplayState(SameDiff sd, int segmentIndex, int expectedState) {
        assertSegmentReplayState(sd, segmentIndex, expectedState, null);
    }

    public static void assertSegmentReplayState(SameDiff sd, int segmentIndex, int expectedState,
                                                 String context) {
        int actual = getNativeOps().getPlanSegmentReplayState(getPlanHandle(sd), segmentIndex);
        if (actual != expectedState) {
            fail("assertSegmentReplayState", context,
                    "segment " + segmentIndex + ": expected replay state "
                            + replayStateName(expectedState) + " but was " + replayStateName(actual), null);
        }
    }

    /**
     * Assert the segment is capturable (can be captured as a CUDA graph).
     */
    public static void assertSegmentCapturable(SameDiff sd, int segmentIndex) {
        assertSegmentCapturable(sd, segmentIndex, null);
    }

    public static void assertSegmentCapturable(SameDiff sd, int segmentIndex, String context) {
        boolean capturable = getNativeOps().isPlanSegmentCapturable(getPlanHandle(sd), segmentIndex);
        if (!capturable) {
            fail("assertSegmentCapturable", context,
                    "segment " + segmentIndex + " is NOT capturable", null);
        }
    }

    /**
     * Assert the segment capture did NOT fail.
     */
    public static void assertSegmentCaptureNotFailed(SameDiff sd, int segmentIndex) {
        assertSegmentCaptureNotFailed(sd, segmentIndex, null);
    }

    public static void assertSegmentCaptureNotFailed(SameDiff sd, int segmentIndex, String context) {
        boolean failed = getNativeOps().isPlanSegmentCaptureFailed(getPlanHandle(sd), segmentIndex);
        if (failed) {
            fail("assertSegmentCaptureNotFailed", context,
                    "segment " + segmentIndex + " capture FAILED", null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Non-Asserting Queries — all plan-level state
    // ═══════════════════════════════════════════════════════════════════════

    /** Total CUDA graph replays across all segments. */
    public static int getTotalGraphReplays(SameDiff sd) {
        return getNativeOps().getPlanTotalGraphReplays(getPlanHandle(sd));
    }

    /** Whether buffer pointers are stable. 1=stable, 0=not, -1=invalid. */
    public static int getPointersStable(SameDiff sd) {
        return getNativeOps().getPlanPointersStable(getPlanHandle(sd));
    }

    /** Executions since shapes were frozen. -1 if not frozen. */
    public static int getFrozenExecCount(SameDiff sd) {
        return getNativeOps().getPlanFrozenExecutionCount(getPlanHandle(sd));
    }

    /** Whether compilation is sealed. 1=sealed, 0=not, -1=invalid. */
    public static int isCompilationSealed(SameDiff sd) {
        return getNativeOps().isPlanCompilationSealed(getPlanHandle(sd));
    }

    /** Mid-execution recompiles after seal. 0 is healthy. */
    public static long getMidExecutionRecompileCount(SameDiff sd) {
        return getNativeOps().getPlanMidExecutionCompileCount(getPlanHandle(sd));
    }

    /** Number of segments with captured graphs. */
    public static int getCapturedGraphSegmentCount(SameDiff sd) {
        return getNativeOps().getPlanNumCapturedGraphSegments(getPlanHandle(sd));
    }

    /** Per-segment replay count (graph replays, not total executions). */
    public static int getSegmentReplayCount(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentReplayCount(getPlanHandle(sd), segmentIndex);
    }

    /** Per-segment replay state (0=EMPTY..4=ERROR). */
    public static int getSegmentReplayState(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentReplayState(getPlanHandle(sd), segmentIndex);
    }

    /** Per-segment capturable flag. */
    public static boolean isSegmentCapturable(SameDiff sd, int segmentIndex) {
        return getNativeOps().isPlanSegmentCapturable(getPlanHandle(sd), segmentIndex);
    }

    /** Per-segment capture failed flag. */
    public static boolean isSegmentCaptureFailed(SameDiff sd, int segmentIndex) {
        return getNativeOps().isPlanSegmentCaptureFailed(getPlanHandle(sd), segmentIndex);
    }

    /** Per-segment monolithic handle readiness. 1=ready, 0=not, -1=invalid. */
    public static int isMonolithicHandleReady(SameDiff sd, int segmentIndex) {
        return getNativeOps().getPlanSegmentMonolithicHandleReady(getPlanHandle(sd), segmentIndex);
    }

    /** Per-segment island handle readiness. 1=ready, 0=not, -1=invalid. */
    public static int isIslandHandleReady(SameDiff sd, int segmentIndex, int islandIndex) {
        return getNativeOps().getPlanSegmentIslandHandleReady(getPlanHandle(sd), segmentIndex, islandIndex);
    }

    /**
     * Snapshot all queryable state for a segment into a human-readable string.
     * Useful for logging the full picture at a specific point in execution.
     */
    public static String snapshotSegmentState(SameDiff sd, int segmentIndex) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        StringBuilder sb = new StringBuilder();
        sb.append("seg[").append(segmentIndex).append("] {");
        sb.append(" execCount=").append(ops.getSegmentExecutionCount(handle, segmentIndex));
        sb.append(" replayCount=").append(ops.getPlanSegmentReplayCount(handle, segmentIndex));
        sb.append(" replayState=").append(replayStateName(ops.getPlanSegmentReplayState(handle, segmentIndex)));
        sb.append(" replayMode=").append(replayModeName(ops.getPlanSegmentReplayMode(handle, segmentIndex)));
        sb.append(" capturable=").append(ops.isPlanSegmentCapturable(handle, segmentIndex));
        sb.append(" captureFailed=").append(ops.isPlanSegmentCaptureFailed(handle, segmentIndex));
        sb.append(" gapUnits=").append(ops.getPlanSegmentGapUnitCount(handle, segmentIndex));
        sb.append(" islandUnits=").append(ops.getPlanSegmentIslandUnitCount(handle, segmentIndex));
        sb.append(" gapSlots=").append(ops.getPlanSegmentGapSlotCount(handle, segmentIndex));
        sb.append(" monolithicReady=").append(ops.getPlanSegmentMonolithicHandleReady(handle, segmentIndex));
        sb.append(" }");
        return sb.toString();
    }

    /**
     * Snapshot all queryable plan-level state.
     */
    public static String snapshotPlanState(SameDiff sd) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int numSegs = ops.getPlanSegmentCount(handle);
        StringBuilder sb = new StringBuilder();
        sb.append("plan {");
        sb.append(" phase=").append(ops.getPlanPhase(handle));
        sb.append(" segs=").append(numSegs);
        sb.append(" capturedSegs=").append(ops.getPlanNumCapturedGraphSegments(handle));
        sb.append(" totalReplays=").append(ops.getPlanTotalGraphReplays(handle));
        sb.append(" pointersStable=").append(ops.getPlanPointersStable(handle));
        sb.append(" frozenExecCount=").append(ops.getPlanFrozenExecutionCount(handle));
        sb.append(" sealed=").append(ops.isPlanCompilationSealed(handle));
        sb.append(" midExecRecompiles=").append(ops.getPlanMidExecutionCompileCount(handle));
        sb.append(" slots=").append(ops.getPlanNumSlots(handle));
        sb.append(" extInputs=").append(ops.getPlanNumExternalInputs(handle));
        sb.append(" varInputs=").append(ops.getPlanNumVariableExternalInputs(handle));
        sb.append(" stagingBufs=").append(ops.getPlanNumStagingBuffers(handle));
        sb.append(" }\n");
        for (int i = 0; i < numSegs; i++) {
            sb.append("  ").append(snapshotSegmentState(sd, i)).append("\n");
        }
        return sb.toString();
    }

    private static String replayStateName(int state) {
        switch (state) {
            case REPLAY_STATE_EMPTY: return "EMPTY";
            case REPLAY_STATE_CAPTURING: return "CAPTURING";
            case REPLAY_STATE_CAPTURED: return "CAPTURED";
            case REPLAY_STATE_READY: return "READY";
            case REPLAY_STATE_ERROR: return "ERROR";
            default: return "UNKNOWN(" + state + ")";
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // External input introspection
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that ext[extIdx] is classified as variable in the native plan.
     */
    public static void assertExtInputIsVariable(SameDiff sd, int extIdx) {
        assertExtInputIsVariable(sd, extIdx, null);
    }

    public static void assertExtInputIsVariable(SameDiff sd, int extIdx, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        boolean isVar = ops.getPlanIsExternalInputVariable(handle, extIdx);
        if (!isVar) {
            fail("assertExtInputIsVariable", context,
                    "ext[" + extIdx + "] is NOT variable — staging D2D will not run for this input",
                    null);
        }
    }

    /**
     * Assert that ext[extIdx] is classified as placeholder in the native plan.
     */
    public static void assertExtInputIsPlaceholder(SameDiff sd, int extIdx) {
        assertExtInputIsPlaceholder(sd, extIdx, null);
    }

    public static void assertExtInputIsPlaceholder(SameDiff sd, int extIdx, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        boolean isPh = ops.getPlanIsExternalInputPlaceholder(handle, extIdx);
        if (!isPh) {
            fail("assertExtInputIsPlaceholder", context,
                    "ext[" + extIdx + "] is NOT placeholder — H2D sync will not be forced",
                    null);
        }
    }

    /**
     * Assert that the plan has exactly the expected number of variable ext inputs.
     */
    public static void assertVariableExtInputCount(SameDiff sd, int expected) {
        assertVariableExtInputCount(sd, expected, null);
    }

    public static void assertVariableExtInputCount(SameDiff sd, int expected, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int actual = ops.getPlanNumVariableExternalInputs(handle);
        if (actual != expected) {
            fail("assertVariableExtInputCount", context,
                    "expected " + expected + " variable ext inputs but got " + actual,
                    null);
        }
    }

    /**
     * Assert that ext[extIdx] has an allocated staging buffer (non-zero device address).
     */
    public static void assertExtInputHasStagingBuffer(SameDiff sd, int extIdx) {
        assertExtInputHasStagingBuffer(sd, extIdx, null);
    }

    public static void assertExtInputHasStagingBuffer(SameDiff sd, int extIdx, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        long addr = ops.getPlanStagingBufferAddress(handle, extIdx);
        if (addr == 0) {
            fail("assertExtInputHasStagingBuffer", context,
                    "ext[" + extIdx + "] has NO staging buffer — D2D copy will skip this input. "
                            + "isVariable=" + ops.getPlanIsExternalInputVariable(handle, extIdx)
                            + " numStagingBuffers=" + ops.getPlanNumStagingBuffers(handle),
                    null);
        }
    }

    /**
     * Assert that the plan has at least minCount staging buffers allocated.
     */
    public static void assertStagingBufferCountAtLeast(SameDiff sd, int minCount) {
        assertStagingBufferCountAtLeast(sd, minCount, null);
    }

    public static void assertStagingBufferCountAtLeast(SameDiff sd, int minCount, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int actual = ops.getPlanNumStagingBuffers(handle);
        if (actual < minCount) {
            fail("assertStagingBufferCountAtLeast", context,
                    "expected at least " + minCount + " staging buffers but only " + actual
                            + " allocated. variableCount=" + ops.getPlanNumVariableExternalInputs(handle),
                    null);
        }
    }

    /**
     * Snapshot of ext input state for diagnostics.
     */
    public static String snapshotExtInputState(SameDiff sd) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int numExt = ops.getPlanNumExternalInputs(handle);
        int numVar = ops.getPlanNumVariableExternalInputs(handle);
        int numStaging = ops.getPlanNumStagingBuffers(handle);
        StringBuilder sb = new StringBuilder();
        sb.append("extInputState { total=").append(numExt)
                .append(" variable=").append(numVar)
                .append(" staging=").append(numStaging)
                .append(" }\n");
        for (int i = 0; i < numExt; i++) {
            boolean isVar = ops.getPlanIsExternalInputVariable(handle, i);
            boolean isPh = ops.getPlanIsExternalInputPlaceholder(handle, i);
            if (isVar || isPh) {
                long stagingAddr = ops.getPlanStagingBufferAddress(handle, i);
                long effectiveAddr = ops.getPlanEffectiveExternalAddress(handle, i);
                sb.append("  ext[").append(i).append("]:")
                        .append(" var=").append(isVar)
                        .append(" ph=").append(isPh)
                        .append(" staging=0x").append(Long.toHexString(stagingAddr))
                        .append(" effective=0x").append(Long.toHexString(effectiveAddr))
                        .append("\n");
            }
        }
        return sb.toString();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Internals
    // ═══════════════════════════════════════════════════════════════════════

    private static void requirePlanAvailable(DspDebugger.PlanReport report, String context) {
        if (report.errorMessage != null) {
            fail("plan availability", context,
                    "plan not compiled: " + report.errorMessage, report);
        }
    }

    /**
     * Get the native plan handle for direct NativeOps queries.
     * Public so tests can call NativeOps methods not wrapped by this class.
     */
    public static org.bytedeco.javacpp.Pointer getPlanHandleForQuery(SameDiff sd) {
        return getPlanHandle(sd);
    }

    private static org.bytedeco.javacpp.Pointer getPlanHandle(SameDiff sd) {
        var session = sd.getOrCreateSession();
        var executor = session.getDynamicShapePlanExecutor();
        if (executor == null) {
            throw new AssertionError("DspPlanAssertions: no DynamicShapePlanExecutor — "
                    + "call sd.output() at least once before asserting plan state");
        }
        var handle = executor.getNativePlanHandle();
        if (handle == null) {
            throw new AssertionError("DspPlanAssertions: native plan handle is null — "
                    + "plan has not been compiled yet");
        }
        return handle;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // D2D Copy Assertions (Q1, Q10)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that D2D copies fired for ALL variable ext inputs with staging buffers.
     * Detects the bug where staging buffer == original buffer (copy is a no-op)
     * or where staging was never allocated.
     */
    public static void assertAllD2DCopiesFired(SameDiff sd) {
        assertAllD2DCopiesFired(sd, null);
    }

    public static void assertAllD2DCopiesFired(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) {
            fail("assertAllD2DCopiesFired", context, "plan not compiled", null);
        }
        DspHandle.StepSnapshot snap = h.captureStepSnapshot();
        List<Integer> failed = snap.failedD2DExtIndices();
        if (!failed.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append(failed.size()).append(" variable ext input(s) did NOT get D2D copy:\n");
            for (int idx : failed) {
                DspHandle.D2DStatus s = snap.d2dStatusByExtIdx.get(idx);
                sb.append("  ").append(s).append("\n");
            }
            fail("assertAllD2DCopiesFired", context, sb.toString(), null);
        }
    }

    /**
     * Assert no variable ext input has address drift (staging != effective).
     * This detects the case where the CUDA graph was captured with one address
     * but is replaying against a different one.
     */
    public static void assertNoStagingAddressDrift(SameDiff sd) {
        assertNoStagingAddressDrift(sd, null);
    }

    public static void assertNoStagingAddressDrift(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        DspHandle.StepSnapshot snap = h.captureStepSnapshot();
        List<Integer> drifting = snap.driftingExtIndices();
        if (!drifting.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append(drifting.size()).append(" ext input(s) with address drift:\n");
            for (int idx : drifting) {
                DspHandle.D2DStatus s = snap.d2dStatusByExtIdx.get(idx);
                sb.append("  ").append(s).append("\n");
            }
            fail("assertNoStagingAddressDrift", context, sb.toString(), null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Pointer Stability Assertions (Q2)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert no segment has pointer drift (captured addresses == current addresses).
     * Checks the per-segment tracked pointers JSON for any mismatch.
     */
    public static void assertNoAddressDrift(SameDiff sd) {
        assertNoAddressDrift(sd, null);
    }

    public static void assertNoAddressDrift(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        Map<Integer, Boolean> matches = h.allSegmentsPointersMatch();
        List<Integer> drifting = new ArrayList<>();
        for (Map.Entry<Integer, Boolean> e : matches.entrySet()) {
            if (!e.getValue()) drifting.add(e.getKey());
        }
        if (!drifting.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append(drifting.size()).append(" segment(s) with pointer drift:\n");
            for (int segIdx : drifting) {
                sb.append("  seg[").append(segIdx).append("]: ")
                        .append(h.segmentTrackedPointersJson(segIdx)).append("\n");
            }
            fail("assertNoAddressDrift", context, sb.toString(), null);
        }
    }

    /**
     * Assert a specific segment has matching tracked pointers.
     */
    public static void assertSegmentPointersMatch(SameDiff sd, int segIdx) {
        assertSegmentPointersMatch(sd, segIdx, null);
    }

    public static void assertSegmentPointersMatch(SameDiff sd, int segIdx, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        String json = h.segmentTrackedPointersJson(segIdx);
        if (json != null && json.contains("\"match\":false")) {
            fail("assertSegmentPointersMatch", context,
                    "segment " + segIdx + " has pointer drift: " + json, null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Output Validity Assertions (Q6, Q7)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert all outputs are valid (no NaN, Inf, null, or all-zero).
     * Gate behind isDebug — calls dspValidateOutputs which requires sync-to-host.
     */
    public static void assertOutputsValid(SameDiff sd) {
        assertOutputsValid(sd, null);
    }

    public static void assertOutputsValid(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        int[] flags = h.validateOutputs();
        List<String> issues = new ArrayList<>();
        for (int i = 0; i < flags.length; i++) {
            if (flags[i] != 0) {
                issues.add("output[" + i + "] flags=0x" + Integer.toHexString(flags[i])
                        + " (" + describeOutputFlags(flags[i]) + ")");
            }
        }
        if (!issues.isEmpty()) {
            fail("assertOutputsValid", context,
                    issues.size() + " output(s) with issues: " + issues, null);
        }
    }

    /**
     * Assert no stale outputs detected (norm diff above epsilon).
     * Compares current norms against prevNorms, updates prevNorms in-place.
     */
    public static void assertNoStaleOutputs(SameDiff sd, float[] prevNorms,
                                             boolean[] staleOut, float epsilon) {
        assertNoStaleOutputs(sd, prevNorms, staleOut, epsilon, null);
    }

    public static void assertNoStaleOutputs(SameDiff sd, float[] prevNorms,
                                             boolean[] staleOut, float epsilon, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        int staleCount = h.detectStaleOutputs(prevNorms, staleOut, epsilon);
        if (staleCount > 0) {
            List<Integer> staleIndices = new ArrayList<>();
            for (int i = 0; i < staleOut.length; i++) {
                if (staleOut[i]) staleIndices.add(i);
            }
            fail("assertNoStaleOutputs", context,
                    staleCount + " stale output(s) detected at indices: " + staleIndices, null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Capture Quality Assertions (Q8, Q9)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert zero permanent capture failures.
     */
    public static void assertZeroPermCaptureFailures(SameDiff sd) {
        assertZeroPermCaptureFailures(sd, null);
    }

    public static void assertZeroPermCaptureFailures(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        DspHandle.CaptureStats cs = h.parsedCaptureStats();
        if (cs.permFailed > 0) {
            fail("assertZeroPermCaptureFailures", context,
                    cs.permFailed + " permanent capture failure(s): " + cs, null);
        }
    }

    /**
     * Assert no ops escaped CUDA graph capture (all ops captured).
     */
    public static void assertNoHostOnlyOps(SameDiff sd) {
        assertNoHostOnlyOps(sd, null);
    }

    public static void assertNoHostOnlyOps(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        int count = h.numHostOnlyOps();
        if (count > 0) {
            String names = h.hostOnlyOpNames();
            fail("assertNoHostOnlyOps", context,
                    count + " op(s) escaped CUDA graph capture: " + names, null);
        }
    }

    /**
     * Assert captured graph covers all ops.
     */
    public static void assertCaptureComplete(SameDiff sd) {
        assertCaptureComplete(sd, null);
    }

    public static void assertCaptureComplete(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return;
        if (!h.isCaptureComplete()) {
            fail("assertCaptureComplete", context,
                    "capture incomplete — host-only ops: " + h.hostOnlyOpNames()
                            + " stats: " + h.captureStats(), null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // StepSnapshot Assertions
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that the plan progressed between two step snapshots.
     * Checks: executeCount incremented, no new drifting segments, no D2D regressions.
     */
    public static void assertStepProgressed(DspHandle.StepSnapshot before,
                                             DspHandle.StepSnapshot after) {
        assertStepProgressed(before, after, null);
    }

    public static void assertStepProgressed(DspHandle.StepSnapshot before,
                                             DspHandle.StepSnapshot after, String context) {
        if (after.executeCount <= before.executeCount) {
            fail("assertStepProgressed", context,
                    "executeCount did not advance: " + before.executeCount + " -> " + after.executeCount,
                    null);
        }
        if (after.driftingSegments.size() > before.driftingSegments.size()) {
            fail("assertStepProgressed", context,
                    "new pointer drift detected: " + before.driftingSegments + " -> " + after.driftingSegments,
                    null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Full State Snapshot
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Snapshot full plan + segment + capture + D2D state as a human-readable string.
     * Combines snapshotPlanState with the new introspection APIs.
     */
    public static String snapshotFullState(SameDiff sd) {
        DspHandle h = sd.dsp();
        if (!h.isCompiled()) return "Plan not compiled";

        StringBuilder sb = new StringBuilder();
        sb.append(snapshotPlanState(sd)).append("\n");

        // Capture quality
        DspHandle.CaptureStats cs = h.parsedCaptureStats();
        sb.append("Capture: ").append(cs).append("\n");
        int hostOps = h.numHostOnlyOps();
        if (hostOps > 0) {
            sb.append("Host-only ops (").append(hostOps).append("): ")
                    .append(h.hostOnlyOpNames()).append("\n");
        }

        // D2D state
        DspHandle.StepSnapshot snap = h.captureStepSnapshot();
        sb.append("D2D: ").append(snap.d2dStatusByExtIdx.size()).append(" variable inputs, ");
        long fired = snap.d2dStatusByExtIdx.values().stream()
                .filter(s -> s.fired).count();
        sb.append(fired).append(" fired");
        if (!snap.driftingExtIndices().isEmpty()) {
            sb.append(", DRIFT at ext").append(snap.driftingExtIndices());
        }
        sb.append("\n");

        // Pointer drift
        if (!snap.driftingSegments.isEmpty()) {
            sb.append("POINTER DRIFT in segments: ").append(snap.driftingSegments).append("\n");
        }

        return sb.toString();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Arg generation & refresh assertions (Task 1)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that a segment's arg table is fresh — current generation matches captured generation.
     * A mismatch means the segment's CUDA graph was captured with stale arguments.
     */
    public static void assertArgTableFresh(SameDiff sd, int segIdx) {
        assertArgTableFresh(sd, segIdx, null);
    }

    public static void assertArgTableFresh(SameDiff sd, int segIdx, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        long current = ops.getPlanSegmentArgGeneration(handle, segIdx);
        long captured = ops.getPlanSegmentCapturedArgGeneration(handle, segIdx);
        if (current != captured) {
            fail("assertArgTableFresh", context,
                    "segment[" + segIdx + "] arg generation mismatch: current=" + current
                            + " captured=" + captured + " (delta=" + (current - captured) + ")",
                    null);
        }
    }

    /**
     * Assert no segment has arg generation drift (current != captured).
     * Call after warmup to verify all segments have consistent arg state.
     */
    public static void assertNoArgDrift(SameDiff sd) {
        assertNoArgDrift(sd, null);
    }

    public static void assertNoArgDrift(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        int segCount = h.numSegments();
        List<String> drifted = new ArrayList<>();
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        for (int i = 0; i < segCount; i++) {
            long current = ops.getPlanSegmentArgGeneration(handle, i);
            long captured = ops.getPlanSegmentCapturedArgGeneration(handle, i);
            if (current != captured) {
                drifted.add("seg[" + i + "] current=" + current + " captured=" + captured);
            }
        }
        if (!drifted.isEmpty()) {
            fail("assertNoArgDrift", context,
                    drifted.size() + " segment(s) have arg generation drift: " + drifted,
                    null);
        }
    }

    /**
     * Assert no segment needs arg refresh. If any segment reports needsArgRefresh=true,
     * the replay may use stale data.
     */
    public static void assertArgRefreshNotNeeded(SameDiff sd) {
        assertArgRefreshNotNeeded(sd, null);
    }

    public static void assertArgRefreshNotNeeded(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        int segCount = h.numSegments();
        List<Integer> needRefresh = new ArrayList<>();
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        for (int i = 0; i < segCount; i++) {
            if (ops.getPlanSegmentNeedsArgRefresh(handle, i) == 1) {
                needRefresh.add(i);
            }
        }
        if (!needRefresh.isEmpty()) {
            fail("assertArgRefreshNotNeeded", context,
                    needRefresh.size() + " segment(s) need arg refresh: " + needRefresh,
                    null);
        }
    }

    /** Get the current arg generation for a segment (non-asserting query). */
    public static long getArgGeneration(SameDiff sd, int segIdx) {
        return getNativeOps().getPlanSegmentArgGeneration(getPlanHandle(sd), segIdx);
    }

    /** Get the captured arg generation for a segment (non-asserting query). */
    public static long getCapturedArgGeneration(SameDiff sd, int segIdx) {
        return getNativeOps().getPlanSegmentCapturedArgGeneration(getPlanHandle(sd), segIdx);
    }

    /** Get the captured input address key for a segment (non-asserting query). */
    public static long getCapturedInputAddrKey(SameDiff sd, int segIdx) {
        return getNativeOps().getPlanSegmentCapturedInputAddrKey(getPlanHandle(sd), segIdx);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Backend selection assertions (Task 2)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert a specific segment was compiled with the expected backend.
     * @param expectedBackend e.g. "triton", "cublas", "native"
     */
    public static void assertSegmentBackend(SameDiff sd, int segIdx, String expectedBackend) {
        assertSegmentBackend(sd, segIdx, expectedBackend, null);
    }

    public static void assertSegmentBackend(SameDiff sd, int segIdx, String expectedBackend, String context) {
        DspHandle h = sd.dsp();
        String actual = h.segmentBackendName(segIdx);
        if (actual == null || !actual.equalsIgnoreCase(expectedBackend)) {
            fail("assertSegmentBackend", context,
                    "segment[" + segIdx + "] backend: expected '" + expectedBackend
                            + "' but was '" + actual + "'",
                    null);
        }
    }

    /**
     * Assert all segments were compiled with the expected backend.
     */
    public static void assertAllSegmentsCompiledWith(SameDiff sd, String expectedBackend) {
        assertAllSegmentsCompiledWith(sd, expectedBackend, null);
    }

    public static void assertAllSegmentsCompiledWith(SameDiff sd, String expectedBackend, String context) {
        DspHandle h = sd.dsp();
        int segCount = h.numSegments();
        List<String> mismatched = new ArrayList<>();
        for (int i = 0; i < segCount; i++) {
            String actual = h.segmentBackendName(i);
            if (actual == null || !actual.equalsIgnoreCase(expectedBackend)) {
                mismatched.add("seg[" + i + "]=" + actual);
            }
        }
        if (!mismatched.isEmpty()) {
            fail("assertAllSegmentsCompiledWith", context,
                    mismatched.size() + " segment(s) not compiled with '" + expectedBackend
                            + "': " + mismatched,
                    null);
        }
    }

    /**
     * Assert no segment is executing in SLOT_BY_SLOT fallback mode.
     * Checks that every segment has a replay mode > 0 (i.e., not slot-by-slot).
     */
    public static void assertNoSlotBySlotFallback(SameDiff sd) {
        assertNoSlotBySlotFallback(sd, null);
    }

    public static void assertNoSlotBySlotFallback(SameDiff sd, String context) {
        DspHandle h = sd.dsp();
        int segCount = h.numSegments();
        List<Integer> fallbackSegs = new ArrayList<>();
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        for (int i = 0; i < segCount; i++) {
            int replayMode = ops.getPlanSegmentReplayMode(handle, i);
            if (replayMode == 0) {
                fallbackSegs.add(i);
            }
        }
        if (!fallbackSegs.isEmpty()) {
            fail("assertNoSlotBySlotFallback", context,
                    fallbackSegs.size() + " segment(s) in slot-by-slot fallback: " + fallbackSegs,
                    null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // KV cache position assertions (Task 3)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the KV cache position equals the expected value.
     */
    public static void assertKvCachePositionEquals(SameDiff sd, long expected) {
        assertKvCachePositionEquals(sd, expected, null);
    }

    public static void assertKvCachePositionEquals(SameDiff sd, long expected, String context) {
        DspHandle h = sd.dsp();
        long actual = h.kvCachePosition();
        if (actual != expected) {
            fail("assertKvCachePositionEquals", context,
                    "KV cache position: expected " + expected + " but was " + actual,
                    null);
        }
    }

    /**
     * Assert the KV cache position has advanced from the given previous value.
     * @param previous the position before the decode step(s)
     */
    public static void assertKvCachePositionAdvanced(SameDiff sd, long previous) {
        assertKvCachePositionAdvanced(sd, previous, null);
    }

    public static void assertKvCachePositionAdvanced(SameDiff sd, long previous, String context) {
        DspHandle h = sd.dsp();
        long current = h.kvCachePosition();
        if (current <= previous) {
            fail("assertKvCachePositionAdvanced", context,
                    "KV cache position did not advance: was " + previous + ", now " + current,
                    null);
        }
    }

    /** Get the current KV cache position (non-asserting query). */
    public static long getKvCachePosition(SameDiff sd) {
        return sd.dsp().kvCachePosition();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Slot flag decoded assertions (Task 4)
    // ═══════════════════════════════════════════════════════════════════════

    // Slot flag bit positions (from NativeDynamicShapePlan.h SlotFlags)
    public static final int FLAG_VIEW_CAPABLE = 0;
    public static final int FLAG_DATA_DEPENDENT = 1;
    public static final int FLAG_SHAPE_DEPENDS_ON_VALUES = 2;
    public static final int FLAG_IDENTITY = 3;
    public static final int FLAG_IN_PLACE_FUSED = 4;
    public static final int FLAG_FUSED_CHAIN_HEAD = 5;
    public static final int FLAG_FUSED_CHAIN_TAIL = 6;
    public static final int FLAG_NEEDS_ZEROED = 7;
    public static final int FLAG_NEEDS_INT_LONG_SYNC = 8;
    public static final int FLAG_SHAPE_STATIC = 9;
    public static final int FLAG_FROZEN_CONSTANT = 10;

    /**
     * Assert that a specific slot has a given trait flag set.
     * @param flagBit one of FLAG_* constants
     */
    public static void assertSlotHasTrait(SameDiff sd, int slotIdx, int flagBit) {
        assertSlotHasTrait(sd, slotIdx, flagBit, null);
    }

    public static void assertSlotHasTrait(SameDiff sd, int slotIdx, int flagBit, String context) {
        int flags = getNativeOps().getPlanSlotFlags(getPlanHandle(sd), slotIdx);
        if ((flags & (1 << flagBit)) == 0) {
            fail("assertSlotHasTrait", context,
                    "slot[" + slotIdx + "] expected flag bit " + flagBit + " ("
                            + flagBitName(flagBit) + ") set, but flags=0x"
                            + Integer.toHexString(flags) + " (" + decodeSlotFlags(flags) + ")",
                    null);
        }
    }

    /**
     * Assert a slot does NOT have the DATA_DEPENDENT flag.
     * Data-dependent slots require value-based shape computation — expensive.
     */
    public static void assertSlotNotDataDependent(SameDiff sd, int slotIdx) {
        assertSlotNotDataDependent(sd, slotIdx, null);
    }

    public static void assertSlotNotDataDependent(SameDiff sd, int slotIdx, String context) {
        int flags = getNativeOps().getPlanSlotFlags(getPlanHandle(sd), slotIdx);
        if ((flags & (1 << FLAG_DATA_DEPENDENT)) != 0) {
            String opName = getNativeOps().getPlanSlotOpName(getPlanHandle(sd), slotIdx);
            fail("assertSlotNotDataDependent", context,
                    "slot[" + slotIdx + "] (op=" + opName
                            + ") has DATA_DEPENDENT flag set, flags=0x"
                            + Integer.toHexString(flags) + " (" + decodeSlotFlags(flags) + ")",
                    null);
        }
    }

    /**
     * Assert a slot is in the FROZEN_CONSTANT state (slotState == 3).
     */
    public static void assertSlotIsFrozenConstant(SameDiff sd, int slotIdx) {
        assertSlotIsFrozenConstant(sd, slotIdx, null);
    }

    public static void assertSlotIsFrozenConstant(SameDiff sd, int slotIdx, String context) {
        int state = getNativeOps().getPlanSlotState(getPlanHandle(sd), slotIdx);
        if (state != 3) { // FROZEN_CONSTANT = 3
            fail("assertSlotIsFrozenConstant", context,
                    "slot[" + slotIdx + "] state=" + state + " ("
                            + slotStateName(state) + "), expected FROZEN_CONSTANT (3)",
                    null);
        }
    }

    /**
     * Assert a slot has isDynamicShape set (its shape varies across executions).
     */
    public static void assertSlotIsDynamicShape(SameDiff sd, int slotIdx) {
        assertSlotIsDynamicShape(sd, slotIdx, null);
    }

    public static void assertSlotIsDynamicShape(SameDiff sd, int slotIdx, String context) {
        int flags = getNativeOps().getPlanSlotFlags(getPlanHandle(sd), slotIdx);
        // isDynamicShape is NOT a bitmask flag — it's a separate bool.
        // Check that SHAPE_STATIC is NOT set (bit 9).
        if ((flags & (1 << FLAG_SHAPE_STATIC)) != 0) {
            fail("assertSlotIsDynamicShape", context,
                    "slot[" + slotIdx + "] has SHAPE_STATIC flag set (not dynamic), flags=0x"
                            + Integer.toHexString(flags) + " (" + decodeSlotFlags(flags) + ")",
                    null);
        }
    }

    /**
     * Assert no slot in the plan has the FUSED_CHAIN_TAIL flag without a
     * corresponding FUSED_CHAIN_HEAD. This catches dangling tail regressions.
     */
    public static void assertNoFusionDanglingTails(SameDiff sd) {
        assertNoFusionDanglingTails(sd, null);
    }

    public static void assertNoFusionDanglingTails(SameDiff sd, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int slotCount = ops.getTotalPlanOutputSlots(handle);
        boolean headSeen = false;
        List<String> danglingTails = new ArrayList<>();
        for (int i = 0; i < slotCount; i++) {
            int flags = ops.getPlanSlotFlags(handle, i);
            if ((flags & (1 << FLAG_FUSED_CHAIN_HEAD)) != 0) {
                headSeen = true;
            }
            if ((flags & (1 << FLAG_FUSED_CHAIN_TAIL)) != 0) {
                if (!headSeen) {
                    String opName = ops.getPlanSlotOpName(handle, i);
                    danglingTails.add("slot[" + i + "] op=" + opName);
                }
                headSeen = false;
            }
        }
        if (!danglingTails.isEmpty()) {
            fail("assertNoFusionDanglingTails", context,
                    danglingTails.size() + " FUSED_CHAIN_TAIL slot(s) without HEAD: "
                            + danglingTails,
                    null);
        }
    }

    /** Get decoded slot flags as a human-readable string (non-asserting query). */
    public static String getSlotFlagsDecoded(SameDiff sd, int slotIdx) {
        int flags = getNativeOps().getPlanSlotFlags(getPlanHandle(sd), slotIdx);
        return decodeSlotFlags(flags);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Slot generation assertions (Task 8)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert a slot's write generation has advanced from a previous value.
     * This detects slots that are stuck (never written to during execution).
     * @param previousGeneration the generation counter before the execution step(s)
     */
    public static void assertSlotGenerationAdvanced(SameDiff sd, int slotIdx, int previousGeneration) {
        assertSlotGenerationAdvanced(sd, slotIdx, previousGeneration, null);
    }

    public static void assertSlotGenerationAdvanced(SameDiff sd, int slotIdx,
                                                     int previousGeneration, String context) {
        int current = getNativeOps().getPlanSlotGeneration(getPlanHandle(sd), slotIdx);
        if (current <= previousGeneration) {
            String opName = getNativeOps().getPlanSlotOpName(getPlanHandle(sd), slotIdx);
            fail("assertSlotGenerationAdvanced", context,
                    "slot[" + slotIdx + "] (op=" + opName + ") generation did not advance: was "
                            + previousGeneration + ", now " + current,
                    null);
        }
    }

    /**
     * Assert no non-constant slot is stuck (generation == 0 after execution).
     * Constant slots (FROZEN_CONSTANT state) are expected to have generation 0 after freeze.
     */
    public static void assertNoStuckSlots(SameDiff sd) {
        assertNoStuckSlots(sd, null);
    }

    public static void assertNoStuckSlots(SameDiff sd, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int slotCount = ops.getTotalPlanOutputSlots(handle);
        List<String> stuck = new ArrayList<>();
        for (int i = 0; i < slotCount; i++) {
            int gen = ops.getPlanSlotGeneration(handle, i);
            int state = ops.getPlanSlotState(handle, i);
            // FROZEN_CONSTANT (state 3) slots may legitimately have gen 0
            if (gen == 0 && state != 3) {
                String opName = ops.getPlanSlotOpName(handle, i);
                stuck.add("slot[" + i + "] op=" + opName + " state=" + slotStateName(state));
            }
        }
        if (!stuck.isEmpty()) {
            fail("assertNoStuckSlots", context,
                    stuck.size() + " non-constant slot(s) have generation 0 (never written): "
                            + stuck,
                    null);
        }
    }

    /** Get the current slot generation counter (non-asserting query). */
    public static int getSlotGeneration(SameDiff sd, int slotIdx) {
        return getNativeOps().getPlanSlotGeneration(getPlanHandle(sd), slotIdx);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Replay cache assertions (Task 5)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the replay cache hit rate is above the given threshold.
     * @param minRate minimum hit rate (0.0 to 1.0)
     */
    public static void assertReplayCacheHitRateAbove(SameDiff sd, double minRate) {
        assertReplayCacheHitRateAbove(sd, minRate, null);
    }

    public static void assertReplayCacheHitRateAbove(SameDiff sd, double minRate, String context) {
        var ops = getNativeOps();
        int hits = ops.getReplayCacheHits();
        int misses = ops.getReplayCacheMisses();
        int total = hits + misses;
        if (total == 0) {
            fail("assertReplayCacheHitRateAbove", context,
                    "replay cache has 0 total lookups (not exercised)", null);
        }
        double rate = (double) hits / total;
        if (rate < minRate) {
            fail("assertReplayCacheHitRateAbove", context,
                    "replay cache hit rate " + String.format("%.2f", rate)
                            + " < minimum " + String.format("%.2f", minRate)
                            + " (hits=" + hits + " misses=" + misses + ")",
                    null);
        }
    }

    /**
     * Assert the replay cache has zero misses after warmup.
     * Call this after warmup phase is complete — any miss means a shape key collision
     * or plan recompilation.
     */
    public static void assertNoReplayCacheMissesAfterWarmup(SameDiff sd) {
        assertNoReplayCacheMissesAfterWarmup(sd, null);
    }

    public static void assertNoReplayCacheMissesAfterWarmup(SameDiff sd, String context) {
        var ops = getNativeOps();
        int misses = ops.getReplayCacheMisses();
        if (misses > 0) {
            int hits = ops.getReplayCacheHits();
            fail("assertNoReplayCacheMissesAfterWarmup", context,
                    "replay cache has " + misses + " miss(es) after warmup (hits=" + hits + ")",
                    null);
        }
    }

    /** Get the replay cache hit count (non-asserting query). */
    public static int getReplayCacheHits(SameDiff sd) {
        return getNativeOps().getReplayCacheHits();
    }

    /** Get the replay cache miss count (non-asserting query). */
    public static int getReplayCacheMisses(SameDiff sd) {
        return getNativeOps().getReplayCacheMisses();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Frozen plan build pass assertions (Task 6)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert the frozen plan is fully sealed (build pass count >= 2).
     * A build pass count of 0 means the plan hasn't warmed up,
     * 1 means it hasn't compiled, 2+ means it's sealed and ready for replay.
     */
    public static void assertFrozenPlanFullySealed(SameDiff sd) {
        assertFrozenPlanFullySealed(sd, null);
    }

    public static void assertFrozenPlanFullySealed(SameDiff sd, String context) {
        int passCount = getNativeOps().getFrozenPlanBuildPassCount(getPlanHandle(sd));
        if (passCount < 2) {
            fail("assertFrozenPlanFullySealed", context,
                    "frozen plan build pass count = " + passCount
                            + " (expected >= 2 for sealed); "
                            + (passCount == 0 ? "not warmed up" : "not compiled"),
                    null);
        }
    }

    /**
     * Assert the frozen plan has completed at least the given number of build passes.
     */
    public static void assertFrozenPlanBuildPassCountAtLeast(SameDiff sd, int minPasses) {
        assertFrozenPlanBuildPassCountAtLeast(sd, minPasses, null);
    }

    public static void assertFrozenPlanBuildPassCountAtLeast(SameDiff sd, int minPasses, String context) {
        int passCount = getNativeOps().getFrozenPlanBuildPassCount(getPlanHandle(sd));
        if (passCount < minPasses) {
            fail("assertFrozenPlanBuildPassCountAtLeast", context,
                    "frozen plan build pass count = " + passCount
                            + " (expected >= " + minPasses + ")",
                    null);
        }
    }

    /** Get the frozen plan build pass count (non-asserting query). */
    public static int getFrozenPlanBuildPassCount(SameDiff sd) {
        return getNativeOps().getFrozenPlanBuildPassCount(getPlanHandle(sd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Per-execution segment stats assertions (Task 9)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert all segments replayed during the last execute() (no warmup/capture/fallback).
     */
    public static void assertAllSegmentsReplayed(SameDiff sd) {
        assertAllSegmentsReplayed(sd, null);
    }

    public static void assertAllSegmentsReplayed(SameDiff sd, String context) {
        var handle = getPlanHandle(sd);
        var ops = getNativeOps();
        int replayed = ops.getLastExecSegmentsReplayed(handle);
        int total = ops.getLastExecSegmentsTotal(handle);
        if (total < 0) {
            fail("assertAllSegmentsReplayed", context,
                    "no execution context available (plan never executed in steady state)", null);
        }
        if (replayed != total) {
            int warmup = ops.getLastExecSegmentsWarmup(handle);
            int captured = ops.getLastExecSegmentsCaptured(handle);
            int slotBySlot = ops.getLastExecSegmentsSlotBySlot(handle);
            int failed = ops.getLastExecSegmentsFailed(handle);
            fail("assertAllSegmentsReplayed", context,
                    "only " + replayed + "/" + total + " segments replayed"
                            + " (warmup=" + warmup + " captured=" + captured
                            + " slotBySlot=" + slotBySlot + " failed=" + failed + ")",
                    null);
        }
    }

    /**
     * Assert no segment failed during the last execute().
     */
    public static void assertNoSegmentFailures(SameDiff sd) {
        assertNoSegmentFailures(sd, null);
    }

    public static void assertNoSegmentFailures(SameDiff sd, String context) {
        int failed = getNativeOps().getLastExecSegmentsFailed(getPlanHandle(sd));
        if (failed > 0) {
            fail("assertNoSegmentFailures", context,
                    failed + " segment(s) failed during last execute()", null);
        }
    }

    /** Get per-execution segment stats (non-asserting queries). */
    public static int getLastExecSegmentsReplayed(SameDiff sd) {
        return getNativeOps().getLastExecSegmentsReplayed(getPlanHandle(sd));
    }
    public static int getLastExecSegmentsTotal(SameDiff sd) {
        return getNativeOps().getLastExecSegmentsTotal(getPlanHandle(sd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stream sync level assertions (Task 12)
    // ═══════════════════════════════════════════════════════════════════════

    /** Sync level constants matching C++ SyncLevel enum. */
    public static final int SYNC_NONE = 0;
    public static final int SYNC_EVENT = 1;
    public static final int SYNC_STREAM = 2;
    public static final int SYNC_FULL_DEVICE = 3;

    /**
     * Assert the last execute() did not use full device synchronization.
     * Full device sync is expensive (~1.4ms) and should be avoided in steady state.
     */
    public static void assertNoFullDeviceSync(SameDiff sd) {
        assertNoFullDeviceSync(sd, null);
    }

    public static void assertNoFullDeviceSync(SameDiff sd, String context) {
        int level = getNativeOps().getLastExecSyncLevel(getPlanHandle(sd));
        if (level == SYNC_FULL_DEVICE) {
            int syncCount = getNativeOps().getLastExecStreamSyncCount(getPlanHandle(sd));
            fail("assertNoFullDeviceSync", context,
                    "last execute() used FULL_DEVICE sync (syncCount=" + syncCount + ")", null);
        }
    }

    /**
     * Assert the stream sync count is below a maximum.
     * Too many stream syncs indicate unnecessary serialization points.
     */
    public static void assertStreamSyncCountBelow(SameDiff sd, int maxSyncs) {
        assertStreamSyncCountBelow(sd, maxSyncs, null);
    }

    public static void assertStreamSyncCountBelow(SameDiff sd, int maxSyncs, String context) {
        int count = getNativeOps().getLastExecStreamSyncCount(getPlanHandle(sd));
        if (count >= maxSyncs) {
            fail("assertStreamSyncCountBelow", context,
                    "stream sync count " + count + " >= maximum " + maxSyncs, null);
        }
    }

    /** Get the sync level from the last execute() (non-asserting query). */
    public static int getLastExecSyncLevel(SameDiff sd) {
        return getNativeOps().getLastExecSyncLevel(getPlanHandle(sd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Variable fingerprint drift assertions (Task 13)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that variable external inputs are not stuck (unchanging across executions).
     * A high consecutive-unchanged count means the decode loop is feeding identical data,
     * which typically indicates a KV cache bug or stale staging buffer.
     * @param maxUnchanged maximum allowed consecutive unchanged steps
     */
    public static void assertExtInputsNotStuck(SameDiff sd, int maxUnchanged) {
        assertExtInputsNotStuck(sd, maxUnchanged, null);
    }

    public static void assertExtInputsNotStuck(SameDiff sd, int maxUnchanged, String context) {
        int unchanged = getNativeOps().getLastExecConsecutiveUnchangedCount(getPlanHandle(sd));
        if (unchanged > maxUnchanged) {
            fail("assertExtInputsNotStuck", context,
                    "variable ext inputs unchanged for " + unchanged
                            + " consecutive steps (max=" + maxUnchanged + ")",
                    null);
        }
    }

    /** Get the consecutive unchanged count (non-asserting query). */
    public static int getConsecutiveUnchangedCount(SameDiff sd) {
        return getNativeOps().getLastExecConsecutiveUnchangedCount(getPlanHandle(sd));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Diagnostic event counts (Task 10)
    // ═══════════════════════════════════════════════════════════════════════

    /** Category indices matching DspDiagCategory bit positions in DspDiagnostics.h */
    public static final int DIAG_COMPILE       = 0;
    public static final int DIAG_JIT           = 1;
    public static final int DIAG_EXECUTE       = 2;
    public static final int DIAG_TIMING        = 3;
    public static final int DIAG_MEMORY        = 4;
    public static final int DIAG_BACKEND       = 5;
    public static final int DIAG_SHAPE         = 6;
    public static final int DIAG_SEGMENT       = 7;
    public static final int DIAG_FUSION        = 8;
    public static final int DIAG_VERIFY        = 9;
    public static final int DIAG_KV_CACHE      = 10;
    public static final int DIAG_FALLBACK      = 11;
    public static final int DIAG_TRANSFER      = 12;
    public static final int DIAG_EMULATED_REPLAY = 13;
    public static final int DIAG_STREAM_SYNC   = 14;
    public static final int DIAG_MULTI_DEVICE  = 15;
    public static final int DIAG_GRAPH_REPLAY  = 16;
    public static final int DIAG_SEGMENT_BUCKETS = 17;
    public static final int DIAG_LIFECYCLE     = 18;

    /**
     * Assert that diagnostics recorded at least one event overall.
     * Call after plan execution with diagnostics enabled.
     */
    public static void assertDiagnosticsRecorded(SameDiff sd) {
        assertDiagnosticsRecorded(sd, null);
    }
    public static void assertDiagnosticsRecorded(SameDiff sd, String context) {
        long total = getNativeOps().dspDiagGetTotalEventCount();
        if (total <= 0) {
            fail("assertDiagnosticsRecorded", context,
                    "Expected diagnostic events to be recorded, but totalEventCount=" + total, null);
        }
    }

    /**
     * Assert that a specific diagnostic category has at least minEvents events.
     */
    public static void assertDiagCategoryHasEvents(SameDiff sd, int categoryIndex, long minEvents) {
        assertDiagCategoryHasEvents(sd, categoryIndex, minEvents, null);
    }
    public static void assertDiagCategoryHasEvents(SameDiff sd, int categoryIndex, long minEvents, String context) {
        long count = getNativeOps().dspDiagGetCategoryEventCount(categoryIndex);
        if (count < minEvents) {
            fail("assertDiagCategoryHasEvents", context,
                    "Category " + diagCategoryName(categoryIndex) + " has " + count +
                    " events, expected at least " + minEvents, null);
        }
    }

    /**
     * Assert that no events were recorded for a specific category.
     * Useful for verifying that fallback/error paths were not triggered.
     */
    public static void assertNoCategoryEvents(SameDiff sd, int categoryIndex) {
        assertNoCategoryEvents(sd, categoryIndex, null);
    }
    public static void assertNoCategoryEvents(SameDiff sd, int categoryIndex, String context) {
        long count = getNativeOps().dspDiagGetCategoryEventCount(categoryIndex);
        if (count > 0) {
            fail("assertNoCategoryEvents", context,
                    "Category " + diagCategoryName(categoryIndex) + " should have 0 events but has " + count, null);
        }
    }

    /**
     * Assert that no fallback events were recorded (category FALLBACK=11).
     */
    public static void assertNoFallbacks(SameDiff sd) {
        assertNoFallbacks(sd, null);
    }
    public static void assertNoFallbacks(SameDiff sd, String context) {
        assertNoCategoryEvents(sd, DIAG_FALLBACK, context);
    }

    /** Get the diagnostic step count from the global singleton. */
    public static int getDiagStepCount(SameDiff sd) {
        return getNativeOps().dspDiagGetStepCount();
    }

    /** Get total event count from the global diagnostic singleton. */
    public static long getDiagTotalEventCount(SameDiff sd) {
        return getNativeOps().dspDiagGetTotalEventCount();
    }

    /** Get per-category event count from the global diagnostic singleton. */
    public static long getDiagCategoryEventCount(SameDiff sd, int categoryIndex) {
        return getNativeOps().dspDiagGetCategoryEventCount(categoryIndex);
    }

    private static String diagCategoryName(int index) {
        switch (index) {
            case DIAG_COMPILE: return "COMPILE";
            case DIAG_JIT: return "JIT";
            case DIAG_EXECUTE: return "EXECUTE";
            case DIAG_TIMING: return "TIMING";
            case DIAG_MEMORY: return "MEMORY";
            case DIAG_BACKEND: return "BACKEND";
            case DIAG_SHAPE: return "SHAPE";
            case DIAG_SEGMENT: return "SEGMENT";
            case DIAG_FUSION: return "FUSION";
            case DIAG_VERIFY: return "VERIFY";
            case DIAG_KV_CACHE: return "KV_CACHE";
            case DIAG_FALLBACK: return "FALLBACK";
            case DIAG_TRANSFER: return "TRANSFER";
            case DIAG_EMULATED_REPLAY: return "EMULATED_REPLAY";
            case DIAG_STREAM_SYNC: return "STREAM_SYNC";
            case DIAG_MULTI_DEVICE: return "MULTI_DEVICE";
            case DIAG_GRAPH_REPLAY: return "GRAPH_REPLAY";
            case DIAG_SEGMENT_BUCKETS: return "SEGMENT_BUCKETS";
            case DIAG_LIFECYCLE: return "LIFECYCLE";
            default: return "UNKNOWN_" + index;
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Cross-mode output comparison (Task 11)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Assert that a single named output matches a reference value within tolerance.
     * Use to compare outputs from two different execution modes (e.g., SLOT_BY_SLOT vs TRITON).
     *
     * @param actual     the output from the mode under test
     * @param reference  the golden output (e.g., from SLOT_BY_SLOT)
     * @param name       output variable name (for error messages)
     * @param rtol       relative tolerance (e.g., 1e-4)
     * @param atol       absolute tolerance (e.g., 1e-6)
     */
    public static void assertSlotOutputMatchesReference(INDArray actual, INDArray reference,
                                                         String name, double rtol, double atol) {
        assertSlotOutputMatchesReference(actual, reference, name, rtol, atol, null);
    }

    public static void assertSlotOutputMatchesReference(INDArray actual, INDArray reference,
                                                         String name, double rtol, double atol,
                                                         String context) {
        if (reference == null && actual == null) return;
        if (reference == null) {
            fail("assertSlotOutputMatchesReference", context,
                    "Output '" + name + "': reference is null but actual is non-null " + actual.shapeInfoToString(), null);
            return;
        }
        if (actual == null) {
            fail("assertSlotOutputMatchesReference", context,
                    "Output '" + name + "': actual is null but reference is non-null " + reference.shapeInfoToString(), null);
            return;
        }
        if (!actual.equalShapes(reference)) {
            fail("assertSlotOutputMatchesReference", context,
                    "Output '" + name + "': shape mismatch — actual=" + actual.shapeInfoToString() +
                    " vs reference=" + reference.shapeInfoToString(), null);
            return;
        }
        if (actual.dataType() != reference.dataType()) {
            fail("assertSlotOutputMatchesReference", context,
                    "Output '" + name + "': dtype mismatch — actual=" + actual.dataType() +
                    " vs reference=" + reference.dataType(), null);
            return;
        }
        // Element-wise comparison with combined tolerance: |actual - ref| <= atol + rtol * |ref|
        INDArray diff = Transforms.abs(actual.sub(reference));
        INDArray threshold = Transforms.abs(reference).mul(rtol).addi(atol);
        long violations = diff.gt(threshold).castTo(org.nd4j.linalg.api.buffer.DataType.INT64).sumNumber().longValue();
        if (violations > 0) {
            long totalElements = actual.length();
            double maxDiff = diff.maxNumber().doubleValue();
            fail("assertSlotOutputMatchesReference", context,
                    "Output '" + name + "': " + violations + "/" + totalElements +
                    " elements exceed tolerance (rtol=" + rtol + ", atol=" + atol +
                    "), maxDiff=" + maxDiff, null);
        }
    }

    /**
     * Assert that all outputs from one execution match the reference outputs within tolerance.
     * Keys present in reference but missing from actual are flagged. Extra keys in actual are ignored.
     *
     * @param actual     outputs from the mode under test
     * @param reference  golden outputs (e.g., from SLOT_BY_SLOT)
     * @param rtol       relative tolerance
     * @param atol       absolute tolerance
     */
    public static void assertAllOutputsMatchReference(Map<String, INDArray> actual,
                                                       Map<String, INDArray> reference,
                                                       double rtol, double atol) {
        assertAllOutputsMatchReference(actual, reference, rtol, atol, null);
    }

    public static void assertAllOutputsMatchReference(Map<String, INDArray> actual,
                                                       Map<String, INDArray> reference,
                                                       double rtol, double atol,
                                                       String context) {
        List<String> failures = new ArrayList<>();
        for (Map.Entry<String, INDArray> entry : reference.entrySet()) {
            String name = entry.getKey();
            INDArray refArr = entry.getValue();
            INDArray actArr = actual.get(name);
            if (actArr == null) {
                failures.add("'" + name + "': missing from actual outputs");
                continue;
            }
            try {
                assertSlotOutputMatchesReference(actArr, refArr, name, rtol, atol, context);
            } catch (AssertionError e) {
                failures.add(e.getMessage());
            }
        }
        if (!failures.isEmpty()) {
            fail("assertAllOutputsMatchReference", context,
                    failures.size() + " output(s) failed cross-mode comparison:\n  " +
                    String.join("\n  ", failures), null);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helpers
    // ═══════════════════════════════════════════════════════════════════════

    private static String describeOutputFlags(int flags) {
        List<String> parts = new ArrayList<>();
        if ((flags & 1) != 0) parts.add("NULL");
        if ((flags & 2) != 0) parts.add("NaN");
        if ((flags & 4) != 0) parts.add("Inf");
        if ((flags & 8) != 0) parts.add("ALL_ZERO");
        return String.join("|", parts);
    }

    private static String flagBitName(int bit) {
        switch (bit) {
            case FLAG_VIEW_CAPABLE: return "VIEW_CAPABLE";
            case FLAG_DATA_DEPENDENT: return "DATA_DEPENDENT";
            case FLAG_SHAPE_DEPENDS_ON_VALUES: return "SHAPE_DEPENDS_ON_VALUES";
            case FLAG_IDENTITY: return "IDENTITY";
            case FLAG_IN_PLACE_FUSED: return "IN_PLACE_FUSED";
            case FLAG_FUSED_CHAIN_HEAD: return "FUSED_CHAIN_HEAD";
            case FLAG_FUSED_CHAIN_TAIL: return "FUSED_CHAIN_TAIL";
            case FLAG_NEEDS_ZEROED: return "NEEDS_ZEROED";
            case FLAG_NEEDS_INT_LONG_SYNC: return "NEEDS_INT_LONG_SYNC";
            case FLAG_SHAPE_STATIC: return "SHAPE_STATIC";
            case FLAG_FROZEN_CONSTANT: return "FROZEN_CONSTANT";
            default: return "BIT_" + bit;
        }
    }

    private static String decodeSlotFlags(int flags) {
        List<String> parts = new ArrayList<>();
        for (int bit = 0; bit <= 10; bit++) {
            if ((flags & (1 << bit)) != 0) {
                parts.add(flagBitName(bit));
            }
        }
        return parts.isEmpty() ? "NONE" : String.join("|", parts);
    }

    private static String slotStateName(int state) {
        switch (state) {
            case 0: return "WARMUP";
            case 1: return "SHAPE_CACHED";
            case 2: return "FROZEN";
            case 3: return "FROZEN_CONSTANT";
            default: return "UNKNOWN(" + state + ")";
        }
    }

    private static org.nd4j.nativeblas.NativeOps getNativeOps() {
        return org.nd4j.linalg.factory.Nd4j.getNativeOps();
    }

    private static void fail(String check, String context, String detail,
                             DspDebugger.PlanReport report) {
        StringBuilder sb = new StringBuilder();
        sb.append("DspPlanAssertions.").append(check).append(" FAILED");
        if (context != null && !context.isEmpty()) {
            sb.append(" [").append(context).append("]");
        }
        sb.append(": ").append(detail);
        if (report != null && report.errorMessage == null) {
            sb.append("\n").append(report);
        }
        throw new AssertionError(sb.toString());
    }
}
