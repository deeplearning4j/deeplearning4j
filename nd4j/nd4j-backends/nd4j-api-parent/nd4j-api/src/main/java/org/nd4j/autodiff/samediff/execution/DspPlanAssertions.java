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

import java.util.ArrayList;
import java.util.List;

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
 *   DspPlanAssertions.assertPhaseReached(sd, PlanPhase.POINTERS_STABLE);
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
    // Internals
    // ═══════════════════════════════════════════════════════════════════════

    private static void requirePlanAvailable(DspDebugger.PlanReport report, String context) {
        if (report.errorMessage != null) {
            fail("plan availability", context,
                    "plan not compiled: " + report.errorMessage, report);
        }
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
