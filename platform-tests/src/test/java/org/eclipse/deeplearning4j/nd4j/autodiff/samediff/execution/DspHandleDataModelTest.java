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
package org.eclipse.deeplearning4j.nd4j.autodiff.samediff.execution;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.autodiff.samediff.execution.DspHandle.CaptureStats;
import org.nd4j.autodiff.samediff.execution.DspHandle.D2DStatus;
import org.nd4j.autodiff.samediff.execution.DspHandle.SegmentStepInfo;
import org.nd4j.autodiff.samediff.execution.DspHandle.StepSnapshot;

import java.util.*;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Isolated unit tests for {@link DspHandle} inner data classes:
 * {@link CaptureStats}, {@link StepSnapshot}, {@link D2DStatus}, {@link SegmentStepInfo}.
 * No native handle required — pure Java data model tests.
 */
@Tag(TagNames.SMOKE)
@DisplayName("DspHandle data model unit tests")
public class DspHandleDataModelTest {

    // ─── CaptureStats ─────────────────────────────────────────────────────────

    @Test
    @DisplayName("CaptureStats parse: valid pipe-delimited string")
    void testCaptureStatsParse_Valid() {
        String raw = "captured=3|oomRetrying=0|permFailed=1|nonCapt=2|tooSmall=0|addrUnstable=0";
        CaptureStats stats = CaptureStats.parse(raw);

        assertEquals(3, stats.captured);
        assertEquals(0, stats.oomRetrying);
        assertEquals(1, stats.permFailed);
        assertEquals(2, stats.nonCapturable);
        assertEquals(0, stats.tooSmall);
        assertEquals(0, stats.addrUnstable);
    }

    @Test
    @DisplayName("CaptureStats parse: empty string")
    void testCaptureStatsParse_Empty() {
        CaptureStats stats = CaptureStats.parse("");
        assertEquals(0, stats.captured);
        assertEquals(0, stats.permFailed);
    }

    @Test
    @DisplayName("CaptureStats parse: null")
    void testCaptureStatsParse_Null() {
        CaptureStats stats = CaptureStats.parse(null);
        assertEquals(0, stats.captured);
        assertEquals(0, stats.permFailed);
    }

    @Test
    @DisplayName("CaptureStats parse: malformed entries ignored")
    void testCaptureStatsParse_Malformed() {
        String raw = "captured=5|bogus|permFailed=3|tooSmall=NaN";
        CaptureStats stats = CaptureStats.parse(raw);

        assertEquals(5, stats.captured);
        assertEquals(3, stats.permFailed);
        // tooSmall=NaN should be ignored (NumberFormatException caught)
        assertEquals(0, stats.tooSmall);
    }

    @Test
    @DisplayName("CaptureStats parse: missing fields default to zero")
    void testCaptureStatsParse_Partial() {
        String raw = "captured=7";
        CaptureStats stats = CaptureStats.parse(raw);

        assertEquals(7, stats.captured);
        assertEquals(0, stats.oomRetrying);
        assertEquals(0, stats.permFailed);
        assertEquals(0, stats.nonCapturable);
        assertEquals(0, stats.tooSmall);
        assertEquals(0, stats.addrUnstable);
    }

    @Test
    @DisplayName("CaptureStats toString round-trip")
    void testCaptureStatsToString() {
        CaptureStats stats = new CaptureStats(3, 1, 0, 2, 0, 1);
        String s = stats.toString();
        assertTrue(s.contains("captured=3"));
        assertTrue(s.contains("oomRetrying=1"));
        assertTrue(s.contains("permFailed=0"));
        assertTrue(s.contains("nonCapt=2"));
        assertTrue(s.contains("tooSmall=0"));
        assertTrue(s.contains("addrUnstable=1"));
    }

    // ─── StepSnapshot ────────────────────────────────────────────────────────

    private static StepSnapshot makeSnapshot(int execCount, int phase, boolean ptrStable,
                                              int frozenExec, Map<Integer, D2DStatus> d2d,
                                              List<SegmentStepInfo> segments,
                                              CaptureStats captureStats,
                                              List<Integer> drifting) {
        return new StepSnapshot(execCount, phase, ptrStable, frozenExec,
                d2d, segments, captureStats, drifting);
    }

    @Test
    @DisplayName("StepSnapshot: allD2DCopiesFired — all fired")
    void testStepSnapshot_AllD2DFired() {
        Map<Integer, D2DStatus> d2d = new HashMap<>();
        d2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // fired
        d2d.put(1, new D2DStatus(1, 0x2000L, 0x2000L)); // fired

        StepSnapshot snap = makeSnapshot(5, 3, true, 3, d2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        assertTrue(snap.allD2DCopiesFired());
    }

    @Test
    @DisplayName("StepSnapshot: allD2DCopiesFired — one not fired")
    void testStepSnapshot_NotAllD2DFired() {
        Map<Integer, D2DStatus> d2d = new HashMap<>();
        d2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // fired
        d2d.put(1, new D2DStatus(1, 0x2000L, 0x3000L)); // not fired (drift)

        StepSnapshot snap = makeSnapshot(5, 3, true, 3, d2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        assertFalse(snap.allD2DCopiesFired());
    }

    @Test
    @DisplayName("StepSnapshot: failedD2DExtIndices")
    void testStepSnapshot_FailedD2DExtIndices() {
        Map<Integer, D2DStatus> d2d = new HashMap<>();
        d2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // fired
        d2d.put(1, new D2DStatus(1, 0x2000L, 0x3000L)); // drift
        d2d.put(2, new D2DStatus(2, 0x0L, 0x4000L));    // no staging (stagingAddr=0)

        StepSnapshot snap = makeSnapshot(5, 3, true, 3, d2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        List<Integer> failed = snap.failedD2DExtIndices();
        // Only extIdx=1 should be in failed: staging exists (0x2000) but effective differs (0x3000)
        assertEquals(List.of(1), failed);
    }

    @Test
    @DisplayName("StepSnapshot: replayedSegments")
    void testStepSnapshot_ReplayedSegments() {
        List<SegmentStepInfo> segments = new ArrayList<>();
        segments.add(new SegmentStepInfo(0, 5, 3, 3)); // phase=3 REPLAYING
        segments.add(new SegmentStepInfo(1, 5, 3, 4)); // phase=4 SLOT_BY_SLOT
        segments.add(new SegmentStepInfo(2, 5, 3, 3)); // phase=3 REPLAYING

        StepSnapshot snap = makeSnapshot(5, 3, true, 3, Map.of(),
                segments, new CaptureStats(3, 0, 0, 0, 0, 0), List.of());

        List<Integer> replayed = snap.replayedSegments();
        assertEquals(List.of(0, 2), replayed);
    }

    @Test
    @DisplayName("StepSnapshot: slotBySlotSegments")
    void testStepSnapshot_SlotBySlotSegments() {
        List<SegmentStepInfo> segments = new ArrayList<>();
        segments.add(new SegmentStepInfo(0, 5, 0, 3)); // REPLAYING
        segments.add(new SegmentStepInfo(1, 5, 0, 4)); // SLOT_BY_SLOT
        segments.add(new SegmentStepInfo(2, 5, 0, 4)); // SLOT_BY_SLOT

        StepSnapshot snap = makeSnapshot(5, 4, true, 3, Map.of(),
                segments, new CaptureStats(0, 0, 0, 0, 0, 0), List.of());

        List<Integer> sbs = snap.slotBySlotSegments();
        assertEquals(List.of(1, 2), sbs);
    }

    @Test
    @DisplayName("StepSnapshot: hasPointerDrift — no drift")
    void testStepSnapshot_NoPointerDrift() {
        StepSnapshot snap = makeSnapshot(5, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        assertFalse(snap.hasPointerDrift());
    }

    @Test
    @DisplayName("StepSnapshot: hasPointerDrift — with drift")
    void testStepSnapshot_WithPointerDrift() {
        StepSnapshot snap = makeSnapshot(5, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of(0, 2));

        assertTrue(snap.hasPointerDrift());
    }

    @Test
    @DisplayName("StepSnapshot: driftingExtIndices")
    void testStepSnapshot_DriftingExtIndices() {
        Map<Integer, D2DStatus> d2d = new HashMap<>();
        d2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // no drift
        d2d.put(1, new D2DStatus(1, 0x2000L, 0x3000L)); // drift
        d2d.put(2, new D2DStatus(2, 0x4000L, 0x5000L)); // drift
        d2d.put(3, new D2DStatus(3, 0x0L, 0x6000L));    // no staging

        StepSnapshot snap = makeSnapshot(5, 3, true, 3, d2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        List<Integer> drifting = snap.driftingExtIndices();
        assertEquals(List.of(1, 2), drifting);
    }

    @Test
    @DisplayName("StepSnapshot: describeChangesFrom — first call")
    void testStepSnapshot_DescribeChanges_FirstCall() {
        StepSnapshot snap = makeSnapshot(5, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        String desc = snap.describeChangesFrom(null);
        assertEquals("no previous snapshot", desc);
    }

    @Test
    @DisplayName("StepSnapshot: describeChangesFrom — phase transition")
    void testStepSnapshot_DescribeChanges_PhaseTransition() {
        StepSnapshot prev = makeSnapshot(4, 1, false, 2, Map.of(),
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());
        StepSnapshot curr = makeSnapshot(5, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(3, 0, 0, 0, 0, 0), List.of());

        String desc = curr.describeChangesFrom(prev);
        assertTrue(desc.contains("step 4 -> 5"));
        assertTrue(desc.contains("phase 1->3"));
        assertTrue(desc.contains("ptrsStable false->true"));
    }

    @Test
    @DisplayName("StepSnapshot: describeChangesFrom — D2D change")
    void testStepSnapshot_DescribeChanges_D2DChange() {
        Map<Integer, D2DStatus> prevD2d = new HashMap<>();
        prevD2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // fired

        Map<Integer, D2DStatus> currD2d = new HashMap<>();
        currD2d.put(0, new D2DStatus(0, 0x1000L, 0x1000L)); // fired
        currD2d.put(1, new D2DStatus(1, 0x2000L, 0x2000L)); // fired

        StepSnapshot prev = makeSnapshot(4, 3, true, 3, prevD2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());
        StepSnapshot curr = makeSnapshot(5, 3, true, 3, currD2d,
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());

        String desc = curr.describeChangesFrom(prev);
        assertTrue(desc.contains("d2dFired 1->2"));
    }

    @Test
    @DisplayName("StepSnapshot: describeChangesFrom — capture change")
    void testStepSnapshot_DescribeChanges_CaptureChange() {
        StepSnapshot prev = makeSnapshot(4, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(2, 0, 0, 0, 0, 0), List.of());
        StepSnapshot curr = makeSnapshot(5, 3, true, 3, Map.of(),
                List.of(), new CaptureStats(3, 0, 0, 0, 0, 0), List.of());

        String desc = curr.describeChangesFrom(prev);
        assertTrue(desc.contains("captured 2->3"));
    }

    // ─── D2DStatus ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("D2DStatus: fired when staging matches effective")
    void testD2DStatus_Fired() {
        D2DStatus status = new D2DStatus(0, 0x1000L, 0x1000L);
        assertTrue(status.fired);
        assertFalse(status.addressDrift);
    }

    @Test
    @DisplayName("D2DStatus: addressDrift when staging != effective")
    void testD2DStatus_AddressDrift() {
        D2DStatus status = new D2DStatus(0, 0x1000L, 0x2000L);
        assertFalse(status.fired);
        assertTrue(status.addressDrift);
    }

    @Test
    @DisplayName("D2DStatus: no staging buffer")
    void testD2DStatus_NoStaging() {
        D2DStatus status = new D2DStatus(0, 0x0L, 0x2000L);
        assertFalse(status.fired);
        assertFalse(status.addressDrift);
    }

    @Test
    @DisplayName("D2DStatus: toString format")
    void testD2DStatus_ToString() {
        D2DStatus status = new D2DStatus(1, 0x1000L, 0x2000L);
        String s = status.toString();
        assertTrue(s.contains("ext[1]"));
        assertTrue(s.contains("staging=0x1000"));
        assertTrue(s.contains("effective=0x2000"));
        assertTrue(s.contains("fired=false"));
        assertTrue(s.contains("drift=true"));
    }

    // ─── SegmentStepInfo ──────────────────────────────────────────────────────

    @Test
    @DisplayName("SegmentStepInfo: wasReplayed when phase=REPLAYING")
    void testSegmentStepInfo_WasReplayed() {
        SegmentStepInfo info = new SegmentStepInfo(0, 5, 3, 3); // phase=3 = REPLAYING
        assertTrue(info.wasReplayed);
    }

    @Test
    @DisplayName("SegmentStepInfo: not wasReplayed when phase!=REPLAYING")
    void testSegmentStepInfo_NotReplayed() {
        SegmentStepInfo info = new SegmentStepInfo(0, 5, 0, 0); // phase=0 = WARMUP
        assertFalse(info.wasReplayed);
    }

    @Test
    @DisplayName("SegmentStepInfo: toString format")
    void testSegmentStepInfo_ToString() {
        SegmentStepInfo info = new SegmentStepInfo(2, 10, 5, 3);
        String s = info.toString();
        assertTrue(s.contains("seg[2]"));
        assertTrue(s.contains("exec=10"));
        assertTrue(s.contains("replay=5"));
        assertTrue(s.contains("phase=3"));
        assertTrue(s.contains("replaying=true"));
    }
}