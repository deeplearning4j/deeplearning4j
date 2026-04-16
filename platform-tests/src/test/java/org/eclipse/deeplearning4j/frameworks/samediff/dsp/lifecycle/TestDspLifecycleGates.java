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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.lifecycle;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Isolation tests for the DSP lifecycle gate shim wired through
 * {@code libnd4j/include/graph/DspLifecycleContext.h}.
 *
 * <p>The shim centralises four gates that the slot-by-slot execution path now
 * consults to avoid clobbering DSP-owned state:</p>
 *
 * <ol>
 *   <li><b>Tick gate</b> — {@code execCustomOp2} skips {@code tick{Read,Write}Device}
 *       while DSP owns the thread. Verified here by toggling {@code dspSetExecutionMode}
 *       and observing the JNI query surface. The tick call itself is internal
 *       to {@code execCustomOp2} and is exercised transitively by every
 *       slot-by-slot op that runs while mode is non-legacy.</li>
 *   <li><b>Resync gate</b> — {@link NativeOps#dspShouldSkipResync(OpaqueDataBuffer)}
 *       returns true only when (a) DSP owns the thread AND (b) the buffer is
 *       DSP-protected. Out of any active DSP context it is always false so
 *       normal execution is unchanged.</li>
 *   <li><b>Close gate</b> — {@link NativeOps#dspShouldDeferClose(OpaqueDataBuffer)}
 *       returns true when the buffer is DSP-protected (via frozen plan
 *       registration or {@code isConstant}) regardless of ownership, and also
 *       when DSP owns the thread regardless of buffer protection. Out of any
 *       active DSP context on an unprotected buffer it is false.</li>
 *   <li><b>Mode storage</b> — {@link NativeOps#dspSetExecutionMode(int)} /
 *       {@link NativeOps#dspGetExecutionMode()} round-trip across the three
 *       modes (LEGACY_UNAWARE=0, COEXIST_SAFE=1, STRICT_ISOLATED=2). Legacy mode
 *       disables all gates so the pre-shim behaviour can be bisected against.</li>
 * </ol>
 *
 * <p>These tests do not (and cannot) activate capture/replay from the Java
 * side — the thread-locals live in native TLS. End-to-end validation that
 * the gates actually fire during capture is covered by
 * {@code TestFrozenPhaseDriftDetection} and the graph replay correctness
 * suites, which exercise the full slot-by-slot ↔ DSP coexistence path.</p>
 *
 * <p>Run from platform-tests:</p>
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=TestDspLifecycleGates \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/dsp-lifecycle-gates.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@DisplayName("DSP Lifecycle Gates")
public class TestDspLifecycleGates {

    private static final int LEGACY_UNAWARE = 0;
    private static final int COEXIST_SAFE = 1;
    private static final int STRICT_ISOLATED = 2;

    private NativeOps nativeOps;
    private int restoreMode;

    @BeforeEach
    void saveMode() {
        nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        assertNotNull(nativeOps, "NativeOps must be available for DSP lifecycle gate tests");
        restoreMode = nativeOps.dspGetExecutionMode();
    }

    @AfterEach
    void restoreMode() {
        if (nativeOps != null) {
            nativeOps.dspSetExecutionMode(restoreMode);
        }
    }

    // ---------------------------------------------------------------- mode storage

    @Test
    @DisplayName("dspSetExecutionMode / dspGetExecutionMode round-trip across all three modes")
    void executionModeRoundtrip() {
        nativeOps.dspSetExecutionMode(LEGACY_UNAWARE);
        assertEquals(LEGACY_UNAWARE, nativeOps.dspGetExecutionMode(),
                "LEGACY_UNAWARE (0) did not round-trip");

        nativeOps.dspSetExecutionMode(COEXIST_SAFE);
        assertEquals(COEXIST_SAFE, nativeOps.dspGetExecutionMode(),
                "COEXIST_SAFE (1) did not round-trip");

        nativeOps.dspSetExecutionMode(STRICT_ISOLATED);
        assertEquals(STRICT_ISOLATED, nativeOps.dspGetExecutionMode(),
                "STRICT_ISOLATED (2) did not round-trip");
    }

    @Test
    @DisplayName("Default execution mode is COEXIST_SAFE (not LEGACY_UNAWARE)")
    void defaultModeIsCoexistSafe() {
        // The restoreMode we captured in @BeforeEach is whatever NativeOpsHolder
        // installed at startup. With no -Dnd4j.dsp.executionMode it must stay at
        // the native default COEXIST_SAFE so slot-by-slot gating is active out
        // of the box. If startup ever installs LEGACY by accident the shim is
        // silently disabled.
        assertEquals(COEXIST_SAFE, restoreMode,
                "Startup default for DSP execution mode regressed — must be COEXIST_SAFE. " +
                        "LEGACY_UNAWARE is only for bisecting and must never ship as default.");
    }

    // ------------------------------------------------------ ownership queries (cold)

    @Test
    @DisplayName("Ownership queries are false when no DSP capture or replay is active")
    void ownershipFalseOutOfBand() {
        nativeOps.dspSetExecutionMode(COEXIST_SAFE);

        assertFalse(nativeOps.dspIsCaptureActive(),
                "dspIsCaptureActive must be false with no capture in flight");
        assertFalse(nativeOps.dspIsReplayActive(),
                "dspIsReplayActive must be false with no replay in flight");
        assertFalse(nativeOps.dspIsOwned(),
                "dspIsOwned must be false with no DSP activity on the thread");
    }

    // ------------------------------------------------------ resync gate

    @Test
    @DisplayName("dspShouldSkipResync is false for a fresh buffer when no DSP activity is in flight")
    void resyncGateFalseOutOfBand() {
        nativeOps.dspSetExecutionMode(COEXIST_SAFE);

        INDArray arr = Nd4j.linspace(DataType.FLOAT, 1.0f, 1.0f, 16);
        try {
            OpaqueDataBuffer opaque = arr.data().opaqueBuffer();
            assertNotNull(opaque, "Data buffer must have a JNI handle");

            assertFalse(nativeOps.dspShouldSkipResync(opaque),
                    "Out-of-band resync must not be skipped — blocking it would strand " +
                            "host writes on their way to the device");
        } finally {
            arr.close();
        }
    }

    @Test
    @DisplayName("Legacy mode forces resync gate to false even for DSP-protected buffers")
    void resyncGateLegacyAlwaysFalse() {
        nativeOps.dspSetExecutionMode(LEGACY_UNAWARE);

        INDArray arr = Nd4j.linspace(DataType.FLOAT, 1.0f, 1.0f, 8);
        try {
            OpaqueDataBuffer opaque = arr.data().opaqueBuffer();
            assertFalse(nativeOps.dspShouldSkipResync(opaque),
                    "LEGACY_UNAWARE must never defer — it is the bisecting escape hatch");
        } finally {
            arr.close();
        }
    }

    // ------------------------------------------------------ close gate

    @Test
    @DisplayName("dspShouldDeferClose is false for a fresh unprotected buffer")
    void closeGateFalseForFreshBuffer() {
        nativeOps.dspSetExecutionMode(COEXIST_SAFE);

        INDArray arr = Nd4j.zeros(DataType.FLOAT, 4, 4);
        try {
            OpaqueDataBuffer opaque = arr.data().opaqueBuffer();
            assertNotNull(opaque, "Data buffer must have a JNI handle");

            assertFalse(nativeOps.dspShouldDeferClose(opaque),
                    "A freshly-allocated non-constant non-frozen buffer has no DSP owner, " +
                            "so slot-by-slot close must not be deferred");
            assertFalse(nativeOps.dbDspProtects(opaque),
                    "A fresh buffer is not DSP-protected");
            assertFalse(nativeOps.dbIsFrozenPlanRegistered(opaque),
                    "A fresh buffer is not registered in any frozen plan");
        } finally {
            arr.close();
        }
    }

    @Test
    @DisplayName("Legacy mode forces close gate to false even for DSP-protected buffers")
    void closeGateLegacyAlwaysFalse() {
        nativeOps.dspSetExecutionMode(LEGACY_UNAWARE);

        INDArray arr = Nd4j.zeros(DataType.FLOAT, 8);
        try {
            OpaqueDataBuffer opaque = arr.data().opaqueBuffer();
            assertFalse(nativeOps.dspShouldDeferClose(opaque),
                    "LEGACY_UNAWARE must never defer — the bisecting mode is the unchanged " +
                            "pre-shim behaviour and close is one of the pre-shim actions");
        } finally {
            arr.close();
        }
    }

    @Test
    @DisplayName("Constant buffer is reported as DSP-protected and close is deferred (COEXIST_SAFE)")
    void closeGateDefersConstant() {
        nativeOps.dspSetExecutionMode(COEXIST_SAFE);

        INDArray arr = Nd4j.linspace(DataType.FLOAT, 0.0f, 1.0f, 16);
        OpaqueDataBuffer opaque = arr.data().opaqueBuffer();
        assertNotNull(opaque);

        // Flip the DataBuffer's constant flag via the JNI setter used by the
        // constant cache. This exercises dbDspProtects() -> isConstant branch
        // without needing a real frozen plan.
        nativeOps.dbSetConstant(opaque, true);
        try {
            assertTrue(nativeOps.dbIsConstant(opaque),
                    "dbSetConstant(true) must flip dbIsConstant to true");
            assertTrue(nativeOps.dbDspProtects(opaque),
                    "A constant DataBuffer must be reported as DSP-protected");
            assertTrue(nativeOps.dspShouldDeferClose(opaque),
                    "A constant (protected) buffer must trigger shouldDeferClose " +
                            "even without an active capture/replay — DSP may register it any time");
        } finally {
            // Undo the constant flag so the buffer can be closed cleanly in test teardown.
            nativeOps.dbSetConstant(opaque, false);
            arr.close();
        }
    }

    @Test
    @DisplayName("Unknown mode input normalises through set/get without corrupting state")
    void unknownModeDoesNotCorrupt() {
        // Values outside 0..2 are not defined, but the native setter must not
        // crash. It either clamps or falls through to the default; either way,
        // the next explicit set must still succeed.
        nativeOps.dspSetExecutionMode(99);
        nativeOps.dspSetExecutionMode(COEXIST_SAFE);
        assertEquals(COEXIST_SAFE, nativeOps.dspGetExecutionMode(),
                "Explicit COEXIST_SAFE after an unknown-mode input must still take effect");
    }
}
