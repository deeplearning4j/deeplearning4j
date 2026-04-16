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
package org.eclipse.deeplearning4j.frameworks.samediff.dsp.regression;

import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.factory.Environment;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * argTableStable fast-replay performance floor.
 *
 * <p>When {@code argTableStable == true}, the Triton graph-replay path skips
 * refresh + EXT_INPUT_SYNC on every frozen replay step — only D2D capture
 * buffer copies and graph launch are needed. The "stable" path must be
 * <i>at least as fast</i> as the "unstable" path (which redoes the refresh
 * on every step); if it ever becomes slower there's a bug in the
 * stability-tracking code. See
 * {@code TritonGraphBackend_kernel.cu#refreshArgTablesForReplay} and the
 * fast-replay notes in MEMORY.md.
 *
 * <p>This test measures tok/s with argTableStable ON and OFF (via the
 * Triton arg dirty tracking knob, which is the mechanism that produces
 * stable-table windows), then asserts {@code onTps >= 0.95 * offTps}. The
 * 5% slack covers measurement noise; a genuine regression is many
 * multiples of this.
 *
 * <p><b>Assumption.</b> "argTableStable" is controlled by
 * {@link Environment#setTritonArgDirtyTracking(boolean)}: when dirty
 * tracking is enabled, stability is computed per-step and the fast path is
 * taken whenever arg pointers are unchanged. When it is disabled, the fast
 * path is never taken (every step runs the full refresh). This matches
 * the wiring in {@code TritonGraphBackend_kernel.cu}.
 *
 * <p>Run from platform-tests:
 * <pre>
 *   cd platform-tests &amp;&amp; mvn test \
 *       -Dtest=TestArgTableStablePerfFloor \
 *       -Dbackend.artifactId=nd4j-cuda-12.9 \
 *       2&gt;&amp;1 | tee /tmp/argtable-stable-perf.log
 * </pre>
 */
@Slf4j
@NativeTag
@Tag(TagNames.SAMEDIFF)
@Tag(TagNames.LONG_TEST)
@DisplayName("DSP argTableStable fast-replay perf floor")
public class TestArgTableStablePerfFloor extends DspRegressionHarness {

    private static final int WARMUP = 10;
    private static final int MEASURE = 50;

    /**
     * Minimum ratio of (stable on) / (stable off) — 0.95 allows a 5%
     * measurement-noise margin. A real regression pushes the ratio well
     * below this (argTableStable off is materially slower per-step).
     */
    private static final double MIN_RATIO = 0.95;

    private EnvSnapshot snapshot;

    @BeforeEach
    public void setUp() {
        snapshot = captureEnv();
    }

    @AfterEach
    public void tearDown() {
        commit();
        if (snapshot != null) snapshot.restore();
    }

    @Test
    @DisplayName("argTableStable ON must be ≥ 0.95 × argTableStable OFF tok/s")
    public void testArgTableStablePerfFloor() {
        assumeCudaAvailable();
        assumeTritonAvailable();

        double tpsOff = measureTokPerSec(false);
        double tpsOn = measureTokPerSec(true);
        double ratio = tpsOn / Math.max(tpsOff, 1e-9);

        log.info("argTableStable perf floor: OFF={} tok/s, ON={} tok/s, ratio={} (floor={})",
                fmt(tpsOff), fmt(tpsOn), fmt(ratio), fmt(MIN_RATIO));

        assertTrue(ratio >= MIN_RATIO,
                "argTableStable fast path regressed: ON=" + fmt(tpsOn)
                        + " tok/s, OFF=" + fmt(tpsOff) + " tok/s, ratio=" + fmt(ratio)
                        + " < floor " + fmt(MIN_RATIO));
    }

    private double measureTokPerSec(boolean argDirtyTracking) {
        Environment env = Nd4j.getEnvironment();
        // Ensure Triton graph capture + consolidated arg table are both on —
        // they're the precondition for the fast path to be reachable at all.
        env.setTritonGraphCapture(true);
        env.setTritonSectionFusion(true);
        env.setTritonCompileAll(true);
        env.setTritonConsolidatedArgTable(true);
        env.setTritonArgDirtyTracking(argDirtyTracking);

        try (DspReplayFixture fix = new DspReplayFixture()) {
            // Warmup
            for (int i = 0; i < WARMUP; i++) fix.stepAndDiscard();
            commit();
            // Measure
            long start = System.nanoTime();
            for (int i = 0; i < MEASURE; i++) fix.stepAndDiscard();
            commit();
            long end = System.nanoTime();
            double seconds = (end - start) / 1_000_000_000.0;
            return MEASURE / seconds;
        }
    }

    private static String fmt(double d) {
        return String.format(Locale.ROOT, "%.3f", d);
    }
}
