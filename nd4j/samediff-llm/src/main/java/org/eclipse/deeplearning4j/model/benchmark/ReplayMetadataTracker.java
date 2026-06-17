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

package org.eclipse.deeplearning4j.model.benchmark;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.Getter;

/**
 * Tracks replay metadata across validation iterations: phase progression, replay
 * identity/signature stability, replay unit counts, execution counts, and phase
 * violations.
 *
 * <h3>What this tracks</h3>
 * <ul>
 *   <li><b>Phase progression</b> — COMPILE → CAPTURE → REPLAY transitions per iteration</li>
 *   <li><b>Replay signature stability</b> — Hash of each segment's replay identity; must
 *       remain constant across iterations on the same compiled native handle</li>
 *   <li><b>Replay unit counts</b> — Number of replay units per segment; should not increase
 *       after warmup (consolidation should reduce or keep stable)</li>
 *   <li><b>Execution counts</b> — Per-segment execution counts; must be monotonically
 *       increasing (each iteration increments counters)</li>
 *   <li><b>Phase violations</b> — Any unexpected phase transition (e.g., REPLAY → CAPTURE
 *       after shapes are frozen, or fallback to slot-by-slot when replay should be active)</li>
 * </ul>
 *
 * <h3>Integration with NativeOps</h3>
 * <p>When a compiled native plan handle is available, this tracker reads metadata
 * via JNI hooks on NativeOps: {@code getPlanSegmentCount}, {@code getPlanReplaySignatureHash},
 * {@code getPlanReplayUnitCount}, {@code getSegmentExecutionCount}.</p>
 *
 * <h3>Usage</h3>
 * <pre>
 *   ReplayMetadataTracker tracker = new ReplayMetadataTracker("OPTIMAL");
 *   tracker.recordIteration(0, nativeOps, planHandle);  // warmup
 *   for (int i = 1; i &lt; N; i++) {
 *       tracker.recordIteration(i, nativeOps, planHandle);
 *   }
 *   List&lt;String&gt; violations = tracker.getPhaseViolations();
 *   ValidationResult result = tracker.buildValidationResult();
 *   tracker.close();
 * </pre>
 */
public class ReplayMetadataTracker implements AutoCloseable {

    private final String configName;
    private final Map<Integer, List<Long>> segmentSignatures = new LinkedHashMap<>();
    private final Map<Integer, List<Integer>> segmentUnitCounts = new LinkedHashMap<>();
    private final Map<Integer, List<Integer>> segmentExecCounts = new LinkedHashMap<>();
    private final List<String> phaseViolations = new ArrayList<>();
    private final List<PhaseTransition> phaseTransitions = new ArrayList<>();
    private Phase currentPhase = Phase.COMPILE;
    private int totalIterations = 0;
    private boolean nativeHandleAvailable = false;

    public ReplayMetadataTracker(String configName) {
        this.configName = configName;
    }

    /**
     * Record metadata from one iteration. Uses NativeOps introspection when
     * a compiled plan handle is available.
     *
     * @param iteration  zero-based iteration index (0 = warmup)
     * @param nativeOps  NativeOps instance for JNI metadata reads
     * @param planHandle compiled native plan handle (may be null)
     */
    public void recordIteration(int iteration, org.nd4j.nativeblas.NativeOps nativeOps,
                                 org.bytedeco.javacpp.Pointer planHandle) {
        totalIterations++;
        nativeHandleAvailable = (planHandle != null);

        if (planHandle == null) {
            // No native plan — record as slot-by-slot fallback
            phaseViolations.add("Iteration " + iteration + ": no native plan handle — falling back to slot-by-slot");
            return;
        }

        try {
            int segCount = nativeOps.getPlanSegmentCount(planHandle);
            for (int i = 0; i < segCount; i++) {
                long sig = nativeOps.getPlanReplaySignatureHash(planHandle, i);
                int units = nativeOps.getPlanReplayUnitCount(planHandle, i);
                int execCount = nativeOps.getSegmentExecutionCount(planHandle, i);

                segmentSignatures.computeIfAbsent(i, k -> new ArrayList<>()).add(sig);
                segmentUnitCounts.computeIfAbsent(i, k -> new ArrayList<>()).add(units);
                segmentExecCounts.computeIfAbsent(i, k -> new ArrayList<>()).add(execCount);
            }

            // Detect phase transitions
            detectPhaseTransition(iteration, nativeOps, planHandle);

        } catch (Exception e) {
            phaseViolations.add("Iteration " + iteration + ": failed to read replay metadata: " + e.getMessage());
        }
    }

    /**
     * Record metadata from one iteration without native plan handle
     * (e.g., SLOT_BY_SLOT mode where no native compilation occurred).
     */
    public void recordIterationSlotBySlot(int iteration) {
        totalIterations++;
        nativeHandleAvailable = false;
        PhaseTransition transition = new PhaseTransition(iteration, currentPhase, Phase.SLOT_BY_SLOT,
                "No native plan — slot-by-slot execution");
        phaseTransitions.add(transition);
        currentPhase = Phase.SLOT_BY_SLOT;
    }

    /**
     * Manually record a phase violation (e.g., unexpected fallback from replay to capture).
     */
    public void recordPhaseViolation(int iteration, String message) {
        phaseViolations.add("Iteration " + iteration + ": " + message);
    }

    /**
     * Manually record a phase transition.
     */
    public void recordPhaseTransition(int iteration, Phase from, Phase to, String reason) {
        phaseTransitions.add(new PhaseTransition(iteration, from, to, reason));
        currentPhase = to;
    }

    // ─── Validation Checks ──────────────────────────────────────────────────

    /**
     * Check replay signature stability: all iterations must have the same
     * signature hash per segment.
     *
     * @return list of failure messages (empty if stable)
     */
    public List<String> checkSignatureStability() {
        List<String> failures = new ArrayList<>();
        for (Map.Entry<Integer, List<Long>> entry : segmentSignatures.entrySet()) {
            int segIdx = entry.getKey();
            List<Long> sigs = entry.getValue();
            if (sigs.size() < 2) continue;
            long baseline = sigs.get(0);
            for (int i = 1; i < sigs.size(); i++) {
                if (sigs.get(i) != baseline) {
                    failures.add(String.format(
                            "seg[%d] replay signature changed at iteration %d: %s → %s",
                            segIdx, i, Long.toHexString(baseline), Long.toHexString(sigs.get(i))));
                }
            }
        }
        return failures;
    }

    /**
     * Check execution count monotonicity: per-segment exec counts must be
     * non-decreasing across iterations.
     *
     * @return list of failure messages (empty if monotonic)
     */
    public List<String> checkExecCountMonotonicity() {
        List<String> failures = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : segmentExecCounts.entrySet()) {
            int segIdx = entry.getKey();
            List<Integer> counts = entry.getValue();
            for (int i = 1; i < counts.size(); i++) {
                if (counts.get(i) < counts.get(i - 1)) {
                    failures.add(String.format(
                            "seg[%d] execution count decreased at iteration %d: %d → %d",
                            segIdx, i, counts.get(i - 1), counts.get(i)));
                }
            }
        }
        return failures;
    }

    /**
     * Check that replay unit counts do not increase after warmup.
     * Consolidation should reduce or keep unit counts stable.
     *
     * @return list of failure messages (empty if stable/decreasing)
     */
    public List<String> checkUnitCountStability() {
        List<String> failures = new ArrayList<>();
        for (Map.Entry<Integer, List<Integer>> entry : segmentUnitCounts.entrySet()) {
            int segIdx = entry.getKey();
            List<Integer> units = entry.getValue();
            // Skip warmup (iteration 0) — check iterations 1+
            for (int i = 2; i < units.size(); i++) {
                if (units.get(i) > units.get(1)) {
                    failures.add(String.format(
                            "seg[%d] replay unit count increased after warmup at iteration %d: %d → %d",
                            segIdx, i, units.get(1), units.get(i)));
                }
            }
        }
        return failures;
    }

    /**
     * Get all phase violations recorded during tracking.
     */
    public List<String> getPhaseViolations() {
        return Collections.unmodifiableList(phaseViolations);
    }

    /**
     * Get total iterations recorded.
     */
    public int getTotalIterations() {
        return totalIterations;
    }

    /**
     * Whether a native plan handle was available during tracking.
     */
    public boolean isNativeHandleAvailable() {
        return nativeHandleAvailable;
    }

    /**
     * Build a ReplayMetadata snapshot for inclusion in ValidationResult.
     */
    public ReplayMetadata buildMetadata() {
        Map<Integer, Long> latestSignatures = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<Long>> e : segmentSignatures.entrySet()) {
            List<Long> sigs = e.getValue();
            if (!sigs.isEmpty()) {
                latestSignatures.put(e.getKey(), sigs.get(sigs.size() - 1));
            }
        }
        Map<Integer, Integer> latestUnitCounts = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<Integer>> e : segmentUnitCounts.entrySet()) {
            List<Integer> units = e.getValue();
            if (!units.isEmpty()) {
                latestUnitCounts.put(e.getKey(), units.get(units.size() - 1));
            }
        }
        Map<Integer, Integer> latestExecCounts = new LinkedHashMap<>();
        for (Map.Entry<Integer, List<Integer>> e : segmentExecCounts.entrySet()) {
            List<Integer> execs = e.getValue();
            if (!execs.isEmpty()) {
                latestExecCounts.put(e.getKey(), execs.get(execs.size() - 1));
            }
        }

        return new ReplayMetadata(
                configName, currentPhase, new ArrayList<>(phaseTransitions),
                Collections.unmodifiableList(new ArrayList<>(phaseViolations)),
                Collections.unmodifiableMap(latestSignatures),
                Collections.unmodifiableMap(latestUnitCounts),
                Collections.unmodifiableMap(latestExecCounts),
                totalIterations, nativeHandleAvailable);
    }

    @Override
    public void close() {
        // No native resources to free — metadata is in Java objects
    }

    // ─── Phase Detection ────────────────────────────────────────────────────

    private void detectPhaseTransition(int iteration, org.nd4j.nativeblas.NativeOps nativeOps,
                                        org.bytedeco.javacpp.Pointer planHandle) {
        // Infer phase from segment characteristics
        int segCount = nativeOps.getPlanSegmentCount(planHandle);
        int capturedSegments = 0;
        for (int i = 0; i < segCount; i++) {
            int units = nativeOps.getPlanReplayUnitCount(planHandle, i);
            if (units > 0) capturedSegments++;
        }

        Phase newPhase;
        String reason;
        if (iteration == 0) {
            if (capturedSegments == segCount) {
                newPhase = Phase.REPLAY;
                reason = "All segments captured on warmup";
            } else if (capturedSegments > 0) {
                newPhase = Phase.CAPTURE;
                reason = capturedSegments + "/" + segCount + " segments captured";
            } else {
                newPhase = Phase.COMPILE;
                reason = "No segments captured — compile-only";
            }
        } else {
            if (capturedSegments == segCount) {
                newPhase = Phase.REPLAY;
                reason = "Full replay active (" + segCount + " segments)";
            } else if (capturedSegments > 0) {
                newPhase = Phase.CAPTURE;
                reason = "Partial capture: " + capturedSegments + "/" + segCount;
            } else {
                newPhase = Phase.COMPILE;
                reason = "No replay segments detected";
            }
        }

        if (newPhase != currentPhase) {
            phaseTransitions.add(new PhaseTransition(iteration, currentPhase, newPhase, reason));
            currentPhase = newPhase;
        }

        // Detect violations: REPLAY → CAPTURE after warmup is unexpected
        if (iteration > 0 && currentPhase == Phase.CAPTURE && phaseTransitions.size() > 1) {
            PhaseTransition prev = phaseTransitions.get(phaseTransitions.size() - 2);
            if (prev.toPhase == Phase.REPLAY) {
                phaseViolations.add("Unexpected fallback from REPLAY to CAPTURE at iteration " + iteration);
            }
        }
    }

    // ─── Phase Enum ─────────────────────────────────────────────────────────

    public enum Phase {
        /** Initial plan compilation — no capture yet. */
        COMPILE,
        /** Graph capture in progress — segments being recorded. */
        CAPTURE,
        /** Frozen replay active — replaying captured graphs. */
        REPLAY,
        /** Slot-by-slot execution — no native plan or fallback. */
        SLOT_BY_SLOT
    }

    // ─── Inner Classes ──────────────────────────────────────────────────────

    static class PhaseTransition {
        final int iteration;
        final Phase fromPhase;
        final Phase toPhase;
        final String reason;

        PhaseTransition(int iteration, Phase from, Phase to, String reason) {
            this.iteration = iteration;
            this.fromPhase = from;
            this.toPhase = to;
            this.reason = reason;
        }
    }

    /**
     * Immutable snapshot of replay metadata at a point in time.
     */
    @Getter
    public static class ReplayMetadata {
        private final String configName;
        private final Phase currentPhase;
        private final List<PhaseTransition> phaseTransitions;
        private final List<String> phaseViolations;
        private final Map<Integer, Long> segmentSignatures;
        private final Map<Integer, Integer> segmentUnitCounts;
        private final Map<Integer, Integer> segmentExecCounts;
        private final int totalIterations;
        private final boolean nativeHandleAvailable;

        public ReplayMetadata(String configName, Phase currentPhase,
                               List<PhaseTransition> phaseTransitions,
                               List<String> phaseViolations,
                               Map<Integer, Long> segmentSignatures,
                               Map<Integer, Integer> segmentUnitCounts,
                               Map<Integer, Integer> segmentExecCounts,
                               int totalIterations, boolean nativeHandleAvailable) {
            this.configName = configName;
            this.currentPhase = currentPhase;
            this.phaseTransitions = Collections.unmodifiableList(new ArrayList<>(phaseTransitions));
            this.phaseViolations = Collections.unmodifiableList(new ArrayList<>(phaseViolations));
            this.segmentSignatures = Collections.unmodifiableMap(new LinkedHashMap<>(segmentSignatures));
            this.segmentUnitCounts = Collections.unmodifiableMap(new LinkedHashMap<>(segmentUnitCounts));
            this.segmentExecCounts = Collections.unmodifiableMap(new LinkedHashMap<>(segmentExecCounts));
            this.totalIterations = totalIterations;
            this.nativeHandleAvailable = nativeHandleAvailable;
        }

        /**
         * Human-readable replay metadata report.
         */
        public String toReport() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== Replay Metadata ===\n");
            sb.append("Config:           ").append(configName).append("\n");
            sb.append("Current phase:    ").append(currentPhase).append("\n");
            sb.append("Iterations:       ").append(totalIterations).append("\n");
            sb.append("Native handle:    ").append(nativeHandleAvailable ? "yes" : "no").append("\n");

            if (!phaseTransitions.isEmpty()) {
                sb.append("\nPhase transitions:\n");
                for (PhaseTransition t : phaseTransitions) {
                    sb.append(String.format("  iter %d: %s → %s (%s)\n",
                            t.iteration, t.fromPhase, t.toPhase, t.reason));
                }
            }

            if (!phaseViolations.isEmpty()) {
                sb.append("\nPhase violations:\n");
                for (String v : phaseViolations) {
                    sb.append("  - ").append(v).append("\n");
                }
            }

            if (!segmentSignatures.isEmpty()) {
                sb.append("\nSegment signatures:\n");
                for (Map.Entry<Integer, Long> e : segmentSignatures.entrySet()) {
                    sb.append(String.format("  seg[%d]: %s\n", e.getKey(), Long.toHexString(e.getValue())));
                }
            }

            if (!segmentUnitCounts.isEmpty()) {
                sb.append("\nSegment replay unit counts:\n");
                for (Map.Entry<Integer, Integer> e : segmentUnitCounts.entrySet()) {
                    sb.append(String.format("  seg[%d]: %d units\n", e.getKey(), e.getValue()));
                }
            }

            if (!segmentExecCounts.isEmpty()) {
                sb.append("\nSegment execution counts:\n");
                for (Map.Entry<Integer, Integer> e : segmentExecCounts.entrySet()) {
                    sb.append(String.format("  seg[%d]: %d execs\n", e.getKey(), e.getValue()));
                }
            }

            return sb.toString();
        }
    }
}
