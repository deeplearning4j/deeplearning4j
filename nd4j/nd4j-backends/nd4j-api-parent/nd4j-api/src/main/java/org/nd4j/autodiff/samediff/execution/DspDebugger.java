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

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.*;

/**
 * DSP Debug Framework: diagnose op issues within segments.
 *
 * <p>Provides tools to inspect, validate, and debug DSP plan execution at the
 * op level. Use this to find which ops in a segment are causing errors, detect
 * NaN/Inf/stale data, and identify risky ops before they cause CUDA graph failures.</p>
 *
 * <h3>Usage:</h3>
 * <pre>
 *   SameDiff sd = ...;
 *   DspDebugger debugger = DspDebugger.attach(sd);
 *
 *   // Analyze the plan structure
 *   DspDebugger.PlanReport report = debugger.analyzePlan();
 *   System.out.println(report);
 *
 *   // Validate one execution step
 *   Map&lt;String, INDArray&gt; placeholders = Map.of("input", input);
 *   DspDebugger.StepReport stepReport = debugger.validateStep(placeholders, "output");
 *   if (stepReport.hasErrors()) {
 *       System.err.println(stepReport);
 *   }
 * </pre>
 */
@Slf4j
public class DspDebugger {

    // ─── Slot flag bit positions (must match NativeOps.h) ──────────────
    public static final int FLAG_VIEW_CAPABLE       = 1 << 0;
    public static final int FLAG_DATA_DEPENDENT     = 1 << 1;
    public static final int FLAG_SHAPE_DEPENDS_ON_VALUES = 1 << 2;
    public static final int FLAG_IDENTITY           = 1 << 3;
    public static final int FLAG_IN_PLACE_FUSED     = 1 << 4;
    public static final int FLAG_FUSED_CHAIN_HEAD   = 1 << 5;
    public static final int FLAG_FUSED_CHAIN_TAIL   = 1 << 6;
    public static final int FLAG_NEEDS_ZEROED       = 1 << 7;
    public static final int FLAG_NEEDS_INT_LONG_SYNC = 1 << 8;
    public static final int FLAG_SHAPE_STATIC       = 1 << 9;
    public static final int FLAG_FROZEN_CONSTANT    = 1 << 10;

    private final SameDiff sd;

    private DspDebugger(SameDiff sd) {
        this.sd = sd;
    }

    /**
     * Attach a debugger to a SameDiff instance.
     * Enables DSP if not already enabled.
     */
    public static DspDebugger attach(SameDiff sd) {
        sd.setDspAutoCompileEnabled(true);
        sd.setDspNativeAutoCompileEnabled(true);
        return new DspDebugger(sd);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Plan Analysis
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Analyze the compiled plan. Triggers compilation if needed.
     * Returns a structured report of segments, ops, and risk factors.
     */
    public PlanReport analyzePlan() {
        // Trigger plan compilation by doing a dummy output call if needed
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        Pointer handle = executor != null ? executor.getNativePlanHandle() : null;

        if (handle == null || handle.isNull()) {
            return new PlanReport("Plan not compiled. Execute once to compile.");
        }

        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int numSlots = ops.getPlanNumSlots(handle);
        int numSegments = ops.getPlanNumSegments(handle);
        int planPhase = ops.getPlanPhase(handle);

        List<SlotInfo> slots = new ArrayList<>();
        for (int i = 0; i < numSlots; i++) {
            String opName = ops.getPlanSlotOpName(handle, i);
            int flags = ops.getPlanSlotFlags(handle, i);
            int stateCode = ops.getPlanSlotState(handle, i);
            SlotState state = SlotState.fromNativeCode(stateCode);
            slots.add(new SlotInfo(i, opName != null ? opName : "unknown", flags, state));
        }

        List<SegmentReport> segments = new ArrayList<>();
        for (int s = 0; s < numSegments; s++) {
            int execCount = ops.getPlanSegmentExecutionCount(handle, s);
            int phaseCode = ops.getPlanSegmentExecutionPhase(handle, s);
            boolean capturable = ops.isPlanSegmentCapturable(handle, s);
            boolean captureFailed = ops.isPlanSegmentCaptureFailed(handle, s);

            // Collect slots for this segment by scanning segment boundaries
            // (segments are contiguous ranges of slots)
            // We need start/end slot — get from the plan's Java side
            segments.add(new SegmentReport(s, capturable, captureFailed, execCount,
                    ExecutionPhase.fromNativeCode(phaseCode)));
        }

        return new PlanReport(numSlots, numSegments, PlanPhase.fromNativeCode(planPhase),
                slots, segments);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Step Validation
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Execute one step and validate all outputs for NaN/Inf/stale data.
     */
    public StepReport validateStep(Map<String, INDArray> placeholders, String... outputs) {
        StepReport report = new StepReport();

        try {
            Map<String, INDArray> results = sd.output(placeholders, outputs);

            for (Map.Entry<String, INDArray> entry : results.entrySet()) {
                INDArray arr = entry.getValue();
                String name = entry.getKey();

                if (arr == null) {
                    report.addError(name, "Output is NULL");
                    continue;
                }

                if (arr.wasClosed()) {
                    report.addError(name, "Output array is CLOSED (use-after-free)");
                    continue;
                }

                double sum = arr.sumNumber().doubleValue();
                if (Double.isNaN(sum)) {
                    report.addError(name, "Output contains NaN — likely stale/uninitialized buffer");
                }
                if (Double.isInfinite(sum)) {
                    report.addError(name, "Output contains Inf — likely numerical overflow");
                }

                // Check for all-zeros (suspicious stale data)
                if (arr.length() > 0) {
                    double norm = arr.norm2Number().doubleValue();
                    if (norm == 0.0 && arr.dataType().isFPType()) {
                        report.addWarning(name, "Output is all-zeros — may be stale/uninitialized");
                    }
                }
            }
        } catch (Exception e) {
            report.addError("EXECUTION", e.getClass().getSimpleName() + ": " + e.getMessage());
        }

        return report;
    }

    /**
     * Execute multiple steps with different random inputs and validate each.
     * Detects stale data by verifying outputs change between steps.
     */
    public MultiStepReport validateMultipleSteps(int numSteps, String placeholderName,
                                                  long[] shape, DataType dtype,
                                                  String... outputs) {
        MultiStepReport report = new MultiStepReport(numSteps);
        List<Map<String, INDArray>> allResults = new ArrayList<>();

        for (int step = 0; step < numSteps; step++) {
            INDArray input = Nd4j.randn(dtype, shape).muli(step + 1);
            Map<String, INDArray> placeholders = Map.of(placeholderName, input);
            StepReport stepReport = validateStep(placeholders, outputs);
            report.addStepReport(step, stepReport);

            if (!stepReport.hasErrors()) {
                Map<String, INDArray> results = sd.output(placeholders, outputs);
                Map<String, INDArray> duped = new LinkedHashMap<>();
                for (var e : results.entrySet()) {
                    duped.put(e.getKey(), e.getValue().dup());
                }
                allResults.add(duped);
            }
        }

        // Check for stale data: consecutive steps should produce different outputs
        for (int i = 1; i < allResults.size(); i++) {
            for (String outputName : outputs) {
                INDArray prev = allResults.get(i - 1).get(outputName);
                INDArray curr = allResults.get(i).get(outputName);
                if (prev != null && curr != null) {
                    double diff = curr.sub(prev).norm2Number().doubleValue();
                    if (diff < 1e-10) {
                        report.addStaleDataWarning(i, outputName,
                                "Output identical to previous step — STALE DATA suspected");
                    }
                }
            }
        }

        return report;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Report Classes
    // ═══════════════════════════════════════════════════════════════════════

    /** Per-slot metadata from the native plan. */
    public static class SlotInfo {
        public final int index;
        public final String opName;
        public final int flags;
        public final SlotState state;

        public SlotInfo(int index, String opName, int flags, SlotState state) {
            this.index = index;
            this.opName = opName;
            this.flags = flags;
            this.state = state;
        }

        public boolean isViewCapable()       { return (flags & FLAG_VIEW_CAPABLE) != 0; }
        public boolean isDataDependent()     { return (flags & FLAG_DATA_DEPENDENT) != 0; }
        public boolean isShapeDependsOnValues() { return (flags & FLAG_SHAPE_DEPENDS_ON_VALUES) != 0; }
        public boolean isIdentity()          { return (flags & FLAG_IDENTITY) != 0; }
        public boolean isInPlaceFused()      { return (flags & FLAG_IN_PLACE_FUSED) != 0; }
        public boolean isFusedChainHead()    { return (flags & FLAG_FUSED_CHAIN_HEAD) != 0; }
        public boolean isFusedChainTail()    { return (flags & FLAG_FUSED_CHAIN_TAIL) != 0; }
        public boolean needsZeroed()         { return (flags & FLAG_NEEDS_ZEROED) != 0; }
        public boolean needsIntLongSync()    { return (flags & FLAG_NEEDS_INT_LONG_SYNC) != 0; }
        public boolean isShapeStatic()       { return (flags & FLAG_SHAPE_STATIC) != 0; }
        public boolean isFrozenConstant()    { return (flags & FLAG_FROZEN_CONSTANT) != 0; }

        /** Check if this op has properties that make it risky for graph capture. */
        public boolean isRisky() {
            return isDataDependent() || isShapeDependsOnValues() || isViewCapable();
        }

        /** Human-readable risk description. */
        public String getRiskDescription() {
            List<String> risks = new ArrayList<>();
            if (isDataDependent()) risks.add("DATA_DEPENDENT(output shape depends on input data)");
            if (isShapeDependsOnValues()) risks.add("SHAPE_DEPENDS_ON_VALUES(shape changes with input values)");
            if (isViewCapable()) risks.add("VIEW_CAPABLE(output may share buffer with input)");
            if (needsIntLongSync()) risks.add("NEEDS_INT_LONG_SYNC(requires host→device sync)");
            return risks.isEmpty() ? "none" : String.join(", ", risks);
        }

        public String formatFlags() {
            List<String> active = new ArrayList<>();
            if (isViewCapable()) active.add("view");
            if (isDataDependent()) active.add("data_dep");
            if (isShapeDependsOnValues()) active.add("shape_dep_vals");
            if (isIdentity()) active.add("identity");
            if (isInPlaceFused()) active.add("inplace");
            if (isFusedChainHead()) active.add("fused_head");
            if (isFusedChainTail()) active.add("fused_tail");
            if (needsZeroed()) active.add("zeroed");
            if (needsIntLongSync()) active.add("int_sync");
            if (isShapeStatic()) active.add("static");
            if (isFrozenConstant()) active.add("frozen_const");
            return active.isEmpty() ? "-" : String.join("|", active);
        }

        @Override
        public String toString() {
            return String.format("slot[%d] %-25s state=%-15s flags=[%s] %s",
                    index, opName,
                    state != null ? state.name() : "?",
                    formatFlags(),
                    isRisky() ? " *** RISKY: " + getRiskDescription() : "");
        }
    }

    /** Per-segment report. */
    public static class SegmentReport {
        public final int index;
        public final boolean capturable;
        public final boolean captureFailed;
        public final int executionCount;
        public final ExecutionPhase phase;

        SegmentReport(int index, boolean capturable, boolean captureFailed,
                      int executionCount, ExecutionPhase phase) {
            this.index = index;
            this.capturable = capturable;
            this.captureFailed = captureFailed;
            this.executionCount = executionCount;
            this.phase = phase;
        }

        @Override
        public String toString() {
            return String.format("seg[%d] capturable=%s captureFailed=%s execCount=%d phase=%s",
                    index, capturable, captureFailed, executionCount,
                    phase != null ? phase.name() : "?");
        }
    }

    /** Full plan analysis report. */
    public static class PlanReport {
        public final int numSlots;
        public final int numSegments;
        public final PlanPhase planPhase;
        public final List<SlotInfo> slots;
        public final List<SegmentReport> segments;
        public final String errorMessage;

        PlanReport(String errorMessage) {
            this.numSlots = 0;
            this.numSegments = 0;
            this.planPhase = null;
            this.slots = Collections.emptyList();
            this.segments = Collections.emptyList();
            this.errorMessage = errorMessage;
        }

        PlanReport(int numSlots, int numSegments, PlanPhase planPhase,
                   List<SlotInfo> slots, List<SegmentReport> segments) {
            this.numSlots = numSlots;
            this.numSegments = numSegments;
            this.planPhase = planPhase;
            this.slots = slots;
            this.segments = segments;
            this.errorMessage = null;
        }

        /** Get all risky ops that could cause issues during graph capture/replay. */
        public List<SlotInfo> getRiskyOps() {
            List<SlotInfo> risky = new ArrayList<>();
            for (SlotInfo s : slots) {
                if (s.isRisky()) risky.add(s);
            }
            return risky;
        }

        /** Get op histogram: opName → count. */
        public Map<String, Integer> getOpHistogram() {
            Map<String, Integer> hist = new LinkedHashMap<>();
            for (SlotInfo s : slots) {
                hist.merge(s.opName, 1, Integer::sum);
            }
            return hist;
        }

        /** Get ops that failed to freeze (still dynamic after frozen execution). */
        public List<SlotInfo> getUnfrozenOps() {
            List<SlotInfo> unfrozen = new ArrayList<>();
            for (SlotInfo s : slots) {
                if (s.state != null && !s.state.isAtLeast(SlotState.FROZEN) && !s.isIdentity()) {
                    unfrozen.add(s);
                }
            }
            return unfrozen;
        }

        @Override
        public String toString() {
            if (errorMessage != null) return "PlanReport: " + errorMessage;

            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append("  DSP Plan Report\n");
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append(String.format("  Slots: %d | Segments: %d | Phase: %s\n",
                    numSlots, numSegments, planPhase));
            sb.append("───────────────────────────────────────────────────────\n");

            // Op histogram
            sb.append("  Op Histogram:\n");
            Map<String, Integer> hist = getOpHistogram();
            hist.entrySet().stream()
                    .sorted(Map.Entry.<String, Integer>comparingByValue().reversed())
                    .forEach(e -> sb.append(String.format("    %-25s %d\n", e.getKey(), e.getValue())));

            // Segments
            sb.append("───────────────────────────────────────────────────────\n");
            sb.append("  Segments:\n");
            for (SegmentReport seg : segments) {
                sb.append("    ").append(seg).append("\n");
            }

            // Risky ops
            List<SlotInfo> risky = getRiskyOps();
            if (!risky.isEmpty()) {
                sb.append("───────────────────────────────────────────────────────\n");
                sb.append("  ⚠ Risky Ops (" + risky.size() + "):\n");
                for (SlotInfo s : risky) {
                    sb.append("    ").append(s).append("\n");
                }
            }

            // All slots (condensed)
            sb.append("───────────────────────────────────────────────────────\n");
            sb.append("  All Slots:\n");
            for (SlotInfo s : slots) {
                sb.append("    ").append(s).append("\n");
            }
            sb.append("═══════════════════════════════════════════════════════\n");
            return sb.toString();
        }
    }

    /** Single-step validation report. */
    public static class StepReport {
        private final List<String> errors = new ArrayList<>();
        private final List<String> warnings = new ArrayList<>();

        void addError(String output, String message) {
            errors.add("[ERROR] " + output + ": " + message);
        }

        void addWarning(String output, String message) {
            warnings.add("[WARN] " + output + ": " + message);
        }

        public boolean hasErrors() { return !errors.isEmpty(); }
        public boolean hasWarnings() { return !warnings.isEmpty(); }
        public List<String> getErrors() { return errors; }
        public List<String> getWarnings() { return warnings; }

        @Override
        public String toString() {
            if (errors.isEmpty() && warnings.isEmpty()) return "StepReport: OK";
            StringBuilder sb = new StringBuilder("StepReport:\n");
            errors.forEach(e -> sb.append("  ").append(e).append("\n"));
            warnings.forEach(w -> sb.append("  ").append(w).append("\n"));
            return sb.toString();
        }
    }

    /** Multi-step validation report with stale data detection. */
    public static class MultiStepReport {
        private final int numSteps;
        private final Map<Integer, StepReport> stepReports = new LinkedHashMap<>();
        private final List<String> staleDataWarnings = new ArrayList<>();

        MultiStepReport(int numSteps) {
            this.numSteps = numSteps;
        }

        void addStepReport(int step, StepReport report) {
            stepReports.put(step, report);
        }

        void addStaleDataWarning(int step, String output, String message) {
            staleDataWarnings.add("[STALE] step " + step + " " + output + ": " + message);
        }

        public boolean hasErrors() {
            return stepReports.values().stream().anyMatch(StepReport::hasErrors);
        }

        public boolean hasStaleData() {
            return !staleDataWarnings.isEmpty();
        }

        public int getErrorCount() {
            return stepReports.values().stream().mapToInt(r -> r.getErrors().size()).sum();
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("MultiStepReport: %d steps, %d errors, %d stale warnings\n",
                    numSteps, getErrorCount(), staleDataWarnings.size()));
            for (var entry : stepReports.entrySet()) {
                StepReport r = entry.getValue();
                if (r.hasErrors() || r.hasWarnings()) {
                    sb.append("  Step ").append(entry.getKey()).append(": ").append(r).append("\n");
                }
            }
            staleDataWarnings.forEach(w -> sb.append("  ").append(w).append("\n"));
            return sb.toString();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Graph Replay Analysis
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Analyze graph replay readiness and phase progression across segments.
     * Enables GRAPH_REPLAY diagnostics category during analysis.
     *
     * <p>Validates:</p>
     * <ul>
     *   <li>Segment capturable flags vs actual capture success</li>
     *   <li>Phase progression (WARMUP -> COMPILING -> COMPILED -> REPLAYING)</li>
     *   <li>Pointer stability for replay-ready segments</li>
     *   <li>Capture buffer counts and tracked pointer consistency</li>
     * </ul>
     */
    public GraphReplayReport analyzeGraphReplay() {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Enable GRAPH_REPLAY diagnostics for the duration of analysis
        int previousMask = ops.dspDiagGetEnabledMask();
        ops.dspDiagEnableCategories(DspDiagnostics.GRAPH_REPLAY | DspDiagnostics.SEGMENT);

        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        Pointer handle = executor != null ? executor.getNativePlanHandle() : null;

        if (handle == null || handle.isNull()) {
            ops.dspDiagSetCategories(previousMask);
            return new GraphReplayReport("Plan not compiled. Execute once to compile.");
        }

        int numSegments = ops.getPlanNumSegments(handle);
        int planPhaseCode = ops.getPlanPhase(handle);
        int pointersStable = ops.getPlanPointersStable(handle);
        int frozenExecCount = ops.getPlanFrozenExecutionCount(handle);

        List<SegmentReplayInfo> segmentInfos = new ArrayList<>();
        for (int s = 0; s < numSegments; s++) {
            boolean capturable = ops.isPlanSegmentCapturable(handle, s);
            boolean captureFailed = ops.isPlanSegmentCaptureFailed(handle, s);
            int replayState = ops.getPlanSegmentReplayState(handle, s);
            int replayCount = ops.getPlanSegmentReplayCount(handle, s);
            int execCount = ops.getPlanSegmentExecutionCount(handle, s);
            int phaseCode = ops.getPlanSegmentExecutionPhase(handle, s);
            String backendName = ops.getPlanSegmentBackendName(handle, s);
            int numCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(handle, s);
            String trackedPtrs = ops.getPlanSegmentTrackedPointers(handle, s);

            segmentInfos.add(new SegmentReplayInfo(s, capturable, captureFailed,
                    replayState, replayCount, execCount,
                    ExecutionPhase.fromNativeCode(phaseCode),
                    backendName != null ? backendName : "unknown",
                    numCaptureBuffers, trackedPtrs));
        }

        // Restore previous diagnostic mask
        ops.dspDiagSetCategories(previousMask);

        return new GraphReplayReport(numSegments,
                PlanPhase.fromNativeCode(planPhaseCode),
                pointersStable == 1, frozenExecCount, segmentInfos);
    }

    /**
     * Run N steps and track graph replay phase transitions per segment.
     * Returns a report showing which segments progressed through replay phases
     * and which are stuck or regressing.
     */
    public GraphReplayProgressReport trackReplayProgression(int numSteps,
                                                             Map<String, INDArray> placeholders,
                                                             String... outputs) {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        ops.dspDiagEnableCategories(DspDiagnostics.GRAPH_REPLAY);
        ops.dspDiagSetLevel(DspDiagnostics.LEVEL_FULL);

        GraphReplayProgressReport report = new GraphReplayProgressReport(numSteps);

        for (int step = 0; step < numSteps; step++) {
            try {
                sd.output(placeholders, outputs);
            } catch (Exception e) {
                report.addStepError(step, e.getMessage());
                continue;
            }

            Pointer handle = getHandle();
            if (handle == null || handle.isNull()) continue;

            int numSegments = ops.getPlanNumSegments(handle);
            int planPhaseCode = ops.getPlanPhase(handle);
            report.recordPlanPhase(step, PlanPhase.fromNativeCode(planPhaseCode));

            for (int s = 0; s < numSegments; s++) {
                int phaseCode = ops.getPlanSegmentExecutionPhase(handle, s);
                report.recordSegmentPhase(step, s, ExecutionPhase.fromNativeCode(phaseCode));
            }
        }

        return report;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stream Sync Analysis
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Validate output consistency across multiple steps to detect stream sync issues.
     *
     * <p>Stream sync problems manifest as:</p>
     * <ul>
     *   <li>Intermittently wrong outputs (data from previous step leaking)</li>
     *   <li>NaN/Inf appearing sporadically (partial writes visible)</li>
     *   <li>Outputs that are correct individually but appear in wrong order</li>
     * </ul>
     *
     * <p>This method enables STREAM_SYNC diagnostics and runs deterministic
     * input patterns, checking that outputs change monotonically and that no
     * stale data from prior steps bleeds through.</p>
     */
    public StreamSyncReport validateStreamSync(int numSteps,
                                                String placeholderName,
                                                long[] shape, DataType dtype,
                                                String... outputNames) {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int previousMask = ops.dspDiagGetEnabledMask();
        ops.dspDiagEnableCategories(DspDiagnostics.STREAM_SYNC | DspDiagnostics.EXECUTE);
        ops.dspDiagSetLevel(DspDiagnostics.LEVEL_FULL);

        StreamSyncReport report = new StreamSyncReport(numSteps);
        List<Map<String, INDArray>> allOutputs = new ArrayList<>();

        for (int step = 0; step < numSteps; step++) {
            // Deterministic input: scaled identity-like pattern so we can detect bleed
            INDArray input = Nd4j.ones(dtype, shape).muli(step + 1);

            try {
                Map<String, INDArray> result = sd.output(
                        Map.of(placeholderName, input), outputNames);

                Map<String, INDArray> duped = new LinkedHashMap<>();
                for (Map.Entry<String, INDArray> e : result.entrySet()) {
                    INDArray arr = e.getValue();
                    if (arr == null) {
                        report.addIssue(step, e.getKey(), StreamSyncIssue.NULL_OUTPUT,
                                "Output is NULL at step " + step);
                        continue;
                    }
                    if (arr.wasClosed()) {
                        report.addIssue(step, e.getKey(), StreamSyncIssue.CLOSED_BUFFER,
                                "Output buffer closed at step " + step);
                        continue;
                    }

                    // Check for NaN/Inf
                    double sum = arr.sumNumber().doubleValue();
                    if (Double.isNaN(sum)) {
                        report.addIssue(step, e.getKey(), StreamSyncIssue.NAN_DETECTED,
                                "NaN in output — possible partial write from async kernel");
                    }
                    if (Double.isInfinite(sum)) {
                        report.addIssue(step, e.getKey(), StreamSyncIssue.INF_DETECTED,
                                "Inf in output — possible numerical issue or stale data");
                    }

                    duped.put(e.getKey(), arr.dup());
                }
                allOutputs.add(duped);
            } catch (Exception e) {
                report.addIssue(step, "EXECUTION", StreamSyncIssue.EXECUTION_ERROR,
                        e.getClass().getSimpleName() + ": " + e.getMessage());
            }
        }

        // Cross-step consistency: detect stale data bleed and output regression
        for (int i = 1; i < allOutputs.size(); i++) {
            for (String outputName : outputNames) {
                INDArray prev = allOutputs.get(i - 1).get(outputName);
                INDArray curr = allOutputs.get(i).get(outputName);
                if (prev == null || curr == null) continue;

                double diff = curr.sub(prev).norm2Number().doubleValue();
                if (diff < 1e-10) {
                    report.addIssue(i, outputName, StreamSyncIssue.STALE_DATA,
                            "Output identical to step " + (i - 1) + " — stream sync may be missing");
                }

                // Check for backward regression: if scaling input up should scale output
                double prevNorm = prev.norm2Number().doubleValue();
                double currNorm = curr.norm2Number().doubleValue();
                if (prevNorm > 0 && currNorm > 0) {
                    // With linear ops, larger input should give proportionally larger output
                    // A sudden drop could indicate data from a wrong step
                    double ratio = currNorm / prevNorm;
                    if (i > 2 && ratio < 0.01) {
                        report.addIssue(i, outputName, StreamSyncIssue.OUTPUT_REGRESSION,
                                String.format("Output norm dropped %.2fx vs previous — possible out-of-order execution",
                                        1.0 / ratio));
                    }
                }
            }
        }

        ops.dspDiagSetCategories(previousMask);
        return report;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Multi-Device Analysis
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Analyze multi-device placement and transfer patterns for the compiled plan.
     * Reports which segments target which devices and identifies potential
     * cross-device transfer bottlenecks.
     */
    public MultiDeviceReport analyzeMultiDevice() {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();
        int previousMask = ops.dspDiagGetEnabledMask();
        ops.dspDiagEnableCategories(DspDiagnostics.MULTI_DEVICE | DspDiagnostics.TRANSFER);

        Pointer handle = getHandle();
        if (handle == null || handle.isNull()) {
            ops.dspDiagSetCategories(previousMask);
            return new MultiDeviceReport("Plan not compiled.");
        }

        int numSegments = ops.getPlanNumSegments(handle);
        int numSlots = ops.getPlanNumSlots(handle);

        List<SegmentDeviceInfo> segmentDevices = new ArrayList<>();
        for (int s = 0; s < numSegments; s++) {
            String backendName = ops.getPlanSegmentBackendName(handle, s);
            int execCount = ops.getPlanSegmentExecutionCount(handle, s);
            boolean capturable = ops.isPlanSegmentCapturable(handle, s);
            String statsJson = ops.getPlanSegmentStatisticsJson(handle, s);
            int numCaptureBuffers = ops.getPlanSegmentNumCaptureBuffers(handle, s);

            segmentDevices.add(new SegmentDeviceInfo(s,
                    backendName != null ? backendName : "unknown",
                    execCount, capturable, numCaptureBuffers, statsJson));
        }

        // Collect slot data types to identify potential cross-type transfer issues
        List<SlotInfo> slots = new ArrayList<>();
        for (int i = 0; i < numSlots; i++) {
            String opName = ops.getPlanSlotOpName(handle, i);
            int flags = ops.getPlanSlotFlags(handle, i);
            int stateCode = ops.getPlanSlotState(handle, i);
            slots.add(new SlotInfo(i, opName != null ? opName : "unknown", flags,
                    SlotState.fromNativeCode(stateCode)));
        }

        // Check replay cache device stats
        String replayCacheDeviceStats = "";
        try {
            replayCacheDeviceStats = ops.getReplayCacheDeviceStatsJson();
        } catch (Exception ignored) {}

        ops.dspDiagSetCategories(previousMask);
        return new MultiDeviceReport(numSegments, numSlots, segmentDevices,
                slots, replayCacheDeviceStats);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Full Diagnostic Sweep
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Run a comprehensive diagnostic sweep: plan analysis + graph replay +
     * stream sync validation + multi-device analysis.
     *
     * @param warmupSteps   Steps to run before analysis (to populate plan state)
     * @param validateSteps Steps for stream sync validation
     * @param placeholders  Input data for each step
     * @param outputs       Output variable names
     * @return Combined diagnostic report
     */
    public FullDiagnosticReport runFullDiagnostics(int warmupSteps, int validateSteps,
                                                    Map<String, INDArray> placeholders,
                                                    String... outputs) {
        NativeOps ops = NativeOpsHolder.getInstance().getDeviceNativeOps();

        // Enable ALL diagnostic categories at FULL level
        ops.dspDiagSetCategories(DspDiagnostics.ALL);
        ops.dspDiagSetLevel(DspDiagnostics.LEVEL_FULL);
        ops.dspDiagClear();

        // Warmup: populate plan state
        for (int i = 0; i < warmupSteps; i++) {
            try {
                sd.output(placeholders, outputs);
            } catch (Exception e) {
                log.warn("Warmup step {} failed: {}", i, e.getMessage());
            }
        }

        // Collect reports
        PlanReport planReport = analyzePlan();
        GraphReplayReport replayReport = analyzeGraphReplay();
        MultiDeviceReport deviceReport = analyzeMultiDevice();

        // Track replay progression over validate steps
        GraphReplayProgressReport progressReport =
                trackReplayProgression(validateSteps, placeholders, outputs);

        // Get the DSP diagnostic report from C++
        String nativeReport = ops.dspDiagGetPlanReport();
        String jsonReport = ops.dspDiagGetJsonReport();

        return new FullDiagnosticReport(planReport, replayReport, deviceReport,
                progressReport, nativeReport, jsonReport);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Helper
    // ═══════════════════════════════════════════════════════════════════════

    private Pointer getHandle() {
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor executor = session.getDynamicShapePlanExecutor();
        return executor != null ? executor.getNativePlanHandle() : null;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Graph Replay Report Classes
    // ═══════════════════════════════════════════════════════════════════════

    /** Replay state for a single segment. */
    public static class SegmentReplayInfo {
        public final int index;
        public final boolean capturable;
        public final boolean captureFailed;
        public final int replayState;
        public final int replayCount;
        public final int executionCount;
        public final ExecutionPhase phase;
        public final String backendName;
        public final int numCaptureBuffers;
        public final String trackedPointers;

        SegmentReplayInfo(int index, boolean capturable, boolean captureFailed,
                          int replayState, int replayCount, int executionCount,
                          ExecutionPhase phase, String backendName,
                          int numCaptureBuffers, String trackedPointers) {
            this.index = index;
            this.capturable = capturable;
            this.captureFailed = captureFailed;
            this.replayState = replayState;
            this.replayCount = replayCount;
            this.executionCount = executionCount;
            this.phase = phase;
            this.backendName = backendName;
            this.numCaptureBuffers = numCaptureBuffers;
            this.trackedPointers = trackedPointers;
        }

        /** Whether replay is actively working for this segment. */
        public boolean isReplaying() {
            return phase == ExecutionPhase.REPLAYING;
        }

        /** Whether capture was attempted but failed. */
        public boolean hasCaptureFailure() {
            return capturable && captureFailed;
        }

        /** Whether the segment is stuck in warmup despite multiple executions. */
        public boolean isStuckInWarmup() {
            return phase == ExecutionPhase.WARMUP && executionCount > 3;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("seg[%d] backend=%-12s phase=%-12s capturable=%s",
                    index, backendName, phase != null ? phase.name() : "?", capturable));
            if (captureFailed) sb.append(" CAPTURE_FAILED");
            sb.append(String.format(" replayState=%d replayCount=%d execCount=%d captureBuffers=%d",
                    replayState, replayCount, executionCount, numCaptureBuffers));
            return sb.toString();
        }
    }

    /** Full graph replay analysis report. */
    public static class GraphReplayReport {
        public final int numSegments;
        public final PlanPhase planPhase;
        public final boolean pointersStable;
        public final int frozenExecutionCount;
        public final List<SegmentReplayInfo> segments;
        public final String errorMessage;

        GraphReplayReport(String errorMessage) {
            this.numSegments = 0;
            this.planPhase = null;
            this.pointersStable = false;
            this.frozenExecutionCount = 0;
            this.segments = Collections.emptyList();
            this.errorMessage = errorMessage;
        }

        GraphReplayReport(int numSegments, PlanPhase planPhase, boolean pointersStable,
                          int frozenExecutionCount, List<SegmentReplayInfo> segments) {
            this.numSegments = numSegments;
            this.planPhase = planPhase;
            this.pointersStable = pointersStable;
            this.frozenExecutionCount = frozenExecutionCount;
            this.segments = segments;
            this.errorMessage = null;
        }

        /** Segments where graph capture failed. */
        public List<SegmentReplayInfo> getCaptureFailures() {
            List<SegmentReplayInfo> failures = new ArrayList<>();
            for (SegmentReplayInfo s : segments) {
                if (s.hasCaptureFailure()) failures.add(s);
            }
            return failures;
        }

        /** Segments stuck in warmup. */
        public List<SegmentReplayInfo> getStuckSegments() {
            List<SegmentReplayInfo> stuck = new ArrayList<>();
            for (SegmentReplayInfo s : segments) {
                if (s.isStuckInWarmup()) stuck.add(s);
            }
            return stuck;
        }

        /** Segments actively replaying. */
        public List<SegmentReplayInfo> getReplayingSegments() {
            List<SegmentReplayInfo> replaying = new ArrayList<>();
            for (SegmentReplayInfo s : segments) {
                if (s.isReplaying()) replaying.add(s);
            }
            return replaying;
        }

        /** Whether the plan has reached full replay state. */
        public boolean isFullyReplaying() {
            return planPhase == PlanPhase.REPLAYING && pointersStable;
        }

        @Override
        public String toString() {
            if (errorMessage != null) return "GraphReplayReport: " + errorMessage;

            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append("  Graph Replay Report\n");
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append(String.format("  Plan phase: %s | Pointers stable: %s | Frozen exec count: %d\n",
                    planPhase, pointersStable, frozenExecutionCount));

            List<SegmentReplayInfo> failures = getCaptureFailures();
            List<SegmentReplayInfo> stuck = getStuckSegments();
            List<SegmentReplayInfo> replaying = getReplayingSegments();

            sb.append(String.format("  Segments: %d total | %d replaying | %d capture-failed | %d stuck\n",
                    numSegments, replaying.size(), failures.size(), stuck.size()));
            sb.append("───────────────────────────────────────────────────────\n");

            for (SegmentReplayInfo seg : segments) {
                sb.append("  ").append(seg).append("\n");
            }

            if (!failures.isEmpty()) {
                sb.append("───────────────────────────────────────────────────────\n");
                sb.append("  CAPTURE FAILURES:\n");
                for (SegmentReplayInfo f : failures) {
                    sb.append("    seg[").append(f.index).append("] backend=")
                            .append(f.backendName).append("\n");
                }
            }

            sb.append("═══════════════════════════════════════════════════════\n");
            return sb.toString();
        }
    }

    /** Tracks phase transitions over multiple execution steps. */
    public static class GraphReplayProgressReport {
        private final int numSteps;
        private final List<PlanPhase> planPhases = new ArrayList<>();
        private final Map<Integer, List<ExecutionPhase>> segmentPhases = new LinkedHashMap<>();
        private final List<String> errors = new ArrayList<>();

        GraphReplayProgressReport(int numSteps) {
            this.numSteps = numSteps;
        }

        void recordPlanPhase(int step, PlanPhase phase) {
            while (planPhases.size() <= step) planPhases.add(null);
            planPhases.set(step, phase);
        }

        void recordSegmentPhase(int step, int segmentIdx, ExecutionPhase phase) {
            segmentPhases.computeIfAbsent(segmentIdx, k -> new ArrayList<>());
            List<ExecutionPhase> phases = segmentPhases.get(segmentIdx);
            while (phases.size() <= step) phases.add(null);
            phases.set(step, phase);
        }

        void addStepError(int step, String message) {
            errors.add("[ERROR] step " + step + ": " + message);
        }

        /** Check if any segment regressed to an earlier phase. */
        public List<String> getPhaseRegressions() {
            List<String> regressions = new ArrayList<>();
            for (Map.Entry<Integer, List<ExecutionPhase>> entry : segmentPhases.entrySet()) {
                List<ExecutionPhase> phases = entry.getValue();
                for (int i = 1; i < phases.size(); i++) {
                    ExecutionPhase prev = phases.get(i - 1);
                    ExecutionPhase curr = phases.get(i);
                    if (prev != null && curr != null
                            && curr.getNativeCode() < prev.getNativeCode()) {
                        regressions.add(String.format("seg[%d] regressed: %s -> %s at step %d",
                                entry.getKey(), prev.name(), curr.name(), i));
                    }
                }
            }
            return regressions;
        }

        /** Steps where the plan phase advanced. */
        public List<String> getPlanPhaseTransitions() {
            List<String> transitions = new ArrayList<>();
            for (int i = 1; i < planPhases.size(); i++) {
                PlanPhase prev = planPhases.get(i - 1);
                PlanPhase curr = planPhases.get(i);
                if (prev != curr && prev != null && curr != null) {
                    transitions.add(String.format("step %d: %s -> %s", i, prev.name(), curr.name()));
                }
            }
            return transitions;
        }

        public boolean hasErrors() { return !errors.isEmpty(); }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("GraphReplayProgress: %d steps\n", numSteps));

            List<String> transitions = getPlanPhaseTransitions();
            if (!transitions.isEmpty()) {
                sb.append("  Plan phase transitions:\n");
                transitions.forEach(t -> sb.append("    ").append(t).append("\n"));
            }

            List<String> regressions = getPhaseRegressions();
            if (!regressions.isEmpty()) {
                sb.append("  PHASE REGRESSIONS:\n");
                regressions.forEach(r -> sb.append("    ").append(r).append("\n"));
            }

            if (!errors.isEmpty()) {
                sb.append("  Errors:\n");
                errors.forEach(e -> sb.append("    ").append(e).append("\n"));
            }

            return sb.toString();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Stream Sync Report Classes
    // ═══════════════════════════════════════════════════════════════════════

    /** Types of stream sync issues. */
    public enum StreamSyncIssue {
        NULL_OUTPUT,
        CLOSED_BUFFER,
        NAN_DETECTED,
        INF_DETECTED,
        STALE_DATA,
        OUTPUT_REGRESSION,
        EXECUTION_ERROR
    }

    /** Stream sync validation report. */
    public static class StreamSyncReport {
        private final int numSteps;
        private final List<StreamSyncEntry> issues = new ArrayList<>();

        StreamSyncReport(int numSteps) {
            this.numSteps = numSteps;
        }

        void addIssue(int step, String outputName, StreamSyncIssue type, String message) {
            issues.add(new StreamSyncEntry(step, outputName, type, message));
        }

        public boolean hasIssues() { return !issues.isEmpty(); }

        public List<StreamSyncEntry> getIssuesOfType(StreamSyncIssue type) {
            List<StreamSyncEntry> filtered = new ArrayList<>();
            for (StreamSyncEntry e : issues) {
                if (e.type == type) filtered.add(e);
            }
            return filtered;
        }

        public int countByType(StreamSyncIssue type) {
            return getIssuesOfType(type).size();
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("StreamSyncReport: %d steps, %d issues\n", numSteps, issues.size()));
            if (issues.isEmpty()) {
                sb.append("  No stream sync issues detected.\n");
            } else {
                for (StreamSyncEntry e : issues) {
                    sb.append("  ").append(e).append("\n");
                }
            }
            return sb.toString();
        }
    }

    /** Single stream sync issue entry. */
    public static class StreamSyncEntry {
        public final int step;
        public final String outputName;
        public final StreamSyncIssue type;
        public final String message;

        StreamSyncEntry(int step, String outputName, StreamSyncIssue type, String message) {
            this.step = step;
            this.outputName = outputName;
            this.type = type;
            this.message = message;
        }

        @Override
        public String toString() {
            return String.format("[%s] step=%d output=%s: %s",
                    type.name(), step, outputName, message);
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Multi-Device Report Classes
    // ═══════════════════════════════════════════════════════════════════════

    /** Per-segment device placement info. */
    public static class SegmentDeviceInfo {
        public final int index;
        public final String backendName;
        public final int executionCount;
        public final boolean capturable;
        public final int numCaptureBuffers;
        public final String statisticsJson;

        SegmentDeviceInfo(int index, String backendName, int executionCount,
                          boolean capturable, int numCaptureBuffers, String statisticsJson) {
            this.index = index;
            this.backendName = backendName;
            this.executionCount = executionCount;
            this.capturable = capturable;
            this.numCaptureBuffers = numCaptureBuffers;
            this.statisticsJson = statisticsJson;
        }

        @Override
        public String toString() {
            return String.format("seg[%d] backend=%-12s capturable=%s execCount=%d captureBuffers=%d",
                    index, backendName, capturable, executionCount, numCaptureBuffers);
        }
    }

    /** Multi-device analysis report. */
    public static class MultiDeviceReport {
        public final int numSegments;
        public final int numSlots;
        public final List<SegmentDeviceInfo> segmentDevices;
        public final List<SlotInfo> slots;
        public final String replayCacheDeviceStats;
        public final String errorMessage;

        MultiDeviceReport(String errorMessage) {
            this.numSegments = 0;
            this.numSlots = 0;
            this.segmentDevices = Collections.emptyList();
            this.slots = Collections.emptyList();
            this.replayCacheDeviceStats = "";
            this.errorMessage = errorMessage;
        }

        MultiDeviceReport(int numSegments, int numSlots,
                          List<SegmentDeviceInfo> segmentDevices,
                          List<SlotInfo> slots, String replayCacheDeviceStats) {
            this.numSegments = numSegments;
            this.numSlots = numSlots;
            this.segmentDevices = segmentDevices;
            this.slots = slots;
            this.replayCacheDeviceStats = replayCacheDeviceStats;
            this.errorMessage = null;
        }

        /** Get backend distribution: backendName -> count. */
        public Map<String, Integer> getBackendDistribution() {
            Map<String, Integer> dist = new LinkedHashMap<>();
            for (SegmentDeviceInfo s : segmentDevices) {
                dist.merge(s.backendName, 1, Integer::sum);
            }
            return dist;
        }

        /** Slots with int/long sync requirements (cross-device transfer potential). */
        public List<SlotInfo> getIntLongSyncSlots() {
            List<SlotInfo> syncSlots = new ArrayList<>();
            for (SlotInfo s : slots) {
                if (s.needsIntLongSync()) syncSlots.add(s);
            }
            return syncSlots;
        }

        @Override
        public String toString() {
            if (errorMessage != null) return "MultiDeviceReport: " + errorMessage;

            StringBuilder sb = new StringBuilder();
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append("  Multi-Device Report\n");
            sb.append("═══════════════════════════════════════════════════════\n");
            sb.append(String.format("  Segments: %d | Slots: %d\n", numSegments, numSlots));

            Map<String, Integer> dist = getBackendDistribution();
            sb.append("  Backend distribution: ");
            dist.forEach((k, v) -> sb.append(k).append("=").append(v).append(" "));
            sb.append("\n");

            List<SlotInfo> syncSlots = getIntLongSyncSlots();
            if (!syncSlots.isEmpty()) {
                sb.append(String.format("  Slots requiring INT/LONG sync: %d\n", syncSlots.size()));
            }

            sb.append("───────────────────────────────────────────────────────\n");
            for (SegmentDeviceInfo seg : segmentDevices) {
                sb.append("  ").append(seg).append("\n");
            }

            if (replayCacheDeviceStats != null && !replayCacheDeviceStats.isEmpty()) {
                sb.append("───────────────────────────────────────────────────────\n");
                sb.append("  Replay cache device stats: ").append(replayCacheDeviceStats).append("\n");
            }

            sb.append("═══════════════════════════════════════════════════════\n");
            return sb.toString();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Full Diagnostic Report
    // ═══════════════════════════════════════════════════════════════════════

    /** Combined report from all diagnostic analyses. */
    public static class FullDiagnosticReport {
        public final PlanReport planReport;
        public final GraphReplayReport replayReport;
        public final MultiDeviceReport deviceReport;
        public final GraphReplayProgressReport progressReport;
        public final String nativePlanReport;
        public final String jsonReport;

        FullDiagnosticReport(PlanReport planReport, GraphReplayReport replayReport,
                             MultiDeviceReport deviceReport,
                             GraphReplayProgressReport progressReport,
                             String nativePlanReport, String jsonReport) {
            this.planReport = planReport;
            this.replayReport = replayReport;
            this.deviceReport = deviceReport;
            this.progressReport = progressReport;
            this.nativePlanReport = nativePlanReport;
            this.jsonReport = jsonReport;
        }

        /** Quick summary of all issues found. */
        public List<String> getAllIssues() {
            List<String> issues = new ArrayList<>();

            // Plan issues
            if (planReport != null && planReport.errorMessage == null) {
                List<SlotInfo> risky = planReport.getRiskyOps();
                if (!risky.isEmpty()) {
                    issues.add(risky.size() + " risky ops detected (data-dependent/view-capable)");
                }
                List<SlotInfo> unfrozen = planReport.getUnfrozenOps();
                if (!unfrozen.isEmpty()) {
                    issues.add(unfrozen.size() + " ops failed to freeze");
                }
            }

            // Replay issues
            if (replayReport != null && replayReport.errorMessage == null) {
                List<SegmentReplayInfo> failures = replayReport.getCaptureFailures();
                if (!failures.isEmpty()) {
                    issues.add(failures.size() + " segments had graph capture failures");
                }
                List<SegmentReplayInfo> stuck = replayReport.getStuckSegments();
                if (!stuck.isEmpty()) {
                    issues.add(stuck.size() + " segments stuck in warmup");
                }
                if (!replayReport.pointersStable) {
                    issues.add("Pointer instability detected — graph replay may be unsafe");
                }
            }

            // Progress regressions
            if (progressReport != null) {
                List<String> regressions = progressReport.getPhaseRegressions();
                if (!regressions.isEmpty()) {
                    issues.add(regressions.size() + " phase regressions detected");
                }
            }

            return issues;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("╔═══════════════════════════════════════════════════════╗\n");
            sb.append("║          FULL DSP DIAGNOSTIC REPORT                  ║\n");
            sb.append("╚═══════════════════════════════════════════════════════╝\n\n");

            List<String> issues = getAllIssues();
            if (issues.isEmpty()) {
                sb.append("  No issues detected.\n\n");
            } else {
                sb.append("  ISSUES FOUND: ").append(issues.size()).append("\n");
                for (String issue : issues) {
                    sb.append("    - ").append(issue).append("\n");
                }
                sb.append("\n");
            }

            if (planReport != null) sb.append(planReport).append("\n");
            if (replayReport != null) sb.append(replayReport).append("\n");
            if (deviceReport != null) sb.append(deviceReport).append("\n");
            if (progressReport != null) sb.append(progressReport).append("\n");

            if (nativePlanReport != null && !nativePlanReport.isEmpty()) {
                sb.append("── Native C++ Plan Report ─────────────────────────────\n");
                sb.append(nativePlanReport).append("\n");
            }

            return sb.toString();
        }
    }
}
