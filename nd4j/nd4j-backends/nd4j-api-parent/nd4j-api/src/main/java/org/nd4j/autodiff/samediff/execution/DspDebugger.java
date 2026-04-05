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
}
