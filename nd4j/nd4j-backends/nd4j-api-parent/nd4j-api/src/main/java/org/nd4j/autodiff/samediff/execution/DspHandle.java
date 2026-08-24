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
package org.nd4j.autodiff.samediff.execution;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.diagnostics.DspDiagnostics;
import org.nd4j.autodiff.samediff.internal.InferenceSession;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueNDArray;

import java.util.*;
import java.util.stream.Collectors;

/**
 * Thin facade over SameDiff for direct DSP plan replay and buffer inspection.
 *
 * <h3>Usage</h3>
 * <pre>
 *   // Warm the plan
 *   sd.output(placeholders, "logits");
 *
 *   // Get the handle
 *   DspHandle h = sd.dsp();
 *
 *   // Replay and inspect
 *   h.replay(placeholders);
 *   int nanSlot = h.firstNaNSlot();
 *   h.snapshotAllSlots().forEach((slot, summary) -&gt;
 *       System.out.println(summary));
 * </pre>
 *
 * <h3>Thread safety</h3>
 * Not thread-safe. Must be used on the same thread that called sd.dsp().
 * Shares the underlying executor with InferenceSession.
 *
 * @author Adam Gibson
 */
@Slf4j
public final class DspHandle {

    private final SameDiff sd;

    public DspHandle(SameDiff sd) {
        this.sd = sd;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Executor / handle access
    // ─────────────────────────────────────────────────────────────────────────

    private DynamicShapePlanExecutor requireExecutor() {
        InferenceSession session = sd.getOrCreateSession();
        DynamicShapePlanExecutor exec = session.getDynamicShapePlanExecutor();
        if (exec == null) {
            // Training executes on the gradient-function instance in its own
            // TrainingSession — fall back to it so DSP plan state can be inspected
            // right after fit(). The executor is thread-local to the thread that
            // ran the training loop.
            InferenceSession trainingSession = sd.getLastTrainingSession();
            if (trainingSession != null) {
                exec = trainingSession.getDynamicShapePlanExecutor();
            }
        }
        if (exec == null) {
            throw new IllegalStateException(
                "DspHandle: no DynamicShapePlanExecutor. " +
                "Call sd.output(), sd.fit() or sd.compileNativeDynamicShapePlan() first.");
        }
        return exec;
    }

    private Pointer requireHandle() {
        Pointer handle = requireExecutor().getNativePlanHandle();
        if (handle == null || handle.isNull()) {
            throw new IllegalStateException(
                "DspHandle: native plan handle is null. Plan not compiled to native C++.");
        }
        return handle;
    }

    /**
     * Get the native plan handle, or null if not yet compiled.
     */
    public Pointer getNativePlanHandle() {
        DynamicShapePlanExecutor exec = sd.getOrCreateSession().getDynamicShapePlanExecutor();
        return exec != null ? exec.getNativePlanHandle() : null;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plan state query
    // ─────────────────────────────────────────────────────────────────────────

    /** Whether the native plan has been compiled. */
    public boolean isCompiled() {
        Pointer handle = getNativePlanHandle();
        return handle != null && !handle.isNull();
    }

    /** Total number of internal slots (all intermediate + output). */
    public int totalSlots() {
        return nativeOps().getTotalPlanOutputSlots(requireHandle());
    }

    /** Number of external inputs (constants + variables + placeholders). */
    public int numExternalInputs() {
        return nativeOps().getPlanNumExternalInputs(requireHandle());
    }

    /** Get ext input index for a named variable. Returns -1 if not found. */
    public int extInputIndex(String name) {
        return requireExecutor().findExternalInputIndex(name);
    }

    /** Get output slot index for a named output. Returns -1 if not found. */
    public int slotIndexForOutput(String name) {
        return requireExecutor().findOutputIndex(name);
    }

    /**
     * Find the first slot whose op name contains the given substring.
     * Returns -1 if not found.
     */
    public int slotIndexForOp(String opNameSubstring) {
        NativeOps ops = nativeOps();
        Pointer handle = requireHandle();
        int total = ops.getPlanNumSlots(handle);
        for (int i = 0; i < total; i++) {
            String name = ops.getPlanSlotOpName(handle, i);
            if (name != null && name.contains(opNameSubstring)) return i;
        }
        return -1;
    }

    /**
     * Find ALL slots whose op name contains the given substring.
     * Returns a list of slot indices (may be empty).
     */
    public List<Integer> allSlotsForOp(String opNameSubstring) {
        NativeOps ops = nativeOps();
        Pointer handle = requireHandle();
        int total = ops.getPlanNumSlots(handle);
        List<Integer> result = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            String name = ops.getPlanSlotOpName(handle, i);
            if (name != null && name.contains(opNameSubstring)) result.add(i);
        }
        return result;
    }

    /** Human-readable plan summary. */
    public String planSummary() {
        return DspPlanAssertions.snapshotPlanState(sd);
    }

    /** Current native plan execution count. */
    public int executeCount() {
        return nativeOps().getPlanExecuteCount(requireHandle());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Staging buffer introspection (CUDA graph replay debugging)
    // ─────────────────────────────────────────────────────────────────────────

    /** Number of variable ext inputs with allocated staging buffers. */
    public int numStagingBuffers() {
        return nativeOps().getPlanNumStagingBuffers(requireHandle());
    }

    /** Number of cached variable ext input indices in the D2D fast path. */
    public int numCachedVariableExtIndices() {
        return nativeOps().getPlanNumCachedVariableExtIndices(requireHandle());
    }

    /** Get the i-th cached variable ext input index. Returns -1 if out of range. */
    public int cachedVariableExtIndex(int i) {
        return nativeOps().getPlanCachedVariableExtIndex(requireHandle(), i);
    }

    /** Get all cached variable ext input indices. */
    public List<Integer> cachedVariableExtIndices() {
        int n = numCachedVariableExtIndices();
        List<Integer> result = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            result.add(cachedVariableExtIndex(i));
        }
        return result;
    }

    /** Device address of the staging buffer for ext[extIdx], or 0. */
    public long stagingBufferAddress(int extIdx) {
        return nativeOps().getPlanStagingBufferAddress(requireHandle(), extIdx);
    }

    /** Device address the CUDA graph reads from for ext[extIdx]. */
    public long effectiveExternalAddress(int extIdx) {
        return nativeOps().getPlanEffectiveExternalAddress(requireHandle(), extIdx);
    }

    /** Device address of the last externalArrays[extIdx] passed to execute/executeSteadyState. */
    public long lastExternalInputAddress(int extIdx) {
        return nativeOps().getPlanLastExternalInputAddress(requireHandle(), extIdx);
    }

    /**
     * Get a COPY of the staging buffer content for ext[extIdx].
     * Returns null if no staging buffer exists for this ext input.
     * Use immediately after replay() while staging memory is valid.
     *
     * Uses atomic native copy (copyPlanStagingToBuffer) to avoid the stale-pointer
     * race that occurs when extracting specialBuffer() in one JNI call and copying
     * in another — the staging buffer's device memory can be freed/reallocated
     * between calls, causing CUDA illegal memory access (error 700/719).
     */
    public INDArray getStagingBufferContent(int extIdx) {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();

        // Step 1: Get shape info from the staging buffer (read-only, no device memory access)
        OpaqueNDArray opaque = owned(ops.getPlanStagingBufferArray(handle, extIdx));
        if (opaque == null || opaque.isNull()) return null;

        long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaque);
        if (shapeInfo == null || shapeInfo.length == 0) return null;

        long rank = shapeInfo[0];
        if (rank < 0 || rank > Shape.MAX_RANK || shapeInfo.length < Shape.shapeInfoLength(rank)) return null;

        long[] shape = Shape.shape(shapeInfo);
        for (long dim : shape) {
            if (dim <= 0 || dim > 10_000_000) return null;
        }
        DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
        long length = OpaqueNDArray.getOpaqueNDArrayLength(opaque);
        char ordering = Shape.order(shapeInfo);

        if (length <= 0) return null;

        // Step 2: Allocate destination array — use simple shape (no explicit strides)
        // to ensure contiguous allocation matching the staging buffer's element count.
        INDArray result = Nd4j.createUninitialized(dtype, shape, ordering);

        // Step 3: Atomic copy — native code reads staging specialBuffer and copies
        // to dst DataBuffer in a single C++ call, no stale pointer window.
        OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
        if (dstOdb == null) return null;

        int rc = ops.copyPlanStagingToBuffer(handle, extIdx, dstOdb);
        if (rc != 0) {
            // Copy failed — staging buffer was invalidated between shape read and copy
            result.close();
            return null;
        }

        Nd4j.getExecutioner().commit();
        return result;
    }

    /**
     * Snapshot staging buffer state for all variable ext inputs.
     * Returns map: extIndex -> {name, stagingAddr, effectiveAddr, match, stagingSum}.
     * The "match" flag indicates whether staging and effective addresses are the same
     * (they should be for variables with staging buffers during replay).
     */
    public Map<Integer, String> snapshotStagingState() {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();
        int numCached = ops.getPlanNumCachedVariableExtIndices(handle);
        int numExt = ops.getPlanNumExternalInputs(handle);

        Map<Integer, String> result = new LinkedHashMap<>();
        for (int ci = 0; ci < numCached; ci++) {
            int extIdx = ops.getPlanCachedVariableExtIndex(handle, ci);
            if (extIdx < 0 || extIdx >= numExt) continue;

            long stagingAddr = ops.getPlanStagingBufferAddress(handle, extIdx);
            long effectiveAddr = ops.getPlanEffectiveExternalAddress(handle, extIdx);
            boolean match = (stagingAddr != 0 && stagingAddr == effectiveAddr);

            String summary;
            try {
                INDArray staging = getStagingBufferContent(extIdx);
                if (staging == null) {
                    summary = String.format("ext[%d] staging=0x%x effective=0x%x match=%s [no staging]",
                            extIdx, stagingAddr, effectiveAddr, match);
                } else if (!staging.dataType().isNumerical()) {
                    summary = String.format("ext[%d] staging=0x%x effective=0x%x match=%s shape=%s dtype=%s",
                            extIdx, stagingAddr, effectiveAddr, match,
                            Arrays.toString(staging.shape()), staging.dataType());
                } else {
                    double sum = staging.sumNumber().doubleValue();
                    summary = String.format("ext[%d] staging=0x%x effective=0x%x match=%s shape=%s dtype=%s sum=%.6f%s",
                            extIdx, stagingAddr, effectiveAddr, match,
                            Arrays.toString(staging.shape()), staging.dataType(), sum,
                            Double.isNaN(sum) ? " *** NaN ***" : "");
                }
            } catch (Exception e) {
                summary = String.format("ext[%d] staging=0x%x effective=0x%x match=%s [error: %s]",
                        extIdx, stagingAddr, effectiveAddr, match, e.getMessage());
            }
            result.put(extIdx, summary);
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Buffer injection
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Inject an array directly into external input slot.
     * Effect is cleared on the next sd.output() call.
     */
    public DspHandle setExtInput(int extIdx, INDArray array) {
        Objects.requireNonNull(array, "array must not be null");
        requireExecutor().overrideExternalInput(extIdx, array);
        return this;
    }

    /**
     * Inject an array by variable name.
     * @throws IllegalArgumentException if name not found in external inputs
     */
    public DspHandle setExtInput(String variableName, INDArray array) {
        int idx = extInputIndex(variableName);
        if (idx < 0) {
            throw new IllegalArgumentException(
                "DspHandle.setExtInput: '" + variableName + "' not found in external inputs.");
        }
        return setExtInput(idx, array);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Variable / placeholder marking
    // ─────────────────────────────────────────────────────────────────────────

    /** Mark ext input as variable (D2D staging buffer). */
    public DspHandle markVariable(int extIdx) {
        nativeOps().markPlanExternalInputVariable(requireHandle(), extIdx);
        return this;
    }

    /** Mark named variable as variable. */
    public DspHandle markVariable(String variableName) {
        int idx = extInputIndex(variableName);
        if (idx < 0)
            throw new IllegalArgumentException("markVariable: '" + variableName + "' not found");
        return markVariable(idx);
    }

    /** Mark ext input as placeholder (variable + H2D sync). */
    public DspHandle markPlaceholder(int extIdx) {
        nativeOps().markPlanExternalInputPlaceholder(requireHandle(), extIdx);
        return this;
    }

    /** Mark named variable as placeholder. */
    public DspHandle markPlaceholder(String variableName) {
        int idx = extInputIndex(variableName);
        if (idx < 0)
            throw new IllegalArgumentException("markPlaceholder: '" + variableName + "' not found");
        return markPlaceholder(idx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plan execution
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Replay the plan with the given placeholder arrays.
     * Delegates to the existing executor — same execution path as sd.output()
     * but skips InferenceSession bookkeeping.
     */
    public Map<String, INDArray> replay(Map<String, INDArray> placeholders) {
        DynamicShapePlanExecutor exec = requireExecutor();
        DynamicShapePlan plan = exec.getCurrentPlan();
        if (plan == null) {
            throw new IllegalStateException(
                "DspHandle.replay: no compiled plan. Call sd.output() once first.");
        }
        return exec.execute(plan, placeholders);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Intermediate slot access
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get a COPY of the output array at the given internal slot index.
     * Call immediately after replay() while slot memory is valid.
     * Returns null if the slot is empty or out of range.
     */
    public INDArray getSlotOutput(int slotIdx) {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();

        OpaqueNDArray opaqueOut = owned(ops.getPlanSlotOutputArray(handle, slotIdx));
        if (opaqueOut == null || opaqueOut.isNull()) return null;

        long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
        if (shapeInfo == null) {
            // Null-read tracing (task #52, longViewChain/JIT null-slot): the native
            // slot array is non-null but its shapeInfo read came back null/invalid —
            // signature of a released shared ConstantShapeBuffer (view replaced, old
            // view's deferred delete dropped a shape buffer the replacement still
            // references). Record WHICH branch nulled so the native root is targetable.
            DspDiagnostics.record(DspDiagnostics.MEMORY,
                    "Java getSlotOutput NULL: slot=" + slotIdx + " reason=shapeInfo-null (opaque non-null)");
            return null;
        }

        long[] shape = Shape.shape(shapeInfo);
        long[] strides = Shape.stride(shapeInfo);
        DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
        long length = OpaqueNDArray.getOpaqueNDArrayLength(opaqueOut);
        char ordering = Shape.order(shapeInfo);

        if (length == 0) {
            // Null-read tracing (task #52): zero-length at read time — empty view
            // or length computed from a released/garbage shapeInfo.
            DspDiagnostics.record(DspDiagnostics.MEMORY,
                    "Java getSlotOutput NULL: slot=" + slotIdx + " reason=length-0 (shapeInfo="
                            + java.util.Arrays.toString(shapeInfo) + ")");
            return null;
        }

        INDArray result = Nd4j.createUninitialized(dtype, shape, strides, ordering);

        Pointer nativeSpecial = ops.getOpaqueNDArraySpecialBuffer(opaqueOut);
        Pointer nativePrimary = (nativeSpecial == null || nativeSpecial.isNull())
                ? ops.getOpaqueNDArrayBuffer(opaqueOut) : null;

        OpaqueDataBuffer srcOdb = ops.dbCreateExternalDataBuffer(
                length, dtype.toInt(), nativePrimary, nativeSpecial);
        if (srcOdb != null) {
            try {
                OpaqueDataBuffer dstOdb = result.data().opaqueBuffer();
                if (dstOdb != null) {
                    ops.copyBuffer(dstOdb, length, srcOdb, 0, 0);
                }
            } finally {
                ops.deleteDataBuffer(srcOdb);
            }
        }

        Nd4j.getExecutioner().commit();
        return result;
    }

    /**
     * Get the output array for the first slot matching the op name substring.
     * Returns null if not found or empty.
     */
    public INDArray getSlotOutput(String opNameSubstring) {
        int idx = slotIndexForOp(opNameSubstring);
        if (idx < 0) return null;
        return getSlotOutput(idx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // NaN debugging utilities
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Snapshot all non-null intermediate slots after a replay.
     * Returns a map from slot index to a summary string with NaN/Inf flags.
     */
    public Map<Integer, String> snapshotAllSlots() {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();
        int total = ops.getTotalPlanOutputSlots(handle);

        Map<Integer, String> result = new LinkedHashMap<>();
        for (int i = 0; i < total; i++) {
            OpaqueNDArray opaqueOut = ops.getPlanSlotOutputArray(handle, i);
            if (opaqueOut == null || opaqueOut.isNull()) continue;

            owned(opaqueOut);
            String opName = ops.getPlanSlotOpName(handle, i);
            long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
            if (shapeInfo == null) continue;

            long[] shape = Shape.shape(shapeInfo);
            DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);

            String summary;
            try {
                INDArray arr = getSlotOutput(i);
                if (arr == null) {
                    summary = String.format("slot[%d] op=%-30s shape=%s dtype=%s [empty]",
                            i, opName, Arrays.toString(shape), dtype);
                } else if (!dtype.isNumerical()) {
                    summary = String.format("slot[%d] op=%-30s shape=%s dtype=%s [non-numerical]",
                            i, opName, Arrays.toString(shape), dtype);
                } else {
                    double sum = arr.sumNumber().doubleValue();
                    boolean hasNaN = Double.isNaN(sum);
                    boolean hasInf = Double.isInfinite(sum);
                    summary = String.format("slot[%d] op=%-30s shape=%s dtype=%s sum=%.6f%s",
                            i, opName, Arrays.toString(shape), dtype, sum,
                            hasNaN ? " *** NaN ***" : hasInf ? " *** Inf ***" : "");
                }
            } catch (Exception e) {
                summary = String.format("slot[%d] op=%-30s shape=%s dtype=%s [stale buffer]",
                        i, opName, Arrays.toString(shape), dtype);
            }
            result.put(i, summary);
        }
        return result;
    }

    /**
     * Find the first slot with NaN values after a replay.
     * Returns -1 if no NaN is found.
     */
    public int firstNaNSlot() {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();
        int total = ops.getTotalPlanOutputSlots(handle);

        for (int i = 0; i < total; i++) {
            OpaqueNDArray opaqueOut = owned(ops.getPlanSlotOutputArray(handle, i));
            if (opaqueOut == null || opaqueOut.isNull()) continue;

            long[] shapeInfo = OpaqueNDArray.getOpaqueNDArrayShapeInfo(opaqueOut);
            if (shapeInfo != null) {
                DataType dtype = ArrayOptionsHelper.dataType(shapeInfo);
                if (!dtype.isNumerical()) continue;
            }

            try {
                INDArray arr = getSlotOutput(i);
                if (arr == null) continue;
                double sum = arr.sumNumber().doubleValue();
                if (Double.isNaN(sum)) return i;
            } catch (Exception e) {
                // Slot buffer may be stale/reclaimed — skip it
                log.trace("firstNaNSlot: skipping slot {} due to: {}", i, e.getMessage());
            }
        }
        return -1;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Capture quality & validation (Q7, Q8, Q9)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Raw capture stats string from the last capture attempt.
     * Format: "captured=N|oomRetrying=N|permFailed=N|nonCapt=N|tooSmall=N|addrUnstable=N"
     * Returns empty string if not compiled.
     */
    public String captureStats() {
        if (!isCompiled()) return "";
        return nativeOps().getPlanCaptureStats(requireHandle());
    }

    /** Parsed capture stats as a structured record. */
    public CaptureStats parsedCaptureStats() {
        return CaptureStats.parse(captureStats());
    }

    /** Number of ops that ran on host during capture (escaped CUDA graph). */
    public int numHostOnlyOps() {
        return nativeOps().getPlanNumHostOnlyOps(requireHandle());
    }

    /** Pipe-delimited list of host-only op names. */
    public String hostOnlyOpNames() {
        return nativeOps().getPlanHostOnlyOpNames(requireHandle());
    }

    /** Parsed list of host-only op names. */
    public List<String> hostOnlyOpNameList() {
        String raw = hostOnlyOpNames();
        if (raw == null || raw.isEmpty()) return Collections.emptyList();
        return Arrays.asList(raw.split("\\|"));
    }

    /** Whether the captured graph covers all ops (no host-only escapes). */
    public boolean isCaptureComplete() {
        return nativeOps().validatePlanCapturedGraph(requireHandle());
    }

    /**
     * Validate all plan outputs for NaN/Inf/null/all-zeros.
     * Returns flags per output: bit 0=NULL, bit 1=NaN, bit 2=Inf, bit 3=ALL_ZERO.
     * Gate behind isDebug — requires sync-to-host.
     */
    public int[] validateOutputs() {
        if (!isCompiled()) return new int[0];
        int numOutputs = nativeOps().getPlanNumRequestedOutputs(requireHandle());
        if (numOutputs <= 0) return new int[0];
        int[] flags = new int[numOutputs];
        nativeOps().dspValidateOutputs(requireHandle(), flags);
        return flags;
    }

    /** True if any output has validation issues (NaN/Inf/null/all-zero) after the last replay. */
    public boolean hasOutputIssues() {
        int[] flags = validateOutputs();
        for (int f : flags) {
            if (f != 0) return true;
        }
        return false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Pointer & address tracking (Q2, Q10)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Per-segment tracked pointer state as JSON.
     * Contains: [{inputIdx, capturedAddr, currentAddr, match}, ...]
     * Non-empty only after capture has occurred.
     */
    public String segmentTrackedPointersJson(int segmentIdx) {
        return nativeOps().getPlanSegmentTrackedPointers(requireHandle(), segmentIdx);
    }

    /** Capture buffers JSON for a segment. */
    public String segmentCaptureBuffersJson(int segmentIdx) {
        return nativeOps().getPlanSegmentCaptureBuffersJson(requireHandle(), segmentIdx);
    }

    /** Number of buffers registered at capture time for a segment. */
    public int segmentNumCaptureBuffers(int segmentIdx) {
        return nativeOps().getPlanSegmentNumCaptureBuffers(requireHandle(), segmentIdx);
    }

    /** Number of pinned host pointers in a segment's replay handle. */
    public int segmentNumHostPointers(int segmentIdx) {
        return nativeOps().getPlanSegmentNumHostPointers(requireHandle(), segmentIdx);
    }

    /**
     * Check all segments for pointer drift (captured addr != current addr).
     * Returns map: segmentIdx -> true if all pointers match, false if drift detected.
     */
    public Map<Integer, Boolean> allSegmentsPointersMatch() {
        Pointer handle = requireHandle();
        NativeOps ops = nativeOps();
        int numSegs = ops.getPlanSegmentCount(handle);
        Map<Integer, Boolean> result = new LinkedHashMap<>();
        for (int s = 0; s < numSegs; s++) {
            String json = ops.getPlanSegmentTrackedPointers(handle, s);
            boolean allMatch = json == null || json.isEmpty()
                    || json.equals("[]") || !json.contains("\"match\":false");
            result.put(s, allMatch);
        }
        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Stale output detection (Q6)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Compare current output norms against prevNorms array.
     * Updates prevNorms in-place with current norms.
     * Returns number of stale outputs (norm diff below epsilon).
     * Gate behind isDebug — requires sync-to-host.
     */
    public int detectStaleOutputs(float[] prevNorms, boolean[] staleOut, float epsilon) {
        if (!isCompiled()) return 0;
        return nativeOps().dspDetectStaleOutputs(requireHandle(), prevNorms, staleOut, epsilon);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Segment introspection (Q4, Q5)
    // ─────────────────────────────────────────────────────────────────────────

    /** Number of segments in the plan. */
    public int numSegments() {
        return nativeOps().getPlanSegmentCount(requireHandle());
    }

    /** Execution phase for a segment (see ExecutionPhase ordinals). */
    public int segmentExecutionPhase(int segIdx) {
        return nativeOps().getPlanSegmentExecutionPhase(requireHandle(), segIdx);
    }

    /** Number of graph replays for a segment. */
    public int segmentReplayCount(int segIdx) {
        return nativeOps().getPlanSegmentReplayCount(requireHandle(), segIdx);
    }

    /** Total execution count for a segment (slot-by-slot + replay). */
    public int segmentExecCount(int segIdx) {
        return nativeOps().getSegmentExecutionCount(requireHandle(), segIdx);
    }

    /** Whether a segment is eligible for CUDA graph capture. */
    public boolean isSegmentCapturable(int segIdx) {
        return nativeOps().isPlanSegmentCapturable(requireHandle(), segIdx);
    }

    /** Whether capture permanently failed for a segment. */
    public boolean isSegmentCaptureFailed(int segIdx) {
        return nativeOps().isPlanSegmentCaptureFailed(requireHandle(), segIdx);
    }

    /** Replay state: 0=EMPTY, 1=CAPTURING, 2=CAPTURED, 3=READY, 4=ERROR. */
    public int segmentReplayState(int segIdx) {
        return nativeOps().getPlanSegmentReplayState(requireHandle(), segIdx);
    }

    /** Replay mode: 0=NONE, 1=MONOLITHIC, 2=COMPOSITE. */
    public int segmentReplayMode(int segIdx) {
        return nativeOps().getPlanSegmentReplayMode(requireHandle(), segIdx);
    }

    /** Backend name for a segment (e.g. "triton", "cublas"). */
    public String segmentBackendName(int segIdx) {
        return nativeOps().getPlanSegmentBackendName(requireHandle(), segIdx);
    }

    /** Per-segment statistics as JSON. */
    public String segmentStatisticsJson(int segIdx) {
        return nativeOps().getPlanSegmentStatisticsJson(requireHandle(), segIdx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plan lifecycle (Q5)
    // ─────────────────────────────────────────────────────────────────────────

    /** Current plan phase ordinal (see PlanPhase). */
    public int planPhase() {
        return nativeOps().getPlanPhase(requireHandle());
    }

    /** Whether buffer pointers are stable across executions. */
    public boolean pointersStable() {
        return nativeOps().getPlanPointersStable(requireHandle()) == 1;
    }

    /** Number of executions since shapes were frozen. -1 if not frozen. */
    public int frozenExecCount() {
        return nativeOps().getPlanFrozenExecutionCount(requireHandle());
    }

    /** Number of mid-execution recompiles (should be 0 after seal). */
    public int midExecutionCompileCount() {
        return (int) nativeOps().getPlanMidExecutionCompileCount(requireHandle());
    }

    /** Whether the plan's compilation has been sealed. */
    public boolean isCompilationSealed() {
        return nativeOps().isPlanCompilationSealed(requireHandle()) == 1;
    }

    /** Total CUDA graph replays across all segments. */
    public int totalGraphReplays() {
        return nativeOps().getPlanTotalGraphReplays(requireHandle());
    }

    /** Applied native graph execution mode. */
    public GraphExecutionMode graphExecutionMode() {
        return GraphExecutionMode.fromNativeCode(
                nativeOps().getPlanGraphExecutionMode(requireHandle()));
    }

    /** Number of segments with captured CUDA graphs. */
    public int numCapturedGraphSegments() {
        return nativeOps().getPlanNumCapturedGraphSegments(requireHandle());
    }

    /** Frozen plan build pass count: 0=needs warmup, 1=needs compile, 2+=sealed. */
    public int frozenPlanBuildPassCount() {
        return nativeOps().getFrozenPlanBuildPassCount(requireHandle());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Slot introspection
    // ─────────────────────────────────────────────────────────────────────────

    /** Per-slot state: 0=WARMUP, 1=SHAPE_CACHED, 2=FROZEN, 3=FROZEN_CONSTANT. */
    public int slotState(int slotIdx) {
        return nativeOps().getPlanSlotState(requireHandle(), slotIdx);
    }

    /**
     * Per-slot flags bitmask.
     * Bit 0: VIEW_CAPABLE, Bit 1: DATA_DEPENDENT, Bit 2: SHAPE_DEPENDS_ON_VALUES,
     * Bit 3: IDENTITY, Bit 4: IN_PLACE_FUSED, Bit 5: FUSED_CHAIN_HEAD,
     * Bit 6: FUSED_CHAIN_TAIL, Bit 7: NEEDS_ZEROED, Bit 8: NEEDS_INT_LONG_SYNC,
     * Bit 9: SHAPE_STATIC, Bit 10: FROZEN_CONSTANT.
     */
    public int slotFlags(int slotIdx) {
        return nativeOps().getPlanSlotFlags(requireHandle(), slotIdx);
    }

    /** Slot op name. */
    public String slotOpName(int slotIdx) {
        return nativeOps().getPlanSlotOpName(requireHandle(), slotIdx);
    }

    /**
     * Monotonic write-generation counter for a slot.
     * Incremented on kernel dispatch, prezero, nullify, fused chain emit, view install.
     * Returns -1 if invalid.
     */
    public int slotGeneration(int slotIdx) {
        return nativeOps().getPlanSlotGeneration(requireHandle(), slotIdx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Backend & visualization
    // ─────────────────────────────────────────────────────────────────────────

    /** JSON array of available backends. */
    public String availableBackends() {
        return nativeOps().getPlanAvailableBackends(requireHandle());
    }

    /** Which backend compiled a specific segment. */
    public String segmentCompiledBackend(int segIdx) {
        return nativeOps().getPlanSegmentCompiledBackend(requireHandle(), segIdx);
    }

    /** Per-segment compilation audit as JSON. */
    public String segmentCompilationAudit(int segIdx) {
        return nativeOps().getPlanSegmentCompilationAudit(requireHandle(), segIdx);
    }

    /** Invalidate one segment's backend/replay artifact and force clean rebuilding. */
    public void invalidateSegmentCache(int segIdx) {
        nativeOps().invalidatePlanSegmentCache(requireHandle(), segIdx);
    }

    /** Invalidate all artifacts owned by one backend name (empty means all). */
    public void invalidateBackendCaches(String backendName) {
        nativeOps().invalidatePlanBackendCaches(
                requireHandle(), backendName == null ? "" : backendName);
    }

    /** Aggregated backend cache stats as JSON. */
    public String backendCacheStats() {
        return nativeOps().getPlanBackendCacheStats(requireHandle());
    }

    /** Export Chrome trace JSON for the CUDA graph timeline. */
    public boolean exportChromeTrace(String outputPath) {
        return nativeOps().exportPlanCudaGraphChromeTrace(requireHandle(), outputPath);
    }

    /** Export CUDA graph visualization as HTML. */
    public boolean exportCudaGraphHtml(String outputPath) {
        return nativeOps().exportPlanCudaGraphHtml(requireHandle(), outputPath);
    }

    /** Get Chrome trace JSON for the CUDA graph timeline. */
    public String cudaGraphChromeTraceJson() {
        return nativeOps().getPlanCudaGraphChromeTraceJson(requireHandle());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Replay cache
    // ─────────────────────────────────────────────────────────────────────────

    public boolean isReplayCacheEnabled() { return nativeOps().isReplayCacheEnabled(); }
    public int replayCacheHits() { return nativeOps().getReplayCacheHits(); }
    public int replayCacheMisses() { return nativeOps().getReplayCacheMisses(); }
    public String replayCacheDeviceStatsJson() { return nativeOps().getReplayCacheDeviceStatsJson(); }

    // ─────────────────────────────────────────────────────────────────────────
    // KV cache introspection
    // ─────────────────────────────────────────────────────────────────────────

    /** Current KV cache position (decode step count). */
    public long kvCachePosition() {
        return nativeOps().getPlanKvCachePosition(requireHandle());
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Native diagnostics bridge
    // ─────────────────────────────────────────────────────────────────────────

    /** Human-readable plan report from the C++ diagnostic ring buffer. */
    public String nativePlanReport() {
        return nativeOps().dspDiagGetPlanReport();
    }

    /** JSON diagnostic report from the C++ diagnostic ring buffer. */
    public String nativeJsonReport() {
        return nativeOps().dspDiagGetJsonReport();
    }

    /** Total diagnostic steps executed by the C++ singleton. */
    public int diagStepCount() {
        return nativeOps().dspDiagGetStepCount();
    }

    /** Total event count across all diagnostic categories. */
    public long diagTotalEventCount() {
        return nativeOps().dspDiagGetTotalEventCount();
    }

    /** Event count for a specific diagnostic category by index (0..18). */
    public long diagCategoryEventCount(int categoryIndex) {
        return nativeOps().dspDiagGetCategoryEventCount(categoryIndex);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // StepSnapshot — structured per-step state capture
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Capture a complete state snapshot immediately after replay().
     * Answers all 10 monologue questions without log parsing.
     * Only calls metadata queries (addresses, counts, phases) — no syncToHost.
     */
    public StepSnapshot captureStepSnapshot() {
        NativeOps ops = nativeOps();
        Pointer handle = requireHandle();

        int execCount = ops.getPlanExecuteCount(handle);
        int phase = ops.getPlanPhase(handle);
        boolean ptrStable = ops.getPlanPointersStable(handle) == 1;
        int frozenExec = ops.getPlanFrozenExecutionCount(handle);

        // D2D status for each cached variable ext input
        Map<Integer, D2DStatus> d2dMap = new LinkedHashMap<>();
        int numCached = ops.getPlanNumCachedVariableExtIndices(handle);
        for (int ci = 0; ci < numCached; ci++) {
            int extIdx = ops.getPlanCachedVariableExtIndex(handle, ci);
            if (extIdx < 0) continue;
            long stagingAddr = ops.getPlanStagingBufferAddress(handle, extIdx);
            long effectiveAddr = ops.getPlanEffectiveExternalAddress(handle, extIdx);
            d2dMap.put(extIdx, new D2DStatus(extIdx, stagingAddr, effectiveAddr));
        }

        // Segment execution state
        int numSegs = ops.getPlanSegmentCount(handle);
        List<SegmentStepInfo> segInfos = new ArrayList<>(numSegs);
        List<Integer> driftingSegs = new ArrayList<>();
        for (int s = 0; s < numSegs; s++) {
            int segExec = ops.getSegmentExecutionCount(handle, s);
            int segReplay = ops.getPlanSegmentReplayCount(handle, s);
            int segPhase = ops.getPlanSegmentExecutionPhase(handle, s);
            segInfos.add(new SegmentStepInfo(s, segExec, segReplay, segPhase));

            String trackedJson = ops.getPlanSegmentTrackedPointers(handle, s);
            if (trackedJson != null && trackedJson.contains("\"match\":false")) {
                driftingSegs.add(s);
            }
        }

        CaptureStats cs = CaptureStats.parse(ops.getPlanCaptureStats(handle));

        return new StepSnapshot(execCount, phase, ptrStable, frozenExec,
                d2dMap, segInfos, cs, driftingSegs);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Inner data classes
    // ─────────────────────────────────────────────────────────────────────────

    /** D2D copy status for a single external input. */
    public static final class D2DStatus {
        public final int extIdx;
        public final long stagingAddr;
        public final long effectiveAddr;
        /** True when staging buffer exists and effective address matches it. */
        public final boolean fired;
        /** True when staging exists but effective differs (graph reads stale data). */
        public final boolean addressDrift;

        public D2DStatus(int extIdx, long stagingAddr, long effectiveAddr) {
            this.extIdx = extIdx;
            this.stagingAddr = stagingAddr;
            this.effectiveAddr = effectiveAddr;
            this.fired = stagingAddr != 0 && stagingAddr == effectiveAddr;
            this.addressDrift = stagingAddr != 0 && stagingAddr != effectiveAddr;
        }

        @Override
        public String toString() {
            return String.format("ext[%d] staging=0x%x effective=0x%x fired=%s drift=%s",
                    extIdx, stagingAddr, effectiveAddr, fired, addressDrift);
        }
    }

    /** Per-segment execution state at snapshot time. */
    public static final class SegmentStepInfo {
        public final int segmentIdx;
        public final int execCount;
        public final int replayCount;
        /** ExecutionPhase ordinal: 0=WARMUP, 1=COMPILING, 2=COMPILED, 3=REPLAYING, 4=SLOT_BY_SLOT. */
        public final int phase;
        public final boolean wasReplayed;

        public SegmentStepInfo(int segmentIdx, int execCount, int replayCount, int phase) {
            this.segmentIdx = segmentIdx;
            this.execCount = execCount;
            this.replayCount = replayCount;
            this.phase = phase;
            this.wasReplayed = (phase == 3); // REPLAYING
        }

        @Override
        public String toString() {
            return String.format("seg[%d] exec=%d replay=%d phase=%d replaying=%s",
                    segmentIdx, execCount, replayCount, phase, wasReplayed);
        }
    }

    /** Parsed capture stats from the pipe-delimited native string. */
    public static final class CaptureStats {
        public final int captured;
        public final int oomRetrying;
        public final int permFailed;
        public final int nonCapturable;
        public final int tooSmall;
        public final int addrUnstable;

        public CaptureStats(int captured, int oomRetrying, int permFailed,
                            int nonCapturable, int tooSmall, int addrUnstable) {
            this.captured = captured;
            this.oomRetrying = oomRetrying;
            this.permFailed = permFailed;
            this.nonCapturable = nonCapturable;
            this.tooSmall = tooSmall;
            this.addrUnstable = addrUnstable;
        }

        /** Parse "captured=N|oomRetrying=N|permFailed=N|nonCapt=N|tooSmall=N|addrUnstable=N". */
        public static CaptureStats parse(String raw) {
            if (raw == null || raw.isEmpty()) {
                return new CaptureStats(0, 0, 0, 0, 0, 0);
            }
            int captured = 0, oomRetrying = 0, permFailed = 0;
            int nonCapt = 0, tooSmall = 0, addrUnstable = 0;
            for (String part : raw.split("\\|")) {
                String[] kv = part.split("=", 2);
                if (kv.length != 2) continue;
                try {
                    int val = Integer.parseInt(kv[1].trim());
                    switch (kv[0].trim()) {
                        case "captured": captured = val; break;
                        case "oomRetrying": oomRetrying = val; break;
                        case "permFailed": permFailed = val; break;
                        case "nonCapt": nonCapt = val; break;
                        case "tooSmall": tooSmall = val; break;
                        case "addrUnstable": addrUnstable = val; break;
                    }
                } catch (NumberFormatException ignored) {}
            }
            return new CaptureStats(captured, oomRetrying, permFailed, nonCapt, tooSmall, addrUnstable);
        }

        @Override
        public String toString() {
            return String.format("captured=%d oomRetrying=%d permFailed=%d nonCapt=%d tooSmall=%d addrUnstable=%d",
                    captured, oomRetrying, permFailed, nonCapturable, tooSmall, addrUnstable);
        }
    }

    /** Immutable snapshot of plan state captured immediately after a replay() call. */
    public static final class StepSnapshot {
        public final int executeCount;
        public final int planPhase;
        public final boolean pointersStable;
        public final int frozenExecCount;
        public final Map<Integer, D2DStatus> d2dStatusByExtIdx;
        public final List<SegmentStepInfo> segments;
        public final CaptureStats captureStats;
        public final List<Integer> driftingSegments;

        public StepSnapshot(int executeCount, int planPhase, boolean pointersStable,
                            int frozenExecCount, Map<Integer, D2DStatus> d2dStatusByExtIdx,
                            List<SegmentStepInfo> segments, CaptureStats captureStats,
                            List<Integer> driftingSegments) {
            this.executeCount = executeCount;
            this.planPhase = planPhase;
            this.pointersStable = pointersStable;
            this.frozenExecCount = frozenExecCount;
            this.d2dStatusByExtIdx = Collections.unmodifiableMap(d2dStatusByExtIdx);
            this.segments = Collections.unmodifiableList(segments);
            this.captureStats = captureStats;
            this.driftingSegments = Collections.unmodifiableList(driftingSegments);
        }

        /** Q1: Did D2D copy fire for every variable ext input with a staging buffer? */
        public boolean allD2DCopiesFired() {
            return d2dStatusByExtIdx.values().stream().allMatch(s -> s.fired);
        }

        /** Ext input indices where D2D has a staging buffer but address drifted (did not fire correctly). */
        public List<Integer> failedD2DExtIndices() {
            return d2dStatusByExtIdx.values().stream()
                    .filter(s -> s.addressDrift)
                    .map(s -> s.extIdx)
                    .collect(Collectors.toList());
        }

        /** Q4: Segment indices that were replayed (phase == REPLAYING). */
        public List<Integer> replayedSegments() {
            return segments.stream()
                    .filter(s -> s.wasReplayed)
                    .map(s -> s.segmentIdx)
                    .collect(Collectors.toList());
        }

        /** Segment indices that ran slot-by-slot (phase == SLOT_BY_SLOT). */
        public List<Integer> slotBySlotSegments() {
            return segments.stream()
                    .filter(s -> s.phase == 4) // SLOT_BY_SLOT
                    .map(s -> s.segmentIdx)
                    .collect(Collectors.toList());
        }

        /** Q2: Any segments with pointer drift? */
        public boolean hasPointerDrift() {
            return !driftingSegments.isEmpty();
        }

        /** Q10: Ext input indices where staging address != 0 but has address drift. */
        public List<Integer> driftingExtIndices() {
            return d2dStatusByExtIdx.values().stream()
                    .filter(s -> s.addressDrift)
                    .map(s -> s.extIdx)
                    .collect(Collectors.toList());
        }

        /**
         * Q3/Q5: Describe what changed between this snapshot and a previous one.
         * Returns a human-readable summary of phase transitions, D2D status changes,
         * and capture stats deltas.
         */
        public String describeChangesFrom(StepSnapshot prev) {
            if (prev == null) return "no previous snapshot";
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("step %d -> %d: ", prev.executeCount, executeCount));

            if (planPhase != prev.planPhase) {
                sb.append(String.format("phase %d->%d ", prev.planPhase, planPhase));
            }
            if (pointersStable != prev.pointersStable) {
                sb.append(String.format("ptrsStable %s->%s ", prev.pointersStable, pointersStable));
            }

            // D2D changes
            long prevFired = prev.d2dStatusByExtIdx.values().stream().filter(s -> s.fired).count();
            long nowFired = d2dStatusByExtIdx.values().stream().filter(s -> s.fired).count();
            if (prevFired != nowFired) {
                sb.append(String.format("d2dFired %d->%d ", prevFired, nowFired));
            }

            // Capture stats delta
            if (captureStats.captured != prev.captureStats.captured) {
                sb.append(String.format("captured %d->%d ",
                        prev.captureStats.captured, captureStats.captured));
            }
            if (captureStats.permFailed != prev.captureStats.permFailed) {
                sb.append(String.format("permFailed %d->%d ",
                        prev.captureStats.permFailed, captureStats.permFailed));
            }

            // Segment replay progression
            int prevReplaying = (int) prev.segments.stream().filter(s -> s.wasReplayed).count();
            int nowReplaying = (int) segments.stream().filter(s -> s.wasReplayed).count();
            if (prevReplaying != nowReplaying) {
                sb.append(String.format("segsReplaying %d->%d ", prevReplaying, nowReplaying));
            }

            // Drifting segments
            if (!driftingSegments.isEmpty()) {
                sb.append("DRIFT:seg").append(driftingSegments).append(" ");
            }

            return sb.toString().trim();
        }

        @Override
        public String toString() {
            return String.format(
                    "StepSnapshot{exec=%d phase=%d ptrsStable=%s frozenExec=%d d2d=%d/%d segs=%d capture=%s drift=%s}",
                    executeCount, planPhase, pointersStable, frozenExecCount,
                    d2dStatusByExtIdx.values().stream().filter(s -> s.fired).count(),
                    d2dStatusByExtIdx.size(),
                    segments.size(), captureStats, driftingSegments);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Cross-stream testing API
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Create an independent CUDA stream for testing cross-stream sync.
     * The returned pointer must be freed with {@link #destroyTestStream(Pointer)}.
     * @return opaque stream pointer, or null on CPU
     */
    public Pointer createTestStream() {
        return nativeOps().dspCreateTestStream();
    }

    /**
     * Destroy a test stream created by {@link #createTestStream()}.
     */
    public void destroyTestStream(Pointer streamPtr) {
        if (streamPtr != null) {
            nativeOps().dspDestroyTestStream(streamPtr);
        }
    }

    /**
     * Write host data to an ext input's device buffer on the LC default stream.
     * After this call, the device buffer is authoritative.
     * @param extIdx  external input index
     * @param srcHost host pointer to source data
     * @param numBytes bytes to copy
     * @return 0 on success
     */
    public int writeDeviceBufferOnDefaultStream(int extIdx, Pointer srcHost, long numBytes) {
        return nativeOps().dspWriteDeviceBufferOnDefaultStream(
            requireHandle(), extIdx, srcHost, numBytes);
    }

    /**
     * Write host data to an ext input's device buffer on an explicit stream.
     * Creates a cross-stream hazard that performPreReplaySync must handle.
     * @param extIdx    external input index
     * @param srcHost   host pointer to source data
     * @param numBytes  bytes to copy
     * @param streamPtr pointer from {@link #createTestStream()}
     * @return 0 on success
     */
    public int writeDeviceBufferOnExplicitStream(int extIdx, Pointer srcHost, long numBytes,
                                                  Pointer streamPtr) {
        return nativeOps().dspWriteDeviceBufferOnExplicitStream(
            requireHandle(), extIdx, srcHost, numBytes, streamPtr);
    }

    /**
     * Synchronize a CUDA stream. Use to insert explicit sync barriers in tests.
     * @param streamPtr pointer from {@link #createTestStream()}, or null for default
     * @return 0 on success
     */
    public int syncStream(Pointer streamPtr) {
        return nativeOps().dspSyncStream(streamPtr);
    }

    /**
     * Query whether an ext input's device buffer is authoritative (device has fresh data).
     * @return true if device is authoritative (isPrimaryActual() == false)
     */
    public boolean isExtInputDeviceAuthoritative(int extIdx) {
        return nativeOps().dspIsExtInputDeviceAuthoritative(requireHandle(), extIdx) == 1;
    }

    /**
     * Get the DSP execution stream pointer.
     * @return opaque pointer to the DSP stream, or null
     */
    public Pointer getExecutionStream() {
        return nativeOps().dspGetExecutionStream(requireHandle());
    }

    /**
     * Get the LaunchContext default stream pointer.
     * @return opaque pointer to the default stream, or null
     */
    public Pointer getDefaultStream() {
        return nativeOps().dspGetDefaultStream();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Arg generation & refresh (per-segment)
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Current arg generation counter for a segment.
     * Incremented each time the segment's argument table is refreshed.
     */
    public long segmentArgGeneration(int segIdx) {
        return nativeOps().getPlanSegmentArgGeneration(requireHandle(), segIdx);
    }

    /**
     * Arg generation that was active when the segment's CUDA graph was captured.
     * If current generation != captured generation, args may be stale.
     */
    public long segmentCapturedArgGeneration(int segIdx) {
        return nativeOps().getPlanSegmentCapturedArgGeneration(requireHandle(), segIdx);
    }

    /**
     * Whether a segment needs its argument table refreshed before replay.
     * @return 1 if refresh needed, 0 if args are current
     */
    public boolean segmentNeedsArgRefresh(int segIdx) {
        return nativeOps().getPlanSegmentNeedsArgRefresh(requireHandle(), segIdx) == 1;
    }

    /**
     * Hash key of the captured input addresses for a segment.
     * Used to detect pointer drift since capture.
     */
    public long segmentCapturedInputAddrKey(int segIdx) {
        return nativeOps().getPlanSegmentCapturedInputAddrKey(requireHandle(), segIdx);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-execution stats (from last execute() call)
    // ─────────────────────────────────────────────────────────────────────────

    /** Segments that ran warmup during last execute(). -1 if unavailable. */
    public int lastExecSegmentsWarmup() { return nativeOps().getLastExecSegmentsWarmup(requireHandle()); }
    /** Segments that completed CUDA graph capture during last execute(). */
    public int lastExecSegmentsCaptured() { return nativeOps().getLastExecSegmentsCaptured(requireHandle()); }
    /** Segments that ran graph replay during last execute(). */
    public int lastExecSegmentsReplayed() { return nativeOps().getLastExecSegmentsReplayed(requireHandle()); }
    /** Segments that ran slot-by-slot during last execute(). */
    public int lastExecSegmentsSlotBySlot() { return nativeOps().getLastExecSegmentsSlotBySlot(requireHandle()); }
    /** Segments that failed during last execute(). */
    public int lastExecSegmentsFailed() { return nativeOps().getLastExecSegmentsFailed(requireHandle()); }
    /** Total segments dispatched during last execute(). */
    public int lastExecSegmentsTotal() { return nativeOps().getLastExecSegmentsTotal(requireHandle()); }
    /** Sync level from last execute(): 0=NONE, 1=EVENT, 2=STREAM, 3=FULL_DEVICE. */
    public int lastExecSyncLevel() { return nativeOps().getLastExecSyncLevel(requireHandle()); }
    /** Stream sync count from last execute(). */
    public int lastExecStreamSyncCount() { return nativeOps().getLastExecStreamSyncCount(requireHandle()); }
    /** Consecutive executions with no variable ext input change. */
    public int lastExecConsecutiveUnchangedCount() { return nativeOps().getLastExecConsecutiveUnchangedCount(requireHandle()); }

    // ─────────────────────────────────────────────────────────────────────────
    // Buffer coloring
    // ─────────────────────────────────────────────────────────────────────────

    /** Whether buffer coloring is currently applied on this plan. */
    public boolean bufferColoringApplied() { return nativeOps().getPlanBufferColoringApplied(requireHandle()); }
    /** Number of colors assigned by coloring (0 if not computed). */
    public int bufferColoringNumColors() { return nativeOps().getPlanBufferColoringNumColors(requireHandle()); }
    /** Estimated bytes saved by coloring. */
    public long bufferColoringBytesSaved() { return nativeOps().getPlanBufferColoringBytesSaved(requireHandle()); }
    /** Color assigned to a specific slot (-1 if uncolored). */
    public int slotColor(int slotIdx) { return nativeOps().getPlanSlotColor(requireHandle(), slotIdx); }

    // ─────────────────────────────────────────────────────────────────────────
    // Buffer pool (static — not plan-specific)
    // ─────────────────────────────────────────────────────────────────────────

    /** Total bytes currently pooled on the given device. */
    public static long bufferPoolPooledBytes(int deviceId) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getBufferPoolPooledBytes(deviceId);
    }
    /** Number of buffers currently in the pool on the given device. */
    public static int bufferPoolPooledCount(int deviceId) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getBufferPoolPooledCount(deviceId);
    }
    /** Lifetime acquire count on the given device. */
    public static long bufferPoolTotalAcquired(int deviceId) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getBufferPoolTotalAcquired(deviceId);
    }
    /** Lifetime reuse count (acquires satisfied from pool) on the given device. */
    public static long bufferPoolTotalReused(int deviceId) {
        return NativeOpsHolder.getInstance().getDeviceNativeOps().getBufferPoolTotalReused(deviceId);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Debugger bridge
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Get a DspDebugger bound to the same SameDiff instance.
     */
    public DspDebugger debugger() {
        return DspDebugger.attach(sd);
    }

    private NativeOps nativeOps() {
        return NativeOpsHolder.getInstance().getDeviceNativeOps();
    }

    /**
     * Attach the process-primary backend owner to a natively returned array
     * wrapper. DspHandle is primary-backend-scoped (see nativeOps()), and raw
     * wrappers coming back over JNI carry no owner of their own.
     */
    private static OpaqueNDArray owned(OpaqueNDArray array) {
        if (array != null && !array.isNull()) {
            array.attachOwner(OpaqueDataBuffer.primaryOwner());
        }
        return array;
    }
}
