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
import org.nd4j.autodiff.functions.DifferentialFunction;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.VariableType;
import org.nd4j.autodiff.samediff.internal.SessionMemMgr;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.BaseNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.nd4j.nativeblas.OpaqueConstantShapeBuffer;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.nativeblas.OpaqueLaunchContext;
import org.nd4j.linalg.api.memory.deallocation.OpaqueDataBufferDeallocator;
import org.nd4j.nativeblas.OpaqueShapeList;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;

import java.io.Closeable;
import java.util.*;

/**
 * Executes a compiled {@link DynamicShapePlan} with pre-wired index-based slot access.
 *
 * <p>Key optimizations over standard InferenceSession execution:</p>
 * <ol>
 *   <li><b>Array-indexed variable storage</b> — {@code INDArray[] slots} replaces
 *       {@code Map<String, SDValue>}</li>
 *   <li><b>Pre-resolved input wiring</b> — integer indices replace {@code op.argNames()}
 *       + HashMap.get()</li>
 *   <li><b>Pre-computed liveness</b> — {@code releaseAtStep[i]} replaces
 *       {@code arrayUseTracker}</li>
 *   <li><b>shapeFunctionOverride(true)</b> — tells C++ to skip redundant shape calc +
 *       prepareOutputs</li>
 *   <li><b>Pre-allocated OpContext pool</b> — {@code OpContext[step]} replaces per-op
 *       pool polling</li>
 *   <li><b>Per-slot shape cache</b> — avoids hash key computation when shapes unchanged</li>
 * </ol>
 *
 * @see DynamicShapePlan
 * @see DynamicShapeSlot
 */
@Slf4j
public class DynamicShapePlanExecutor implements Closeable {

    private static final boolean TIMING_ENABLED = Boolean.parseBoolean(
            System.getProperty("org.nd4j.inference.timing", "false"));

    private static final boolean SHAPE_OVERRIDE = Boolean.parseBoolean(
            System.getProperty("org.nd4j.inference.dynamicShapePlan.shapeOverride", "true"));


    private final SameDiff sd;
    private final SessionMemMgr mmgr;

    /** The plan this executor is currently configured for. */
    private DynamicShapePlan currentPlan;

    /** Flat output array slots: stores op outputs by slot index. */
    private INDArray[] outputSlots;

    /** External input array cache: resolved constant/variable/placeholder arrays. */
    private INDArray[] externalInputs;

    /** BitSet tracking which slots are currently live (have valid arrays). */
    private BitSet liveSlots;

    /** Pending DataBuffers to close. Accumulated during execution and periodically flushed
     *  to reclaim GPU memory mid-execution (not just at the end). */
    private ArrayList<DataBuffer> pendingClose;

    /** Persistent dedup sets across all flushes within one execute() call.
     *  Identity dedup prevents processing the same DataBuffer object twice (views sharing parents).
     *  ODB dedup prevents double-close of the same native OpaqueDataBuffer.
     *  GPU address dedup is per-batch (local to each freePendingBuffers call) to prevent
     *  double-free of the same GPU address within a single flush batch, while allowing
     *  pool-reused addresses to be freed in subsequent flushes. */
    private Set<DataBuffer> seenIdentity;
    private HashSet<Long> closedOdbAddresses;

    /** Buffers deferred from a mid-execution flush because their GPU address was still
     *  used by a live slot (view of parent). Re-checked on the next flush when the view
     *  slot may have been released. */
    private ArrayList<DataBuffer> deferredClose;

    /** Flush pendingClose every RELEASE_FLUSH_INTERVAL ops during execution to reduce
     *  peak GPU memory. Vision encoder with 1962 ops accumulates ~10GB of dead intermediates
     *  if we only flush at the end. Periodic flushing reduces peak by ~50%. */
    private static final int RELEASE_FLUSH_INTERVAL = Integer.getInteger("nd4j.dsp.flushInterval", 100);

    /** Persistent buffer pool for cross-execution array reuse (avoids mmgr round-trip each step). */
    private LocalBufferPool localPool;

    /** Slot-indexed array cache: persists across execute() calls for O(1) array reuse.
     *  Same slot always produces the same shape in autoregressive decoding, so we cache
     *  by slot index instead of using TreeMap lookup. Non-view arrays are stored here
     *  when released; views go to pendingClose. Output slots are NOT cached (claimed by caller). */
    private INDArray[] slotArrayCache;

    /** Tracks which slots produce view outputs (C++ replaces our pre-allocated buffer).
     *  Set after first execution; skip allocation for these slots on subsequent steps. */
    private boolean[] slotIsViewProducer;

    /** Tracks which slot indices are final outputs (claimed by caller). These slots must
     *  always have real allocations even if they're view producers. */
    private BitSet outputSlotSet;

    /** Persistent OpContext pool (avoids native allocation each step). */
    private final ArrayDeque<OpContext> ctxPool = new ArrayDeque<>();

    // Timing accumulators
    private long timingWireInputsNs, timingSyncNs, timingShapeNs, timingAllocNs, timingExecNs, timingReleaseNs;
    private int timingShapeHits, timingShapeMisses;
    private int timingZeroSkipped, timingZeroApplied;
    private int timingPoolHits, timingPoolMisses;
    // Cache miss diagnostics
    private Map<String, Integer> timingCacheMissReasons = new HashMap<>();
    private int timingCacheLeakedConstant;
    private long timingCacheLeakedConstantBytes;
    // Release diagnostics
    private int pendingCloseCount, pendingCloseViewCount;
    private long pendingCloseBytes;
    private int totalFlushedCount;
    private long totalFlushedBytes;
    // Cross-device replica diagnostics
    private int replicaCount;
    private long replicaBytes;
    private int replicaToDev0Count, replicaToDev1Count;
    private long replicaToDev0Bytes, replicaToDev1Bytes;
    private int wrongDeviceCacheEjections;

    public DynamicShapePlanExecutor(SameDiff sd, SessionMemMgr mmgr) {
        this.sd = sd;
        this.mmgr = mmgr;
    }

    /**
     * Initialize the executor for a specific plan.
     *
     * <p>IMPORTANT: This always clears the plan's per-slot shape caches, even if the same plan
     * is being reused. After a session reset, the cached DynamicShapePlan may contain stale
     * DataBuffer references in slot.cachedOutputShapes that point to freed GPU memory.
     * Using these stale references causes memory corruption ("double free or corruption (out)").</p>
     */
    public void initialize(DynamicShapePlan plan) {
        // ALWAYS clear shape caches to avoid stale DataBuffer references from previous sessions.
        plan.clearAllShapeCaches();

        if (currentPlan == plan) {
            return;
        }
        // Plan changed — flush old pool and slot cache when switching plans
        if (localPool != null) {
            localPool.flushTo(mmgr);
        }
        closeSlotArrayCache();
        currentPlan = plan;
        int totalSlots = plan.getTotalOutputSlots();
        outputSlots = new INDArray[totalSlots];
        externalInputs = new INDArray[plan.getExternalInputKeys().length];
        liveSlots = new BitSet(totalSlots);
        pendingClose = new ArrayList<>();
        localPool = new LocalBufferPool();
        slotArrayCache = new INDArray[totalSlots];
        slotIsViewProducer = new boolean[totalSlots];
    }

    /**
     * Execute the plan with the given placeholder arrays.
     *
     * @param plan              the compiled plan
     * @param placeholderArrays placeholder name -> INDArray
     * @return map of requested output variable name -> INDArray
     */
    public Map<String, INDArray> execute(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        if (currentPlan != plan) {
            initialize(plan);
        }

        if (TIMING_ENABLED) {
            timingWireInputsNs = timingSyncNs = timingShapeNs = timingAllocNs = timingExecNs = timingReleaseNs = 0;
            timingShapeHits = timingShapeMisses = 0;
            timingZeroSkipped = timingZeroApplied = 0;
            timingPoolHits = timingPoolMisses = 0;
            timingCacheMissReasons.clear();
            timingCacheLeakedConstant = 0;
            timingCacheLeakedConstantBytes = 0;
            pendingCloseCount = pendingCloseViewCount = 0;
            pendingCloseBytes = 0;
        }

        // Clear output slots from the previous execution.
        Arrays.fill(outputSlots, null);
        Arrays.fill(externalInputs, null);
        liveSlots.clear();
        if (pendingClose == null) pendingClose = new ArrayList<>();
        pendingClose.clear();

        // Initialize persistent dedup sets for this execution.
        seenIdentity = Collections.newSetFromMap(new IdentityHashMap<>());
        closedOdbAddresses = new HashSet<>();
        if (deferredClose == null) deferredClose = new ArrayList<>();
        deferredClose.clear();
        totalFlushedCount = 0;
        totalFlushedBytes = 0;
        replicaCount = 0;
        replicaBytes = 0;
        replicaToDev0Count = replicaToDev1Count = 0;
        replicaToDev0Bytes = replicaToDev1Bytes = 0;
        wrongDeviceCacheEjections = 0;

        // Resolve external inputs (constants, variables, placeholders)
        resolveExternalInputs(plan, placeholderArrays);

        NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
        DynamicShapeSlot[] slots = plan.getSlots();
        if (localPool == null) localPool = new LocalBufferPool();

        // Cache the execution stream once per execute() call.
        Pointer execStream = null;
        try {
            OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
            if (lc != null) {
                execStream = nativeOps.lcExecutionStream(lc);
                if (execStream != null) execStream.retainReference();
            }
        } catch (Exception e) {
            // CPU backend or unavailable
        }

        // Build set of output slot indices — these slots must always have real allocations
        // even if they're view producers, because the caller needs the data.
        outputSlotSet = new BitSet();
        for (int si : plan.getOutputNameToSlotIndex().values()) {
            if (si >= 0) outputSlotSet.set(si);
        }

        Map<String, INDArray> results = new LinkedHashMap<>();
        try {
            for (int stepIdx = 0; stepIdx < slots.length; stepIdx++) {
                DynamicShapeSlot slot = slots[stepIdx];
                OpContext ctx = ctxPool.pollFirst();
                if (ctx == null) {
                    ctx = Nd4j.getExecutioner().buildContext();
                }
                ctx.purge();

                try {
                    if (stepIdx % 500 == 0 || stepIdx == slots.length - 1) {
                        log.info("DSP step {}/{}: op={}", stepIdx, slots.length, slot.getOpName());
                    }
                    executeSlot(slot, ctx, nativeOps, localPool, execStream);
                } catch (Exception e) {
                    log.error("Error executing slot {} ({}): {}", stepIdx, slot.getOpName(), e.getMessage());
                    // Clear the slot cache on failure — a failed execution may leave
                    // cached arrays with corrupted shape info or stale GPU pointers.
                    // Without this, every subsequent execute() hits the same stale cache.
                    closeSlotArrayCache();
                    throw new RuntimeException("DynamicShapePlan execution failed at step " + stepIdx +
                            " (" + slot.getOpName() + ")", e);
                }

                ctx.purgeForReuse();
                ctxPool.offerFirst(ctx);

                // Mark dead slots for deferred close. Don't close now because:
                // (1) GPU kernels may still be using the buffer on the execution stream
                // (2) View arrays share GPU pointers — dedup is only safe post-commit
                long tRelease0 = TIMING_ENABLED ? System.nanoTime() : 0;
                int[] toRelease = plan.getReleaseAtStep()[stepIdx];
                for (int slotIdx : toRelease) {
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null && liveSlots.get(slotIdx)) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && !buf.isConstant()) {
                            boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[slotIdx];
                            if (isViewSlot) {
                                // View producer — the buffer belongs to the input (C++ made
                                // this a view of the input's GPU memory). Don't close or cache
                                // it; the input slot manages its own buffer lifecycle.
                            } else if (buf.closeable() && slotArrayCache != null) {
                                // Non-view producer — cache for O(1) reuse on next execute().
                                INDArray prev = slotArrayCache[slotIdx];
                                if (prev != null && !prev.wasClosed()) {
                                    DataBuffer pbuf = prev.data();
                                    if (pbuf != null && !pbuf.wasClosed() && pbuf.closeable() && !pbuf.isConstant()) {
                                        pendingClose.add(pbuf);
                                    }
                                }
                                slotArrayCache[slotIdx] = arr;
                            } else if (buf.closeable()) {
                                pendingClose.add(buf);
                            }
                            if (TIMING_ENABLED && !isViewSlot) {
                                pendingCloseBytes += buf.length() * buf.getElementSize();
                                pendingCloseCount++;
                            }
                        }
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                // Periodically flush dead buffers to reclaim GPU memory mid-execution.
                // This prevents ~10GB intermediate accumulation in vision encoder (1962 ops).
                // The sync cost (~1ms per flush) is negligible vs. the GPU memory savings.
                if (stepIdx > 0 && stepIdx % RELEASE_FLUSH_INTERVAL == 0 && !pendingClose.isEmpty()) {
                    flushPendingClose(nativeOps, execStream);
                }

                if (TIMING_ENABLED) timingReleaseNs += System.nanoTime() - tRelease0;
            }

            // Claim output arrays directly from slots — no dup, no D2H sync.
            // The caller owns the returned arrays and must close them when done.
            // This avoids 61 per-output dup + commit + getFloat calls per decode step
            // (~3.5x speedup for autoregressive decoding with KV cache).
            Nd4j.getExecutioner().commit();
            {
                Map<String, Integer> outputMap = plan.getOutputNameToSlotIndex();
                int viewFlagFixCount = 0;
                int lengthViewCount = 0;
                for (Map.Entry<String, Integer> entry : outputMap.entrySet()) {
                    int slotIdx = entry.getValue();
                    INDArray arr = outputSlots[slotIdx];
                    if (arr != null) {
                        // Fix IS_VIEW flag set by C++ ops. C++ ops (concat, reshape, etc.)
                        // sometimes set the IS_VIEW flag in shapeInfo even for outputs that
                        // OWN their GPU memory (not actual views). This makes isView()=true,
                        // which causes arr.close() to skip freeing GPU memory → 30MB/step leak.
                        // Only clear the flag if the array actually owns its buffer (not a
                        // true view-producer slot) and length()==data().length().
                        if (arr.isView() && arr.data() != null && !arr.data().wasClosed()) {
                            boolean isViewProducer = slotIsViewProducer != null && slotIsViewProducer[slotIdx];
                            long arrLen = arr.length();
                            long dataLen = arr.data().length();
                            boolean lengthView = arrLen < dataLen;
                            boolean flagView = ArrayOptionsHelper.isView(arr.shapeInfoJava());
                            if (!isViewProducer && !lengthView && flagView) {
                                // C++ set the IS_VIEW flag but this is not a true view.
                                // Clear the flag so close() works correctly.
                                long[] shapeInfo = arr.shapeInfoJava();
                                long options = shapeInfo[shapeInfo.length - 3];
                                options &= ~ArrayOptionsHelper.IS_VIEW;
                                shapeInfo[shapeInfo.length - 3] = options;
                                viewFlagFixCount++;
                            } else if (lengthView) {
                                lengthViewCount++;
                            }
                        }
                        results.put(entry.getKey(), arr);
                        // Remove from slots so closePendingBuffers won't free it
                        outputSlots[slotIdx] = null;
                        liveSlots.clear(slotIdx);
                    }
                }
                if (viewFlagFixCount > 0 || lengthViewCount > 0) {
                    log.info("  Output view fix: {} flag-only views fixed, {} length-based views (unfixable)",
                            viewFlagFixCount, lengthViewCount);
                }
                // Log first few length-based view details to understand why data().length() > length()
                if (lengthViewCount > 0) {
                    int logged = 0;
                    for (Map.Entry<String, Integer> entry2 : outputMap.entrySet()) {
                        INDArray arr2 = results.get(entry2.getKey());
                        if (arr2 != null && arr2.data() != null && arr2.length() < arr2.data().length()) {
                            if (logged < 3) {
                                log.info("    View detail: {} arrLen={} dataLen={} ratio={:.2f} shape={} dtype={}",
                                        entry2.getKey(), arr2.length(), arr2.data().length(),
                                        (double) arr2.data().length() / arr2.length(),
                                        java.util.Arrays.toString(arr2.shape()), arr2.dataType());
                                logged++;
                            }
                        }
                    }
                }
            }

            // Collect remaining live slots (non-output intermediates) for cleanup.
            // Skip view-producer slots — their buffers belong to the input.
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && liveSlots.get(i)) {
                    boolean isViewSlot = slotIsViewProducer != null && slotIsViewProducer[i];
                    if (!isViewSlot) {
                        DataBuffer buf = arr.data();
                        if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                            pendingClose.add(buf);
                        }
                    }
                    outputSlots[i] = null;
                    liveSlots.clear(i);
                }
            }

            // Now close ALL pending buffers in one batch. commit() syncs the execution stream
            // first, ensuring all GPU kernels are done before we free their buffers.
            // GPU address dedup prevents double-free for view arrays sharing parent buffers.
            // No new allocations happen between closes, so address reuse is impossible.
            closePendingBuffers(nativeOps, execStream);

            if (TIMING_ENABLED) {
                printTimingSummary(slots.length, localPool);
            }

            return results;
        } finally {
            // Safety: null out remaining slots without closing (already handled above)
            if (outputSlots != null) {
                Arrays.fill(outputSlots, null);
            }
            if (liveSlots != null) {
                liveSlots.clear();
            }
        }
    }

    /**
     * Flush pending dead buffers mid-execution to reclaim GPU memory.
     * Syncs the execution stream, frees dead buffers on that same stream via
     * dbFreeBuffersOnStream(), then trims the pool on the same stream so the
     * pool can reuse the freed memory for subsequent allocations.
     */
    private void flushPendingClose(NativeOps nativeOps, Pointer execStream) {
        // Merge previously deferred buffers (parents whose GPU memory was still live).
        if (!deferredClose.isEmpty()) {
            pendingClose.addAll(deferredClose);
            deferredClose.clear();
        }
        if (pendingClose.isEmpty()) return;

        // Re-fetch a fresh stream pointer. The cached execStream may be stale — C++
        // ContextBuffers::release() can free the underlying cudaStream_t during device
        // context switches in executeSlot(). Dereferencing a freed stream → SIGSEGV.
        Pointer freshStream = getFreshExecStream(nativeOps);
        if (freshStream == null) freshStream = execStream; // fallback to cached

        // Sync execution stream so all GPU kernels using these buffers have completed.
        Nd4j.getExecutioner().commit();
        // Build sorted array of GPU addresses from live slots. Owner buffers whose
        // allocation range overlaps with any live address are deferred — a view in a
        // live slot still needs the parent's GPU memory. Range check catches offset
        // views (e.g., strided_slice) that exact-match would miss.
        long[] liveGpuAddresses = collectLiveGpuAddresses(nativeOps);
        int[] stats = freePendingBuffers(nativeOps, freshStream, liveGpuAddresses);
        // Trim the pool on the execution stream so freed memory is immediately reusable.
        // Without this, cudaFreeAsync enqueues frees but the pool can't reuse until synced.
        if (freshStream != null) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            nativeOps.trimMemoryPoolOnStream(currentDevice, freshStream);
            // Also trim device 1 if multi-GPU. Cross-device frees go to device 1's
            // default stream (see dbFreeBuffersOnStream cross-device path). Without
            // trimming device 1, these frees accumulate as "used" in pool stats.
            int numDevices = nativeOps.getAvailableDevices();
            if (numDevices > 1) {
                for (int d = 0; d < numDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPool(d);
                    }
                }
            }
        }
        if (!deferredClose.isEmpty()) {
            log.info("  Mid-exec flush: freed {}/{} buffers ({}MB), deferred {} (live views), total freed: {}MB",
                    stats[0], stats[1], stats[2], deferredClose.size(), totalFlushedBytes / (1024 * 1024));
        } else {
            log.info("  Mid-exec flush: freed {}/{} buffers ({}MB), total freed so far: {}MB",
                    stats[0], stats[1], stats[2], totalFlushedBytes / (1024 * 1024));
        }
        pendingClose.clear();
    }

    /**
     * Close all pending DataBuffers after execution completes (final flush).
     * Sync the execution stream first, then close with GPU address dedup.
     * Frees on the execution stream so pool memory is immediately reusable.
     */
    private void closePendingBuffers(NativeOps nativeOps, Pointer execStream) {
        // Merge any remaining deferred buffers from mid-execution flushes.
        // At final close, no slots are live so all deferred buffers can be freed.
        if (!deferredClose.isEmpty()) {
            pendingClose.addAll(deferredClose);
            deferredClose.clear();
        }

        if (pendingClose.isEmpty() && totalFlushedCount == 0) return;

        // Re-fetch a fresh stream pointer (same reason as flushPendingClose).
        Pointer freshStream = getFreshExecStream(nativeOps);
        if (freshStream == null) freshStream = execStream;

        if (!pendingClose.isEmpty()) {
            Nd4j.getExecutioner().commit();
            freePendingBuffers(nativeOps, freshStream, null);
        }

        // Trim the pool so cudaFreeAsync-enqueued frees are processed and memory is
        // returned to the driver. Without this, poolUsed grows across execute() calls
        // because the frees are stream-ordered but never synced before the next allocation.
        if (freshStream != null) {
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            nativeOps.trimMemoryPoolOnStream(currentDevice, freshStream);
            // Trim all other devices — cross-device frees went to their default streams
            int numDevices = nativeOps.getAvailableDevices();
            if (numDevices > 1) {
                for (int d = 0; d < numDevices; d++) {
                    if (d != currentDevice) {
                        nativeOps.trimMemoryPool(d);
                    }
                }
            }
        }

        log.info("  Deferred close: {}/{} buffers ({}MB)",
                totalFlushedCount, totalFlushedCount, totalFlushedBytes / (1024 * 1024));
        pendingClose.clear();
    }

    /**
     * Core buffer freeing logic shared by flushPendingClose() and closePendingBuffers().
     * Uses persistent dedup sets (seenIdentity, closedOdbAddresses)
     * that span all flushes within one execute() call.
     *
     * Frees GPU memory on the execution stream (not stream 0) so the pool can reuse
     * freed memory for subsequent allocations on the same stream without cross-stream sync.
     *
     * @param liveGpuAddresses Sorted GPU addresses from live slots (may be null for final close).
     *                         Owner buffers whose allocation range overlaps with any live address
     *                         are deferred — a view in a live slot still references the parent memory.
     * @return int[3] = {freedCount, totalInBatch, freedMB}
     */
    private int[] freePendingBuffers(NativeOps nativeOps, Pointer execStream, long[] liveGpuAddresses) {
        int freedCount = 0;
        long freedBytes = 0;
        int batchSize = pendingClose.size();

        // GPU address dedup is per-batch (not persistent across flushes).
        // Between flushes, the CUDA pool may reuse freed addresses for new allocations.
        // Those are legitimate new allocations that must be freed in the next flush.
        // Persistent GPU dedup would incorrectly skip them → memory leak.
        HashSet<Long> batchGpuAddresses = new HashSet<>();

        for (DataBuffer buf : pendingClose) {
            if (buf == null || buf.wasClosed() || !buf.closeable() || buf.isConstant()) continue;

            OpaqueDataBuffer odb = buf.opaqueBuffer();
            if (odb == null || odb.isNull()) continue;

            boolean isOwner = nativeOps.dbIsOwner(odb);
            long gpuAddr = 0;

            if (isOwner) {
                Pointer special = nativeOps.dbSpecialBuffer(odb);
                if (special != null && special.address() != 0) {
                    gpuAddr = special.address();
                }
            }

            // Check if this owner buffer's GPU memory is still used by a live slot.
            // This happens when C++ creates a view (reshape/slice/identity) — the view
            // ODB points into the parent's allocation range. If the parent slot is marked
            // dead while a view slot is still live, freeing the parent would invalidate
            // the view's memory. Range check catches both zero-offset and offset views.
            if (isOwner && gpuAddr != 0 && liveGpuAddresses != null && liveGpuAddresses.length > 0) {
                long allocSize = buf.length() * buf.getElementSize();
                if (hasLiveViewInRange(liveGpuAddresses, gpuAddr, allocSize)) {
                    deferredClose.add(buf);
                    continue;
                }
            }

            // Layer 1: Java identity dedup (persistent across flushes)
            if (!seenIdentity.add(buf)) continue;

            // Layer 2: OpaqueDataBuffer address dedup (persistent across flushes)
            long odbAddr = odb.address();
            if (odbAddr != 0 && !closedOdbAddresses.add(odbAddr)) continue;

            // Layer 3: GPU address dedup for OWNER ODBs only (per-batch).
            // Only owner ODBs will actually free GPU memory in C++. Non-owner views
            // (isOwner=false) skip this check so the actual owner can be freed later.
            // This prevents: (a) non-owner views blocking owner frees (the old leak bug),
            // and (b) two different owner ODBs double-freeing the same GPU address.
            if (isOwner && gpuAddr != 0 && !batchGpuAddresses.add(gpuAddr)) continue;

            try {
                freedBytes += buf.length() * buf.getElementSize();
                freedCount++;
                // Free on the execution stream so the pool can reuse this memory
                // for the next cudaMallocAsync on the same stream.
                if (execStream != null) {
                    nativeOps.dbFreeBuffersOnStream(odb, execStream);
                } else {
                    nativeOps.dbFreeBuffersOnly(odb);
                }

                // Sync Java-side lifecycle with the native close that just happened.
                odb.tryMarkForDeallocation();
                odb.setNull();
                OpaqueDataBufferDeallocator deallocator = odb.getDeallocator();
                if (deallocator != null) {
                    deallocator.markDeallocated();
                }
            } catch (Exception e) {
                log.warn("  dbFreeBuffersOnStream failed ({}B): {}",
                        buf.length() * buf.getElementSize(), e.getMessage());
            }
        }

        totalFlushedCount += freedCount;
        totalFlushedBytes += freedBytes;
        return new int[]{freedCount, batchSize, (int) (freedBytes / (1024 * 1024))};
    }

    /**
     * Collect GPU addresses from all live slots. Used during mid-execution flushes
     * to prevent freeing parent buffers whose GPU memory is still referenced by
     * view arrays in live slots.
     *
     * Returns a sorted array of GPU addresses for range-based overlap checks.
     * Views created by C++ ops (reshape, slice, etc.) may have non-zero offsets
     * from their parent allocation, so exact-match address checks are insufficient.
     */
    private long[] collectLiveGpuAddresses(NativeOps nativeOps) {
        int count = 0;
        long[] addresses = new long[liveSlots.cardinality()];
        for (int i = liveSlots.nextSetBit(0); i >= 0; i = liveSlots.nextSetBit(i + 1)) {
            INDArray arr = outputSlots[i];
            if (arr == null) continue;
            DataBuffer buf = arr.data();
            if (buf == null || buf.wasClosed()) continue;
            OpaqueDataBuffer odb = buf.opaqueBuffer();
            if (odb == null || odb.isNull()) continue;
            Pointer special = nativeOps.dbSpecialBuffer(odb);
            if (special != null && special.address() != 0) {
                addresses[count++] = special.address();
            }
        }
        long[] result = count == addresses.length ? addresses : Arrays.copyOf(addresses, count);
        Arrays.sort(result);
        return result;
    }

    /**
     * Check if any live GPU address falls within the allocation range [base, base+size).
     * This catches both zero-offset views (same base pointer) and offset views
     * (pointer within the parent's allocation). Uses binary search on sorted live addresses.
     */
    private static boolean hasLiveViewInRange(long[] sortedLiveAddresses, long base, long size) {
        if (sortedLiveAddresses.length == 0 || size <= 0) return false;
        long end = base + size;
        // Binary search for the first address >= base
        int idx = Arrays.binarySearch(sortedLiveAddresses, base);
        if (idx < 0) idx = -(idx + 1); // insertion point
        // Check if any address starting from idx is within [base, end)
        while (idx < sortedLiveAddresses.length && sortedLiveAddresses[idx] < end) {
            return true;
        }
        return false;
    }

    private void resolveExternalInputs(DynamicShapePlan plan, Map<String, INDArray> placeholderArrays) {
        String[] keys = plan.getExternalInputKeys();
        for (int i = 0; i < keys.length; i++) {
            String varName = keys[i];
            INDArray arr = null;

            // Try placeholder first
            if (placeholderArrays != null) {
                arr = placeholderArrays.get(varName);
            }

            // Then try constant/variable from SameDiff
            if (arr == null) {
                SDVariable var = sd.getVariable(varName);
                if (var != null &&
                        (var.getVariableType() == VariableType.CONSTANT ||
                                var.getVariableType() == VariableType.VARIABLE)) {
                    arr = var.getArr();
                }
            }

            externalInputs[i] = arr;
        }
    }

    private void executeSlot(DynamicShapeSlot slot, OpContext ctx, NativeOps nativeOps,
                             LocalBufferPool localPool, Pointer execStream) {
        DifferentialFunction fn = slot.getOp();

        // Step 0: Device placement. Always save the caller's device so we can
        // restore it in the finally block, even if a mid-execution failover
        // switches us to an unexpected device.
        int previousDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        int targetDevice = slot.getTargetDeviceId();
        if (targetDevice >= 0) {
            if (previousDeviceId != targetDevice) {
                Nd4j.getAffinityManager().unsafeSetDevice(targetDevice);
                // Re-fetch execution stream for the target device. The execStream passed
                // in was cached from the original device's launch context — using it for
                // cudaMemsetAsync on a different device's memory fails for non-P2P GPUs.
                try {
                    OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                    if (lc != null) {
                        Pointer deviceStream = nativeOps.lcExecutionStream(lc);
                        if (deviceStream != null) {
                            deviceStream.retainReference();
                            execStream = deviceStream;
                        }
                    }
                } catch (Exception e) {
                    // fall through with original stream
                }
            }
        }

        try {
        // Track replicated input copies for explicit close after execution.
        // replicateToDevice() creates new arrays that leak forever without explicit close
        // (GC cleanup is broken for GPU buffers). Both Step 1b and Step 4c create copies.
        List<DataBuffer> replicatedInputBuffers = null;

        // Step 1: Wire inputs
        long tWire0 = TIMING_ENABLED ? System.nanoTime() : 0;
        int[] inputSourceIndices = slot.getInputSourceIndices();
        INDArray[] inputArrays = slot.getInputArraysBuffer();

        for (int i = 0; i < inputSourceIndices.length; i++) {
            int srcIdx = inputSourceIndices[i];
            if (srcIdx >= 0) {
                inputArrays[i] = outputSlots[srcIdx];
            } else {
                int extIdx = -(srcIdx + 1);
                inputArrays[i] = externalInputs[extIdx];
            }

            if (inputArrays[i] == null) {
                throw new IllegalStateException("Null input at index " + i + " for op " + slot.getOpName() +
                        ", input var: " + slot.getInputVarNames()[i]);
            }
        }
        // Step 1b: Migrate inputs to target device if cross-device.
        // Use dbDeviceId (C++ DataBuffer._deviceId) instead of getDeviceForArray (Java-side)
        // because Java doesn't know about C++ allocation failover — an array may have been
        // intended for device 1 but allocated on device 0 by CudaMemoryPool::allocateFailover.
        //
        // For VIEW inputs: pre-dup to contiguous before replication and track the intermediate
        // buffer for freeing on the execution stream. replicateToDevice's internal dup() also
        // handles views, but its close() frees via dbClose() → default stream (nullptr). DSP
        // trims only the execution stream, so default-stream frees accumulate without being
        // processed → ~30MB/step leak. By pre-duping here and tracking in replicatedInputBuffers,
        // the intermediate is freed on the execution stream by freePendingBuffers → trimmed properly.
        if (targetDevice >= 0) {
            boolean migrated = false;
            for (int i = 0; i < inputArrays.length; i++) {
                INDArray input = inputArrays[i];
                if (input != null && !input.isEmpty() && input.data() != null) {
                    int inputDevice = -1;
                    OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                    if (inputOdb != null && !inputOdb.isNull()) {
                        inputDevice = nativeOps.dbDeviceId(inputOdb);
                    }
                    if (inputDevice >= 0 && inputDevice != targetDevice) {
                        // Pre-dup views to contiguous on the source device. This avoids
                        // replicateToDevice's internal dup() which frees the intermediate
                        // on the default stream (not trimmed by DSP's execution stream trim).
                        INDArray inputToReplicate = input;
                        if (input.isView()) {
                            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                inputToReplicate = input.dup(input.ordering());
                            }
                            // Track contiguous intermediate for freeing on execution stream
                            DataBuffer dupBuf = inputToReplicate.data();
                            if (dupBuf != null && !dupBuf.isConstant()) {
                                if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                replicatedInputBuffers.add(dupBuf);
                            }
                        }
                        INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                        inputArrays[i] = replica;
                        // Track replica for explicit close after execution
                        DataBuffer replicaBuf = replica.data();
                        if (replicaBuf != null && !replicaBuf.isConstant()) {
                            if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                            replicatedInputBuffers.add(replicaBuf);
                        }
                        replicaCount++;
                        long rBytes = replica.length() * replica.data().getElementSize();
                        replicaBytes += rBytes;
                        if (targetDevice == 0) { replicaToDev0Count++; replicaToDev0Bytes += rBytes; }
                        else if (targetDevice == 1) { replicaToDev1Count++; replicaToDev1Bytes += rBytes; }
                        migrated = true;
                    }
                }
            }
            if (migrated) {
                log.trace("Migrated inputs to device {} for op {}", targetDevice, slot.getOpName());
            }
        }
        ctx.setInputArrays(inputArrays);
        if (TIMING_ENABLED) timingWireInputsNs += System.nanoTime() - tWire0;

        // Step 2: Sync INT/LONG inputs if needed
        long tSync0 = TIMING_ENABLED ? System.nanoTime() : 0;
        if (slot.isNeedsIntLongSync()) {
            syncIntLongInputs(inputArrays, slot.isDataDependent(), nativeOps);
        }
        if (TIMING_ENABLED) timingSyncNs += System.nanoTime() - tSync0;

        // Step 3: Compute output shapes
        long tShape0 = TIMING_ENABLED ? System.nanoTime() : 0;
        List<DataBuffer> outShapes = getOrComputeShapes(slot, ctx, fn, inputArrays, nativeOps);
        if (outShapes == null || outShapes.isEmpty()) {
            throw new IllegalStateException("No output shapes for op " + slot.getOpName());
        }
        if (TIMING_ENABLED) timingShapeNs += System.nanoTime() - tShape0;

        // Step 4: Allocate outputs
        long tAlloc0 = TIMING_ENABLED ? System.nanoTime() : 0;
        int[] outputSlotIndices = slot.getOutputSlotIndices();
        INDArray[] outputArrays = new INDArray[outShapes.size()];

        for (int i = 0; i < outShapes.size(); i++) {
            DataBuffer shapeBuffer = outShapes.get(i);
            long[] shapeInfo = shapeBuffer.asLong();
            DataType dt = Shape.dataType(shapeInfo);
            long[] actualShape = Shape.shape(shapeInfo);

            INDArray out = null;
            int slotIdx = (i < outputSlotIndices.length) ? outputSlotIndices[i] : -1;

            // For known view-producing INTERMEDIATE CustomOp slots, use an empty placeholder.
            // C++ will modify it in-place to point to the input's buffer (view).
            // We skip OUTPUT slots — the caller needs the data, so we must allocate
            // even though C++ will replace the buffer. The orphaned allocation for
            // output slots is a one-time cost (they're views, caller can't close them
            // via isView() anyway — the parent in the slot cache holds the GPU memory).
            // We also skip non-CustomOp (legacy) ops because the Java executor validates
            // X.length == Z.length BEFORE calling C++. An empty Z with length=0 fails
            // this check. C++ would handle the empty output (replacing it with a view),
            // but Java rejects it first, causing shape mismatch errors.
            if (slotIdx >= 0 && slotIsViewProducer != null && slotIsViewProducer[slotIdx]
                    && !outputSlotSet.get(slotIdx) && slot.isCustomOp()) {
                out = Nd4j.empty(dt);
                outputArrays[i] = out;
                outputSlots[slotIdx] = out;
                liveSlots.set(slotIdx);
                continue;
            }

            if (Shape.isEmpty(shapeInfo) || numElements(actualShape) == 0) {
                out = Nd4j.emptyWithShape(actualShape, dt);
            } else {
                // Try slot-indexed cache first (O(1) lookup, no TreeMap)
                if (slotIdx >= 0 && slotArrayCache != null) {
                    INDArray cached = slotArrayCache[slotIdx];
                    if (cached != null && !cached.wasClosed()) {
                        DataBuffer cbuf = cached.data();
                        if (cbuf != null && !cbuf.wasClosed() && cbuf.closeable()
                                && cached.dataType() == dt
                                && cbuf.length() >= numElements(actualShape)) {
                            // Check device locality for multi-GPU. If the cached array is on
                            // a different device, DON'T use it — allocate fresh on the target
                            // device instead. Using cross-device memory causes illegal memory
                            // access for non-P2P GPUs, and relying on Step 4c to "detect" it
                            // as a failover creates unnecessary cross-device migrations that
                            // compound memory pressure and corrupt CUDA stream state.
                            boolean wrongDevice = false;
                            if (targetDevice >= 0) {
                                OpaqueDataBuffer cachedOdb = cbuf.opaqueBuffer();
                                if (cachedOdb != null && !cachedOdb.isNull()) {
                                    int cachedDevice = nativeOps.dbDeviceId(cachedOdb);
                                    if (cachedDevice >= 0 && cachedDevice != targetDevice) {
                                        wrongDevice = true;
                                    }
                                }
                            }
                            if (wrongDevice) {
                                // Wrong device — put cached array in pendingClose.
                                // Don't set out — let the code fall through to fresh allocation.
                                pendingClose.add(cbuf);
                                slotArrayCache[slotIdx] = null;
                                wrongDeviceCacheEjections++;
                            } else {
                            reshapeBuffer(cached, actualShape);
                            // Invalidate the cached OpaqueNDArray so a fresh C++ NDArray*
                            // wrapper is created from the current Java-side shape info.
                            // Without this, the stale OpaqueNDArray from the previous execution
                            // may have a C++ shape info pointer that was modified by the C++ op
                            // (e.g., scalar ops can modify output shapes in-place). The Java
                            // side still has the correct shape, but the cached C++ wrapper doesn't.
                            cached.clearOpaqueNDArray();
                            fastZero(cached, nativeOps, execStream);
                            out = cached;
                            slotArrayCache[slotIdx] = null;
                            if (TIMING_ENABLED) { timingPoolHits++; timingZeroApplied++; }
                            } // end else (correct device)
                        } else if (TIMING_ENABLED && cbuf != null) {
                            // Diagnose cache miss reason
                            String reason = !cbuf.closeable() ? "not-closeable(const=" + cbuf.isConstant() + ")"
                                    : cbuf.wasClosed() ? "closed"
                                    : cached.dataType() != dt ? "dtype-mismatch"
                                    : cbuf.length() < numElements(actualShape) ? "too-small(" + cbuf.length() + "<" + numElements(actualShape) + ")"
                                    : "unknown";
                            if (timingCacheMissReasons.size() < 20) {
                                timingCacheMissReasons.merge(reason, 1, Integer::sum);
                            }
                        }
                    }
                    if (out == null) {
                        // Cached array is stale or wrong type — close it
                        if (cached != null && !cached.wasClosed()) {
                            DataBuffer cbuf = cached.data();
                            if (cbuf != null && !cbuf.wasClosed() && cbuf.closeable() && !cbuf.isConstant()) {
                                pendingClose.add(cbuf);
                            } else if (TIMING_ENABLED && cbuf != null && !cbuf.wasClosed() && (cbuf.isConstant() || !cbuf.closeable())) {
                                timingCacheLeakedConstant++;
                                timingCacheLeakedConstantBytes += cbuf.length() * cbuf.getElementSize();
                            }
                        }
                        slotArrayCache[slotIdx] = null;
                    }
                }
                if (out == null) {
                    if (slotIdx >= 0 && outputSlotSet.get(slotIdx)) {
                        // Output arrays are claimed by caller and NOT reused by slot cache.
                        // Allocate DIRECTLY via Nd4j.create() — NOT through mmgr (ArrayCacheMemoryMgr)
                        // which adds 5% headroom (growthFactor=1.05), making data().length() > length()
                        // → isView()=true → closeable()=false → close() silently fails → 30MB/step leak.
                        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            out = Nd4j.create(dt, actualShape);
                        }
                    } else if (slotIdx >= 0 && slotArrayCache != null) {
                        // Intermediate slots get 2x headroom for slot cache reuse.
                        out = allocateForSlotCache(dt, actualShape);
                    } else {
                        out = allocateWithHeadroom(dt, actualShape);
                    }
                    if (TIMING_ENABLED) timingPoolMisses++;
                }
            }
            outputArrays[i] = out;

            if (slotIdx >= 0) {
                outputSlots[outputSlotIndices[i]] = outputArrays[i];
                liveSlots.set(outputSlotIndices[i]);
            }
        }
        ctx.setOutputArrays(outputArrays);
        if (TIMING_ENABLED) timingAllocNs += System.nanoTime() - tAlloc0;

        // Save GPU buffer addresses BEFORE execution for view-producer detection.
        // After execution, if C++ modifies an output in-place to be a view, the GPU
        // address changes (points to input's buffer instead of our pre-allocation).
        // We compare pre/post addresses to detect TRUE view producers, avoiding false
        // positives from 2x headroom (which makes isView()=true for all large arrays).
        // We save raw long addresses (not ODB objects) because C++ modifies the native
        // DataBuffer in-place — the ODB wraps the same C++ object and would return the
        // NEW address after modification.
        long[] preExecGpuAddrs = new long[outputArrays.length];
        for (int i = 0; i < outputArrays.length; i++) {
            INDArray arr = outputArrays[i];
            if (arr != null && !arr.isEmpty()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed()) {
                    OpaqueDataBuffer odb = buf.opaqueBuffer();
                    if (odb != null && !odb.isNull()) {
                        Pointer special = nativeOps.dbSpecialBuffer(odb);
                        if (special != null) preExecGpuAddrs[i] = special.address();
                    }
                }
            }
        }

        // Step 4c: If output allocation failed over to a different device (target OOM),
        // handle based on P2P capability:
        // - P2P available: switch execution to failover device (cross-device memory works)
        // - No P2P: emergency reclaim on original device and retry there (cross-device
        //   memory access causes cudaErrorIllegalAddress on non-P2P GPUs)
        if (targetDevice >= 0) {
            int failoverDevice = -1;
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray out = outputArrays[i];
                if (out != null && !out.isEmpty() && out.data() != null) {
                    OpaqueDataBuffer odb = out.data().opaqueBuffer();
                    if (odb != null && !odb.isNull()) {
                        int actualDevice = nativeOps.dbDeviceId(odb);
                        if (actualDevice >= 0 && actualDevice != targetDevice) {
                            failoverDevice = actualDevice;
                            break;
                        }
                    }
                }
            }
            if (failoverDevice >= 0) {
                boolean p2pAccess = nativeOps.isPeerAccessSupported(targetDevice, failoverDevice);

                if (!p2pAccess) {
                    // Non-P2P failover: can't execute cross-device. Free misplaced outputs,
                    // emergency flush pending buffers on original device, then retry allocation.
                    log.info("Op {} output OOM on device {}, failover to non-P2P device {} — emergency reclaim on device {}",
                            slot.getOpName(), targetDevice, failoverDevice, targetDevice);

                    // Save shapes/dtypes and free misplaced outputs on the failover device
                    int originalTarget = targetDevice;
                    DataType[] retryDtypes = new DataType[outputArrays.length];
                    long[][] retryShapes = new long[outputArrays.length][];
                    for (int i = 0; i < outputArrays.length; i++) {
                        INDArray out = outputArrays[i];
                        if (out == null || out.isEmpty()) continue;
                        DataBuffer db = out.data();
                        if (db == null) continue;
                        OpaqueDataBuffer odb = db.opaqueBuffer();
                        if (odb == null || odb.isNull()) continue;
                        int actualDevice = nativeOps.dbDeviceId(odb);
                        if (actualDevice == failoverDevice) {
                            retryDtypes[i] = out.dataType();
                            retryShapes[i] = out.shape();
                            try {
                                nativeOps.dbFreeBuffersOnly(odb);
                                odb.tryMarkForDeallocation();
                                odb.setNull();
                                OpaqueDataBufferDeallocator dealloc = odb.getDeallocator();
                                if (dealloc != null) dealloc.markDeallocated();
                            } catch (Exception e) {
                                log.warn("Failed to free misplaced output on device {}: {}", failoverDevice, e.getMessage());
                            }
                            outputArrays[i] = null;
                        }
                    }

                    // Switch back to original device
                    Nd4j.getAffinityManager().unsafeSetDevice(originalTarget);
                    nativeOps.clearLastError();
                    Nd4j.getExecutioner().commit();

                    // Emergency flush ALL pending + deferred buffers to maximize memory recovery
                    if (!pendingClose.isEmpty() || !deferredClose.isEmpty()) {
                        flushPendingClose(nativeOps, execStream);
                    }

                    // Re-fetch fresh stream after device switch + flush
                    Pointer freshStream = getFreshExecStream(nativeOps);
                    if (freshStream != null) execStream = freshStream;
                    nativeOps.clearLastError();

                    // Retry allocation on original device
                    boolean retryOk = true;
                    for (int i = 0; i < outputArrays.length; i++) {
                        if (retryShapes[i] == null) continue;
                        try {
                            int slotIdx = (i < outputSlotIndices.length) ? outputSlotIndices[i] : -1;
                            boolean isOutputSlot = slotIdx >= 0 && outputSlotSet.get(slotIdx);
                            INDArray newOut;
                            if (isOutputSlot) {
                                newOut = Nd4j.create(retryDtypes[i], retryShapes[i]);
                            } else if (slotIdx >= 0 && slotArrayCache != null) {
                                newOut = allocateForSlotCache(retryDtypes[i], retryShapes[i]);
                            } else {
                                newOut = Nd4j.create(retryDtypes[i], retryShapes[i]);
                            }
                            // Verify it landed on the correct device
                            OpaqueDataBuffer retryOdb = newOut.data().opaqueBuffer();
                            if (retryOdb != null && !retryOdb.isNull()) {
                                int retryDevice = nativeOps.dbDeviceId(retryOdb);
                                if (retryDevice >= 0 && retryDevice != originalTarget) {
                                    log.error("Emergency reclaim insufficient — retry output for {} also landed on device {}",
                                            slot.getOpName(), retryDevice);
                                    retryOk = false;
                                    try {
                                        nativeOps.dbFreeBuffersOnly(retryOdb);
                                        retryOdb.tryMarkForDeallocation();
                                        retryOdb.setNull();
                                    } catch (Exception ignored) {}
                                    break;
                                }
                            }
                            outputArrays[i] = newOut;
                            fastZero(newOut, nativeOps, execStream);
                            if (slotIdx >= 0) {
                                outputSlots[slotIdx] = newOut;
                                liveSlots.set(slotIdx);
                            }
                            // Update pre-exec GPU address for view-producer detection
                            DataBuffer buf = newOut.data();
                            if (buf != null && !buf.wasClosed()) {
                                OpaqueDataBuffer odb2 = buf.opaqueBuffer();
                                if (odb2 != null && !odb2.isNull()) {
                                    Pointer special = nativeOps.dbSpecialBuffer(odb2);
                                    if (special != null) preExecGpuAddrs[i] = special.address();
                                }
                            }
                        } catch (Exception e) {
                            log.error("Emergency reclaim retry failed for output {} of {}: {}",
                                    i, slot.getOpName(), e.getMessage());
                            retryOk = false;
                            break;
                        }
                    }

                    if (!retryOk) {
                        throw new RuntimeException("OOM on device " + originalTarget + " for op " +
                                slot.getOpName() + " — emergency reclaim freed insufficient memory. " +
                                "Consider reducing model size or sequence length.");
                    }
                    ctx.setOutputArrays(outputArrays);
                    nativeOps.clearLastError();
                    log.info("  Emergency reclaim succeeded — continuing on device {}", originalTarget);

                } else {
                    // P2P failover: transparently switch execution to failover device.
                    // Cross-device memory access works via P2P, so inputs/outputs on different
                    // devices can be accessed directly.
                    log.info("Op {} output failed over from device {} to P2P device {} (OOM) — switching execution",
                            slot.getOpName(), targetDevice, failoverDevice);
                    if (!pendingClose.isEmpty()) {
                        flushPendingClose(nativeOps, execStream);
                    }
                    targetDevice = failoverDevice;
                    Nd4j.getAffinityManager().unsafeSetDevice(targetDevice);
                    nativeOps.clearLastError();
                    Nd4j.getExecutioner().commit();
                    nativeOps.clearLastError();
                    try {
                        OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                        if (lc != null) {
                            Pointer deviceStream = nativeOps.lcExecutionStream(lc);
                            if (deviceStream != null) {
                                deviceStream.retainReference();
                                execStream = deviceStream;
                            }
                        }
                    } catch (Exception e) {
                        // fall through with current stream
                    }
                    // Re-migrate inputs to the failover device
                    for (int i = 0; i < inputArrays.length; i++) {
                        INDArray input = inputArrays[i];
                        if (input != null && !input.isEmpty() && input.data() != null) {
                            int inputDevice = -1;
                            OpaqueDataBuffer inputOdb = input.data().opaqueBuffer();
                            if (inputOdb != null && !inputOdb.isNull()) {
                                inputDevice = nativeOps.dbDeviceId(inputOdb);
                            }
                            if (inputDevice >= 0 && inputDevice != targetDevice) {
                                // Pre-dup views (same reason as Step 1b)
                                INDArray inputToReplicate = input;
                                if (input.isView()) {
                                    try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                                        inputToReplicate = input.dup(input.ordering());
                                    }
                                    DataBuffer dupBuf = inputToReplicate.data();
                                    if (dupBuf != null && !dupBuf.isConstant()) {
                                        if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                        replicatedInputBuffers.add(dupBuf);
                                    }
                                }
                                INDArray replica = Nd4j.getAffinityManager().replicateToDevice(targetDevice, inputToReplicate);
                                inputArrays[i] = replica;
                                DataBuffer replicaBuf = replica.data();
                                if (replicaBuf != null && !replicaBuf.isConstant()) {
                                    if (replicatedInputBuffers == null) replicatedInputBuffers = new ArrayList<>();
                                    replicatedInputBuffers.add(replicaBuf);
                                }
                            }
                        }
                    }
                    ctx.setInputArrays(inputArrays);
                    nativeOps.clearLastError();
                    for (int i = 0; i < outputArrays.length; i++) {
                        INDArray out = outputArrays[i];
                        if (out != null && !out.isEmpty() && out.data() != null) {
                            fastZero(out, nativeOps, execStream);
                        }
                    }
                    nativeOps.clearLastError();
                    ctx.setOutputArrays(outputArrays);
                }
            }
        }

        // Step 5: Execute
        // Clear any stale CUDA errors from fastZero or allocation before op execution.
        // Without this, a failed cudaMemsetAsync (e.g., from cross-device buffer zeroing)
        // leaves a stale error that CudaExecutioner picks up via lastErrorCode() after
        // execCustomOp2, causing a spurious "cudaMemsetAsync failed" exception.
        if (targetDevice >= 0) {
            nativeOps.clearLastError();
        }
        ctx.shapeFunctionOverride(SHAPE_OVERRIDE);

        // Attach native workspace to OpContext if available — this allows C++ ops to
        // use bump allocation for internal temporaries instead of per-op malloc/cudaMalloc.
        // Without this, C++ op temporary buffer overruns corrupt the regular malloc heap
        // metadata, causing "double free or corruption (out)" crashes.
        // The standard InferenceSession path always does this (see executeRegularOperation).
        Pointer wsPtr = mmgr.getNativeWorkspacePointer();
        if (wsPtr != null) {
            ctx.attachWorkspace(wsPtr);
        }

        long tExec0 = TIMING_ENABLED ? System.nanoTime() : 0;
        // Pre-exec shape diagnostic for non-CustomOp (scalar ops)
        if (!slot.isCustomOp()) {
            for (int i = 0; i < outputArrays.length; i++) {
                INDArray outArr = outputArrays[i];
                if (outArr != null) {
                    long javaLen = outArr.length();
                    long[] javaShape = outArr.shape();
                    int[] nativeShape = outArr.shapeInfoDataBuffer().asInt();
                    long nativeRank = nativeShape.length > 0 ? nativeShape[0] : -1;
                    if (nativeRank != javaShape.length) {
                        log.warn("PRE-EXEC shape mismatch for {} output[{}]: javaShape={} javaLen={} nativeShapeInfo={} data.len={}",
                                slot.getOpName(), i, Arrays.toString(javaShape), javaLen,
                                Arrays.toString(nativeShape),
                                outArr.data() != null ? outArr.data().length() : "null");
                    }
                }
            }
        }

        if (slot.isCustomOp()) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            Nd4j.exec((CustomOp) fn, ctx);
        } else {
            Nd4j.exec((Op) fn, ctx);
        }

        // After execution, C++ may have replaced output arrays or modified them in-place
        // to be views. Detect both cases and mark slots as view producers for future steps.
        List<INDArray> ctxOutputs = ctx.getOutputArrays();
        int maxTracked = Math.min(ctxOutputs != null ? ctxOutputs.size() : 0, outputSlotIndices.length);

        if (ctxOutputs != null) {
            for (int i = 0; i < maxTracked; i++) {
                INDArray ctxOut = ctxOutputs.get(i);
                int si = outputSlotIndices[i];
                if (ctxOut == null || si < 0) continue;

                if (ctxOut != outputArrays[i]) {
                    // Case 1: C++ replaced the output with a different object (new view).
                    if (slotIsViewProducer != null) slotIsViewProducer[si] = true;
                    if (!outputArrays[i].isEmpty()) {
                        DataBuffer buf = outputArrays[i].data();
                        if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                            pendingClose.add(buf);
                        }
                    }
                    outputSlots[si] = ctxOut;
                } else if (slotIsViewProducer != null && !slotIsViewProducer[si]) {
                    // Case 2: Check if C++ modified the output's GPU buffer in-place.
                    // Compare pre-execution GPU address with current address.
                    // If they differ, C++ replaced the buffer with a view of the input.
                    // We must NOT use isView() here — 2x headroom allocation makes
                    // isView()=true for ALL arrays >256 elements (false positive).
                    long preAddr = (i < preExecGpuAddrs.length) ? preExecGpuAddrs[i] : 0;
                    if (preAddr != 0) {
                        DataBuffer currentBuf = ctxOut.data();
                        OpaqueDataBuffer currentOdb = (currentBuf != null) ? currentBuf.opaqueBuffer() : null;
                        long currentAddr = 0;
                        if (currentOdb != null && !currentOdb.isNull()) {
                            Pointer special = nativeOps.dbSpecialBuffer(currentOdb);
                            if (special != null) currentAddr = special.address();
                        }
                        if (currentAddr != 0 && preAddr != currentAddr) {
                            // GPU address changed — C++ made this a view of the input.
                            // The original allocation at preAddr is orphaned (one-time cost
                            // on the first execute(); subsequent calls use Nd4j.empty()).
                            slotIsViewProducer[si] = true;
                        }
                    }
                }
            }
        }

        // Release untracked output arrays
        for (int i = 0; i < outputArrays.length; i++) {
            boolean tracked = (i < outputSlotIndices.length && outputSlotIndices[i] >= 0);
            if (!tracked) {
                INDArray arr = outputArrays[i];
                if (ctxOutputs != null && i < ctxOutputs.size()) {
                    INDArray ctxOut = ctxOutputs.get(i);
                    if (ctxOut != null && ctxOut != arr) {
                        arr = ctxOut;
                    }
                }
                if (arr != null) {
                    DataBuffer buf = arr.data();
                    if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                        pendingClose.add(buf);
                    }
                }
            }
        }

        // Release replicated input copies to prevent GPU memory leak.
        // replicateToDevice() creates new arrays for cross-device inputs in Step 1b and
        // Step 4c failover. Without explicit close, these leak forever (GC cleanup is broken
        // for GPU buffers — PhantomRef strong reference cycle). The live view range check
        // in flushPendingClose() safely defers any buffer whose GPU memory is still
        // referenced by a live output slot (e.g., if C++ made an output view of the input).
        if (replicatedInputBuffers != null) {
            for (DataBuffer buf : replicatedInputBuffers) {
                if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                    pendingClose.add(buf);
                }
            }
        }

        if (TIMING_ENABLED) timingExecNs += System.nanoTime() - tExec0;

        } finally {
            // Always restore the caller's device context, even after transparent failover.
            int currentDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (currentDev != previousDeviceId) {
                Nd4j.getAffinityManager().unsafeSetDevice(previousDeviceId);
            }
        }
    }

    private void syncIntLongInputs(INDArray[] inputs, boolean isDataDependent, NativeOps nativeOps) {
        boolean needsSync = isDataDependent;
        if (!needsSync) {
            for (INDArray in : inputs) {
                if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG ||
                        in.dataType() == DataType.BOOL)
                        && !in.data().isConstant()
                        && in.length() <= 32) {
                    needsSync = true;
                    break;
                }
            }
        }

        if (needsSync) {
            Nd4j.getExecutioner().commit();
            for (INDArray in : inputs) {
                if (in != null && !in.isEmpty() && in.data() != null && !in.data().wasClosed()
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG ||
                        in.dataType() == DataType.BOOL)
                        && !in.data().isConstant()) {
                    if (isDataDependent || in.length() <= 32) {
                        nativeOps.dbForceSyncToPrimary(in.data().opaqueBuffer());
                    }
                }
            }
        }
    }

    private List<DataBuffer> getOrComputeShapes(DynamicShapeSlot slot, OpContext ctx,
                                                  DifferentialFunction fn, INDArray[] inputArrays,
                                                  NativeOps nativeOps) {
        long shapeKey = computeShapeKey(slot, inputArrays);

        if (slot.isShapeCacheValid(shapeKey) && !slot.isDataDependent()) {
            if (TIMING_ENABLED) timingShapeHits++;
            return slot.getCachedOutputShapes();
        }

        if (TIMING_ENABLED) timingShapeMisses++;

        List<DataBuffer> outShapes = null;

        if (fn instanceof DynamicCustomOp) {
            ctx.setIArguments(slot.getIArgs());
            ctx.setTArguments(slot.getTArgs());
            ctx.setBArguments(slot.getBArgs());
            ctx.setDArguments(slot.getDArgs());
            outShapes = ((DynamicCustomOp) fn).calculateOutputShapeFromInputs(ctx);
        }

        if (outShapes == null || outShapes.isEmpty()) {
            try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                if (fn instanceof CustomOp) {
                    ctx.setIArguments(slot.getIArgs());
                    ctx.setTArguments(slot.getTArgs());
                    ctx.setBArguments(slot.getBArgs());
                    ctx.setDArguments(slot.getDArgs());

                    long opHash = ((CustomOp) fn).opHash();
                    OpaqueShapeList shapeList;
                    if (!slot.isOutputShapeDependsOnInputValues()) {
                        shapeList = nativeOps.calculateOutputShapesNoSync(null, opHash,
                                ctx.contextPointer());
                    } else {
                        shapeList = nativeOps.calculateOutputShapes2(null, opHash,
                                ctx.contextPointer());
                    }

                    if (nativeOps.lastErrorCode() != 0 || shapeList == null) {
                        throw new RuntimeException("Shape calculation failed for op " +
                                slot.getOpName() + ": " + nativeOps.lastErrorMessage());
                    }

                    outShapes = new ArrayList<>();
                    int numShapes = (int) nativeOps.getShapeListSize(shapeList);
                    for (int e = 0; e < numShapes; e++) {
                        outShapes.add(readShapeFromNative(nativeOps, shapeList, e));
                    }
                    nativeOps.deleteShapeList(shapeList);
                } else {
                    outShapes = fn.calculateOutputShape(ctx);
                }
            }
        }

        if (!slot.isDataDependent() && outShapes != null && !outShapes.isEmpty()) {
            slot.updateShapeCache(shapeKey, outShapes);
        }

        return outShapes;
    }

    private static DataBuffer readShapeFromNative(NativeOps nativeOps, OpaqueShapeList list, int index) {
        LongPointer ptr = new PagedPointer(nativeOps.getShape(list, index)).asLongPointer();
        int rank = (int) ptr.get(0);
        int len = Shape.shapeInfoLength(rank);
        long[] shapeInfo = new long[len];
        ptr.capacity(len);
        ptr.get(shapeInfo, 0, len);

        OpaqueConstantShapeBuffer csb = nativeOps.cacheAndStoreShapeBuffer(shapeInfo);
        if (csb == null) {
            throw new RuntimeException("Failed to cache shape buffer");
        }

        Pointer primaryPtr = nativeOps.getConstantShapeBufferPrimary(csb);
        Pointer specialPtr = nativeOps.getConstantShapeBufferSpecial(csb);

        DataBuffer buffer;
        if (specialPtr != null && specialPtr.address() != 0) {
            buffer = Nd4j.createBuffer(primaryPtr, specialPtr, len, DataType.INT64);
        } else {
            buffer = Nd4j.createBuffer(primaryPtr, len, DataType.INT64);
        }
        buffer.setConstant(true);
        return buffer;
    }

    private long computeShapeKey(DynamicShapeSlot slot, INDArray[] inputArrays) {
        long hash = slot.getOpNameHash();
        boolean includeValues = slot.isOutputShapeDependsOnInputValues();
        for (INDArray in : inputArrays) {
            if (in != null) {
                for (long dim : in.shape()) {
                    hash ^= dim;
                    hash *= 0x517CC1B727220A95L;
                }
                hash ^= in.dataType().ordinal();
                hash *= 0x9E3779B97F4A7C15L;

                if (includeValues
                        && (in.dataType() == DataType.INT || in.dataType() == DataType.LONG)
                        && in.length() > 0 && in.length() <= 32
                        && in.data() != null && !in.data().wasClosed()) {
                    for (long j = 0; j < in.length(); j++) {
                        hash ^= in.getLong(j);
                        hash *= 0x517CC1B727220A95L;
                    }
                }
            }
        }
        long[] iArgs = slot.getIArgs();
        if (iArgs != null) {
            for (long arg : iArgs) {
                hash ^= arg;
                hash *= 0x9E3779B97F4A7C15L;
            }
        }
        return hash;
    }

    // Minimum padding elements added to every DSP output allocation.
    // Some C++ ops overrun their output buffers by a few bytes/elements, corrupting
    // adjacent glibc malloc chunk headers. This padding absorbs the overrun.
    // 256 elements * 4 bytes (FLOAT32) = 1KB safety margin per buffer.
    // For 2000 ops, total overhead is ~2MB — negligible vs the ~9GB of intermediates.
    private static final long OUTPUT_PADDING_ELEMENTS = 256;

    private INDArray allocateWithHeadroom(DataType dataType, long[] shape) {
        long requiredElements = numElements(shape);
        if (requiredElements <= 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }

        // Scalars (shape=[]) and small arrays: allocate exactly, no padding needed.
        // The overruns happen in large intermediate ops, not scalar reductions.
        if (shape.length == 0 || requiredElements <= OUTPUT_PADDING_ELEMENTS) {
            return mmgr.allocate(true, dataType, shape);
        }

        // Over-allocate to protect against C++ op output buffer overruns.
        // The padding absorbs any small overruns that would otherwise corrupt
        // adjacent malloc metadata, causing "double free or corruption" SIGABRT.
        double gf = ArrayCacheMemoryMgr.getGrowthFactor().get();
        long growthElements = gf > 1.0 ? (long) (requiredElements * gf) : requiredElements;
        long allocElements = Math.max(growthElements, requiredElements + OUTPUT_PADDING_ELEMENTS);

        INDArray oversized = mmgr.allocate(true, dataType, allocElements);
        reshapeBuffer(oversized, shape);
        return oversized;
    }

    /** Growth factor for slot cache refills. Arrays that grow each step (attention scores,
     *  KV cache intermediates) need aggressive headroom so the cached array lasts many steps.
     *  2.0 means we allocate 2x the required elements — at seq_len=128, this covers up to
     *  seq_len=256 before needing reallocation. */
    private static final double SLOT_CACHE_GROWTH_FACTOR = Double.parseDouble(
            System.getProperty("org.nd4j.dsp.slotCacheGrowthFactor", "2.0"));

    private INDArray allocateForSlotCache(DataType dataType, long[] shape) {
        long requiredElements = numElements(shape);
        if (requiredElements <= 0) {
            return Nd4j.emptyWithShape(shape, dataType);
        }
        if (shape.length == 0 || requiredElements <= OUTPUT_PADDING_ELEMENTS) {
            return mmgr.allocate(true, dataType, shape);
        }
        long allocElements = Math.max(
                (long) (requiredElements * SLOT_CACHE_GROWTH_FACTOR),
                requiredElements + OUTPUT_PADDING_ELEMENTS);
        INDArray oversized = mmgr.allocate(true, dataType, allocElements);
        reshapeBuffer(oversized, shape);
        return oversized;
    }

    /**
     * Fast buffer zeroing using direct memset instead of the full assign(0) op dispatch path.
     *
     * Always fetches a FRESH stream from the current device's LaunchContext. The execStream
     * pointer cached earlier in executeSlot can become stale: C++ ContextBuffers::release()
     * frees the underlying cudaStream_t* when device context changes occur during intermediate
     * JNI calls (shape computation, sync operations). Dereferencing a freed stream pointer
     * causes SIGSEGV in the CUDA driver (si_addr near 0x0, garbage stream handle).
     *
     * Falls back to synchronous memset if async fails or no valid stream is available.
     */
    /**
     * Re-fetch a fresh execution stream pointer from the current device's launch context.
     * The cached execStream from execute() may be stale — C++ ContextBuffers::release()
     * can free the underlying cudaStream_t during device context switches in executeSlot().
     */
    private static Pointer getFreshExecStream(NativeOps nativeOps) {
        try {
            OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
            if (lc != null) {
                Pointer stream = nativeOps.lcExecutionStream(lc);
                if (stream != null && stream.address() != 0) {
                    return stream;
                }
            }
        } catch (Exception e) {
            // CPU backend or unavailable
        }
        return null;
    }

    private static void fastZero(INDArray arr, NativeOps nativeOps, Pointer execStream) {
        DataBuffer buf = arr.data();
        if (buf == null || buf.wasClosed()) return;

        OpaqueDataBuffer opaque = buf.opaqueBuffer();
        long bytes = buf.length() * buf.getElementSize();

        Pointer specialPtr = nativeOps.dbSpecialBuffer(opaque);
        if (specialPtr != null && specialPtr.address() != 0) {
            // Fetch a fresh stream pointer from the current device's launch context.
            // This avoids using a stale cached pointer that may have been freed by
            // C++ ContextBuffers::release() during intervening device context changes.
            Pointer freshStream = null;
            try {
                OpaqueLaunchContext lc = nativeOps.defaultLaunchContext();
                if (lc != null) {
                    freshStream = nativeOps.lcExecutionStream(lc);
                }
            } catch (Exception e) {
                // Fall through to sync path
            }

            if (freshStream != null && freshStream.address() != 0) {
                nativeOps.memsetAsync(specialPtr, 0, bytes, 0, freshStream);
                if (nativeOps.lastErrorCode() != 0) {
                    nativeOps.clearLastError();
                    nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
                }
            } else {
                nativeOps.memsetSync(specialPtr, 0, bytes, 0, null);
            }
            nativeOps.dbTickDeviceWrite(opaque);
        } else {
            Pointer primaryPtr = nativeOps.dbPrimaryBuffer(opaque);
            if (primaryPtr != null && primaryPtr.address() != 0) {
                nativeOps.memsetSync(primaryPtr, 0, bytes, 0, null);
            }
        }
    }

    private static long numElements(long[] shape) {
        if (shape == null || shape.length == 0) {
            return 1;
        }
        long prod = 1;
        for (long d : shape) {
            prod *= d;
        }
        return prod;
    }

    private static boolean reshapeBuffer(INDArray arr, long[] shape) {
        if (arr == null || shape == null || shape.length == 0) {
            return false;
        }
        if (Arrays.equals(arr.shape(), shape)) {
            return false;
        }
        long[] newStrides = Nd4j.getStrides(shape, arr.ordering());
        int[] intShape = ArrayUtil.toInts(shape);
        int[] intStrides = ArrayUtil.toInts(newStrides);
        ((BaseNDArray) arr).setShapeAndStride(intShape, intStrides);
        ((BaseNDArray) arr).assignNewId();
        return true;
    }

    /**
     * Local buffer pool for intra-execution array reuse.
     */
    private static final class LocalBufferPool {
        private final Map<DataType, TreeMap<Long, ArrayDeque<INDArray>>> pools = new EnumMap<>(DataType.class);
        private final double largerArrayMaxMultiple;
        private final Set<INDArray> pooledRefs = Collections.newSetFromMap(new IdentityHashMap<>());
        private boolean lastAcquireReshaped;
        private int releaseAccepted;
        private int releaseRejected;
        private long currentPoolBytes;
        private static final long MAX_POOL_BYTES = Long.parseLong(
                System.getProperty("org.nd4j.dsp.pool.maxBytes",
                        String.valueOf(2L * 1024 * 1024 * 1024)));

        private LocalBufferPool() {
            this.largerArrayMaxMultiple = ArrayCacheMemoryMgr.getLargerArrayMaxMultiple().get();
        }

        INDArray acquire(DataType dataType, long[] shape) {
            if (shape == null || shape.length == 0) return null;
            long requiredElements = numElements(shape);
            if (requiredElements <= 0) return null;

            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.get(dataType);
            if (tree == null || tree.isEmpty()) return null;

            long maxElements = (long) (requiredElements * largerArrayMaxMultiple);
            Map.Entry<Long, ArrayDeque<INDArray>> entry = tree.ceilingEntry(requiredElements);
            while (entry != null) {
                long bufferElements = entry.getKey();
                if (bufferElements > maxElements) break;
                ArrayDeque<INDArray> deque = entry.getValue();
                while (deque != null && !deque.isEmpty()) {
                    INDArray arr = deque.poll();
                    if (arr == null) continue;
                    if (deque.isEmpty()) tree.remove(bufferElements);
                    DataBuffer buf = arr.data();
                    if (arr.wasClosed() || buf == null || buf.wasClosed() || !buf.closeable()) continue;
                    if (arr.dataType() != dataType) continue;

                    pooledRefs.remove(arr);
                    currentPoolBytes -= bufferElements * dataType.width();
                    lastAcquireReshaped = reshapeBuffer(arr, shape);
                    // Clear stale OpaqueNDArray — see slot cache comment above
                    arr.clearOpaqueNDArray();
                    return arr;
                }
                entry = tree.higherEntry(bufferElements);
            }
            return null;
        }

        boolean release(INDArray arr) {
            if (arr == null || arr.wasClosed()) return false;
            DataBuffer buf = arr.data();
            if (buf == null || buf.wasClosed() || !buf.closeable()) {
                releaseRejected++;
                return false;
            }
            long thisBytes = buf.length() * arr.dataType().width();
            if (currentPoolBytes + thisBytes > MAX_POOL_BYTES) {
                releaseRejected++;
                return false;
            }
            if (!pooledRefs.add(arr)) return false;
            DataType dt = arr.dataType();
            long bufferElements = buf.length();
            TreeMap<Long, ArrayDeque<INDArray>> tree = pools.computeIfAbsent(dt, k -> new TreeMap<>());
            tree.computeIfAbsent(bufferElements, k -> new ArrayDeque<>()).add(arr);
            currentPoolBytes += thisBytes;
            releaseAccepted++;
            return true;
        }

        void flushTo(SessionMemMgr mmgr) {
            try {
                Nd4j.getExecutioner().commit();
            } catch (Exception ignored) {}

            int pooledCount = 0;
            long pooledBytes = 0;
            int closedCount = 0;
            for (TreeMap<Long, ArrayDeque<INDArray>> tree : pools.values()) {
                for (ArrayDeque<INDArray> deque : tree.values()) {
                    for (INDArray arr : deque) {
                        if (arr == null || arr.wasClosed()) continue;
                        DataBuffer buf = arr.data();
                        if (buf == null || buf.wasClosed()) continue;
                        pooledBytes += buf.length() * arr.dataType().width();
                        pooledCount++;
                        if (buf.closeable() && !arr.isView()) {
                            try {
                                buf.close();
                                closedCount++;
                            } catch (Exception ignored) {}
                        }
                    }
                }
            }
            pools.clear();
            pooledRefs.clear();
            currentPoolBytes = 0;
            log.info("  LocalBufferPool flushTo: {} buffers ({}MB), closed={}, pooled={}, rejected={}",
                    pooledCount, pooledBytes / (1024 * 1024), closedCount,
                    releaseAccepted, releaseRejected);
            releaseAccepted = 0;
            releaseRejected = 0;
        }
    }

    private void printTimingSummary(int opCount, LocalBufferPool localPool) {
        double totalMs = (timingWireInputsNs + timingSyncNs + timingShapeNs +
                timingAllocNs + timingExecNs + timingReleaseNs) / 1_000_000.0;
        log.info("=== DynamicShapePlanExecutor Timing ({} ops, {}ms total) ===",
                opCount, String.format("%.1f", totalMs));
        log.info("  Wire inputs:  {}ms", String.format("%.2f", timingWireInputsNs / 1_000_000.0));
        log.info("  INT/LONG sync: {}ms", String.format("%.2f", timingSyncNs / 1_000_000.0));
        log.info("  Shape calc:   {}ms (hits={}, misses={})",
                String.format("%.2f", timingShapeNs / 1_000_000.0), timingShapeHits, timingShapeMisses);
        log.info("  Mem alloc:    {}ms (slot cache hits={}, misses={}, zero skipped={}, zero applied={})",
                String.format("%.2f", timingAllocNs / 1_000_000.0),
                timingPoolHits, timingPoolMisses, timingZeroSkipped, timingZeroApplied);
        if (!timingCacheMissReasons.isEmpty()) {
            log.info("  Cache miss reasons: {}", timingCacheMissReasons);
        }
        if (timingCacheLeakedConstant > 0) {
            log.info("  Cache LEAKED (constant/non-closeable, not freed): {} arrays, {}MB",
                    timingCacheLeakedConstant, timingCacheLeakedConstantBytes / (1024 * 1024));
        }
        log.info("  Native exec:  {}ms", String.format("%.2f", timingExecNs / 1_000_000.0));
        int viewProducerCount = 0;
        if (slotIsViewProducer != null) {
            for (boolean b : slotIsViewProducer) if (b) viewProducerCount++;
        }
        log.info("  Pending close: {} buffers ({}MB), viewProducerSlots={}",
                pendingCloseCount, pendingCloseBytes / (1024 * 1024), viewProducerCount);
        if (replicaCount > 0) {
            log.info("  Cross-device replicas: {} arrays, {}MB (toDev0: {} arrays {}MB, toDev1: {} arrays {}MB)",
                    replicaCount, replicaBytes / (1024 * 1024),
                    replicaToDev0Count, replicaToDev0Bytes / (1024 * 1024),
                    replicaToDev1Count, replicaToDev1Bytes / (1024 * 1024));
        }
        if (wrongDeviceCacheEjections > 0) {
            log.info("  Wrong-device cache ejections: {}", wrongDeviceCacheEjections);
        }
        // GPU memory pool stats (per-device)
        try {
            NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            LongPointer usedPtr = new LongPointer(1);
            LongPointer reservedPtr = new LongPointer(1);
            nativeOps.getMemoryPoolStats(0, usedPtr, reservedPtr);
            long usedMB = usedPtr.get() / (1024 * 1024);
            long reservedMB = reservedPtr.get() / (1024 * 1024);
            int numDevices = nativeOps.getAvailableDevices();
            if (numDevices > 1) {
                LongPointer usedPtr1 = new LongPointer(1);
                LongPointer reservedPtr1 = new LongPointer(1);
                nativeOps.getMemoryPoolStats(1, usedPtr1, reservedPtr1);
                long usedMB1 = usedPtr1.get() / (1024 * 1024);
                long reservedMB1 = reservedPtr1.get() / (1024 * 1024);
                log.info("  GPU memory pool: dev0 used={}MB reserved={}MB, dev1 used={}MB reserved={}MB",
                        usedMB, reservedMB, usedMB1, reservedMB1);
            } else {
                log.info("  GPU memory pool: used={}MB, reserved={}MB", usedMB, reservedMB);
            }
        } catch (Exception e) {
            // Not available on CPU backend
        }
    }

    /** Close all arrays in the slot cache and null out the cache. */
    private void closeSlotArrayCache() {
        if (slotArrayCache == null) return;
        for (int i = 0; i < slotArrayCache.length; i++) {
            INDArray arr = slotArrayCache[i];
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed() && buf.closeable() && !buf.isConstant()) {
                    try { buf.close(); } catch (Exception ignored) {}
                }
            }
            slotArrayCache[i] = null;
        }
    }

    @Override
    public void close() {
        if (localPool != null) {
            localPool.flushTo(mmgr);
            localPool = null;
        }
        closeSlotArrayCache();
        for (OpContext ctx : ctxPool) {
            try { ctx.close(); } catch (Exception ignored) {}
        }
        ctxPool.clear();

        if (outputSlots != null) {
            for (int i = 0; i < outputSlots.length; i++) {
                INDArray arr = outputSlots[i];
                if (arr != null && !arr.wasClosed()) {
                    try {
                        arr.setCloseable(true);
                        arr.close();
                    } catch (Exception ignored) {}
                }
                outputSlots[i] = null;
            }
        }
        if (externalInputs != null) {
            Arrays.fill(externalInputs, null);
        }
        currentPlan = null;
    }
}
