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

package org.nd4j.linalg.jcublas.ops.executioner;


import lombok.Getter;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.LongIndexer;
import org.nd4j.common.base.Preconditions;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.allocator.pointers.CudaPointer;
import org.nd4j.jita.allocator.tad.DeviceTADManager;
import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.device.DeviceDescriptor;
import org.nd4j.linalg.api.device.DeviceMemoryManager;
import org.nd4j.linalg.api.device.MultiGpuTracer;
import org.nd4j.linalg.api.environment.Nd4jEnvironment;
import org.nd4j.linalg.api.memory.pointers.PagedPointer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ndarray.INDArrayStatistics;
import org.nd4j.linalg.api.ops.*;
import org.nd4j.linalg.api.ops.executioner.DefaultOpExecutioner;
import org.nd4j.linalg.api.ops.executioner.OpStatus;
import org.nd4j.linalg.api.ops.impl.scatter.ScatterUpdate;
import org.nd4j.linalg.api.ops.impl.summarystats.Variance;
import org.nd4j.linalg.api.ops.performance.PerformanceTracker;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.api.shape.LongShapeDescriptor;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.api.shape.TadPack;
import org.nd4j.linalg.api.shape.options.ArrayOptionsHelper;
import org.nd4j.linalg.api.shape.options.ArrayType;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.exception.ND4JOpExceptionUtils;
import org.nd4j.linalg.exception.ND4JOpProfilerException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.OpaqueDataBuffer;
import org.nd4j.linalg.jcublas.bindings.Nd4jCuda;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.buffer.BaseCudaDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaLongDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaUtf8Buffer;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.common.primitives.AtomicBoolean;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.nativeblas.*;

import java.util.*;

import static org.bytedeco.cuda.global.cudart.*;
import org.nd4j.linalg.jcublas.JCublasNDArray;

/**
 * JCuda executioner.
 * <p/>
 * Runs ops directly on the gpu
 *
 * If requested Op doesn't exist within GPU context, DefaultOpExecutioner will be used, with arrays/buffers updated after that.
 *
 * @author Adam Gibson
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaExecutioner extends DefaultOpExecutioner {


    @Getter
    protected static TADManager tadManager = new DeviceTADManager();
    protected ThreadLocal<PointerPointer> extraz = new ThreadLocal<>();
    protected volatile transient Properties properties;

    protected ThreadLocal<String> lastOp = new ThreadLocal<>();

    /** Per-thread flag to skip device coherency in exec() calls.
     * Set by DSP parallel workers that handle device placement themselves. */
    private static final ThreadLocal<Boolean> skipDeviceCoherency = ThreadLocal.withInitial(() -> false);

    @Override
    public void setSkipDeviceCoherency(boolean skip) {
        skipDeviceCoherency.set(skip);
    }

    protected Map<String, CustomOpDescriptor> customOps = null;

    protected AtomicBoolean experimentalMode = new AtomicBoolean(false);

    /** Cached device count — queried once, never changes during JVM lifetime. */
    private volatile int cachedDeviceCount = -1;

    /** Diagnostic counter: how many cross-device replications happened since last reset. */
    private static final ThreadLocal<long[]> replicationCounter = ThreadLocal.withInitial(() -> new long[]{0, 0}); // [count, bytes]

    /**
     * Per-op replicated input buffers created by ensureDeviceCoherency.
     * These are temporary copies of cross-device inputs that must be freed after
     * the op completes. Without tracking, each op that uses a cross-device input
     * leaks a replica buffer permanently (GC cleanup is broken for GPU buffers).
     *
     * Cleared after each exec() call in closeReplicatedBuffers().
     */
    private static final ThreadLocal<List<DataBuffer>> pendingReplicaClose =
            ThreadLocal.withInitial(ArrayList::new);

    /**
     * Cache of constant input replicas across ops within a single forward pass.
     * Key: identity hash code of the source DataBuffer + target device ID.
     * Value: replicated INDArray on the target device.
     *
     * Without this cache, every op that uses a cross-device constant creates a
     * new replica (~258MB total for a 1962-op forward pass with 500+ constant-using ops).
     * With the cache, each unique constant is replicated once per forward pass.
     *
     * Must be cleared between forward passes via clearConstantReplicaCache() to allow
     * the underlying buffers to be freed. InferenceSession calls this at cleanup time.
     */
    private static final ThreadLocal<Map<Long, INDArray>> constantReplicaCache =
            ThreadLocal.withInitial(HashMap::new);

    /**
     * Tracks source DataBuffer identity for each constant replica cache entry.
     * Key: same cache key as constantReplicaCache.
     * Value: System.identityHashCode of the SOURCE DataBuffer.
     * Used by clearConstantReplicaCache() to verify we only close genuine replicas
     * (different DataBuffer objects), never the original model constants.
     */
    private static final ThreadLocal<Map<Long, Integer>> constantReplicaSourceIds =
            ThreadLocal.withInitial(HashMap::new);

    /**
     * Close all replicated input buffers from the current op's ensureDeviceCoherency calls.
     * Called after each exec() to prevent per-op replica leaks.
     *
     * IMPORTANT: Skip when skipDeviceCoherency is true (inside ensureDeviceCoherency).
     * During migration, replicateToDevice may trigger inner exec() calls (e.g., dup() for
     * views). Those inner exec() calls have their own finally{closeReplicatedBuffers()},
     * which would prematurely close replicas added by the outer ensureDeviceCoherency loop.
     * The outer exec()'s finally block runs with skipDeviceCoherency=false (restored by
     * ensureDeviceCoherency's own finally), so it properly cleans up all replicas.
     */
    private void closeReplicatedBuffers() {
        if (skipDeviceCoherency.get()) return;
        List<DataBuffer> pending = pendingReplicaClose.get();
        if (pending.isEmpty()) return;
        // Synchronize current device stream before freeing replicas.
        // The op kernel was dispatched on a CUDA stream and may still be reading
        // from these replica buffers. Without this sync, cudaFreeAsync returns the
        // memory while the kernel is still in-flight → use-after-free → SIGSEGV.
        // commit() synchronizes all pending operations on the current thread's stream.
        Nd4j.getExecutioner().commit();
        for (DataBuffer buf : pending) {
            if (buf != null && !buf.wasClosed()) {
                try {
                    // Replicas may have acquired constant flag via directExecHelper's
                    // setCloseable(false) → setConstant(true). Force-unset before closing.
                    if (buf.isConstant()) {
                        buf.setConstant(false);
                    }
                    buf.close();
                } catch (Exception e) {
                    // Non-fatal: buffer may have been closed through an alias
                }
            }
        }
        pending.clear();
    }

    /**
     * Clear the per-thread constant replica cache. Call this between forward passes
     * (e.g., from InferenceSession cleanup) to allow replica buffers to be freed.
     * The cache entries themselves hold INDArray references that prevent GC.
     */
    @Override
    public void clearConstantReplicaCache() {
        Map<Long, INDArray> cache = constantReplicaCache.get();
        Map<Long, Integer> sourceIds = constantReplicaSourceIds.get();
        if (cache.isEmpty()) {
            sourceIds.clear();
            return;
        }
        int closedCount = 0;
        long closedBytes = 0;
        int skippedSameObject = 0;
        for (Map.Entry<Long, INDArray> entry : cache.entrySet()) {
            INDArray arr = entry.getValue();
            if (arr != null && !arr.wasClosed()) {
                DataBuffer buf = arr.data();
                if (buf != null && !buf.wasClosed()) {
                    // Safety: verify the cached buffer is a genuine replica (different
                    // object identity from the source). If somehow the original got cached,
                    // do NOT close it — that would destroy the model constant.
                    Integer srcId = sourceIds.get(entry.getKey());
                    if (srcId != null && System.identityHashCode(buf) == srcId) {
                        skippedSameObject++;
                        continue;
                    }
                    try {
                        // Replicas of constants may acquire the constant flag via
                        // directExecHelper's setCloseable(false) → setConstant(true).
                        // Force-unset it so close() works.
                        if (buf.isConstant()) {
                            buf.setConstant(false);
                        }
                        closedBytes += buf.length() * buf.getElementSize();
                        buf.close();
                        closedCount++;
                    } catch (Exception e) {
                        // Non-fatal
                    }
                }
            }
        }
        if (closedCount > 0 || skippedSameObject > 0) {
            log.info("Cleared constant replica cache: closed {} buffers ({}MB), skipped {} same-object",
                    closedCount, closedBytes / (1024 * 1024), skippedSameObject);
        }
        cache.clear();
        sourceIds.clear();
    }

    @Override
    public long[] resetReplicationCounter() {
        long[] prev = replicationCounter.get();
        long[] result = new long[]{prev[0], prev[1]};
        prev[0] = 0;
        prev[1] = 0;
        return result;
    }

    public CudaExecutioner() {
        experimentalMode.set(Nd4j.getNativeOps().isExperimentalEnabled());
    }

    /**
     * Replicate an input array to the target device, using the constant replica cache
     * for constant inputs and tracking non-constant replicas for cleanup after op execution.
     *
     * @param input           the input array to replicate
     * @param targetDeviceId  the target device
     * @param constCache      per-thread constant replica cache
     * @param replicaPending  per-op list of non-constant replica buffers to close after exec
     * @return the replicated array on the target device
     */
    private INDArray replicateWithCache(INDArray input, int targetDeviceId,
                                         Map<Long, INDArray> constCache,
                                         List<DataBuffer> replicaPending) {
        boolean isConstant = input.data() != null && input.data().isConstant();
        long cacheKey = isConstant ?
                ((long) System.identityHashCode(input.data()) << 32) | targetDeviceId : 0;

        INDArray migrated = null;
        if (isConstant) {
            migrated = constCache.get(cacheKey);
            if (migrated != null && migrated.wasClosed()) {
                constCache.remove(cacheKey);
                constantReplicaSourceIds.get().remove(cacheKey);
                migrated = null;
            }
        }

        if (migrated == null) {
            migrated = Nd4j.getAffinityManager().replicateToDevice(targetDeviceId, input);
            if (isConstant) {
                constCache.put(cacheKey, migrated);
                // Track source identity so clearConstantReplicaCache() can verify
                // the cached entry is a genuine replica, not the original.
                constantReplicaSourceIds.get().put(cacheKey, System.identityHashCode(input.data()));
            } else {
                DataBuffer replicaBuf = migrated.data();
                if (replicaBuf != null) {
                    replicaPending.add(replicaBuf);
                }
            }
        }
        return migrated;
    }

    /**
     * Minimum free memory (bytes) a device must have beyond migration needs to be
     * selected as the target. Prevents choosing a device that is too full for
     * computation workspace (cuDNN, intermediates, output allocations).
     *
     * Set to 1 GB (was 128 MB). Under memory pressure, cudaMemGetInfo reports free
     * memory MINUS the async memory pool's reserved set. The pool can reuse freed
     * entries via cudaMallocAsync even when cudaMemGetInfo reports near-zero free.
     * A 128 MB threshold caused premature cross-device routing: device 0 had 12.7 GB
     * pool_used on a 24 GB GPU, cudaMemGetInfo showed ~25 MB free, but the pool had
     * ~11 GB of reusable entries. Routing to device 1 triggered DIAG-MIGRATE and
     * subsequent CUDA context corruption.
     */
    private static final long MIN_FREE_MEMORY_FOR_TARGET = 1024L * 1024 * 1024;

    /**
     * Select the optimal device for op execution, minimizing cross-device data copies.
     *
     * Strategy (in priority order):
     * 1. Output device: if outputs exist and are already allocated on a device, execute there.
     *    Outputs cannot be safely migrated (DSP tracks output buffer addresses for lifecycle
     *    management — replacing them orphans the originals).
     * 2. Input majority by data size: if no outputs have a device, pick the device where the
     *    most input data (by bytes) already resides — BUT only if that device has enough free
     *    memory for the migration of remaining inputs plus computation workspace. If the
     *    data-locality winner is too full, fall through to capacity-based selection.
     * 3. Capacity tiebreaker: pick the device with the most free memory.
     *
     * @param inputs  input arrays from OpContext (may be null)
     * @param outputs output arrays from OpContext (may be null)
     * @return target device ID (>= 0), or -1 if single-GPU or no arrays
     */
    private int selectTargetDevice(List<INDArray> inputs, List<INDArray> outputs) {
        int numDevices = getDeviceCount();
        if (numDevices <= 1) return 0;

        // When cross-device routing is suppressed (InferenceSession/DSP execution/decode loop),
        // always stay on the current thread's device. The CUDA async memory pool can reuse
        // freed entries via cudaMallocAsync even when cudaMemGetInfo reports near-zero free
        // (pool-reserved memory is reported as "used" but is instantly reclaimable by trim).
        // Routing away from the model's home device during DSP execution invalidates baked
        // graph pointers and causes migrate() H2D failures on non-peer devices.
        if (OpaqueDataBuffer.isCrossDeviceRoutingSuppressed()) {
            return Nd4j.getAffinityManager().getDeviceForCurrentThread();
        }

        // Delegate to DeviceMemoryManager which handles memory pressure,
        // device priorities, simulation mode, and fallback routing centrally.
        // Use input data locality to pick a preferred device.
        DeviceMemoryManager mgr = DeviceMemoryManager.getInstance();

        // Find which device holds the most input data (data-locality preference)
        int preferredDevice = -1;
        if (inputs != null && !inputs.isEmpty()) {
            long[] inputDeviceBytes = new long[numDevices];
            for (INDArray input : inputs) {
                if (input != null && input.data() != null) {
                    int devId = AtomicAllocator.getInstance().getDeviceId(input);
                    if (devId >= 0 && devId < numDevices) {
                        inputDeviceBytes[devId] += input.length() * input.data().getElementSize();
                    }
                }
            }
            long bestBytes = 0;
            for (int d = 0; d < numDevices; d++) {
                if (inputDeviceBytes[d] > bestBytes) {
                    bestBytes = inputDeviceBytes[d];
                    preferredDevice = d;
                }
            }
        }

        // Delegate to DeviceMemoryManager which handles memory pressure,
        // device priorities, and fallback routing centrally.
        long workspaceBytes = MIN_FREE_MEMORY_FOR_TARGET;

        DeviceDescriptor preferredDesc = null;
        if (preferredDevice >= 0) {
            String prefId = String.valueOf(preferredDevice);
            for (DeviceDescriptor d : mgr.getRegisteredDevices()) {
                if (prefId.equals(d.getDeviceId())) {
                    preferredDesc = d;
                    break;
                }
            }
        }

        DeviceDescriptor selected;
        if (preferredDesc != null) {
            selected = mgr.selectDeviceForAllocation(workspaceBytes, preferredDesc);
        } else {
            selected = mgr.selectDeviceForAllocation(workspaceBytes);
        }
        if (selected != null) {
            try {
                return Integer.parseInt(selected.getDeviceId());
            } catch (NumberFormatException e) {
                return 0;
            }
        }
        return 0;
    }

    /** Get the number of CUDA devices (cached after first query). */
    private int getDeviceCount() {
        if (cachedDeviceCount < 0) {
            cachedDeviceCount = Nd4j.getNativeOps().getAvailableDevices();
        }
        return cachedDeviceCount;
    }

    /**
     * Ensure all OpContext arrays are coherent on a single device.
     * Sets the CUDA context to the target device and migrates any inputs
     * that are not on the target device.
     *
     * @return the target device ID, or -1 if no device change was needed
     */
    private int ensureDeviceCoherency(List<INDArray> inputs, List<INDArray> outputs, OpContext oc) {
        int numDevices = getDeviceCount();
        if (numDevices <= 1) return 0;

        int targetDeviceId = selectTargetDevice(inputs, outputs);
        if (targetDeviceId < 0) return -1;

        int previousDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();

        // Fast path: if already on the target device, check if any inputs need migration
        if (previousDevice == targetDeviceId) {
            boolean needsMigration = false;
            if (inputs != null) {
                for (INDArray input : inputs) {
                    if (input != null && input.data() != null) {
                        int inputDeviceId = AtomicAllocator.getInstance().getDeviceId(input);
                        if (inputDeviceId >= 0 && inputDeviceId != targetDeviceId) {
                            needsMigration = true;
                            break;
                        }
                    }
                }
            }
            if (!needsMigration) return targetDeviceId;
        }

        // Set CUDA context to the target device
        DeviceMemoryManager.getInstance().switchDevice(targetDeviceId, "CudaExecutioner.ensureDeviceCoherency", "coherency");
        if (previousDevice != targetDeviceId) {
            MultiGpuTracer.traceDeviceSwitch("CudaExecutioner.ensureDeviceCoherency",
                    previousDevice, targetDeviceId, "target-device-selected");
        }

        // Migrate any inputs/outputs not on the target device.
        // Set skipDeviceCoherency to prevent infinite recursion:
        // replicateToDevice -> dup -> assign -> exec -> ensureDeviceCoherency
        skipDeviceCoherency.set(true);
        try {
            if (inputs != null) {
                Map<Long, INDArray> constCache = constantReplicaCache.get();
                List<DataBuffer> replicaPending = pendingReplicaClose.get();
                for (int i = 0; i < inputs.size(); i++) {
                    INDArray input = inputs.get(i);
                    if (input != null && input.data() != null) {
                        int inputDeviceId = AtomicAllocator.getInstance().getDeviceId(input);
                        if (inputDeviceId >= 0 && inputDeviceId != targetDeviceId) {
                            log.warn("DIAG-MIGRATE: input[{}] shape={} dtype={} const={} devId={} → target={} id={} identityHash={}",
                                    i, java.util.Arrays.toString(input.shape()), input.dataType(),
                                    input.data().isConstant(), inputDeviceId, targetDeviceId,
                                    input.getId(), System.identityHashCode(input));
                            MultiGpuTracer.traceDataTransfer(inputDeviceId, targetDeviceId,
                                    input.shape(), input.dataType(),
                                    input.length() * input.data().getElementSize(),
                                    input.isView(),
                                    input.data().isConstant());
                            INDArray migrated = replicateWithCache(input, targetDeviceId, constCache, replicaPending);
                            log.warn("DIAG-MIGRATE: migrated id={} closed={} identityHash={}",
                                    migrated.getId(), migrated.wasClosed(), System.identityHashCode(migrated));
                            oc.setInputArray(i, migrated);
                            long[] rc = replicationCounter.get();
                            rc[0]++;
                            rc[1] += input.length() * input.data().getElementSize();
                        }
                    }
                }
            }
            // Also migrate output arrays — native ops write results to the output buffer,
            // which must be on the same device as the execution context (streams, workspace).
            // Without this, cross-device writes cause "invalid resource handle" errors.
            if (outputs != null) {
                for (int i = 0; i < outputs.size(); i++) {
                    INDArray output = outputs.get(i);
                    if (output != null && output.data() != null) {
                        int outputDeviceId = AtomicAllocator.getInstance().getDeviceId(output);
                        if (outputDeviceId >= 0 && outputDeviceId != targetDeviceId) {
                            // For outputs, create a fresh buffer on the target device (no data to copy).
                            // The old output buffer belongs to the caller — do NOT close it here.
                            INDArray migrated = Nd4j.createUninitialized(output.dataType(), output.shape(), output.ordering());
                            oc.setOutputArray(i, migrated);
                        }
                    }
                }
            }
        } finally {
            skipDeviceCoherency.set(false);
        }

        return targetDeviceId;
    }

    /**
     * Device coherency for legacy Op interface (no OpContext).
     * Called from exec(Op, null) for ops like assign/dup that don't use OpContext.
     * Ensures all op arrays (x, y, z) are on the same device before kernel launch.
     */
    private void ensureDeviceCoherencyForOp(Op op) {
        int numDevices = getDeviceCount();
        if (numDevices <= 1) return;

        INDArray x = op.x();
        INDArray y = op.y();
        INDArray z = op.z();

        // Build input/output lists for selectTargetDevice
        List<INDArray> inputs = new ArrayList<>(2);
        if (x != null) inputs.add(x);
        if (y != null) inputs.add(y);
        List<INDArray> outputs = (z != null) ? List.of(z) : null;

        int targetDeviceId = selectTargetDevice(inputs, outputs);
        if (targetDeviceId < 0) return;

        int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
        if (targetDeviceId != currentDevice) {
            DeviceMemoryManager.getInstance().switchDevice(targetDeviceId,
                    "CudaExecutioner.ensureDeviceCoherencyForOp", "op-coherency");
        }

        // Migrate inputs not on the target device.
        // Set skipDeviceCoherency to prevent infinite recursion:
        // replicateToDevice -> dup -> assign -> exec -> ensureDeviceCoherencyForOp
        skipDeviceCoherency.set(true);
        try {
            Map<Long, INDArray> constCache = constantReplicaCache.get();
            List<DataBuffer> replicaPending = pendingReplicaClose.get();
            if (x != null && x.data() != null) {
                int xDevice = AtomicAllocator.getInstance().getDeviceId(x);
                if (xDevice >= 0 && xDevice != targetDeviceId) {
                    INDArray migrated = replicateWithCache(x, targetDeviceId, constCache, replicaPending);
                    op.setX(migrated);
                }
            }
            if (y != null && y.data() != null) {
                int yDevice = AtomicAllocator.getInstance().getDeviceId(y);
                if (yDevice >= 0 && yDevice != targetDeviceId) {
                    INDArray migrated = replicateWithCache(y, targetDeviceId, constCache, replicaPending);
                    op.setY(migrated);
                }
            }
        } finally {
            skipDeviceCoherency.set(false);
        }
    }

    @Override
    public String getLastOp() {
        return lastOp.get();
    }

    @Override
    public INDArray exec(BroadcastOp op) {
        // Device coherency: ensure all arrays are on the same device
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherencyForOp(op);
        }

        // Handle in-place broadcast on views: dup to contiguous, execute, assign back.
        // Without this, the CUDA kernel writes to a shared DataBuffer region but only
        // updates the view's portion, leaving the host-side stale for the parent array.
        boolean viewInPlace = false;
        INDArray originalView = null;
        if (op.x() == op.z() && op.x() != null && op.x().isView()) {
            viewInPlace = true;
            originalView = op.x();
            INDArray xDup = op.x().dup(op.x().ordering());
            op.setX(xDup);
            op.setZ(xDup);
        }

        try {
            long st = profilingConfigurableHookIn(op);

            checkForCompression(op);

            // Handle null dimensions or dimensions with null data buffer
            INDArray dimArray = op.dimensions();
            INDArray dim1 = null;
            if (dimArray != null && dimArray.data() != null) {
                dim1 = dimArray.castTo(DataType.LONG);
            }
            val dimension = OpaqueNDArray.fromINDArray(dim1);

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            val context = AtomicAllocator.getInstance().getDeviceContext();

            if (CudaEnvironment.getInstance().getConfiguration().isDebug())
                lastOp.set(op.opName());

            Pointer hostYShapeInfo =
                    op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer()).retainReference();
            Pointer hostZShapeInfo =
                    op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer()).retainReference();

            val x = OpaqueNDArray.fromINDArray(op.x());
            val y = OpaqueNDArray.fromINDArray(op.y());
            val z = OpaqueNDArray.fromINDArray(op.z());

            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(),dim1.toLongVector());

            Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
            Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

            DataBuffer offsets = tadBuffers.getSecond();
            Pointer devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

            Pointer devTadShapeInfoZ = null;
            Pointer devTadOffsetsZ = null;

            // that's the place where we're going to have second TAD in place
            Pair<DataBuffer, DataBuffer> tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dim1.toLongVector());

            devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
            devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);

            PointerPointer xShapeInfoHostPointer = extraz.get().put(
                    AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                    AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                    context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                    hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                    devTadShapeInfoZ, devTadOffsetsZ);
            Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;

            switch (op.getOpType()) {
                case BROADCAST:
                    Nd4j.getNativeOps().execBroadcast(xShapeInfoHostPointer, op.opNum(),
                            x,
                            y,
                            z,
                            extraArgs,
                            dimension);
                    break;
                case BROADCAST_BOOL:
                    Nd4j.getNativeOps().execBroadcastBool(xShapeInfoHostPointer, op.opNum(),
                            x,
                            y,
                            z,
                            extraArgs,
                            dimension);
                    break;
                default:
                    throw new UnsupportedOperationException("Unknown op type: " + op.getOpType());
            }

            if (Nd4j.getNativeOps().lastErrorCode() != 0)
                throw ND4JOpExceptionUtils.opExecutionException(op, Nd4j.getNativeOps().lastErrorMessage());

            profilingConfigurableHookOut(op, null, st);

            // Assign result back to the original view if we duped for in-place broadcast
            if (viewInPlace && originalView != null) {
                originalView.assign(op.z());
                return originalView;
            }

            return op.z();
        } finally {
            closeReplicatedBuffers();
        }
    }

    /**
     *
     * @param op
     * @param dimension
     * @return
     */
    protected INDArray naiveExec(ReduceOp op, long... dimension) {
        long st = profilingConfigurableHookIn(op);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(op.z() != null) {
                Preconditions.checkState(op.x().equalShapes(op.z()), "For empty reductions, result (z) array must have same shape as x shape." +
                        " Got: x=%ndShape, z=%ndShape", op.x(), op.z());
                op.z().assign(op.x());
                return op.z();
            } else {
                op.setZ(op.x().dup());
                return op.z();
            }
        }

        INDArray ret = op.z();

        checkForCompression(op);
        op.validateDataTypes(null);

        // Normalize negative axes first (e.g., -1 → rank-1), then check for "reduce all"
        dimension = Shape.normalizeAxis(op.x().rank(), dimension);
        if (Shape.wholeArrayDimension(dimension)) {
            dimension = new long[0];
        }

        // Validate dimensions are within rank bounds
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= op.x().rank())
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + op.x().rank() + "]");

        // Device coherency: ensure all arrays are on the same device
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherencyForOp(op);
        }

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val hostYShapeInfo = op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        val hostZShapeInfo = op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        // Empty-input guard: if x has a 0-dimension TAD creation will throw.
        // The output z was already allocated by exec(ReduceOp) with the correct shape
        // (e.g. [2,1,3] for keepDims or [2,3] without keepDims) and is zero-initialized.
        if (op.x().isEmpty()) {
            profilingConfigurableHookOut(op, null, st);
            return op.z();
        }

        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(op.x().shapeInfoDataBuffer(), context);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()),
                context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(),
                context.getBufferAllocation(),
                context.getBufferReduction(),
                context.getBufferScalar(),
                context.getBufferSpecial(),
                hostYShapeInfo,
                hostZShapeInfo,
                hostTadShapeInfo,
                devTadShapeInfo,
                devTadOffsets);

        Pointer yDevTadOffsets = null;
        Pointer yDevTadShapeInfo = null;

        if (op.y() != null) {
            // dimension.length == 0 means "reduce all dimensions" (full array reduction)
            if (dimension.length == 0 || op.x().tensorAlongDimension(0, dimension).length() != op.y().length()) {
                if (!op.isComplexAccumulation() && op.x().length() != op.y().length())
                    throw new ND4JIllegalStateException("Op.X [" + op.x().length() + "] and Op.Y [" + op.y().length() + "] lengths should match");

                if (!op.z().isScalar()) {
                    Pair<DataBuffer, DataBuffer> yTadBuffers = tadManager.getTADOnlyShapeInfo(op.y(), dimension);

                    yDevTadShapeInfo = AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);

                    DataBuffer yOffsets = yTadBuffers.getSecond();
                    yDevTadOffsets = yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

                    xShapeInfoHostPointer.put(12, yDevTadShapeInfo);
                    xShapeInfoHostPointer.put(13, yDevTadOffsets);
                } else {
                    // For scalar output, explicitly clear any stale TAD pointers from previous operations
                    xShapeInfoHostPointer.put(12, null);
                    xShapeInfoHostPointer.put(13, null);
                }
            } else {
                // TAD vs full array code branch
                xShapeInfoHostPointer.put(12, AtomicAllocator.getInstance().getPointer(op.y().shapeInfoDataBuffer(), context));
                xShapeInfoHostPointer.put(13, null);
            }
        }

        DataType argsType;
        switch (op.getOpType()) {
            case REDUCE_LONG:
            case REDUCE_BOOL:
                argsType = op.x().dataType();
                break;
            default:
                argsType = op.z().dataType();
        }

        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(argsType), context) : null;

        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());
        INDArray dimArr = Nd4j.createFromArray(dimension);
        val dim = OpaqueNDArray.fromINDArray(dimArr);
        if (op instanceof Variance) {
            if (ret.isScalar()) {
                Nd4j.getNativeOps().execSummaryStatsScalar(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        z,
                        ((Variance) op).isBiasCorrected());
            } else {
                Nd4j.getNativeOps().execSummaryStatsTad(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        z,
                        dim,
                        ((Variance) op).isBiasCorrected());
            }
        } else if (op.y() != null && op.getOpType() == Op.Type.REDUCE3) {
            // Only use Reduce3 for actual two-input reductions (dot, cosine, etc.)
            // NOT for single-input reductions where y contains the axes tensor
            if (ret.isScalar()) {
                Nd4j.getNativeOps().execReduce3Scalar(xShapeInfoHostPointer, op.opNum(),
                        x,
                        extraArgs,
                        y,
                        z);
            } else {
                Nd4j.getNativeOps().execReduce3Tad(xShapeInfoHostPointer, op.opNum(),x
                        ,extraArgs,y,z,dim);
            }
        } else {
            if (ret.isScalar()) {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool(xShapeInfoHostPointer,
                                op.opNum(),
                                x,
                                extraArgs,
                                z,dim);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z,dim);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,
                                z);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            } else {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat2(xShapeInfoHostPointer, op.opNum(),
                                x,
                                extraArgs,z,dim);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool2(xShapeInfoHostPointer, op.opNum(),
                                x,extraArgs,z,dim);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame2(xShapeInfoHostPointer, op.opNum(),
                                x, extraArgs, z,dim);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong2(xShapeInfoHostPointer, op.opNum(),
                                x, extraArgs, z,dim);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        profilingConfigurableHookOut(op, null, st);

        return op.z();
    }

    @Override
    public INDArray exec(Variance op) {
        return exec((ReduceOp) op);
    }

    @Override
    public INDArray exec(ReduceOp op) {
        // Device coherency: ensure all arrays are on the same device
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherencyForOp(op);
        }

        try {
            checkForCompression(op);

            if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
                //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
                //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
                if(op.z() != null) {
                    Preconditions.checkState(op.x().equalShapes(op.z()), "For empty reductions, result (z) array must have same shape as x shape." +
                            " Got: x=%ndShape, z=%ndShape", op.x(), op.z());
                    op.z().assign(op.x());
                    return op.z();
                } else {
                    op.setZ(op.x().dup());
                    return op.z();
                }
            }

            // Match CPU behavior: check for null dimensions before calling toLongVector()
            // op.dimensions() can be null OR return an array with null data buffer
            // In either case, pass null to normalizeAxis which returns empty array meaning "reduce all"
            INDArray dimArray = op.dimensions();
            long[] dimLong = null;
            if (dimArray != null && dimArray.data() != null) {
                dimLong = dimArray.toLongVector();
            }

            // Normalize negative axes first (e.g., -1 → rank-1), then check for "reduce all"
            dimLong = Shape.normalizeAxis(op.x().rank(), dimLong);
            if (Shape.wholeArrayDimension(dimLong)) {
                dimLong = new long[0];
            }

            val dimension = dimLong;

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));


            val wholeDims = Shape.wholeArrayDimension(dimension) || op.x().rank() == dimension.length || dimension.length == 0;
            val retShape = Shape.reductionShape(op.y() == null ? op.x() : op.x().length() > op.y().length() ? op.x() : op.y(), dimension, true, op.isKeepDims());

            if (op.x().isVector() && op.x().length() == ArrayUtil.prod(retShape) && ArrayUtil.prodLong(retShape) > 1 && op.y() == null)
                return op.noOp();

            val dtype = op.resultType();
            INDArray ret = null;
            if (op.z() == null || op.z() == op.x()) {
                if (op.isComplexAccumulation()) {
                    // Complex accumulation requires actual dimensions, not full array reduction
                    if (dimension.length == 0)
                        throw new ND4JIllegalStateException("Complex accumulation requires dimensions to be specified");
                    val xT = op.x().tensorsAlongDimension(dimension);
                    val yT = op.y().tensorsAlongDimension(dimension);

                    // we intentionally want to set it to 0.0
                    ret = Nd4j.createUninitialized(dtype, new long[] {xT, yT});
                } else {
                    if (op.y() != null) {
                        //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
                        if (op.x().length() == op.y().length()) {
                            //Pairwise
                            if (!wholeDims && op.x().tensorsAlongDimension(dimension) != op.y().tensorsAlongDimension(dimension)) {
                                throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                                        Arrays.toString(op.x().shape()) + ", y shape = " + Arrays.toString(op.y().shape()) +
                                        ", dimension = " + Arrays.toString(dimension) + ")");
                            }
                        } else {
                            if (dimension.length == 0)
                                throw new ND4JIllegalStateException("TAD vs TAD comparison requires dimension (or other comparison mode was supposed to be used?)");

                            //Every X TAD vs. entirety of Y
                            val xTADSize = op.x().length() / op.x().tensorsAlongDimension(dimension);

                            if (xTADSize != op.y().length()) {
                                throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                                        " (x TAD size = " + xTADSize + ", y size = " + op.y().length());
                            }
                        }
                    }

                    // in case of regular accumulation we don't care about array state before op
                    ret = Nd4j.create(dtype, retShape);
                }
                op.setZ(ret);
            } else {
                // compare length

                if (op.z().length() != (retShape.length == 0 ? 1 : ArrayUtil.prodLong(retShape)))
                    throw new ND4JIllegalStateException("Shape of target array for reduction [" + Arrays.toString(op.z().shape()) + "] doesn't match expected [" + Arrays.toString(retShape) + "]");
            }

            long st = profilingConfigurableHookIn(op);
            naiveExec(op, dimension);

            profilingConfigurableHookOut(op, null, st);

            return op.z();
        } finally {
            closeReplicatedBuffers();
        }
    }

    @Override
    public INDArray exec(IndexAccumulation op) {
        // Device coherency: ensure all arrays are on the same device
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherencyForOp(op);
        }

        try {
            // Check for null dimensions or null data buffer before calling toLongVector()
            INDArray dimArray = op.dimensions();
            long[] dimLong = null;
            if (dimArray != null && dimArray.data() != null) {
                dimLong = dimArray.toLongVector();
            }
            // Normalize negative axes first, then check for "reduce all"
            dimLong = Shape.normalizeAxis(op.x().rank(), dimLong);
            if (Shape.wholeArrayDimension(dimLong)) {
                dimLong = new long[0];
            }
            val dimension = dimLong;

            if (op.x().isEmpty()) {
                for (val d:dimension) {
                    Preconditions.checkArgument(op.x().size(d) != 0, "IndexReduce can't be issued along axis with 0 in shape");
                }
            }

            if (op.z() == null) {
                val retShape = Shape.reductionShape(op.x(), dimension, true, op.isKeepDims());
                op.setZ(Nd4j.createUninitialized(DataType.LONG, retShape));
            }

            long st = profilingConfigurableHookIn(op);

            checkForCompression(op);


            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            if (op.x().isVector() && op.x().length() == op.z().length()) {
                return op.x();
            }

            if (op.z().isEmpty())
                return op.z();

            if (CudaEnvironment.getInstance().getConfiguration().isDebug())
                lastOp.set(op.opName());

            val context = AtomicAllocator.getInstance().getDeviceContext();

            val hostYShapeInfo =
                    op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
            val hostZShapeInfo =
                    op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());


            Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

            val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
            val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

            val offsets = tadBuffers.getSecond();
            val devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

            PointerPointer xShapeInfoHostPointer = extraz.get().put(
                    AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                    AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                    context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                    hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);
            Pointer extraArgs = op.extraArgs() != null
                    ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.x().dataType()), context) : null;

            val x = OpaqueNDArray.fromINDArray(op.x());
            val y = OpaqueNDArray.fromINDArray(op.y());
            val z = OpaqueNDArray.fromINDArray(op.z());
            // Handle null dimensions or dimensions with null data buffer
            INDArray dimArr = op.dimensions();
            INDArray dim1 = null;
            if (dimArr != null && dimArr.data() != null) {
                dim1 = dimArr.castTo(DataType.LONG);
            } else {
                // Full reduce: pass empty LONG array to native code
                dim1 = Nd4j.empty(DataType.LONG);
            }
            val dimension2 = OpaqueNDArray.fromINDArray(dim1);
            Nd4j.getNativeOps().execIndexReduce(xShapeInfoHostPointer, op.opNum(),
                    x,
                    extraArgs,
                    z,
                    dimension2);

            if (Nd4j.getNativeOps().lastErrorCode() != 0)
                throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

            profilingConfigurableHookOut(op, null, st);

            return op.z();
        } finally {
            closeReplicatedBuffers();
        }
    }


    @Override
    public INDArray exec(Op op) {
        return exec(op, null);
    }

    @Override
    public INDArray exec(Op op, OpContext oc) {
        // Device coherency: ensure all arrays are on the same device
        // and the CUDA context is set to the target device for op execution.
        // Skipped when DSP parallel workers handle device placement themselves.
        if (!skipDeviceCoherency.get()) {
            if (oc != null) {
                ensureDeviceCoherency(oc.getInputArrays(), oc.getOutputArrays(), oc);
            } else {
                ensureDeviceCoherencyForOp(op);
            }
        }

        if (MultiGpuTracer.ENABLED) {
            int currentDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            MultiGpuTracer.verifyDeviceContext(op.opName(), currentDev, currentDev);
        }

        checkForCompression(op);

        try {
            if (op instanceof TransformOp) {
                TransformOp t = (TransformOp) op;
                invoke(t, oc);
            } else if (op instanceof ReduceOp) {
                ReduceOp acc = (ReduceOp) op;
                invoke(acc, oc, acc.dimensionsArr());
            } else if (op instanceof ScalarOp) {
                ScalarOp sc = (ScalarOp) op;
                invoke(sc, oc);
            } else if (op instanceof BroadcastOp) {
                BroadcastOp broadcastOp = (BroadcastOp) op;
                invoke(broadcastOp, oc);
            } else if (op instanceof IndexAccumulation) {
                IndexAccumulation indexAccumulation = (IndexAccumulation) op;
                INDArray idxDims = indexAccumulation.dimensions();
                long[] idxDimLong = (idxDims != null && idxDims.data() != null) ? idxDims.toLongVector() : new long[0];
                invoke(indexAccumulation, oc, idxDimLong);
            } else if (op instanceof RandomOp) {
                exec((RandomOp) op, oc, Nd4j.getRandom());
            } else if (op instanceof CustomOp) {
                exec((CustomOp) op, oc);
            }
        } finally {
            // Free non-constant replicated input buffers created by ensureDeviceCoherency.
            // Without this, every op that uses a cross-device input leaks a replica buffer.
            closeReplicatedBuffers();
        }

        return op.z();
    }


    @Override
    public TransformOp execAndReturn(TransformOp op) {
        // Route through exec(Op, OpContext) to get device coherency
        exec((Op) op, (OpContext) null);
        return op;
    }



    protected CudaContext invoke(BroadcastOp op, OpContext oc) {
        long st = profilingConfigurableHookIn(op);

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        checkForCompression(op);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());



        val hostYShapeInfo =
                y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo =
                z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        val tadBuffers = tadManager.getTADOnlyShapeInfo(x, op.getDimension());

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = tadBuffers.getSecond();
        val devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        // that's the place where we're going to have second TAD in place
        val tadBuffersZ = tadManager.getTADOnlyShapeInfo(z, op.getDimension());

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), // 0
                context.getOldStream(), // 1
                AtomicAllocator.getInstance().getDeviceIdPointer(), // 2
                context.getBufferAllocation(), // 3
                context.getBufferReduction(),  // 4
                context.getBufferScalar(),  // 5
                context.getBufferSpecial(), // 6
                hostYShapeInfo,  // 7
                hostZShapeInfo,  // 8
                hostTadShapeInfo,  // 9
                devTadShapeInfo,  // 10
                devTadOffsets, // 11
                devTadShapeInfoZ,  // 12
                devTadOffsetsZ); // 13
        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        // Handle null dimensions or dimensions with null data buffer
        INDArray dimArr = op.dimensions();
        INDArray dimCast = null;
        if (dimArr != null && dimArr.data() != null) {
            dimCast = dimArr.castTo(DataType.LONG);
        }
        val dimension = OpaqueNDArray.fromINDArray(dimCast);
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;

        switch (op.getOpType()) {
            case BROADCAST:
                Nd4j.getNativeOps().execBroadcast(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        yb,
                        zb,
                        extraArgs,
                        dimension);
                break;
            case BROADCAST_BOOL:
                Nd4j.getNativeOps().execBroadcastBool(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        yb,
                        zb,
                        extraArgs,
                        dimension);
                break;
            default:
                throw new UnsupportedOperationException("Unknown opType: " + op.getOpType());
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        // Mark output array's DEVICE buffer as written to
        if (z != null && !z.isEmpty() && z.data() != null) {
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);
        }

        profilingConfigurableHookOut(op, oc, st);

        return null;
    }



    protected CudaContext invoke(IndexAccumulation op, OpContext oc, long[] dimension) {
        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        // Normalize negative axes first, then check for "reduce all"
        dimension = Shape.normalizeAxis(x.rank(), dimension);
        if (Shape.wholeArrayDimension(dimension)) {
            dimension = new long[0];
        }

        // Empty dimension array means "reduce all dimensions" (scalar output)
        if (dimension.length == 0) {
            if(z == x || z == null) {
                z = Nd4j.createUninitialized(DataType.LONG, new long[0], 'c');
                setZ(z, op, oc);
            }
        }

        boolean keepDims = op.isKeepDims();
        long[] retShape = Shape.reductionShape(x, dimension, true, keepDims);

        if(z == null || x == z) {
            val ret = Nd4j.createUninitialized(DataType.LONG, retShape);

            setZ(ret, op, oc);
            z = ret;
        } else if(!Arrays.equals(retShape, z.shape())){
            throw new IllegalStateException("Z array shape does not match expected return type for op " + op
                    + ": expected shape " + Arrays.toString(retShape) + ", z.shape()=" + Arrays.toString(z.shape()));
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);


        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        // Validate dimensions are within rank bounds
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= x.rank())
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension) + " contains element that higher then rank of op.X: [" + x.rank() + "]");

        val context = AtomicAllocator.getInstance().getDeviceContext();

        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(x.dataType()), context) : null;
        val hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        // Use dimension directly - normalizeAxis already handled conversion to empty array
        Pair<DataBuffer, DataBuffer> tadBuffers = tadManager.getTADOnlyShapeInfo(x, dimension);

        Pointer hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        Pointer devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        DataBuffer offsets = tadBuffers.getSecond();
        Pointer devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer(offsets, context);

        val xb = OpaqueNDArray.fromINDArray(x);
        val zb = OpaqueNDArray.fromINDArray(z);

        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        // Empty dimension means scalar reduction (all dimensions)
        if (z.isScalar() || dimension.length == 0) {
            Nd4j.getNativeOps().execIndexReduceScalar(xShapeInfoHostPointer, op.opNum(),
                    xb,
                    extraArgs,
                    zb);
        } else {
            if (dimension.length > 1)
                Arrays.sort(dimension);

            val dim = Nd4j.createFromArray(dimension);
            val dimPointer = OpaqueNDArray.fromINDArray(dim);

            Nd4j.getNativeOps().execIndexReduce(xShapeInfoHostPointer, op.opNum(),
                    xb,
                    extraArgs,
                    zb,
                    dimPointer);
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        // Mark output array's DEVICE buffer as written to
        if (z != null && !z.isEmpty() && z.data() != null) {
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);
        }

        profilingConfigurableHookOut(op, oc, st);

        return null;

    }


    protected CudaContext invoke(ReduceOp op, OpContext oc, long[] dimension) {
        val context = AtomicAllocator.getInstance().getDeviceContext();

        INDArray x = getX(op, oc);
        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        if(op instanceof BaseReduceOp && ((BaseReduceOp)op).isEmptyReduce()) {
            //Edge case for TF import compatibility: [x,y].reduce(empty) = [x,y]
            //Note that "empty" axis is NOT the same as length 0, as in INDArray.sum(new int[0]), which means "all dimensions"
            if(z != null) {
                if(!x.isScalar() && !z.isScalar())
                    Preconditions.checkState(x.equalShapes(z), "For empty reductions, result (z) array must have same shape as x shape." +
                            " Got: x=%ndShape, z=%ndShape", x, z);
                z.assign(x);
                return context;
            } else {
                setZ(x.dup(), op, oc);
                return context;
            }
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);

        // Normalize negative axes first (e.g., -1 → rank-1), then check for "reduce all"
        dimension = Shape.normalizeAxis(x.rank(), dimension);
        if (Shape.wholeArrayDimension(dimension)) {
            dimension = new long[0];
        }

        // FIXME: this should be moved down to C++ on per-op basis
        // reduce to scalar case, ReduceBool ops require special treatment
        // dimension.length == 0 means "reduce all dimensions"
        if (op instanceof BaseReduceBoolOp && x.isEmpty() && dimension.length == 0) {
            if (z == null) {
                op.setZ(Nd4j.scalar(((BaseReduceBoolOp) op).emptyValue()));
            } else {
                z.assign(((BaseReduceBoolOp) op).emptyValue());
            }

            return context;
        }

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (dimension.length > 1)
            Arrays.sort(dimension);

        // Validate dimensions are within rank bounds
        for (int i = 0; i < dimension.length; i++)
            if (dimension[i] >= x.rank())
                throw new ND4JIllegalStateException("Op target dimension " + Arrays.toString(dimension)
                        + " contains element that higher then rank of op.X: [" + x.rank() + "]");

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val tadBuffers = x.isEmpty() ? Pair.<DataBuffer, DataBuffer>makePair(x.data(), null) : tadManager.getTADOnlyShapeInfo(x, dimension);

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = x.isEmpty() ? null : tadBuffers.getSecond();
        val devTadOffsets = offsets == null ? null : AtomicAllocator.getInstance().getPointer((DataBuffer) offsets, context);

        Pointer xShapeInfo = AtomicAllocator.getInstance().getPointer(x.shapeInfoDataBuffer(), context);

        long[] retShape = Shape.reductionShape(x, dimension, true, op.isKeepDims());

        if (y != null && dimension.length > 0) {
            // Only do TAD validation when we have specific dimensions (not full array reduction)
            //2 options here: either pairwise, equal sizes - OR every X TAD vs. entirety of Y
            if (x.length() == y.length()) {
                //Pairwise
                if (x.tensorsAlongDimension(dimension) != y.tensorsAlongDimension(dimension)) {
                    throw new ND4JIllegalStateException("Number of TADs along dimension don't match: (x shape = " +
                            Arrays.toString(x.shape()) + ", y shape = " + Arrays.toString(y.shape()) +
                            ", dimension = " + Arrays.toString(dimension) + ")");
                }
            } else if(!(op instanceof ReduceOp)) {
                //Every X TAD vs. entirety of Y
                val xTADSize = x.length() / x.tensorsAlongDimension(dimension);

                if (xTADSize != y.length()) {
                    throw new ND4JIllegalStateException("Size of TADs along dimension don't match for pairwise execution:" +
                            " (x TAD size = " + xTADSize + ", y size = " + y.length());
                }
            }
        }

        val dataType = oc != null ? op.resultType(oc) : op.resultType();

        if( z == null ) {
            val ret = Nd4j.createUninitialized(dataType, retShape);
            setZ(ret, op, oc);
            z = ret;
        } else if(z.dataType() != dataType || !Arrays.equals(retShape, z.shape())){
            throw new ND4JIllegalStateException("Output array for op " + op.getClass().getSimpleName() + " should have type " + dataType + " and shape " + Arrays.toString(retShape)
                    + " but has datatype " + z.dataType() + " and shape " + Arrays.toString(z.shape()));
        }

        val eb = op.extraArgsDataBuff(z.dataType() == DataType.BOOL || op.getOpType() == Op.Type.REDUCE_LONG ? x.dataType() : z.dataType());
        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(eb, context) : null;

        val hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        val xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets);

        // Only compute y-TAD for REDUCE3 ops (e.g., euclidean, dot) where y is a data array.
        // For REDUCE_BOOL ops (All, Any), y is the axis array — computing its TAD using the
        // reduction dimensions causes "Dimension index OOB" when axis rank < dimension.length.
        val yTadBuffers = (y == null || op.getOpType() != Op.Type.REDUCE3) ? null
                : tadManager.getTADOnlyShapeInfo(y, dimension);

        val yDevTadShapeInfo = yTadBuffers == null ? null : AtomicAllocator.getInstance().getPointer(yTadBuffers.getFirst(), context);
        val yOffsets = yTadBuffers == null ? null : yTadBuffers.getSecond();
        val yDevTadOffsets = yOffsets == null ? null : AtomicAllocator.getInstance().getPointer(yOffsets, context);

        if (y != null) {
            xShapeInfoHostPointer.put(12L, yDevTadShapeInfo);
            xShapeInfoHostPointer.put(13L, yDevTadOffsets);
        }


        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);
        INDArray dim = Nd4j.createFromArray(dimension);
        OpaqueNDArray dimensionPointer = OpaqueNDArray.fromINDArray(dim);
        op.validateDataTypes(null);

        if (z.isScalar()) {
            if (op instanceof Variance) {
                Nd4j.getNativeOps().execSummaryStatsScalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        extraArgs,
                        zb,
                        ((Variance) op).isBiasCorrected());
            } else if (y != null && op.getOpType() == Op.Type.REDUCE3) {
                // Only use Reduce3 for actual two-input reductions (dot, cosine, etc.)
                Nd4j.getNativeOps().execReduce3Scalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        extraArgs,
                        yb,
                        zb);
            } else {
                switch (op.getOpType()) {
                    case REDUCE_FLOAT:
                        Nd4j.getNativeOps().execReduceFloat(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb);
                        break;
                    case REDUCE_BOOL:
                        Nd4j.getNativeOps().execReduceBool(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb,dimensionPointer);
                        break;
                    case REDUCE_SAME:
                        Nd4j.getNativeOps().execReduceSame(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb);
                        break;
                    case REDUCE_LONG:
                        Nd4j.getNativeOps().execReduceLong(xShapeInfoHostPointer, op.opNum(),
                                xb,
                                extraArgs,
                                zb,dimensionPointer);
                        break;
                    default:
                        throw new UnsupportedOperationException();
                }
            }
        } else {

            if (y != null && op.getOpType() == Op.Type.REDUCE3) {
                // Only use Reduce3 for actual two-input reductions (dot, cosine, etc.)
                Nd4j.getNativeOps().execReduce3Tad(
                        xShapeInfoHostPointer,
                        op.opNum(),
                        xb,
                        extraArgs,
                        yb,
                        zb,dimensionPointer);
            } else {
                if (op instanceof Variance) {
                    Nd4j.getNativeOps().execSummaryStatsTad(xShapeInfoHostPointer, op.opNum(),
                            xb, extraArgs,
                            zb,
                            dimensionPointer,
                            ((Variance) op).isBiasCorrected());
                } else {
                    switch (op.getOpType()) {
                        case REDUCE_FLOAT:
                            Nd4j.getNativeOps().execReduceFloat2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);

                            break;
                        case REDUCE_SAME:
                            Nd4j.getNativeOps().execReduceSame2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        case REDUCE_BOOL:
                            Nd4j.getNativeOps().execReduceBool2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        case REDUCE_LONG:
                            Nd4j.getNativeOps().execReduceLong2(xShapeInfoHostPointer, op.opNum(),
                                    xb,
                                    extraArgs,
                                    zb, dimensionPointer);
                            break;
                        default:
                            throw new UnsupportedOperationException();
                    }
                }
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        // Mark output array's DEVICE buffer as written to
        if (z != null && !z.isEmpty() && z.data() != null) {
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);
        }

        profilingConfigurableHookOut(op, oc, st);

        Nd4j.getExecutioner().commit();

        return context;
    }


    protected CudaContext intercept(ScalarOp op, long[] dimension) {
        long st = profilingConfigurableHookIn(op);

        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        val context = AtomicAllocator.getInstance().getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        val hostYShapeInfo = op.y() == null ? null : AddressRetriever.retrieveHostPointer(op.y().shapeInfoDataBuffer());
        val hostZShapeInfo = op.z() == null ? null : AddressRetriever.retrieveHostPointer(op.z().shapeInfoDataBuffer());

        val tadBuffers = tadManager.getTADOnlyShapeInfo(op.x(), dimension);

        val hostTadShapeInfo = AddressRetriever.retrieveHostPointer(tadBuffers.getFirst());
        val devTadShapeInfo = AtomicAllocator.getInstance().getPointer(tadBuffers.getFirst(), context);

        val offsets = tadBuffers.getSecond();
        val devTadOffsets = AtomicAllocator.getInstance().getPointer(offsets, context);

        Pointer devTadShapeInfoZ = null;
        Pointer devTadOffsetsZ = null;

        val tadBuffersZ = tadManager.getTADOnlyShapeInfo(op.z(), dimension);

        devTadShapeInfoZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getFirst(), context);
        devTadOffsetsZ = AtomicAllocator.getInstance().getPointer(tadBuffersZ.getSecond(), context);


        PointerPointer extraPointers = extraz.get().put(
                AddressRetriever.retrieveHostPointer(op.x().shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, hostTadShapeInfo, devTadShapeInfo, devTadOffsets,
                devTadShapeInfoZ, devTadOffsetsZ);

        val extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.z().dataType()), context) : null;


        val x = OpaqueNDArray.fromINDArray(op.x());
        val y = OpaqueNDArray.fromINDArray(op.y());
        val z = OpaqueNDArray.fromINDArray(op.z());
        val dim = Nd4j.createFromArray(dimension);
        val dimensionPointer = OpaqueNDArray.fromINDArray(dim);
        switch (op.getOpType()) {
            case SCALAR:
                Nd4j.getNativeOps().execScalarTad(extraPointers, op.opNum(),
                        x,
                        z,
                        y,
                        extraArgs,
                        dimensionPointer);

                break;
            case SCALAR_BOOL:
                Nd4j.getNativeOps().execScalarBoolTad(extraPointers, op.opNum(),
                        x,
                        z,
                        y,
                        extraArgs,
                        dimensionPointer);
                break;
            default:
                throw new UnsupportedOperationException();
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        // Mark output array's DEVICE buffer as written to
        INDArray zArr = op.z();
        if (zArr != null && !zArr.isEmpty() && zArr.data() != null) {
            ((BaseCudaDataBuffer) zArr.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(zArr);
        }

        profilingConfigurableHookOut(op, null, st);

        return null;
    }

    @Override
    public INDArray exec(ScalarOp op) {
        // Route through exec(Op, OpContext) to get device coherency
        return exec((Op) op, (OpContext) null);
    }

    protected CudaContext invoke(ScalarOp op, OpContext oc) {
        // Device coherency is handled by exec(Op, OpContext) before dispatch here.
        INDArray x = getX(op, oc);

        // Handle empty arrays - if input is empty (0 elements), return early with empty output
        if (x != null && x.isEmpty()) {
            INDArray z = getZ(op, oc);
            if (z == null) {
                z = Nd4j.create(x.dataType(), x.shape());
                setZ(z, op, oc);
            }
            return null;
        }

        long st = profilingConfigurableHookIn(op);

        checkForCompression(op);

        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);


        if(z == null){
            switch (op.getOpType()) {
                case SCALAR:
                    z = x.ulike();
                    setZ(x.ulike(), op, oc);
                    break;
                case SCALAR_BOOL:
                    z = Nd4j.createUninitialized(DataType.BOOL, x.shape());
                    setZ(z, op, oc);
                    break;
                default:
                    throw new ND4JIllegalStateException("Unknown op type: [" + op.getOpType() +"]");
            }
        }

        if (x.length() != z.length()) {
            // Detailed diagnostic: compare Java-side shape (jvmShapeInfo) vs native shape info buffer
            String xDiag = "x.length()=" + x.length() + " rank=" + x.rank()
                    + " shape=" + java.util.Arrays.toString(x.shape())
                    + " empty=" + x.isEmpty()
                    + " nativeShapeInfo=" + java.util.Arrays.toString(x.shapeInfoDataBuffer().asInt())
                    + " data.length=" + (x.data() != null ? x.data().length() : "null")
                    + " class=" + x.getClass().getSimpleName();
            String zDiag = "z.length()=" + z.length() + " rank=" + z.rank()
                    + " shape=" + java.util.Arrays.toString(z.shape())
                    + " empty=" + z.isEmpty()
                    + " nativeShapeInfo=" + java.util.Arrays.toString(z.shapeInfoDataBuffer().asInt())
                    + " data.length=" + (z.data() != null ? z.data().length() : "null")
                    + " class=" + z.getClass().getSimpleName()
                    + " z==oc.out0=" + (oc != null && oc.getOutputArray(0) == z)
                    + " zFromOc=" + (oc != null ? (oc.getOutputArray(0) != null ? "rank=" + oc.getOutputArray(0).rank() + " len=" + oc.getOutputArray(0).length() : "null") : "no-oc");
            throw new ND4JIllegalStateException("op.X length should be equal to op.Z length: ["
                    + java.util.Arrays.toString(x.shapeInfoDataBuffer().asInt()) + "] != ["
                    + java.util.Arrays.toString(z.shapeInfoDataBuffer().asInt()) + "]\n"
                    + "  X: " + xDiag + "\n"
                    + "  Z: " + zDiag);
        }

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        // Check for null dimensions AND null data buffer before calling toLongVector()
        INDArray dimArr = op.dimensions();
        if (dimArr != null && dimArr.data() != null) {
            intercept(op, dimArr.toLongVector());
            return null;
        }

        val context = AtomicAllocator.getInstance().getDeviceContext();

        // Ensure scalar is on the current device — op.scalar() may be a constant
        // allocated on device 0 (model load), but the kernel target device may differ
        // (e.g. device 1 after emergency reclaim). Non-P2P access causes error 700.
        INDArray scalar = op.scalar();
        if (scalar != null && scalar.data() != null && getDeviceCount() > 1) {
            int scalarDevice = AtomicAllocator.getInstance().getDeviceId(scalar);
            int currentDevice = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            if (scalarDevice >= 0 && scalarDevice != currentDevice) {
                scalar = Nd4j.getAffinityManager().replicateToDevice(currentDevice, scalar);
            }
        }

        val hostYShapeInfo = scalar == null ? null : AddressRetriever.retrieveHostPointer(scalar.shapeInfoDataBuffer());
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        Pointer extraArgs = op.extraArgs() != null ? AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(op.getOpType() == Op.Type.SCALAR_BOOL ? x.dataType() : z.dataType()), context) : null;


        PointerPointer xShapeInfoHostPointer = extraz.get().put(
                AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), context.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(), context.getBufferAllocation(),
                context.getBufferReduction(), context.getBufferScalar(), context.getBufferSpecial(),
                hostYShapeInfo, hostZShapeInfo, null, null);

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(scalar);
        // When x == z (in-place op like muli), reuse xb to avoid a second
        // fromINDArray call on the same buffer. The second call triggers
        // syncToSpecial() which may H→D copy stale host data over the
        // correct device data, causing the scalar to be applied twice.
        val zb = (x == z) ? xb : OpaqueNDArray.fromINDArray(z);

        switch (op.getOpType()) {
            case SCALAR_BOOL:
                Nd4j.getNativeOps().execScalarBool(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        zb,
                        yb,
                        extraArgs);
                break;
            case SCALAR:
                Nd4j.getNativeOps().execScalar(xShapeInfoHostPointer, op.opNum(),
                        xb,
                        zb,
                        yb,
                        extraArgs);
                break;
            default:
                throw new UnsupportedOperationException("Unknown op type: " + op.getOpType());
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        // Mark output array's DEVICE buffer as written to
        if (z != null && !z.isEmpty() && z.data() != null) {
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);
        }

        profilingConfigurableHookOut(op, oc, st);

        return null;
    }

    protected CudaContext invoke(TransformOp op, OpContext oc) {
        // Device coherency is handled by exec(Op, OpContext) before dispatch here.
        INDArray x = getX(op, oc);

        long st = profilingConfigurableHookIn(op);

        INDArray y = getY(op, oc);
        INDArray z = getZ(op, oc);

        checkForCompression(op);


        AtomicAllocator allocator = AtomicAllocator.getInstance();

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val context = allocator.getDeviceContext();

        if (CudaEnvironment.getInstance().getConfiguration().isDebug())
            lastOp.set(op.opName());

        // special temp array for IsMax along dimension
        INDArray ret = null;

        Pointer xShapeInfo = allocator.getPointer(x.shapeInfoDataBuffer(), context);


        Pointer dimensionDevPointer = null;
        Pointer dimensionHostPointer = null;
        Pointer retPointer = null;
        Pointer retHostShape = null;
        int dimension[] = null;

        Pointer hostYShapeInfo = y == null ? null : AddressRetriever.retrieveHostPointer(y.shapeInfoDataBuffer());


        if (z == null) {
            ret = Nd4j.createUninitialized(op.resultType(), x.shape(), x.ordering());
            setZ(ret, op, oc);
            z = ret;
        }

        // In-place transform on a view: the view shares its DataBuffer with a parent array.
        // On CUDA, OpaqueNDArray.fromINDArray() calls syncToSpecial() on the shared DataBuffer
        // which does H→D for the entire buffer. The kernel writes results to the view's region
        // on device, but the shared DataBuffer's actuality counters track the whole buffer, not
        // per-element regions. When the same buffer is later read, syncToPrimary (D→H) may not
        // correctly round-trip the view's portion because the host still holds pre-op values and
        // the counter state can get confused when the same OpaqueNDArray serves as both x and z.
        // Fix: dup the view to a contiguous standalone array, execute the op on it, then copy
        // results back to the original view.
        boolean viewInPlace = false;
        INDArray originalView = null;
        if (x == z && x.isView()) {
            viewInPlace = true;
            originalView = x;
            x = x.dup(x.ordering());
            z = x;
            if (oc != null) {
                oc.setInputArray(0, x);
                oc.setOutputArray(0, z);
            } else {
                op.setX(x);
                op.setZ(z);
            }
        }

        Pointer extraArgs = op.extraArgs() != null ? allocator.getPointer(op.extraArgsDataBuff(op.getOpType() == Op.Type.TRANSFORM_BOOL || op.getOpType() == Op.Type.PAIRWISE_BOOL ? x.dataType() : z.dataType()), context) : null;
        val hostZShapeInfo = z == null ? null : AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer());

        Pointer hostTadShapeInfo = null;
        Pointer devTadShapeInfo = null;

        Pointer hostMaxTadShapeInfo = null;
        Pointer devMaxTadShapeInfo = null;


        Pointer devTadOffsets = null;
        Pointer devMaxTadOffsets = null;

        op.validateDataTypes(oc, experimentalMode.get());

        PointerPointer xShapeInfoHostPointer =
                extraz.get().put(AddressRetriever.retrieveHostPointer(x.shapeInfoDataBuffer()), // 0
                        context.getOldStream(), // 1
                        allocator.getDeviceIdPointer(), // 2
                        context.getBufferAllocation(), // 3
                        context.getBufferReduction(), // 4
                        context.getBufferScalar(), // 5
                        context.getBufferSpecial(), // 6
                        hostYShapeInfo, // 7
                        hostZShapeInfo, // 8
                        hostTadShapeInfo, // 9
                        devTadShapeInfo, // 10
                        devTadOffsets, // 11
                        hostMaxTadShapeInfo, // 12
                        devMaxTadShapeInfo, // 13
                        devMaxTadOffsets, // 14
                        dimensionDevPointer, // special pointer for IsMax  // 15
                        dimensionHostPointer, // special pointer for IsMax  // 16
                        retPointer, // special pointer for IsMax // 17
                        new CudaPointer(dimension == null ? 0 : dimension.length),
                        retHostShape);

        // DSP lifecycle diagnostic: detect closed DataBuffers before kernel launch.
        if (op.getOpType() == Op.Type.TRANSFORM_ANY) {
            if (z != null && z.data() != null && z.data().wasClosed()) {
                throw new IllegalStateException(String.format(
                    "TRANSFORM_ANY: destination has CLOSED DataBuffer. " +
                    "z.shape=%s z.dtype=%s z.id=%d op=%s",
                    java.util.Arrays.toString(z.shape()), z.dataType(), z.getId(), op.opName()));
            }
            if (x != null && x.data() != null && x.data().wasClosed()) {
                throw new IllegalStateException(String.format(
                    "TRANSFORM_ANY: source has CLOSED DataBuffer. " +
                    "x.shape=%s x.dtype=%s x.id=%d op=%s",
                    java.util.Arrays.toString(x.shape()), x.dataType(), x.getId(), op.opName()));
            }
        }

        val xb = OpaqueNDArray.fromINDArray(x);
        val yb = OpaqueNDArray.fromINDArray(y);
        val zb = OpaqueNDArray.fromINDArray(z);

        if (y != null) {
            switch (op.getOpType()) {
                case TRANSFORM_BOOL:
                case PAIRWISE_BOOL:
                    Nd4j.getNativeOps().execPairwiseTransformBool(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            yb,
                            extraArgs,
                            zb );
                    break;
                default:
                    Nd4j.getNativeOps().execPairwiseTransform(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            yb,
                            zb,
                            extraArgs);
                    break;
            }
        } else {
            switch (op.getOpType()) {
                case TRANSFORM_ANY:
                    Nd4j.getNativeOps().execTransformAny(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_FLOAT:
                    Nd4j.getNativeOps().execTransformFloat(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_BOOL:
                    Nd4j.getNativeOps().execTransformBool(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_SAME:
                    Nd4j.getNativeOps().execTransformSame(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                case TRANSFORM_STRICT:
                    Nd4j.getNativeOps().execTransformStrict(xShapeInfoHostPointer, op.opNum(),
                            xb,
                            extraArgs,
                            zb);
                    break;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        if (extraArgs != null)
            extraArgs.address();

        // Mark output array's DEVICE buffer as written to.
        // After native op execution, the result is in the DEVICE buffer.
        // We need to update Java-side counters so that subsequent reads
        // will correctly sync DEVICE→HOST rather than using stale HOST data.
        if (z != null && !z.isEmpty() && z.data() != null) {
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);
        }

        // Copy results from the temporary contiguous array back to the original view.
        if (viewInPlace && originalView != null) {
            originalView.assign(z);
        }

        profilingConfigurableHookOut(op, oc, st);

        return null;
    }


    /**
     * This method executes specified RandomOp using default RNG available via Nd4j.getRandom()
     *
     * @param op
     */
    @Override
    public INDArray exec(RandomOp op) {
        return exec(op, Nd4j.getRandom());
    }


    @Override
    public INDArray exec(RandomOp op, Random rng) {
        return exec(op, null, rng);
    }

    public INDArray exec(RandomOp op, OpContext oc, Random rng) {
        // Device coherency: ensure all arrays are on the same device
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherencyForOp(op);
        }

        try {
            INDArray x = getX(op, oc);
            INDArray y = getY(op, oc);
            INDArray z = getZ(op, oc);

            if(op instanceof BaseRandomOp && ((BaseRandomOp)op).isTripleArgRngOp() && z != null && x == null && y == null){
                //Ugly hack to ensure the triple arg call occurs
                //See GaussianDistribution.setZ etc
                x = z;
                y = z;
            }

            long st = profilingConfigurableHookIn(op);

            checkForCompression(op);


            if (rng.getStatePointer() == null)
                throw new IllegalStateException(
                        "You should use one of NativeRandom classes for NativeOperations execution");

            if (extraz.get() == null)
                extraz.set(new PointerPointer(32));

            if (CudaEnvironment.getInstance().getConfiguration().isDebug())
                lastOp.set(op.opName());

            val context = AtomicAllocator.getInstance().getDeviceContext();

            PointerPointer extraZZ = extraz.get().put(AddressRetriever.retrieveHostPointer(z.shapeInfoDataBuffer()),
                    context.getOldStream(), AtomicAllocator.getInstance().getDeviceIdPointer());


            val xb = OpaqueNDArray.fromINDArray(x);
            val yb = OpaqueNDArray.fromINDArray(y);
            val zb = OpaqueNDArray.fromINDArray(z);

            if (x != null && y != null && z != null) {
                // triple arg call
                Nd4j.getNativeOps().execRandom3(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        xb,
                        yb,
                        zb,
                        AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()), context));

            } else if (x != null && z != null) {
                //double arg call
                Nd4j.getNativeOps().execRandom2(extraZZ, op.opNum(),
                        rng.getStatePointer(), // rng state ptr
                        xb,
                        zb,
                        AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()),context));


            } else {
                // single arg call
                Nd4j.getNativeOps().execRandom(extraZZ, op.opNum(), rng.getStatePointer(), // rng state ptr
                        zb,
                        AtomicAllocator.getInstance().getPointer(op.extraArgsDataBuff(z.dataType()),context));
            }

            if (Nd4j.getNativeOps().lastErrorCode() != 0)
                throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

            // Mark z as device-written so subsequent ops see correct device data
            ((BaseCudaDataBuffer) z.data()).actualizePointerAndIndexer();
            AtomicAllocator.getInstance().tickDeviceWrite(z);

            profilingConfigurableHookOut(op, oc, st);

            return z;
        } finally {
            closeReplicatedBuffers();
        }
    }

    /**
     * This method return set of key/value
     * and key/key/value objects,
     * describing current environment
     *
     * @return
     */
    @Override
    public synchronized Properties getEnvironmentInformation() {
        if (properties == null) {
            Properties props = super.getEnvironmentInformation();

            List<Map<String, Object>> devicesList = new ArrayList<>();

            // fill with per-device information: name, memory, versions
            for (int i = 0; i < Nd4j.getNativeOps().getAvailableDevices(); i++) {
                Map<String, Object> deviceProps = new HashMap<>();

                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_NAME_KEY, Nd4j.getNativeOps().getDeviceName(i));
                deviceProps.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, Nd4j.getNativeOps().getDeviceFreeMemory(i));
                deviceProps.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, Nd4j.getNativeOps().getDeviceTotalMemory(i));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MAJOR_VERSION_KEY, (long) Nd4j.getNativeOps().getDeviceMajor(i));
                deviceProps.put(Nd4jEnvironment.CUDA_DEVICE_MINOR_VERSION_KEY, (long) Nd4j.getNativeOps().getDeviceMinor(i));

                devicesList.add(i, deviceProps);
            }

            // fill with basic general info
            props.put(Nd4jEnvironment.BACKEND_KEY, "CUDA");
            props.put(Nd4jEnvironment.CUDA_NUM_GPUS_KEY, Nd4j.getNativeOps().getAvailableDevices());
            props.put(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY, devicesList);
            props.put(Nd4jEnvironment.BLAS_VENDOR_KEY, (Nd4j.factory().blas()).getBlasVendor().toString());
            props.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

            // fill bandwidth information
            props.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());

            properties = props;
        } else {

            List<Map<String, Object>> devicesList = (List<Map<String, Object>>) properties.get(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY);

            // just update information that might change over time
            for (int i = 0; i < Nd4j.getNativeOps().getAvailableDevices(); i++) {
                Map<String, Object> dev = devicesList.get(i);

                dev.put(Nd4jEnvironment.CUDA_FREE_MEMORY_KEY, Nd4j.getNativeOps().getDeviceFreeMemory(i));
                dev.put(Nd4jEnvironment.CUDA_TOTAL_MEMORY_KEY, Nd4j.getNativeOps().getDeviceTotalMemory(i));
            }

            properties.put(Nd4jEnvironment.CUDA_DEVICE_INFORMATION_KEY, devicesList);
            properties.put(Nd4jEnvironment.HOST_FREE_MEMORY_KEY, Pointer.maxBytes() - Pointer.totalBytes());

            // fill bandwidth information
            properties.put(Nd4jEnvironment.MEMORY_BANDWIDTH_KEY, PerformanceTracker.getInstance().getCurrentBandwidth());
        }
        return properties;
    }

    @Override
    public TADManager getTADManager() {
        return tadManager;
    }

    @Override
    @SuppressWarnings("unchecked")
    public void printEnvironmentInformation() {
        super.printEnvironmentInformation();
    }

    @Override
    public void commit() {
        val ctx = AtomicAllocator.getInstance().getDeviceContext();
        ctx.syncOldStream();
        ctx.syncSpecialStream();
    }

    @Override
    public synchronized Map<String, CustomOpDescriptor> getCustomOperations() {
        if(customOps == null) {
            String list = Nd4j.getNativeOps().getAllCustomOps();

            if (list == null || list.isEmpty()) {
                log.warn("No customs ops available!");
                customOps = Collections.emptyMap();
                return customOps;
            }

            val map = new HashMap<String, CustomOpDescriptor>();

            String[] split = list.split(";");
            for (String op : split) {
                if (op == null || op.isEmpty())
                    continue;

                String[] another = op.split(":");

                CustomOpDescriptor descriptor = CustomOpDescriptor.builder()
                        .hash(Long.valueOf(another[1]))
                        .numInputs(Integer.valueOf(another[2]))
                        .numOutputs(Integer.valueOf(another[3]))
                        .allowsInplace(Integer.valueOf(another[4]) == 1)
                        .numTArgs(Integer.valueOf(another[5]))
                        .numIArgs(Integer.valueOf(another[6]))
                        .build();

                map.put(another[0], descriptor);
            }

            customOps = Collections.unmodifiableMap(map);
        }

        return customOps;
    }


    /**
     * This method executes given CustomOp
     *
     * PLEASE NOTE: You're responsible for input/output validation
     * @param op Operation to execute
     */
    @Override
    public  INDArray[] exec(@NonNull CustomOp op) {
        // Device-aware execution: ensure all inputs are on the same device as the output
        // When an output is pre-specified (e.g., a.add(b) creates result on current device),
        // we use the OUTPUT's device as target because the caller expects result there.
        // This handles cross-device operations by migrating inputs to the output's device.
        List<INDArray> inputs = op.inputArguments();
        List<INDArray> outputs = op.outputArguments();

        // Determine target device: prefer output's device, fall back to first input's device
        int targetDeviceId = -1;
        if (outputs != null && !outputs.isEmpty()) {
            INDArray firstOutput = outputs.get(0);
            if (firstOutput != null && !firstOutput.isEmpty() && firstOutput.data() != null) {
                targetDeviceId = AtomicAllocator.getInstance().getDeviceId(firstOutput);
            }
        }
        if (targetDeviceId < 0 && inputs != null && !inputs.isEmpty()) {
            INDArray firstInput = inputs.get(0);
            if (firstInput != null && !firstInput.isEmpty() && firstInput.data() != null) {
                targetDeviceId = AtomicAllocator.getInstance().getDeviceId(firstInput);
            }
        }

        if (targetDeviceId >= 0) {
            int currentDeviceId = Nd4j.getAffinityManager().getDeviceForCurrentThread();

            // Switch to target device
            if (targetDeviceId != currentDeviceId) {
                DeviceMemoryManager.getInstance().switchDevice(targetDeviceId, "CudaExecutioner.exec", "input-device-align");
            }

            // Migrate inputs that are on different devices to the target device
            if (inputs != null) {
                boolean migrationOccurred = false;

                // Migrate all inputs that are on different devices
                for (int i = 0; i < inputs.size(); i++) {
                    INDArray input = inputs.get(i);
                    if (input != null && !input.isEmpty() && input.data() != null) {
                        int inputDeviceId = AtomicAllocator.getInstance().getDeviceId(input);
                        if (inputDeviceId >= 0 && inputDeviceId != targetDeviceId) {
                            // Migrate this input to the target device
                            INDArray migrated = Nd4j.getAffinityManager().replicateToDevice(targetDeviceId, input);
                            // Replace in the input list directly
                            inputs.set(i, migrated);
                            migrationOccurred = true;
                        }
                    }
                }

                // Sync after migration to ensure data is fully transferred
                if (migrationOccurred) {
                    Nd4j.getExecutioner().commit();
                }
            }
        }

        val name = op.opName();
        // Set allocation context BEFORE any native calls so allocations are tagged with op name
        if (name != null) {
            Nd4j.getNativeOps().setAllocationContext(name);
        }
        try (val context = buildContext()) {
            op.setupOpContextFromCustomOp(context);
            boolean shapeOverride = op.initializeOutputs(context);
            if (shapeOverride) {
                // Shape calculation reuses this native context and leaves fastpath array bindings behind.
                // Reset them before repopulating the execution context with the finalized outputs.
                context.purgeForReuse();
            }
            long start = profilingConfigurableHookIn(op,context);
            initOpContext(op, shapeOverride, context);

            val result = exec(op, context);
            val states = context.getRngStates();


            // pulling states back
            Nd4j.getRandom().setStates(states.getFirst(), states.getSecond());
            // NOTE: Do NOT call profilingConfigurableHookOut here.
            // exec(op, context) already calls hookOut internally (at line 2453) and then
            // calls closeReplicatedBuffers() in its finally block. If we call hookOut here
            // after exec(op, context) returns, the OpContext inputs have already been freed
            // by closeReplicatedBuffers(), causing false "closed before call" errors for
            // non-constant cross-device inputs that were migrated as temporary replicas.
            // The inner exec(op,context) handles the complete hookIn/hookOut/cleanup cycle.

            return result;
        } catch (ND4JOpProfilerException e) {
            throw e;
        } catch (Exception e) {
            throw ND4JOpExceptionUtils.opExecutionException(op, null, e);
        } finally {
            // Clear allocation context after op execution
            Nd4j.getNativeOps().clearAllocationContext();
            // Clear ThreadLocal to prevent stale context references
            clearOpContext();
        }


    }




    @Override
    public ExecutionerType type() {
        return ExecutionerType.CUDA;
    }

    @Override
    public String getString(DataBuffer buffer, long index) {
        Preconditions.checkArgument(buffer instanceof CudaUtf8Buffer, "Expected Utf8Buffer");

        val addr = ((LongIndexer) buffer.indexer()).get(index);
        val ptr = new PagedPointer(addr);
        val str = new Nd4jCuda.utf8string(ptr);
        return str._buffer().capacity(str._length()).getString();
    }

    @Override
    public boolean isExperimentalMode() {
        return experimentalMode.get();
    }



    @Override
    public OpContext buildContext() {
        return new CudaOpContext();
    }

    @Override
    public INDArray[] exec(CustomOp op, OpContext context) {
        // Device coherency: ensure all arrays are on the same device
        // and the CUDA context is set to the target device for op execution.
        // Skipped when DSP parallel workers handle device placement themselves.
        if (!skipDeviceCoherency.get()) {
            ensureDeviceCoherency(context.getInputArrays(), context.getOutputArrays(), context);
        }

        if (MultiGpuTracer.ENABLED) {
            int currentDev = Nd4j.getAffinityManager().getDeviceForCurrentThread();
            MultiGpuTracer.verifyDeviceContext(op.opName(), currentDev, currentDev);
        }

        try {
            Nd4j.getExecutioner().commit();
            long st = profilingConfigurableHookIn(op, context);
            if(op instanceof UserDefinedCustomOp) {
                ((UserDefinedCustomOp) op).exec(context);
                // Tick device write for UserDefinedCustomOp outputs too
                for (val out : context.getOutputArrays()) {
                    if (out != null && !out.isEmpty() && out.data() != null) {
                        ((BaseCudaDataBuffer) out.data()).actualizePointerAndIndexer();
                        AtomicAllocator.getInstance().tickDeviceWrite(out);
                    }
                }
                return context.getOutputArrays().toArray(new INDArray[0]);
            }



            val status = Nd4j.getNativeOps().execCustomOp2(null, op.opHash(), context.contextPointer());
            if (Nd4j.getNativeOps().lastErrorCode() != 0)
                throw ND4JOpExceptionUtils.opExecutionException(op, Nd4j.getNativeOps().lastErrorMessage(), null);

            if (status != 0)
                throw ND4JOpExceptionUtils.opExecutionException(op, "Status code: " + status, null);

            // Actualize pointers for context arrays (NOT op.inputArguments()/op.outputArguments()
            // which are stale arrays from graph construction, not the current execution's arrays).
            for (val in : context.getInputArrays()) {
                if (in != null && !in.isEmpty() && in.data() != null)
                    ((BaseCudaDataBuffer) in.data()).actualizePointerAndIndexer();
            }

            for (val out : context.getOutputArrays()) {
                if (out != null && !out.isEmpty() && out.data() != null) {
                    ((BaseCudaDataBuffer) out.data()).actualizePointerAndIndexer();
                    AtomicAllocator.getInstance().tickDeviceWrite(out);
                }
            }

            profilingConfigurableHookOut(op, context, st);

            if (context.getOutputArrays().isEmpty())
                return new INDArray[0];
            else
                return context.getOutputArrays().toArray(new INDArray[context.getOutputArrays().size()]);
        } finally {
            // Free non-constant replicated input buffers created by ensureDeviceCoherency.
            // Safe to call even if already called by exec(Op, OpContext) — the list is
            // cleared after each call, so the second call finds an empty list.
            closeReplicatedBuffers();
        }
    }

    @Override
    public INDArrayStatistics inspectArray(@NonNull INDArray array) {
        val debugInfo = new Nd4jCuda.DebugInfo();
        val ctx = AtomicAllocator.getInstance().getDeviceContext();
        AtomicAllocator.getInstance().synchronizeHostData(array);

        if (extraz.get() == null)
            extraz.set(new PointerPointer(32));

        val extras = extraz.get().put(
                null,
                ctx.getOldStream(),
                AtomicAllocator.getInstance().getDeviceIdPointer(),
                ctx.getBufferAllocation(),
                ctx.getBufferReduction(),
                ctx.getBufferScalar(),
                ctx.getBufferSpecial());


        Nd4j.getNativeOps().inspectArray(extras, AtomicAllocator.getInstance().getHostPointer(array), (LongPointer) AtomicAllocator.getInstance().getHostPointer(array.shapeInfoDataBuffer()), AtomicAllocator.getInstance().getPointer(array, ctx), (LongPointer) AtomicAllocator.getInstance().getPointer(array.shapeInfoDataBuffer()), debugInfo);

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        return INDArrayStatistics.builder()
                .minValue(debugInfo._minValue())
                .maxValue(debugInfo._maxValue())
                .meanValue(debugInfo._meanValue())
                .stdDevValue(debugInfo._stdDevValue())
                .countInf(debugInfo._infCount())
                .countNaN(debugInfo._nanCount())
                .countNegative(debugInfo._negativeCount())
                .countPositive(debugInfo._positiveCount())
                .countZero(debugInfo._zeroCount())
                .build();
    }


    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        long extras = ArrayOptionsHelper.composeTypicalChecks(empty, dtype, false, false, false, false, false);
        return createShapeInfo(shape, stride, elementWiseStride, order, dtype, extras);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, boolean empty, boolean isView) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        long extras = ArrayOptionsHelper.composeTypicalChecks(empty, dtype, false, false, isView, false, false);
        return createShapeInfo(shape, stride, elementWiseStride, order, dtype, extras);
    }

    @Override
    public DataBuffer createShapeInfo(long[] shape, long[] stride, long elementWiseStride, char order, DataType dtype, long extras) {
        LongPointer shapePointer = new LongPointer(shape);
        LongPointer stridePointer = new LongPointer(stride);
        OpaqueConstantShapeBuffer dbf = Nd4j.getNativeOps().shapeBufferEx(shape.length, shapePointer, stridePointer, dtype.toInt(), order, elementWiseStride, extras);
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        dbf.retainReference();

        Pointer primaryShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferPrimary(dbf);
        Pointer specialShapeInfo = Nd4j.getNativeOps().getConstantShapeBufferSpecial(dbf);
        long shapeInfoLength = Shape.shapeInfoLength(shape.length);

        val result = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);

        return result;
    }

    @Override
    public TadPack tadShapeInfoAndOffsets(INDArray array, long[] dimension) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        OpaqueTadPack pack = Nd4j.getNativeOps().tadOnlyShapeInfo( array.shapeInfoDataBuffer().opaqueBuffer(), new LongPointer(ArrayUtil.toLongArray(dimension)), dimension.length).retainReference();

        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        LongPointer primaryShapeInfo = Nd4j.getNativeOps().getPrimaryShapeInfo(pack).retainReference();
        LongPointer specialShapeInfo = Nd4j.getNativeOps().getSpecialShapeInfo(pack).retainReference();
        long shapeInfoLength =  Nd4j.getNativeOps().getShapeInfoLength(pack);
        LongPointer primaryOffsets = Nd4j.getNativeOps().getPrimaryOffsets(pack).retainReference();
        LongPointer specialOffsets =  Nd4j.getNativeOps().getSpecialOffsets(pack).retainReference();
        long numTads = Nd4j.getNativeOps().getNumberOfTads(pack);

        val tadShape = new CudaLongDataBuffer(primaryShapeInfo, specialShapeInfo, shapeInfoLength, true);
        val tadOffsets = new CudaLongDataBuffer(primaryOffsets, specialOffsets, numTads, true);

        return new TadPack(tadShape, tadOffsets);
    }

    @Override
    public INDArray createFromDescriptor(DataBuffer shapeInformation) {
        JCublasNDArray ndArray = new JCublasNDArray();
        ndArray.setShapeInfoDataBuffer(shapeInformation);
        DataType dt = Shape.dataType(ndArray.shapeInfoJava());
        DataBuffer buff = Nd4j.createBuffer(dt,ndArray.length(),false);
        ndArray.setData(buff);
        return ndArray;
    }


    @Override
    public DataBuffer createConstantBuffer(long[] values, DataType desiredType) {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().constantBufferLong(desiredType.toInt(), new LongPointer(values), values.length);
        dbf.retainReference();
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val buffer = Nd4j.createBuffer(Nd4j.getNativeOps().getConstantDataBufferPrimary(dbf), Nd4j.getNativeOps().getConstantDataBufferSpecial(dbf), values.length, desiredType);
        buffer.setConstant(true);

        return buffer;
    }

    @Override
    public DataBuffer createConstantBuffer(double[] values, DataType desiredType)  {
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val dbf = Nd4j.getNativeOps().constantBufferDouble(desiredType.toInt(), new DoublePointer(values), values.length);
        dbf.retainReference();
        if (Nd4j.getNativeOps().lastErrorCode() != 0)
            throw new RuntimeException(Nd4j.getNativeOps().lastErrorMessage());

        val buffer = Nd4j.createBuffer(Nd4j.getNativeOps().getConstantDataBufferPrimary(dbf), Nd4j.getNativeOps().getConstantDataBufferSpecial(dbf), values.length, desiredType);
        buffer.setConstant(true);

        return buffer;
    }
}

