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

package org.eclipse.deeplearning4j.llm.generation;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.ArrayHolder;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;

/**
 * Memory management utilities for SameDiff models.
 *
 * <p>Addresses the "closeable poisoning" problem: {@code SameDiff.directExecHelper()}
 * calls {@code setCloseable(false)} on all placeholders, which sets
 * {@code dataBuffer.setConstant(true)} — marking the buffer as constant and
 * preventing {@code close()} from freeing memory. {@link #safeClose(INDArray)}
 * undoes this poisoning before closing.</p>
 */
@Slf4j
public final class SameDiffMemoryUtils {
    private static final int CLOSED_GRAPH_RECLAIM_PASSES = 3;

    private SameDiffMemoryUtils() {}

    /**
     * Safely close an INDArray, undoing any "closeable poisoning" from SameDiff execution.
     *
     * <p>{@code SameDiff.directExecHelper()} calls {@code setCloseable(false)} on all
     * placeholder arrays, which internally sets {@code dataBuffer.setConstant(true)}.
     * This prevents {@code close()} from actually freeing the GPU memory. This method
     * calls {@code setCloseable(true)} first to undo the poisoning, then closes.</p>
     *
     * @param arr the array to close, may be null
     */
    public static void safeClose(INDArray arr) {
        if (arr == null) {
            return;
        }
        closeArray(arr, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    /**
     * Free all constant and variable arrays from a SameDiff model.
     *
     * <p>Iterates through the model's constant and variable array holders,
     * removes each array, undoes the constant flag, and closes it.
     * This is useful for freeing model weights after they are no longer needed
     * (e.g., freeing a vision encoder after encoding is complete).</p>
     *
     * @param model the SameDiff model whose arrays should be freed
     * @return the number of arrays freed
     */
    public static int freeModelArrays(SameDiff model) {
        if (model == null) {
            return 0;
        }

        // Model execution is asynchronous on accelerator backends. Never release weights
        // while queued kernels may still reference them; doing so poisons the CUDA stream
        // and causes the next unrelated operation to report error 700.
        Nd4j.getExecutioner().commit();

        List<INDArray> arrays = new ArrayList<>();

        // Free constant arrays
        removeArrayHolder(model.getConstantArrays(), arrays);

        // Free variable arrays
        removeArrayHolder(model.getVariablesArrays(), arrays);

        Set<DataBuffer> closedBuffers = Collections.newSetFromMap(new IdentityHashMap<>());
        int closedArrayCount = 0;
        for (INDArray arr : arrays) {
            if (closeArray(arr, closedBuffers)) {
                closedArrayCount++;
            }
        }

        log.info("Freed {} model arrays ({} unique data buffers)", closedArrayCount, closedBuffers.size());
        return closedArrayCount;
    }

    private static void removeArrayHolder(ArrayHolder holder, List<INDArray> arrays) {
        if (holder == null) {
            return;
        }

        Collection<String> names = holder.arrayNames();
        if (names == null || names.isEmpty()) {
            return;
        }

        // Copy names to avoid ConcurrentModificationException
        List<String> nameList = new ArrayList<>(names);

        for (String name : nameList) {
            try {
                INDArray arr = holder.removeArray(name);
                if (arr != null) {
                    arrays.add(arr);
                }
            } catch (Exception e) {
                log.debug("Exception removing model array '{}': {}", name, e.getMessage());
            }
        }
    }

    private static boolean closeArray(INDArray arr, Set<DataBuffer> closedBuffers) {
        if (arr == null) {
            return false;
        }

        DataBuffer data = null;
        try {
            data = arr.data();
        } catch (Exception e) {
            log.debug("Exception reading array data buffer during close: {}", e.getMessage());
        }

        if (arr.wasClosed() || (data != null && data.wasClosed())) {
            return false;
        }

        try {
            arr.clearOpaqueNDArray();
        } catch (Exception e) {
            log.debug("Exception clearing OpaqueNDArray during close: {}", e.getMessage());
        }

        // Undo constant protection so close() can deallocate
        try {
            if (data != null && !data.wasClosed()) {
                data.setConstant(false);
            }
            arr.setCloseable(true);
            if (arr.closeable()) {
                arr.close();
                if (arr.wasClosed()) {
                    if (data != null && data.wasClosed()) {
                        closedBuffers.add(data);
                    }
                    return true;
                }
            }
        } catch (Exception e) {
            log.debug("Exception during array close, falling back to data buffer close: {}", e.getMessage());
        }

        // Model constants can be stored as views or shape-adjusted wrappers whose
        // INDArray.close() is intentionally conservative (sub-view check blocks close
        // when length < buffer.length). Since freeModelArrays is called during model
        // destruction, we know the entire buffer is being released — close the underlying
        // data buffer directly for each unique buffer.
        if (data != null && !data.wasClosed() && closedBuffers.add(data)) {
            try {
                data.setConstant(false);
                if (data.isAttached()) {
                    // Buffer is in a workspace — detach it first so it can be freed
                    log.debug("Model buffer is attached to workspace, forcing detach for cleanup");
                }
                data.close();
                return data.wasClosed();
            } catch (Exception e) {
                log.warn("Failed to close model data buffer (attached={}, constant={}, length={}): {}",
                        data.isAttached(), data.isConstant(), data.length(), e.getMessage());
            }
        }

        return arr.wasClosed() || (data != null && data.wasClosed());
    }

    /**
     * Free all GPU memory held by a vision encoder's model arrays.
     *
     * <p>Call this after vision encoding is complete and the encoder is no longer needed.
     * Clears placeholders, resets sessions, frees all constant/variable arrays, and
     * closes the model. This reclaims the ~5GB+ of GPU memory used by vision encoder weights.</p>
     *
     * @param visionEncoder the vision encoder SameDiff model to free
     */
    public static void freeVisionEncoder(SameDiff visionEncoder) {
        if (visionEncoder == null) {
            return;
        }
        try {
            visionEncoder.clearPlaceholders(true);
            visionEncoder.clearOpInputs();
            visionEncoder.resetSession();
            freeModelArrays(visionEncoder);
            visionEncoder.close();
            log.info("Vision encoder freed");
        } catch (Exception e) {
            log.warn("Error freeing vision encoder: {}", e.getMessage());
        }
    }

    /**
     * Trim CUDA memory pools on all devices.
     * Syncs pending cudaFreeAsync calls and releases physical memory back to the OS.
     */
    public static void trimAllDevicePools() {
        try {
            var nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
            // Java-owned arrays close via stream-ordered frees after decoder.resetSession().
            // Complete those frees first, then use the established full-pool trim from
            // SameDiff.resetSession(); stream-local trim can run before these frees land.
            Nd4j.getExecutioner().commit();
            int numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
            for (int d = 0; d < numDevices; d++) {
                nativeOps.trimMemoryPool(d);
            }
        } catch (Exception e) {
            log.debug("Failed to trim memory pools: {}", e.getMessage());
        }
    }

    /**
     * Reclaim phantom-managed native wrappers after a complete graph owner closes.
     * Multiple bounded passes are required because OpaqueNDArray cleanup releases
     * DataBuffer referents that become collectible only in the following GC pass.
     * Live references are never flushed: DeallocatorService requires a positive
     * runtime liveness verdict for this production path.
     *
     * @return number of collected registrations reclaimed across all passes
     */
    public static int reclaimClosedGraphResources() {
        int reclaimed = 0;
        for (int pass = 0; pass < CLOSED_GRAPH_RECLAIM_PASSES; pass++) {
            reclaimed += reclaimCollectedNativeResources();
        }
        trimAllDevicePools();
        log.info("Closed-graph reclamation freed {} collected native registrations; {} remain",
                reclaimed, Nd4j.getDeallocatorService().getReferenceMap().size());
        return reclaimed;
    }

    /**
     * Run one collected-only reclamation pass at a completed execution boundary.
     * Live graph, plan, constant, and retained-state registrations remain untouched.
     *
     * @return number of positively collected registrations reclaimed
     */
    public static int reclaimCollectedNativeResources() {
        Nd4j.getMemoryManager().invokeGc();
        return Nd4j.getDeallocatorService().flushCollectedReferences();
    }
}
