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

package org.nd4j.autodiff.samediff.internal;

import lombok.extern.slf4j.Slf4j;
import org.nd4j.autodiff.samediff.config.SDValue;
import org.nd4j.autodiff.samediff.config.SDValueType;
import org.nd4j.autodiff.samediff.internal.memory.ArrayCacheMemoryMgr;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.OpContext;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

@Slf4j
public class BufferLifecycleManager {

    private final InferenceSession session;

    public BufferLifecycleManager(InferenceSession session) {
        this.session = session;
    }

    /** Increment live reference count for the DataBuffer of the given array. */
    public void trackLiveBuffer(INDArray array, IdentityHashMap<DataBuffer, Integer> liveDataBufferRefs) {
        if (array == null || array.data() == null) return;
        DataBuffer buf = array.data();
        liveDataBufferRefs.merge(buf, 1, Integer::sum);
    }

    /** Decrement live reference count and remove entry when it reaches zero. */
    public void untrackLiveBuffer(INDArray array, IdentityHashMap<DataBuffer, Integer> liveDataBufferRefs) {
        if (array == null || array.data() == null) return;
        DataBuffer buf = array.data();
        Integer count = liveDataBufferRefs.get(buf);
        if (count == null) return;
        if (count <= 1) {
            liveDataBufferRefs.remove(buf);
        } else {
            liveDataBufferRefs.put(buf, count - 1);
        }
    }

    /** Return true if the DataBuffer is exclusively owned (ref count == 1 or not tracked). */
    public boolean isBufferExclusivelyOwned(INDArray array, IdentityHashMap<DataBuffer, Integer> liveDataBufferRefs) {
        if (array == null || array.data() == null) return true;
        Integer count = liveDataBufferRefs.get(array.data());
        return count == null || count <= 1;
    }

    public int closeLatestRequestedOutputBuffers(Map<String, SDValue> latestOutputs,
                                                  IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (latestOutputs == null || latestOutputs.isEmpty()) {
            return 0;
        }

        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (SDValue value : latestOutputs.values()) {
            collectCloseableBuffers(value, uniqueBuffers);
        }

        int closed = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (buf == null || protectedBuffers.containsKey(buf)) {
                continue;
            }
            if (buf.closeable()) {
                try {
                    buf.close();
                    closed++;
                } catch (Exception ignored) {
                    // Buffer may already be deallocated or owned elsewhere.
                }
            }
        }
        latestOutputs.clear();
        return closed;
    }

    public int closeNodeValueOutputBuffers(Map<VarId, SDValue> nodeValueOutputs,
                                            IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (nodeValueOutputs.isEmpty()) {
            return 0;
        }

        IdentityHashMap<DataBuffer, Boolean> uniqueBuffers = new IdentityHashMap<>();
        for (SDValue value : nodeValueOutputs.values()) {
            collectCloseableBuffers(value, uniqueBuffers);
        }

        int closed = 0;
        for (DataBuffer buf : uniqueBuffers.keySet()) {
            if (buf == null || protectedBuffers.containsKey(buf)) {
                continue;
            }

            // Never force-close constant buffers — they are model weights that must
            // survive session cleanup. The protectedBuffers snapshot may miss them if
            // identity ops alias a constant's DataBuffer to an output slot, causing the
            // buffer to appear in nodeValueOutputs with a different identity.
            if (buf.isConstant()) {
                continue;
            }

            if (buf.closeable()) {
                try {
                    buf.close();
                    closed++;
                } catch (Exception ignored) {
                    // Buffer may already be deallocated or owned elsewhere.
                }
            }
        }

        nodeValueOutputs.clear();
        return closed;
    }

    /**
     * Close all pooled OpContexts and clear accumulated thread-local state.
     * Must be called during resetSession() to prevent stale native pointers
     * from being reused in subsequent sessions.
     */
    public void closePooledResources(Map<String, OpContext> activeContexts,
                                      java.util.Deque<OpContext> pool,
                                      java.util.Set<Long> freedArrays,
                                      AbstractDependencyTracker<SDValue, Dep> arrayUseTracker,
                                      IdentityHashMap<DataBuffer, Integer> liveDataBufferRefs,
                                      Map<Long, List<DataBuffer>> outputShapeCache,
                                      Map<String, INDArray> outputArrayCache) {
        int activeClosed = 0;
        for (OpContext ctx : activeContexts.values()) {
            if (ctx != null) {
                try {
                    ctx.close();
                    activeClosed++;
                } catch (Exception e) {
                    // Already closed or invalid; ignore
                }
            }
        }
        activeContexts.clear();

        // Close pooled OpContexts that were returned via purge() during execution.
        // These hold live native pointers that become stale after session reset.
        int pooledClosed = 0;
        for (OpContext ctx : pool) {
            if (ctx != null) {
                try {
                    ctx.close();
                    pooledClosed++;
                } catch (Exception e) {
                    // Already closed or invalid; ignore
                }
            }
        }
        pool.clear();

        // Clear freed arrays tracker to avoid stale entries
        freedArrays.clear();

        // Clear dependency tracking and live buffer refs held by this thread's last execution.
        arrayUseTracker.clear();
        liveDataBufferRefs.clear();

        // Clear output shape and array caches
        outputShapeCache.clear();
        outputArrayCache.clear();
    }

    /**
     * Flush the array cache, freeing all cached intermediate arrays.
     *
     * @param mmgr the memory manager
     * @param protectedBuffers DataBuffers to skip (e.g., static KV buffers).
     *                         May be null if no protection is needed.
     */
    public void flushArrayCache(SessionMemMgr mmgr, IdentityHashMap<DataBuffer, Boolean> protectedBuffers) {
        if (mmgr instanceof ArrayCacheMemoryMgr) {
            ((ArrayCacheMemoryMgr) mmgr).close(protectedBuffers);
            log.info("Flushed ArrayCacheMemoryMgr");
        }
    }

    public void collectCloseableBuffers(SDValue value, IdentityHashMap<DataBuffer, Boolean> buffers) {
        if (value == null) {
            return;
        }

        switch (value.getSdValueType()) {
            case TENSOR:
                INDArray tensor = value.getTensorValue();
                if (tensor != null && !tensor.wasClosed() && tensor.data() != null) {
                    buffers.put(tensor.data(), Boolean.TRUE);
                }
                break;
            case LIST:
                if (value.getListValue() != null) {
                    for (INDArray arr : value.getListValue()) {
                        if (arr != null && !arr.wasClosed() && arr.data() != null) {
                            buffers.put(arr.data(), Boolean.TRUE);
                        }
                    }
                }
                break;
            case DICT:
                if (value.getDictValue() != null) {
                    for (INDArray arr : value.getDictValue().values()) {
                        if (arr != null && !arr.wasClosed() && arr.data() != null) {
                            buffers.put(arr.data(), Boolean.TRUE);
                        }
                    }
                }
                break;
            default:
                break;
        }
    }
}
