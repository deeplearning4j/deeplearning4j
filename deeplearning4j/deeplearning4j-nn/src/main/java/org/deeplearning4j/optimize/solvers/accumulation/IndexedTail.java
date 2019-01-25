/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import lombok.var;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.ThresholdCompression;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.AtomicBoolean;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class provides queue-like functionality for multiple readers/multiple writers, with transparent duplication
 * and collapsing ability
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class IndexedTail {
    // here we store positions of individual consumers
    protected ConcurrentHashMap<Long, AtomicLong> positions = new ConcurrentHashMap<>();

    // here we store individual updates
    protected Map<Long, INDArray> updates = new ConcurrentHashMap<>();

    // simple counter for new updates
    protected AtomicLong updatesCounter = new AtomicLong(0);

    // index of last deleted element. used for maintenance, and removal of useless updates
    protected AtomicLong lastDeletedIndex = new AtomicLong(-1);

    // this value is used as max number of possible consumers.
    protected final int expectedConsumers;

    // flag useful for debugging only
    protected AtomicBoolean dead = new AtomicBoolean(false);

    protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    // fields required for collapser
    protected final boolean allowCollapse;
    protected final long[] shape;
    protected final int collapseThreshold = 32;
    protected AtomicBoolean collapsedMode = new AtomicBoolean(false);
    protected AtomicLong collapsedIndex = new AtomicLong(-1);

    public IndexedTail(int expectedConsumers) {
        this(expectedConsumers, false, null);
    }

    public IndexedTail(int expectedConsumers, boolean allowCollapse, long[] shape) {
        this.expectedConsumers = expectedConsumers;
        this.allowCollapse = allowCollapse;

        if (allowCollapse)
            Preconditions.checkArgument(shape != null, "shape can't be null if collapse is allowed");

        this.shape = shape;
    }

    /**
     * This mehtod adds update, with optional collapse
     * @param update
     */
    public void put(@NonNull INDArray update) {
        try {
            lock.writeLock().lock();

            //if we're already in collapsed mode - we just insta-decompress
            if (collapsedMode.get()) {
                val lastUpdateIndex = collapsedIndex.get();
                val lastUpdate = updates.get(lastUpdateIndex);

                Preconditions.checkArgument(!lastUpdate.isCompressed(), "lastUpdate should NOT be compressed during collapse mode");

                smartDecompress(update, lastUpdate);

                // collapser only can work if all consumers are already introduced
            } else if (allowCollapse && positions.size() >= expectedConsumers) {
                // getting last added update
                val lastUpdateIndex = updatesCounter.get();

                // looking for max common non-applied update
                long maxIdx = firstNotAppliedIndexEverywhere();
                val array = Nd4j.create(shape);

                val delta = lastUpdateIndex - maxIdx;
                if (delta >= collapseThreshold) {
                    log.info("Max delta to collapse: {}; Range: <{}...{}>", delta, maxIdx, lastUpdateIndex);
                    for (long e = maxIdx; e < lastUpdateIndex; e++) {
                        val u = updates.get(e);
                        if (u == null)
                            log.error("Failed on index {}", e);
                           // continue;

                        smartDecompress(u, array);

                        // removing updates array
                        updates.remove(e);
                    }

                    // decode latest update
                    smartDecompress(update, array);

                    // putting collapsed array back at last index
                    updates.put(lastUpdateIndex, array);
                    collapsedIndex.set(lastUpdateIndex);

                    // shift counter by 1
                    updatesCounter.getAndIncrement();

                    // we're saying that right now all updates within some range are collapsed into 1 update
                    collapsedMode.set(true);
                } else {
                    updates.put(updatesCounter.getAndIncrement(), update);
                }
            } else {
                updates.put(updatesCounter.getAndIncrement(), update);
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    protected long firstNotAppliedIndexEverywhere() {
        long maxIdx = -1;

        // if there's no updates posted yet - just return negative value
        if (updatesCounter.get() == 0)
            return maxIdx;

        for (val v:positions.values()) {
            if (v.get() > maxIdx)
                maxIdx = v.get();
        }

        return maxIdx + 1;
    }

    protected long maxAppliedIndexEverywhere() {
        long maxIdx = Long.MAX_VALUE;
        for (val v:positions.values()) {
            if (v.get() < maxIdx)
                maxIdx = v.get();
        }

        return maxIdx;
    }

    public boolean hasAnything() {
        return hasAnything(Thread.currentThread().getId());
    }

    /**
     *
     * @return
     */
    public boolean hasAnything(long threadId) {
        var threadPosition = getLocalPosition(threadId);

        val r = threadPosition < updatesCounter.get();
        log.info("hasAnything({}): {}; position: {}; updates: {}", threadId, r, threadPosition, updatesCounter.get());

        return r;
    }

    public boolean drainTo(@NonNull INDArray array) {
        return drainTo(Thread.currentThread().getId(), array);
    }

    protected long getGlobalPosition() {
        try {
            lock.readLock().lock();

            return updatesCounter.get();
        } finally {
            lock.readLock().unlock();
        }
    }

    protected long getLocalPosition() {
        return getLocalPosition(Thread.currentThread().getId());
    }

    protected long getDelta() {
        return getDelta(Thread.currentThread().getId());
    }

    protected long getDelta(long threadId) {
        return getGlobalPosition() - getLocalPosition(threadId);
    }

    protected long getLocalPosition(long threadId) {
        var threadPosition = positions.get(threadId);

        // will be instantiated on first call from any given thread
        if (threadPosition == null) {
            threadPosition = new AtomicLong(-1);
            positions.put(threadId, threadPosition);
        }

        return threadPosition.get() < 0 ? 0 : threadPosition.get();
    }

    public boolean drainTo(long threadId, @NonNull INDArray array) {
        var threadPosition = positions.get(threadId);

        // will be instantiated on first call from any given thread
        if (threadPosition == null) {
            threadPosition = new AtomicLong(-1);
            positions.put(threadId, threadPosition);
        }

        long globalPos = 0;
        long localPos = 0;
        long delta = 0;
        val sessionUpdates = new ArrayList<INDArray>();

        try {
            lock.readLock().lock();

            // since drain fetches all existing updates for a given consumer
            collapsedMode.set(false);

            globalPos = updatesCounter.get();
            localPos = getLocalPosition(threadId);

            // we're finding out, how many arrays we should provide
            delta = getDelta(threadId);

            // within read lock we only move references and tag updates as applied
            for (long e = localPos; e < localPos + delta; e++) {
                val update = updates.get(e);

                if (allowCollapse && update == null)
                    continue;

                // FIXME: just continue here, probably it just means that collapser was working in this position
                if (update == null) {
                    log.info("Global: [{}]; Local: [{}]", globalPos, localPos);
                    throw new RuntimeException("Element [" + e + "] is absent");
                }

                sessionUpdates.add(update);
            }

            // and shifting stuff by one
            threadPosition.set(globalPos);
        } finally {
            lock.readLock().unlock();
        }


        // now we decompress all arrays within delta into provided array
        for (val u:sessionUpdates) {
            smartDecompress(u.unsafeDuplication(true), array);
        }



        // TODO: this call should be either synchronized, or called from outside
        maintenance();

        return delta > 0;
    }

    /**
     * This method does maintenance of updates within
     */
    protected synchronized void maintenance() {
        // first of all we're checking, if all consumers were already registered. if not - just no-op.
        if (positions.size() < expectedConsumers) {
            log.info("Skipping maintanance due to not all expected consumers shown up: [{}] vs [{}]", positions.size(), expectedConsumers);
            return;
        }

        // now we should get minimal id of consumed update
        val minIdx = maxAppliedIndexEverywhere();
        val allPositions = new long[positions.size()];
        int cnt = 0;
        for (val p:positions.values())
            allPositions[cnt++] = p.get();

        log.info("Min idx: {}; last deleted index: {}; stored updates: {}; positions: {}", minIdx, lastDeletedIndex.get(), updates.size(), allPositions);

        // now we're checking, if there are undeleted updates between
        if (minIdx > lastDeletedIndex.get()) {
            // delete everything between them
            for (long e = lastDeletedIndex.get(); e < minIdx; e++) {
                updates.remove(e);
            }

            // now, making sure we won't try to delete stuff twice
            lastDeletedIndex.set(minIdx);
            //System.gc();
        }
    }

    /**
     * This method returns actual number of updates stored within tail
     * @return
     */
    protected int updatesSize() {
        return updates.size();
    }

    protected INDArray smartDecompress(INDArray encoded, @NonNull INDArray target) {
        INDArray result = target;

        if (encoded.isCompressed() || encoded.data().dataType() == DataType.INT) {
            int encoding = encoded.data().getInt(3);
            if (encoding == ThresholdCompression.FLEXIBLE_ENCODING) {
                Nd4j.getExecutioner().thresholdDecode(encoded, result);
            } else if (encoding == ThresholdCompression.BITMAP_ENCODING) {
                Nd4j.getExecutioner().bitmapDecode(encoded, result);
            } else
                throw new ND4JIllegalStateException("Unknown encoding mode: [" + encoding + "]");
        } else {
            result.addi(encoded);
        }

        return result;
    }

    protected boolean isDead() {
        return dead.get();
    }

    protected void notifyDead() {
        dead.set(true);
    }

    public void purge() {
        positions.clear();
        updates.clear();
        updatesCounter.set(0);
        lastDeletedIndex.set(-1);
        collapsedMode.set(false);
        collapsedIndex.set(-1);
    }
}
