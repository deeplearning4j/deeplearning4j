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

package org.nd4j.jita.allocator.time.rings;

import org.nd4j.jita.allocator.time.Ring;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
public class LockedRing implements Ring {

    private final float[] ring;
    private final AtomicInteger position = new AtomicInteger(0);

    private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /**
     * Builds new BasicRing with specified number of elements stored
     *
     * @param ringLength
     */
    public LockedRing(int ringLength) {
        this.ring = new float[ringLength];
    }

    public float getAverage() {
        try {
            lock.readLock().lock();

            float rates = 0.0f;
            int x = 0;
            int existing = 0;
            for (x = 0; x < ring.length; x++) {
                rates += ring[x];
                if (ring[x] > 0) {
                    existing++;
                }
            }
            if (existing > 0) {
                return rates / existing;
            } else {
                return 0.0f;
            }
        } finally {
            lock.readLock().unlock();
        }
    }

    public void store(double rate) {
        store((float) rate);
    }

    public void store(float rate) {
        try {
            lock.writeLock().lock();

            int pos = position.getAndIncrement();
            if (pos >= ring.length) {
                pos = 0;
                position.set(0);
            }
            ring[pos] = rate;
        } finally {
            lock.writeLock().unlock();
        }
    }
}
