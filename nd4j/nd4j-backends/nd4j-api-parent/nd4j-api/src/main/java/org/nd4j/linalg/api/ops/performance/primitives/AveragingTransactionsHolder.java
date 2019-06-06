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

package org.nd4j.linalg.api.ops.performance.primitives;

import lombok.NoArgsConstructor;
import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.nd4j.linalg.memory.MemcpyDirection;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class AveragingTransactionsHolder {
    private final List<List<Long>> storage = new ArrayList<>(MemcpyDirection.values().length);
    private final List<ReentrantReadWriteLock> locks= new ArrayList<>(MemcpyDirection.values().length);

    public AveragingTransactionsHolder() {
        init();
    }

    protected void init() {
        // filling map withi initial keys
        for (val v: MemcpyDirection.values()) {
            val o = v.ordinal();
            storage.add(o, new ArrayList<Long>());

            locks.add(o, new ReentrantReadWriteLock());
        }
    }

    public void clear() {
        for (val v: MemcpyDirection.values()) {
            val o = v.ordinal();
            try {
                locks.get(o).writeLock().lock();

                storage.get(o).clear();
            } finally {
                locks.get(o).writeLock().unlock();
            }
        }
    }


    public void addValue(@NonNull MemcpyDirection direction, Long value) {
        val o = direction.ordinal();
        try {
            locks.get(o).writeLock().lock();

            storage.get(o).add(value);
        } finally {
            locks.get(o).writeLock().unlock();
        }
    }

    public Long getAverageValue(@NonNull MemcpyDirection direction) {
        val o = direction.ordinal();
        try {
            Long r = 0L;
            locks.get(o).readLock().lock();

            val list = storage.get(o);

            if (list.isEmpty())
                return 0L;

            for (val v : list)
                r += v;

            return r / list.size();
        } finally {
            locks.get(o).readLock().unlock();
        }
    }
}
