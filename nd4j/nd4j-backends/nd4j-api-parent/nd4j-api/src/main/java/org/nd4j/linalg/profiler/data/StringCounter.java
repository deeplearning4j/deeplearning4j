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

package org.nd4j.linalg.profiler.data;

import org.nd4j.linalg.profiler.data.primitives.ComparableAtomicLong;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple key-value counter
 *
 * @author raver119@gmail.com
 */
public class StringCounter {
    private Map<String, ComparableAtomicLong> counter = new ConcurrentHashMap<>();
    private AtomicLong totals = new AtomicLong(0);

    public StringCounter() {

    }

    public void reset() {
        for (String key : counter.keySet()) {
            //            counter.remove(key);
            counter.put(key, new ComparableAtomicLong(0));
        }

        totals.set(0);
    }

    public long incrementCount(String key) {
        if (!counter.containsKey(key)) {
            counter.put(key, new ComparableAtomicLong(0));
        }

        ArrayUtil.allUnique(new int[] {});

        totals.incrementAndGet();

        return counter.get(key).incrementAndGet();
    }

    public long getCount(String key) {
        if (!counter.containsKey(key))
            return 0;

        return counter.get(key).get();
    }

    public void totalsIncrement() {
        totals.incrementAndGet();
    }

    public String asString() {
        StringBuilder builder = new StringBuilder();

        Map<String, ComparableAtomicLong> sortedCounter = ArrayUtil.sortMapByValue(counter);

        for (String key : sortedCounter.keySet()) {
            long currentCnt = sortedCounter.get(key).get();
            long totalCnt = totals.get();

            if (totalCnt == 0)
                continue;

            float perc = currentCnt * 100 / totalCnt;

            builder.append(key).append("  >>> [").append(currentCnt).append("]").append(" perc: [").append(perc)
                            .append("]").append("\n");
        }

        return builder.toString();
    }
}
