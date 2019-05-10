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

import org.nd4j.linalg.api.ops.CustomOp;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.profiler.data.primitives.ComparableAtomicLong;
import org.nd4j.linalg.profiler.data.primitives.TimeSet;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class StringAggregator {

    private Map<String, TimeSet> times = new ConcurrentHashMap<>();
    private Map<String, ComparableAtomicLong> longCalls = new ConcurrentHashMap<>();

    private static final long THRESHOLD = 100000;

    public StringAggregator() {

    }

    public void reset() {
        for (String key : times.keySet()) {
            //            times.remove(key);
            times.put(key, new TimeSet());
        }

        for (String key : longCalls.keySet()) {
            //          longCalls.remove(key);
            longCalls.put(key, new ComparableAtomicLong(0));
        }
    }


    public void putTime(String key, Op op, long timeSpent) {
        if (!times.containsKey(key))
            times.put(key, new TimeSet());

        times.get(key).addTime(timeSpent);

        if (timeSpent > THRESHOLD) {
            String keyExt = key + " " + op.opName() + " (" + op.opNum() + ")";
            if (!longCalls.containsKey(keyExt))
                longCalls.put(keyExt, new ComparableAtomicLong(0));

            longCalls.get(keyExt).incrementAndGet();
        }
    }

    public void putTime(String key, CustomOp op, long timeSpent) {
        if (!times.containsKey(key))
            times.put(key, new TimeSet());

        times.get(key).addTime(timeSpent);

        if (timeSpent > THRESHOLD) {
            String keyExt = key + " " + op.opName() + " (" + op.opHash() + ")";
            if (!longCalls.containsKey(keyExt))
                longCalls.put(keyExt, new ComparableAtomicLong(0));

            longCalls.get(keyExt).incrementAndGet();
        }
    }

    public void putTime(String key, long timeSpent) {
        if (!times.containsKey(key))
            times.put(key, new TimeSet());

        times.get(key).addTime(timeSpent);
    }

    protected long getMedian(String key) {
        return times.get(key).getMedian();
    }

    protected long getAverage(String key) {
        return times.get(key).getAverage();
    }

    protected long getMaximum(String key) {
        return times.get(key).getMaximum();
    }

    protected long getMinimum(String key) {
        return times.get(key).getMinimum();
    }

    protected long getSum(String key) {
        return times.get(key).getSum();
    }

    public String asPercentageString() {
        StringBuilder builder = new StringBuilder();

        Map<String, TimeSet> sortedTimes = ArrayUtil.sortMapByValue(times);

        AtomicLong sum = new AtomicLong(0);
        for (String key : sortedTimes.keySet()) {
            sum.addAndGet(getSum(key));
        }
        long lSum = sum.get();
        builder.append("Total time spent: ").append(lSum / 1000000).append(" ms.").append("\n");

        for (String key : sortedTimes.keySet()) {
            long currentSum = getSum(key);
            float perc;
            if (lSum == 0) {
                perc = 0.0f;
            } else {
                perc = currentSum * 100.0f / sum.get();
            }

            long sumMs = currentSum / 1000000;

            builder.append(key).append("  >>> ").append(" perc: ").append(perc).append(" ").append("Time spent: ")
                            .append(sumMs).append(" ms");

            builder.append("\n");
        }

        return builder.toString();
    }

    public String asString() {
        StringBuilder builder = new StringBuilder();

        Map<String, TimeSet> sortedTimes = ArrayUtil.sortMapByValue(times);

        for (String key : sortedTimes.keySet()) {
            long currentMax = getMaximum(key);
            long currentMin = getMinimum(key);
            long currentAvg = getAverage(key);
            long currentMed = getMedian(key);

            builder.append(key).append("  >>> ");

            if (longCalls.size() == 0)
                builder.append(" ").append(sortedTimes.get(key).size()).append(" calls; ");

            builder.append("Min: ").append(currentMin).append(" ns; ").append("Max: ").append(currentMax)
                            .append(" ns; ").append("Average: ").append(currentAvg).append(" ns; ").append("Median: ")
                            .append(currentMed).append(" ns; ");

            builder.append("\n");
        }

        builder.append("\n");

        Map<String, ComparableAtomicLong> sortedCalls = ArrayUtil.sortMapByValue(longCalls);

        for (String key : sortedCalls.keySet()) {
            long numCalls = sortedCalls.get(key).get();
            builder.append(key).append("  >>> ").append(numCalls);

            builder.append("\n");
        }
        builder.append("\n");

        return builder.toString();
    }
}
