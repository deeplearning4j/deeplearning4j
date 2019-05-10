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

package org.nd4j.linalg.profiler.data.primitives;

import java.util.ArrayList;
import java.util.List;


/**
 * Utility holder class, used to store timing sets retrieved with profiler.
 *
 * @author raver119@gmail.com
 */
public class TimeSet implements Comparable<TimeSet> {
    private List<ComparableAtomicLong> times = new ArrayList<>();
    private long sum = 0;

    public void addTime(long time) {
        times.add(new ComparableAtomicLong(time));
        sum = 0;
    }

    public long getSum() {
        if (sum == 0) {
            for (ComparableAtomicLong time : times) {
                sum += time.get();
            }
        }

        return sum;
    }

    public long getAverage() {
        if (times.size() == 0)
            return 0L;

        long tSum = getSum();
        return tSum / times.size();
    }

    public long getMedian() {
        if (times.size() == 0)
            return 0L;

        return times.get(times.size() / 2).longValue();
    }

    public long getMinimum() {
        long min = Long.MAX_VALUE;
        for (ComparableAtomicLong time : times) {
            if (time.get() < min)
                min = time.get();
        }

        return min;
    }

    public long getMaximum() {
        long max = Long.MIN_VALUE;
        for (ComparableAtomicLong time : times) {
            if (time.get() > max)
                max = time.get();
        }

        return max;
    }

    public int size() {
        return times.size();
    }


    @Override
    public int compareTo(TimeSet o) {
        return Long.compare(o.getSum(), this.getSum());
    }
}
