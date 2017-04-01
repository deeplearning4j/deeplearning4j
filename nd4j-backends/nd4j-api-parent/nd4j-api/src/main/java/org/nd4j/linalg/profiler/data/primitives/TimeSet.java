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
