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
