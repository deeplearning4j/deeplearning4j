package org.nd4j.jita.perf.data;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple key-value counter
 *
 * @author raver119@gmail.com
 */
public class StringCounter {
    private Map<String, AtomicLong> counter = new ConcurrentHashMap<>();
    private AtomicLong totals = new AtomicLong(0);

    public StringCounter() {

    }

    public long incrementCount(String key) {
        if (!counter.containsKey(key)) {
            counter.put(key, new AtomicLong(0));
        }

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

        for (String key: counter.keySet()) {
            long currentCnt = counter.get(key).get();
            long totalCnt = totals.get();
            float perc = currentCnt * 100 / totalCnt;

            builder.append(key).append("  >>> [").append(currentCnt).append("]").append(" perc: [").append(perc).append("]").append("\n");
        }

        return builder.toString();
    }
}
