package org.nd4j.jita.perf.data;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class StringAggregator {

    private Map<String, List<Long>> times = new HashMap<>();

    public StringAggregator() {

    }

    public void putTime(String key, long startTime) {
        long currTime = System.nanoTime();
        if (!times.containsKey(key))
            times.put(key, new ArrayList<Long>());

        times.get(key).add(currTime - startTime);
    }

    protected long getMedian(String key) {
        List<Long> values = times.get(key);
        Collections.sort(values);

        return values.get(values.size() / 2);
    }

    protected long getAverage(String key) {
        List<Long> values = times.get(key);
        AtomicLong cnt = new AtomicLong(0);
        for (Long value: values) {
            cnt.addAndGet(value);
        }

        return cnt.get() / values.size();
    }

    protected long getMaximum(String key) {
        List<Long> values = times.get(key);
        AtomicLong cnt = new AtomicLong(0);
        for (Long value: values) {
            if (value > cnt.get())
                cnt.set(value);
        }

        return cnt.get();
    }

    protected long getMinimum(String key) {
        List<Long> values = times.get(key);
        AtomicLong cnt = new AtomicLong(Long.MAX_VALUE - 1);
        for (Long value: values) {
            if (value < cnt.get())
                cnt.set(value);
        }

        return cnt.get();
    }

    public String asString() {
        StringBuilder builder = new StringBuilder();

        for (String key: times.keySet()) {
            long currentMax = getMaximum(key);
            long currentMin = getMinimum(key);
            long currentAvg = getAverage(key);
            long currentMed = getMedian(key);

            builder.append(key).append("  >>> ")
                    .append("Min: ").append(currentMin).append(" ns; ")
                    .append("Max: ").append(currentMax).append(" ns; ")
                    .append("Avg: ").append(currentAvg).append(" ns; ")
                    .append("Med: ").append(currentMed).append(" ns; ");

            builder.append("\n");
        }

        return builder.toString();
    }
}
