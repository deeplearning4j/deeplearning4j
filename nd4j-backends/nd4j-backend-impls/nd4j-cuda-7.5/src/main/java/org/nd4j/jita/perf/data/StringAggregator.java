package org.nd4j.jita.perf.data;

import org.nd4j.linalg.api.ops.Op;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class StringAggregator {

    private Map<String, List<Long>> times = new HashMap<>();
    private Map<String, AtomicLong> longCalls = new HashMap<>();

    private static final long THRESHOLD = 100000;

    public StringAggregator() {

    }

    public void putTime(String key, Op op, long timeSpent) {
        if (!times.containsKey(key))
            times.put(key, new ArrayList<Long>());

        times.get(key).add(timeSpent);

        if (timeSpent > THRESHOLD) {
            String keyExt = key + " " + op.name() + " (" + op.opNum() + ")";
            if (!longCalls.containsKey(keyExt))
                longCalls.put(keyExt, new AtomicLong(0));

            longCalls.get(keyExt).incrementAndGet();
        }
    }

    public void putTime(String key, long timeSpent) {
        if (!times.containsKey(key))
            times.put(key, new ArrayList<Long>());

        times.get(key).add(timeSpent);
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
                    .append("Average: ").append(currentAvg).append(" ns; ")
                    .append("Median: ").append(currentMed).append(" ns; ");

            builder.append("\n");
        }

        builder.append("\n");

        for (String key: longCalls.keySet()) {
            long numCalls = longCalls.get(key).get();
            builder.append(key).append("  >>> ")
                    .append(numCalls);

            builder.append("\n");
        }
        builder.append("\n");

        return builder.toString();
    }
}
