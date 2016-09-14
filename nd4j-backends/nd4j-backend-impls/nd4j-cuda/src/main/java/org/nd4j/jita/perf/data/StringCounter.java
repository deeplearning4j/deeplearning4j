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

    public StringCounter() {

    }

    public long incrementCount(String key) {
        if (!counter.containsKey(key)) {
            counter.put(key, new AtomicLong(0));
        }

        return counter.get(key).incrementAndGet();
    }

    public long getCount(String key) {
        if (!counter.containsKey(key))
            return 0;

        return counter.get(key).get();
    }
}
