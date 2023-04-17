package org.nd4j.profiler;

import lombok.Getter;
import org.nd4j.common.primitives.Counter;

public class MemoryCounter {

    @Getter
    private static Counter<String> allocated = new Counter<>();

    private static Counter<String> instanceCounts = new Counter<>();



    public static void increment(String name, long size) {
        allocated.incrementCount(name, size);
        instanceCounts.incrementCount(name, 1);
    }

    public static void decrement(String name, long size) {
        allocated.incrementCount(name, -size);
        instanceCounts.incrementCount(name, 1);
    }

}
