package org.nd4j.profiler;

import org.nd4j.common.primitives.Counter;

public class MemoryCounter {

    private static Counter<String> allocated = new Counter<>();



    public static void increment(String name, long size) {
        allocated.incrementCount(name, size);
    }

    public static void decrement(String name, long size) {
        allocated.incrementCount(name, -size);
    }

}
