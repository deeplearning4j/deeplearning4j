package org.nd4j.linalg.profiler.data.primitives;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class ComparableAtomicLong extends AtomicLong implements Comparable<ComparableAtomicLong> {

    public ComparableAtomicLong() {
        super();
    }

    public ComparableAtomicLong(long startingValue) {
        super(startingValue);
    }

    @Override
    public int compareTo(ComparableAtomicLong o) {
        return Long.compare(o.get(), this.get());
    }
}
