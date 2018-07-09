package org.deeplearning4j.spark.parameterserver.util;

import lombok.AllArgsConstructor;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

@AllArgsConstructor
public class CountingIterator<T> implements Iterator<T> {

    private final Iterator<T> iter;
    private final AtomicInteger counter;

    @Override
    public boolean hasNext() {
        return iter.hasNext();
    }

    @Override
    public T next() {
        counter.getAndIncrement();
        return iter.next();
    }
}
