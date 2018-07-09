package org.deeplearning4j.spark.parameterserver.util;

import lombok.AllArgsConstructor;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * A simple iterator that adds 1 to the specified counter every time next() is called
 *
 * @param <T> Type of iterator
 * @author Alex Black
 */
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
