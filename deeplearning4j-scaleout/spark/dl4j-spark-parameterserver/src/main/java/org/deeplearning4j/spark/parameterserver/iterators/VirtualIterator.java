package org.deeplearning4j.spark.parameterserver.iterators;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Iterator;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

/**
 * This class is thin wrapper, to provide block-until-depleted functionality in multi-threaded environment
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class VirtualIterator<E> extends java.util.Observable implements Iterator<E> {
    // TODO: use AsyncIterator here?
    protected Iterator<E> iterator;
    protected AtomicBoolean state = new AtomicBoolean(true);

    public VirtualIterator(@NonNull Iterator<E> iterator) {
        this.iterator = iterator;
    }


    @Override
    public boolean hasNext() {
        boolean u = iterator.hasNext();
        state.compareAndSet(true, u);
        if (!state.get()) {
            this.setChanged();
            notifyObservers();
        }
        return u;
    }

    @Override
    public E next() {
        return iterator.next();
    }

    @Override
    public void remove() {
        // no-op, we don't need this call implemented
    }

    @Override
    public void forEachRemaining(Consumer<? super E> action) {
        iterator.forEachRemaining(action);
        state.compareAndSet(true, false);
    }

    /**
     * This method blocks until underlying Iterator is depleted
     */
    public void blockUntilDepleted() {
        while (state.get())
            LockSupport.parkNanos(1000L);
    }
}
