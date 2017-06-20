package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This BlockingQueue implementation is suited only for symmetric gradients updates, and should NOT be used anywhere else.
 *
 * Basic idea: all worker threads requesting via poll()/take() method will be advancing only once all consumers get the same element from Queue.
 * So, multiple consumers are guaranteed to be consuming the same elements in the same order served by this queue.
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class FancyBlockingQueue<E> implements BlockingQueue<E>, Registerable {
    protected BlockingQueue<E> backingQueue;
    protected volatile int consumers;

    protected ThreadLocal<AtomicLong> currentStep = new ThreadLocal<>();
    protected final AtomicLong step = new AtomicLong(0);
    protected final AtomicInteger state = new AtomicInteger(0);
    protected final AtomicInteger currentConsumers = new AtomicInteger(0);

    protected AtomicBoolean isFirst = new AtomicBoolean(false);
    protected AtomicBoolean isDone = new AtomicBoolean(true);

    protected AtomicInteger barrier = new AtomicInteger(0);
    protected AtomicInteger secondary = new AtomicInteger(0);

    protected AtomicInteger numElementsReady = new AtomicInteger(0);
    protected AtomicInteger numElementsDrained = new AtomicInteger(0);
    protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock();


    public FancyBlockingQueue(@NonNull BlockingQueue<E> queue){
        this(queue, -1);
    }

    public FancyBlockingQueue(@NonNull BlockingQueue<E> queue, int consumers) {
        this.backingQueue = queue;
        this.consumers = consumers;
        this.currentConsumers.set(consumers);
    }


    @Override
    public boolean add(E e) {
        return backingQueue.add(e);
    }

    @Override
    public boolean offer(E e) {
        return backingQueue.offer(e);
    }


    @Override
    public E remove() {
        return backingQueue.remove();
    }

    @Override
    public void register(int consumers) {
        lock.readLock().lock();

        this.numElementsReady.set(backingQueue.size());
        this.numElementsDrained.set(0);
        this.consumers = consumers;
        this.currentConsumers.set(consumers);

        lock.readLock().unlock();
    }

    @Override
    public void put(E e) throws InterruptedException {
        lock.writeLock().lock();
        backingQueue.put(e);
        lock.writeLock().unlock();
    }

    @Override
    public boolean isEmpty() {
        return numElementsDrained.get() == numElementsReady.get() || backingQueue.isEmpty();
    }

    protected void synchronize(int consumers) {
        // any first thread entering this block - will reset this field to false
        isDone.compareAndSet(true, false);

        // last thread will set isDone to true
        if (barrier.incrementAndGet() == currentConsumers.get()) {
            secondary.set(0);
            barrier.set(0);
            isFirst.set(false);
            isDone.set(true);
        } else {
            // just wait, till last thread will set isDone to true
            while (!isDone.get())
                LockSupport.parkNanos(1000L);
        }

        // second lock here needed only to ensure we won't get overrun over isDone flag
        if (secondary.incrementAndGet() == currentConsumers.get()) {
            isFirst.set(true);
        } else {
            while (!isFirst.get())
                LockSupport.parkNanos(1000L);
        }

    }

    @Override
    public E poll() {
        // if that's first step, set local step counter to -1
        if (currentStep.get() == null)
            currentStep.set(new AtomicLong(-1));

        // we block until everyone else step forward
        while (step.get() == currentStep.get().get())
            LockSupport.parkNanos(1000L);

        E object = peek();

        // we wait until all consumers peek() this object from queue
        synchronize(consumers);

        currentStep.get().incrementAndGet();


        // last consumer shifts queue on step further
        if (state.incrementAndGet() == consumers) {

            // we're removing current head of queue
            remove();

            numElementsDrained.incrementAndGet();

            // and moving step counter further
            state.set(0);
            step.incrementAndGet();
        }

        // we wait until all consumers know that queue is updated (for isEmpty())
        synchronize(consumers);
        //log.info("Second lock passed");

        // now, every consumer in separate threads will get it's own copy of CURRENT head of the queue
        return object;
    }

    @Override
    public E element() {
        return backingQueue.element();
    }

    @Override
    public void clear() {
        backingQueue.clear();
        step.set(0);
    }

    @Override
    public int size() {
        return backingQueue.size();
    }

    @Override
    public E peek() {
        return backingQueue.peek();
    }

    @Override
    public boolean offer(E e, long timeout, TimeUnit unit) throws InterruptedException {
        return backingQueue.offer(e, timeout, unit);
    }

    @Override
    public E take() throws InterruptedException {
        return null;
    }

    @Override
    public E poll(long timeout, TimeUnit unit) throws InterruptedException {
        return backingQueue.poll(timeout, unit);
    }


    @Override
    public int remainingCapacity() {
        return backingQueue.remainingCapacity();
    }

    @Override
    public boolean remove(Object o) {
        return backingQueue.remove(o);
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        return backingQueue.containsAll(c);
    }

    @Override
    public boolean addAll(Collection<? extends E> c) {
        return backingQueue.addAll(c);
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        return backingQueue.removeAll(c);
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        return backingQueue.retainAll(c);
    }

    @Override
    public boolean contains(Object o) {
        return backingQueue.contains(o);
    }


    @Override
    public Iterator<E> iterator() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @Override
    public <T> T[] toArray(T[] a) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int drainTo(Collection<? super E> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int drainTo(Collection<? super E> c, int maxElements) {
        throw new UnsupportedOperationException();
    }
}
