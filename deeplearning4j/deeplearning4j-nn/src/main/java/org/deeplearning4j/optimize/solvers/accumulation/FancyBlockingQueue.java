/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.optimize.solvers.accumulation;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.deeplearning4j.util.ThreadUtils;

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
    protected AtomicBoolean bypassMode = new AtomicBoolean(false);

    protected boolean isDebug = false;
    protected ReentrantReadWriteLock lock = new ReentrantReadWriteLock();


    public FancyBlockingQueue(@NonNull BlockingQueue<E> queue) {
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
    public void fallbackToSingleConsumerMode(boolean reallyFallback) {
        bypassMode.set(reallyFallback);
    }

    @Override
    public void registerConsumers(int consumers) {
        lock.writeLock().lock();

        this.numElementsReady.set(backingQueue.size());
        this.numElementsDrained.set(0);
        this.consumers = consumers;
        this.currentConsumers.set(consumers);

        lock.writeLock().unlock();
    }

    @Override
    public void put(E e) throws InterruptedException {
        lock.readLock().lock();
        backingQueue.put(e);
        lock.readLock().unlock();
    }

    @Override
    public boolean isEmpty() {
        if (bypassMode.get())
            return backingQueue.isEmpty();


        boolean res = numElementsDrained.get() >= numElementsReady.get();

        if (isDebug)
            log.info("thread {} queries isEmpty: {}", Thread.currentThread().getId(), res);


        return res;
    }

    protected void synchronize(int consumers) {
        if (consumers == 1 || bypassMode.get())
            return;

        if (isDebug)
            log.info("thread {} locking at FBQ", Thread.currentThread().getId());

        // any first thread entering this block - will reset this field to false
        isDone.compareAndSet(true, false);

        // last thread will set isDone to true
        if (barrier.incrementAndGet() == consumers) {
            secondary.set(0);
            barrier.set(0);
            isFirst.set(false);
            isDone.set(true);
        } else {
            // just wait, till last thread will set isDone to true
            while (!isDone.get())
                ThreadUtils.uncheckedSleep(1);
        }

        // second lock here needed only to ensure we won't get overrun over isDone flag
        if (secondary.incrementAndGet() == consumers) {
            isFirst.set(true);
        } else {
            while (!isFirst.get())
                ThreadUtils.uncheckedSleep(1);
        }

        if (isDebug)
            log.info("thread {} unlocking at FBQ", Thread.currentThread().getId());

    }

    @Override
    public E poll() {
        if (bypassMode.get())
            return backingQueue.poll();

        // if that's first step, set local step counter to -1
        if (currentStep.get() == null)
            currentStep.set(new AtomicLong(-1));

        // we block until everyone else step forward
        while (step.get() == currentStep.get().get())
            ThreadUtils.uncheckedSleep(1);

        E object = peek();

        // we wait until all consumers peek() this object from queue
        synchronize(currentConsumers.get());

        currentStep.get().incrementAndGet();


        // last consumer shifts queue on step further
        if (state.incrementAndGet() == currentConsumers.get()) {

            // we're removing current head of queue
            remove();

            numElementsDrained.incrementAndGet();

            // and moving step counter further
            state.set(0);
            step.incrementAndGet();
        }

        // we wait until all consumers know that queue is updated (for isEmpty())
        synchronize(currentConsumers.get());
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
