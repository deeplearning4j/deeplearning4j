package org.deeplearning4j.parallelism;

import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;


import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Limited Queue implementation, suited for multi-gpu prefetch.
 *
 * Basic idea is simple: DataSets are coming from DataSetIterator, and their device location is unknown.
 * So, for better performance DataSets should be transparently moved to the devices where they will be used later, and this should be done in background.
 *
 * @author raver119@gmail.com
 */
public class MagicQueue implements Queue<DataSet> {
    protected final List<Queue<DataSet>> backingQueues;
    protected final AtomicInteger nextBucket = new AtomicInteger(0);
    protected final int numberOfBuckets;
    protected final List<QueueHandler> handlers;

    protected MagicQueue(int numberOfFlows) {
        backingQueues = new ArrayList<>();
        handlers = new ArrayList<>();
        if (numberOfFlows > 1) {
            for (int i = 0; i < numberOfFlows; i++) {
                ConcurrentLinkedQueue<DataSet> queue = new ConcurrentLinkedQueue<>();
                backingQueues.add(queue);

                QueueHandler handler = new QueueHandler(queue);

                Nd4j.getAffinityManager().attachThreadToDevice(handler, i);

                handler.start();
                handlers.add(handler);
            }
        } else {
            ConcurrentLinkedQueue<DataSet> queue = new ConcurrentLinkedQueue<>();
            backingQueues.add(queue);
        }

        numberOfBuckets = numberOfFlows;
    }

    /**
     * This method returns average queue size for all devices
     * @return
     */
    @Override
    public int size() {
        if (numberOfBuckets > 1) {
            long cnt = 0;
            for (int i = 0; i < numberOfBuckets; i++) {
                cnt += backingQueues.get(i).size();
            }

            return (int) Math.floor(cnt / numberOfBuckets);
        } else return backingQueues.get(0).size();
    }

    protected int size(int deviceId) {
        if (deviceId >= backingQueues.size())
            throw new RuntimeException("DeviceID exceeds number of actual backing queues");

        return backingQueues.get(deviceId).size();
    }

    @Override
    public boolean isEmpty() {
        return size() < 1;
    }

    /**
     * This method isn't supported
     * @param o
     * @return
     */
    @Override
    public boolean contains(Object o) {
        throw new UnsupportedOperationException();
    }

    /**
     * This method isn't supported
     * @return
     */
    @Override
    public Iterator<DataSet> iterator() {
        throw new UnsupportedOperationException();
    }

    /**
     * This method isn't supported
     * @return
     */
    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    /**
     * This method isn't supported
     * @param a
     * @param <T>
     * @return
     */
    @Override
    public <T> T[] toArray(T[] a) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(DataSet dataSet) {
        if (numberOfBuckets > 1) {
            synchronized (this) {
                if (nextBucket.get() >= backingQueues.size())
                    nextBucket.set(0);
            }

            handlers.get(nextBucket.getAndIncrement()).put(dataSet);

            return true;
        } else {
            backingQueues.get(0).add(dataSet);
            return true;
        }
    }

    /**
     * This method isn't supported
     * @param o
     * @return
     */
    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    /**
     * This method isn't supported
     * @param c
     * @return
     */
    @Override
    public boolean containsAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean addAll(Collection<? extends DataSet> c) {
        return false;
    }

    /**
     * This method isn't supported
     * @param c
     * @return
     */
    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    /**
     * This method isn't supported
     * @param c
     * @return
     */
    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
        for(Queue<DataSet> queue: backingQueues) {
            queue.clear();
        }
    }

    @Override
    public boolean offer(DataSet dataSet) {
        return false;
    }

    @Override
    public DataSet remove() {
        return null;
    }

    @Override
    public DataSet poll() {
        return null;
    }

    @Override
    public DataSet element() {
        return null;
    }

    @Override
    public DataSet peek() {
        return null;
    }

    public static class Builder {
        private int numberOfBuckets = -1;

        public Builder() {

        }

        public Builder setNumberOfBuckets(int number) {
            this.numberOfBuckets = number;

            return this;
        }

        public MagicQueue build() {
            if (numberOfBuckets < 1)
                numberOfBuckets = Nd4j.getAffinityManager().getNumberOfDevices();

            MagicQueue queue = new MagicQueue(numberOfBuckets);


            return queue;
        }
    }

    private static class QueueHandler extends Thread implements Runnable {
        private final Queue<DataSet> targetQueue;
        private final LinkedBlockingQueue<DataSet> bufferQueue;

        public QueueHandler(Queue<DataSet> queue) {
            this.targetQueue = queue;
            this.bufferQueue = new LinkedBlockingQueue<DataSet>();

            this.setDaemon(true);
        }


        public void put(DataSet dataSet) {
            bufferQueue.add(dataSet);
        }

        @Override
        public void run() {
            while (true) {
                try {
                    DataSet ds = bufferQueue.poll(1, TimeUnit.SECONDS);

                    if (ds != null) {
                        // now we initialize dataset on target device (if applicable)
                        if (ds.getFeaturesMaskArray() != null)
                            Nd4j.getAffinityManager().touch(ds.getFeaturesMaskArray());
                        if (ds.getLabelsMaskArray() != null)
                            Nd4j.getAffinityManager().touch(ds.getLabelsMaskArray());

                        Nd4j.getAffinityManager().touch(ds.getFeatures());
                        Nd4j.getAffinityManager().touch(ds.getLabels());

                        targetQueue.add(ds);
                    }
                } catch (Exception e) {
                    //
                }
            }
        }
    }
}
