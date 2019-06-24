package org.deeplearning4j.rl4j.observation;

import org.apache.commons.collections4.queue.CircularFifoQueue;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class CircularFifoObservationPool implements ObservationPool {

    private final CircularFifoQueue<INDArray> queue;

    public CircularFifoObservationPool(Builder builder) {
        queue = new CircularFifoQueue<>(builder.poolSize);
    }

    public CircularFifoObservationPool(int poolSize) {
        queue = new CircularFifoQueue<>(poolSize);
    }

    @Override
    public void add(INDArray elem) {
        queue.add(elem);
    }

    @Override
    public INDArray[] get() {
        int size = queue.size();
        INDArray[] array = new INDArray[size];
        for (int i = 0; i < size; ++i) {
            array[i] = queue.get(i).castTo(Nd4j.dataType());
        }
        return array;
    }

    public boolean isReady() {
        return queue.isAtFullCapacity();
    }

    public Builder builder(int poolSize) {
        return new Builder(poolSize);
    }

    public static class Builder {
        private final int poolSize;

        public Builder(int poolSize) {

            this.poolSize = poolSize;
        }

        public CircularFifoObservationPool build() {
            return new CircularFifoObservationPool(this);
        }
    }
}
