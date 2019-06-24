package org.deeplearning4j.rl4j.observation.transforms;

import org.deeplearning4j.rl4j.observation.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class PoolingTransform extends PassthroughTransform {

    private final ObservationPool observationPool;

    protected PoolingTransform(Builder builder) {
        observationPool = builder.observationPool;
    }

    @Override
    protected Observation handle(Observation input)
    {
        // FIXME: Should this be added to all observables?
        if(input instanceof VoidObservation) {
            return input;
        }

        observationPool.add(input.toNDArray());
        if(!observationPool.isReady()) {
            return VoidObservation.getInstance();
        }

        INDArray[] pooled = observationPool.get();
        INDArray linear = Nd4j.concat(0, pooled);
        //int[] newShape = Learning.makeShape(1, ArrayUtil.toInts(linear.shape()));
        //INDArray result = linear.reshape(newShape);
        return new BasicObservation(linear);
    }

    @Override
    protected boolean getIsReady() {
        return observationPool.isReady();
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Integer poolSize;
        private ObservationPool observationPool;

        public Builder poolSize(int value) {
            poolSize = value;
            return this;
        }

        public Builder observablePool(ObservationPool observationPool) {
            this.observationPool = observationPool;
            return this;
        }

        public PoolingTransform build() {
            if(observationPool == null && poolSize == null) {
                throw new IllegalArgumentException("An ObservationPool or a pool size must be set.");
            }

            if(observationPool == null) {
                observationPool = new CircularFifoObservationPool(poolSize);
            }

            return new PoolingTransform(this);
        }
    }
}
