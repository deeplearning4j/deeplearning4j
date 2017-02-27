package org.nd4j.linalg.learning;

import lombok.EqualsAndHashCode;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * Created by Alex on 08/09/2016.
 */
public class NoOpUpdater implements GradientUpdater {
    @Override
    public int stateSizeForInputSize(int inputSize) {
        return 0;
    }

    @Override
    public void setStateViewArray(INDArray viewArray, int[] shape, char order, boolean initialize) {
        //No op
    }

    @Override
    public void update(Object... args) {
        //No op
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient;
    }

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis) {
        return new NoOpUpdaterAggregator();
    }

    @EqualsAndHashCode
    private static class NoOpUpdaterAggregator implements GradientUpdaterAggregator {
        @Override
        public GradientUpdater getUpdater() {
            return new NoOpUpdater();
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            //No op
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            return this;
        }
    }
}
