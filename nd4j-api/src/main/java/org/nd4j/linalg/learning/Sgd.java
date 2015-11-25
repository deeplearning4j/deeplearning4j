package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class Sgd implements GradientUpdater {
    private double learningRate = 1e-1;

    public Sgd(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(Object... args) {
        if(args.length > 0) {
            learningRate = (Double) args[0];
        }

    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        return gradient.mul(learningRate);
    }

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis){
        SgdAggregator ag = new SgdAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    public static class SgdAggregator implements GradientUpdaterAggregator {
        private double lrSum;
        private int count = 0;

        @Override
        public GradientUpdater getUpdater() {
            return new Sgd(lrSum/count);
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            if(!(updater instanceof Sgd)) throw new UnsupportedOperationException("Cannot aggregate Sgd with updater: " + updater);
            Sgd sgd = (Sgd)updater;
            lrSum += sgd.learningRate;
            count++;
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            if(!(other instanceof SgdAggregator))
                throw new IllegalArgumentException("Cannot combine SgdAggregator with aggregator: " + other);
            SgdAggregator aggregator = (SgdAggregator)other;
            lrSum += aggregator.lrSum;
            count += aggregator.count;
            return this;
        }
    }
}
