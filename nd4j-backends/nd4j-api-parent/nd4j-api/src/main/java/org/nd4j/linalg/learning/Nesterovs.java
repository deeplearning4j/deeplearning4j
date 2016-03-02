package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.Serializable;

/**
 * Nesterov's momentum.
 * Keep track of the previous layer's gradient
 * and use it as a way of updating the gradient.
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class Nesterovs implements Serializable,GradientUpdater {
    private double momentum = 0.5;
    private INDArray v;
    private double learningRate = 0.1;

    public Nesterovs(double momentum, double learningRate) {
        this.momentum = momentum;
        this.learningRate = learningRate;
    }

    public Nesterovs(double momentum) {
        this.momentum = momentum;

    }

    @Override
    public void update(Object... args) {
        if(args.length > 0) {
            learningRate = (Double) args[0];
            momentum = (Double) args[1];
        }

    }


    /**
     * Get the nesterov update
     * @param gradient the gradient to get the update for
     * @param iteration
     * @return
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(v == null)
            v = Nd4j.zeros(gradient.shape());
        INDArray vPrev = v;
        v = vPrev.mul(momentum).subi(gradient.mul(learningRate));
        //reference https://cs231n.github.io/neural-networks-3/#sgd 2nd equation
        //DL4J default is negative step function thus we flipped the signs:
        // x += mu * v_prev + (-1 - mu) * v
        //i.e., we do params -= updatedGradient, not params += updatedGradient
        
        INDArray ret = vPrev.muli(momentum).addi(v.mul(-momentum - 1));
        return ret;
    }

    @Override
    public GradientUpdaterAggregator getAggregator(boolean addThis){
        NesterovsAggregator ag = new NesterovsAggregator();
        if(addThis) ag.aggregate(this);
        return ag;
    }

    public static class NesterovsAggregator implements GradientUpdaterAggregator {
        private INDArray vSum;
        private double lrSum;
        private double momentumSum;
        private int count = 0;

        @Override
        public GradientUpdater getUpdater() {
            Nesterovs nesterovs = new Nesterovs(momentumSum/count,lrSum/count);
            nesterovs.setV(vSum.div(count));
            return nesterovs;
        }

        @Override
        public void aggregate(GradientUpdater updater) {
            if(!(updater instanceof Nesterovs)) throw new UnsupportedOperationException("Cannot aggregate Nesterovs with updater: " + updater);
            Nesterovs nesterovs = (Nesterovs)updater;
            if(vSum == null){
                vSum = nesterovs.v.dup();
                lrSum = nesterovs.learningRate;
                momentumSum = nesterovs.momentum;
            } else {
                vSum.addi(nesterovs.v);
                lrSum += nesterovs.learningRate;
                momentumSum += nesterovs.momentum;
            }
            count++;
        }

        @Override
        public GradientUpdaterAggregator combine(GradientUpdaterAggregator other) {
            if(!(other instanceof NesterovsAggregator))
                throw new IllegalArgumentException("Cannot combine NesterovsAggregator with aggregator: " + other);
            NesterovsAggregator aggregator = (NesterovsAggregator)other;
            vSum.addi(aggregator.vSum);
            lrSum += aggregator.lrSum;
            momentumSum += aggregator.momentumSum;
            count += aggregator.count;
            return this;
        }
    }
}
