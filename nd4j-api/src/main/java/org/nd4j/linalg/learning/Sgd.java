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
    public void combineUpdaters(GradientUpdater... updaters) {
        if(updaters == null || updaters.length == 0) return;
        //Average learning rates: this usually won't be necessary, but might be used in some cases
        //(slightly different schedules, etc). Done mainly for consistency.
        double lrSum = learningRate;
        for(GradientUpdater u : updaters){
            if(!(u instanceof Sgd)) throw new UnsupportedOperationException("Cannot average Sgd updater with other updater: " + u);
            lrSum += ((Sgd)u).learningRate;
        }
        this.learningRate = lrSum / (updaters.length+1);
    }
}
