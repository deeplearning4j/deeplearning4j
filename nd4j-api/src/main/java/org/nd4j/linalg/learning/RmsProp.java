package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * RMS Prop updates:
 *
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 * http://cs231n.github.io/neural-networks-3/#ada
 *
 * @author Adam Gibson
 */
@Data
@NoArgsConstructor
public class RmsProp implements GradientUpdater {
    private INDArray lastGradient;
    private double rmsDecay = 0.95;
    private double learningRate = 1e-1;
    private double epsilon = 1e-8;

    public RmsProp(double learningRate, double rmsDecay){
    	this.learningRate = learningRate;
    	this.rmsDecay = rmsDecay;
    }

    @Override
    public void update(Object... args) {
        if(args.length > 0) {
            learningRate = (Double) args[0];
        }
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(lastGradient == null)
            lastGradient = Nd4j.zeros(gradient.shape());
        lastGradient.muli(rmsDecay).addi(gradient.mul(gradient).muli(1 - rmsDecay));
        // lr * gradient / sqrt(cache + 1e-8)
        INDArray ret = gradient.mul(learningRate).divi(Transforms.sqrt(lastGradient.add(epsilon)));
        
        return ret;
    }

    @Override
    public void combineUpdaters(GradientUpdater... updaters) {
        if(updaters == null || updaters.length == 0) return;
        //Average learning rates & rmsDecay: this usually won't be necessary, but might be used in some cases
        //(slightly different schedules, etc). Done mainly for consistency.
        //And: average historical/stored gradients
        double lrSum = learningRate;
        double rmsDecaySum = rmsDecay;
        for(GradientUpdater u : updaters){
            if(!(u instanceof RmsProp)) throw new UnsupportedOperationException("Cannot combine RmsProp updater with other updater: " + u);
            RmsProp r = (RmsProp)u;
            lrSum += r.learningRate;
            rmsDecaySum += r.rmsDecay;
            lastGradient.addi(r.lastGradient);
        }
        this.learningRate = lrSum / (updaters.length+1);
        this.rmsDecay = rmsDecaySum / (updaters.length+1);
        lastGradient.divi(updaters.length+1);
    }
}
