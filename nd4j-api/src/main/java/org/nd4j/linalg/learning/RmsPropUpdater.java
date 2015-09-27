package org.nd4j.linalg.learning;

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
@NoArgsConstructor
public class RmsPropUpdater implements GradientUpdater {
    private INDArray lastGradient;
    private double rmsDecay = 0.5;
    private double lr = 1e-1;

    public RmsPropUpdater(double lr, double rmsDecay){
    	this.lr = lr;
    	this.rmsDecay = rmsDecay;
    }

    public void setRmsDecay(double rmsDecay){
    	this.rmsDecay = rmsDecay;
    }

    public double getRmsDecay(){
    	return rmsDecay;
    }

    public void setLR( double lr ){
    	this.lr = lr;
    }

    public double getLR(){
    	return lr;
    }

    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(lastGradient == null)
            lastGradient = Nd4j.zeros(gradient.shape());
        lastGradient.muli(rmsDecay).addi(gradient.mul(gradient).muli(1 - rmsDecay));
        // lr * gradient / sqrt(cache + 1e-8)
        INDArray ret = gradient.mul(lr).divi(Transforms.sqrt(lastGradient.add(Nd4j.EPS_THRESHOLD)));
        
        return ret;
    }
}
