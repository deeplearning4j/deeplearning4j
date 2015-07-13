package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 *
 * RMS Prop updates:
 *
 * http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
 *
 * @author Adam Gibson
 */
public class RmsPropUpdater implements GradientUpdater {
    private INDArray lastGradient;
    private double rmsDecay = 0.5;
    private double lr = 1e-1;


    @Override
    public INDArray getGradient(INDArray gradient) {
        if(lastGradient == null)
            lastGradient = Nd4j.zeros(gradient.shape());
        lastGradient.assign(lastGradient.mul(rmsDecay).addi(Transforms.pow(gradient, 2).muli((1 - rmsDecay))));
        INDArray ret = gradient.mul(lr).negi().divi(Transforms.sqrt(lastGradient.add(Nd4j.EPS_THRESHOLD)));
        return ret;
    }
}
