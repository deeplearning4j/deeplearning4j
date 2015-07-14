package org.nd4j.linalg.learning;

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
public class Nesterovs implements Serializable,GradientUpdater {
    private double momentum = 0.5;
    private INDArray lastGradient;

    public Nesterovs(double momentum) {
        this.momentum = momentum;
    }

    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }



    /**
     * Get the nesterov udpdate
     * @param gradient the gradient to get the update for
     *
     * @param iteration
     * @return
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
      if(lastGradient == null)
          lastGradient = Nd4j.zeros(gradient.shape());
        INDArray ret  = lastGradient.mul(momentum).subi(gradient);
        lastGradient = ret;
        return ret;
    }


}
