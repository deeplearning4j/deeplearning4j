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
    private double lr;

    public Nesterovs(double momentum,double lr) {
        this.momentum = momentum;
        this.lr = lr;

    }

    /**
     * Initialize with a learning rate of 0.1
     * @param momentum the momentum to initialize with
     */
    public Nesterovs(double momentum) {
        this(momentum,0.1);

    }
    public double getMomentum() {
        return momentum;
    }

    public void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public double getLr() {
        return lr;
    }

    public void setLr(double lr) {
        this.lr = lr;
    }

    /**
     * Get the nesterov update
     * @param gradient the gradient to get the update for
     * @param iteration
     * @return
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(lastGradient == null)
            lastGradient = Nd4j.zeros(gradient.shape());
        INDArray v = lastGradient.mul(momentum).sub(gradient.mul(lr));
        //reference https://cs231n.github.io/neural-networks-3/#sgd 2nd equation
        //default is negative step function thus we flipped the signs:
        // x += mu * v_prev + (-1 - mu) * v
        INDArray ret = lastGradient.muli(momentum).addi(v.mul(-momentum - 1));
        lastGradient = v;
        return ret;
    }


}
