package org.nd4j.linalg.learning;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.Serializable;

/**
 * http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
 *
 * Ada delta updater. More robust adagrad that keeps track of a moving window
 * average of the gradient rather than the every decaying learning rates of adagrad
 *
 * @author Adam Gibson
 */
public class AdaDelta implements Serializable,GradientUpdater {
    private INDArray msg;
    private INDArray msdx;
    private double rho = 0.95;

    public AdaDelta(double rho) {
        this.rho = rho;
    }

    public AdaDelta() {
        this.rho = 0.95;
    }

    public INDArray getMsg() {
        return msg;
    }

    public void setMsg(INDArray msg) {
        this.msg = msg;
    }

    public INDArray getMsdx() {
        return msdx;
    }

    public void setMsdx(INDArray msdx) {
        this.msdx = msdx;
    }

    public double getRho() {
        return rho;
    }

    public void setRho(double rho) {
        this.rho = rho;
    }

    /**
     * Get the updated gradient for the given gradient
     * and also update the state of ada delta.
     * @param gradient the gradient to get the
     *                 updated gradient for
     * @return the update gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient) {
        if(msg == null)
            msg = Nd4j.zeros(gradient.shape());

        if(msdx == null)
            msdx = Nd4j.zeros(gradient.shape());

        msg.muli(rho);
        msg.addi(1 - rho).muli(gradient.mul(gradient));
        INDArray ret = Transforms.sqrt(msdx.add(Nd4j.EPS_THRESHOLD).divi(msg.addi(Nd4j.EPS_THRESHOLD))).muli(gradient);
        msdx.muli(rho);
        msdx.addi(1 - rho).muli(ret).muli(ret);

        return ret;
    }


}
