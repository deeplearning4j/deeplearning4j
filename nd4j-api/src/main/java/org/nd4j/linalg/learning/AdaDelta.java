package org.nd4j.linalg.learning;

import lombok.Data;
import lombok.NoArgsConstructor;
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
@Data
@NoArgsConstructor
public class AdaDelta implements Serializable,GradientUpdater {
    private INDArray msg;
    private INDArray msdx;
    private double rho = 0.95;


    public AdaDelta(double rho) {
        this.rho = rho;
    }

    @Override
    public void update(Object... args) {
        //no op
    }

    /**
     * Get the updated gradient for the given gradient
     * and also update the state of ada delta.
     * @param gradient the gradient to get the
     *                 updated gradient for
     * @param iteration
     * @return the update gradient
     */
    @Override
    public INDArray getGradient(INDArray gradient, int iteration) {
        if(msg == null)
            msg = Nd4j.zeros(gradient.shape());

        if(msdx == null)
            msdx = Nd4j.zeros(gradient.shape());

        msg.muli(rho);
        msg.addi(1 - rho).muli(gradient.mul(gradient));
        // modifiedGradient = sqrt(modifiedGradient^2)_t-1 / sqrt(avgSquaredRawGradient^2)_t * rawGradient
        INDArray ret = Transforms.sqrt(msdx.add(Nd4j.EPS_THRESHOLD))
        		.divi(Transforms.sqrt(msg.add(Nd4j.EPS_THRESHOLD))).muli(gradient);
        msdx.muli(rho);
        INDArray dxSquared = ret.mul(ret);
        msdx.addi(dxSquared.muli(1 - rho));

        return ret;
    }

    @Override
    public void combineUpdaters(GradientUpdater... updaters) {
        if(updaters == null || updaters.length == 0) return;
        //Average rho, msg, msdx
        double rhoSum = rho;
        for(GradientUpdater u : updaters){
            if(!(u instanceof AdaDelta)) throw new UnsupportedOperationException("Cannot combine AdaDelta updater with other updater: " + u);
            AdaDelta a = (AdaDelta)u;
            rhoSum += a.rho;
            msg.addi(a.msg);
            msdx.addi(a.msdx);
        }
        this.rho = rhoSum / (updaters.length + 1);
        msg.divi(updaters.length + 1);
        msdx.divi(updaters.length + 1);
    }

}
