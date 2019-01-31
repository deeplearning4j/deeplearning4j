package org.nd4j.linalg.learning.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * WeightDecay regularization: Updater is not applied to the regularization term gradients, and (optionally) applies the learning rate.
 *
 * For all cases, {@code w -= update}<br>
 * If {@code applyLR == true}, we have:
 * {@code update = updater(gradient) + lr * coeff * w}<br>
 * Where {@code lr} is the learning rate for the current iteration/epoch (accounting for LR schedules if present).<br>
 * <br>
 * If {@code applyLR == false}, we have:<br>
 * {@code update = updater(gradient) + coeff * w}<br>
 *
 * @author Alex Black
 */
public class WeightDecay implements Regularization {

    protected final double coeff;
    protected final boolean applyLR;

    /**
     * @param coeff   Weight decay regularization coefficient
     * @param applyLR If true, multiply the current learning rate. If false, do not multiply by LR.
     */
    public WeightDecay(double coeff, boolean applyLR){
        this.coeff = coeff;
        this.applyLR = applyLR;
    }

    @Override
    public ApplyStep applyStep() {
        return ApplyStep.POST_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr) {
        //L = loss + coeff * 0.5 * sum_i x[i]^2
        //dL/dx[i] = coeff * x[i]
        //update(x[i]) = coeff * x[i] * ( applyLR ? lr : )
        double scale = applyLR ? lr * coeff : coeff;
        Nd4j.getBlasWrapper().level1().axpy(param.length(), scale, param, gradView);        //update += scale * param
    }

    @Override
    public double score(INDArray param) {
        return coeff * param.norm2Number().doubleValue();
    }
}
