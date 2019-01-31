package org.nd4j.linalg.learning.regularization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;

/**
 * L2 regularization: very similar to weight decay, but is applied before the updater is applied, not after.
 * <br>
 * <br>
 * Implements updating as follows:<br>
 * {@code w -= updater(gradient + l2 * w}<br>
 */
public class L2Regularization implements Regularization {

    protected final ISchedule l2;

    /**
     * @param l2   L2 regularization coefficient
     */
    public L2Regularization(double l2) {
        this(new FixedSchedule(l2));
    }

    public L2Regularization(ISchedule l2) {
        this.l2 = l2;
    }

    @Override
    public ApplyStep applyStep(){
        return ApplyStep.BEFORE_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch) {
        //L = loss + l2 * 0.5 * sum_i x[i]^2
        //dL/dx[i] = dloss/dx[i] + l2 * x[i]
        double coeff = l2.valueAt(iteration, epoch);
        Nd4j.getBlasWrapper().level1().axpy(param.length(), coeff, param, gradView);        //Gradient += scale * param
    }

    @Override
    public double score(INDArray param, int iteration, int epoch) {
        return l2.valueAt(iteration, epoch) * param.norm2Number().doubleValue();
    }
}
