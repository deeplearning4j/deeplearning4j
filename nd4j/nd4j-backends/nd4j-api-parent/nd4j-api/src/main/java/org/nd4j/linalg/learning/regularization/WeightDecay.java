package org.nd4j.linalg.learning.regularization;

import lombok.Data;
import lombok.NonNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.schedule.FixedSchedule;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.shade.jackson.annotation.JsonProperty;

/**
 * WeightDecay regularization: Updater is not applied to the regularization term gradients, and (optionally) applies the learning rate.
 *
 * Loss: {@code L = loss + coeff * 0.5 * sum_i w[i]^2}<br>
 *
 * For all cases, {@code w -= update}<br>
 * If {@code applyLR == true}, we have:
 * {@code update = updater(gradient) + lr * coeff * w}<br>
 * Where {@code lr} is the learning rate for the current iteration/epoch (accounting for LR schedules if present).<br>
 * <br>
 * If {@code applyLR == false}, we have:<br>
 * {@code update = updater(gradient) + coeff * w}<br>
 *
 * <br>
 * Similar to L2 regularization, but WeightDecay should usually be preferred in practice.
 * See <a href="https://www.fast.ai/2018/07/02/adam-weight-decay/">https://www.fast.ai/2018/07/02/adam-weight-decay/</a>
 * for further details.
 *
 * @author Alex Black
 */
@Data
public class WeightDecay implements Regularization {

    protected final ISchedule coeff;
    protected final boolean applyLR;

    /**
     * @param coeff   Weight decay regularization coefficient
     * @param applyLR If true, multiply the current learning rate. If false, do not multiply by LR.
     */
    public WeightDecay(double coeff, boolean applyLR) {
        this(new FixedSchedule(coeff), applyLR);
    }

    public WeightDecay(@JsonProperty("coeff") @NonNull ISchedule coeff, @JsonProperty("applyLR") boolean applyLR){
        this.coeff = coeff;
        this.applyLR = applyLR;
    }

    @Override
    public ApplyStep applyStep() {
        return ApplyStep.POST_UPDATER;
    }

    @Override
    public void apply(INDArray param, INDArray gradView, double lr, int iteration, int epoch) {
        //L = loss + coeff * 0.5 * sum_i x[i]^2
        //dL/dx[i] = coeff * x[i]
        //update(x[i]) = coeff * x[i] * ( applyLR ? lr : )
        double scale = applyLR ? lr * coeff.valueAt(iteration, epoch) : coeff.valueAt(iteration, epoch);
        Nd4j.getBlasWrapper().level1().axpy(param.length(), scale, param, gradView);        //update += scale * param
    }

    @Override
    public double score(INDArray param, int iteration, int epoch) {
        //Score: L = 0.5 * sum_i x[i]^2
        double norm2 = param.norm2Number().doubleValue();   //Norm2 is sqrt(sum_i x[i]^2)
        return coeff.valueAt(iteration, epoch) * 0.5 * norm2 * norm2;
    }

    @Override
    public Regularization clone() {
        return new WeightDecay(coeff.clone(), applyLR);
    }
}
