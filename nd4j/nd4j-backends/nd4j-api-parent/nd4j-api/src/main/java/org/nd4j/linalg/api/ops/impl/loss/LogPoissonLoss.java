package org.nd4j.linalg.api.ops.impl.loss;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

import java.util.Arrays;
import java.util.List;

/**
 * Log Poisson loss
 *
 * Note: This expects that the input/predictions are log(x) not x!
 *
 * @author Paul Dubs
 */
public class LogPoissonLoss extends BaseLoss {
    private boolean full;

    public LogPoissonLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        this(sameDiff, lossReduce, predictions, weights, labels, false);
    }

    public LogPoissonLoss(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, boolean full){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.full = full;
        addArgs();
    }

    public LogPoissonLoss(){ }

    protected void addArgs(){
        super.addArgs();
        if(full){
            iArguments.add((long) 1);
        }
    }

    @Override
    public String opName() {
        return "log_poisson_loss";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> grad){
        //No external gradient
        //Args are: predictions, weights, label

        SDVariable[] grads;
        if(full) {
            grads = f().lossLogPoissonFullBp(arg(2), arg(0), arg(1), lossReduce);
        }else{
            grads = f().lossLogPoissonBp(arg(2), arg(0), arg(1), lossReduce);
        }
        return Arrays.asList(grads);
    }

}
