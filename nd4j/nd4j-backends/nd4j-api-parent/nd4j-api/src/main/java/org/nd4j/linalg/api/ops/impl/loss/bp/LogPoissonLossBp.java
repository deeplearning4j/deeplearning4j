package org.nd4j.linalg.api.ops.impl.loss.bp;

import org.nd4j.autodiff.loss.LossReduce;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

/**
 * Log Poisson loss backprop
 *
 * @author Paul Dubs
 */
public class LogPoissonLossBp extends BaseLossBp {


    private boolean full = false;

    public LogPoissonLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels){
        super(sameDiff, lossReduce, predictions, weights, labels);
    }
    
    public LogPoissonLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, boolean full){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.full = full;
        if(full){
            iArguments.add((long) 1);
        }
    }

    public LogPoissonLossBp(){ }

    @Override
    public String opName() {
        return "log_poisson_loss_grad";
    }


}
