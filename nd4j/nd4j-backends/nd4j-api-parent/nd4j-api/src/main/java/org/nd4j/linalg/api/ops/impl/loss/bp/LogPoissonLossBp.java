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
        this(sameDiff, lossReduce, predictions, weights, labels, false);
    }
    
    public LogPoissonLossBp(SameDiff sameDiff, LossReduce lossReduce, SDVariable predictions, SDVariable weights, SDVariable labels, boolean full){
        super(sameDiff, lossReduce, predictions, weights, labels);
        this.full = full;
        addArgs();
    }

    public LogPoissonLossBp(){ }


    protected void addArgs(){
       super.addArgs();
        if(full){
            iArguments.add((long) 1);
        }
    }

    @Override
    public String opName() {
        return "log_poisson_loss_grad";
    }


}
