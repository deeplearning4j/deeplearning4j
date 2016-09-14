package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseOutputLayer extends FeedForwardLayer {
    protected ILossFunction lossFn;
    @Deprecated
    protected String customLossFunction;

    protected BaseOutputLayer(Builder builder) {
    	super(builder);
        this.lossFn = builder.lossFn;
        this.customLossFunction = builder.customLossFunction;
    }

    /**
     *
     * @deprecated As of 0.6.0. Use {@link #getLossFn()} instead
     */
    @Deprecated
    public LossFunction getLossFunction(){
        //To maintain backward compatibility only (as much as possible)
        if(lossFn instanceof LossNegativeLogLikelihood) {
            return LossFunction.NEGATIVELOGLIKELIHOOD;
        } else if(lossFn instanceof LossMCXENT){
            return LossFunction.MCXENT;
        } else if(lossFn instanceof LossMSE){
            return LossFunction.MSE;
        } else if(lossFn instanceof LossBinaryXENT) {
            return LossFunction.XENT;
        } else {
            //TODO: are there any others??
            return null;
        }
    }
    
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected ILossFunction lossFn = new LossMCXENT();
        @Deprecated
        protected String customLossFunction;

        public Builder() {}

        public Builder(LossFunction lossFunction) {
            lossFunction(lossFunction);
        }

        public Builder(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
        }

        public T lossFunction(LossFunction lossFunction) {
            return lossFunction(lossFunction.getILossFunction());
        }

        public T lossFunction(ILossFunction lossFunction) {
            this.lossFn = lossFunction;
            return (T)this;
        }


        @Deprecated
        public T customLossFunction(String customLossFunction) {
            this.customLossFunction = customLossFunction;
            return (T)this;
        }
    }
}
