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

    protected BaseOutputLayer(Builder builder) {
    	super(builder);
        this.lossFn = builder.lossFn;
    }
    
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected ILossFunction lossFn = new LossMCXENT();

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
    }
}
