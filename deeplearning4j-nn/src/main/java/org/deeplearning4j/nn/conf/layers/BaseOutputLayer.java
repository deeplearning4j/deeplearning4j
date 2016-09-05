package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseOutputLayer extends FeedForwardLayer {
	protected LossFunction lossFunction;
    protected ILossFunction iLossFunction;
    protected String customLossFunction;

    protected BaseOutputLayer(Builder builder) {
    	super(builder);
        this.lossFunction = builder.lossFunction;
        this.customLossFunction = builder.customLossFunction;
        this.iLossFunction = builder.iLossFunction;
    }
    
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected LossFunction lossFunction = LossFunction.RMSE_XENT;
        protected ILossFunction iLossFunction = new LossMCXENT();
        protected String customLossFunction;

        public Builder() {}

        public Builder(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }
        public Builder(ILossFunction lossFunction) {
            this.iLossFunction = lossFunction;
        }

        public T lossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return (T)this;
        }
        public T lossFunction(ILossFunction lossFunction) {
            this.iLossFunction = lossFunction;
            return (T)this;
        }

        public T customLossFunction(String customLossFunction) {
            this.customLossFunction = customLossFunction;
            return (T)this;
        }
    }
}
