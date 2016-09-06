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
        this.iLossFunction = builder.ilossFunction;
    }
    
    public static abstract class Builder<T extends Builder<T>> extends FeedForwardLayer.Builder<T> {
        protected ILossFunction lossFunction = new LossMCXENT();
        protected String customLossFunction;

        public Builder() {}

        public Builder(LossFunction lossFunction) {
            /*
                SWITCH STATEMENT
                case MSE:
                    this.lossfunction = new LossMSE()
            */
        }
        public Builder(ILossFunction lossFunction) {
            this.lossFunction = lossFunction;
        }

        public T lossFunction(LossFunction lossFunction) {
            /*
                SWITCH STATEMENT
                case MSE:
                    this.lossfunction = new LossMSE()
            */
        }
        public T lossFunction(ILossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return (T)this;
        }

        //NOTE: Not needed?
        public T customLossFunction(String customLossFunction) {
            this.customLossFunction = customLossFunction;
            return (T)this;
        }
    }
}
