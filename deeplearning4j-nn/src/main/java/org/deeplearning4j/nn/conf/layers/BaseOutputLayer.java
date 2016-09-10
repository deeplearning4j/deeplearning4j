package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.ToString;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;

@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class BaseOutputLayer extends FeedForwardLayer {
	protected LossFunction lossFunction;
    protected ILossFunction lossFn;
    @Deprecated
    protected String customLossFunction;

    protected BaseOutputLayer(Builder builder) {
    	super(builder);
        this.lossFn = builder.lossFn;
        this.customLossFunction = builder.customLossFunction;
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
            switch(lossFunction){
                case MSE:
                    return lossFunction(new LossMSE());
                case EXPLL:
                    throw new UnsupportedOperationException("Not yet implemented");
                case XENT:
                    return lossFunction(new LossBinaryXENT());
                case MCXENT:
                    return lossFunction(new LossMCXENT());
                case RMSE_XENT:
                    throw new UnsupportedOperationException("Not yet implemented");
                case SQUARED_LOSS:
                    throw new UnsupportedOperationException("Not yet implemented");
                case RECONSTRUCTION_CROSSENTROPY:
                    throw new UnsupportedOperationException("Not yet implemented");
                case NEGATIVELOGLIKELIHOOD:
                    return lossFunction(new LossMCXENT());  //TODO have a separate NLL class??
                case CUSTOM:
                    throw new UnsupportedOperationException("Not yet implemented");
                default:
                    throw new IllegalStateException("Unknown loss function: " + lossFunction);
            }
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
