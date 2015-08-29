package org.deeplearning4j.nn.conf.layers;

import lombok.*;

/**
 * @author jeffreytang
 */
@Data @NoArgsConstructor
@ToString(callSuper = true)
@EqualsAndHashCode(callSuper = true)
public abstract class FeedForwardLayer extends Layer {
    protected int nIn;
    protected int nOut;
    
    public FeedForwardLayer( Builder builder ){
    	super(builder);
    	this.nIn = builder.nIn;
    	this.nOut = builder.nOut;
    }

    public abstract static class Builder<T extends Builder<T>> extends Layer.Builder<T> {
        protected int nIn = 0;
        protected int nOut = 0;

        public T nIn(int nIn) {
            this.nIn = nIn;
            return (T) this;
        }

        public T nOut(int nOut) {
            this.nOut = nOut;
            return (T) this;
        }
    }
}
