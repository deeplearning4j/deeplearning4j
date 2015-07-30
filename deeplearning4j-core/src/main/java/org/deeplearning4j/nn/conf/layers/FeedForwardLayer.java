package org.deeplearning4j.nn.conf.layers;

import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Created by jeffreytang on 7/21/15.
 */
@Data
@NoArgsConstructor
public abstract class FeedForwardLayer extends Layer {
    private static final long serialVersionUID = 492217000569721428L;
    protected int nIn;
    protected int nOut;
    
    public FeedForwardLayer( Builder builder ){
    	super(builder);
    	this.nIn = builder.nIn;
    	this.nOut = builder.nOut;
    }

    public abstract static class Builder extends Layer.Builder {
        protected int nIn = Integer.MIN_VALUE;
        protected int nOut = Integer.MIN_VALUE;

        public Builder nIn(int nIn) {
            this.nIn = nIn;
            return this;
        }

        public Builder nOut(int nOut) {
            this.nOut = nOut;
            return this;
        }
    }
}
