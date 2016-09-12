package org.deeplearning4j.arbiter.layers;

import org.deeplearning4j.nn.conf.layers.ActivationLayer;

/**
 * Layer space for {@link ActivationLayer}
 *
 * @author Alex Black
 */
public class ActivationLayerSpace extends FeedForwardLayerSpace<ActivationLayer> {
    protected ActivationLayerSpace(Builder builder) {
        super(builder);
    }

    @Override
    public ActivationLayer getValue(double[] parameterValues) {
        ActivationLayer.Builder b = new ActivationLayer.Builder();
        super.setLayerOptionsBuilder(b, parameterValues);
        return b.build();
    }

    public static class Builder extends FeedForwardLayerSpace.Builder<Builder>{

        @SuppressWarnings("unchecked")
        public ActivationLayerSpace build(){
            return new ActivationLayerSpace(this);
        }
    }
}
