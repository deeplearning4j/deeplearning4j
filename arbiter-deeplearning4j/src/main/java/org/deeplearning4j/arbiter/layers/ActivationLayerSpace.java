package org.deeplearning4j.arbiter.layers;

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.arbiter.util.CollectionUtils;
import org.deeplearning4j.nn.conf.layers.ActivationLayer;

/**
 * Layer space for {@link ActivationLayer}
 *
 * @author Alex Black
 */
@Data
@NoArgsConstructor(access = AccessLevel.PRIVATE) //For Jackson JSON/YAML deserialization
public class ActivationLayerSpace extends FeedForwardLayerSpace<ActivationLayer> {
    protected ActivationLayerSpace(Builder builder) {
        super(builder);

        this.numParameters = CollectionUtils.countUnique(collectLeaves());
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

    @Override
    public String toString() {
        return toString(", ");
    }

    @Override
    public String toString(String delim) {
        return "ActivationLayerSpace(" + super.toString(delim) + ")";
    }
}
