package org.deeplearning4j.nn.conf.override;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Configuration override
 * @author Adam Gibson
 */
public class ComposableOverride implements ConfOverride {
    private ConfOverride[] overrides;

    public ComposableOverride(ConfOverride...overrides) {
        this.overrides = overrides;
    }

    @Override
    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
        for(ConfOverride override : overrides)
            override.overrideLayer(i,builder);
    }
}
