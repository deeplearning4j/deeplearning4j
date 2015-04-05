package org.deeplearning4j.nn.conf.override;

import org.deeplearning4j.nn.api.LayerFactory;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;

/**
 * Layer factory override
 * @author Adam Gibson
 */
public class LayerFactoryOverride implements ConfOverride {
    private int layer;
    private LayerFactory layerFactory;

    public LayerFactoryOverride(int layer, LayerFactory layerFactory) {
        this.layer = layer;
        this.layerFactory = layerFactory;
    }

    @Override
    public void overrideLayer(int i, NeuralNetConfiguration.Builder builder) {
        if(i == layer)
            builder.layerFactory(layerFactory);
    }
}
