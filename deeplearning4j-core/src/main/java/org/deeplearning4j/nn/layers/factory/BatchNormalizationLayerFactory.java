package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.BatchNormalizationParamInitializer;

/**
 * Created by agibsonccc on 9/27/15.
 */
public class BatchNormalizationLayerFactory extends DefaultLayerFactory {
    public BatchNormalizationLayerFactory(Class<? extends Layer> layerConfig) {
        super(layerConfig);
    }

    @Override
    public ParamInitializer initializer() {
        return new BatchNormalizationParamInitializer();
    }
}
