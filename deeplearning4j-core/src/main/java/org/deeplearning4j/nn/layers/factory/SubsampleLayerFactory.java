package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.EmptyParamInitializer;

/**
 * @author Adam Gibson
 */
public class SubsampleLayerFactory extends DefaultLayerFactory {
    public SubsampleLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }

    @Override
    public ParamInitializer initializer() {
        return new EmptyParamInitializer();
    }
}
