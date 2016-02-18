package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.EmptyParamInitializer;

/**
 */
public class EmptyFactory extends DefaultLayerFactory {
    public EmptyFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }


    @Override
    public ParamInitializer initializer() {
        return new EmptyParamInitializer();
    }
}
