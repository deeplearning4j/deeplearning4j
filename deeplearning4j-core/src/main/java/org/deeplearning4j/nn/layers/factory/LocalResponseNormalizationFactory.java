package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.params.EmptyParamInitializer;

/**
 * Created by nyghtowl on 10/29/15.
 */
public class LocalResponseNormalizationFactory extends DefaultLayerFactory {
    public LocalResponseNormalizationFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }


    @Override
    public ParamInitializer initializer() {
        return new EmptyParamInitializer();
    }
}
