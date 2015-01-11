package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.params.RecursiveParamInitializer;

/**
 * Recursive parameter initializer
 * @author Adam Gibson
 */
public class RecursiveAutoEncoderLayerFactory extends DefaultLayerFactory {

    public RecursiveAutoEncoderLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }

    @Override
    public ParamInitializer initializer() {
        return new RecursiveParamInitializer();
    }
}
