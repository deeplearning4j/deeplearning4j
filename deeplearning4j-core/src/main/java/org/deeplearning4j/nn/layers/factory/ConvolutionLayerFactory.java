package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;

/**
 * Create a convolution layer
 * @author Adam Gibson
 */
public class ConvolutionLayerFactory extends DefaultLayerFactory {
    public ConvolutionLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }

    @Override
    public ParamInitializer initializer() {
        return new ConvolutionParamInitializer();
    }
}
