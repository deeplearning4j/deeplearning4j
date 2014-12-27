package org.deeplearning4j.nn.layers.factory;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;

/**
 * Used for creating pretrain neural net layers
 * @author Adam Gibson
 */
public class PretrainLayerFactory extends DefaultLayerFactory {


    public PretrainLayerFactory(Class<? extends Layer> layerClazz) {
        super(layerClazz);
    }



    @Override
    public ParamInitializer getInitializer() {
        return new PretrainParamInitializer();
    }
}
