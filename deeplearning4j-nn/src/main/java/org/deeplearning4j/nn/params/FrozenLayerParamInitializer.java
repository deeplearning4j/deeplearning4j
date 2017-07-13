package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Parameter initializer for {@link FrozenLayer} instances. Relies on underlying layer's param initializer.
 *
 * @author Alex Black
 */
public class FrozenLayerParamInitializer implements ParamInitializer {

    private static final FrozenLayerParamInitializer INSTANCE = new FrozenLayerParamInitializer();

    public static FrozenLayerParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public int numParams(Layer layer) {
        FrozenLayer fl = (FrozenLayer) layer;
        ParamInitializer initializer = fl.getLayer().initializer();
        return initializer.numParams(fl.getLayer());
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        FrozenLayer fl = (FrozenLayer) conf.getLayer();
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        conf.setLayer(innerLayer);
        Map<String, INDArray> m = initializer.init(conf, paramsView, initializeParams);
        conf.setLayer(fl);

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        FrozenLayer fl = (FrozenLayer) conf.getLayer();
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        conf.setLayer(innerLayer);
        Map<String, INDArray> m = initializer.getGradientsFromFlattened(conf, gradientView);
        conf.setLayer(fl);
        return m;
    }
}
