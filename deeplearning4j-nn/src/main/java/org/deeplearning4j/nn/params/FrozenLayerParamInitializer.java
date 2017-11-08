package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.misc.FrozenLayer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;
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
    public int numParams(Layer layer) {
        FrozenLayer fl = (FrozenLayer) layer;
        ParamInitializer initializer = fl.getLayer().initializer();
        return initializer.numParams(fl.getLayer());
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return Collections.emptyList();
    }

    @Override
    public boolean isWeightParam(String key) {
        return false;
    }

    @Override
    public boolean isBiasParam(String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(Layer layer, INDArray paramsView, boolean initializeParams) {
        FrozenLayer fl = (FrozenLayer) layer;
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        Map<String, INDArray> m = initializer.init(innerLayer, paramsView, initializeParams);

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(Layer layer, INDArray gradientView) {
        FrozenLayer fl = (FrozenLayer) layer;
        Layer innerLayer = fl.getLayer();
        ParamInitializer initializer = innerLayer.initializer();
        Map<String, INDArray> m = initializer.getGradientsFromFlattened(layer, gradientView);
        return m;
    }
}
