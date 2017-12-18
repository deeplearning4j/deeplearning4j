package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.List;
import java.util.Map;

public class SameDiffParamInitializer implements ParamInitializer {

    private static final SameDiffParamInitializer INSTANCE = new SameDiffParamInitializer();

    public static SameDiffParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return 0;
    }

    @Override
    public int numParams(Layer layer) {
        return 0;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return null;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return null;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return null;
    }

    @Override
    public boolean isWeightParam(Layer layer, String key) {
        return false;
    }

    @Override
    public boolean isBiasParam(Layer layer, String key) {
        return false;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        return null;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        return null;
    }
}
