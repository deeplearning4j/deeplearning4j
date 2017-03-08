package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.Map;

/**
 * @author Adam Gibson
 */
public class EmptyParamInitializer implements ParamInitializer {

    private static final EmptyParamInitializer INSTANCE = new EmptyParamInitializer();

    public static EmptyParamInitializer getInstance() {
        return INSTANCE;
    }

    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return 0;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        return Collections.EMPTY_MAP;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        return Collections.emptyMap();
    }
}
