package org.deeplearning4j.nn.api;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Param initializer for a layer
 *
 * @author Adam Gibson
 */
public interface ParamInitializer {

    /**
     * Initialize the parameters
     * @param params the parameters to initialize
     * @param conf the configuration
     */
    void init(Map<String, INDArray> params, NeuralNetConfiguration conf);
}
