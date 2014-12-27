package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;

/**
 * Pretrain weight initializer.
 * Has the visible bias as well as hidden and weight matrix.
 *
 * @author Adam Gibson
 */
public class PretrainParamInitializer extends DefaultParamInitializer {
    public final static String VISIBLE_BIAS_KEY = "vb";

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf) {
        super.init(params, conf);
        params.put(VISIBLE_BIAS_KEY, Nd4j.zeros(conf.getnIn()));
    }


}
