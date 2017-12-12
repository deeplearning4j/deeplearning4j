package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.recurrent.SimpleRnn;
import org.deeplearning4j.nn.weights.WeightInitUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;

import java.util.*;

import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

public class SimpleRnnParamInitializer implements ParamInitializer {

    private static final SimpleRnnParamInitializer INSTANCE = new SimpleRnnParamInitializer();

    public static SimpleRnnParamInitializer getInstance(){
        return INSTANCE;
    }

    public static final String WEIGHT_KEY = DefaultParamInitializer.WEIGHT_KEY;
    public static final String RECURRENT_WEIGHT_KEY = "RW";
    public static final String BIAS_KEY = DefaultParamInitializer.BIAS_KEY;

    private static final List<String> PARAM_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY, BIAS_KEY));
    private static final List<String> WEIGHT_KEYS = Collections.unmodifiableList(Arrays.asList(WEIGHT_KEY, RECURRENT_WEIGHT_KEY));
    private static final List<String> BIAS_KEYS = Collections.singletonList(BIAS_KEY);


    @Override
    public int numParams(NeuralNetConfiguration conf) {
        return numParams(conf.getLayer());
    }

    @Override
    public int numParams(Layer layer) {
        SimpleRnn c = (SimpleRnn)layer;
        int nIn = c.getNIn();
        int nOut = c.getNOut();
        return nIn * nOut + nOut * nOut + nOut;
    }

    @Override
    public List<String> paramKeys(Layer layer) {
        return PARAM_KEYS;
    }

    @Override
    public List<String> weightKeys(Layer layer) {
        return WEIGHT_KEYS;
    }

    @Override
    public List<String> biasKeys(Layer layer) {
        return BIAS_KEYS;
    }

    @Override
    public boolean isWeightParam(String key) {
        return WEIGHT_KEY.equals(key) || RECURRENT_WEIGHT_KEY.equals(key);
    }

    @Override
    public boolean isBiasParam(String key) {
        return BIAS_KEY.equals(key);
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        SimpleRnn c = (SimpleRnn)conf.getLayer();
        int nIn = c.getNIn();
        int nOut = c.getNOut();

        Map<String,INDArray> m;

        if(initializeParams){
            Distribution dist = Distributions.createDistribution(c.getDist());

             m = getSubsets(paramsView, nIn, nOut, false);
             INDArray w = WeightInitUtil.initWeights(nIn, nOut, new int[]{nIn, nOut}, c.getWeightInit(), dist, 'c', m.get(WEIGHT_KEY));
             m.put(WEIGHT_KEY, w);

            INDArray rw = WeightInitUtil.initWeights(nOut, nOut, new int[]{nOut, nOut}, c.getWeightInit(), dist, 'c', m.get(WEIGHT_KEY));
            m.put(RECURRENT_WEIGHT_KEY, rw);

        } else {
             m = getSubsets(paramsView, nIn, nOut, true);
        }

        return m;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        SimpleRnn c = (SimpleRnn)conf.getLayer();
        int nIn = c.getNIn();
        int nOut = c.getNOut();

        return getSubsets(gradientView, nIn, nOut, true);
    }

    private static Map<String,INDArray> getSubsets(INDArray in, int nIn, int nOut, boolean reshape){
        int pos = nIn * nOut;
        INDArray w = in.get(point(0), interval(0, pos));
        INDArray rw = in.get(point(0), interval(pos, pos + nOut * nOut));
        pos += nOut * nOut;
        INDArray b = in.get(point(0), interval(pos, pos + nOut));

        if(reshape){
            w = w.reshape('c', nIn, nOut);
            rw = rw.reshape('c', nOut, nOut);
        }

        Map<String,INDArray> m = new LinkedHashMap<>();
        m.put(WEIGHT_KEY, w);
        m.put(RECURRENT_WEIGHT_KEY, rw);
        m.put(BIAS_KEY, b);
        return m;
    }
}
