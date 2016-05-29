package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Batch normalization variable init
 */

public class BatchNormalizationParamInitializer implements ParamInitializer {
    public final static String GAMMA = "gamma";
    public final static String BETA = "beta";

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop){
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        return 2*layer.getNOut();
    }

    @Override
    public void init(Map<String, INDArray> params, NeuralNetConfiguration conf, INDArray paramView) {
        // gamma & beta per activation for DNN and per per feature matrix for CNN layers
        // TODO setup for CNN & RNN
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        int nOut = layer.getNOut();

        INDArray gammaView = paramView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nOut));
        INDArray betaView = paramView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut,2*nOut));

        params.put(GAMMA,createGamma(conf, gammaView));
        conf.addVariable(GAMMA);
        params.put(BETA, createBeta(conf, betaView));
        conf.addVariable(BETA);
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        int nOut = layer.getNOut();

        INDArray gammaView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(0,nOut));
        INDArray betaView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(nOut,2*nOut));

        Map<String,INDArray> out = new LinkedHashMap<>();
        out.put(GAMMA, gammaView);
        out.put(BETA, betaView);

        return out;
    }

    protected INDArray createBeta(NeuralNetConfiguration conf, INDArray betaView) {
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        betaView.assign(layer.getBeta());
        return betaView;
    }

    protected INDArray createGamma(NeuralNetConfiguration conf, INDArray gammaView) {
        BatchNormalization layer = (BatchNormalization) conf.getLayer();
        gammaView.assign(layer.getGamma());
        return gammaView;
    }

}
