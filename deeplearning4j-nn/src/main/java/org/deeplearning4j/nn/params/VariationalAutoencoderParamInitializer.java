package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.Distributions;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.HashMap;
import java.util.Map;

/**
 * Created by Alex on 25/11/2016.
 */
public class VariationalAutoencoderParamInitializer extends DefaultParamInitializer {

    private static final VariationalAutoencoderParamInitializer INSTANCE = new VariationalAutoencoderParamInitializer();
    public static VariationalAutoencoderParamInitializer getInstance(){
        return INSTANCE;
    }

    public static final String WEIGHT_KEY_SUFFIX = "W";
    public static final String BIAS_KEY_SUFFIX = "b";

    @Override
    public int numParams(NeuralNetConfiguration conf, boolean backprop) {
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        int nIn = layer.getNIn();
        int nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        int paramCount = 0;
        for( int i=0; i<encoderLayerSizes.length; i++ ){
            int encoderLayerIn;
            if(i == 0){
                encoderLayerIn = nIn;
            } else {
                encoderLayerIn = encoderLayerSizes[i-1];
            }
            paramCount += (encoderLayerIn + 1) * encoderLayerSizes[i]; //weights + bias
        }

        //Between the last encoder layer and the parameters for p(z|x):
        int lastEncLayerSize = encoderLayerSizes[encoderLayerSizes.length-1];
        if(backprop){
            paramCount += (lastEncLayerSize + 1) * nOut;        //Just mean parameters used in backprop
        } else {
            paramCount += (lastEncLayerSize + 1) * 2 * nOut;    //Mean and variance parameters used in unsupervised training
        }

        if(!backprop) {
            //Decoder:
            for (int i = 0; i < decoderLayerSizes.length; i++) {
                int decoderLayerNIn;
                if (i == 0) {
                    decoderLayerNIn = nOut;
                } else {
                    decoderLayerNIn = decoderLayerSizes[i - 1];
                }
                paramCount += (decoderLayerNIn + 1) * decoderLayerSizes[i];
            }

            //Between last decoder layer and parameters for p(x|z):
            int nDistributionParams = layer.getOutputDistribution().distributionParamCount(nIn);
            int lastDecLayerSize = decoderLayerSizes[decoderLayerSizes.length - 1];
            paramCount += (lastDecLayerSize + 1) * nDistributionParams;
        }

        return paramCount;
    }

    @Override
    public Map<String, INDArray> init(NeuralNetConfiguration conf, INDArray paramsView, boolean initializeParams) {
        if(paramsView.length() != numParams(conf,true)){
            throw new IllegalArgumentException("Incorrect paramsView length: Expected length " + numParams(conf,true) + ", got length " + paramsView.length());
        }

        Map<String,INDArray> ret = new HashMap<>();
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        int nIn = layer.getNIn();
        int nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        WeightInit weightInit = layer.getWeightInit();
        Distribution dist = Distributions.createDistribution(layer.getDist());

        int soFar = 0;
        for( int i=0; i<encoderLayerSizes.length; i++ ){
            int encoderLayerNIn;
            if(i == 0){
                encoderLayerNIn = nIn;
            } else {
                encoderLayerNIn = encoderLayerSizes[i-1];
            }
            int weightParamCount = encoderLayerNIn * encoderLayerSizes[i];
            INDArray weightView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+weightParamCount));
            soFar += weightParamCount;
            INDArray biasView = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+encoderLayerSizes[i]));
            soFar += encoderLayerSizes[i];

            INDArray layerWeights = createWeightMatrix(encoderLayerNIn, encoderLayerSizes[i], weightInit, dist, weightView, initializeParams);
            INDArray layerBiases = createBias(encoderLayerSizes[i], 0.0, biasView, initializeParams);       //TODO don't hardcode 0

            String sW = "e" + i + WEIGHT_KEY_SUFFIX;
            String sB = "e" + i + BIAS_KEY_SUFFIX;
            ret.put(sW, layerWeights);
            ret.put(sB, layerBiases);

            conf.addVariable(sW);
            conf.addVariable(sB);
        }

        //Last encoder layer -> p(z|x)
        int nWeightsPzx = encoderLayerSizes[encoderLayerSizes.length-1] * nOut;
        INDArray pzxWeightsMean = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasMean = paramsView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar+nOut));

        INDArray pzxWeightsMeanReshaped = createWeightMatrix(encoderLayerSizes[encoderLayerSizes.length-1], nOut, weightInit, dist, pzxWeightsMean, initializeParams);
        INDArray pzxBiasMeanReshaped = createBias(nOut, 0.0, pzxBiasMean, initializeParams);       //TODO don't hardcode 0

        String sW = "eZXMean" + WEIGHT_KEY_SUFFIX;
        String sB = "eZXMean" + BIAS_KEY_SUFFIX;
        ret.put(sW, pzxWeightsMeanReshaped);
        ret.put(sB, pzxBiasMeanReshaped);
        conf.addVariable(sW);
        conf.addVariable(sB);


        //Allocate array for additional pretrain params
        //TODO this won't work with deserialization/model loading
        int pretrainSpecificNumParams = numParams(conf, false) - numParams(conf, true);
        INDArray pretrainParams = Nd4j.createUninitialized(pretrainSpecificNumParams);
        soFar = 0;
        INDArray pzxWeightsLogStdev2 = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasLogStdev2 = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar+nOut));
        soFar += nOut;

        INDArray pzxWeightsLogStdev2Reshaped = createWeightMatrix(encoderLayerSizes[encoderLayerSizes.length-1], nOut, weightInit, dist, pzxWeightsLogStdev2, initializeParams);
        INDArray pzxBiasLogStdev2Reshaped = createBias(nOut, 0.0, pzxBiasLogStdev2, initializeParams);         //TODO don't hardcode 0

        sW = "eZXLogStdev2" + WEIGHT_KEY_SUFFIX;
        sB = "eZXLogStdev2" + BIAS_KEY_SUFFIX;
        ret.put(sW, pzxWeightsLogStdev2Reshaped);
        ret.put(sB, pzxBiasLogStdev2Reshaped);
        conf.addVariable(sW);
        conf.addVariable(sB);

        for( int i=0; i<decoderLayerSizes.length; i++ ){
            int decoderLayerNIn;
            if(i == 0){
                decoderLayerNIn = nOut;
            } else {
                decoderLayerNIn = decoderLayerSizes[i-1];
            }
            int weightParamCount = decoderLayerNIn * decoderLayerSizes[i];
            INDArray weightView = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+weightParamCount));
            soFar += weightParamCount;
            INDArray biasView = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+decoderLayerSizes[i]));
            soFar += decoderLayerSizes[i];

            INDArray layerWeights = createWeightMatrix(decoderLayerNIn, decoderLayerSizes[i], weightInit, dist, weightView, initializeParams);
            INDArray layerBiases = createBias(decoderLayerSizes[i], 0.0, biasView, initializeParams);          //TODO don't hardcode 0

            sW = "d" + i + WEIGHT_KEY_SUFFIX;
            sB = "d" + i + BIAS_KEY_SUFFIX;
            ret.put(sW, layerWeights);
            ret.put(sB, layerBiases);
            conf.addVariable(sW);
            conf.addVariable(sB);
        }

        //Finally, p(x|z):
        int nDistributionParams = layer.getOutputDistribution().distributionParamCount(nIn);
        int pxzWeightCount = decoderLayerSizes[decoderLayerSizes.length-1] * nDistributionParams;
        int pxzBiasCount = nDistributionParams;
        INDArray pxzWeightView = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+pxzWeightCount));
        soFar += pxzWeightCount;
        INDArray pxzBiasView = pretrainParams.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+pxzBiasCount));

        INDArray pxzWeightsReshaped = createWeightMatrix(decoderLayerSizes[decoderLayerSizes.length-1], nDistributionParams, weightInit, dist,  pxzWeightView, initializeParams);
        INDArray pxzBiasReshaped = createBias(nDistributionParams, 0.0, pxzBiasView, initializeParams);       //TODO don't hardcode 0

        sW = "dXZ" + WEIGHT_KEY_SUFFIX;
        sB = "dXZ" + BIAS_KEY_SUFFIX;
        ret.put(sW, pxzWeightsReshaped);
        ret.put(sB, pxzBiasReshaped);
        conf.addVariable(sW);
        conf.addVariable(sB);

        return ret;
    }

    @Override
    public Map<String, INDArray> getGradientsFromFlattened(NeuralNetConfiguration conf, INDArray gradientView) {
        Map<String,INDArray> ret = new HashMap<>();
        VariationalAutoencoder layer = (VariationalAutoencoder) conf.getLayer();

        int nIn = layer.getNIn();
        int nOut = layer.getNOut();
        int[] encoderLayerSizes = layer.getEncoderLayerSizes();
        int[] decoderLayerSizes = layer.getDecoderLayerSizes();

        WeightInit weightInit = layer.getWeightInit();
        Distribution dist = Distributions.createDistribution(layer.getDist());

        int soFar = 0;
        for( int i=0; i<encoderLayerSizes.length; i++ ){
            int encoderLayerNIn;
            if(i == 0){
                encoderLayerNIn = nIn;
            } else {
                encoderLayerNIn = encoderLayerSizes[i-1];
            }
            int weightParamCount = encoderLayerNIn * encoderLayerSizes[i];
            INDArray weightGradView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+weightParamCount));
            soFar += weightParamCount;
            INDArray biasGradView = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar,soFar+encoderLayerSizes[i]));
            soFar += encoderLayerSizes[i];

            INDArray layerWeights = weightGradView.reshape('f',encoderLayerNIn, encoderLayerSizes[i]);
            INDArray layerBiases = biasGradView;    //Arready correct shape (row vector)

            ret.put("e" + i + WEIGHT_KEY_SUFFIX, layerWeights);
            ret.put("e" + i + BIAS_KEY_SUFFIX, layerBiases);
        }

        //Last encoder layer -> p(z|x)
        int nWeightsPzx = encoderLayerSizes[encoderLayerSizes.length-1] * nOut;
        INDArray pzxWeightsMean = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar + nWeightsPzx));
        soFar += nWeightsPzx;
        INDArray pzxBiasMean = gradientView.get(NDArrayIndex.point(0), NDArrayIndex.interval(soFar, soFar+nOut));

        INDArray pzxWeightGradMeanReshaped = pzxWeightsMean.reshape('f', encoderLayerSizes[encoderLayerSizes.length-1], nOut);
        INDArray pzxBiasGradMeanReshaped = pzxBiasMean;  //Already correct shape (row vector)

        ret.put("eZXMean" + WEIGHT_KEY_SUFFIX, pzxWeightGradMeanReshaped);
        ret.put("eZXMean" + BIAS_KEY_SUFFIX, pzxBiasGradMeanReshaped);


        //TODO handle backprop params..

        return ret;
    }
}
