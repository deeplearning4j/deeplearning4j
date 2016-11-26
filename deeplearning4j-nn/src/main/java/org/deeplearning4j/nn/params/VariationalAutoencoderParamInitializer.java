package org.deeplearning4j.nn.params;

import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.variational.VariationalAutoencoder;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Map;

/**
 * Created by Alex on 25/11/2016.
 */
public class VariationalAutoencoderParamInitializer implements ParamInitializer {

    private static final VariationalAutoencoderParamInitializer INSTANCE = new VariationalAutoencoderParamInitializer();
    public static VariationalAutoencoderParamInitializer getInstance(){
        return INSTANCE;
    }

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
        paramCount += (lastEncLayerSize + 1) * nOut;

        //Decoder:
        for( int i=0; i<decoderLayerSizes.length; i++ ){
            int decoderLayerNIn;
            if(i == 0){
                decoderLayerNIn = nOut;
            } else {
                decoderLayerNIn = decoderLayerSizes[i-1];
            }
            paramCount += (decoderLayerNIn + 1) * decoderLayerSizes[i];
        }


        return paramCount;
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
