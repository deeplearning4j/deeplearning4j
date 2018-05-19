package org.deeplearning4j.nn.layers.samediff.testlayers;

import lombok.Data;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.samediff.BaseSameDiffLayer;
import org.deeplearning4j.nn.conf.layers.samediff.SDLayerParams;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Data
public class MinimalSameDiffDense extends BaseSameDiffLayer {

    private int nIn;
    private int nOut;
    private Activation activation;

    public MinimalSameDiffDense(int nIn, int nOut, Activation activation, WeightInit weightInit){
        this.nIn = nIn;
        this.nOut = nOut;
        this.activation = activation;
        this.weightInit = weightInit;
    }

    protected MinimalSameDiffDense(){
        //For JSON serialization
    }

    @Override
    public List<SDVariable> defineLayer(SameDiff sd, SDVariable layerInput, Map<String, SDVariable> paramTable) {
        SDVariable weights = paramTable.get(DefaultParamInitializer.WEIGHT_KEY);
        SDVariable bias = paramTable.get(DefaultParamInitializer.BIAS_KEY);

        SDVariable mmul = sd.mmul("mmul", layerInput, weights);
        SDVariable z = mmul.add("z", bias);
        return Collections.singletonList(activation.asSameDiff("out", sd, z));
    }

    @Override
    public InputType getOutputType(int layerIndex, InputType inputType) {
        return InputType.feedForward(nOut);
    }

    @Override
    public void defineParameters(SDLayerParams params) {
        params.addWeightParam(DefaultParamInitializer.WEIGHT_KEY, new long[]{nIn, nOut});
        params.addBiasParam(DefaultParamInitializer.BIAS_KEY, new long[]{1, nOut});
    }

    @Override
    public void initializeParameters(Map<String, INDArray> params) {
        params.get(DefaultParamInitializer.BIAS_KEY).assign(0);
        initWeights(nIn, nOut, weightInit, params.get(DefaultParamInitializer.WEIGHT_KEY));
    }

    //OPTIONAL methods:
//    public void setNIn(InputType inputType, boolean override)
//    public InputPreProcessor getPreProcessorForInputType(InputType inputType)
//    public void applyGlobalConfigToLayer(NeuralNetConfiguration.Builder globalConfig)
}
