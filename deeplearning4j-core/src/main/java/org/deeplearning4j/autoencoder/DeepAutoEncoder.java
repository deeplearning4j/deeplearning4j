package org.deeplearning4j.autoencoder;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.ArrayUtil;
import org.jblas.DoubleMatrix;

/**
 * Encapsulates a deep auto encoder and decoder (the transpose of an encoder)
 */
public class DeepAutoEncoder implements Serializable {

    /**
     *
     */
    private static final long serialVersionUID = -3571832097247806784L;
    private BaseMultiLayerNetwork encoder;
    private BaseMultiLayerNetwork decoder;
    private Object[] trainingParams;

    public DeepAutoEncoder(BaseMultiLayerNetwork encoder, Object[] trainingParams) {
        this.encoder = encoder;
        this.trainingParams = trainingParams;
    }


    public void train(DoubleMatrix input, double lr,int n) {
        //final hidden layer needs to be linear activation
        encoder.pretrain(input,trainingParams);
        encoder.getSigmoidLayers()[encoder.getSigmoidLayers().length - 1].setActivationFunction(Activations.linear());
        //initialize the decoder from the newly trained weights
        initDecoder();




    }


    private void initDecoder() {
        //encoder hasn't been pretrained yet
        if(encoder.getLayers() == null || encoder.getSigmoidLayers() == null)
            return;

        int[] hiddenLayerSizes = new int[encoder.getHiddenLayerSizes().length - 1];
        System.arraycopy(encoder.getHiddenLayerSizes(),0,hiddenLayerSizes,0,hiddenLayerSizes.length);
        ArrayUtil.reverse(hiddenLayerSizes);

        if (encoder.getClass().isAssignableFrom(DBN.class)) {
            DBN d = (DBN) encoder;
            //note the gaussian visible unit, we want a GBRBM here for the continuous inputs for the real value codes from the encoder
            decoder = new DBN.Builder().withHiddenUnits(d.getHiddenUnit()).withVisibleUnits(RBM.VisibleUnit.GAUSSIAN).withOutputLossFunction(OutputLayer.LossFunction.XENT)
                    .numberOfInputs(encoder.getHiddenLayerSizes()[encoder.getHiddenLayerSizes().length - 1]).numberOfOutPuts(encoder.getnIns()).withClazz(encoder.getClass())
                    .hiddenLayerSizes(hiddenLayerSizes).renderWeights(encoder.getRenderWeightsEveryNEpochs())
                    .useRegularization(encoder.isUseRegularization()).withDropOut(encoder.getDropOut()).withLossFunction(encoder.getLossFunction())
                    .withSparsity(encoder.getSparsity()).useAdaGrad(encoder.isUseAdaGrad()).withOptimizationAlgorithm(encoder.getOptimizationAlgorithm())
                    .build();


        }
        else {
            decoder = new BaseMultiLayerNetwork.Builder().withClazz(encoder.getClass())
                    .withOutputLossFunction(OutputLayer.LossFunction.XENT)
                    .activateForLayer(encoder.getActivationFunctionForLayer())
                    .numberOfInputs(encoder.getHiddenLayerSizes()[encoder.getHiddenLayerSizes().length - 1]).numberOfOutPuts(encoder.getnIns()).withClazz(encoder.getClass())
                    .hiddenLayerSizes(hiddenLayerSizes).renderWeights(encoder.getRenderWeightsEveryNEpochs())
                    .useRegularization(encoder.isUseRegularization()).withDropOut(encoder.getDropOut()).withLossFunction(encoder.getLossFunction())
                    .withSparsity(encoder.getSparsity()).useAdaGrad(encoder.isUseAdaGrad()).withOptimizationAlgorithm(encoder.getOptimizationAlgorithm())
                    .build();


        }




        NeuralNetwork[] cloned = new NeuralNetwork[encoder.getnLayers()];
        HiddenLayer[] clonedHidden = new HiddenLayer[encoder.getnLayers()];


        for(int i = 0; i < cloned.length; i++)
            cloned[i] = encoder.getLayers()[i].transpose();

        for(int i = 0; i < cloned.length; i++) {
            clonedHidden[i] = encoder.getSigmoidLayers()[i].transpose();
            clonedHidden[i].setB(cloned[i].gethBias());
        }



        ArrayUtil.reverse(cloned);
        ArrayUtil.reverse(clonedHidden);






        decoder.setSigmoidLayers(clonedHidden);
        decoder.setLayers(cloned);


        //weights on the first layer are n times bigger

    }

    /**
     * Trains the decoder on the given input
     * @param input the given input to train on
     */
    public void finetune(DoubleMatrix input,double lr,int epochs) {
        List<DoubleMatrix> activations = encoder.feedForward(input);
        if(decoder == null)
           initDecoder();
        DoubleMatrix decoderInput = activations.get(activations.size() - 2);

        decoder.setInput(decoderInput);
        decoder.initializeLayers(decoderInput);
        decoder.finetune(input,lr,epochs);

    }


    /**
     * Reconstructs the given input by running the input
     * through the auto encoder followed by the decoder
     * @param input the input to reconstruct
     * @return the reconstructed input from input ---> encode ---> decode
     */
    public DoubleMatrix reconstruct(DoubleMatrix input) {
        List<DoubleMatrix> activations = encoder.feedForward(input);

        DoubleMatrix decoderInput = activations.get(activations.size() - 2);
        List<DoubleMatrix> decoderActivations =  decoder.feedForward(decoderInput);
        return decoderActivations.get(decoderActivations.size() - 1);
    }



    public BaseMultiLayerNetwork getEncoder() {
        return encoder;
    }

    public void setEncoder(BaseMultiLayerNetwork encoder) {
        this.encoder = encoder;
    }

    public BaseMultiLayerNetwork getDecoder() {
        return decoder;
    }

    public void setDecoder(BaseMultiLayerNetwork decoder) {
        this.decoder = decoder;
    }

    public DoubleMatrix encode(DoubleMatrix input) {
        return encoder.output(input);
    }

    public DoubleMatrix decode(DoubleMatrix input) {
        return decoder.output(input);
    }



}
