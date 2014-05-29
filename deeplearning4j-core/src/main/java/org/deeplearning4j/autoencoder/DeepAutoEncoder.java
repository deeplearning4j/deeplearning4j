package org.deeplearning4j.autoencoder;

import java.io.Serializable;

import static org.deeplearning4j.util.MatrixUtil.binomial;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.round;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.rbm.RBM;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.RBMUtil;
import org.jblas.DoubleMatrix;

/**
 * Encapsulates a deep auto encoder and decoder (the transpose of an encoder)
 *
 * The focus of a deep auto encoder is the code layer.
 * This code layer is the end of the encoder
 * and the input to the decoder.
 *
 * For real valued data, this is a gaussian rectified linear layer.
 *
 * For binary, its binary/binary
 *
 *
 * @author Adam Gibson
 *
 *
 */
public class DeepAutoEncoder implements Serializable {

    /**
     *
     */
    private static final long serialVersionUID = -3571832097247806784L;
    private BaseMultiLayerNetwork encoder;
    private BaseMultiLayerNetwork decoder;
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;

    public DeepAutoEncoder(BaseMultiLayerNetwork encoder) {
        this.encoder = encoder;
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
            //note the gaussian visible unit, we want a GBRBM here for
            //the continuous inputs for the real value codes from the encoder
            Map<Integer,RBM.VisibleUnit> m = Collections.singletonMap(0, visibleUnit);
            Map<Integer,RBM.HiddenUnit> m2 = Collections.singletonMap(0, hiddenUnit);

            Map<Integer,Double> learningRateForLayerReversed = new HashMap<>();
            int count = 0;
            for(int i = encoder.getnLayers() - 1; i >= 0; i--) {
                if(encoder.getLayerLearningRates().get(count) != null) {
                    learningRateForLayerReversed.put(i,encoder.getLayerLearningRates().get(count));
                }
                count++;
            }

            decoder = new DBN.Builder()
                    .withVisibleUnitsByLayer(m)
                    .withHiddenUnitsByLayer(m2)
                    .withHiddenUnits(d.getHiddenUnit())
                    .withVisibleUnits(d.getVisibleUnit())
                    .withVisibleUnits(d.getVisibleUnit())
                    .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                    .learningRateForLayer(learningRateForLayerReversed)
                    .numberOfInputs(encoder.getHiddenLayerSizes()[encoder.getHiddenLayerSizes().length - 1])
                    .numberOfOutPuts(encoder.getnIns()).withClazz(encoder.getClass())
                    .hiddenLayerSizes(hiddenLayerSizes).renderWeights(encoder.getRenderWeightsEveryNEpochs())
                    .useRegularization(encoder.isUseRegularization()).withDropOut(encoder.getDropOut())
                    .withLossFunction(encoder.getLossFunction()).renderByLayer(encoder.getRenderByLayer())
                    .withOutputActivationFunction(Activations.sigmoid())
                    .withSparsity(encoder.getSparsity()).useAdaGrad(encoder.isUseAdaGrad())
                    .withOptimizationAlgorithm(encoder.getOptimizationAlgorithm())
                    .build();


        }
        else {
            decoder = new BaseMultiLayerNetwork.Builder().withClazz(encoder.getClass())
                    .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
                    .activateForLayer(encoder.getActivationFunctionForLayer()).renderByLayer(encoder.getRenderByLayer())
                    .numberOfInputs(encoder.getHiddenLayerSizes()[encoder.getHiddenLayerSizes().length - 1])
                    .numberOfOutPuts(encoder.getnIns()).withClazz(encoder.getClass())
                    .hiddenLayerSizes(hiddenLayerSizes).renderWeights(encoder.getRenderWeightsEveryNEpochs())
                    .useRegularization(encoder.isUseRegularization()).withDropOut(encoder.getDropOut())
                    .withLossFunction(encoder.getLossFunction())
                    .withSparsity(encoder.getSparsity()).useAdaGrad(encoder.isUseAdaGrad())
                    .withOptimizationAlgorithm(encoder.getOptimizationAlgorithm())
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


    }




    /**
     * Trains the decoder on the given input
     * @param input the given input to train on
     */
    public void finetune(DoubleMatrix input,double lr,int epochs) {
        if(decoder == null)
            initDecoder();
        DoubleMatrix encode = encode(input);
        //round the input for the binary codes for input, this is only applicable for the forward layer.
        DoubleMatrix decoderInput = round(sigmoid(encode));
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
        DoubleMatrix encode = encode(input);
        //round the input for the binary codes for input, this is only applicable for the forward layer.
        DoubleMatrix decoderInput = round(encode);
        MatrixUtil.scaleByMax(decoderInput);
        List<DoubleMatrix> decoderActivations =  decoder.feedForward(decoderInput);
        return decoderActivations.get(decoderActivations.size() - 1);
    }

    public RBM.VisibleUnit getVisibleUnit() {
        return visibleUnit;
    }

    public void setVisibleUnit(RBM.VisibleUnit visibleUnit) {
        this.visibleUnit = visibleUnit;
    }

    public RBM.HiddenUnit getHiddenUnit() {
        return hiddenUnit;
    }

    public void setHiddenUnit(RBM.HiddenUnit hiddenUnit) {
        this.hiddenUnit = hiddenUnit;
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
        return encoder.sampleHiddenGivenVisible(input,encoder.getLayers().length);
    }

    public DoubleMatrix decode(DoubleMatrix input) {
        return decoder.output(input);
    }



}
