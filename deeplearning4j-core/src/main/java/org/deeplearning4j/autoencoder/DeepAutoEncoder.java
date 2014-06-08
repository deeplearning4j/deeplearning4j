package org.deeplearning4j.autoencoder;

import java.io.Serializable;

import static org.deeplearning4j.util.MatrixUtil.binomial;
import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.round;
import static org.jblas.MatrixFunctions.abs;

import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.dbn.DBN;
import org.deeplearning4j.nn.BaseMultiLayerNetwork;
import org.deeplearning4j.nn.HiddenLayer;
import org.deeplearning4j.nn.NeuralNetwork;
import org.deeplearning4j.nn.OutputLayer;
import org.deeplearning4j.nn.activation.ActivationFunction;
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
public class DeepAutoEncoder extends BaseMultiLayerNetwork {

    /**
     *
     */
    private static final long serialVersionUID = -3571832097247806784L;
    private BaseMultiLayerNetwork encoder;
    private BaseMultiLayerNetwork decoder;
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.GAUSSIAN;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
    private OutputLayer.LossFunction outputLayerLossFunction = OutputLayer.LossFunction.RMSE_XENT;
    private ActivationFunction outputLayerActivation = Activations.sigmoid();
    private boolean roundCodeLayerInput = false;
    private boolean normalizeCodeLayerOutput = false;

    public DeepAutoEncoder(){}

    public DeepAutoEncoder(BaseMultiLayerNetwork encoder) {
        this.encoder = encoder;
    }

    /**
     * Train the network running some unsupervised
     * pretraining followed by SGD/finetune
     *
     * @param input       the input to train on
     * @param labels      the labels for the training examples(a matrix of the following format:
     *                    [0,1,0] where 0 represents the labels its not and 1 represents labels for the positive outcomes
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void trainNetwork(DoubleMatrix input, DoubleMatrix labels, Object[] otherParams) {
        throw new IllegalStateException("Not implemented");
    }

    /**
     * Pretrain the network with the given parameters
     *
     * @param input       the input to train ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void pretrain(DoubleMatrix input, Object[] otherParams) {
        throw new IllegalStateException("Not implemented");

    }

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.rbm.RBM} for continuous inputs.
     *
     * @param input    the input to the layer
     * @param nVisible the number of visible inputs
     * @param nHidden  the number of hidden units
     * @param W        the weight vector
     * @param hbias    the hidden bias
     * @param vBias    the visible bias
     * @param rng      the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
     * @param index    the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.rbm.RBM}
     */
    @Override
    public NeuralNetwork createLayer(DoubleMatrix input, int nVisible, int nHidden, DoubleMatrix W, DoubleMatrix hbias, DoubleMatrix vBias, RandomGenerator rng, int index) {
        return null;
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[0];
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
                    .withOutputLossFunction(outputLayerLossFunction)
                    .sampleOrActivateByLayer(encoder.getSampleOrActivate())
                    .withHiddenUnitsByLayer(((DBN) encoder).getHiddenUnitByLayer())
                    .withVisibleUnitsByLayer(((DBN) encoder).getVisibleUnitByLayer())
                    .learningRateForLayer(learningRateForLayerReversed)
                    .numberOfInputs(encoder.getHiddenLayerSizes()[encoder.getHiddenLayerSizes().length - 1])
                    .numberOfOutPuts(encoder.getnIns()).withClazz(encoder.getClass())
                    .hiddenLayerSizes(hiddenLayerSizes).renderWeights(encoder.getRenderWeightsEveryNEpochs())
                    .useRegularization(encoder.isUseRegularization()).withDropOut(encoder.getDropOut())
                    .withLossFunction(encoder.getLossFunction()).renderByLayer(encoder.getRenderByLayer())
                    .withOutputActivationFunction(outputLayerActivation)
                    .withSparsity(encoder.getSparsity()).useAdaGrad(encoder.isUseAdaGrad())
                    .withOptimizationAlgorithm(encoder.getOptimizationAlgorithm())
                    .build();

            if(encoder.isForceNumEpochs())
                decoder.setForceNumEpochs(true);



        }
        else {
            decoder = new BaseMultiLayerNetwork.Builder().withClazz(encoder.getClass())
                    .withOutputLossFunction(outputLayerLossFunction)
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


        for(int i = 0; i < cloned.length; i++) {
            cloned[i] = encoder.getLayers()[i].transpose();
            //decoder
            if(i == cloned.length - 1) {
                if(cloned[i] instanceof  RBM) {
                    RBM r = (RBM) cloned[i];
                    r.setHiddenType(hiddenUnit);
                    r.setVisibleType(visibleUnit);
                    cloned[i] = r;
                }
            }

        }
        for(int i = 0; i < cloned.length; i++) {
            clonedHidden[i] = encoder.getSigmoidLayers()[i].transpose();
            clonedHidden[i].setB(cloned[i].gethBias());
        }



        ArrayUtil.reverse(cloned);
        ArrayUtil.reverse(clonedHidden);


        decoder.setSigmoidLayers(clonedHidden);
        decoder.setLayers(cloned);

        DoubleMatrix encoded = encodeWithScaling(input);
        decoder.setInput(encoded);
        decoder.initializeLayers(encoded);

        //copy the params
        update(decoder);



        //set the combined hidden layers and associated sizes,...
        this.hiddenLayerSizes = ArrayUtil.combine(encoder.getHiddenLayerSizes(),decoder.getHiddenLayerSizes());
        this.layers = ArrayUtil.combine(encoder.getLayers(),decoder.getLayers());
        this.sigmoidLayers = ArrayUtil.combine(encoder.getSigmoidLayers(),decoder.getSigmoidLayers());
        this.outputLayer = decoder.getOutputLayer();


    }




    /**
     * Trains the decoder on the given input
     * @param input the given input to train on
     */
    public void finetune(DoubleMatrix input,double lr,int epochs) {
        this.input = input;

        if(decoder == null)
            initDecoder();


        setInput(input);
        setLabels(labels);

        super.finetune(input,lr,epochs);

    }

    /**
     * Encodes with rounding and sigmoid taken in to account
     * @param input the input to encode
     * @return the encoded input scaled and rounded relative to the configuration
     * of the auto encoder
     */
    public DoubleMatrix encodeWithScaling(DoubleMatrix input) {
        DoubleMatrix encode = encode(input);
        //rounding would make these straight probabilities
        if(!isRoundCodeLayerInput() && isNormalizeCodeLayerOutput())
            MatrixUtil.normalizeZeroMeanAndUnitVariance(encode);


        DoubleMatrix decoderInput = isRoundCodeLayerInput() ? round(sigmoid(encode)) :  sigmoid(encode);
        return decoderInput;
    }

    /**
     * Reconstructs the given input by running the input
     * through the auto encoder followed by the decoder
     * @param input the input to reconstruct
     * @return the reconstructed input from input ---> encode ---> decode
     */
    public DoubleMatrix reconstruct(DoubleMatrix input) {
        DoubleMatrix decoderInput = encodeWithScaling(input);

        List<DoubleMatrix> decoderActivations =  decoder.feedForward(decoderInput);
        return decoderActivations.get(decoderActivations.size() - 1);
    }

    public OutputLayer.LossFunction getOutputLayerLossFunction() {
        return outputLayerLossFunction;
    }

    public void setOutputLayerLossFunction(OutputLayer.LossFunction outputLayerLossFunction) {
        this.outputLayerLossFunction = outputLayerLossFunction;
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

    public boolean isRoundCodeLayerInput() {
        return roundCodeLayerInput;
    }

    public void setRoundCodeLayerInput(boolean roundCodeLayerInput) {
        this.roundCodeLayerInput = roundCodeLayerInput;
    }

    public ActivationFunction getOutputLayerActivation() {
        return outputLayerActivation;
    }

    public void setOutputLayerActivation(ActivationFunction outputLayerActivation) {
        this.outputLayerActivation = outputLayerActivation;
    }

    public boolean isNormalizeCodeLayerOutput() {
        return normalizeCodeLayerOutput;
    }

    public void setNormalizeCodeLayerOutput(boolean normalizeCodeLayerOutput) {
        this.normalizeCodeLayerOutput = normalizeCodeLayerOutput;
    }


    public static class Builder extends BaseMultiLayerNetwork.Builder<DeepAutoEncoder> {
        public Builder() {
            clazz = DeepAutoEncoder.class;
        }




    }


}
