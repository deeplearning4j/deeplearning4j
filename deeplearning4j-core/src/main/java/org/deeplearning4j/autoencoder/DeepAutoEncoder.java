package org.deeplearning4j.autoencoder;


import static org.deeplearning4j.util.MatrixUtil.sigmoid;
import static org.deeplearning4j.util.MatrixUtil.round;

import java.util.*;

import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
 * A few notes from the science 2006 paper:
 * On decode, use straight activations
 * On encode, use sampling from activations
 *
 * The decoder is the transpose of the output layer.
 *
 * Back prop happens twice. Once on the encoder,
 * once on the decoder (globally)
 *
 * Both time should use a loss function that simulates reconstructions:
 * that could be RMSE_XENT or SQUARED_LOSS
 *
 * The code layer should always be gaussian.
 *
 * If the goal is binary codes, the output layer's activation function should be sigmoid.
 *
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
    private RBM.VisibleUnit visibleUnit = RBM.VisibleUnit.BINARY;
    private RBM.HiddenUnit hiddenUnit = RBM.HiddenUnit.BINARY;
    //linear code layer
    private ActivationFunction codeLayerAct = Activations.linear();
    //reconstruction error
    private OutputLayer.LossFunction outputLayerLossFunction = OutputLayer.LossFunction.RMSE_XENT;
    //learn binary codes
    private ActivationFunction outputLayerActivation = Activations.sigmoid();
    private boolean roundCodeLayerInput = false;
    //could be useful for gaussian/rectified
    private boolean normalizeCodeLayerOutput = false;
    private static Logger log = LoggerFactory.getLogger(DeepAutoEncoder.class);
    private boolean alreadyInitialized = false;

    public DeepAutoEncoder(){}

    public DeepAutoEncoder(BaseMultiLayerNetwork encoder) {
        this.encoder = encoder;
        this.initDecoder();
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
        throw new IllegalStateException("Not implemented");
    }


    /**
     * Compute activations from input to output of the output layer
     * @return the list of activations for each layer
     */
    public  List<DoubleMatrix> feedForward() {
        DoubleMatrix currInput = this.input;

        List<DoubleMatrix> activations = new ArrayList<>();
        activations.add(currInput);
        NeuralNetwork[] layers = getLayers();
        for(int i = 0; i < layers.length; i++) {
            NeuralNetwork layer = getLayers()[i];
            HiddenLayer l = getSigmoidLayers()[i];
            if(layer == null) {
                log.warn("Null layer found going to break");
                break;
            }
            if(l == null) {
                log.warn("Null sigmoid layer found going to break");
                break;
            }
            layer.setInput(currInput);
            l.setInput(currInput);

            if(getSampleOrActivate() != null && getSampleOrActivate().get(i) != null && getSampleOrActivate().get(i) || !sampleFromHiddenActivations) {
                currInput = l.activate(currInput);

                if(roundCodeLayerInput && (layer instanceof  RBM)) {
                    RBM r = (RBM)  layer;
                    if(r.getHiddenType() == RBM.HiddenUnit.GAUSSIAN) {
                        currInput = round(currInput);
                    }
                }
            }

            else  if(sampleFromHiddenActivations) {
                currInput = layer.sampleHiddenGivenVisible(l.getActivationFunction().apply(currInput)).getSecond();

                if(roundCodeLayerInput && (layer instanceof  RBM)) {
                    RBM r = (RBM)  layer;
                    if(r.getHiddenType() != RBM.HiddenUnit.GAUSSIAN) {
                        currInput = round(currInput);
                    }
                }
            }
            else
                currInput = layer.sampleHiddenGivenVisible(currInput).getSecond();
            activations.add(currInput);
        }


        if(getOutputLayer() != null) {
            getOutputLayer().setInput(currInput);
            if(getOutputLayer().getActivationFunction() == null)
                if(outputActivationFunction != null)
                    outputLayer.setActivationFunction(outputActivationFunction);
                else
                    outputLayer.setActivationFunction(Activations.sigmoid());

            activations.add(getOutputLayer().output(currInput));

        }
        return activations;
    }

    /**
     * Pretrain with a data set iterator.
     * This will run through each neural net at a time and train on the input.
     *
     * @param iter        the iterator to use
     * @param otherParams
     */
    @Override
    public void pretrain(DataSetIterator iter, Object[] otherParams) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[numLayers];
    }

    private void initDecoder() {
        //encoder hasn't been pretrained yet
        if(encoder.getLayers() == null || encoder.getSigmoidLayers() == null)
            return;

        //infer input from encoder
        if(encoder.getInput() != null && input == null)
            this.input = encoder.getInput();

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

        //real valued activities on decoding setp
        this.sampleFromHiddenActivations = false;

        NeuralNetwork[] cloned = new NeuralNetwork[encoder.getLayers().length];
        HiddenLayer[] clonedHidden = new HiddenLayer[encoder.getLayers().length];


        for(int i = 0; i < cloned.length ; i++) {
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
            cloned[i].setW(cloned[i].getW());
            clonedHidden[i].setB(cloned[i].gethBias());
        }

        ActivationFunction codeLayerActivation = encoder.getSigmoidLayers()[encoder.getSigmoidLayers().length - 1].getActivationFunction();
        ActivationFunction firstActivation = encoder.getSigmoidLayers()[0].getActivationFunction();

        ArrayUtil.reverse(cloned);
        ArrayUtil.reverse(clonedHidden);

        //reverse causes activation function to be out of line
        clonedHidden[0].setActivationFunction(firstActivation);
        clonedHidden[clonedHidden.length - 1].setActivationFunction(codeLayerActivation);

        NeuralNetwork[] decoderLayers = new NeuralNetwork[cloned.length - 1];
        for(int i = 0 ; i < decoderLayers.length; i++)
            decoderLayers[i] = cloned[i];
        HiddenLayer[] decoderHiddenLayers = new HiddenLayer[clonedHidden.length - 1];
        for(int i = 0; i < decoderHiddenLayers.length; i++)
            decoderHiddenLayers[i] = clonedHidden[i];

        decoder.setSigmoidLayers(decoderHiddenLayers);
        decoder.setLayers(decoderLayers);

        DoubleMatrix encoded = encodeWithScaling(input);
        //decoder.setInput(encoded);
        //decoder.initializeLayers(encoded);


        this.sampleOrActivate = decoder.getSampleOrActivate();
        this.layerLearningRates = decoder.getLayerLearningRates();
        this.normalizeByInputRows = decoder.isNormalizeByInputRows();
        this.useAdaGrad = decoder.isUseAdaGrad();
        this.hiddenLayerSizes = decoder.getHiddenLayerSizes();
        this.rng = decoder.getRng();
        this.dist = decoder.getDist();
        this.activation = decoder.getActivation();
        this.useRegularization = decoder.isUseRegularization();
        this.columnMeans = decoder.getColumnMeans();
        this.columnStds = decoder.getColumnStds();
        this.columnSums = decoder.getColumnSums();
        this.errorTolerance = decoder.getErrorTolerance();
        this.renderWeightsEveryNEpochs = decoder.getRenderWeightsEveryNEpochs();
        this.forceNumEpochs = decoder.isForceNumEpochs();
        this.l2 = decoder.getL2();
        this.fanIn = decoder.getFanIn();
        this.momentum = decoder.getMomentum();
        this.learningRateUpdate = decoder.getLearningRateUpdate();
        this.shouldBackProp = decoder.isShouldBackProp();
        this.sparsity = decoder.getSparsity();
        this.dropOut = decoder.getDropOut();
        this.optimizationAlgorithm = decoder.getOptimizationAlgorithm();
        this.lossFunction = decoder.getLossFunction();
        this.outputActivationFunction = decoder.getOutputActivationFunction();
        this.lossFunctionByLayer = decoder.getLossFunctionByLayer();
        this.outputLossFunction = decoder.getOutputLossFunction();



        //set the combined hidden layers and associated sizes,...
        this.hiddenLayerSizes = ArrayUtil.combine(encoder.getHiddenLayerSizes(),decoder.getHiddenLayerSizes());
        RBM r = (RBM) decoder.getLayers()[0];

        r.setVisibleType(visibleUnit);
        r.setHiddenType(hiddenUnit);

        this.layers = ArrayUtil.combine(encoder.getLayers(),decoder.getLayers());
        this.sigmoidLayers = ArrayUtil.combine(encoder.getSigmoidLayers(),decoder.getSigmoidLayers());
        //for the code layer everything should be linear
        this.sigmoidLayers[encoder.getSigmoidLayers().length - 1].setActivationFunction(codeLayerAct);
        this.outputLayer = decoder.getOutputLayer();

        //set the output layer weights to be the initial input weights
        this.outputLayer.setW(encoder.getLayers()[0].getW().transpose());
        this.outputLayer.setB(encoder.getLayers()[0].getvBias().dup());
        this.outputLayer.setLossFunction(outputLossFunction);
        this.outputLayer.setActivationFunction(outputActivationFunction);

        dimensionCheck();

        //done with encoder/decoder
        this.encoder = null;
        this.decoder = null;
        alreadyInitialized = true;

    }




    /**
     * Trains the decoder on the given input
     * @param input the given input to train on
     */
    public void finetune(DoubleMatrix input,double lr,int epochs) {
        this.input = input;

        if(decoder == null && !alreadyInitialized)
            initDecoder();


        setInput(input);
        setLabels(input);

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

    public OutputLayer.LossFunction getOutputLayerLossFunction() {
        return outputLayerLossFunction;
    }

    public void setOutputLayerLossFunction(OutputLayer.LossFunction outputLayerLossFunction) {
        this.outputLayerLossFunction = outputLayerLossFunction;
        if(outputLayer != null)
            outputLayer.setLossFunction(outputLayerLossFunction);
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


    public void setCodeLayerActivationFunction(ActivationFunction act) {
        this.codeLayerAct = act;
    }

    public static class Builder extends BaseMultiLayerNetwork.Builder<DeepAutoEncoder> {
        public Builder() {
            clazz = DeepAutoEncoder.class;
        }






    }


}
