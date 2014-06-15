package org.deeplearning4j.autoencoder;


import java.util.*;

import org.apache.commons.math3.distribution.RealDistribution;
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
import org.deeplearning4j.transformation.MatrixTransform;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.MatrixUtil;
import org.deeplearning4j.util.RBMUtil;
import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.deeplearning4j.util.MatrixUtil.*;
import static org.deeplearning4j.util.MatrixUtil.normal;

/**
 * Encapsulates a deep auto encoder and decoder (the transpose of an encoder).
 *
 * The idea here will be to train a DBN first using pretrain. This creates the weights
 * for the autoencoder. Here is an example configuration for training a DBN for binary images.
 *
 *


        Map<Integer,Double> layerLearningRates = new HashMap<>();
        layerLearningRates.put(codeLayer,1e-1);
        RandomGenerator rng = new MersenneTwister(123);


        DBN dbn = new DBN.Builder()
        .learningRateForLayer(layerLearningRates)
        .hiddenLayerSizes(new int[]{1000, 500, 250, 30}).withRng(rng)
        .useRBMPropUpAsActivation(true)
        .activateForLayer(Collections.singletonMap(3, Activations.linear()))
        .withHiddenUnitsByLayer(Collections.singletonMap(codeLayer, RBM.HiddenUnit.GAUSSIAN))
        .numberOfInputs(784)
        .sampleFromHiddenActivations(true)
        .sampleOrActivateByLayer(Collections.singletonMap(3,false))
        .lineSearchBackProp(true).useRegularization(true).withL2(1e-3)
        .withOutputActivationFunction(Activations.sigmoid())
        .numberOfOutPuts(784).withMomentum(0.5)
        .withOutputLossFunction(OutputLayer.LossFunction.RMSE_XENT)
 *       .build();
 *
 * Reduction of dimensionality with neural nets Hinton 2006
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
            AutoEncoder layer = (AutoEncoder) getLayers()[i];
            layer.setInput(currInput);

            if(getSampleOrActivate() != null && getSampleOrActivate().get(i) != null && getSampleOrActivate().get(i) || !sampleFromHiddenActivations) {
                currInput = layer.reconstruct(currInput);

            }

            else  if(sampleFromHiddenActivations) {
                currInput = layer.sampleHiddenGivenVisible(layer.reconstruct(currInput)).getSecond();
            }
            else
                currInput = layer.sampleHiddenGivenVisible(currInput).getSecond();
            //code layer gaussian noise


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



    /**
     * Trains the decoder on the given input
     * @param input the given input to train on
     */
    public void finetune(DoubleMatrix input,double lr,int epochs) {
        this.input = input;


        setInput(input);
        setLabels(input);

        super.finetune(input,lr,epochs);

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
        private  BaseMultiLayerNetwork encoder;

        public Builder() {
            clazz = DeepAutoEncoder.class;
        }

        public Builder withEncoder(BaseMultiLayerNetwork encoder) {
            this.encoder = encoder;
            return this;
        }


        @Override
        public Builder lineSearchBackProp(boolean lineSearchBackProp) {
            super.lineSearchBackProp(lineSearchBackProp);
            return this;
        }

        /**
         * Loss function by layer
         *
         * @param lossFunctionByLayer the loss function per layer
         * @return builder pattern
         */
        @Override
        public Builder lossFunctionByLayer(Map<Integer, NeuralNetwork.LossFunction> lossFunctionByLayer) {
            super.lossFunctionByLayer(lossFunctionByLayer);
            return this;
        }

        /**
         * Sample or activate by layer allows for deciding to sample or just pass straight activations
         * for each layer
         *
         * @param sampleOrActivateByLayer
         * @return
         */
        @Override
        public Builder sampleOrActivateByLayer(Map<Integer, Boolean> sampleOrActivateByLayer) {
            super.sampleOrActivateByLayer(sampleOrActivateByLayer);
            return this;
        }

        /**
         * Override rendering for a given layer only
         *
         * @param renderByLayer
         * @return
         */
        @Override
        public Builder renderByLayer(Map<Integer, Integer> renderByLayer) {
            super.renderByLayer(renderByLayer);
            return this;
        }

        /**
         * Override learning rate by layer only
         *
         * @param learningRates
         * @return
         */
        @Override
        public Builder learningRateForLayer(Map<Integer, Double> learningRates) {
             super.learningRateForLayer(learningRates);
            return this;
        }

        @Override
        public Builder activateForLayer(Map<Integer, ActivationFunction> activationForLayer) {
            super.activateForLayer(activationForLayer);
            return this;
            
        }

        @Override
        public Builder activateForLayer(int layer, ActivationFunction function) {
            super.activateForLayer(layer, function);
            return this;
        }

        /**
         * Activation function for output layer
         *
         * @param outputActivationFunction the output activation function to use
         * @return builder pattern
         */
        @Override
        public Builder withOutputActivationFunction(ActivationFunction outputActivationFunction) {
            super.withOutputActivationFunction(outputActivationFunction);
            return this;
        }

        /**
         * Output loss function
         *
         * @param outputLossFunction the output loss function
         * @return
         */
        @Override
        public Builder withOutputLossFunction(OutputLayer.LossFunction outputLossFunction) {
            super.withOutputLossFunction(outputLossFunction);
            return this;
        }

        /**
         * Which optimization algorithm to use with neural nets and Logistic regression
         *
         * @param optimizationAlgo which optimization algorithm to use with
         *                         neural nets and logistic regression
         * @return builder pattern
         */
        @Override
        public Builder withOptimizationAlgorithm(NeuralNetwork.OptimizationAlgorithm optimizationAlgo) {
            super.withOptimizationAlgorithm(optimizationAlgo);
            return this;
        }

        /**
         * Loss function to use with neural networks
         *
         * @param lossFunction loss function to use with neural networks
         * @return builder pattern
         */
        @Override
        public Builder withLossFunction(NeuralNetwork.LossFunction lossFunction) {
            super.withLossFunction(lossFunction);
            return this;
        }

        /**
         * Whether to use drop out on the neural networks or not:
         * random zero out of examples
         *
         * @param dropOut the dropout to use
         * @return builder pattern
         */
        @Override
        public Builder withDropOut(double dropOut) {
            super.withDropOut(dropOut);
            return this;
        }

        /**
         * Whether to use hidden layer activations or neural network sampling
         * on feed forward pass
         *
         * @param useHiddenActivationsForwardProp true if use hidden activations, false otherwise
         * @return builder pattern
         */
        @Override
        public Builder sampleFromHiddenActivations(boolean useHiddenActivationsForwardProp) {
            super.sampleFromHiddenActivations(useHiddenActivationsForwardProp);
            return this;
        }

        /**
         * Turn this off for full dataset training
         *
         * @param normalizeByInputRows whether to normalize the changes
         *                             by the number of input rows
         * @return builder pattern
         */
        @Override
        public Builder normalizeByInputRows(boolean normalizeByInputRows) {
            super.normalizeByInputRows(normalizeByInputRows);
            return this;
        }

        @Override
        public Builder useAdaGrad(boolean useAdaGrad) {
            super.useAdaGrad(useAdaGrad);
            return this;
        }

        @Override
        public Builder withSparsity(double sparsity) {
            super.withSparsity(sparsity);
            return this;
        }

        @Override
        public Builder withVisibleBiasTransforms(Map<Integer, MatrixTransform> visibleBiasTransforms) {
            super.withVisibleBiasTransforms(visibleBiasTransforms);
            return this;
        }

        @Override
        public Builder withHiddenBiasTransforms(Map<Integer, MatrixTransform> hiddenBiasTransforms) {
            super.withHiddenBiasTransforms(hiddenBiasTransforms);
            return this;
        }

        /**
         * Forces use of number of epochs for training
         * SGD style rather than conjugate gradient
         *
         * @return
         */
        @Override
        public Builder forceEpochs() {
            super.forceEpochs();
            return this;
        }

        /**
         * Disables back propagation
         *
         * @return
         */
        @Override
        public Builder disableBackProp() {
            super.disableBackProp();
            return this;
        }

        /**
         * Transform the weights at the given layer
         *
         * @param layer     the layer to transform
         * @param transform the function used for transformation
         * @return
         */
        @Override
        public Builder transformWeightsAt(int layer, MatrixTransform transform) {
            super.transformWeightsAt(layer, transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given layers
         *
         * @param transforms
         * @return
         */
        @Override
        public Builder transformWeightsAt(Map<Integer, MatrixTransform> transforms) {
            super.transformWeightsAt(transforms);
            return this;
        }

        /**
         * Probability distribution for generating weights
         *
         * @param dist
         * @return
         */
        @Override
        public Builder withDist(RealDistribution dist) {
            super.withDist(dist);
            return this;
        }

        /**
         * Specify momentum
         *
         * @param momentum
         * @return
         */
        @Override
        public Builder withMomentum(double momentum) {
            super.withMomentum(momentum);
            return this;
        }

        /**
         * Use l2 reg
         *
         * @param useRegularization
         * @return
         */
        @Override
        public Builder useRegularization(boolean useRegularization) {
            super.useRegularization(useRegularization);
            return this;
        }

        /**
         * L2 coefficient
         *
         * @param l2
         * @return
         */
        @Override
        public Builder withL2(double l2) {
            super.withL2(l2);
            return this;
        }

        /**
         * Whether to plot weights or not
         *
         * @param everyN
         * @return
         */
        @Override
        public Builder renderWeights(int everyN) {
            super.renderWeights(everyN);
            return this;
        }

        @Override
        public Builder withFanIn(Double fanIn) {
            super.withFanIn(fanIn);
            return this;
        }

        /**
         * Pick an activation function, default is sigmoid
         *
         * @param activation
         * @return
         */
        @Override
        public Builder withActivation(ActivationFunction activation) {
            super.withActivation(activation);
            return this;
        }

        @Override
        public Builder numberOfInputs(int nIns) {
            super.numberOfInputs(nIns);
            return this;
        }



        @Override
        public Builder hiddenLayerSizes(Integer[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }

        @Override
        public Builder hiddenLayerSizes(int[] hiddenLayerSizes) {
            super.hiddenLayerSizes(hiddenLayerSizes);
            return this;
        }

        @Override
        public Builder numberOfOutPuts(int nOuts) {
            super.numberOfOutPuts(nOuts);
            return this;
        }

        @Override
        public Builder withRng(RandomGenerator gen) {
            super.withRng(gen);
            return this;
        }

        @Override
        public Builder withInput(DoubleMatrix input) {
            super.withInput(input);
            return this;
        }

        @Override
        public Builder withLabels(DoubleMatrix labels) {
            super.withLabels(labels);
            return this;
        }

        @Override
        public Builder withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            super.withClazz(clazz);
            return this;
        }

        @Override
        public DeepAutoEncoder buildEmpty() {
          return  super.buildEmpty();

        }

        @Override
        public DeepAutoEncoder build() {


            //everything but output layer
            int inverseCount = encoder.getLayers().length - 1;
            NeuralNetwork[] autoEncoders = new NeuralNetwork[encoder.getLayers().length * 2 - 1];
            HiddenLayer[] hiddenLayers = new HiddenLayer[encoder.getLayers().length * 2 - 1];
            for(int i = 0; i < autoEncoders.length; i++) {
                if(i < encoder.getLayers().length) {
                    AutoEncoder a = new AutoEncoder.Builder().withActivation(i < encoder.getLayers().length - 1 ? Activations.sigmoid() : Activations.linear())
                            .numberOfVisible(encoder.getLayers()[i].getnVisible())
                            .numHidden(encoder.getLayers()[i].getnHidden())
                            .withWeights(encoder.getLayers()[i].getW().dup())
                           .applySparsity(encoder.getLayers()[i].isApplySparsity())
                            .normalizeByInputRows(encoder.getLayers()[i].normalizeByInputRows())
                            .withDropOut(encoder.getLayers()[i].dropOut())
                            .useRegularization(encoder.getLayers()[i].isUseRegularization())
                            .useAdaGrad(encoder.getLayers()[i].isUseAdaGrad())
                            .withVisibleBias(encoder.getLayers()[i].getvBias().dup())
                            .withHBias(encoder.getLayers()[i].gethBias().dup())
                            .withDistribution(encoder.getLayers()[i].getDist())
                            .renderWeights(encoder.getLayers()[i].getRenderEpochs())
                            .withL2(encoder.getLayers()[i].getL2()).withMomentum(encoder.getLayers()[i].getMomentum())
                            .withLossFunction(encoder.getLayers()[i].getLossFunction())
                            .withRandom(encoder.getLayers()[i].getRng())
                            .build();
                    //code layer linear
                   if(i == encoder.getLayers().length - 1) {
                       a.act = Activations.linear();
                   }

                    HiddenLayer h = encoder.getSigmoidLayers()[i].clone();
                    h.setActivationFunction(Activations.linear());
                    hiddenLayers[i] = h;
                    autoEncoders[i] = a;

                }
                else {
                    AutoEncoder a = new AutoEncoder.Builder()
                            .numberOfVisible(encoder.getLayers()[inverseCount].getnHidden())
                            .numHidden(encoder.getLayers()[inverseCount].getnVisible())
                            .withWeights(encoder.getLayers()[inverseCount].getW().transpose())
                            .applySparsity(encoder.getLayers()[inverseCount].isApplySparsity())
                            .normalizeByInputRows(encoder.getLayers()[inverseCount].normalizeByInputRows())
                            .withDropOut(encoder.getLayers()[inverseCount].dropOut())
                            .useRegularization(encoder.getLayers()[inverseCount].isUseRegularization())
                            .useAdaGrad(encoder.getLayers()[inverseCount].isUseAdaGrad())
                            .withVisibleBias(encoder.getLayers()[inverseCount].gethBias().dup())
                            .withHBias(encoder.getLayers()[inverseCount].getvBias().dup())
                            .withDistribution(encoder.getLayers()[inverseCount].getDist())
                            .renderWeights(encoder.getLayers()[inverseCount].getRenderEpochs())
                            .withL2(encoder.getLayers()[inverseCount].getL2()).withMomentum(encoder.getLayers()[inverseCount].getMomentum())
                            .withLossFunction(encoder.getLayers()[inverseCount].getLossFunction())
                            .withRandom(encoder.getLayers()[inverseCount].getRng())
                            .build();

                    autoEncoders[i] = a;
                    hiddenLayers[i] = encoder.getSigmoidLayers()[inverseCount].transpose();
                    hiddenLayers[i].setActivationFunction(Activations.linear());
                    inverseCount--;
                }
            }

            OutputLayer o = new OutputLayer.Builder().normalizeByInputRows(encoder.getLayers()[0].normalizeByInputRows())
                    .numberOfInputs(encoder.getLayers()[0].getnHidden()).numberOfOutputs(encoder.getnIns())
                    .useAdaGrad(encoder.getLayers()[0].isUseAdaGrad()).useRegularization(encoder.getLayers()[0].isUseRegularization())
                    .withBias(encoder.getLayers()[0].getvBias()).withActivationFunction(encoder.getOutputActivationFunction())
                    .withL2(encoder.getLayers()[0].getL2()).withWeights(encoder.getLayers()[0].getW().transpose())
                    .build();

            DeepAutoEncoder e = new DeepAutoEncoder();
            e.setLayers(autoEncoders);
            e.setSigmoidLayers(hiddenLayers);
            e.setOutputLayer(o);
            e.setLossFunctionByLayer(encoder.getLossFunctionByLayer());
            e.setSampleOrActivate(encoder.getSampleOrActivate());
            e.setRenderByLayer(encoder.getRenderByLayer());
            e.setNormalizeByInputRows(encoder.isNormalizeByInputRows());
            e.setnOuts(encoder.getnIns());
            e.setnIns(encoder.getnIns());
            e.setRng(encoder.getRng());
            e.setShouldBackProp(this.backProp);
            e.setSampleFromHiddenActivations(encoder.isSampleFromHiddenActivations());
            e.setLineSearchBackProp(encoder.isLineSearchBackProp());
            e.setMomentum(encoder.getMomentum());
            e.activationFunctionForLayer.putAll(encoder.getActivationFunctionForLayer());
            e.setSparsity(encoder.getSparsity());
            e.setRenderWeightsEveryNEpochs(encoder.getRenderWeightsEveryNEpochs());
            e.setL2(encoder.getL2());
            e.setForceNumEpochs(shouldForceEpochs);
            e.setUseRegularization(encoder.isUseRegularization());
            e.setUseAdaGrad(encoder.isUseAdaGrad());
            e.setDropOut(encoder.getDropOut());
            e.setOptimizationAlgorithm(encoder.getOptimizationAlgorithm());
            e.setLossFunction(encoder.getLossFunction());
            e.setOutputActivationFunction(encoder.getOutputActivationFunction());
            e.setOutputLossFunction(encoder.getOutputLossFunction());



            return e;

        }



    }


}
