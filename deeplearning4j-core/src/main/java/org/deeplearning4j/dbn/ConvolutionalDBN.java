package org.deeplearning4j.dbn;

import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.nn.*;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.rbm.ConvolutionalRBM;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.deeplearning4j.util.MatrixUtil;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.List;

/**
 * Convolutional Deep Belief Network
 * @author Adam Gibson
 */
public class ConvolutionalDBN extends BaseMultiLayerNetwork {


    //filter size per layer
    protected int[][] filterSizes;
    //the per datum input size (2d image dimensions)
    protected int[] inputSize;

    //rbm wise nfm
    protected int[] nFm;
    //layer wise stride (arr[k] is an array)
    protected int[][] stride;
    //layer wise filter number of filters
    protected int[] numFilters;
    //sparse gain for each rbm
    protected double sparseGain;
    //layer type: convolution or subsampling
    protected LayerType[] layerTypes;
    public  static enum LayerType {
        SUBSAMPLE,CONVOLUTION
    }






    protected ConvolutionalDBN() {
    }

    protected ConvolutionalDBN(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers, RandomGenerator rng) {
        super(nIns, hiddenLayerSizes, nOuts, nLayers, rng);
    }

    protected ConvolutionalDBN(int nIn, int[] hiddenLayerSizes, int nOuts, int nLayers, RandomGenerator rng, DoubleMatrix input, DoubleMatrix labels) {
        super(nIn, hiddenLayerSizes, nOuts, nLayers, rng, input, labels);
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

    }

    /**
     * Pretrain the network with the given parameters
     *
     * @param input       the input to train ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    @Override
    public void pretrain(DoubleMatrix input, Object[] otherParams) {

    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @param input the input data to feed forward
     * @return the list of activations for each layer
     */
    @Override
    public List<DoubleMatrix> feedForward(DoubleMatrix input) {
        /* Number of tensors is equivalent to the number of mini batches */
        FourDTensor tensor = new FourDTensor(input,inputSize[0],inputSize[1],  input.rows / inputSize[0],input.rows / input.length);
        for(int i = 0; i < getnLayers(); i++) {

        }
        return super.feedForward(input);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    @Override
    public List<DoubleMatrix> feedForward() {
        return super.feedForward();
    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     *
     * @param revert the best network so far
     * @param lr     the learning rate to use for training
     * @param epoch  the epoch to use
     * @return whether the training should converge or not
     */
    @Override
    protected void backPropStep(BaseMultiLayerNetwork revert, double lr, int epoch) {
        super.backPropStep(revert, lr, epoch);
    }

    /**
     * Run SGD based on the given labels
     *
     * @param labels the labels to use
     * @param lr     the learning rate during training
     * @param epochs the number of times to iterate
     */
    @Override
    public void finetune(DoubleMatrix labels, double lr, int epochs) {
        super.finetune(labels, lr, epochs);
    }

    /**
     * Backpropagation of errors for weights
     *
     * @param lr     the learning rate to use
     * @param epochs the number of epochs to iterate (this is already called in finetune)
     */
    @Override
    public void backProp(double lr, int epochs) {
        super.backProp(lr, epochs);
    }

    @Override
    public void init() {
        DoubleMatrix layerInput = input;
        if(!(rng instanceof SynchronizedRandomGenerator))
            rng = new SynchronizedRandomGenerator(rng);
        int inputSize;
        if(getnLayers() < 1)
            throw new IllegalStateException("Unable to create network layers; number specified is less than 1");

        if(this.dist == null)
            dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);

        this.layers = new NeuralNetwork[getnLayers()];

        // construct multi-layer
        for(int i = 0; i < this.getnLayers(); i++) {
            if(i == 0)
                inputSize = this.nIns;
            else
                inputSize = this.hiddenLayerSizes[i-1];

            if(i == 0) {
                // construct sigmoid_layer
                sigmoidLayers[i] = createHiddenLayer(i,inputSize,this.hiddenLayerSizes[i],activation,rng,layerInput,dist);
            }
            else {
                if(input != null) {
                    if(this.useHiddenActivationsForwardProp)
                        layerInput = sigmoidLayers[i - 1].sampleHiddenGivenVisible();
                    else
                        layerInput = getLayers()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();

                }

                // construct sigmoid_layer
                sigmoidLayers[i] = createHiddenLayer(i,inputSize,this.hiddenLayerSizes[i],activation,rng,layerInput,dist);


            }

            this.layers[i] = createLayer(layerInput,inputSize, this.hiddenLayerSizes[i], this.sigmoidLayers[i].getW(), this.sigmoidLayers[i].getB(), null, rng,i);

        }



        // layer for output using LogisticRegression
        this.logLayer = new LogisticRegression.Builder()
                .useAdaGrad(useAdaGrad).optimizeBy(getOptimizationAlgorithm())
                .normalizeByInputRows(normalizeByInputRows)
                .useRegularization(useRegularization)
                .numberOfInputs(hiddenLayerSizes[getnLayers()-1])
                .numberOfOutputs(nOuts).withL2(l2).build();

        synchonrizeRng();

        applyTransforms();
        initCalled = true;

    }

    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overridden
     * for other kinds of hidden units
     *
     * @param index
     * @param nIn        the number of inputs
     * @param nOut       the number of outputs
     * @param activation the activation function for the layer
     * @param rng        the rng to use for sampling
     * @param layerInput the layer starting input
     * @param dist       the probability distribution to use
     *                   for generating weights
     * @return a hidden layer with the given parameters
     */
    @Override
    public DownSamplingLayer createHiddenLayer(int index, int nIn, int nOut, ActivationFunction activation, RandomGenerator rng, DoubleMatrix layerInput, RealDistribution dist) {
        ConvolutionalRBM r = (ConvolutionalRBM) getLayers()[index - 1];
        DownSamplingLayer layer = new DownSamplingLayer.Builder().dist(dist).withInput(layerInput)
                .withFmSize(MatrixFunctions.floor(MatrixUtil.toMatrix(r.getFmSize())).div(MatrixUtil.toMatrix(stride[index])))
                .numFeatureMaps(nFm[index])
                .nIn(nIn).nOut(nOut).withActivation(activation).build();
        return layer;
    }

    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.rbm.RBM} for continuous inputs.
     * <p/>
     * Please be sure to call super.initializeNetwork
     * <p/>
     * to handle the passing of baseline parameters such as fanin
     * and rendering.
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
        ConvolutionalRBM r = new ConvolutionalRBM.Builder().withDistribution(getDist())
                .withDropOut(getDropOut()).withOptmizationAlgo(getOptimizationAlgorithm())
                .withFilterSize(filterSizes[index]).withInput(input).numberOfVisible(nVisible)
                .useAdaGrad(isUseAdaGrad()).normalizeByInputRows(normalizeByInputRows)
                .numHidden(nHidden).withHBias(hbias).withMomentum(getMomentum())
                .withL2(getL2()).useRegularization(isUseRegularization())
                .withRandom(rng).withLossFunction(getLossFunction())
                .withStride(stride[index]).withNumFilters(numFilters[index])
                .withSparseGain(sparseGain).withSparsity(getSparsity())
                .withWeights(W).build();

        return r;
    }

    @Override
    public NeuralNetwork[] createNetworkLayers(int numLayers) {
        return new NeuralNetwork[numLayers];
    }
}
