package org.deeplearning4j.nn;

import static org.jblas.MatrixFunctions.powi;
import static org.jblas.MatrixFunctions.pow;

import static org.deeplearning4j.util.MatrixUtil.binomial;
import static org.deeplearning4j.util.MatrixUtil.assertNaN;
import static org.deeplearning4j.util.MatrixUtil.valueMatrixOf;
import static org.deeplearning4j.util.MatrixUtil.complainAboutMissMatchedMatrices;
import static org.deeplearning4j.util.MatrixUtil.isNaN;


import org.apache.commons.math3.distribution.NormalDistribution;
import org.apache.commons.math3.distribution.RealDistribution;
import org.apache.commons.math3.random.MersenneTwister;
import org.apache.commons.math3.random.RandomGenerator;
import org.deeplearning4j.autoencoder.AutoEncoder;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.nn.NeuralNetwork.LossFunction;
import org.deeplearning4j.nn.NeuralNetwork.OptimizationAlgorithm;
import org.deeplearning4j.nn.activation.ActivationFunction;
import org.deeplearning4j.nn.activation.Activations;
import org.deeplearning4j.nn.activation.Sigmoid;
import org.deeplearning4j.optimize.*;
import org.deeplearning4j.rng.SynchronizedRandomGenerator;
import org.deeplearning4j.transformation.MatrixTransform;
import org.deeplearning4j.util.ArrayUtil;
import org.deeplearning4j.util.Dl4jReflection;
import org.deeplearning4j.util.SerializationUtils;
import org.jblas.DoubleMatrix;

import org.jblas.SimpleBlas;
import org.jblas.ranges.RangeUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.OutputStream;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.util.*;


/**
 * A base class for a multi layer neural network with a logistic output layer
 * and multiple hidden layers.
 * @author Adam Gibson
 *
 */
public abstract class BaseMultiLayerNetwork implements Serializable,Persistable,Output {



    private static Logger log = LoggerFactory.getLogger(BaseMultiLayerNetwork.class);
    private static final long serialVersionUID = -5029161847383716484L;
    //number of columns in the input matrix
    protected int nIns;
    //the hidden layer sizes at each layer
    protected int[] hiddenLayerSizes;
    //the number of outputs/labels for logistic regression
    protected int nOuts;
    //the hidden layers
    protected HiddenLayer[] sigmoidLayers;
    //logistic regression output layer (aka the softmax layer) for translating network outputs in to probabilities
    protected OutputLayer outputLayer;
    protected RandomGenerator rng;
    /* probability distribution for generation of weights */
    protected RealDistribution dist;
    protected double momentum = 0.1;
    //default training examples and associated layers
    protected DoubleMatrix input,labels;
    protected MultiLayerNetworkOptimizer optimizer;
    //activation function for each hidden layer
    protected ActivationFunction activation = new Sigmoid();
    //l2 regularization constant for weight decay
    protected double l2 = 2e-4;
    //whether to initialize layers
    protected boolean shouldInit = true;
    //whether to render weights or not; anything <=0 will not render the weights
    protected int renderWeightsEveryNEpochs = -1;
    protected boolean useRegularization = false;
    //sometimes we may need to transform weights; this allows a
    //weight transform upon layer setup
    protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<>();
    //hidden bias transforms; for initialization
    protected Map<Integer,MatrixTransform> hiddenBiasTransforms = new HashMap<>();
    //visible bias transforms for initialization
    protected Map<Integer,MatrixTransform> visibleBiasTransforms = new HashMap<>();

    //momentum after n iterations
    protected Map<Integer,Double> momentumAfter = new HashMap<>();
    //momentum after n iterations by layer
    protected Map<Integer,Map<Integer,Double>> momentumAfterByLayer = new HashMap<>();
    //reset adagrad historical gradient after n iterations
    protected Map<Integer,Integer> resetAdaGradIterationsByLayer = new HashMap<>();
    //reset adagrad historical gradient after n iterations
    protected int resetAdaGradIterations = -1;

    protected boolean shouldBackProp = true;
    //whether to only train a certain number of epochs
    protected boolean forceNumEpochs = false;
    //don't use sparsity by default
    protected double sparsity = 0;
    //optional: used in normalizing input. This is used in saving the model for prediction purposes in normalizing incoming data
    protected DoubleMatrix columnSums;
    //subtract input by column means for zero mean
    protected DoubleMatrix  columnMeans;
    //divide by the std deviation
    protected DoubleMatrix columnStds;
    protected boolean initCalled = false;
    /* Sample if true, otherwise use the straight activation function */
    protected boolean sampleFromHiddenActivations = true;
    /* weight initialization scheme, this is either a probability distribution or a weighting scheme for initialization */
    protected WeightInit weightInit = null;
    /* weight init for layers */
    protected Map<Integer,WeightInit> weightInitByLayer = new HashMap<>();
    /* output layer weight intialization */
    protected WeightInit outputLayerWeightInit = null;

    /*
     * Use adagrad or not
     */
    protected boolean useAdaGrad = false;
    /**
     * Override the activation function for a particular layer
     */
    protected Map<Integer,ActivationFunction> activationFunctionForLayer = new HashMap<>();

    /*
     * Hinton's Practical guide to RBMS:
     *
     * Learning rate updates over time.
     * Usually when weights have a large fan in (starting point for a random distribution during initialization)
     * you want a smaller update rate.
     *
     * For biases this can be bigger.
     */
    protected double learningRateUpdate = 0.95;
    /*
     * Any neural networks used as layers.
     * This will always have an equivalent sigmoid layer
     * with shared weight matrices for training.
     *
     * Typically, this is some mix of:
     *        RBMs,Denoising AutoEncoders, or their Continuous counterparts
     */
    protected NeuralNetwork[] layers;

    /*
     * The delta from the previous iteration to this iteration for
     * cross entropy must change >= this amount in order to continue.
     */
    protected double errorTolerance = 0.0001;

    /*
     * Drop out: randomly zero examples
     */
    protected double dropOut = 0;
    /*
     * Output layer drop out
     */
    protected double outputLayerDropout = 0;
    /*
     * Normalize by input rows with gradients or not
     */
    protected boolean normalizeByInputRows = false;

    /**
     * Which optimization algorithm to use: SGD or CG
     */
    protected OptimizationAlgorithm optimizationAlgorithm;
    /**
     * Which loss function to use:
     * Squared loss, Reconstruction entropy, negative log likelihood
     */
    protected LossFunction lossFunction;

    /**
     * Loss function for output layer
     */
    protected OutputLayer.LossFunction outputLossFunction;
    /**
     * Output activation function
     */
    protected ActivationFunction outputActivationFunction;
    /**
     * Layer specific learning rates
     */
    protected Map<Integer,Double> layerLearningRates = new HashMap<>();
    /**
     * Render by layer
     */
    protected Map<Integer,Integer> renderByLayer = new HashMap<>();

    /**
     * Sample or activate by layer. Default value is true
     */
    protected Map<Integer,Boolean> sampleOrActivate = new HashMap<>();

    /**
     * Loss function by layer
     */
    protected Map<Integer,LossFunction> lossFunctionByLayer = new HashMap<>();


    protected double fineTuneLearningRate = 1e-2;

    /**
     * Whether to use conjugate gradient line search for back prop or
     * normal SGD backprop
     */
    protected boolean lineSearchBackProp = false;

    /*
      Binary drop connect mask
     */
    protected DoubleMatrix mask;

    /*
      Use drop connect knob
     */
    protected boolean useDropConnect = false;

    /*
       Damping factor for gradient
     */
    protected double dampingFactor = 10;


    /*
      Use gauss newton vector product: hessian free
     */
    protected boolean useGaussNewtonVectorProductBackProp = false;

    /*
       Concat the biases vs adding them
     */
    protected boolean concatBiases = false;

    /*
        Constrains the gradient to unit norm
     */
    protected boolean constrainGradientToUnitNorm = false;


    /* Reflection/factory constructor */
    protected BaseMultiLayerNetwork() {}

    protected BaseMultiLayerNetwork(int nIns, int[] hiddenLayerSizes, int nOuts, int nLayers, RandomGenerator rng) {
        this(nIns,hiddenLayerSizes,nOuts,nLayers,rng,null,null);
    }


    protected BaseMultiLayerNetwork(int nIn, int[] hiddenLayerSizes, int nOuts
            , int nLayers, RandomGenerator rng,DoubleMatrix input,DoubleMatrix labels) {
        this.nIns = nIn;
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.input = input.dup();
        this.labels = labels.dup();

        if(hiddenLayerSizes.length != nLayers)
            throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + nLayers);

        this.nOuts = nOuts;
        this.setnLayers(nLayers);

        this.sigmoidLayers = new HiddenLayer[nLayers];

        if(rng == null)
            this.rng = new SynchronizedRandomGenerator(new MersenneTwister(123));
        else
            this.rng = rng;

        if(input != null)
            initializeLayers(input);
    }



    /* sanity check for hidden layer and inter layer dimensions */
    public void dimensionCheck() {

        for(int i = 0; i < getnLayers(); i++) {
            HiddenLayer h = sigmoidLayers[i];
            NeuralNetwork network = layers[i];
            h.getW().assertSameSize(network.getW());
            h.getB().assertSameSize(network.gethBias());

            if(i < getnLayers() - 1) {
                HiddenLayer h1 = sigmoidLayers[i + 1];
                NeuralNetwork network1 = layers[i + 1];
                if(h1.getnIn() != h.getnOut())
                    throw new IllegalStateException("Invalid structure: hidden layer in for " + (i + 1) + " not equal to number of ins " + i);
                if(network.getnHidden() != network1.getnVisible())
                    throw new IllegalStateException("Invalid structure: network hidden for " + (i + 1) + " not equal to number of visible " + i);

            }
        }

        if(sigmoidLayers[sigmoidLayers.length - 1].getnOut() != outputLayer.getnIn())
            throw new IllegalStateException("Number of outputs for final hidden layer not equal to the number of logistic input units for output layer");


    }


    /**
     * Synchronizes the rng, this is mean for use with scale out methods
     */
    public void synchonrizeRng() {
        RandomGenerator rgen = new SynchronizedRandomGenerator(rng);
        for(int i = 0; i < getnLayers(); i++) {
            layers[i].setRng(rgen);
            sigmoidLayers[i].setRng(rgen);
        }


    }







    /**
     * Base class for initializing the layers based on the input.
     * This is meant for capturing numbers such as input columns or other things.
     * @param input the input matrix for training
     */
    public void initializeLayers(DoubleMatrix input) {
        if(input == null)
            throw new IllegalArgumentException("Unable to initialize layers with empty input");

        if(input.columns != nIns)
            throw new IllegalArgumentException(String.format("Unable to train on number of inputs; columns should be equal to number of inputs. Number of inputs was %d while number of columns was %d",nIns,input.columns));

        if(this.layers == null)
            this.layers = new NeuralNetwork[getnLayers()];

        for(int i = 0; i < hiddenLayerSizes.length; i++)
            if(hiddenLayerSizes[i] < 1)
                throw new IllegalArgumentException("All hidden layer sizes must be >= 1");



        this.input = input.dup();
        if(!initCalled) {
            init();
            log.info("Initializing layers with input of dims " + input.rows + " x " + input.columns);

        }


    }

    public void init() {
        DoubleMatrix layerInput = input;
        if(!(rng instanceof SynchronizedRandomGenerator))
            rng = new SynchronizedRandomGenerator(rng);
        int inputSize;
        if(getnLayers() < 1)
            throw new IllegalStateException("Unable to create network layers; number specified is less than 1");

        if(this.dist == null)
            this.dist = new NormalDistribution(rng,0,.01,NormalDistribution.DEFAULT_INVERSE_ABSOLUTE_ACCURACY);



        if(this.layers == null || sigmoidLayers == null || this.sigmoidLayers[0] == null || this.layers[0] == null) {
            this.layers = new NeuralNetwork[getnLayers()];
            // construct multi-layer
            for(int i = 0; i < this.getnLayers(); i++) {
                ActivationFunction currLayerActivation = activationFunctionForLayer.get(i) != null ? activationFunctionForLayer.get(i) : activation;

                if(i == 0)
                    inputSize = this.nIns;
                else
                    inputSize = this.hiddenLayerSizes[i-1];

                if(i == 0) {
                    // construct sigmoid_layer
                    sigmoidLayers[i] = createHiddenLayer(i,inputSize,this.hiddenLayerSizes[i],currLayerActivation,rng,layerInput,dist);
                }
                else {
                    if(input != null) {
                        if(this.sampleFromHiddenActivations)
                            layerInput = layers[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();
                        else
                            layerInput = getLayers()[i - 1].sampleHiddenGivenVisible(layerInput).getSecond();

                    }

                    // construct sigmoid_layer
                    sigmoidLayers[i] = createHiddenLayer(i,inputSize,this.hiddenLayerSizes[i],currLayerActivation,rng,layerInput,dist);


                }

                this.layers[i] = createLayer(layerInput,inputSize, this.hiddenLayerSizes[i], this.sigmoidLayers[i].getW(), this.sigmoidLayers[i].getB(), null, rng,i);

            }

        }



        // layer for output using OutputLayer
        this.outputLayer = new OutputLayer.Builder()
                .withDropout(outputLayerDropout).weightInit(outputLayerWeightInit)
                .useAdaGrad(useAdaGrad).optimizeBy(getOptimizationAlgorithm())
                .normalizeByInputRows(normalizeByInputRows).withLossFunction(outputLossFunction)
                .useRegularization(useRegularization).withActivationFunction(outputActivationFunction)
                .numberOfInputs(getSigmoidLayers()[getSigmoidLayers().length - 1].nOut)
                .numberOfOutputs(nOuts).withL2(l2).build();

        synchonrizeRng();
        dimensionCheck();
        applyTransforms();
        initCalled = true;
        initMask();

    }

    /**
     * Triggers the activation of the last hidden layer ie: not logistic regression
     * @return the activation of the last hidden layer given the last input to the network
     */
    public DoubleMatrix activate() {
        return getSigmoidLayers()[getSigmoidLayers().length - 1].activate();
    }

    /**
     * Triggers the activation for a given layer
     * @param layer the layer to activate on
     * @return the activation for a given layer
     */
    public DoubleMatrix activate(int layer) {
        return getSigmoidLayers()[layer].activate();
    }

    /**
     * Triggers the activation of the given layer
     * @param layer the layer to trigger on
     * @param input the input to the hidden layer
     * @return the activation of the layer based on the input
     */
    public DoubleMatrix activate(int layer, DoubleMatrix input) {
        return getSigmoidLayers()[layer].activate(input);
    }


    /**
     * Finetunes with the current cached labels
     * @param lr the learning rate to use
     * @param epochs the max number of epochs to finetune with
     */
    public void finetune(double lr, int epochs) {
        finetune(this.labels,lr,epochs);

    }

    /**
     * Sets the input and labels from this dataset
     * @param data the dataset to initialize with
     */
    public void initialize(DataSet data) {
        setInput(data.getFirst());
        feedForward(data.getFirst());
        this.labels = data.getSecond();
        outputLayer.setLabels(labels);
    }


    public Map<Integer, Boolean> getSampleOrActivate() {
        return sampleOrActivate;
    }

    public void setSampleOrActivate(Map<Integer, Boolean> sampleOrActivate) {
        this.sampleOrActivate = sampleOrActivate;
    }

    /**
     * Compute activations from input to output of the output layer
     * @return the list of activations for each layer
     */
    public  List<DoubleMatrix> feedForwardConcat() {
        DoubleMatrix currInput = DoubleMatrix.concatHorizontally(this.input,DoubleMatrix.ones(input.rows,1));
        if(this.input.columns != nIns)
            throw new IllegalStateException("Illegal input length");
        List<DoubleMatrix> activations = new ArrayList<>();
        activations.add(currInput);
        for(int i = 0; i < getnLayers(); i++) {
            NeuralNetwork layer = getLayers()[i];
            HiddenLayer l = getSigmoidLayers()[i];

            layer.setInput(currInput);
            l.setInput(currInput);

            if(getSampleOrActivate() != null && getSampleOrActivate().get(i) != null && getSampleOrActivate().get(i))
                currInput = getSigmoidLayers()[i].activate(layer.sampleHiddenGivenVisible(currInput).getSecond());
            else  if(sampleFromHiddenActivations)
                currInput = layer.sampleHiddenGivenVisible(l.getActivationFunction().apply(currInput)).getSecond();
            else
                currInput = layer.sampleHiddenGivenVisible(currInput).getSecond();
            //applies drop connect to the activation
            applyDropConnectIfNecessary(currInput);
            activations.add(currInput);
        }

        if(getOutputLayer() != null) {
            getOutputLayer().setInput(activations.get(activations.size() - 1));
            if(getOutputLayer().getActivationFunction() == null)
                if(outputActivationFunction != null)
                    outputLayer.setActivationFunction(outputActivationFunction);
                else
                    outputLayer.setActivationFunction(Activations.sigmoid());

            activations.add(getOutputLayer().output(activations.get(activations.size() - 1)));

        }
        return activations;
    }



    /**
     * Compute activations from input to output of the output layer
     * @return the list of activations for each layer
     */
    public  List<DoubleMatrix> feedForward() {
        DoubleMatrix currInput = this.input;
        if(this.input.columns != nIns)
            throw new IllegalStateException("Illegal input length");

        List<DoubleMatrix> activations = new ArrayList<>();
        activations.add(currInput);

        for(int i = 0; i < getnLayers(); i++) {
            NeuralNetwork layer = getLayers()[i];
            HiddenLayer l = getSigmoidLayers()[i];

            layer.setInput(currInput);
            l.setInput(currInput);

            if(getSampleOrActivate() != null && getSampleOrActivate().get(i) != null && getSampleOrActivate().get(i))
                currInput = getSigmoidLayers()[i].activate(layer.reconstruct(currInput));
            else  if(sampleFromHiddenActivations)
                currInput = layer.sampleHiddenGivenVisible(l.getActivationFunction().apply(currInput)).getSecond();
            else
                currInput = layer.sampleHiddenGivenVisible(currInput).getSecond();
            //applies drop connect to the activation
            applyDropConnectIfNecessary(currInput);
            activations.add(currInput);
        }

        if(getOutputLayer() != null) {
            getOutputLayer().setInput(activations.get(activations.size() - 1));
            if(getOutputLayer().getActivationFunction() == null)
                if(outputActivationFunction != null)
                    outputLayer.setActivationFunction(outputActivationFunction);
                else
                    outputLayer.setActivationFunction(Activations.sigmoid());

            activations.add(getOutputLayer().output(activations.get(activations.size() - 1)));

        }
        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     * @return the list of activations for each layer
     */
    public  List<DoubleMatrix> feedForward(DoubleMatrix input) {
        if(input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");

        else
            this.input = input;
        return feedForward();
    }

    /**
     * Compute activations from input to output of the output layer
     * with the R operator
     * @return the list of activations for each layer
     */
    public  List<DoubleMatrix> feedForwardR(DoubleMatrix input,DoubleMatrix v) {
        if(input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");

        else
            this.input = input;
        return feedForwardR(v);
    }


    /**
     * Applies drop connect relative to connections.
     * This should be used on the activation of a neural net. (Post sigmoid layer)
     * @param input the input to apply drop connect to
     */
    protected void applyDropConnectIfNecessary(DoubleMatrix input) {
        if(useDropConnect) {
            DoubleMatrix mask = binomial(valueMatrixOf(0.5,input.rows,input.columns),1,rng);
            input.muli(mask);
            //apply l2 for drop connect
            if(l2 > 0)
                input.muli(l2);
        }
    }



    /* delta computation for back prop with the R operator */
    protected  List<DoubleMatrix> computeDeltasR(DoubleMatrix v) {
        List<DoubleMatrix> deltaRet = new ArrayList<>();
        assertNaN(v);
        DoubleMatrix[] deltas = new DoubleMatrix[getnLayers() + 1];
        List<DoubleMatrix> activations = feedForward();
        List<DoubleMatrix> rActivations = feedForwardR(activations,v);
      /*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<DoubleMatrix> weights = new ArrayList<>();
        List<DoubleMatrix> biases = new ArrayList<>();
        List<ActivationFunction> activationFunctions = new ArrayList<>();


        for(int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getW());
            biases.add(getLayers()[j].gethBias());
            activationFunctions.add(getSigmoidLayers()[j].getActivationFunction());
        }

        weights.add(getOutputLayer().getW());
        biases.add(getOutputLayer().getB());
        activationFunctions.add(outputLayer.getActivationFunction());

        DoubleMatrix rix = rActivations.get(rActivations.size() - 1).divi(input.rows);
        assertNaN(rix);

        //errors
        for(int i = getnLayers(); i >= 0; i--) {
            //W^t * error^l + 1
            deltas[i] =  activations.get(i).transpose().mmul(rix);
            applyDropConnectIfNecessary(deltas[i]);

            if(i > 0)
                rix = rix.mmul(weights.get(i).addRowVector(biases.get(i)).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));




        }

        for(int i = 0; i < deltas.length; i++) {
            if(constrainGradientToUnitNorm)
                deltaRet.add(deltas[i].div(deltas[i].norm2()));

            else
                deltaRet.add(deltas[i]);
            assertNaN(deltaRet.get(i));
        }

        return deltaRet;
    }

    /**
     * Unpack in a condensed form (weight,bias as one matrix) with
     * the param vector as flattened in each element in the list
     * @param param the param vector to convert
     * @return an unpacked list in a condensed form (weight/bias merged)
     */
    public List<DoubleMatrix> unPackCondensed(DoubleMatrix param) {
        //more sanity checks!
        int numParams = numParams();
        if(param.length != numParams)
            throw new IllegalArgumentException("Parameter vector not equal of length to " + numParams);

        List<DoubleMatrix> ret = new ArrayList<>();
        int curr = 0;
        for(int i = 0; i < layers.length; i++) {
            int layerLength = layers[i].getW().length + layers[i].gethBias().length;
            DoubleMatrix subMatrix = param.get(RangeUtils.all(),RangeUtils.interval(curr,curr + layerLength));
            DoubleMatrix weightPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(0,layers[i].getW().length));

            int beginHBias = layers[i].getW().length;
            int endHbias = subMatrix.length;
            DoubleMatrix hBiasPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(beginHBias,endHbias));
            int layerLengthSum = weightPortion.length + hBiasPortion.length;
            if(layerLengthSum != layerLength) {
                if(hBiasPortion.length != layers[i].gethBias().length)
                    throw new IllegalStateException("Hidden bias on layer " + i + " was off");
                if(weightPortion.length != layers[i].getW().length)
                    throw new IllegalStateException("Weight portion on layer " + i + " was off");

            }

            DoubleMatrix d = new DoubleMatrix(1,weightPortion.length + hBiasPortion.length);
            d.put(RangeUtils.all(),RangeUtils.interval(0,weightPortion.length),weightPortion);
            d.put(RangeUtils.all(),RangeUtils.interval(weightPortion.length,d.length),hBiasPortion);

            ret.add(d);
            curr += layerLength;
        }

        int outputLayerWeightLength = outputLayer.getW().length;
        int biasLength = outputLayer.getB().length;
        int layerLength = outputLayerWeightLength + biasLength;

        DoubleMatrix subMatrix = param.get(RangeUtils.all(),RangeUtils.interval(curr,curr + layerLength));
        DoubleMatrix weightPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(0,outputLayer.getW().length));
        DoubleMatrix hBiasPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(weightPortion.length,subMatrix.length));
        DoubleMatrix d = new DoubleMatrix(1,weightPortion.length + hBiasPortion.length);
        d.put(RangeUtils.all(),RangeUtils.interval(0,weightPortion.length),weightPortion);
        d.put(RangeUtils.all(),RangeUtils.interval(weightPortion.length,d.length),hBiasPortion);

        ret.add(d);





        return ret;
    }


    /* delta computation for back prop with the R operator */
    protected  void computeDeltasR2(List<Pair<DoubleMatrix,DoubleMatrix>> deltaRet,DoubleMatrix v) {
        if(mask == null)
            initMask();
        DoubleMatrix[] deltas = new DoubleMatrix[getnLayers() + 2];
        DoubleMatrix[] preCons = new DoubleMatrix[getnLayers() + 2];
        List<DoubleMatrix> activations = feedForward();
        List<DoubleMatrix> rActivations = feedForwardR(v);

		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<DoubleMatrix> weights = new ArrayList<>();
        List<DoubleMatrix> biases = new ArrayList<>();
        List<ActivationFunction> activationFunctions = new ArrayList<>();

        for(int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getW());
            biases.add(getLayers()[j].gethBias());
            activationFunctions.add(getSigmoidLayers()[j].getActivationFunction());
        }

        weights.add(getOutputLayer().getW());
        biases.add(getOutputLayer().getB());
        activationFunctions.add(outputLayer.getActivationFunction());
        DoubleMatrix rix = rActivations.get(rActivations.size() - 1).div(input.rows);

        //errors
        for(int i = getnLayers() + 1; i >= 0; i--) {
            //output layer
            if(i >= getnLayers() + 1) {
                //-( y - h) .* f'(z^l) where l is the output layer
                //note the one difference here, back prop the error from the gauss newton vector R operation rather than the normal output
                deltas[i] = rix;
                preCons[i] = pow(rix,2);
                applyDropConnectIfNecessary(deltas[i]);

            }
            else {

                //W^t * error^l + 1
                deltas[i] =  activations.get(i).transpose().mmul(rix);
                preCons[i] = powi(activations.get(i).transpose(),2).mmul(pow(rix,2));

                applyDropConnectIfNecessary(deltas[i]);
                //calculate gradient for layer
                DoubleMatrix weightsPlusBias = weights.get(i).addRowVector(biases.get(i).transpose()).transpose();
                DoubleMatrix activation = activations.get(i);
                if(i > 0)
                    rix = rix.mmul(weightsPlusBias).mul(activationFunctions.get(i - 1).applyDerivative(activation));


            }

        }




        for(int i = 0; i < deltas.length; i++) {
            if(constrainGradientToUnitNorm)
                deltaRet.add(new Pair<>(deltas[i].divi(deltas[i].norm2()),preCons[i]));

            else
                deltaRet.add(new Pair<>(deltas[i],preCons[i]));
        }

    }



    private DoubleMatrix compress(DoubleMatrix[] matrices) {
        double[][] data = new double[matrices.length][];
        for(int i = 0; i < data.length; i++)
            data[i] = matrices[i].data;
        DoubleMatrix ret =  new DoubleMatrix(ArrayUtil.combine(data));
        return ret.reshape(1,ret.length);
    }



    //damping update after line search
    public void dampingUpdate(double rho,double boost,double decrease) {
        if(rho < 0.25 || Double.isNaN(rho)) {
            this.dampingFactor *= boost;
        }
        else if(rho > 0.75)
            this.dampingFactor *= decrease;


    }

    /* p and gradient are same length */
    public double reductionRatio(DoubleMatrix p,double currScore,double score,DoubleMatrix gradient) {
        double currentDamp = dampingFactor;
        this.dampingFactor = 0;
        DoubleMatrix denom = getBackPropRGradient(p);
        denom.muli(0.5).muli(p.mul(denom)).rowSums();
        denom.subi(gradient.mul(p).rowSums());
        double rho = (currScore - score) / denom.get(0);
        this.dampingFactor = currentDamp;
        if(score - currScore > 0)
            return Double.NEGATIVE_INFINITY;
        return rho;
    }



    /* delta computation for back prop with precon for SFH */
    protected  List<Pair<DoubleMatrix,DoubleMatrix>>  computeDeltas2() {
        List<Pair<DoubleMatrix,DoubleMatrix>> deltaRet = new ArrayList<>();
        List<DoubleMatrix> activations = feedForward();
        DoubleMatrix[] deltas = new DoubleMatrix[activations.size() - 1];
        DoubleMatrix[] preCons = new DoubleMatrix[activations.size() - 1];


        //- y - h
        DoubleMatrix ix = activations.get(activations.size() - 1).sub(labels).div(labels.rows);
        log.info("Ix mean " + ix.sum());

       	/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<DoubleMatrix> weights = new ArrayList<>();
        List<DoubleMatrix> biases = new ArrayList<>();

        List<ActivationFunction> activationFunctions = new ArrayList<>();
        for(int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getW());
            biases.add(getLayers()[j].gethBias());
            activationFunctions.add(getSigmoidLayers()[j].getActivationFunction());
        }

        biases.add(getOutputLayer().getB());
        weights.add(getOutputLayer().getW());
        activationFunctions.add(outputLayer.getActivationFunction());



        //errors
        for(int i = weights.size() - 1; i >= 0; i--) {
            deltas[i] = activations.get(i).transpose().mmul(ix);
            log.info("Delta sum at " + i + " is " + deltas[i].sum());
            preCons[i] = powi(activations.get(i) .transpose(),2).mmul(pow(ix, 2)).mul(labels.rows);
            applyDropConnectIfNecessary(deltas[i]);

            if(i > 0) {
                //W[i] + b[i] * f'(z[i - 1])
                ix = ix.mmul(weights.get(i).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));
}
        }

        for(int i = 0; i < deltas.length; i++) {
            if(constrainGradientToUnitNorm)
                deltaRet.add(new Pair<>(deltas[i].divi(deltas[i].norm2()),preCons[i]));

            else
                deltaRet.add(new Pair<>(deltas[i],preCons[i]));

        }

        return deltaRet;
    }





    /* delta computation for back prop */
    protected  void computeDeltas(List<DoubleMatrix> deltaRet) {
        DoubleMatrix[] deltas = new DoubleMatrix[getnLayers() + 2];

        List<DoubleMatrix> activations = feedForward();

        //- y - h
        DoubleMatrix  ix = activations.get(activations.size() - 1).sub(labels).divi(input.rows);

		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<DoubleMatrix> weights = new ArrayList<>();
        List<DoubleMatrix> biases = new ArrayList<>();

        List<ActivationFunction> activationFunctions = new ArrayList<>();
        for(int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getW());
            biases.add(getLayers()[j].gethBias());
            activationFunctions.add(getSigmoidLayers()[j].getActivationFunction());
        }

        biases.add(getOutputLayer().getB());
        weights.add(getOutputLayer().getW());
        activationFunctions.add(outputLayer.getActivationFunction());



        //errors
        for(int i = getnLayers() + 1; i >= 0; i--) {
            //output layer
            if(i >= getnLayers() + 1) {
                //-( y - h) .* f'(z^l) where l is the output layer
                deltas[i] = ix;

            }
            else {
                DoubleMatrix delta = activations.get(i).transpose().mmul(ix);
                deltas[i] = delta;
                applyDropConnectIfNecessary(deltas[i]);
                DoubleMatrix weightsPlusBias = weights.get(i).addRowVector(biases.get(i)).transpose();
                DoubleMatrix activation = activations.get(i);
                if(i > 0)
                    ix = ix.mmul(weightsPlusBias).muli(activationFunctions.get(i - 1).applyDerivative(activation));

            }

        }

        for(int i = 0; i < deltas.length; i++) {
            if(constrainGradientToUnitNorm)
                deltaRet.add(deltas[i].divi(deltas[i].norm2()));

            else
                deltaRet.add(deltas[i]);
        }



    }

    /**
     * One step of back prop
     */
    public void backPropStep() {
        List<Pair<DoubleMatrix,DoubleMatrix>> deltas = backPropGradient();
        for(int i = 0; i < layers.length; i++) {
            if(deltas.size() < layers.length) {
                layers[i].getW().subi(deltas.get(i).getFirst());
                layers[i].gethBias().subi(deltas.get(i).getSecond());
                sigmoidLayers[i].setW(layers[i].getW());
                sigmoidLayers[i].setB(layers[i].gethBias());
            }

        }

        outputLayer.getW().subi(deltas.get(deltas.size() - 1).getFirst());
        outputLayer.getB().subi(deltas.get(deltas.size() - 1).getSecond());

    }


    /**
     * One step of back prop with the R operator
     * @param v for gaussian vector, this is for g * v. This is optional.
     */
    public void backPropStepR(DoubleMatrix v) {
        List<Pair<DoubleMatrix,DoubleMatrix>> deltas = backPropGradientR(v);
        for(int i = 0; i < layers.length; i++) {
            if(deltas.size() < layers.length) {
                layers[i].getW().subi(deltas.get(i).getFirst());
                layers[i].gethBias().subi(deltas.get(i).getSecond());
                sigmoidLayers[i].setW(layers[i].getW());
                sigmoidLayers[i].setB(layers[i].gethBias());
            }

        }

        outputLayer.getW().subi(deltas.get(deltas.size() - 1).getFirst());
        outputLayer.getB().subi(deltas.get(deltas.size() - 1).getSecond());

    }



    /**
     * Gets the back prop gradient with the r operator (gauss vector)
     * This is also called computeGV
     * @param v the v in gaussian newton vector g * v
     * @return the back prop with r gradient
     */
    public DoubleMatrix getBackPropRGradient(DoubleMatrix v) {
        return pack(backPropGradientR(v));
    }


    /**
     * Gets the back prop gradient with the r operator (gauss vector)
     * and the associated precon matrix
     * This is also called computeGV
     * @return the back prop with r gradient
     */
    public Pair<DoubleMatrix,DoubleMatrix> getBackPropGradient2() {
        List<Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>>> deltas = backPropGradient2();
        List<Pair<DoubleMatrix,DoubleMatrix>> deltaNormal = new ArrayList<>();
        List<Pair<DoubleMatrix,DoubleMatrix>> deltasPreCon = new ArrayList<>();
        for(int i = 0; i < deltas.size(); i++) {
            deltaNormal.add(deltas.get(i).getFirst());
            deltasPreCon.add(deltas.get(i).getSecond());
        }


        return new Pair<>(pack(deltaNormal),pack(deltasPreCon));
    }



    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BaseMultiLayerNetwork)) return false;

        BaseMultiLayerNetwork that = (BaseMultiLayerNetwork) o;

        if (Double.compare(that.dropOut, dropOut) != 0) return false;
        if (Double.compare(that.errorTolerance, errorTolerance) != 0) return false;
        if (forceNumEpochs != that.forceNumEpochs) return false;
        if (initCalled != that.initCalled) return false;
        if (Double.compare(that.l2, l2) != 0) return false;
        if (Double.compare(that.learningRateUpdate, learningRateUpdate) != 0) return false;
        if (Double.compare(that.momentum, momentum) != 0) return false;
        if (nIns != that.nIns) return false;
        if (nOuts != that.nOuts) return false;
        if (normalizeByInputRows != that.normalizeByInputRows) return false;
        if (renderWeightsEveryNEpochs != that.renderWeightsEveryNEpochs) return false;
        if (shouldBackProp != that.shouldBackProp) return false;
        if (shouldInit != that.shouldInit) return false;
        if (Double.compare(that.sparsity, sparsity) != 0) return false;
        if (useAdaGrad != that.useAdaGrad) return false;
        if (sampleFromHiddenActivations != that.sampleFromHiddenActivations) return false;
        if (useRegularization != that.useRegularization) return false;
        if (activation != null ? !activation.equals(that.activation) : that.activation != null) return false;
        if (activationFunctionForLayer != null ? !activationFunctionForLayer.equals(that.activationFunctionForLayer) : that.activationFunctionForLayer != null)
            return false;
        if (columnMeans != null ? !columnMeans.equals(that.columnMeans) : that.columnMeans != null) return false;
        if (columnStds != null ? !columnStds.equals(that.columnStds) : that.columnStds != null) return false;
        if (columnSums != null ? !columnSums.equals(that.columnSums) : that.columnSums != null) return false;
        if (hiddenBiasTransforms != null ? !hiddenBiasTransforms.equals(that.hiddenBiasTransforms) : that.hiddenBiasTransforms != null)
            return false;
        if (!Arrays.equals(hiddenLayerSizes, that.hiddenLayerSizes)) return false;
        if (input != null ? !input.equals(that.input) : that.input != null) return false;
        if (labels != null ? !labels.equals(that.labels) : that.labels != null) return false;
        if (!Arrays.equals(layers, that.layers)) return false;
        if (lossFunction != that.lossFunction) return false;
        if (optimizationAlgorithm != that.optimizationAlgorithm) return false;
        if (optimizer != null ? !optimizer.equals(that.optimizer) : that.optimizer != null) return false;
        if (outputActivationFunction != null ? !outputActivationFunction.equals(that.outputActivationFunction) : that.outputActivationFunction != null)
            return false;
        if (outputLayer != null ? !outputLayer.equals(that.outputLayer) : that.outputLayer != null) return false;
        if (outputLossFunction != that.outputLossFunction) return false;
        if (!Arrays.equals(sigmoidLayers, that.sigmoidLayers)) return false;
        if (visibleBiasTransforms != null ? !visibleBiasTransforms.equals(that.visibleBiasTransforms) : that.visibleBiasTransforms != null)
            return false;
        if (weightTransforms != null ? !weightTransforms.equals(that.weightTransforms) : that.weightTransforms != null)
            return false;

        return true;
    }

    @Override
    public int hashCode() {
        int result;
        long temp;
        result = nIns;
        result = 31 * result + (hiddenLayerSizes != null ? Arrays.hashCode(hiddenLayerSizes) : 0);
        result = 31 * result + nOuts;
        result = 31 * result + (sigmoidLayers != null ? Arrays.hashCode(sigmoidLayers) : 0);
        result = 31 * result + (outputLayer != null ? outputLayer.hashCode() : 0);
        temp = Double.doubleToLongBits(momentum);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (input != null ? input.hashCode() : 0);
        result = 31 * result + (labels != null ? labels.hashCode() : 0);
        result = 31 * result + (optimizer != null ? optimizer.hashCode() : 0);
        result = 31 * result + (activation != null ? activation.hashCode() : 0);
        temp = Double.doubleToLongBits(l2);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (shouldInit ? 1 : 0);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + renderWeightsEveryNEpochs;
        result = 31 * result + (useRegularization ? 1 : 0);
        result = 31 * result + (weightTransforms != null ? weightTransforms.hashCode() : 0);
        result = 31 * result + (hiddenBiasTransforms != null ? hiddenBiasTransforms.hashCode() : 0);
        result = 31 * result + (visibleBiasTransforms != null ? visibleBiasTransforms.hashCode() : 0);
        result = 31 * result + (shouldBackProp ? 1 : 0);
        result = 31 * result + (forceNumEpochs ? 1 : 0);
        temp = Double.doubleToLongBits(sparsity);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (columnSums != null ? columnSums.hashCode() : 0);
        result = 31 * result + (columnMeans != null ? columnMeans.hashCode() : 0);
        result = 31 * result + (columnStds != null ? columnStds.hashCode() : 0);
        result = 31 * result + (initCalled ? 1 : 0);
        result = 31 * result + (sampleFromHiddenActivations ? 1 : 0);
        result = 31 * result + (useAdaGrad ? 1 : 0);
        result = 31 * result + (activationFunctionForLayer != null ? activationFunctionForLayer.hashCode() : 0);
        temp = Double.doubleToLongBits(learningRateUpdate);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (layers != null ? Arrays.hashCode(layers) : 0);
        temp = Double.doubleToLongBits(errorTolerance);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        temp = Double.doubleToLongBits(dropOut);
        result = 31 * result + (int) (temp ^ (temp >>> 32));
        result = 31 * result + (normalizeByInputRows ? 1 : 0);
        result = 31 * result + (optimizationAlgorithm != null ? optimizationAlgorithm.hashCode() : 0);
        result = 31 * result + (lossFunction != null ? lossFunction.hashCode() : 0);
        result = 31 * result + (outputLossFunction != null ? outputLossFunction.hashCode() : 0);
        result = 31 * result + (outputActivationFunction != null ? outputActivationFunction.hashCode() : 0);
        return result;
    }

    @Override
    public String toString() {
        final StringBuilder sb = new StringBuilder("BaseMultiLayerNetwork{");
        sb.append("nIns=").append(nIns);
        sb.append(", hiddenLayerSizes=").append(Arrays.toString(hiddenLayerSizes));
        sb.append(", nOuts=").append(nOuts);
        sb.append(", sigmoidLayers=").append(Arrays.toString(sigmoidLayers));
        sb.append(", outputLayer=").append(outputLayer);
        sb.append(", rng=").append(rng);
        sb.append(", dist=").append(dist);
        sb.append(", momentum=").append(momentum);
        sb.append(", input=").append(input);
        sb.append(", labels=").append(labels);
        sb.append(", optimizer=").append(optimizer);
        sb.append(", activation=").append(activation);
        sb.append(", l2=").append(l2);
        sb.append(", shouldInit=").append(shouldInit);
        sb.append(", renderWeightsEveryNEpochs=").append(renderWeightsEveryNEpochs);
        sb.append(", useRegularization=").append(useRegularization);
        sb.append(", weightTransforms=").append(weightTransforms);
        sb.append(", hiddenBiasTransforms=").append(hiddenBiasTransforms);
        sb.append(", visibleBiasTransforms=").append(visibleBiasTransforms);
        sb.append(", shouldBackProp=").append(shouldBackProp);
        sb.append(", forceNumEpochs=").append(forceNumEpochs);
        sb.append(", sparsity=").append(sparsity);
        sb.append(", columnSums=").append(columnSums);
        sb.append(", columnMeans=").append(columnMeans);
        sb.append(", columnStds=").append(columnStds);
        sb.append(", initCalled=").append(initCalled);
        sb.append(", sampleFromHiddenActivations=").append(sampleFromHiddenActivations);
        sb.append(", useAdaGrad=").append(useAdaGrad);
        sb.append(", activationFunctionForLayer=").append(activationFunctionForLayer);
        sb.append(", learningRateUpdate=").append(learningRateUpdate);
        sb.append(", layers=").append(Arrays.toString(layers));
        sb.append(", errorTolerance=").append(errorTolerance);
        sb.append(", dropOut=").append(dropOut);
        sb.append(", normalizeByInputRows=").append(normalizeByInputRows);
        sb.append(", optimizationAlgorithm=").append(optimizationAlgorithm);
        sb.append(", lossFunction=").append(lossFunction);
        sb.append(", outputLossFunction=").append(outputLossFunction);
        sb.append(", outputActivationFunction=").append(outputActivationFunction);
        sb.append(", layerLearningRates=").append(layerLearningRates);
        sb.append(", renderByLayer=").append(renderByLayer);
        sb.append(", sampleOrActivate=").append(sampleOrActivate);
        sb.append(", lossFunctionByLayer=").append(lossFunctionByLayer);
        sb.append('}');
        return sb.toString();
    }

    @Override
    public BaseMultiLayerNetwork clone() {
        BaseMultiLayerNetwork ret = new Builder<>().withClazz(getClass()).buildEmpty();
        ret.update(this);
        return ret;
    }




    public List<DoubleMatrix> hiddenBiases() {
        List<DoubleMatrix> ret = new ArrayList<>();
        for(int i = 0; i < layers.length; i++)
            ret.add(layers[i].gethBias());
        ret.add(outputLayer.getB());
        return ret;
    }



    public List<DoubleMatrix> weightMatrices() {
        List<DoubleMatrix> ret = new ArrayList<>();
        for(int i = 0; i < layers.length; i++)
            ret.add(layers[i].getW());
        ret.add(outputLayer.getW());
        return ret;
    }


    /**
     * Feed forward with the r operator
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    public List<DoubleMatrix> feedForwardR(List<DoubleMatrix> acts,DoubleMatrix v) {
        List<DoubleMatrix> R = new ArrayList<>();
        R.add(DoubleMatrix.zeros(input.rows,input.columns));
        List<Pair<DoubleMatrix,DoubleMatrix>> vWvB = unPack(v);
        List<DoubleMatrix> W = weightMatrices();

        for(int i = 0; i < layers.length; i++) {
            ActivationFunction derivative = getSigmoidLayers()[i].getActivationFunction();
            if(getLayers()[i] instanceof AutoEncoder) {
                AutoEncoder a = (AutoEncoder) getLayers()[i];
                derivative = a.getAct();
            }

            //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
            R.add(R.get(i).mmul(W.get(i)).addi(acts.get(i).mmul(vWvB.get(i).getFirst().addRowVector(vWvB.get(i).getSecond()))).muli((derivative.applyDerivative(acts.get(i + 1)))));
        }

        //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
        R.add(R.get(R.size() - 1).mmul(W.get(W.size() - 1)).addi(acts.get(acts.size() - 2).mmul(vWvB.get(vWvB.size() - 1).getFirst().addRowVector(vWvB.get(vWvB.size() - 1).getSecond()))).muli((outputLayer.getActivationFunction().applyDerivative(acts.get(acts.size() - 1)))));

        return R;
    }


    /**
     * Feed forward with the r operator
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    public List<DoubleMatrix> feedForwardR(DoubleMatrix v) {
        return feedForwardR(feedForward(),v);
    }






    /**
     * Backpropagation of errors for weights
     * @param lr the learning rate to use
     * @param epochs  the number of epochs to iterate (this is already called in finetune)
     * @param eval the evaluator for stopping
     */
    public void backProp(double lr,int epochs,TrainingEvaluator eval) {
        if(useGaussNewtonVectorProductBackProp) {
            BackPropROptimizer opt = new BackPropROptimizer(this,lr,epochs);
            opt.optimize(eval,epochs,lineSearchBackProp);

        }
        else {
            BackPropOptimizer opt = new BackPropOptimizer(this,lr,epochs);
            opt.optimize(eval,epochs,lineSearchBackProp);

        }
    }

    /**
     * Backpropagation of errors for weights
     * @param lr the learning rate to use
     * @param epochs  the number of epochs to iterate (this is already called in finetune)
     */
    public void backProp(double lr,int epochs) {
        backProp(lr,epochs,null);

    }

    public Map<Integer, ActivationFunction> getActivationFunctionForLayer() {
        return activationFunctionForLayer;
    }

    public void setActivationFunctionForLayer(Map<Integer, ActivationFunction> activationFunctionForLayer) {
        this.activationFunctionForLayer = activationFunctionForLayer;
    }

    public boolean isUseDropConnect() {
        return useDropConnect;
    }

    public void setUseDropConnect(boolean useDropConnect) {
        this.useDropConnect = useDropConnect;
    }

    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various layers(w,hbias NOT VBIAS) and output layer
     * @return the params for this neural net
     */
    public DoubleMatrix params() {
        double[][] list = new double[layers.length * 2 + 2][];

        int deltaCount = 0;

        for(int i = 0; i < list.length - 1; i+= 2) {
            if(deltaCount >= layers.length) {
                list[i] = outputLayer.getW().dup().reshape(1,outputLayer.getW().length).data;
                list[i + 1] = outputLayer.getB().dup().reshape(1,outputLayer.getB().length).data;
                deltaCount++;
            }
            else {
                list[i] = layers[deltaCount].getW().dup().reshape(1,layers[deltaCount].getW().length).data;
                list[i + 1] = layers[deltaCount].gethBias().dup().reshape(1,layers[deltaCount].gethBias().length).data;
                deltaCount++;
            }

        }


        DoubleMatrix ret = new DoubleMatrix(ArrayUtil.combine(list));
        return ret.reshape(1,ret.length);

    }


    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various layers and output layer
     * @return the params for this neural net
     */
    public int numParams() {
        int length = 0;


        for(int i = 0; i < layers.length; i++) {
            length += layers[i].gethBias().length;
            length += layers[i].getW().length;
        }
        length +=  outputLayer.getB().length;
        length += outputLayer.getW().length;

        return length;

    }

    /**
     * Packs a set of matrices in to one vector,
     * where the matrices in this case are the w,hbias at each layer
     * and the output layer w,bias
     * @return a singular matrix of all of the layers packed in to one matrix
     */
    public  DoubleMatrix pack() {
        List<Pair<DoubleMatrix,DoubleMatrix>> vWvB = new ArrayList<>();
        for(int i = 0; i < layers.length; i++) {
            vWvB.add(new Pair<>(layers[i].getW(),layers[i].gethBias()));
        }

        vWvB.add(new Pair<>(outputLayer.getW(), outputLayer.getB()));

        return pack(vWvB);

    }

    /**
     * Packs a set of matrices in to one vector
     * @param layers the layers to pack
     * @return a singular matrix of all of the layers packed in to one matrix
     */
    public  DoubleMatrix pack(List<Pair<DoubleMatrix,DoubleMatrix>> layers) {
        if(layers.size() != this.layers.length + 1)
            throw new IllegalArgumentException("Illegal number of layers passed in. Was " + layers.size() + " when should have been " + (this.layers.length + 1));


        double[][] allMatrices = new double[(this.layers.length + 1) * 2][];

        int count = 0;
        for(int i = 0; i < this.layers.length; i++) {
            allMatrices[count++] = layers.get(i).getFirst().data;
            if(allMatrices[count - 1].length != getLayers()[i].getW().length)
                throw new IllegalArgumentException("Unable to convert the given matrix to the given length. not of same length as W. Length is " +allMatrices[count].length + " when should be " + getLayers()[i].getW().length);
            allMatrices[count++] = layers.get(i).getSecond().data;
            if(allMatrices[count - 1].length != getLayers()[i].gethBias().length)
                throw new IllegalArgumentException("Unable to convert the given matrix to the given length. not of same length as W. Length is " +allMatrices[count].length + " when should be " + getLayers()[i].gethBias().length);

        }

        allMatrices[count++] = layers.get(layers.size() - 1).getFirst().data;
        allMatrices[count++] = layers.get(layers.size() - 1).getSecond().data;



        DoubleMatrix ret =  new DoubleMatrix(ArrayUtil.combine(allMatrices));
        int params = numParams();

        if(ret.length != params)
            throw new IllegalStateException("Returning an invalid matrix of length " + ret.length + " when should have been " + params);


        return ret.reshape(1,ret.length);
    }


    /**
     * Unpacks a parameter matrix in to a
     * set of pairs(w,hbias)
     * triples with layer wise
     * @param param the param vector
     * @return a segmented list of the param vector
     */
    public  List<Pair<DoubleMatrix,DoubleMatrix>> unPack(DoubleMatrix param) {
        //more sanity checks!
        int numParams = numParams();
        if(param.length != numParams)
            throw new IllegalArgumentException("Parameter vector not equal of length to " + numParams);
        if(param.rows != 1)
            param = param.reshape(1,param.length);
        List<Pair<DoubleMatrix,DoubleMatrix>> ret = new ArrayList<>();
        int curr = 0;
        for(int i = 0; i < layers.length; i++) {
            int layerLength = layers[i].getW().length + layers[i].gethBias().length;
            DoubleMatrix subMatrix = param.get(RangeUtils.all(),RangeUtils.interval(curr,curr + layerLength));
            DoubleMatrix weightPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(0,layers[i].getW().length));

            int beginHBias = layers[i].getW().length;
            int endHbias = subMatrix.length;
            DoubleMatrix hBiasPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(beginHBias,endHbias));
            int layerLengthSum = weightPortion.length + hBiasPortion.length;
            if(layerLengthSum != layerLength) {
                if(hBiasPortion.length != layers[i].gethBias().length)
                    throw new IllegalStateException("Hidden bias on layer " + i + " was off");
                if(weightPortion.length != layers[i].getW().length)
                    throw new IllegalStateException("Weight portion on layer " + i + " was off");

            }
            ret.add(new Pair<>(weightPortion.reshape(layers[i].getW().rows,layers[i].getW().columns),hBiasPortion.reshape(layers[i].gethBias().rows,layers[i].gethBias().columns)));
            curr += layerLength;
        }

        int outputLayerWeightLength = outputLayer.getW().length;
        int biasLength = outputLayer.getB().length;
        int layerLength = outputLayerWeightLength + biasLength;

        DoubleMatrix subMatrix = param.get(RangeUtils.all(),RangeUtils.interval(curr,curr + layerLength));
        DoubleMatrix weightPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(0,outputLayer.getW().length));
        DoubleMatrix hBiasPortion = subMatrix.get(RangeUtils.all(),RangeUtils.interval(weightPortion.length,subMatrix.length));
        ret.add(new Pair<>(weightPortion.reshape(outputLayer.getW().rows,outputLayer.getW().columns),hBiasPortion.reshape(outputLayer.getB().rows,outputLayer.getB().columns)));


        return ret;
    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     * @return whether the training should converge or not
     */
    protected List<Pair<DoubleMatrix,DoubleMatrix>> backPropGradient() {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);

        //precompute deltas
        List<DoubleMatrix> deltas = new ArrayList<>();
        //compute derivatives and gradients given activations
        computeDeltas(deltas);
        List<Pair<DoubleMatrix,DoubleMatrix>> vWvB = new ArrayList<>();
        for(int i = 0; i < layers.length; i++) {
            vWvB.add(new Pair<>(layers[i].getW(),layers[i].gethBias()));
        }

        vWvB.add(new Pair<>(outputLayer.getW(), outputLayer.getB()));


        List<Pair<DoubleMatrix,DoubleMatrix>> list = new ArrayList<>();

        for(int l = 0; l < getnLayers(); l++) {
            DoubleMatrix gradientChange = deltas.get(l);
            if(gradientChange.length != getLayers()[l].getW().length)
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            DoubleMatrix deltaColumnSums = deltas.get(l).columnMeans();



            list.add(new Pair<>(gradientChange,deltaColumnSums));


        }

        DoubleMatrix logLayerGradient = deltas.get(getnLayers());
        DoubleMatrix biasGradient = deltas.get(getnLayers()).columnMeans();


        if(mask == null)
            initMask();



        list.add(new Pair<>(logLayerGradient,biasGradient));
        DoubleMatrix gradient = pack(list);
        gradient.addi(mask.mul(params()).mul(l2));


        list = unPack(gradient);

        return list;

    }


    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     * @return whether the training should converge or not
     */
    protected List<Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>>> backPropGradient2() {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);

        //precompute deltas
        List<Pair<DoubleMatrix,DoubleMatrix>> deltas = computeDeltas2();


        List<Pair<Pair<DoubleMatrix,DoubleMatrix>,Pair<DoubleMatrix,DoubleMatrix>>> list = new ArrayList<>();
        List<Pair<DoubleMatrix,DoubleMatrix>> grad = new ArrayList<>();
        List<Pair<DoubleMatrix,DoubleMatrix>> preCon = new ArrayList<>();

        for(int l = 0; l < deltas.size(); l++) {
            DoubleMatrix gradientChange = deltas.get(l).getFirst();
            DoubleMatrix preConGradientChange = deltas.get(l).getSecond();


            if(l < getLayers().length && gradientChange.length != getLayers()[l].getW().length)
                throw new IllegalStateException("Gradient change not equal to weight change");
            else if(l == getLayers().length && gradientChange.length != outputLayer.getW().length)
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            DoubleMatrix deltaColumnSums = deltas.get(l).getFirst().columnMeans();
            DoubleMatrix preConColumnSums = deltas.get(l).getSecond().columnMeans();

            grad.add(new Pair<>(gradientChange,deltaColumnSums));
            preCon.add(new Pair<>(preConGradientChange,preConColumnSums));
            if(l < getLayers().length && deltaColumnSums.length != layers[l].gethBias().length)
                throw new IllegalStateException("Bias change not equal to weight change");
            else if(l == getLayers().length && deltaColumnSums.length != outputLayer.getB().length)
                throw new IllegalStateException("Bias change not equal to weight change");





        }

        DoubleMatrix g = pack(grad);
        DoubleMatrix con = pack(preCon);
        DoubleMatrix theta = params();


        if(mask == null)
            initMask();

        g.addi(theta.mul(l2).muli(mask));

        DoubleMatrix conAdd = powi(mask.mul(l2).add(valueMatrixOf(dampingFactor, g.rows, g.columns)), 3.0 / 4.0);

        con.addi(conAdd);

        List<Pair<DoubleMatrix,DoubleMatrix>> gUnpacked = unPack(g);

        List<Pair<DoubleMatrix,DoubleMatrix>> conUnpacked = unPack(con);

        for(int i = 0; i < gUnpacked.size(); i++)
            list.add(new Pair<>(gUnpacked.get(i),conUnpacked.get(i)));



        return list;

    }





    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last layers weights
     * to revert to in case of convergence, the learning rate being used to train
     * and the current epoch
     * @param v the v in gaussian newton vector g * v
     * @return whether the training should converge or not
     */
    protected List<Pair<DoubleMatrix,DoubleMatrix>> backPropGradientR(DoubleMatrix v) {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);
        if(mask == null)
            initMask();
        //precompute deltas
        List<DoubleMatrix> deltas = computeDeltasR(v);
        //compute derivatives and gradients given activations



        List<Pair<DoubleMatrix,DoubleMatrix>> list = new ArrayList<>();

        for(int l = 0; l < getnLayers(); l++) {
            DoubleMatrix gradientChange = deltas.get(l);
            if(isNaN(gradientChange)) {
                log.info("Attempted v back prop was " + v.sum());
                throw new IllegalStateException("Gradient change was NaN");

            }
            if(gradientChange.length != getLayers()[l].getW().length)
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            DoubleMatrix deltaColumnSums = deltas.get(l).columnMeans();
            if(deltaColumnSums.length != layers[l].gethBias().length)
                throw new IllegalStateException("Bias change not equal to weight change");



            list.add(new Pair<>(gradientChange, deltaColumnSums));


        }

        DoubleMatrix logLayerGradient = deltas.get(getnLayers());
        DoubleMatrix biasGradient = deltas.get(getnLayers()).columnMeans();

        list.add(new Pair<>(logLayerGradient,biasGradient));

        DoubleMatrix pack = pack(list).addi(mask.mul(l2).mul(v)).addi(v.mul(dampingFactor));
        return unPack(pack);

    }




    /**
     * Run SGD based on the given labels
     * @param iter fine tune based on the labels
     * @param lr the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DataSetIterator iter,double lr, int iterations) {
        iter.reset();

        while(iter.hasNext()) {
            DataSet data = iter.next();
            if(data.getFirst() == null || data.getSecond() == null)
                break;

            setInput(data.getFirst());
            setLabels(data.getSecond());
            feedForward();
            this.fineTuneLearningRate = lr;
            optimizer = new MultiLayerNetworkOptimizer(this,lr);
            optimizer.optimize(data.getSecond(), lr,iterations);
        }



    }

    /**
     * Run training algorithm based on the datastet iterator
     * @param iter the labels to use
     * @param lr the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DataSetIterator iter,double lr, int iterations,TrainingEvaluator eval) {
        iter.reset();

        while(iter.hasNext()) {
            DataSet data = iter.next();
            if(data.getFirst() == null || data.getSecond() == null)
                break;


            setInput(data.getFirst());
            setLabels(data.getSecond());


            this.fineTuneLearningRate = lr;
            optimizer = new MultiLayerNetworkOptimizer(this,lr);
            optimizer.optimize(this.labels, lr,iterations,eval);
        }


    }







    /**
     * Run SGD based on the given labels
     * @param labels the labels to use
     * @param lr the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DoubleMatrix labels,double lr, int iterations) {
        this.labels = labels;
        feedForward();
        complainAboutMissMatchedMatrices(labels,outputLayer.getInput());
        if(labels != null)
            this.labels = labels;
        this.fineTuneLearningRate = lr;
        optimizer = new MultiLayerNetworkOptimizer(this,lr);
        optimizer.optimize(this.labels, lr,iterations);

    }

    /**
     * Run training algorithm based on the given labels
     * @param labels the labels to use
     * @param lr the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DoubleMatrix labels,double lr, int iterations,TrainingEvaluator eval) {
        feedForward();

        if(labels != null)
            this.labels = labels;

        this.fineTuneLearningRate = lr;
        optimizer = new MultiLayerNetworkOptimizer(this,lr);
        optimizer.optimize(this.labels, lr,iterations,eval);
    }





    /**
     * Returns the predictions for each example in the dataset
     * @param d the matrix to predict
     * @return the prediction for the dataset
     */
    public int[] predict(DoubleMatrix d) {
        DoubleMatrix output = output(d);
        int[] ret = new int[d.rows];
        for(int i = 0; i < ret.length; i++)
            ret[i] = SimpleBlas.iamax(output);
        return ret;
    }

    /**
     * Label the probabilities of the input
     * @param x the input to label
     * @return a vector of probabilities
     * given each label.
     *
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public DoubleMatrix output(DoubleMatrix x) {
        List<DoubleMatrix> activations = feedForward(x);
        if(columnSums != null) {
            for(int i = 0; i < x.columns; i++) {
                DoubleMatrix col = x.getColumn(i);
                col = col.div(columnSums.get(0,i));
                x.putColumn(i, col);
            }
        }

        if(columnMeans != null) {
            for(int i = 0; i < x.columns; i++) {
                DoubleMatrix col = x.getColumn(i);
                col = col.sub(columnMeans.get(0,i));
                x.putColumn(i, col);
            }
        }

        if(columnStds != null) {
            for(int i = 0; i < x.columns; i++) {
                DoubleMatrix col = x.getColumn(i);
                col = col.div(columnStds.get(0,i));
                x.putColumn(i, col);
            }
        }

        //last activation is input
        DoubleMatrix predicted = activations.get(activations.size() - 1);
        return predicted;
    }

    /**
     * Reconstructs the input.
     * This is equivalent functionality to a
     * deep autoencoder.
     * @param x the input to reconstruct
     * @param layerNum the layer to output for encoding
     * @return a reconstructed matrix
     * relative to the size of the last hidden layer.
     * This is great for data compression and visualizing
     * high dimensional data (or just doing dimensionality reduction).
     *
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public DoubleMatrix sampleHiddenGivenVisible(DoubleMatrix x,int layerNum) {
        DoubleMatrix currInput = x;
        List<DoubleMatrix> forward = feedForward(currInput);
        return getLayers()[layerNum - 1].sampleHiddenGivenVisible(forward.get(layerNum - 1)).getSecond();
    }

    /**
     * Reconstructs the input.
     * This is equivalent functionality to a
     * deep autoencoder.
     * @param x the input to reconstruct
     * @param layerNum the layer to output for encoding
     * @return a reconstructed matrix
     * relative to the size of the last hidden layer.
     * This is great for data compression and visualizing
     * high dimensional data (or just doing dimensionality reduction).
     *
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public DoubleMatrix reconstruct(DoubleMatrix x,int layerNum) {
        DoubleMatrix currInput = x;
        List<DoubleMatrix> forward = feedForward(currInput);
        return forward.get(layerNum - 1);
    }

    /**
     * Reconstruct from the final layer
     * @param x the input to reconstruct
     * @return the reconstructed input
     */
    public DoubleMatrix reconstruct(DoubleMatrix x) {
        return reconstruct(x,sigmoidLayers.length);
    }



    @Override
    public void write(OutputStream os) {
        SerializationUtils.writeObject(this, os);
    }

    /**
     * Load (using {@link ObjectInputStream}
     * @param is the input stream to load from (usually a file)
     */
    @Override
    public void load(InputStream is) {
        BaseMultiLayerNetwork loaded = SerializationUtils.readObject(is);
        update(loaded);
    }



    /**
     * Assigns the parameters of this model to the ones specified by this
     * network. This is used in loading from input streams, factory methods, etc
     * @param network the network to get parameters from
     */
    public  void update(BaseMultiLayerNetwork network) {

        if(network.layers != null && network.getnLayers() > 0) {
            this.setnLayers(network.getLayers().length);
            this.layers = new NeuralNetwork[network.getLayers().length];
            for(int i = 0; i < layers.length; i++)
                if((this.getnLayers()) > i && (network.getnLayers() > i)) {
                    if(network.getLayers()[i] == null) {
                        throw new IllegalStateException("Will not clone uninitialized network, layer " + i + " of network was null");
                    }

                    this.getLayers()[i] = network.getLayers()[i].clone();

                }
        }
        this.sampleOrActivate = network.sampleOrActivate;
        this.layerLearningRates = network.layerLearningRates;
        this.normalizeByInputRows = network.normalizeByInputRows;
        this.useAdaGrad = network.useAdaGrad;
        this.outputLayerDropout = network.outputLayerDropout;
        this.hiddenLayerSizes = network.hiddenLayerSizes;
        if(network.outputLayer != null)
            this.outputLayer = network.outputLayer.clone();
        this.nIns = network.nIns;
        this.nOuts = network.nOuts;
        this.rng = network.rng;
        this.dist = network.dist;
        this.renderByLayer = network.renderByLayer;
        this.activation = network.activation;
        this.useRegularization = network.useRegularization;
        this.columnMeans = network.columnMeans;
        this.columnStds = network.columnStds;
        this.columnSums = network.columnSums;
        this.errorTolerance = network.errorTolerance;
        this.renderWeightsEveryNEpochs = network.renderWeightsEveryNEpochs;
        this.forceNumEpochs = network.forceNumEpochs;
        this.input = network.input;
        this.l2 = network.l2;
        this.labels =  network.labels;
        this.momentum = network.momentum;
        this.learningRateUpdate = network.learningRateUpdate;
        this.shouldBackProp = network.shouldBackProp;
        this.weightTransforms = network.weightTransforms;
        this.sparsity = network.sparsity;
        this.visibleBiasTransforms = network.visibleBiasTransforms;
        this.hiddenBiasTransforms = network.hiddenBiasTransforms;
        this.dropOut = network.dropOut;
        this.optimizationAlgorithm = network.optimizationAlgorithm;
        this.lossFunction = network.lossFunction;
        this.outputLayer = network.outputLayer;
        this.outputActivationFunction = network.outputActivationFunction;
        this.lossFunctionByLayer = network.lossFunctionByLayer;
        this.outputLossFunction = network.outputLossFunction;
        this.useDropConnect = network.useDropConnect;
        this.useGaussNewtonVectorProductBackProp = network.useGaussNewtonVectorProductBackProp;
        this.constrainGradientToUnitNorm = network.constrainGradientToUnitNorm;

        if(network.sigmoidLayers != null && network.sigmoidLayers.length > 0) {
            this.sigmoidLayers = new HiddenLayer[network.sigmoidLayers.length];
            for(int i = 0; i < sigmoidLayers.length; i++)
                this.getSigmoidLayers()[i] = network.getSigmoidLayers()[i].clone();

        }


    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     * @param input the input to score
     * @param labels the true labels
     * @return the score for the given input,label pairs
     */
    public double score(DoubleMatrix input,DoubleMatrix labels) {
        feedForward(input);
        setLabels(labels);
        return score();
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    public double score(DataSet data) {
        feedForward(data.getFirst());
        setLabels(data.getSecond());
        return score();
    }


    /**
     * Score of the model (relative to the objective function)
     * @return the score of the model (relative to the objective function)
     */
    public  double score() {
        feedForward();
        return outputLayer.score();
    }
    /**
     * Score of the model (relative to the objective function)
     * @param param the current parameters
     * @return the score of the model (relative to the objective function)
     */
    public  double score(DoubleMatrix param) {
        DoubleMatrix params = params();
        setParameters(param);
        double ret =  score();
        double regCost = 0.5 * l2 * powi(mask.mul(param),2).sum();
        setParameters(params);
        return ret + regCost;
    }


    /**
     * Pretrain with a data set iterator.
     * This will run through each neural net at a time and train on the input.
     * @param iter the iterator to use
     * @param otherParams
     */
    public abstract  void pretrain(DataSetIterator iter, Object[] otherParams);

    /**
     * Pretrain the network with the given parameters
     * @param input the input to train ons
     * @param otherParams the other parameters for child classes (algorithm specific parameters such as corruption level for SDA)
     */
    public abstract void pretrain(DoubleMatrix input,Object[] otherParams);


    protected void applyTransforms() {
        if(layers == null || layers.length < 1) {
            throw new IllegalStateException("Layers not initialized");
        }

        for(int i = 0; i < layers.length; i++) {
            if(weightTransforms.containsKey(i))
                layers[i].setW(weightTransforms.get(i).apply(layers[i].getW()));
            if(hiddenBiasTransforms.containsKey(i))
                layers[i].sethBias(getHiddenBiasTransforms().get(i).apply(layers[i].gethBias()));
            if(this.visibleBiasTransforms.containsKey(i))
                layers[i].setvBias(getVisibleBiasTransforms().get(i).apply(layers[i].getvBias()));
        }
    }




    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.rbm.RBM} for continuous inputs.
     *
     * @param input the input to the layer
     * @param nVisible the number of visible inputs
     * @param nHidden the number of hidden units
     * @param W the weight vector
     * @param hbias the hidden bias
     * @param vBias the visible bias
     * @param rng the rng to use (THiS IS IMPORTANT; YOU DO NOT WANT TO HAVE A MIS REFERENCED RNG OTHERWISE NUMBERS WILL BE MEANINGLESS)
     * @param index the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.rbm.RBM}
     */
    public abstract NeuralNetwork createLayer(DoubleMatrix input,int nVisible,int nHidden, DoubleMatrix W,DoubleMatrix hbias,DoubleMatrix vBias,RandomGenerator rng,int index);


    public abstract NeuralNetwork[] createNetworkLayers(int numLayers);


    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overridden
     * for other kinds of hidden units
     * @param nIn the number of inputs
     * @param nOut the number of outputs
     * @param activation the activation function for the layer
     * @param rng the rng to use for sampling
     * @param layerInput the layer starting input
     * @param dist the probability distribution to use
     * for generating weights
     * @return a hidden layer with the given parameters
     */
    public  HiddenLayer createHiddenLayer(int index,int nIn,int nOut,ActivationFunction activation,RandomGenerator rng,DoubleMatrix layerInput,RealDistribution dist) {
        return new HiddenLayer.Builder()
                .nIn(nIn).nOut(nOut).concatBiases(concatBiases).weightInit(weightInitByLayer.get(index) != null ?weightInitByLayer.get(index) : weightInit)
                .withActivation(activationFunctionForLayer.get(index) != null ? activationFunctionForLayer.get(index) : activation)
                .withRng(rng).withInput(layerInput).dist(dist)
                .build();

    }



    /**
     * Merges this network with the other one.
     * This is a weight averaging with the update of:
     * a += b - a / n
     * where a is a matrix on the network
     * b is the incoming matrix and n
     * is the batch size.
     * This update is performed across the network layers
     * as well as hidden layers and logistic layers
     *
     * @param network the network to merge with
     * @param batchSize the batch size (number of training examples)
     * to average by
     */
    public void merge(BaseMultiLayerNetwork network,int batchSize) {
        if(network.getnLayers() != getnLayers())
            throw new IllegalArgumentException("Unable to merge networks that are not of equal length");
        for(int i = 0; i < getnLayers(); i++) {
            NeuralNetwork n = layers[i];
            NeuralNetwork otherNetwork = network.layers[i];
            n.merge(otherNetwork, batchSize);
            //tied weights: must be updated at the same time
            getSigmoidLayers()[i].setB(n.gethBias());
            getSigmoidLayers()[i].setW(n.getW());

        }

        getOutputLayer().merge(network.outputLayer, batchSize);
    }



    public  DoubleMatrix getLabels() {
        return labels;
    }

    public OutputLayer getOutputLayer() {
        return outputLayer;
    }

    /**
     * Note that if input isn't null
     * and the layers are null, this is a way
     * of initializing the neural network
     * @param input
     */
    public  void setInput(DoubleMatrix input) {
        if(input != null && this.layers == null)
            this.initializeLayers(input);
        this.input = input;

    }

    private void initMask() {
        setMask(DoubleMatrix.ones(1,pack().length));
        List<Pair<DoubleMatrix,DoubleMatrix>> mask = unPack(getMask());
        for(int i = 0; i < mask.size(); i++)
            mask.get(i).setSecond(DoubleMatrix.zeros(mask.get(i).getSecond().rows,mask.get(i).getSecond().columns));
        setMask(pack(mask));

    }

    public boolean isShouldBackProp() {
        return shouldBackProp;
    }


    public OptimizationAlgorithm getOptimizationAlgorithm() {
        return optimizationAlgorithm;
    }

    public void setOptimizationAlgorithm(OptimizationAlgorithm optimizationAlgorithm) {
        this.optimizationAlgorithm = optimizationAlgorithm;
    }

    public LossFunction getLossFunction() {
        return lossFunction;
    }

    public void setLossFunction(LossFunction lossFunction) {
        this.lossFunction = lossFunction;
    }


    public  DoubleMatrix getInput() {
        return input;
    }

    public  synchronized HiddenLayer[] getSigmoidLayers() {
        return sigmoidLayers;
    }

    public  synchronized NeuralNetwork[] getLayers() {
        return layers;
    }



    public boolean isForceNumEpochs() {
        return forceNumEpochs;
    }



    public DoubleMatrix getColumnSums() {
        return columnSums;
    }

    public void setColumnSums(DoubleMatrix columnSums) {
        this.columnSums = columnSums;
    }

    public  int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public  void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public  RandomGenerator getRng() {
        return rng;
    }

    public  void setRng(RandomGenerator rng) {
        this.rng = rng;
    }

    public  RealDistribution getDist() {
        return dist;
    }

    public  void setDist(RealDistribution dist) {
        this.dist = dist;
    }

    public  MultiLayerNetworkOptimizer getOptimizer() {
        return optimizer;
    }

    public  void setOptimizer(MultiLayerNetworkOptimizer optimizer) {
        this.optimizer = optimizer;
    }

    public  ActivationFunction getActivation() {
        return activation;
    }

    public  void setActivation(ActivationFunction activation) {
        this.activation = activation;
    }


    public  int getRenderWeightsEveryNEpochs() {
        return renderWeightsEveryNEpochs;
    }

    public  void setRenderWeightsEveryNEpochs(
            int renderWeightsEveryNEpochs) {
        this.renderWeightsEveryNEpochs = renderWeightsEveryNEpochs;
    }

    public  Map<Integer, MatrixTransform> getWeightTransforms() {
        return weightTransforms;
    }

    public  void setWeightTransforms(
            Map<Integer, MatrixTransform> weightTransforms) {
        this.weightTransforms = weightTransforms;
    }

    public  double getSparsity() {
        return sparsity;
    }

    public  void setSparsity(double sparsity) {
        this.sparsity = sparsity;
    }

    public  double getLearningRateUpdate() {
        return learningRateUpdate;
    }

    public  void setLearningRateUpdate(double learningRateUpdate) {
        this.learningRateUpdate = learningRateUpdate;
    }

    public  double getErrorTolerance() {
        return errorTolerance;
    }

    public  void setErrorTolerance(double errorTolerance) {
        this.errorTolerance = errorTolerance;
    }

    public  void setLabels(DoubleMatrix labels) {
        this.labels = labels;
    }

    public  void setForceNumEpochs(boolean forceNumEpochs) {
        this.forceNumEpochs = forceNumEpochs;
    }



    public DoubleMatrix getColumnMeans() {
        return columnMeans;
    }

    public void setColumnMeans(DoubleMatrix columnMeans) {
        this.columnMeans = columnMeans;
    }

    public DoubleMatrix getColumnStds() {
        return columnStds;
    }

    public void setColumnStds(DoubleMatrix columnStds) {
        this.columnStds = columnStds;
    }

    public WeightInit getWeightInit() {
        return weightInit;
    }

    public void setWeightInit(WeightInit weightInit) {
        this.weightInit = weightInit;
    }

    public Map<Integer, WeightInit> getWeightInitByLayer() {
        return weightInitByLayer;
    }

    public void setWeightInitByLayer(Map<Integer, WeightInit> weightInitByLayer) {
        this.weightInitByLayer = weightInitByLayer;
    }

    public WeightInit getOutputLayerWeightInit() {
        return outputLayerWeightInit;
    }

    public void setOutputLayerWeightInit(WeightInit outputLayerWeightInit) {
        this.outputLayerWeightInit = outputLayerWeightInit;
    }

    public double getFineTuneLearningRate() {
        return fineTuneLearningRate;
    }

    public void setFineTuneLearningRate(double fineTuneLearningRate) {
        this.fineTuneLearningRate = fineTuneLearningRate;
    }

    public  boolean isUseAdaGrad() {
        return useAdaGrad;
    }

    public  void setUseAdaGrad(boolean useAdaGrad) {
        this.useAdaGrad = useAdaGrad;
    }



    public  boolean isNormalizeByInputRows() {
        return normalizeByInputRows;
    }

    public  void setNormalizeByInputRows(boolean normalizeByInputRows) {
        this.normalizeByInputRows = normalizeByInputRows;
    }

    public double getOutputLayerDropout() {
        return outputLayerDropout;
    }

    public void setOutputLayerDropout(double outputLayerDropout) {
        this.outputLayerDropout = outputLayerDropout;
    }

    public boolean isSampleFromHiddenActivations() {
        return sampleFromHiddenActivations;
    }

    public void setSampleFromHiddenActivations(
            boolean sampleFromHiddenActivations) {
        this.sampleFromHiddenActivations = sampleFromHiddenActivations;
    }


    public double getDropOut() {
        return dropOut;
    }

    public void setDropOut(double dropOut) {
        this.dropOut = dropOut;
    }

    public  Map<Integer, MatrixTransform> getHiddenBiasTransforms() {
        return hiddenBiasTransforms;
    }

    public  Map<Integer, MatrixTransform> getVisibleBiasTransforms() {
        return visibleBiasTransforms;
    }

    public  int getnIns() {
        return nIns;
    }

    public  void setnIns(int nIns) {
        this.nIns = nIns;
    }

    public  int getnOuts() {
        return nOuts;
    }

    public  void setnOuts(int nOuts) {
        this.nOuts = nOuts;
    }

    public  int getnLayers() {
        return layers.length;
    }

    public  void setnLayers(int nLayers) {
        this.layers = createNetworkLayers(nLayers);
    }

    public  double getMomentum() {
        return momentum;
    }

    public  void setMomentum(double momentum) {
        this.momentum = momentum;
    }

    public  double getL2() {
        return l2;
    }

    public  void setL2(double l2) {
        this.l2 = l2;
    }

    public  boolean isUseRegularization() {
        return useRegularization;
    }

    public  void setUseRegularization(boolean useRegularization) {
        this.useRegularization = useRegularization;
    }

    public  void setSigmoidLayers(HiddenLayer[] sigmoidLayers) {
        this.sigmoidLayers = sigmoidLayers;
    }

    public  void setOutputLayer(OutputLayer outputLayer) {
        this.outputLayer = outputLayer;
    }

    public  void setShouldBackProp(boolean shouldBackProp) {
        this.shouldBackProp = shouldBackProp;
    }

    public  void setLayers(NeuralNetwork[] layers) {
        this.layers = layers;
    }

    public Map<Integer, Integer> getRenderByLayer() {
        return renderByLayer;
    }

    public void setRenderByLayer(Map<Integer, Integer> renderByLayer) {
        this.renderByLayer = renderByLayer;
    }

    public boolean isUseGaussNewtonVectorProductBackProp() {
        return useGaussNewtonVectorProductBackProp;
    }

    public void setUseGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
        this.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
    }

    public double getDampingFactor() {
        return dampingFactor;
    }

    public void setDampingFactor(double dampingFactor) {
        this.dampingFactor = dampingFactor;
    }

    public DoubleMatrix getMask() {
        return mask;
    }

    public void setMask(DoubleMatrix mask) {
        this.mask = mask;
    }

    public boolean isConcatBiases() {
        return concatBiases;
    }

    public void setConcatBiases(boolean concatBiases) {
        this.concatBiases = concatBiases;
    }

    public boolean isConstrainGradientToUnitNorm() {
        return constrainGradientToUnitNorm;
    }

    public void setConstrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
        this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
    }

    /**
     * Clears the input from all of the layers
     */
    public void clearInput() {
        this.input = null;
        for(int i = 0; i < layers.length; i++)
            layers[i].clearInput();
        outputLayer.setInput(null);
    }

    public OutputLayer.LossFunction getOutputLossFunction() {
        return outputLossFunction;
    }

    public void setOutputLossFunction(OutputLayer.LossFunction outputLossFunction) {
        this.outputLossFunction = outputLossFunction;
        if(outputLayer != null)
            outputLayer.setLossFunction(outputLossFunction);
    }

    public ActivationFunction getOutputActivationFunction() {
        return outputActivationFunction;
    }

    public void setOutputActivationFunction(ActivationFunction outputActivationFunction) {
        this.outputActivationFunction = outputActivationFunction;
        if(outputLayer != null)
            outputLayer.setActivationFunction(outputActivationFunction);
    }

    public Map<Integer, Double> getLayerLearningRates() {
        return layerLearningRates;
    }

    public void setLayerLearningRates(Map<Integer, Double> layerLearningRates) {
        this.layerLearningRates = layerLearningRates;
    }

    public Map<Integer, LossFunction> getLossFunctionByLayer() {
        return lossFunctionByLayer;
    }

    public void setLossFunctionByLayer(Map<Integer, LossFunction> lossFunctionByLayer) {
        this.lossFunctionByLayer = lossFunctionByLayer;
    }

    public int getResetAdaGradIterations() {
        return resetAdaGradIterations;
    }

    public void setResetAdaGradIterations(int resetAdaGradIterations) {
        this.resetAdaGradIterations = resetAdaGradIterations;
    }

    public Map<Integer, Integer> getResetAdaGradIterationsByLayer() {
        return resetAdaGradIterationsByLayer;
    }

    public void setResetAdaGradIterationsByLayer(Map<Integer, Integer> resetAdaGradIterationsByLayer) {
        this.resetAdaGradIterationsByLayer = resetAdaGradIterationsByLayer;
    }

    public Map<Integer, Map<Integer, Double>> getMomentumAfterByLayer() {
        return momentumAfterByLayer;
    }

    public void setMomentumAfterByLayer(Map<Integer, Map<Integer, Double>> momentumAfterByLayer) {
        this.momentumAfterByLayer = momentumAfterByLayer;
    }

    public Map<Integer, Double> getMomentumAfter() {
        return momentumAfter;
    }

    public void setMomentumAfter(Map<Integer, Double> momentumAfter) {
        this.momentumAfter = momentumAfter;
    }

    public boolean isLineSearchBackProp() {
        return lineSearchBackProp;
    }

    public void setLineSearchBackProp(boolean lineSearchBackProp) {
        this.lineSearchBackProp = lineSearchBackProp;
    }


    /**
     * Sets parameters for the model.
     * This is used to manipulate the weights and biases across
     * all layers (including the output layer)
     * @param params a parameter vector equal 1,numParameters
     */
    public void setParameters(DoubleMatrix params) {
        for(int i = 0; i < getLayers().length; i++) {
            ParamRange range = startIndexForLayer(i);
            DoubleMatrix w = params.get(RangeUtils.all(),RangeUtils.interval(range.getwStart(),range.getwEnd()));
            DoubleMatrix bias = params.get(RangeUtils.all(),RangeUtils.interval(range.getBiasStart(),range.getBiasEnd()));
            int rows = getLayers()[i].getW().rows,columns = getLayers()[i].getW().columns;
            getLayers()[i].setW(w.reshape(rows,columns));
            getLayers()[i].sethBias(bias.reshape(getLayers()[i].gethBias().rows,getLayers()[i].gethBias().columns));
        }


        ParamRange range = startIndexForLayer(getLayers().length);
        DoubleMatrix w = params.get(RangeUtils.all(),RangeUtils.interval(range.getwStart(),range.getwEnd()));
        DoubleMatrix bias = params.get(RangeUtils.all(),RangeUtils.interval(range.getBiasStart(),range.getBiasEnd()));
        int rows = getOutputLayer().getW().rows,columns =  getOutputLayer().getW().columns;
        getOutputLayer().setW(w.reshape(rows, columns));
        getOutputLayer().setB(bias.reshape(getOutputLayer().getB().rows, getOutputLayer().getB().columns));



    }


    /**
     * Returns a start index for a given layer (neural net or outputlayer)
     * @param layer the layer to get the index for
     * @return the index for the layer
     */
    public ParamRange startIndexForLayer(int layer) {
        int start = 0;
        for(int i = 0; i < layer; i++) {
            start += getLayers()[i].getW().length;
            start += getLayers()[i].gethBias().length;
        }
        if(layer < getLayers().length) {
            int wEnd = start + getLayers()[layer].getW().length;
            return new ParamRange(start,wEnd,wEnd,wEnd + getLayers()[layer].gethBias().length);

        }

        else {
            int wEnd = start + getOutputLayer().getW().length;
            return new ParamRange(start,wEnd,wEnd,wEnd + getOutputLayer().getB().length);

        }


    }

    public static class ParamRange implements  Serializable {
        private int wStart,wEnd,biasStart,biasEnd;

        private ParamRange(int wStart, int wEnd, int biasStart, int biasEnd) {
            this.wStart = wStart;
            this.wEnd = wEnd;
            this.biasStart = biasStart;
            this.biasEnd = biasEnd;
        }

        public int getwStart() {
            return wStart;
        }

        public void setwStart(int wStart) {
            this.wStart = wStart;
        }

        public int getwEnd() {
            return wEnd;
        }

        public void setwEnd(int wEnd) {
            this.wEnd = wEnd;
        }

        public int getBiasStart() {
            return biasStart;
        }

        public void setBiasStart(int biasStart) {
            this.biasStart = biasStart;
        }

        public int getBiasEnd() {
            return biasEnd;
        }

        public void setBiasEnd(int biasEnd) {
            this.biasEnd = biasEnd;
        }
    }








    public static class Builder<E extends BaseMultiLayerNetwork> {
        protected Class<? extends BaseMultiLayerNetwork> clazz;
        private Map<Integer,Double> layerLearningRates = new HashMap<>();
        private int nIns;
        private int[] hiddenLayerSizes;
        private int nOuts;
        private int nLayers;
        private RandomGenerator rng = new MersenneTwister(1234);
        private DoubleMatrix input,labels;
        private ActivationFunction activation;
        private int renderWeithsEveryNEpochs = -1;
        private double l2 = 0.01;
        private boolean useRegularization = false;
        private double momentum;
        private RealDistribution dist;
        protected Map<Integer,MatrixTransform> weightTransforms = new HashMap<>();
        protected boolean backProp = true;
        protected boolean shouldForceEpochs = false;
        private double sparsity = 0;
        private Map<Integer,MatrixTransform> hiddenBiasTransforms = new HashMap<>();
        private Map<Integer,MatrixTransform> visibleBiasTransforms = new HashMap<>();
        private boolean useAdaGrad = true;
        private boolean normalizeByInputRows = true;
        private boolean useHiddenActivationsForwardProp = true;
        private double dropOut = 0;
        private Map<Integer,ActivationFunction> activationForLayer = new HashMap<>();
        private LossFunction lossFunction = LossFunction.RECONSTRUCTION_CROSSENTROPY;
        private OptimizationAlgorithm optimizationAlgo = OptimizationAlgorithm.CONJUGATE_GRADIENT;
        private OutputLayer.LossFunction outputLossFunction = OutputLayer.LossFunction.MCXENT;
        private ActivationFunction outputActivationFunction = Activations.softmax();
        private Map<Integer,Integer> renderByLayer = new HashMap<>();
        private Map<Integer,Boolean> sampleOrActivateByLayer = new HashMap<>();
        private Map<Integer,LossFunction> lossFunctionByLayer = new HashMap<>();
        private boolean lineSearchBackProp = false;
        private boolean concatBiases = false;
        //momentum after n iterations
        private Map<Integer,Double> momentumAfter = new HashMap<>();
        //momentum after n iterations by layer
        private Map<Integer,Map<Integer,Double>> momentumAfterByLayer = new HashMap<>();
        //reset adagrad historical gradient after n iterations
        private Map<Integer,Integer> resetAdaGradIterationsByLayer = new HashMap<>();
        //reset adagrad historical gradient after n iterations
        private int resetAdaGradIterations = -1;
        private double outputLayerDropout = 0.0;
        private boolean useDropConnect = false;
        private boolean useGaussNewtonVectorProductBackProp = false;
        private boolean constrainGradientToUnitNorm = false;
        /* weight initialization scheme, this is either a probability distribution or a weighting scheme for initialization */
        private WeightInit weightInit = null;
        /* weight init for layers */
        private Map<Integer,WeightInit> weightInitByLayer = new HashMap<>();
        /* output layer weight intialization */
        private WeightInit outputLayerWeightInit = null;


        /**
         * Output layer weight initialization
         * @param outputLayerWeightInit
         * @return
         */
        public Builder<E> outputLayerWeightInit(WeightInit outputLayerWeightInit) {
            this.outputLayerWeightInit = outputLayerWeightInit;
            return this;
        }

        /**
         * Layer specific weight init
         * @param weightInitByLayer
         * @return
         */
        public Builder<E> weightInitByLayer(Map<Integer,WeightInit> weightInitByLayer) {
            this.weightInitByLayer = weightInitByLayer;
            return this;
        }

        /**
         * Default weight init scheme
         * @param weightInit
         * @return
         */
        public Builder<E> weightInit(WeightInit weightInit) {
            this.weightInit = weightInit;
            return this;
        }


        /**
         * Whether to constrain gradient to unit norm or not
         * @param constrainGradientToUnitNorm
         * @return
         */
        public Builder<E> constrainGradientToUnitNorm(boolean constrainGradientToUnitNorm) {
            this.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
            return this;
        }

        /**
         * Whether to concate biases or add them
         * @param concatBiases true if concatneating biases,
         *        false add them
         * @return
         */
        public Builder<E> concatBiases(boolean concatBiases) {
            this.concatBiases = concatBiases;
            return this;
        }

        /**
         * Use gauss newton back prop - this is for hessian free
         * @param useGaussNewtonVectorProductBackProp whether to use gauss newton vector backprop
         * @return
         */
        public Builder<E> useGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
            this.useGaussNewtonVectorProductBackProp= useGaussNewtonVectorProductBackProp;
            return this;
        }


        /**
         * Use drop connect on activations or not
         * @param useDropConnect use drop connect or not
         * @return builder pattern
         */
        public Builder<E> useDropConnection(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }


        /**
         * Output layer drop out
         * @param outputLayerDropout
         * @return
         */
        public Builder<E> outputLayerDropout(double outputLayerDropout) {
            this.outputLayerDropout = outputLayerDropout;
            return this;
        }


        /**
         * Reset the adagrad epochs  after n iterations
         * @param resetAdaGradIterations the number of iterations to reset adagrad after
         * @return
         */
        public Builder<E> resetAdaGradIterations(int resetAdaGradIterations) {
            this.resetAdaGradIterations = resetAdaGradIterations;
            return this;
        }

        /**
         * Reset map for adagrad historical gradient by layer
         * @param resetAdaGradEpochsByLayer
         * @return
         */
        public Builder<E> resetAdaGradEpochsByLayer(Map<Integer,Integer> resetAdaGradEpochsByLayer) {
            this.resetAdaGradIterationsByLayer = resetAdaGradEpochsByLayer;
            return this;
        }

        /**
         * Sets the momentum to the specified value for a given layer after a specified number of iterations
         * @param momentumAfterByLayer the by layer momentum changes
         * @return
         */
        public Builder<E> momentumAfterByLayer(Map<Integer,Map<Integer,Double>> momentumAfterByLayer) {
            this.momentumAfterByLayer = momentumAfterByLayer;
            return this;
        }

        /**
         * Set the momentum to the value after the desired number of iterations
         * @param momentumAfter the momentum after n iterations
         * @return
         */
        public Builder<E> momentumAfter(Map<Integer,Double> momentumAfter) {
            this.momentumAfter = momentumAfter;
            return this;
        }

        /**
         * Whether to use line search back prop instead of SGD
         * @param lineSearchBackProp
         * @return
         */
        public Builder<E> lineSearchBackProp(boolean lineSearchBackProp) {
            this.lineSearchBackProp = lineSearchBackProp;
            return this;
        }


        /**
         * Loss function by layer
         * @param lossFunctionByLayer the loss function per layer
         * @return builder pattern
         */

        public Builder<E> lossFunctionByLayer(Map<Integer,LossFunction> lossFunctionByLayer) {
            this.lossFunctionByLayer = lossFunctionByLayer;
            return this;
        }


        /**
         * Sample or activate by layer allows for deciding to sample or just pass straight activations
         * for each layer
         * @param sampleOrActivateByLayer
         * @return
         */
        public Builder<E> sampleOrActivateByLayer( Map<Integer, Boolean> sampleOrActivateByLayer) {
            this.sampleOrActivateByLayer = sampleOrActivateByLayer;
            return this;
        }

        /**
         * Override rendering for a given layer only
         * @param renderByLayer
         * @return
         */
        public Builder renderByLayer(Map<Integer,Integer> renderByLayer) {
            this.renderByLayer = renderByLayer;
            return this;
        }

        /**
         * Override learning rate by layer only
         * @param learningRates
         * @return
         */
        public Builder<E> learningRateForLayer(Map<Integer,Double> learningRates) {
            this.layerLearningRates.putAll(learningRates);
            return this;
        }

        public Builder<E> activateForLayer(Map<Integer,ActivationFunction> activationForLayer) {
            this.activationForLayer.putAll(activationForLayer);
            return this;
        }

        public Builder activateForLayer(int layer,ActivationFunction function) {
            activationForLayer.put(layer,function);
            return this;
        }

        /**
         * Activation function for output layer
         * @param outputActivationFunction the output activation function to use
         * @return builder pattern
         */
        public Builder<E> withOutputActivationFunction(ActivationFunction outputActivationFunction) {
            this.outputActivationFunction = outputActivationFunction;
            return this;
        }


        /**
         * Output loss function
         * @param outputLossFunction the output loss function
         * @return
         */
        public Builder<E> withOutputLossFunction(OutputLayer.LossFunction outputLossFunction) {
            this.outputLossFunction = outputLossFunction;
            return this;
        }

        /**
         * Which optimization algorithm to use with neural nets and Logistic regression
         * @param optimizationAlgo which optimization algorithm to use with
         * neural nets and logistic regression
         * @return builder pattern
         */
        public Builder<E> withOptimizationAlgorithm(OptimizationAlgorithm optimizationAlgo) {
            this.optimizationAlgo = optimizationAlgo;
            return this;
        }

        /**
         * Loss function to use with neural networks
         * @param lossFunction loss function to use with neural networks
         * @return builder pattern
         */
        public Builder<E> withLossFunction(LossFunction lossFunction) {
            this.lossFunction = lossFunction;
            return this;
        }

        /**
         * Whether to use drop out on the neural networks or not:
         * random zero out of examples
         * @param dropOut the dropout to use
         * @return builder pattern
         */
        public Builder<E> withDropOut(double dropOut) {
            this.dropOut = dropOut;
            return this;
        }

        /**
         * Whether to use hidden layer activations or neural network sampling
         * on feed forward pass
         * @param useHiddenActivationsForwardProp true if use hidden activations, false otherwise
         * @return builder pattern
         */
        public Builder<E> sampleFromHiddenActivations(boolean useHiddenActivationsForwardProp) {
            this.useHiddenActivationsForwardProp = useHiddenActivationsForwardProp;
            return this;
        }

        /**
         * Turn this off for full dataset training
         * @param normalizeByInputRows whether to normalize the changes
         * by the number of input rows
         * @return builder pattern
         */
        public Builder<E> normalizeByInputRows(boolean normalizeByInputRows) {
            this.normalizeByInputRows = normalizeByInputRows;
            return this;
        }




        public Builder<E> useAdaGrad(boolean useAdaGrad) {
            this.useAdaGrad = useAdaGrad;
            return this;
        }

        public Builder<E> withSparsity(double sparsity) {
            this.sparsity = sparsity;
            return this;
        }


        public Builder<E> withVisibleBiasTransforms(Map<Integer,MatrixTransform> visibleBiasTransforms) {
            this.visibleBiasTransforms = visibleBiasTransforms;
            return this;
        }

        public Builder<E> withHiddenBiasTransforms(Map<Integer,MatrixTransform> hiddenBiasTransforms) {
            this.hiddenBiasTransforms = hiddenBiasTransforms;
            return this;
        }

        /**
         * Forces use of number of epochs for training
         * SGD style rather than conjugate gradient
         * @return
         */
        public Builder<E> forceEpochs() {
            shouldForceEpochs = true;
            return this;
        }

        /**
         * Disables back propagation
         * @return
         */
        public Builder<E> disableBackProp() {
            backProp = false;
            return this;
        }

        /**
         * Transform the weights at the given layer
         * @param layer the layer to transform
         * @param transform the function used for transformation
         * @return
         */
        public Builder<E> transformWeightsAt(int layer,MatrixTransform transform) {
            weightTransforms.put(layer,transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given layers
         * @param transforms
         * @return
         */
        public Builder<E> transformWeightsAt(Map<Integer,MatrixTransform> transforms) {
            weightTransforms.putAll(transforms);
            return this;
        }

        /**
         * Probability distribution for generating weights
         * @param dist
         * @return
         */
        public Builder<E> withDist(RealDistribution dist) {
            this.dist = dist;
            return this;
        }

        /**
         * Specify momentum
         * @param momentum
         * @return
         */
        public Builder<E> withMomentum(double momentum) {
            this.momentum = momentum;
            return this;
        }

        /**
         * Use l2 reg
         * @param useRegularization
         * @return
         */
        public Builder<E> useRegularization(boolean useRegularization) {
            this.useRegularization = useRegularization;
            return this;
        }

        /**
         * L2 coefficient
         * @param l2
         * @return
         */
        public Builder<E> withL2(double l2) {
            this.l2 = l2;
            return this;
        }

        /**
         * Whether to plot weights or not
         * @param everyN
         * @return
         */
        public Builder<E> renderWeights(int everyN) {
            this.renderWeithsEveryNEpochs = everyN;
            return this;
        }


        /**
         * Pick an activation function, default is sigmoid
         * @param activation
         * @return
         */
        public Builder<E> withActivation(ActivationFunction activation) {
            this.activation = activation;
            return this;
        }

        /**
         * Number of inputs: aka number of columns in your feature matrix
         * @param nIns
         * @return
         */
        public Builder<E> numberOfInputs(int nIns) {
            this.nIns = nIns;
            return this;
        }


        /**
         * Size of the hidden layers
         * @param hiddenLayerSizes
         * @return
         */
        public Builder<E> hiddenLayerSizes(Integer[] hiddenLayerSizes) {
            this.hiddenLayerSizes = new int[hiddenLayerSizes.length];
            this.nLayers = hiddenLayerSizes.length;
            for(int i = 0; i < hiddenLayerSizes.length; i++)
                this.hiddenLayerSizes[i] = hiddenLayerSizes[i];
            return this;
        }

        public Builder<E> hiddenLayerSizes(int[] hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            this.nLayers = hiddenLayerSizes.length;
            return this;
        }

        /**
         * Number of outputs, this can be labels,
         * or one if you're doing regression.
         * @param nOuts
         * @return
         */
        public Builder<E> numberOfOutPuts(int nOuts) {
            this.nOuts = nOuts;
            return this;
        }

        public Builder<E> withRng(RandomGenerator gen) {
            this.rng = gen;
            return this;
        }

        public Builder<E> withInput(DoubleMatrix input) {
            this.input = input;
            return this;
        }

        public Builder<E> withLabels(DoubleMatrix labels) {
            this.labels = labels;
            return this;
        }

        public Builder<E> withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz =  clazz;
            return this;
        }


        @SuppressWarnings("unchecked")
        public E buildEmpty() {
            try {
                E ret = null;
                Constructor<?> c = Dl4jReflection.getEmptyConstructor(clazz);
                c.setAccessible(true);

                ret = (E) c.newInstance();

                return ret;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        @SuppressWarnings("unchecked")
        public E build() {
            try {
                E ret;
                Constructor<?> c = Dl4jReflection.getEmptyConstructor(clazz);
                c.setAccessible(true);

                ret = (E) c.newInstance();
                ret.constrainGradientToUnitNorm = constrainGradientToUnitNorm;
                ret.weightInit = weightInit;
                ret.weightInitByLayer = weightInitByLayer;
                ret.outputLayerWeightInit = outputLayerWeightInit;
                ret.concatBiases = concatBiases;
                ret.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
                ret.setUseDropConnect(useDropConnect);
                ret.setOutputLayerDropout(outputLayerDropout);
                ret.setResetAdaGradIterations(resetAdaGradIterations);
                ret.setResetAdaGradIterationsByLayer(resetAdaGradIterationsByLayer);
                ret.setMomentumAfter(momentumAfter);
                ret.setMomentumAfterByLayer(momentumAfterByLayer);
                ret.setLossFunctionByLayer(lossFunctionByLayer);
                ret.setSampleOrActivate(sampleOrActivateByLayer);
                ret.setRenderByLayer(renderByLayer);
                ret.setNormalizeByInputRows(normalizeByInputRows);
                ret.setInput(this.input);
                ret.setnOuts(this.nOuts);
                ret.setnIns(this.nIns);
                ret.setLabels(this.labels);
                ret.setHiddenLayerSizes(this.hiddenLayerSizes);
                ret.setnLayers(this.nLayers);
                ret.setRng(this.rng);
                ret.setShouldBackProp(this.backProp);
                ret.setSigmoidLayers(new HiddenLayer[ret.getnLayers()]);
                ret.setSampleFromHiddenActivations(useHiddenActivationsForwardProp);
                ret.setInput(this.input);
                ret.setLineSearchBackProp(lineSearchBackProp);
                ret.setMomentum(momentum);
                ret.setLabels(labels);
                ret.activationFunctionForLayer.putAll(activationForLayer);
                ret.setSparsity(sparsity);
                ret.setRenderWeightsEveryNEpochs(renderWeithsEveryNEpochs);
                ret.setL2(l2);
                ret.setForceNumEpochs(shouldForceEpochs);
                ret.setUseRegularization(useRegularization);
                ret.setUseAdaGrad(useAdaGrad);
                ret.setDropOut(dropOut);
                ret.setOptimizationAlgorithm(optimizationAlgo);
                ret.setLossFunction(lossFunction);
                ret.setOutputActivationFunction(outputActivationFunction);
                ret.setOutputLossFunction(outputLossFunction);
                ret.initializeLayers(DoubleMatrix.zeros(1,nIns));
                if(activation != null)
                    ret.setActivation(activation);
                if(dist != null)
                    ret.setDist(dist);
                ret.getWeightTransforms().putAll(weightTransforms);
                ret.getVisibleBiasTransforms().putAll(visibleBiasTransforms);
                ret.getHiddenBiasTransforms().putAll(hiddenBiasTransforms);
                ret.getLayerLearningRates().putAll(layerLearningRates);
                if(hiddenLayerSizes == null)
                    throw new IllegalStateException("Unable to build network, no hidden layer sizes defined");
                return ret;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }

    }


}