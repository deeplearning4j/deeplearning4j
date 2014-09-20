package org.deeplearning4j.nn;



import org.deeplearning4j.models.featuredetectors.autoencoder.AutoEncoder;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.nd4j.linalg.api.activation.ActivationFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.sampling.Sampling;
import org.nd4j.linalg.transformation.MatrixTransform;
import org.nd4j.linalg.util.ArrayUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Layer;


import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.optimize.api.TrainingEvaluator;
import org.deeplearning4j.optimize.optimizers.BackPropOptimizer;
import org.deeplearning4j.optimize.optimizers.BackPropROptimizer;
import org.deeplearning4j.optimize.optimizers.MultiLayerNetworkOptimizer;

import org.deeplearning4j.util.Dl4jReflection;
import org.deeplearning4j.util.SerializationUtils;


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
 * and multiple hidden neuralNets.
 * @author Adam Gibson
 *
 */
public abstract class BaseMultiLayerNetwork implements Serializable,Persistable,Classifier {


    private static Logger log = LoggerFactory.getLogger(BaseMultiLayerNetwork.class);
    private static final long serialVersionUID = -5029161847383716484L;
    //the hidden layer sizes at each layer
    protected int[] hiddenLayerSizes;
    //the hidden neuralNets
    protected Layer[] layers;
    //default training examples and associated neuralNets
    protected INDArray input, labels;
    protected MultiLayerNetworkOptimizer optimizer;
    //sometimes we may need to applyTransformToOrigin weights; this allows a
    //weight applyTransformToOrigin upon layer setup
    protected Map<Integer, MatrixTransform> weightTransforms = new HashMap<>();
    //hidden bias transforms; for initialization
    protected Map<Integer, MatrixTransform> hiddenBiasTransforms = new HashMap<>();
    //visible bias transforms for initialization
    protected Map<Integer, MatrixTransform> visibleBiasTransforms = new HashMap<>();

    protected boolean shouldBackProp = true;
    //whether to only iterate a certain number of epochs
    protected boolean forceNumEpochs = false;
    protected boolean initCalled = false;
    /* Sample if true, otherwise use the straight activation function */
    protected boolean sampleFromHiddenActivations = true;

    protected NeuralNetConfiguration defaultConfiguration;
    protected List<NeuralNetConfiguration> layerWiseConfigurations;

    /*
     * Hinton's Practical guide to RBMS:
     *
     * Learning rate updates over time.
     * Usually when weights have a large fan in (starting point for a random distribution during initialization)
     * you want a smaller update rate.
     *
     * For biases this can be bigger.
     */
    protected float learningRateUpdate = 0.95f;
    /*
     * Any neural networks used as neuralNets.
     * This will always have an equivalent sigmoid layer
     * with shared weight matrices for training.
     *
     * Typically, this is some mix of:
     *        RBMs,Denoising AutoEncoders, or their Continuous counterparts
     */
    protected NeuralNetwork[] neuralNets;

    /*
     * The delta from the previous iteration to this iteration for
     * cross entropy must change >= this amount in order to continue.
     */
    protected float errorTolerance = 0.0001f;


    /**
     * Whether to use conjugate gradient line search for back prop or
     * normal SGD backprop
     */
    protected boolean lineSearchBackProp = false;

    /*
      Binary drop connect mask
     */
    protected INDArray mask;

    /*
      Use drop connect knob
     */
    protected boolean useDropConnect = false;

    /*
       Damping factor for gradient
     */
    protected float dampingFactor = 10;


    /*
      Use gauss newton vector product: hessian free
     */
    protected boolean useGaussNewtonVectorProductBackProp = false;


    /* Reflection/factory constructor */
    protected BaseMultiLayerNetwork() {
    }

    protected BaseMultiLayerNetwork(  int[] hiddenLayerSizes ,int nLayers) {
        this(hiddenLayerSizes,nLayers, null, null);
    }


    protected BaseMultiLayerNetwork( int[] hiddenLayerSizes
            , int nLayers, INDArray input, INDArray labels) {
        this.hiddenLayerSizes = hiddenLayerSizes;
        this.input = input.dup();
        this.labels = labels.dup();

        if (hiddenLayerSizes.length != nLayers)
            throw new IllegalArgumentException("The number of hidden layer sizes must be equivalent to the nLayers argument which is a value of " + nLayers);

        this.setnLayers(nLayers);

        //+ output layer
        this.layers = new org.deeplearning4j.nn.layers.Layer[nLayers + 1];

        intializeConfigurations();

        if (input != null)
            initializeLayers(input);
    }




    protected void intializeConfigurations() {

        if(layerWiseConfigurations == null)
            layerWiseConfigurations = new ArrayList<>();

        if(layers == null)
            layers = new Layer[getnLayers() + 1];
        if(neuralNets == null)
            neuralNets = new NeuralNetwork[getnLayers()];

        if(defaultConfiguration == null)
            defaultConfiguration = new NeuralNetConfiguration.Builder()
                    .build();

        //add a default configuration for each hidden layer + output layer
        for(int i = 0; i < hiddenLayerSizes.length + 1; i++) {
            layerWiseConfigurations.add(defaultConfiguration.clone());
        }


    }


    /* sanity check for hidden layer and inter layer dimensions */
    public void dimensionCheck() {
        assert layers.length == neuralNets.length + 1 : "Missing output layer";

        for (int i = 0; i < getnLayers(); i++) {
            Layer h = layers[i];
            NeuralNetwork network = neuralNets[i];
            LinAlgExceptions.assertSameShape(network.getW(), h.getW());
            LinAlgExceptions.assertSameShape(h.getB(), network.gethBias());

            assert h.conf().getnIn() == h.getW().rows() : "Number of inputs not consistent with number of rows in weight matrix";
            assert h.conf().getnOut() == h.getW().columns()  : "Number of inputs not consistent with number of rows in weight matrix";

            if (i < getnLayers() - 1) {
                Layer h1 = layers[i + 1];
                NeuralNetwork network1 = neuralNets[i + 1];



                assert h1.conf().getnIn() == h1.getW().rows() : "Number of inputs not consistent with number of rows in weight matrix";
                assert h1.conf().getnOut() == h1.getW().columns()  : "Number of inputs not consistent with number of rows in weight matrix";
                assert network1.conf().getnIn() == network1.getW().rows() : "Number of inputs not consistent with number of rows in weight matrix";
                assert network1.conf().getnOut() == network1.getW().columns()  : "Number of inputs not consistent with number of rows in weight matrix";

                if (h1.conf().getnIn() != h.conf().getnOut())
                    throw new IllegalStateException("Invalid structure: hidden layer in for " + (i + 1) + " not equal to number of ins " + i);
                if (network.conf().getnOut() != network1.conf().getnIn())
                    throw new IllegalStateException("Invalid structure: network hidden for " + (i + 1) + " not equal to number of visible " + i);

            }
        }


    }


    /**
     * Transform the data based on the model's output.
     * This can be anything from a number to reconstructions.
     *
     * @param data the data to transform
     * @return the transformed data
     */
    @Override
    public INDArray transform(INDArray data) {
        return output(data);
    }


    public NeuralNetConfiguration getDefaultConfiguration() {
        return defaultConfiguration;
    }

    public void setDefaultConfiguration(NeuralNetConfiguration defaultConfiguration) {
        this.defaultConfiguration = defaultConfiguration;
    }

    public List<NeuralNetConfiguration> getLayerWiseConfigurations() {
        return layerWiseConfigurations;
    }

    public void setLayerWiseConfigurations(List<NeuralNetConfiguration> layerWiseConfigurations) {
        this.layerWiseConfigurations = layerWiseConfigurations;
    }

    /**
     * Base class for initializing the neuralNets based on the input.
     * This is meant for capturing numbers such as input columns or other things.
     *
     * @param input the input matrix for training
     */
    public void initializeLayers(INDArray input) {
        if (input == null)
            throw new IllegalArgumentException("Unable to initialize neuralNets with empty input");

        if (input.columns() != defaultConfiguration.getnIn())
            throw new IllegalArgumentException(String.format("Unable to iterate on number of inputs; columns should be equal to number of inputs. Number of inputs was %d while number of columns was %d", defaultConfiguration.getnIn(), input.columns()));

        if (this.neuralNets == null)
            this.neuralNets = new NeuralNetwork[getnLayers()];

        for (int i = 0; i < hiddenLayerSizes.length; i++)
            if (hiddenLayerSizes[i] < 1)
                throw new IllegalArgumentException("All hidden layer sizes must be >= 1");


        this.input = input.dup();
        if (!initCalled) {
            init();
            log.info("Initializing neuralNets with input of dims " + input.rows() + " x " + input.columns());

        }


    }

    public void init() {
        if(layerWiseConfigurations == null) {
            intializeConfigurations();
        }

        INDArray layerInput = input;
        int inputSize;
        if (getnLayers() < 1)
            throw new IllegalStateException("Unable to createComplex network neuralNets; number specified is less than 1");


        if (this.neuralNets == null || neuralNets == null || this.neuralNets[0] == null || this.neuralNets[0] == null) {
            this.neuralNets = new NeuralNetwork[getnLayers()];
            // construct multi-layer
            for (int i = 0; i < this.getnLayers(); i++) {

                if (i == 0)
                    inputSize = defaultConfiguration.getnIn();
                else
                    inputSize = this.hiddenLayerSizes[i - 1];

                if (i == 0) {
                    layerWiseConfigurations.get(i).setnIn(inputSize);
                    layerWiseConfigurations.get(i).setnOut(this.hiddenLayerSizes[i]);
                    // construct sigmoid_layer
                    layers[i] = createHiddenLayer(i,layerInput);
                } else {
                    if (input != null)
                        layerInput = activationFromPrevLayer(i - 1,layerInput);

                    layerWiseConfigurations.get(i).setnIn(inputSize);
                    layerWiseConfigurations.get(i).setnOut(this.hiddenLayerSizes[i]);
                    // construct sigmoid_layer
                    layers[i] = createHiddenLayer(i, layerInput);



                }

                this.neuralNets[i] = createLayer(layerInput,  this.layers[i].getW(), this.layers[i].getB(), null, i);

            }

        }


        NeuralNetConfiguration last = layerWiseConfigurations.get(layerWiseConfigurations.size() - 1);
        NeuralNetConfiguration secondToLast = layerWiseConfigurations.get(layerWiseConfigurations.size() - 2);
        last.setnIn(secondToLast.getnOut());


        this.layers[layers.length - 1] = new OutputLayer.Builder().configure(last).build();



        dimensionCheck();
        applyTransforms();
        initCalled = true;
        initMask();

    }

    /**
     * Triggers the activation of the last hidden layer ie: not logistic regression
     *
     * @return the activation of the last hidden layer given the last input to the network
     */
    public INDArray activate() {
        return getLayers()[getNeuralNets().length - 1].activate();
    }

    /**
     * Triggers the activation for a given layer
     *
     * @param layer the layer to activate on
     * @return the activation for a given layer
     */
    public INDArray activate(int layer) {
        return getLayers()[layer].activate();
    }

    /**
     * Triggers the activation of the given layer
     *
     * @param layer the layer to trigger on
     * @param input the input to the hidden layer
     * @return the activation of the layer based on the input
     */
    public INDArray activate(int layer, INDArray input) {
        return getLayers()[layer].activate(input);
    }


    /**
     * Finetunes with the current cached labels
     *
     * @param lr     the learning rate to use
     * @param epochs the max number of epochs to finetune with
     */
    public void finetune(float lr, int epochs) {
        finetune(this.labels, lr, epochs);

    }

    /**
     * Sets the input and labels from this dataset
     *
     * @param data the dataset to initialize with
     */
    public void initialize(DataSet data) {
        setInput(data.getFeatureMatrix());
        feedForward(data.getFeatureMatrix());
        this.labels = data.getLabels();

        getOutputLayer().setLabels(labels);
    }




    public INDArray activationFromPrevLayer(int curr,INDArray input) {
        //output layer
        if(curr == neuralNets.length) {
            return getOutputLayer().labelProbabilities(input);
        }

        switch(layers[curr].conf().getActivationType()) {
            case HIDDEN_LAYER_ACTIVATION: return layers[curr].activate(input);
            case NET_ACTIVATION: return neuralNets[curr].hiddenActivation(input);
            case SAMPLE: return neuralNets[curr].sampleHiddenGivenVisible(input).getSecond();
            default: throw new IllegalStateException("Invalid activation type");
        }
    }




    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward() {
        INDArray currInput = this.input;
        if (this.input.columns() != defaultConfiguration.getnIn())
            throw new IllegalStateException("Illegal input length");

        List<INDArray> activations = new ArrayList<>();
        activations.add(currInput);

        for (int i = 0; i < layers.length; i++) {
            currInput = activationFromPrevLayer(i,currInput);
            //applies drop connect to the activation
            applyDropConnectIfNecessary(currInput);
            activations.add(currInput);
        }


        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(INDArray input) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");

        else
            this.input = input;
        return feedForward();
    }


    /**
     * Applies drop connect relative to connections.
     * This should be used on the activation of a neural net. (Post sigmoid layer)
     *
     * @param input the input to apply drop connect to
     */
    protected void applyDropConnectIfNecessary(INDArray input) {
        if (useDropConnect) {
            INDArray mask = Sampling.binomial(Nd4j.valueArrayOf(input.rows(), input.columns(), 0.5), 1, defaultConfiguration.getRng());
            input.muli(mask);
            //apply l2 for drop connect
            if (defaultConfiguration.getL2() > 0)
                input.muli(defaultConfiguration.getL2());
        }
    }


    /* delta computation for back prop with the R operator */
    protected List<INDArray> computeDeltasR(INDArray v) {
        List<INDArray> deltaRet = new ArrayList<>();

        INDArray[] deltas = new INDArray[getnLayers() + 1];
        List<INDArray> activations = feedForward();
        List<INDArray> rActivations = feedForwardR(activations, v);
      /*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();
        List<ActivationFunction> activationFunctions = new ArrayList<>();


        for (int j = 0; j < getNeuralNets().length; j++) {
            weights.add(getNeuralNets()[j].getW());
            biases.add(getNeuralNets()[j].gethBias());
            activationFunctions.add(getNeuralNets()[j].conf().getActivationFunction());
        }


        INDArray rix = rActivations.get(rActivations.size() - 1).divi(input.rows());
        LinAlgExceptions.assertValidNum(rix);

        //errors
        for (int i = getnLayers(); i >= 0; i--) {
            //W^t * error^l + 1
            deltas[i] = activations.get(i).transpose().mmul(rix);
            applyDropConnectIfNecessary(deltas[i]);

            if (i > 0)
                rix = rix.mmul(weights.get(i).addRowVector(biases.get(i)).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));


        }

        for (int i = 0; i < deltas.length; i++) {
            if (defaultConfiguration.isConstrainGradientToUnitNorm())
                deltaRet.add(deltas[i].div(deltas[i].norm2(Integer.MAX_VALUE)));

            else
                deltaRet.add(deltas[i]);
            LinAlgExceptions.assertValidNum(deltaRet.get(i));
        }

        return deltaRet;
    }



    //damping update after line search
    public void dampingUpdate(float rho, float boost, float decrease) {
        if (rho < 0.25 || Double.isNaN(rho)) {
            this.dampingFactor *= boost;
        } else if (rho > 0.75)
            this.dampingFactor *= decrease;


    }

    /* p and gradient are same length */
    public float reductionRatio(INDArray p, float currScore, float score, INDArray gradient) {
        float currentDamp = dampingFactor;
        this.dampingFactor = 0;
        INDArray denom = getBackPropRGradient(p);
        denom.muli(0.5).muli(p.mul(denom)).sum(0);
        denom.subi(gradient.mul(p).sum(0));
        float rho = (currScore - score) / (float) denom.getScalar(0).element();
        this.dampingFactor = currentDamp;
        if (score - currScore > 0)
            return Float.NEGATIVE_INFINITY;
        return rho;
    }


    /* delta computation for back prop with precon for SFH */
    protected List<Pair<INDArray, INDArray>> computeDeltas2() {
        List<Pair<INDArray, INDArray>> deltaRet = new ArrayList<>();
        List<INDArray> activations = feedForward();
        INDArray[] deltas = new INDArray[activations.size() - 1];
        INDArray[] preCons = new INDArray[activations.size() - 1];


        //- y - h
        INDArray ix = activations.get(activations.size() - 1).sub(labels).div(labels.rows());
        log.info("Ix mean " + ix.sum(Integer.MAX_VALUE));

       	/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();

        List<ActivationFunction> activationFunctions = new ArrayList<>();
        for (int j = 0; j < getNeuralNets().length; j++) {
            weights.add(getNeuralNets()[j].getW());
            biases.add(getNeuralNets()[j].gethBias());
            activationFunctions.add(getLayers()[j].conf().getActivationFunction());
        }


        //errors
        for (int i = weights.size() - 1; i >= 0; i--) {
            deltas[i] = activations.get(i).transpose().mmul(ix);
            log.info("Delta sum at " + i + " is " + deltas[i].sum(Integer.MAX_VALUE));
            preCons[i] = Transforms.pow(activations.get(i).transpose(), 2).mmul(Transforms.pow(ix, 2)).mul(labels.rows());
            applyDropConnectIfNecessary(deltas[i]);

            if (i > 0) {
                //W[i] + b[i] * f'(z[i - 1])
                ix = ix.mmul(weights.get(i).transpose()).muli(activationFunctions.get(i - 1).applyDerivative(activations.get(i)));
            }
        }

        for (int i = 0; i < deltas.length; i++) {
            if (defaultConfiguration.isConstrainGradientToUnitNorm())
                deltaRet.add(new Pair<>(deltas[i].divi(deltas[i].norm2(Integer.MAX_VALUE)), preCons[i]));

            else
                deltaRet.add(new Pair<>(deltas[i], preCons[i]));

        }

        return deltaRet;
    }


    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        int start = 0;
        //loop to output layer
        for(int i = 0; i < neuralNets.length; i++) {
            int numParams = getNeuralNets()[i].numParams();
            //set the params for the neural net AND bias, note that these are the same reference, not duplications
            getNeuralNets()[i].setParams(params.get(NDArrayIndex.interval(start,start + numParams)));
            getLayers()[i].setW(getNeuralNets()[i].getW());
            getLayers()[i].setB(getNeuralNets()[i].gethBias());
            start += numParams;
        }

        getOutputLayer().setParams(params.get(NDArrayIndex.interval(start, params.length())));

    }

    /**
     * Fit the model to the given data
     *
     * @param data the data to fit the model to
     */
    @Override
    public void fit(INDArray data) {
        this.pretrain(data,new Object[]{1});
    }

    /* delta computation for back prop */
    protected void computeDeltas(List<INDArray> deltaRet) {
        INDArray[] deltas = new INDArray[getnLayers() + 2];

        List<INDArray> activations = feedForward();

        //- y - h
        INDArray ix = labels.sub(activations.get(activations.size() - 1)).negi().subi(getOutputLayer().conf().getActivationFunction().applyDerivative(activations.get(activations.size() - 1)));

		/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();

        List<ActivationFunction> activationFunctions = new ArrayList<>();
        for (int j = 0; j < getNeuralNets().length; j++) {
            weights.add(getNeuralNets()[j].getW());
            biases.add(getNeuralNets()[j].gethBias());
            activationFunctions.add(getLayers()[j].conf().getActivationFunction());
        }


        weights.add(getOutputLayer().getW());
        biases.add(getOutputLayer().getB());
        activationFunctions.add(getOutputLayer().conf().getActivationFunction());


        //errors
        for (int i = getnLayers() + 1; i >= 0; i--) {
            //output layer
            if (i >= getnLayers() + 1) {
                //-( y - h) .* f'(z^l) where l is the output layer
                deltas[i] = ix;

            } else {
                INDArray delta = activations.get(i).transpose().mmul(ix);
                deltas[i] = delta;
                applyDropConnectIfNecessary(deltas[i]);
                INDArray weightsPlusBias = weights.get(i).addRowVector(biases.get(i)).transpose();
                INDArray activation = activations.get(i);
                if (i > 0)
                    ix = ix.mmul(weightsPlusBias).muli(activationFunctions.get(i - 1).applyDerivative(activation));

            }

        }

        for (int i = 0; i < deltas.length; i++) {
            if (defaultConfiguration.isConstrainGradientToUnitNorm())
                deltaRet.add(deltas[i].divi(deltas[i].norm2(Integer.MAX_VALUE)));

            else
                deltaRet.add(deltas[i]);
        }


    }

    /**
     * One step of back prop
     */
    public void backPropStep() {
        List<Pair<INDArray, INDArray>> deltas = backPropGradient();
        for (int i = 0; i < layers.length; i++) {
            layers[i].getW().subi(deltas.get(i).getFirst());
            layers[i].getB().subi(deltas.get(i).getSecond());
            if(i < neuralNets.length) {
                neuralNets[i].setW(layers[i].getW());
                neuralNets[i].sethBias(layers[i].getB());
            }
        }

    }


    /**
     * One step of back prop with the R operator
     *
     * @param v for gaussian vector, this is for g * v. This is optional.
     */
    public void backPropStepR(INDArray v) {
        List<Pair<INDArray, INDArray>> deltas = backPropGradientR(v);
        for (int i = 0; i < neuralNets.length; i++) {
            if (deltas.size() < neuralNets.length) {
                neuralNets[i].getW().subi(deltas.get(i).getFirst());
                neuralNets[i].gethBias().subi(deltas.get(i).getSecond());
                layers[i].setW(neuralNets[i].getW());
                layers[i].setB(neuralNets[i].gethBias());
            }

        }

        getOutputLayer().getW().subi(deltas.get(deltas.size() - 1).getFirst());
        getOutputLayer().getB().subi(deltas.get(deltas.size() - 1).getSecond());

    }

    public Layer[] getLayers() {
        return layers;
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public void setNeuralNets(NeuralNetwork[] neuralNets) {
        this.neuralNets = neuralNets;
    }

    /**
     * Gets the back prop gradient with the r operator (gauss vector)
     * This is also called computeGV
     *
     * @param v the v in gaussian newton vector g * v
     * @return the back prop with r gradient
     */
    public INDArray getBackPropRGradient(INDArray v) {
        return pack(backPropGradientR(v));
    }


    /**
     * Gets the back prop gradient with the r operator (gauss vector)
     * and the associated precon matrix
     * This is also called computeGV
     *
     * @return the back prop with r gradient
     */
    public Pair<INDArray, INDArray> getBackPropGradient2() {
        List<Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>>> deltas = backPropGradient2();
        List<Pair<INDArray, INDArray>> deltaNormal = new ArrayList<>();
        List<Pair<INDArray, INDArray>> deltasPreCon = new ArrayList<>();
        for (int i = 0; i < deltas.size(); i++) {
            deltaNormal.add(deltas.get(i).getFirst());
            deltasPreCon.add(deltas.get(i).getSecond());
        }


        return new Pair<>(pack(deltaNormal), pack(deltasPreCon));
    }


    @Override
    public BaseMultiLayerNetwork clone() {
        BaseMultiLayerNetwork ret = null;
        try {
            ret = getClass().newInstance();
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        }
        ret.update(this);
        return ret;
    }


    public List<INDArray> weightMatrices() {
        List<INDArray> ret = new ArrayList<>();
        for (int i = 0; i < neuralNets.length; i++)
            ret.add(neuralNets[i].getW());
        ret.add(getOutputLayer().getW());
        return ret;
    }


    /**
     * Feed forward with the r operator
     *
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    public List<INDArray> feedForwardR(List<INDArray> acts, INDArray v) {
        List<INDArray> R = new ArrayList<>();
        R.add(Nd4j.zeros(input.rows(), input.columns()));
        List<Pair<INDArray, INDArray>> vWvB = unPack(v);
        List<INDArray> W = weightMatrices();

        for (int i = 0; i < neuralNets.length; i++) {
            ActivationFunction derivative = getNeuralNets()[i].conf().getActivationFunction();
            if (getNeuralNets()[i] instanceof AutoEncoder) {
                AutoEncoder a = (AutoEncoder) getNeuralNets()[i];
                derivative = a.conf.getActivationFunction();
            }

            //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
            R.add(R.get(i).mmul(W.get(i)).addi(acts.get(i).mmul(vWvB.get(i).getFirst().addRowVector(vWvB.get(i).getSecond()))).muli((derivative.applyDerivative(acts.get(i + 1)))));
        }

        //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
        R.add(R.get(R.size() - 1).mmul(W.get(W.size() - 1)).addi(acts.get(acts.size() - 2).mmul(vWvB.get(vWvB.size() - 1).getFirst().addRowVector(vWvB.get(vWvB.size() - 1).getSecond()))).muli((getOutputLayer().conf().getActivationFunction().applyDerivative(acts.get(acts.size() - 1)))));

        return R;
    }


    /**
     * Feed forward with the r operator
     *
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    public List<INDArray> feedForwardR(INDArray v) {
        return feedForwardR(feedForward(), v);
    }


    /**
     * Backpropagation of errors for weights
     *
     * @param lr     the learning rate to use
     * @param epochs the number of epochs to iterate (this is already called in finetune)
     * @param eval   the evaluator for stopping
     */
    public void backProp(float lr, int epochs, TrainingEvaluator eval) {
        if (useGaussNewtonVectorProductBackProp) {
            BackPropROptimizer opt = new BackPropROptimizer(this, lr, epochs);
            opt.optimize(eval, epochs, lineSearchBackProp);

        } else {
            BackPropOptimizer opt = new BackPropOptimizer(this, lr, epochs);
            opt.optimize(eval, epochs, lineSearchBackProp);

        }
    }

    /**
     * Backpropagation of errors for weights
     *
     * @param lr     the learning rate to use
     * @param epochs the number of epochs to iterate (this is already called in finetune)
     */
    public void backProp(float lr, int epochs) {
        backProp(lr, epochs, null);

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
     * various neuralNets(w,hbias NOT VBIAS) and output layer
     *
     * @return the params for this neural net
     */
    @Override
    public INDArray params() {
        List<INDArray> params = new ArrayList<>();
        for(int i = 0; i < getnLayers(); i++) {
            params.add(getNeuralNets()[i].getW());
            params.add(getNeuralNets()[i].gethBias());
        }
        params.add(getOutputLayer().params());
        return Nd4j.toFlattened(params);
    }


    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various neuralNets and output layer
     *
     * @return the params for this neural net
     */
    @Override
    public int numParams() {
        int length = 0;


        for (int i = 0; i < neuralNets.length; i++) {
            length += neuralNets[i].numParams() - neuralNets[i].getvBias().length();
        }

        length += getOutputLayer().numParams();

        return length;

    }

    /**
     * Packs a set of matrices in to one vector,
     * where the matrices in this case are the w,hbias at each layer
     * and the output layer w,bias
     *
     * @return a singular matrix of all of the neuralNets packed in to one matrix
     */

    public INDArray pack() {
        List<Pair<INDArray, INDArray>> vWvB = new ArrayList<>();
        for (int i = 0; i < neuralNets.length; i++) {
            vWvB.add(new Pair<>(neuralNets[i].getW(), neuralNets[i].gethBias()));
        }

        vWvB.add(new Pair<>(getOutputLayer().getW(), getOutputLayer().getB()));

        return pack(vWvB);

    }

    /**
     * Packs a set of matrices in to one vector
     *
     * @param layers the neuralNets to pack
     * @return a singular matrix of all of the neuralNets packed in to one matrix
     */
    public INDArray pack(List<Pair<INDArray, INDArray>> layers) {
        if (layers.size() != this.neuralNets.length + 1)
            throw new IllegalArgumentException("Illegal number of neuralNets passed in. Was " + layers.size() + " when should have been " + (this.neuralNets.length + 1));

        List<INDArray> list = new ArrayList<>();
        for(int i = 0; i < layers.size(); i++) {
            list.add(layers.get(i).getFirst());
            list.add(layers.get(i).getSecond());
        }



        INDArray ret = Nd4j.toFlattened(list);
        if(ret.length() != numParams())
            throw new IllegalStateException("Illegal number of parameters found in the layers with a difference of " + Math.abs(ret.length() - numParams()));
        return ret;
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public float score(org.nd4j.linalg.dataset.api.DataSet data) {
        return score(data.getFeatureMatrix(), data.getLabels());
    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last neuralNets weights
     * to revert to in case of convergence, the learning rate being used to iterate
     * and the current epoch
     *
     * @return whether the training should converge or not
     */
    protected List<Pair<INDArray, INDArray>> backPropGradient() {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);

        //precompute deltas
        List<INDArray> deltas = new ArrayList<>();
        //compute derivatives and gradients given activations
        computeDeltas(deltas);

        List<Pair<INDArray, INDArray>> vWvB = new ArrayList<>();
        for (int i = 0; i < neuralNets.length; i++)
            vWvB.add(new Pair<>(neuralNets[i].getW(), neuralNets[i].gethBias()));


        vWvB.add(new Pair<>(getOutputLayer().getW(), getOutputLayer().getB()));


        List<Pair<INDArray, INDArray>> list = new ArrayList<>();

        for (int l = 0; l < getnLayers() + 1; l++) {
            INDArray gradientChange = deltas.get(l);
            if (gradientChange.length() != getLayers()[l].getW().length())
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).sum(0);


            list.add(new Pair<>(gradientChange, deltaColumnSums));


        }


        if (mask == null)
            initMask();



        for(int i = 0; i < list.size(); i++) {
            if(i < getnLayers()) {
                INDArray weightGradientForLayer = list.get(i).getFirst();
                INDArray biasGradientForLayer = list.get(i).getSecond();
                assert Arrays.equals(weightGradientForLayer.shape(),getNeuralNets()[i].getW().shape()) :
                        "Illegal shape for layer " + i + " weight gradient, should have been " +
                                Arrays.toString(getNeuralNets()[i].getW().shape()) + " but was " + Arrays.toString(weightGradientForLayer.shape()) ;
                assert Arrays.equals(biasGradientForLayer.shape(),getNeuralNets()[i].gethBias().shape()) :
                        "Illegal shape for layer " + i + " bias   gradient, should have been " +
                                Arrays.toString(getNeuralNets()[i].gethBias().shape()) + " but was " + Arrays.toString(biasGradientForLayer.shape());
            }
            else {
                INDArray weightGradientForLayer = list.get(i).getFirst();
                INDArray biasGradientForLayer = list.get(i).getSecond();
                assert Arrays.equals(weightGradientForLayer.shape(),getOutputLayer().getW().shape()) :
                        "Illegal shape for output  weight gradient, should have been " +
                                Arrays.toString(getOutputLayer().getW().shape()) + " but was " + Arrays.toString(weightGradientForLayer.shape());
                ;
                assert Arrays.equals(biasGradientForLayer.shape(),getOutputLayer().getB().shape()) :
                        "Illegal shape for output layer  bias   gradient, should have been " +
                                Arrays.toString(getOutputLayer().getB().shape()) + " but was " + Arrays.toString(biasGradientForLayer.shape());
                ;

            }
        }


        INDArray gradient = pack(list);
        INDArray params = params().mul(defaultConfiguration.getL2());
        gradient.addi(mask.mul(params));
        list = unPack(gradient);

        return list;

    }


    /**
     * Unpacks a parameter matrix in to a
     * applyTransformToDestination of pairs(w,hbias)
     * triples with layer wise
     * @param param the param vector
     * @return a segmented list of the param vector
     */
    public  List<Pair<INDArray,INDArray>> unPack(INDArray param) {
        //more sanity checks!
        int numParams = numParams();
        if(param.length() != numParams)
            throw new IllegalArgumentException("Parameter vector not equal of length to " + numParams);
        if(param.rows() != 1)
            param = param.reshape(1,param.length());
        List<Pair<INDArray,INDArray>> ret = new ArrayList<>();
        int curr = 0;
        for(int i = 0; i < layers.length; i++) {
            int layerLength = layers[i].getW().length() + layers[i].getB().length();
            INDArray subMatrix = param.get(NDArrayIndex.interval(curr,curr + layerLength));
            INDArray weightPortion = subMatrix.get(NDArrayIndex.interval(0,layers[i].getW().length()));

            int beginHBias = layers[i].getW().length();
            int endHbias = subMatrix.length();
            INDArray hBiasPortion = subMatrix.get(NDArrayIndex.interval(beginHBias,endHbias));
            int layerLengthSum = weightPortion.length() + hBiasPortion.length();
            if(layerLengthSum != layerLength) {
                if(hBiasPortion.length() != layers[i].getB().length())
                    throw new IllegalStateException("Hidden bias on layer " + i + " was off");
                if(weightPortion.length() != layers[i].getW().length())
                    throw new IllegalStateException("Weight portion on layer " + i + " was off");

            }
            ret.add(new Pair<>(weightPortion.reshape(layers[i].getW().rows(),layers[i].getW().columns()),hBiasPortion.reshape(layers[i].getB().rows(),layers[i].getB().columns())));
            curr += layerLength;
        }


        return ret;
    }

    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last neuralNets weights
     * to revert to in case of convergence, the learning rate being used to iterate
     * and the current epoch
     *
     * @return whether the training should converge or not
     */
    protected List<Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>>> backPropGradient2() {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);

        //precompute deltas
        List<Pair<INDArray, INDArray>> deltas = computeDeltas2();


        List<Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>>> list = new ArrayList<>();
        List<Pair<INDArray, INDArray>> grad = new ArrayList<>();
        List<Pair<INDArray, INDArray>> preCon = new ArrayList<>();

        for (int l = 0; l < deltas.size(); l++) {
            INDArray gradientChange = deltas.get(l).getFirst();
            INDArray preConGradientChange = deltas.get(l).getSecond();


            if (l < getNeuralNets().length && gradientChange.length() != getNeuralNets()[l].getW().length())
                throw new IllegalStateException("Gradient change not equal to weight change");
            else if (l == getNeuralNets().length && gradientChange.length() != getOutputLayer().getW().length())
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).getFirst().mean(1);
            INDArray preConColumnSums = deltas.get(l).getSecond().mean(1);

            grad.add(new Pair<>(gradientChange, deltaColumnSums));
            preCon.add(new Pair<>(preConGradientChange, preConColumnSums));
            if (l < getNeuralNets().length && deltaColumnSums.length() != neuralNets[l].gethBias().length())
                throw new IllegalStateException("Bias change not equal to weight change");
            else if (l == getNeuralNets().length && deltaColumnSums.length() != getOutputLayer().getB().length())
                throw new IllegalStateException("Bias change not equal to weight change");


        }

        INDArray g = pack(grad);
        INDArray con = pack(preCon);
        INDArray theta = params();


        if (mask == null)
            initMask();

        g.addi(theta.mul(defaultConfiguration.getL2()).muli(mask));

        INDArray conAdd = Transforms.pow(mask.mul(defaultConfiguration.getL2()).add(Nd4j.valueArrayOf(g.rows(), g.columns(), dampingFactor)), 3.0 / 4.0);

        con.addi(conAdd);

        List<Pair<INDArray, INDArray>> gUnpacked = unPack(g);

        List<Pair<INDArray, INDArray>> conUnpacked = unPack(con);

        for (int i = 0; i < gUnpacked.size(); i++)
            list.add(new Pair<>(gUnpacked.get(i), conUnpacked.get(i)));


        return list;

    }


    /**
     * Do a back prop iteration.
     * This involves computing the activations, tracking the last neuralNets weights
     * to revert to in case of convergence, the learning rate being used to iterate
     * and the current epoch
     *
     * @param v the v in gaussian newton vector g * v
     * @return whether the training should converge or not
     */
    protected List<Pair<INDArray, INDArray>> backPropGradientR(INDArray v) {
        //feedforward to compute activations
        //initial error
        //log.info("Back prop step " + epoch);
        if (mask == null)
            initMask();
        //precompute deltas
        List<INDArray> deltas = computeDeltasR(v);
        //compute derivatives and gradients given activations


        List<Pair<INDArray, INDArray>> list = new ArrayList<>();

        for (int l = 0; l < getnLayers(); l++) {
            INDArray gradientChange = deltas.get(l);

            if (gradientChange.length() != getNeuralNets()[l].getW().length())
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).mean(1);
            if (deltaColumnSums.length() != neuralNets[l].gethBias().length())
                throw new IllegalStateException("Bias change not equal to weight change");


            list.add(new Pair<>(gradientChange, deltaColumnSums));


        }

        INDArray logLayerGradient = deltas.get(getnLayers());
        INDArray biasGradient = deltas.get(getnLayers()).mean(1);

        list.add(new Pair<>(logLayerGradient, biasGradient));

        INDArray pack = pack(list).addi(mask.mul(defaultConfiguration.getL2()).mul(v)).addi(v.mul(dampingFactor));
        return unPack(pack);

    }


    /**
     * Run SGD based on the given labels
     *
     * @param iter       fine tune based on the labels
     * @param lr         the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DataSetIterator iter, float lr, int iterations) {
        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();
            if (data.getFeatureMatrix() == null || data.getLabels() == null)
                break;

            setInput(data.getFeatureMatrix());
            setLabels(data.getLabels());
            feedForward();
            optimizer = new MultiLayerNetworkOptimizer(this, lr);
            optimizer.optimize(data.getLabels(), lr, iterations);
        }


    }

    /**
     * Run training algorithm based on the datastet iterator
     *
     * @param iter       the labels to use
     * @param lr         the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(DataSetIterator iter, float lr, int iterations, TrainingEvaluator eval) {
        iter.reset();

        while (iter.hasNext()) {
            DataSet data = iter.next();
            if (data.getFeatureMatrix() == null || data.getLabels() == null)
                break;


            setInput(data.getFeatureMatrix());
            setLabels(data.getLabels());


            optimizer = new MultiLayerNetworkOptimizer(this, lr);
            optimizer.optimize(this.labels, lr, iterations, eval);
        }


    }


    /**
     * Run SGD based on the given labels
     *
     * @param labels     the labels to use
     * @param lr         the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(INDArray labels, float lr, int iterations) {
        this.labels = labels;
        getOutputLayer().setLabels(labels);
        feedForward();
        if (labels != null)
            this.labels = labels;
        optimizer = new MultiLayerNetworkOptimizer(this, lr);
        optimizer.optimize(this.labels, lr, iterations);

    }

    /**
     * Run training algorithm based on the given labels
     *
     * @param labels     the labels to use
     * @param lr         the learning rate during training
     * @param iterations the number of times to iterate
     */
    public void finetune(INDArray labels, float lr, int iterations, TrainingEvaluator eval) {
        feedForward();

        if (labels != null)
            this.labels = labels;

        optimizer = new MultiLayerNetworkOptimizer(this, lr);
        optimizer.optimize(this.labels, lr, iterations, eval);
    }


    /**
     * Returns the predictions for each example in the dataset
     *
     * @param d the matrix to predict
     * @return the prediction for the dataset
     */
    @Override
    public int[] predict(INDArray d) {
        INDArray output = output(d);
        int[] ret = new int[d.rows()];
        for (int i = 0; i < ret.length; i++)
            ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        return ret;
    }

    /**
     * Returns the probabilities for each label
     * for each example row wise
     *
     * @param examples the examples to classify (one example in each row)
     * @return the likelihoods of each example and each label
     */
    @Override
    public INDArray labelProbabilities(INDArray examples) {
        List<INDArray> feed = feedForward(examples);
        return getOutputLayer().labelProbabilities(feed.get(feed.size() - 1));
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray examples, INDArray labels) {
        pretrain(examples,new Object[]{defaultConfiguration.getK(), defaultConfiguration.getLr()});
        finetune(labels,defaultConfiguration.getLr(),defaultConfiguration.getNumIterations());
    }

    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(org.nd4j.linalg.dataset.api.DataSet data) {
        fit(data.getFeatureMatrix(),data.getLabels());
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        getOutputLayer().fit(examples, labels);
    }

    /**
     * Label the probabilities of the input
     *
     * @param x the input to label
     * @return a vector of probabilities
     * given each label.
     * <p/>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray x) {
        List<INDArray> activations = feedForward(x);


        //last activation is input
        INDArray predicted = activations.get(activations.size() - 1);
        return predicted;
    }


    /**
     * Reconstructs the input.
     * This is equivalent functionality to a
     * deep autoencoder.
     *
     * @param x        the input to transform
     * @param layerNum the layer to output for encoding
     * @return a reconstructed matrix
     * relative to the size of the last hidden layer.
     * This is great for data compression and visualizing
     * high dimensional data (or just doing dimensionality reduction).
     * <p/>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray reconstruct(INDArray x, int layerNum) {
        INDArray currInput = x;
        List<INDArray> forward = feedForward(currInput);
        return forward.get(layerNum - 1);
    }


    /**
     * Prints the configuration
     */
    public void printConfiguration() {
        StringBuffer sb = new StringBuffer();
        int count = 0;
        for(NeuralNetConfiguration conf : getLayerWiseConfigurations()) {
            sb.append(" Layer " + count++ + " conf " + conf);
        }

        log.info(sb.toString());
    }

    @Override
    public void write(OutputStream os) {
        SerializationUtils.writeObject(this, os);
    }

    /**
     * Load (using {@link ObjectInputStream}
     *
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
     *
     * @param network the network to getFromOrigin parameters from
     */
    public void update(BaseMultiLayerNetwork network) {

        if (network.neuralNets != null && network.getnLayers() > 0) {
            this.setnLayers(network.getNeuralNets().length);
            this.neuralNets = new NeuralNetwork[network.getNeuralNets().length];
            for (int i = 0; i < neuralNets.length; i++)
                if ((this.getnLayers()) > i && (network.getnLayers() > i)) {
                    if (network.getNeuralNets()[i] == null) {
                        throw new IllegalStateException("Will not clone uninitialized network, layer " + i + " of network was null");
                    }

                    this.getNeuralNets()[i] = network.getNeuralNets()[i].clone();

                }
        }

        this.hiddenLayerSizes = network.hiddenLayerSizes;
        this.defaultConfiguration = network.defaultConfiguration;
        this.errorTolerance = network.errorTolerance;
        this.forceNumEpochs = network.forceNumEpochs;
        this.input = network.input;
        this.labels = network.labels;
        this.learningRateUpdate = network.learningRateUpdate;
        this.shouldBackProp = network.shouldBackProp;
        this.weightTransforms = network.weightTransforms;
        this.visibleBiasTransforms = network.visibleBiasTransforms;
        this.hiddenBiasTransforms = network.hiddenBiasTransforms;
        this.useDropConnect = network.useDropConnect;
        this.useGaussNewtonVectorProductBackProp = network.useGaussNewtonVectorProductBackProp;

        if (network.neuralNets != null && network.neuralNets.length > 0) {
            this.neuralNets = new NeuralNetwork[network.neuralNets.length];
            for (int i = 0; i < neuralNets.length; i++)
                this.getNeuralNets()[i] = network.getNeuralNets()[i].clone();

        }


    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param input  the input to score
     * @param labels the true labels
     * @return the score for the given input,label pairs
     */
    @Override
    public float score(INDArray input, INDArray labels) {
        feedForward(input);
        setLabels(labels);
        Evaluation eval = new Evaluation();
        eval.eval(labels, labelProbabilities(input));
        return (float) eval.f1();
    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        return labels.columns();
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    public float score(DataSet data) {
        feedForward(data.getFeatureMatrix());
        setLabels(data.getLabels());
        return score();
    }


    /**
     * Score of the model (relative to the objective function)
     *
     * @return the score of the model (relative to the objective function)
     */
    @Override
    public float score() {
        feedForward();
        return getOutputLayer().score();
    }

    /**
     * Score of the model (relative to the objective function)
     *
     * @param param the current parameters
     * @return the score of the model (relative to the objective function)
     */

    public float score(INDArray param) {
        INDArray params = params();
        setParameters(param);
        float ret = score();
        float regCost = 0.5f * defaultConfiguration.getL2() * (float) Transforms.pow(mask.mul(param), 2).sum(Integer.MAX_VALUE).element();
        setParameters(params);
        return ret + regCost;
    }


    protected void applyTransforms() {
        if (neuralNets == null || neuralNets.length < 1) {
            throw new IllegalStateException("Layers not initialized");
        }

        for (int i = 0; i < neuralNets.length; i++) {
            if (weightTransforms.containsKey(i))
                neuralNets[i].setW(weightTransforms.get(i).apply(neuralNets[i].getW()));
            if (hiddenBiasTransforms.containsKey(i))
                neuralNets[i].sethBias(getHiddenBiasTransforms().get(i).apply(neuralNets[i].gethBias()));
            if (this.visibleBiasTransforms.containsKey(i))
                neuralNets[i].setvBias(getVisibleBiasTransforms().get(i).apply(neuralNets[i].getvBias()));
        }
    }


    /**
     * Creates a layer depending on the index.
     * The main reason this matters is for continuous variations such as the {@link org.deeplearning4j.models.classifiers.dbn.DBN}
     * where the first layer needs to be an {@link org.deeplearning4j.models.featuredetectors.rbm.RBM} for continuous inputs.
     *
     * @param input    the input to the layer
     * @param W        the weight vector
     * @param hbias    the hidden bias
     * @param vBias    the visible bias
     * @param index    the index of the layer
     * @return a neural network layer such as {@link org.deeplearning4j.models.featuredetectors.rbm.RBM}
     */
    public abstract NeuralNetwork createLayer(INDArray input, INDArray W, INDArray hbias, INDArray vBias, int index);


    public abstract void pretrain(DataSetIterator iter, Object[] otherParams);

    public abstract void pretrain(INDArray input, Object[] otherParams);


    public abstract NeuralNetwork[] createNetworkLayers(int numLayers);


    /**
     * Creates a hidden layer with the given parameters.
     * The default implementation is a binomial sampling
     * hidden layer, but this can be overridden
     * for other kinds of hidden units
     * @param layerInput the layer starting input
     *                   for generating weights
     * @return a hidden layer with the given parameters
     */
    public Layer createHiddenLayer(int index, INDArray layerInput) {

        return new org.deeplearning4j.nn.layers.Layer.Builder()
                .withInput(layerInput).conf(layerWiseConfigurations.get(index))
                .build();

    }


    /**
     * Merges this network with the other one.
     * This is a weight averaging with the update of:
     * a += b - a / n
     * where a is a matrix on the network
     * b is the incoming matrix and n
     * is the batch size.
     * This update is performed across the network neuralNets
     * as well as hidden neuralNets and logistic neuralNets
     *
     * @param network   the network to merge with
     * @param batchSize the batch size (number of training examples)
     *                  to average by
     */
    public void merge(BaseMultiLayerNetwork network, int batchSize) {
        if (network.getnLayers() != getnLayers())
            throw new IllegalArgumentException("Unable to merge networks that are not of equal length");
        for (int i = 0; i < getnLayers(); i++) {
            NeuralNetwork n = neuralNets[i];
            NeuralNetwork otherNetwork = network.neuralNets[i];
            n.merge(otherNetwork, batchSize);
            //tied weights: must be updated at the same time
            getLayers()[i].setB(n.gethBias());
            getLayers()[i].setW(n.getW());

        }

        getOutputLayer().merge(network.getOutputLayer(), batchSize);
    }


    public INDArray getLabels() {
        return labels;
    }


    /**
     * Note that if input isn't null
     * and the neuralNets are null, this is a way
     * of initializing the neural network
     *
     * @param input
     */
    public void setInput(INDArray input) {
        if (input != null && this.neuralNets == null)
            this.initializeLayers(input);
        this.input = input;

    }

    private void initMask() {
        setMask(Nd4j.ones(1, pack().length()));
        List<Pair<INDArray, INDArray>> mask = unPack(getMask());
        for (int i = 0; i < mask.size(); i++)
            mask.get(i).setSecond(Nd4j.zeros(mask.get(i).getSecond().rows(), mask.get(i).getSecond().columns()));
        setMask(pack(mask));

    }

    public boolean isShouldBackProp() {
        return shouldBackProp;
    }


    public INDArray getInput() {
        return input;
    }

    public synchronized NeuralNetwork[] getNeuralNets() {
        return neuralNets;
    }


    public boolean isForceNumEpochs() {
        return forceNumEpochs;
    }


    public int[] getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    public void setHiddenLayerSizes(int[] hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }


    public Map<Integer, MatrixTransform> getWeightTransforms() {
        return weightTransforms;
    }


    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    public void setForceNumEpochs(boolean forceNumEpochs) {
        this.forceNumEpochs = forceNumEpochs;
    }


    public boolean isSampleFromHiddenActivations() {
        return sampleFromHiddenActivations;
    }

    public void setSampleFromHiddenActivations(
            boolean sampleFromHiddenActivations) {
        this.sampleFromHiddenActivations = sampleFromHiddenActivations;
    }


    public Map<Integer, MatrixTransform> getHiddenBiasTransforms() {
        return hiddenBiasTransforms;
    }

    public Map<Integer, MatrixTransform> getVisibleBiasTransforms() {
        return visibleBiasTransforms;
    }

    public int getnLayers() {
        return neuralNets.length;
    }

    public void setnLayers(int nLayers) {
        this.neuralNets = createNetworkLayers(nLayers);
    }


    public void setShouldBackProp(boolean shouldBackProp) {
        this.shouldBackProp = shouldBackProp;
    }

    public void setLayers(NeuralNetwork[] layers) {
        this.neuralNets = layers;
    }

    public boolean isUseGaussNewtonVectorProductBackProp() {
        return useGaussNewtonVectorProductBackProp;
    }

    public void setUseGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
        this.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
    }


    public INDArray getMask() {
        return mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }


    /**
     * Clears the input from all of the neuralNets
     */
    public void clearInput() {
        this.input = null;
        for (int i = 0; i < neuralNets.length; i++) {
            neuralNets[i].clearInput();
            layers[i].setInput(null);
        }

    }


    public OutputLayer getOutputLayer() {
        return (OutputLayer) getLayers()[getLayers().length - 1];
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
     * all neuralNets (including the output layer)
     *
     * @param params a parameter vector equal 1,numParameters
     */
    public void setParameters(INDArray params) {
        for (int i = 0; i < getNeuralNets().length; i++) {
            ParamRange range = startIndexForLayer(i);
            INDArray w = params.get(NDArrayIndex.interval(0, params.rows()), NDArrayIndex.interval(range.getwStart(), range.getwEnd()));
            INDArray bias = params.get(NDArrayIndex.interval(0, params.rows()), NDArrayIndex.interval(range.getBiasStart(), range.getBiasEnd()));
            int rows = getNeuralNets()[i].getW().rows(), columns = getNeuralNets()[i].getW().columns();
            getNeuralNets()[i].setW(w.reshape(rows, columns));
            getNeuralNets()[i].sethBias(bias.reshape(getNeuralNets()[i].gethBias().rows(), getNeuralNets()[i].gethBias().columns()));
        }


        ParamRange range = startIndexForLayer(getNeuralNets().length);
        INDArray w = params.get(NDArrayIndex.interval(0, params.rows()), NDArrayIndex.interval(range.getwStart(), range.getwEnd()));
        INDArray bias = params.get(NDArrayIndex.interval(0, params.rows()), NDArrayIndex.interval(range.getBiasStart(), range.getBiasEnd()));
        int rows = getOutputLayer().getW().rows(), columns = getOutputLayer().getW().columns();
        getOutputLayer().setW(w.reshape(rows, columns));
        getOutputLayer().setB(bias.reshape(getOutputLayer().getB().rows(), getOutputLayer().getB().columns()));


    }


    /**
     * Returns a start index for a given layer (neural net or outputlayer)
     *
     * @param layer the layer to getFromOrigin the index for
     * @return the index for the layer
     */
    public ParamRange startIndexForLayer(int layer) {
        int start = 0;
        for (int i = 0; i < layer; i++) {
            start += getNeuralNets()[i].getW().length();
            start += getNeuralNets()[i].gethBias().length();
        }
        if (layer < getNeuralNets().length) {
            int wEnd = start + getNeuralNets()[layer].getW().length();
            return new ParamRange(start, wEnd, wEnd, wEnd + getNeuralNets()[layer].gethBias().length());

        } else {
            int wEnd = start + getOutputLayer().getW().length();
            return new ParamRange(start, wEnd, wEnd, wEnd + getOutputLayer().getB().length());

        }


    }


    /**
     * Run one iteration
     *
     * @param input  the input to iterate on
     * @param params the extra params for the neural network(k, corruption level, max epochs,...)
     */
    @Override
    public void iterate(INDArray input, Object[] params) {

    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the example labels(a binary outcome matrix)
     * @param params   extra parameters
     */
    @Override
    public void fit(INDArray examples, INDArray labels, Object[] params) {

    }

    /**
     * Fit the model
     *
     * @param data   the data to train on
     * @param params extra parameters
     */
    @Override
    public void fit(org.nd4j.linalg.dataset.api.DataSet data, Object[] params) {

    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     *                 the number of rows in the example
     * @param params   extra parameters
     */
    @Override
    public void fit(INDArray examples, int[] labels, Object[] params) {

    }

    /**
     * Iterate once on the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     *                 the number of rows in the example
     * @param params   extra parameters
     */
    @Override
    public void iterate(INDArray examples, int[] labels, Object[] params) {

    }



    public static class ParamRange implements Serializable {
        private int wStart, wEnd, biasStart, biasEnd;

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
        private int[] hiddenLayerSizes;
        private int nLayers;
        private INDArray input, labels;
        protected Map<Integer, MatrixTransform> weightTransforms = new HashMap<>();
        protected boolean backProp = true;
        protected boolean shouldForceEpochs = false;
        private Map<Integer, MatrixTransform> hiddenBiasTransforms = new HashMap<>();
        private Map<Integer, MatrixTransform> visibleBiasTransforms = new HashMap<>();
        private boolean lineSearchBackProp = false;
        private boolean useDropConnect = false;
        private boolean useGaussNewtonVectorProductBackProp = false;
        protected NeuralNetConfiguration conf;
        protected List<NeuralNetConfiguration> layerWiseConfiguration;



        public Builder<E> layerWiseCOnfiguration(List<NeuralNetConfiguration> layerWiseConfiguration) {
            this.layerWiseConfiguration = layerWiseConfiguration;
            return this;
        }

        public Builder<E> configure(NeuralNetConfiguration conf) {
            this.conf = conf;
            return this;
        }

        /**
         * Use gauss newton back prop - this is for hessian free
         *
         * @param useGaussNewtonVectorProductBackProp whether to use gauss newton vector backprop
         * @return
         */
        public Builder<E> useGaussNewtonVectorProductBackProp(boolean useGaussNewtonVectorProductBackProp) {
            this.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
            return this;
        }


        /**
         * Use drop connect on activations or not
         *
         * @param useDropConnect use drop connect or not
         * @return builder pattern
         */
        public Builder<E> useDropConnection(boolean useDropConnect) {
            this.useDropConnect = useDropConnect;
            return this;
        }


        /**
         * Whether to use line search back prop instead of SGD
         *
         * @param lineSearchBackProp
         * @return
         */
        public Builder<E> lineSearchBackProp(boolean lineSearchBackProp) {
            this.lineSearchBackProp = lineSearchBackProp;
            return this;
        }

        public Builder<E> withVisibleBiasTransforms(Map<Integer, MatrixTransform> visibleBiasTransforms) {
            this.visibleBiasTransforms = visibleBiasTransforms;
            return this;
        }

        public Builder<E> withHiddenBiasTransforms(Map<Integer, MatrixTransform> hiddenBiasTransforms) {
            this.hiddenBiasTransforms = hiddenBiasTransforms;
            return this;
        }

        /**
         * Forces use of number of epochs for training
         * SGD style rather than conjugate gradient
         *
         * @return
         */
        public Builder<E> forceEpochs() {
            shouldForceEpochs = true;
            return this;
        }

        /**
         * Disables back propagation
         *
         * @return
         */
        public Builder<E> disableBackProp() {
            backProp = false;
            return this;
        }

        /**
         * Transform the weights at the given layer
         *
         * @param layer     the layer to applyTransformToOrigin
         * @param transform the function used for transformation
         * @return
         */
        public Builder<E> transformWeightsAt(int layer, MatrixTransform transform) {
            weightTransforms.put(layer, transform);
            return this;
        }

        /**
         * A map of transformations for transforming
         * the given neuralNets
         *
         * @param transforms
         * @return
         */
        public Builder<E> transformWeightsAt(Map<Integer, MatrixTransform> transforms) {
            weightTransforms.putAll(transforms);
            return this;
        }


        /**
         * Size of the hidden neuralNets
         *
         * @param hiddenLayerSizes
         * @return
         */
        public Builder<E> hiddenLayerSizes(Integer[] hiddenLayerSizes) {
            this.hiddenLayerSizes = new int[hiddenLayerSizes.length];
            this.nLayers = hiddenLayerSizes.length;
            for (int i = 0; i < hiddenLayerSizes.length; i++)
                this.hiddenLayerSizes[i] = hiddenLayerSizes[i];
            return this;
        }

        public Builder<E> hiddenLayerSizes(int[] hiddenLayerSizes) {
            this.hiddenLayerSizes = hiddenLayerSizes;
            this.nLayers = hiddenLayerSizes.length;
            return this;
        }



        public Builder<E> withInput(INDArray input) {
            this.input = input;
            return this;
        }

        public Builder<E> withLabels(INDArray labels) {
            this.labels = labels;
            return this;
        }

        public Builder<E> withClazz(Class<? extends BaseMultiLayerNetwork> clazz) {
            this.clazz = clazz;
            return this;
        }


        @SuppressWarnings("unchecked")
        public E buildEmpty() {
            try {
                E ret;
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
                ret.setDefaultConfiguration(conf);
                ret.useGaussNewtonVectorProductBackProp = useGaussNewtonVectorProductBackProp;
                ret.setUseDropConnect(useDropConnect);
                ret.setInput(this.input);
                ret.setLabels(this.labels);
                ret.setHiddenLayerSizes(this.hiddenLayerSizes);
                ret.setnLayers(this.nLayers);
                ret.setShouldBackProp(this.backProp);
                ret.setLayerWiseConfigurations(layerWiseConfiguration);
                ret.neuralNets = new NeuralNetwork[nLayers];
                ret.setInput(this.input);
                ret.setLineSearchBackProp(lineSearchBackProp);
                ret.setLabels(labels);
                ret.setForceNumEpochs(shouldForceEpochs);
                ret.getWeightTransforms().putAll(weightTransforms);
                ret.getVisibleBiasTransforms().putAll(visibleBiasTransforms);
                ret.getHiddenBiasTransforms().putAll(hiddenBiasTransforms);
                if (hiddenLayerSizes == null)
                    throw new IllegalStateException("Unable to build network, no hidden layer sizes defined");
                return ret;
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }
    }
}



