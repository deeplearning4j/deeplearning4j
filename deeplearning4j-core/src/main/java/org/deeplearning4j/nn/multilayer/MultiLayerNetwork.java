/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.nn.multilayer;


import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.OutputLayer;
import org.deeplearning4j.nn.layers.convolution.subsampling.SubsamplingLayer;
import org.deeplearning4j.nn.layers.factory.LayerFactories;
import org.deeplearning4j.nn.layers.recurrent.BaseRecurrentLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.MultiLayerUtil;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.util.LinAlgExceptions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.lang.String;
import java.lang.reflect.Constructor;
import java.util.*;


/**
 * A base class for a multi
 * layer neural network with a logistic output layer
 * and multiple hidden neuralNets.
 *
 * @author Adam Gibson
 */
public class MultiLayerNetwork implements Serializable, Classifier, Layer {


    private static final Logger log = LoggerFactory.getLogger(MultiLayerNetwork.class);
    //the hidden neuralNets
    protected Layer[] layers;


    //default training examples and associated neuralNets
    protected INDArray input, labels;
    //sometimes we may need to transform weights; this allows a
    protected boolean initCalled = false;
    private Collection<IterationListener> listeners = new ArrayList<>();

    protected NeuralNetConfiguration defaultConfiguration;
    protected MultiLayerConfiguration layerWiseConfigurations;
    protected Gradient gradient;
    protected double score;
    private INDArray params;
    /*
      Binary drop connect mask
     */
    protected INDArray mask;

    protected int layerIndex;	//For Layer.get/setIndex()

    protected transient Solver solver;	//Used to call optimizers during backprop


    public MultiLayerNetwork(MultiLayerConfiguration conf) {
        this.layerWiseConfigurations = conf;
        this.defaultConfiguration = conf.getConf(0).clone();
    }

    /**
     * Initialize the network based on the configuration
     *
     * @param conf   the configuration json
     * @param params the parameters
     */
    public MultiLayerNetwork(String conf, INDArray params) {
        this(MultiLayerConfiguration.fromJson(conf));
        init();
        setParameters(params);
    }


    /**
     * Initialize the network based on the configuraiton
     *
     * @param conf   the configuration
     * @param params the parameters
     */
    public MultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
        this(conf);
        init();
        setParameters(params);
    }


    protected void intializeConfigurations() {

        if (layerWiseConfigurations == null)
            layerWiseConfigurations = new MultiLayerConfiguration.Builder().build();

        if (layers == null)
            layers = new Layer[getnLayers()];

        if (defaultConfiguration == null)
            defaultConfiguration = new NeuralNetConfiguration.Builder()
                    .build();
    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     *
     * @param iter the input to iterate on
     *             The typical tip is that the higher k is the closer to the model
     *             you will be approximating due to more sampling. K = 1
     *             usually gives very good results and is the default in quite a few situations.
     */
    public void pretrain(DataSetIterator iter) {
        if (!layerWiseConfigurations.isPretrain())
            return;

        INDArray layerInput;

        for (int i = 0; i < getnLayers(); i++) {
            if (i == 0) {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    if(getLayerWiseConfigurations().getInputPreProcess(i) != null)
                        layerInput = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(next.getFeatureMatrix(),layers[i]);
                    else
                        layerInput = next.getFeatureMatrix();
                    setInput(layerInput);
                      /*During pretrain, feed forward expected activations of network, use activation cooccurrences during pretrain  */
                    if (this.getInput() == null || this.getLayers() == null)
                        initializeLayers(input());
                    getLayers()[i].fit(input());
                    log.info("Training on layer " + (i + 1) + " with " + input().slices() + " examples");
                }

            } else {
                while (iter.hasNext()) {
                    DataSet next = iter.next();
                    layerInput = next.getFeatureMatrix();
                    for (int j = 1; j <= i; j++)
                        layerInput = activationFromPrevLayer(j - 1, layerInput,true);

                    log.info("Training on layer " + (i + 1) + " with " + layerInput.slices() + " examples");
                    getLayers()[i].fit(layerInput);

                }
            }
            iter.reset();
        }
    }


    /**
     * This unsupervised learning method runs
     * contrastive divergence on each RBM layer in the network.
     *
     * @param input the input to iterate on
     *              The typical tip is that the higher k is the closer to the model
     *              you will be approximating due to more sampling. K = 1
     *              usually gives very good results and is the default in quite a few situations.
     */
    public void pretrain(INDArray input) {

        if (!layerWiseConfigurations.isPretrain())
            return;
        /* During pretrain, feed forward expected activations of network, use activation cooccurrences during pretrain  */


        INDArray layerInput = null;

        for (int i = 0; i < getnLayers() - 1; i++) {
            if (i == 0)
                if(getLayerWiseConfigurations().getInputPreProcess(i) != null)
                    layerInput = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input,layers[i]);
                else
                    layerInput = input;
            else
                layerInput = activationFromPrevLayer(i - 1, layerInput,true);
            log.info("Training on layer " + (i + 1) + " with " + layerInput.slices() + " examples");
            getLayers()[i].fit(layerInput);

        }
    }


    @Override
    public int batchSize() {
        return input.slices();
    }

    @Override
    public NeuralNetConfiguration conf() {
        return defaultConfiguration;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray input() {
        return input;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray getParam(String param) {
        //Get params for MultiLayerNetwork sub layers.
        //Parameter keys here: same as MultiLayerNetwork.backprop().
        int idx = param.indexOf("_");
        if( idx == -1 ) throw new IllegalStateException("Invalid param key: not have layer separator: \""+param+"\"");
        int layerIdx = Integer.parseInt(param.substring(0, idx));
        String newKey = param.substring(idx+1);

        return layers[layerIdx].getParam(newKey);
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<String, INDArray> paramTable() {
        //Get all parameters from all layers
        Map<String,INDArray> allParams = new HashMap<>();
        for( int i=0; i<layers.length; i++ ){
            Map<String,INDArray> paramMap = layers[i].paramTable();
            for( Map.Entry<String, INDArray> entry : paramMap.entrySet() ){
                String newKey = i + "_" + entry.getKey();
                allParams.put(newKey, entry.getValue());
            }
        }
        return allParams;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setParam(String key, INDArray val) {
        //Set params for MultiLayerNetwork sub layers.
        //Parameter keys here: same as MultiLayerNetwork.backprop().
        int idx = key.indexOf("_");
        if( idx == -1 ) throw new IllegalStateException("Invalid param key: not have layer separator: \""+key+"\"");
        int layerIdx = Integer.parseInt(key.substring(0, idx));
        String newKey = key.substring(idx+1);

        layers[layerIdx].setParam(newKey,val);
    }


    public MultiLayerConfiguration getLayerWiseConfigurations() {
        return layerWiseConfigurations;
    }

    public void setLayerWiseConfigurations(MultiLayerConfiguration layerWiseConfigurations) {
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

        this.input = input;
        if(input != null)
            setInputMiniBatchSize(input.size(0));

        if (!initCalled)
            init();
    }

    /**
     * Initialize
     */
    public void init() {
        if (layerWiseConfigurations == null || layers == null)
            intializeConfigurations();
        if (initCalled)
            return;

        if (getnLayers() < 1)
            throw new IllegalStateException("Unable to createComplex network neuralNets; number specified is less than 1");

        if (this.layers == null || this.layers[0] == null) {
            if (this.layers == null)
                this.layers = new Layer[getnLayers()];

            // construct multi-layer
            for (int i = 0; i < getnLayers(); i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                layers[i] = LayerFactories.getFactory(conf).create(conf, listeners, i);
            }
            initCalled = true;
            initMask();
        }

        //Set parameters in MultiLayerNetwork.defaultConfiguration for later use in BaseOptimizer.setupSearchState() etc
        //Keyed as per backprop()
        defaultConfiguration.clearVariables();
        for( int i=0; i<layers.length; i++ ){
            for( String s : layers[i].conf().variables() ){
                defaultConfiguration.addVariable(i+"_"+s);
            }
        }

        //all params are views
        reDistributeParams();
    }


    /**
     * Triggers the activation of the last hidden layer ie: not logistic regression
     *
     * @return the activation of the last hidden layer given the last input to the network
     */
    public INDArray activate() {
        return getLayers()[getLayers().length - 1].activate();
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

    @Override
    public INDArray activate(INDArray input) {
        throw new UnsupportedOperationException();
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

    @Override
    public INDArray activationMean() {
        //TODO determine how to pass back all activationMean for MLN
        throw new UnsupportedOperationException();
//        List<INDArray> avgActivations =  new ArrayList<>();
//
//        for( Layer layer: getLayers() ){
//            avgActivations.add(layer.activationMean());
//            }
//        return Nd4j.toFlattened(avgActivations);
    }


    /**
     * Redistribute parameters handles
     * having parameters as a view
     */
    public void reDistributeParams() {
        List<INDArray> params = new ArrayList<>();
        for(Layer l : layers) {
            INDArray paramsForL = l.params();
            params.add(paramsForL);
        }

        this.params = Nd4j.toFlattened(params);
        int idx = 0;
        for(Layer l : layers) {
            INDArray paramsForL = l.params();
            params.add(paramsForL);
            int range = l.numParams();
            INDArray get = this.params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
            if (get.length() < 1)
                continue;
            l.setParams(get);
            idx += range;
        }


    }

    /**
     * Sets the input and labels from this dataset
     *
     * @param data the dataset to initialize with
     */
    public void initialize(DataSet data) {
        setInput(data.getFeatureMatrix());
        feedForward(getInput());
        this.labels = data.getLabels();
        if (getOutputLayer() instanceof BaseOutputLayer) {
            BaseOutputLayer o = (BaseOutputLayer) getOutputLayer();
            o.setLabels(labels);
        }
    }


    /**
     * Compute input linear transformation (z) from previous layer
     * Apply pre processing transformation where necessary
     *
     * @param curr  the current layer
     * @param input the input
     * @param training training or test mode
     * @return the activation from the previous layer
     */
    public INDArray zFromPrevLayer(int curr, INDArray input,boolean training) {
        if(getLayerWiseConfigurations().getInputPreProcess(curr) != null)
            input = getLayerWiseConfigurations().getInputPreProcess(curr).preProcess(input,layers[curr]);

        INDArray ret = layers[curr].preOutput(input, training);
        return ret;
    }

    /**
     * Calculate activation from previous layer including pre processing where necessary
     *
     * @param curr  the current layer
     * @param input the input
     * @return the activation from the previous layer
     */
    public INDArray activationFromPrevLayer(int curr, INDArray input,boolean training) {
        if(getLayerWiseConfigurations().getInputPreProcess(curr) != null)
            input = getLayerWiseConfigurations().getInputPreProcess(curr).preProcess(input,layers[curr]);
        INDArray ret = layers[curr].activate(input, training);
        return ret;
    }


    /**
     * * Compute input linear transformation (z) of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> computeZ(boolean training) {
        INDArray currInput = this.input;

        List<INDArray> activations = new ArrayList<>();
        activations.add(currInput);

        for (int i = 0; i < layers.length; i++) {
            currInput = zFromPrevLayer(i, currInput,training);
            //applies drop connect to the activation
            activations.add(currInput);
        }


        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> computeZ(INDArray input,boolean training) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        else if (this.getLayerWiseConfigurations().getInputPreProcess(0) != null)
            setInput(getLayerWiseConfigurations().getInputPreProcess(0).preProcess(input,layers[0]));
        else
            setInput(input);
        return computeZ(training);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(INDArray input, boolean test) {
        setInput(input);
        return feedForward(test);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(boolean test) {
        INDArray currInput = input;
        List<INDArray> activations = new ArrayList<>();
        activations.add(currInput);

        for (int i = 0; i < layers.length; i++) {
            currInput = activationFromPrevLayer(i, currInput,test);
            //applies drop connect to the activation
            activations.add(currInput);
        }
        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward() {
        return feedForward(false);
    }


    /**
     * Compute input linear transformation (z)
     * Compute activations (applies activation transformation to z)
     *
     * @return a pair of activations and corresponding derivatives
     */
    public Pair<List<INDArray>,List<INDArray>> feedForwardActivationsAndDerivatives(boolean training) {
        INDArray currInput = input;

        List<INDArray> activations = new ArrayList<>();
        List<INDArray> derivatives = new ArrayList<>();
        activations.add(currInput);

        for (int i = 0; i < layers.length; i++) {
            currInput = zFromPrevLayer(i, currInput,training); // w*x+b for each layer
            //special case: row wise softmax
            if (layers[i].conf().getLayer().getActivationFunction().equals("softmax"))
                activations.add(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform("softmax", currInput.dup()), 1));
            else
                activations.add(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(layerWiseConfigurations.getConf(i).getLayer().getActivationFunction(), currInput)));
        }

        currInput = this.input;
        for (int i = 0; i < layers.length; i++) {
            currInput = zFromPrevLayer(i, currInput,training); // w*x+b for each layer
            INDArray dup = currInput.dup();
            //special case: row wise softmax
            if (layers[i].conf().getLayer().getActivationFunction().equals("softmax"))
                derivatives.add(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(layerWiseConfigurations.getConf(i).getLayer().getActivationFunction(), dup).derivative(), 1));
            else
                derivatives.add(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(layerWiseConfigurations.getConf(i).getLayer().getActivationFunction(), dup).derivative()));
        }
        // Duplicating last layer derivative to keep pair list equal
        derivatives.add(derivatives.get(layers.length - 1));
        return new Pair<>(activations, derivatives);
    }


    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(INDArray input) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        else if (this.getLayerWiseConfigurations().getInputPreProcess(0) != null)
            setInput(getLayerWiseConfigurations().getInputPreProcess(0).preProcess(input,layers[0]));
        else
            setInput(input);
        return feedForward();
    }


    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
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
        List<String> activationFunctions = new ArrayList<>();


        for (int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getParam(DefaultParamInitializer.WEIGHT_KEY));
            biases.add(getLayers()[j].getParam(DefaultParamInitializer.BIAS_KEY));
            activationFunctions.add(getLayers()[j].conf().getLayer().getActivationFunction());
        }


        INDArray rix = rActivations.get(rActivations.size() - 1).divi((double) input.slices());
        LinAlgExceptions.assertValidNum(rix);

        //errors
        for (int i = getnLayers() - 1; i >= 0; i--) {
            //W^t * error^l + 1
            deltas[i] = activations.get(i).transpose().mmul(rix);

            if (i > 0)
                rix = rix.mmul(weights.get(i).addRowVector(biases.get(i)).transpose()).muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunctions.get(i - 1), activations.get(i)).derivative()));

        }

        for (int i = 0; i < deltas.length - 1; i++) {
            if (defaultConfiguration.isConstrainGradientToUnitNorm()) {
                double sum = deltas[i].sum(Integer.MAX_VALUE).getDouble(0);
                if (sum > 0)
                    deltaRet.add(deltas[i].div(deltas[i].norm2(Integer.MAX_VALUE)));
                else
                    deltaRet.add(deltas[i]);
            } else
                deltaRet.add(deltas[i]);
            LinAlgExceptions.assertValidNum(deltaRet.get(i));
        }

        return deltaRet;
    }


    //damping update after line search
    public void dampingUpdate(double rho, double boost, double decrease) {
        if (rho < 0.25 || Double.isNaN(rho))
            layerWiseConfigurations.setDampingFactor(getLayerWiseConfigurations().getDampingFactor() * boost);


        else if (rho > 0.75)
            layerWiseConfigurations.setDampingFactor(getLayerWiseConfigurations().getDampingFactor() * decrease);
    }

    /* p and gradient are same length */
    public double reductionRatio(INDArray p, double currScore, double score, INDArray gradient) {
        double currentDamp = layerWiseConfigurations.getDampingFactor();
        layerWiseConfigurations.setDampingFactor(0);
        INDArray denom = getBackPropRGradient(p);
        denom.muli(0.5).muli(p.mul(denom)).sum(0);
        denom.subi(gradient.mul(p).sum(0));
        double rho = (currScore - score) / (double) denom.getScalar(0).element();
        layerWiseConfigurations.setDampingFactor(currentDamp);
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
        INDArray ix = activations.get(activations.size() - 1).sub(labels).div(labels.slices());

       	/*
		 * Precompute activations and z's (pre activation network outputs)
		 */
        List<INDArray> weights = new ArrayList<>();
        List<INDArray> biases = new ArrayList<>();

        List<String> activationFunctions = new ArrayList<>();
        for (int j = 0; j < getLayers().length; j++) {
            weights.add(getLayers()[j].getParam(DefaultParamInitializer.WEIGHT_KEY));
            biases.add(getLayers()[j].getParam(DefaultParamInitializer.BIAS_KEY));
            activationFunctions.add(getLayers()[j].conf().getLayer().getActivationFunction());
        }


        //errors
        for (int i = weights.size() - 1; i >= 0; i--) {
            deltas[i] = activations.get(i).transpose().mmul(ix);
            preCons[i] = Transforms.pow(activations.get(i).transpose(), 2).mmul(Transforms.pow(ix, 2)).muli(labels.slices());

            if (i > 0) {
                //W[i] + b[i] * f'(z[i - 1])
                ix = ix.mmul(weights.get(i).transpose()).muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(activationFunctions.get(i - 1), activations.get(i)).derivative()));
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
    public MultiLayerNetwork clone() {
        MultiLayerNetwork ret;
        try {
            Constructor<MultiLayerNetwork> constructor = (Constructor<MultiLayerNetwork>) getClass().getDeclaredConstructor(MultiLayerConfiguration.class);
            ret = constructor.newInstance(getLayerWiseConfigurations());
            ret.update(this);

        } catch (Exception e) {
            throw new IllegalStateException("Unable to cloe network");
        }
        return ret;
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
        if(params != null)
            return params;

        List<INDArray> params = new ArrayList<>();
        for (Layer layer: getLayers())
            if(!(layer instanceof SubsamplingLayer)) {
                params.add(layer.params());
            }

        return Nd4j.toFlattened(params);
    }

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray
     * which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        if(this.params != null) {
            this.params = params;
            int idx = 0;
            for (int i = 0; i < getLayers().length; i++) {
                Layer layer = getLayer(i);
                int range = layer.numParams();
                INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
//                if (get.length() < 1)
//                    throw new IllegalStateException("Unable to retrieve layer. No params found (length was 0");
                layer.setParams(get);
                idx += range;

            }
        }
        int idx = 0;
        for (int i = 0; i < getLayers().length; i++) {
            Layer layer = getLayer(i);
            int range = layer.numParams();
            INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
//            if (get.length() < 1)
//                throw new IllegalStateException("Unable to retrieve layer. No params found (length was 0");
            layer.setParams(get);
            idx += range;

        }
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
        for (int i = 0; i < layers.length; i++)
            length += layers[i].numParams();

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
        return params();

    }

    /**
     * Packs a set of matrices in to one vector
     *
     * @param layers the neuralNets to pack
     * @return a singular matrix of all of the neuralNets packed in to one matrix
     */
    public INDArray pack(List<Pair<INDArray, INDArray>> layers) {
        List<INDArray> list = new ArrayList<>();

        for (Pair<INDArray, INDArray> layer : layers) {
            list.add(layer.getFirst());
            list.add(layer.getSecond());
        }
        return Nd4j.toFlattened(list);
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(org.nd4j.linalg.dataset.api.DataSet data) {
        return f1Score(data.getFeatureMatrix(), data.getLabels());
    }


    /**
     * Unpacks a parameter matrix in to a
     * transform of pairs(w,hbias)
     * triples with layer wise
     *
     * @param param the param vector
     * @return a segmented list of the param vector
     */
    public List<Pair<INDArray, INDArray>> unPack(INDArray param) {
        //more sanity checks!
        if (param.slices() != 1)
            param = param.reshape(1, param.length());
        List<Pair<INDArray, INDArray>> ret = new ArrayList<>();
        int curr = 0;
        for (int i = 0; i < layers.length; i++) {
            int layerLength = layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).length() + layers[i].getParam(DefaultParamInitializer.BIAS_KEY).length();
            INDArray subMatrix = param.get(NDArrayIndex.interval(curr, curr + layerLength));
            INDArray weightPortion = subMatrix.get(NDArrayIndex.interval(0, layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).length()));

            int beginHBias = layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).length();
            int endHbias = subMatrix.length();
            INDArray hBiasPortion = subMatrix.get(NDArrayIndex.interval(beginHBias, endHbias));
            int layerLengthSum = weightPortion.length() + hBiasPortion.length();
            if (layerLengthSum != layerLength) {
                if (hBiasPortion.length() != layers[i].getParam(DefaultParamInitializer.BIAS_KEY).length())
                    throw new IllegalStateException("Hidden bias on layer " + i + " was off");
                if (weightPortion.length() != layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).length())
                    throw new IllegalStateException("Weight portion on layer " + i + " was off");

            }

            ret.add(new Pair<>(weightPortion.reshape(layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).slices(), layers[i].getParam(DefaultParamInitializer.WEIGHT_KEY).columns()), hBiasPortion.reshape(layers[i].getParam(DefaultParamInitializer.BIAS_KEY).slices(), layers[i].getParam(DefaultParamInitializer.BIAS_KEY).columns())));
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

        //precompute deltas
        List<Pair<INDArray, INDArray>> deltas = computeDeltas2();


        List<Pair<Pair<INDArray, INDArray>, Pair<INDArray, INDArray>>> list = new ArrayList<>();
        List<Pair<INDArray, INDArray>> grad = new ArrayList<>();
        List<Pair<INDArray, INDArray>> preCon = new ArrayList<>();

        for (int l = 0; l < deltas.size(); l++) {
            INDArray gradientChange = deltas.get(l).getFirst();
            INDArray preConGradientChange = deltas.get(l).getSecond();


            if (l < layers.length && gradientChange.length() != layers[l].getParam(DefaultParamInitializer.WEIGHT_KEY).length())
                throw new IllegalStateException("Gradient change not equal to weight change");

            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).getFirst().mean(0);
            INDArray preConColumnSums = deltas.get(l).getSecond().mean(0);

            grad.add(new Pair<>(gradientChange, deltaColumnSums));
            preCon.add(new Pair<>(preConGradientChange, preConColumnSums));
            if (l < layers.length && deltaColumnSums.length() != layers[l].getParam(DefaultParamInitializer.BIAS_KEY).length())
                throw new IllegalStateException("Bias change not equal to weight change");
            else if (l == getLayers().length && deltaColumnSums.length() != getOutputLayer().getParam(DefaultParamInitializer.BIAS_KEY).length())
                throw new IllegalStateException("Bias change not equal to weight change");


        }

        INDArray g = pack(grad);
        INDArray con = pack(preCon);
        INDArray theta = params();


        if (mask == null)
            initMask();

        g.addi(theta.mul(defaultConfiguration.getL2()).muli(mask));

        INDArray conAdd = Transforms.pow(mask.mul(defaultConfiguration.getL2()).add(Nd4j.valueArrayOf(g.slices(), g.columns(), layerWiseConfigurations.getDampingFactor())), 3.0 / 4.0);

        con.addi(conAdd);

        List<Pair<INDArray, INDArray>> gUnpacked = unPack(g);

        List<Pair<INDArray, INDArray>> conUnpacked = unPack(con);

        for (int i = 0; i < gUnpacked.size(); i++)
            list.add(new Pair<>(gUnpacked.get(i), conUnpacked.get(i)));


        return list;

    }


    @Override
    public void fit(DataSetIterator iter) {
        if (layerWiseConfigurations.isPretrain()) {
            pretrain(iter);
            iter.reset();
            while (iter.hasNext()) {
                DataSet next = iter.next();
                if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;
                setInput(next.getFeatureMatrix());
                setLabels(next.getLabels());
                finetune();
            }

        }
        if (layerWiseConfigurations.isBackprop()) {
            iter.reset();
            while (iter.hasNext()) {
                DataSet next = iter.next();
                if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;
                setInput(next.getFeatureMatrix());
                setLabels(next.getLabels());
                if( solver == null ){
                    solver = new Solver.Builder()
                            .configure(conf())
                            .listeners(getListeners())
                            .model(this).build();
                }
                solver.optimize();
            }
        }
    }

    protected void backprop() {
        String multiGradientKey;
        gradient = new DefaultGradient();
        Layer currLayer;

        if(!(getOutputLayer() instanceof BaseOutputLayer)) {
            log.warn("Warning: final layer isn't output layer. You cannot use backprop without an output layer.");
            return;
        }

        BaseOutputLayer outputLayer = (BaseOutputLayer) getOutputLayer();
        if(labels == null)
            throw new IllegalStateException("No labels found");
        if(outputLayer.conf().getLayer().getWeightInit() == WeightInit.ZERO){
            throw new IllegalStateException("Output layer weights cannot be initialized to zero when using backprop.");
        }

        outputLayer.setLabels(labels);

        //calculate and apply the backward gradient for every layer
        /**
         * Skip the output layer for the indexing and just loop backwards updating the coefficients for each layer.
         *
         * Activate applies the activation function for each layer and sets that as the input for the following layer.
         *
         * Typical literature contains most trivial case for the error calculation: wT * weights
         * This interpretation transpose a few things to get mini batch because ND4J is rows vs columns organization for params
         */
        int numLayers = getnLayers();
        //Store gradients is a list; used to ensure iteration order in DefaultGradient linked hash map. i.e., layer 0 first instead of output layer
        LinkedList<Pair<String,INDArray>> gradientList = new LinkedList<>();

        Pair<Gradient,INDArray> currPair = outputLayer.backpropGradient(null);

        for( Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
            multiGradientKey = String.valueOf(numLayers - 1) + "_" + entry.getKey();
            gradientList.addLast(new Pair<>(multiGradientKey,entry.getValue()));
        }

        if(getLayerWiseConfigurations().getInputPreProcess(numLayers-1) != null)
            currPair = new Pair<> (currPair.getFirst(), this.layerWiseConfigurations.getInputPreProcess(numLayers - 1).backprop(currPair.getSecond(),layers[numLayers-1]));

        // Calculate gradients for previous layers & drops output layer in count
        for(int j = numLayers - 2; j >= 0; j--) {
            currLayer = getLayer(j);
            currPair = currLayer.backpropGradient(currPair.getSecond());

            LinkedList<Pair<String,INDArray>> tempList = new LinkedList<>();
            for( Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet() ){
                multiGradientKey = String.valueOf(j) + "_" + entry.getKey();
                tempList.addFirst(new Pair<>(multiGradientKey,entry.getValue()));
            }
            for(Pair<String,INDArray> pair : tempList) gradientList.addFirst(pair);

            //Pass epsilon through input processor before passing to next layer (if applicable)
            if(getLayerWiseConfigurations().getInputPreProcess(j) != null)
                currPair = new Pair<> (currPair.getFirst(), getLayerWiseConfigurations().getInputPreProcess(j).backprop(currPair.getSecond(),layers[j]));
        }

        //Add gradients to Gradients, in correct order
        for( Pair<String,INDArray> pair : gradientList ) gradient.setGradientFor(pair.getFirst(), pair.getSecond());
    }


    /**
     *
     * @return
     */
    public Collection<IterationListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.listeners = listeners;

        if (layers == null) {
            init();
        }
        for (Layer layer : layers) {
            layer.setListeners(listeners);
        }
    }


    @Override
    public void setListeners(IterationListener... listeners) {
        Collection<IterationListener> cListeners = new ArrayList<>();
        for(IterationListener listener : listeners)
            cListeners.add(listener);
        setListeners(cListeners);
    }


    /**
     * Run SGD based on the given labels
     *
     */
    public void finetune() {
        if (!(getOutputLayer() instanceof BaseOutputLayer)) {
            log.warn("Output layer not instance of output layer returning.");
            return;
        }
        if(labels == null)
            throw new IllegalStateException("No labels found");

        log.info("Finetune phase");
        BaseOutputLayer output = (BaseOutputLayer) getOutputLayer();
        if (output.conf().getOptimizationAlgo() != OptimizationAlgorithm.HESSIAN_FREE) {
            feedForward();
            output.fit(output.input(), labels);
        } else {
            throw new UnsupportedOperationException();
        }
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
        int[] ret = new int[d.slices()];
        if (d.isRowVector()) ret[0] = Nd4j.getBlasWrapper().iamax(output);
        else {
            for (int i = 0; i < ret.length; i++)
                ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        }
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
        BaseOutputLayer o = (BaseOutputLayer) getOutputLayer();
        return o.labelProbabilities(feed.get(feed.size() - 1));
    }

    /**
     * Fit the model
     *
     * @param data   the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray data, INDArray labels) {
        setInput(data.dup());
        setLabels(labels.dup());

        if (layerWiseConfigurations.isPretrain()) {
            pretrain(data);
            finetune();
        }

        if(layerWiseConfigurations.isBackprop()) {
            if( solver == null ){
                solver = new Solver.Builder()
                        .configure(conf())
                        .listeners(getListeners())
                        .model(this).build();
            }

            solver.optimize();
        }
    }

    /**
     * Fit the unsupervised model
     *
     * @param data the examples to classify (one example in each row)
     */

    @Override
    public void fit(INDArray data) {
        setInput(data.dup());
        pretrain(data);
    }

    @Override
    public void iterate(INDArray input) {
        pretrain(input);
    }


    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(org.nd4j.linalg.dataset.api.DataSet data) {
        fit(data.getFeatureMatrix(), data.getLabels());
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        org.deeplearning4j.nn.conf.layers.OutputLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.OutputLayer) getOutputLayer().conf().getLayer();
        fit(examples, FeatureUtil.toOutcomeMatrix(labels, layerConf.getNOut()));
    }


    /**
     * Label the probabilities of the input
     *
     * @param input    the input to label
     * @param test whether the output
     *             is test or train. This mainly
     *             affect hyper parameters such as
     *             drop out where certain things should
     *             be applied with activations
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray input, boolean test) {
        List<INDArray> activations = feedForward(input, test);
        //last activation is input
        return activations.get(activations.size() - 1);
    }

    /**
     * Label the probabilities of the input
     *
     * @param input the input to label
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray input) {
        return output(input, false);
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
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray reconstruct(INDArray x, int layerNum) {
        List<INDArray> forward = feedForward(x);
        return forward.get(layerNum - 1);
    }


    /**
     * Prints the configuration
     */
    public void printConfiguration() {
        StringBuilder sb = new StringBuilder();
        int count = 0;
        for (NeuralNetConfiguration conf : getLayerWiseConfigurations().getConfs()) {
            sb.append(" Layer " + count++ + " conf " + conf);
        }

        log.info(sb.toString());
    }


    /**
     * Assigns the parameters of this model to the ones specified by this
     * network. This is used in loading from input streams, factory methods, etc
     *
     * @param network the network to getFromOrigin parameters from
     */
    public void update(MultiLayerNetwork network) {
        this.defaultConfiguration = network.defaultConfiguration;
        setInput(network.input);
        this.labels = network.labels;
        this.layers = ArrayUtils.clone(network.layers);
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
    public double f1Score(INDArray input, INDArray labels) {
        feedForward(input);
        setLabels(labels);
        Evaluation eval = new Evaluation();
        eval.eval(labels, labelProbabilities(input));
        return eval.f1();
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
    public double score(DataSet data) {
        feedForward(data.getFeatureMatrix());
        setLabels(data.getLabels());
        if( getOutputLayer() instanceof BaseOutputLayer ){
            BaseOutputLayer ol = (BaseOutputLayer)getOutputLayer();
            ol.setLabels(data.getLabels());
            ol.computeScore(calcL1(),calcL2());
            this.score = ol.score();
        } else {
            log.warn("Cannot calculate score wrt labels without an OutputLayer");
            return 0.0;
        }
        return score();
    }


    @Override
    public void fit() {
        fit(input, labels);
    }

    @Override
    public void update(INDArray gradient, String paramType) {
    }


    /**
     * Score of the model (relative to the objective function)
     *
     * @return the score of the model (relative to the objective function)
     */
    @Override
    public double score() {
        return score;
    }

    @Override
    public void computeGradientAndScore() {
        //Calculate activations (which are stored in each layer, and used in backprop)
        feedForward();
        backprop();
        score = ((BaseOutputLayer)getOutputLayer()).computeScore(calcL1(),calcL2());
        // Updating activations based on new gradients

    }

    @Override
    public void accumulateScore(double accum) {

    }

    /**
     * Clear the inputs. Clears optimizer state.
     */
    public void clear() {
        for (Layer layer : layers)
            layer.clear();

        input = null;
        labels = null;
        solver = null;
    }

    /**
     * Score of the model (relative to the objective function)
     *
     * @param param the current parameters
     * @return the score of the model (relative to the objective function)
     */

    public double score(INDArray param) {
        INDArray params = params();
        setParameters(param);
        double ret = score();
        double regCost = 0.5f * defaultConfiguration.getL2() * (double) Transforms.pow(mask.mul(param), 2).sum(Integer.MAX_VALUE).element();
        setParameters(params);
        return ret + regCost;
    }

    /**
     * Averages the given logistic regression
     * from a mini batch in to this one
     *
     * @param layer     the logistic regression to average in to this one
     * @param batchSize the batch size
     */
    @Override
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
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
    public void merge(MultiLayerNetwork network, int batchSize) {
        if (network.layers.length != layers.length)
            throw new IllegalArgumentException("Unable to merge networks that are not of equal length");
        for (int i = 0; i < getnLayers(); i++) {
            Layer n = layers[i];
            Layer otherNetwork = network.layers[i];
            n.merge(otherNetwork, batchSize);

        }

        getOutputLayer().merge(network.getOutputLayer(), batchSize);
    }


    /**
     * Note that if input isn't null
     * and the neuralNets are null, this is a way
     * of initializing the neural network
     *
     * @param input
     */
    public void setInput(INDArray input) {
        this.input = input;
        if (this.layers == null)
            this.initializeLayers(getInput());
        if(input != null)
            setInputMiniBatchSize(input.size(0));
    }

    private void initMask() {
        setMask(Nd4j.ones(1, pack().length()));
    }


    /**
     * Get the output layer
     *
     * @return
     */
    public Layer getOutputLayer() {
        return getLayers()[getLayers().length - 1];
    }


    /**
     * Sets parameters for the model.
     * This is used to manipulate the weights and biases across
     * all neuralNets (including the output layer)
     *
     * @param params a parameter vector equal 1,numParameters
     */
    public void setParameters(INDArray params) {
        int idx = 0;
        for (int i = 0; i < getLayers().length; i++) {
            Layer layer = getLayer(i);
            if(!(layer instanceof SubsamplingLayer)) {
                int range = layer.numParams();
                INDArray get = params.get(NDArrayIndex.point(0),NDArrayIndex.interval(idx, range + idx));
                if (get.length() < 1)
                    throw new IllegalStateException("Unable to retrieve layer. No params found (length was 0");
                layer.setParams(get);
                idx += range;
            }
        }
    }


    /**
     * Feed forward with the r operator
     *
     * @param v the v for the r operator
     * @return the activations based on the r operator
     */
    public List<INDArray> feedForwardR(List<INDArray> acts, INDArray v) {
        List<INDArray> R = new ArrayList<>();
        R.add(Nd4j.zeros(input.slices(), input.columns()));
        List<Pair<INDArray, INDArray>> vWvB = unPack(v);
        List<INDArray> W = MultiLayerUtil.weightMatrices(this);

        for (int i = 0; i < layers.length; i++) {
            String derivative = getLayers()[i].conf().getLayer().getActivationFunction();
            //R[i] * W[i] + acts[i] * (vW[i] + vB[i]) .* f'([acts[i + 1])
            R.add(R.get(i).mmul(W.get(i)).addi(acts.get(i)
                    .mmul(vWvB.get(i).getFirst().addiRowVector(vWvB.get(i).getSecond())))
                    .muli((Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(derivative, acts.get(i + 1)).derivative()))));
        }

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

            if (gradientChange.length() != getLayers()[l].getParam(DefaultParamInitializer.WEIGHT_KEY).length())
                throw new IllegalStateException("Gradient change not equal to weight change");


            //update hidden bias
            INDArray deltaColumnSums = deltas.get(l).mean(0);
            if (deltaColumnSums.length() != layers[l].getParam(DefaultParamInitializer.BIAS_KEY).length())
                throw new IllegalStateException("Bias change not equal to weight change");


            list.add(new Pair<>(gradientChange, deltaColumnSums));


        }

        INDArray pack = pack(list).addi(mask.mul(defaultConfiguration.getL2())
                .muli(v)).addi(v.mul(layerWiseConfigurations.getDampingFactor()));
        return unPack(pack);

    }


    public INDArray getLabels() {
        return labels;
    }

    public INDArray getInput() {
        return input;
    }


    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    /**
     * Get the number of layers in the network
     *
     * @return the number of layers in the network
     */
    public int getnLayers() {
        return layerWiseConfigurations.getConfs().size();
    }

    public Layer[] getLayers() {
        return layers;
    }

    public Layer getLayer(int i) {
        return layers[i];
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public INDArray getMask() {
        return mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }

    //==========
    //Layer methods

    @Override
    public Gradient error(INDArray errorSignal) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Type type() {
        return Type.MULTILAYER;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x) {
        INDArray lastLayerActivation = x;
        for( int i=0; i<layers.length-1; i++ ){
        	if(getLayerWiseConfigurations().getInputPreProcess(i) != null)
                lastLayerActivation = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(lastLayerActivation,layers[i]);
            lastLayerActivation = layers[i].activate(lastLayerActivation);
        }
        if(getLayerWiseConfigurations().getInputPreProcess(layers.length-1) != null)
            lastLayerActivation = getLayerWiseConfigurations().getInputPreProcess(layers.length-1).preProcess(lastLayerActivation,layers[layers.length-1]);
        return layers[layers.length-1].preOutput(lastLayerActivation);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<Gradient,INDArray> backpropGradient(INDArray epsilon) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public void setIndex(int index){
        layerIndex = index;
    }

    @Override
    public int getIndex(){
        return layerIndex;
    }

    @Override
    public double calcL2() {
        double l2 = 0.0;
        for( int i=0; i<layers.length; i++ ){
            l2 += layers[i].calcL2();
        }
        return l2;
    }

    @Override
    public double calcL1() {
        double l1 = 0.0;
        for( int i=0; i<layers.length; i++ ){
            l1 += layers[i].calcL1();
        }
        return l1;
    }

    @Override
    public void update(Gradient gradient) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray activate(boolean training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setInputMiniBatchSize(int size){
        if(layers != null)
            for(Layer l : layers)
                l.setInputMiniBatchSize(size);
    }

    @Override
    public int getInputMiniBatchSize(){
        return layers[0].getInputMiniBatchSize();
    }
    
    /**If this MultiLayerNetwork contains one or more RNN layers: conduct forward pass (prediction)
     * but using previous stored state for any RNN layers. The activations for the final step are
     * also stored in the RNN layers for use next time rnnTimeStep() is called.<br>
     * If no previous state is present in RNN layers, the default value (typically 0) is used.<br>
     * This can be used to generate output one or a few step/s at a time instead of always having to do
     * forward pass from t=0
     * @param input Input to lowest layer. May be for one or multiple time steps.
     * @return Output activations.
     */
    public INDArray rnnTimeStep(INDArray input){
    	for( int i=0; i<layers.length; i++ ){
    		if(getLayerWiseConfigurations().getInputPreProcess(i) != null)
                input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input,layers[i]);
    		if(layers[i] instanceof BaseRecurrentLayer){
    			input = ((BaseRecurrentLayer<?>)layers[i]).rnnTimeStep(input);
    		} else {
    			input = layers[i].activate(input, false);
    		}
    	}
    	return input;
    }
    
    /**Get the state of the RNN layer, as used in rnnTimeStep().
     * @param layer Number/index of the layer.
     * @return Hidden state, or null if layer is not an RNN layer
     */
    public Map<String,INDArray> rnnGetPreviousState(int layer){
    	if(layer < 0 || layer >= layers.length ) throw new IllegalArgumentException("Invalid layer number");
    	if( !(layers[layer] instanceof BaseRecurrentLayer) ) throw new IllegalArgumentException("Layer is not an RNN layer");
    	return ((BaseRecurrentLayer<?>)layers[layer]).rnnGetPreviousState();
    }
    
    /**Set the state of the RNN layer.
     * @param layer The number/index of the layer.
     * @param state The state to set the specified layer to
     */
	public void rnnSetPreviousState(int layer, Map<String,INDArray> state){
    	if(layer < 0 || layer >= layers.length ) throw new IllegalArgumentException("Invalid layer number");
    	if( !(layers[layer] instanceof BaseRecurrentLayer) ) throw new IllegalArgumentException("Layer is not an RNN layer");
    	
    	BaseRecurrentLayer<?> r = (BaseRecurrentLayer<?>)layers[layer];
    	r.rnnSetPreviousState(state);
    }
    
    /** Clear the previous state of the RNN layers (if any).
     */
    public void rnnClearPreviousState(){
    	if( layers == null ) return;
    	for( int i=0; i<layers.length; i++ ){
    		if( layers[i] instanceof BaseRecurrentLayer ) ((BaseRecurrentLayer<?>)layers[i]).rnnClearPreviousState();
    	}
    }
}
