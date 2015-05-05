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

package org.deeplearning4j.nn.layers;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import java.util.Map;

/**
 * A layer with a bias and activation function
 * @author Adam Gibson
 */
public abstract class BaseLayer implements Layer {

    protected INDArray input;
    protected Map<String,INDArray> params;
    protected NeuralNetConfiguration conf;
    protected INDArray dropoutMask;
    protected ParamInitializer paramInitializer;
    protected double score = 0.0;
    protected ConvexOptimizer optimizer;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
    
    public BaseLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    public BaseLayer(NeuralNetConfiguration conf, INDArray input) {
        this.input = input;
        this.conf = conf;
    }
    
    public Collection<IterationListener> getIterationListeners() {
        return iterationListeners;
    }

    public void setIterationListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }

    @Override
    public Gradient error(INDArray errorSignal) {
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        Gradient nextLayerGradient = new DefaultGradient();
        INDArray wErrorSignal = errorSignal.mmul(W);
        nextLayerGradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,wErrorSignal);
        return nextLayerGradient;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        INDArray deriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getActivationFunction(), activate(input)).derivative());
        return deriv;
    }

    @Override
    public Gradient errorSignal(Gradient error, INDArray input) {
        INDArray derivative = derivativeActivation(input);
        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,derivative.mul(error.getGradientFor(DefaultParamInitializer.WEIGHT_KEY)));
        return ret;
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        Gradient ret = new DefaultGradient();
        INDArray weightErrorSignal = layerError.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray weightError = weightErrorSignal.transpose().mmul(activation).divi(weightErrorSignal.rows()).transpose();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,weightError);
        INDArray biasGradient = weightError.sum(0).divi(weightErrorSignal.rows());
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasGradient);

        return ret;
    }

    @Override
    public Gradient backwardGradient(INDArray activation, Gradient errorSignal) {
        Gradient propError = error(activation);
        INDArray deriv = derivativeActivation(activation);
        Gradient ret = new DefaultGradient();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,propError.getGradientFor(DefaultParamInitializer.WEIGHT_KEY).mul(deriv));
        return ret;
    }

    @Override
    public void fit() {
        fit(this.input);
    }

    @Override
    public void setScore() {

    }


    /**
     * iterate one iteration of the network
     *
     * @param input  the input to iterate on
     */
    @Override
    public void iterate(INDArray input) {
        this.input = input;
        Gradient gradient = gradient();
        update(gradient);
    }



    @Override
    public void update(Gradient gradient) {
        for(String s : conf.variables()) {
            if(gradient.gradientForVariable().containsKey(s))
                getParam(s).addi(gradient.gradientForVariable().get(s));
        }
    }


    @Override
    public ConvexOptimizer getOptimizer() {
        if(optimizer == null) {
            Solver solver = new Solver.Builder()
                    .model(this)
                    .configure(conf())
                    .listeners(getIterationListeners())
                    .build();
            this.optimizer = solver.getOptimizer();
        }
        return optimizer;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    @Override
    public void setParam(String key, INDArray val) {
        params.put(key,val);
    }

    /**
     * Returns the parameters of the neural network
     *
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        List<INDArray> ret = new ArrayList<>();
        for(String s : params.keySet())
            ret.add(params.get(s));
        return Nd4j.toFlattened(ret);
    }

    @Override
    public void setParams(INDArray params) {
        List<String> gradientList = conf.variables();
        int length = 0;
        for(String s : gradientList)
            length += getParam(s).length();
        if(params.length() != length)
            throw new IllegalArgumentException("Unable to set parameters: must be of length " + length);
        int idx = 0;
        for(int i = 0; i < gradientList.size(); i++) {
            INDArray param = getParam(gradientList.get(i));
            INDArray get = params.get(NDArrayIndex.interval(idx,idx + param.length()));
            if(param.length() != get.length())
                throw new IllegalStateException("Parameter " + gradientList.get(i) + " should have been of length " + param.length() + " but was " + get.length());
            param.assign(get.reshape(param.shape()));
            idx += param.length();
        }

        setScore();

    }

    @Override
    public void initParams() {
        paramInitializer.init(paramTable(),conf());
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return params;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    /**
     * Classify input
     * @param x the input (can either be a matrix or vector)
     * If it's a matrix, each row is considered an example
     * and associated rows are classified accordingly.
     * Each row will be the likelihood of a label given that example
     * @return a probability distribution for each row
     */
    @Override
    public  INDArray preOutput(INDArray x) {
        if(x == null)
            throw new IllegalArgumentException("No null input allowed");

        this.input = x;
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);


        INDArray ret = input().mmul(W).addiRowVector(b);
        return ret;


    }


    @Override
    public int batchSize() {
        return input.rows();
    }

    @Override
    public  INDArray activate() {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.getActivationFunction().equals("softmax"))
            return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), input().mmul(W).addiRowVector(b)));
        else
            return Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), input().mmul(W).addiRowVector(b)));

    }

    @Override
    public  INDArray activate(INDArray input) {
        this.input = input;
        return activate();
    }


    @Override
    public INDArray activationMean() {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        return input().mmul(W).addiRowVector(b);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }



    @Override
    public void clear() {
        if(input != null) {
            input.data().destroy();
            input = null;
        }
    }

    protected void applyDropOutIfNecessary(INDArray input) {
        if(conf.getDropOut() > 0) {
            this.dropoutMask = Nd4j.rand(input.rows(), input.columns()).gt(conf.getDropOut());
        }

        else if(this.dropoutMask != null)
            this.dropoutMask = Nd4j.ones(input.rows(), conf.getnOut());

        //actually apply drop out
        if(conf.getDropOut() > 0)
            input.linearView().muli(dropoutMask);

    }

    /**
     * Averages the given logistic regression
     * from a mini batch in to this one
     * @param l the logistic regression to average in to this one
     * @param batchSize  the batch size
     */
    @Override
    public void merge(Layer l,int batchSize) {
        setParams(params().addi(l.params().divi(batchSize)));
    }


    @Override
    public Layer clone() {
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);


        Layer layer = null;
        try {
            Constructor c = getClass().getConstructor(NeuralNetConfiguration.class, INDArray.class, INDArray.class, INDArray.class);
            layer = (Layer) c.newInstance(conf, W.dup(), b.dup(), input != null ? input.dup() : null);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return layer;

    }

    @Override
    public Type type() {
        return Type.FEED_FORWARD;
    }

    /**
     * The number of parameters for the model
     *
     * @return the number of parameters for the model
     */
    @Override
    public int numParams() {
        int ret = 0;
        for(INDArray val : params.values())
            ret += val.length();
        return ret;
    }

    @Override
    public void fit(INDArray input) {
        if(input != null)
            this.input = input;
        Solver solver = new Solver.Builder()
                .model(this).configure(conf()).listeners(getIterationListeners())
                .build();
        this.optimizer = solver.getOptimizer();
        solver.optimize();
    }


    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(),score());
    }

    @Override
    public double score() {
        return score;
    }

    @Override
    public INDArray input() {
        return input;
    }

    @Override
    public void validateInput() {

    }

    /**
     * Create a gradient list based on the passed in parameters.
     * Will throw an IllegalArgumentException if the number of gradient matrices
     * isn't equal to the number of keys in the parameter list
     * @param gradients the gradients to create from
     * @return the create based on the passed in ndarrays
     */
    protected Gradient createGradient(INDArray...gradients) {
        Gradient ret = new DefaultGradient();
        if(gradients.length != conf.variables().size())
            throw new IllegalArgumentException("Unable to create gradients...not equal to number of parameters");
        for(int i = 0; i < gradients.length; i++) {
            INDArray paramI = getParam(conf.variables().get(i));
            if(!Arrays.equals(paramI.shape(),gradients[i].shape()))
                throw new IllegalArgumentException("Gradient at index " + i + " had wrong gradient size of " + Arrays.toString(gradients[i].shape()) + " when should have been " + Arrays.toString(paramI.shape()));
            ret.gradientForVariable().put(conf.variables().get(i),gradients[i]);
        }
        return ret;
    }

    @Override
    public Layer transpose() {
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);


        Layer layer = null;
        try {
            Constructor c = getClass().getConstructor(NeuralNetConfiguration.class, INDArray.class, INDArray.class, INDArray.class);
            NeuralNetConfiguration clone = conf.clone();
            int nIn = clone.getnOut(),nOut = clone.getnIn();
            clone.setnIn(nIn);
            clone.setnOut(nOut);
            layer = (Layer) c.newInstance(conf, W.transpose().dup(), b.transpose().dup(), input != null ? input.transpose().dup() : null);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return layer;
    }

    @Override
    public Pair<Gradient, Gradient> backWard(Gradient ixes, Gradient deltas, INDArray activation,String previousActivation) {
        //figure out how to set ixes and deltas
        INDArray delta = activation.transpose().mmul(ixes.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
        INDArray biasDelta = delta.mean(0);
        INDArray weightsPlusBias = getParam(DefaultParamInitializer.WEIGHT_KEY).transpose();
        Gradient ret = new DefaultGradient();
        INDArray errorForEach = ixes.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray nextIx =  errorForEach.mmul(weightsPlusBias).muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(previousActivation, activation).derivative()));
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, nextIx);
        INDArray deltaColumnSums = nextIx.isVector() ? nextIx.dup() : nextIx.mean(0);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,deltaColumnSums);
        Gradient weightDelta = new DefaultGradient();
        weightDelta.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,delta);
        weightDelta.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasDelta);
        return new Pair<>(ret,weightDelta);

    }

    @Override
    public void accumulateScore(double accum) {
        score += accum;
    }


}
