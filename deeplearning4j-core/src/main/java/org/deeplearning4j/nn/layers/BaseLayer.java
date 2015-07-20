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
import org.nd4j.linalg.api.ops.LossFunction;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.lang.reflect.Constructor;
import java.util.*;

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
    protected int index=0;

    public BaseLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    public BaseLayer(NeuralNetConfiguration conf, INDArray input) {
        this.input = input;
        this.conf = conf;
    }

    public INDArray getInput() {
        return input;
    }

    public void setInput(INDArray input) {
        this.input = input;
    }

    @Override
    public int getIndex() {
        return index;
    }

    @Override
    public void setIndex(int index) {
        this.index = index;
    }


    @Override
    public Collection<IterationListener> getIterationListeners() {
        return iterationListeners;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }

    @Override
    public Gradient error(INDArray errorSignal) {
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        Gradient nextLayerGradient = new DefaultGradient();
        INDArray wErrorSignal = errorSignal.mmul(W.transpose());
        nextLayerGradient.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,wErrorSignal);
        return nextLayerGradient;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        INDArray deriv = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getActivationFunction(), input).derivative());
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
        INDArray weightError = weightErrorSignal.transpose().mmul(activation).transpose();
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY,weightError);
        INDArray biasGradient = weightError.mean(0);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY,biasGradient);

        return ret;
    }

    @Override
    public Pair<Gradient,INDArray> backwardGradient(INDArray derivative, INDArray epsilon, INDArray activation) {
        //needs to be number of features by examples
        Gradient ret = new DefaultGradient();
        //If this layer is layer L, then
        //Epsilon is (w^(L+1)*(d^(L+1))^T).
        INDArray delta = epsilon.muli(derivative);

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, delta.transpose().mmul(activation).transpose());
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, delta.sum(0));
        
        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(delta.transpose()).transpose();
        
        return new Pair<>(ret,epsilonNext);
    }

    public void fit() {
        fit(this.input);
    }

    @Override
    public void setScore() {
        if (this.input == null)
            return;

        INDArray output = transform(input);
        if (conf.getLossFunction() == LossFunctions.LossFunction.CUSTOM) {
            LossFunction create = Nd4j.getOpFactory().createLossFunction(conf.getCustomLossFunction(), input, output);
            create.exec();
            score = create.currentResult().doubleValue();
        } else{
            score = LossFunctions.score(
                    input,
                    conf.getLossFunction(),
                    output,
                    conf.getL2(),
                    conf.getL1()
                    ,l1Magnitude()
                    ,l2Magnitude(),
                    conf.isUseRegularization());
        }

    }


    /**
     * Objective function:  the specified objective
     * @return the score for the objective
     */

    @Override
    public double score() {
        return score;
    }

    /**
     * iterate one iteration of the network
     *
     * @param input  the input to iterate on
     */
    @Override
    public void iterate(INDArray input) {
        this.input = input.dup();
        applyDropOutIfNecessary(this.input);
        Gradient gradient = gradient();
        for(String paramType : gradient.gradientForVariable().keySet()) {
            update(gradient.getGradientFor(paramType), paramType);
        }
    }


    @Override
    public void update(INDArray gradient, String paramType) {
		setParam(paramType, getParam(paramType).addi(gradient));
    }


    @Override
    public ConvexOptimizer getOptimizer() {
        if(optimizer == null) {
            Solver solver = new Solver.Builder()
                    .model(this).configure(conf())
                    .build();
            this.optimizer = solver.getOptimizer();
        }
        return optimizer;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    /**
     * Returns the parameters of the neural network
     *
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        int length = 0;
        for(String s : params.keySet()) {
            length += params.get(s).length();
        }

        INDArray ret = Nd4j.create(1, length);
        int count = 0;
        for(String s : params.keySet()) {
            INDArray get = params.get(s).linearView();
            for(int i = 0; i < get.length(); i++) {
                ret.putScalar(count++,get.getDouble(i));
            }
        }
        return ret;
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void setParam(String key, INDArray val) {
        params.put(key, val);
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
        Set<String> paramKeySet = this.params.keySet();
        for(String s : paramKeySet ){
        	INDArray param = getParam(s);
            INDArray get = params.get(NDArrayIndex.interval(idx,idx + param.length()));
            if(param.length() != get.length())
                throw new IllegalStateException("Parameter " + s + " should have been of length " + param.length() + " but was " + get.length());
            param.linearView().assign(get);
            idx += param.length();
        }

    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public void initParams() {
        paramInitializer.init(paramTable(), conf());
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return params;
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

        this.input = x.dup();
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect()) {
            if (conf.getDropOut() > 0) {
                W = W.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(W.shape()).divi(conf.getDropOut()));
            }
        }
        INDArray ret = input().mmul(W).addiRowVector(b);
        return ret;
    }

    @Override
    public double l2Magnitude() {
        return Transforms.pow(getParam(DefaultParamInitializer.WEIGHT_KEY),2).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public double l1Magnitude() {
        return Transforms.abs(getParam(DefaultParamInitializer.WEIGHT_KEY)).sum(Integer.MAX_VALUE).getDouble(0);
    }

    @Override
    public int batchSize() {
        return input.rows();
    }

    @Override
    public  INDArray activate() {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect()) {
            if (conf.getDropOut() > 0) {
                W = W.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(W.shape()).divi(conf.getDropOut()));
            }
        }
        INDArray ret = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf.getActivationFunction(), input().mmul(W).addiRowVector(b)));
        return ret;
    }

    @Override
    public  INDArray activate(INDArray input) {
        this.input = input.dup();
        return activate();
    }


    @Override
    public INDArray activationMean() {
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        if(conf.isUseDropConnect()) {
            if (conf.getDropOut() > 0) {
                W = W.mul(Nd4j.getDistributions().createBinomial(1,conf.getDropOut()).sample(W.shape()).divi(conf.getDropOut()));
            }
        }
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
        if(conf.getDropOut() > 0 && !conf.isUseDropConnect()) {
            input.muli(dropoutMask);
        }
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
        setScore();
    }


    @Override
    public Layer clone() {


        Layer layer = null;
        try {
            Constructor c = getClass().getConstructor(NeuralNetConfiguration.class);
            layer = (Layer) c.newInstance(conf);
            Map<String,INDArray> linkedTable = new LinkedHashMap<>();
            for(String s: params.keySet())
                linkedTable.put(s,params.get(s).dup());
            layer.setParamTable(linkedTable);
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
        if(input != null) {
            this.input = input.dup();
            applyDropOutIfNecessary(this.input);
        }
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
    public String toString() {
        return getClass().getName() + "{" +
                "conf=" + conf +
                ", input=" + input +
                ", params=" + params +
                ", dropoutMask=" + dropoutMask +
                ", paramInitializer=" + paramInitializer +
                ", score=" + score +
                ", optimizer=" + optimizer +
                ", iterationListeners=" + iterationListeners +
                '}';
    }

    @Override
    public Layer transpose() {
        INDArray W = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);


        Layer layer = null;
        try {
            Constructor c = getClass().getConstructor(NeuralNetConfiguration.class, INDArray.class, INDArray.class, INDArray.class);
            NeuralNetConfiguration clone = conf.clone();
            int nIn = clone.getNOut(),nOut = clone.getNIn();
            clone.setNIn(nIn);
            clone.setNOut(nOut);
            layer = (Layer) c.newInstance(conf, W.transpose(), b.transpose(), input != null ? input.transpose() : null);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return layer;
    }

    @Deprecated
    @Override
    public Pair<Gradient, Gradient> backWard(Gradient ixes, Gradient deltas, INDArray activation,String previousActivation) {
        //figure out how to set ixes and deltas
        INDArray delta = activation.transpose().mmul(ixes.getGradientFor(DefaultParamInitializer.WEIGHT_KEY));
        INDArray biasDelta = delta.mean(0);
        INDArray weights = getParam(DefaultParamInitializer.WEIGHT_KEY).transpose();
        Gradient ret = new DefaultGradient();
        INDArray errorForEach = ixes.getGradientFor(DefaultParamInitializer.WEIGHT_KEY);
        INDArray nextIx =  errorForEach.mmul(weights).muli(Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(previousActivation, activation).derivative()));
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
