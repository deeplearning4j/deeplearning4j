/*-
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

import lombok.val;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.params.PretrainParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.broadcast.BroadcastMulOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.primitives.Pair;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 * A layer with parameters
 * @author Adam Gibson
 */
public abstract class BaseLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.BaseLayer>
        extends AbstractLayer<LayerConfT> {

    protected INDArray paramsFlattened;
    protected INDArray gradientsFlattened;
    protected Map<String, INDArray> params;
    protected transient Map<String, INDArray> gradientViews;
    protected double score = 0.0;
    protected ConvexOptimizer optimizer;
    protected Gradient gradient;
    protected Solver solver;

    protected Map<String,INDArray> weightNoiseParams = new HashMap<>();

    public BaseLayer(NeuralNetConfiguration conf) {
        super(conf);
    }

    public BaseLayer(NeuralNetConfiguration conf, INDArray input) {
        this(conf);
        this.input = input;
    }

    public LayerConfT layerConf() {
        return (LayerConfT) this.conf.getLayer();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        INDArray z = preOutput(true, workspaceMgr); //Note: using preOutput(INDArray) can't be used as this does a setInput(input) and resets the 'appliedDropout' flag
        //INDArray activationDerivative = Nd4j.getExecutioner().execAndReturn(Nd4j.getOpFactory().createTransform(conf().getLayer().getActivationFunction(), z).derivative());
        //        INDArray activationDerivative = conf().getLayer().getActivationFn().getGradient(z);
        //        INDArray delta = epsilon.muli(activationDerivative);
        INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst(); //TODO handle activation function params

        if (maskArray != null) {
            applyMask(delta);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY); //f order
        Nd4j.gemm(input, delta, weightGrad, true, false, 1.0, 0.0);

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);

        if(hasBias()){
            INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
            delta.sum(biasGrad, 0); //biasGrad is initialized/zeroed first
            ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);
        }

        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        INDArray epsilonNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, new long[]{W.size(0), delta.size(0)}, 'f');
        epsilonNext = W.mmuli(delta.transpose(),epsilonNext).transpose();   //W.mmul(delta.transpose()).transpose();

        weightNoiseParams.clear();

        epsilonNext = backpropDropOutIfPresent(epsilonNext);
        return new Pair<>(ret, epsilonNext);
    }

    public void fit() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        if (this.input == null)
            return;

        INDArray output = activate(true, workspaceMgr);
        setScoreWithZ(output);
    }


    protected void setScoreWithZ(INDArray z) {}

    /**
     * Objective function:  the specified objective
     * @return the score for the objective
     */

    @Override
    public double score() {
        return score;
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public void update(Gradient gradient) {
        for (String paramType : gradient.gradientForVariable().keySet()) {
            update(gradient.getGradientFor(paramType), paramType);
        }
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        setParam(paramType, getParam(paramType).addi(gradient));
    }


    @Override
    public ConvexOptimizer getOptimizer() {
        if (optimizer == null) {
            Solver solver = new Solver.Builder().model(this).configure(conf()).build();
            this.optimizer = solver.getOptimizer();
        }
        return optimizer;
    }

    /**Returns the parameters of the neural network as a flattened row vector
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        return paramsFlattened;
    }

    @Override
    public INDArray getParam(String param) {
        return params.get(param);
    }

    @Override
    public void setParam(String key, INDArray val) {
        if (params.containsKey(key))
            params.get(key).assign(val);
        else
            params.put(key, val);
    }

    @Override
    public void setParams(INDArray params) {
        if (params == paramsFlattened)
            return; //no op
        setParams(params, 'f');
    }

    protected void setParams(INDArray params, char order) {
        List<String> parameterList = conf.variables();
        int length = 0;
        for (String s : parameterList)
            length += getParam(s).length();
        if (params.length() != length)
            throw new IllegalArgumentException("Unable to set parameters: must be of length " + length
                    + ", got params of length " + params.length() + " - " + layerId());
        int idx = 0;
        Set<String> paramKeySet = this.params.keySet();
        for (String s : paramKeySet) {
            INDArray param = getParam(s);
            INDArray get = params.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, idx + param.length()));
            if (param.length() != get.length())
                throw new IllegalStateException("Parameter " + s + " should have been of length " + param.length()
                        + " but was " + get.length() + " - " + layerId());
            param.assign(get.reshape(order, param.shape())); //Use assign due to backprop params being a view of a larger array
            idx += param.length();
        }
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        if (this.params != null && params.length() != numParams())
            throw new IllegalArgumentException("Invalid input: expect params of length " + numParams()
                    + ", got params of length " + params.length() + " - " + layerId());

        this.paramsFlattened = params;
    }

    @Override
    public INDArray getGradientsViewArray() {
        return gradientsFlattened;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if (this.params != null && gradients.length() != numParams())
            throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams(true)
                    + ", got array of length " + gradients.length() + " - " + layerId());

        this.gradientsFlattened = gradients;
        this.gradientViews = conf.getLayer().initializer().getGradientsFromFlattened(conf, gradients);
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        this.params = paramTable;
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException("Deprecated - no longer used - " + layerId());
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return params;
    }

    /**
     * Get the parameter, after applying any weight noise (such as DropConnect) if necessary.
     * Note that during training, this will store the post-noise parameters, as these should be used
     * for both forward pass and backprop, for a single iteration.
     * Consequently, the parameters (post noise) should be cleared after each training iteration
     *
     * @param param    Parameter key
     * @param training If true: during training
     * @return The parameter, after applying any noise
     */
    protected INDArray getParamWithNoise(String param, boolean training, LayerWorkspaceMgr workspaceMgr){
        INDArray p;
        if(layerConf().getWeightNoise() != null){
            if(training && weightNoiseParams.size() > 0 && weightNoiseParams.containsKey(param) ){
                //Re-use these weights for both forward pass and backprop - don't want to use 2 different params here
                //These should be cleared during  backprop
                return weightNoiseParams.get(param);
            } else {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    p = layerConf().getWeightNoise().getParameter(this, param, getIterationCount(), getEpochCount(), training, workspaceMgr);
                }
            }

            if(training){
                //Store for re-use in backprop
                weightNoiseParams.put(param, p);
            }
        } else {
            return getParam(param);
        }

        return p;
    }

    protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(false);
        applyDropOutIfNecessary(training, workspaceMgr);
        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training, workspaceMgr);

        //Input validation:
        if (input.rank() != 2 || input.columns() != W.rows()) {
            if (input.rank() != 2) {
                throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank "
                        + input.rank() + " array with shape " + Arrays.toString(input.shape())
                        + ". Missing preprocessor or wrong input type? " + layerId());
            }
            throw new DL4JInvalidInputException(
                    "Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape())
                            + ") is invalid: does not match layer input size (layer # inputs = "
                            + W.size(0) + ") " + layerId());
        }


        INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, input.size(0), W.size(1));
        input.mmuli(W, ret);
        if(hasBias()){
            ret.addiRowVector(b);
        }

        if (maskArray != null) {
            applyMask(ret);
        }

        return ret;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray z = preOutput(training, workspaceMgr);
        INDArray ret = layerConf().getActivationFn().getActivation(z, training);

        if (maskArray != null) {
            applyMask(ret);
        }

        return ret;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        double l2Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l2 = conf.getL2ByParam(entry.getKey());
            if (l2 > 0) {
                double norm2 = getParam(entry.getKey()).norm2Number().doubleValue();
                l2Sum += 0.5 * l2 * norm2 * norm2;
            }
        }

        return l2Sum;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        double l1Sum = 0.0;
        for (Map.Entry<String, INDArray> entry : paramTable().entrySet()) {
            double l1 = conf.getL1ByParam(entry.getKey());
            if (l1 > 0) {
                double norm1 = getParam(entry.getKey()).norm1Number().doubleValue();
                l1Sum += l1 * norm1;
            }
        }

        return l1Sum;
    }

    @Override
    public Layer clone() {
        Layer layer = null;
        try {
            Constructor c = getClass().getConstructor(NeuralNetConfiguration.class);
            layer = (Layer) c.newInstance(conf);
            Map<String, INDArray> linkedTable = new LinkedHashMap<>();
            for (Map.Entry<String, INDArray> entry : params.entrySet()) {
                linkedTable.put(entry.getKey(), entry.getValue().dup());
            }
            layer.setParamTable(linkedTable);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return layer;

    }

    /**
     * The number of parameters for the model
     *
     * @return the number of parameters for the model
     */
    @Override
    public int numParams() {
        int ret = 0;
        for (INDArray val : params.values())
            ret += val.length();
        return ret;
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        if (input != null) {
            setInput(input, workspaceMgr);
            applyDropOutIfNecessary(true, workspaceMgr);
        }
        if (solver == null) {
            solver = new Solver.Builder().model(this).configure(conf()).listeners(getListeners()).build();
        }
        this.optimizer = solver.getOptimizer();
        solver.optimize(workspaceMgr);
    }

    @Override
    public String toString() {
        return getClass().getName() + "{" + "conf=" + conf + ", score=" + score
                + ", optimizer=" + optimizer + ", listeners=" + trainingListeners + '}';
    }

    @Override
    public Layer transpose() {
        if (!(conf.getLayer() instanceof org.deeplearning4j.nn.conf.layers.FeedForwardLayer))
            throw new UnsupportedOperationException(
                    "Unsupported layer type: " + conf.getLayer().getClass().getName() + " - " + layerId());

        INDArray w = getParam(DefaultParamInitializer.WEIGHT_KEY);
        INDArray b = getParam(DefaultParamInitializer.BIAS_KEY);
        INDArray vb = getParam(PretrainParamInitializer.VISIBLE_BIAS_KEY);
        Layer layer;
        try {
            NeuralNetConfiguration clone = conf.clone(); // assume a deep clone here

            org.deeplearning4j.nn.conf.layers.FeedForwardLayer clonedLayerConf =
                    (org.deeplearning4j.nn.conf.layers.FeedForwardLayer) clone.getLayer();
            val nIn = clonedLayerConf.getNOut();
            val nOut = clonedLayerConf.getNIn();
            clonedLayerConf.setNIn(nIn);
            clonedLayerConf.setNOut(nOut);

            //Need to swap the hidden and visible biases for pretrain layers
            INDArray newB;
            INDArray newVB = null;

            long totalParams = w.length();
            if (vb != null) {
                newB = vb.dup();
                newVB = b.dup();
                totalParams += newB.length() + newVB.length();
            } else {
                newB = Nd4j.create(1, nOut);
                totalParams += newB.length();
            }

            INDArray paramsView = Nd4j.create(1, totalParams);
            layer = clone.getLayer().instantiate(clone, trainingListeners, this.index, paramsView, true);

            layer.setParam(DefaultParamInitializer.WEIGHT_KEY, w.transpose().dup());
            layer.setParam(DefaultParamInitializer.BIAS_KEY, newB);
            if (vb != null)
                layer.setParam(PretrainParamInitializer.VISIBLE_BIAS_KEY, newVB);
        } catch (Exception e) {
            throw new RuntimeException("Unable to construct transposed layer: " + layerId(), e);
        }

        return layer;
    }

    @Override
    public void accumulateScore(double accum) {
        score += accum;
    }

    @Override
    public void clear(){
        super.clear();
        weightNoiseParams.clear();
    }

    @Override
    public void clearNoiseWeightParams(){
        weightNoiseParams.clear();;
    }

    /**
     * Does this layer have no bias term? Many layers (dense, convolutional, output, embedding) have biases by
     * default, but no-bias versions are possible via configuration
     *
     * @return True if a bias term is present, false otherwise
     */
    public boolean hasBias(){
        //Overridden by layers supporting no bias mode: dense, output, convolutional, embedding
        return true;
    }
}
