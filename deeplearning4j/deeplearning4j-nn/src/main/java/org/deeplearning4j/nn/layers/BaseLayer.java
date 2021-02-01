/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.nn.layers;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNormBp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.learning.regularization.Regularization;
import org.nd4j.common.primitives.Pair;

import java.lang.reflect.Constructor;
import java.util.*;

/**
 * A layer with parameters
 * @author Adam Gibson
 */
@Slf4j
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

    public BaseLayer(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    public LayerConfT layerConf() {
        return (LayerConfT) this.conf.getLayer();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(true);
        //If this layer is layer L, then epsilon is (w^(L+1)*(d^(L+1))^T) (or equivalent)
        Pair<INDArray, INDArray> zAndPreNorm = preOutputWithPreNorm(true, true, workspaceMgr);
        INDArray z = zAndPreNorm.getFirst(); //Note: using preOutput(INDArray) can't be used as this does a setInput(input) and resets the 'appliedDropout' flag
        INDArray preNorm = zAndPreNorm.getSecond();
        INDArray delta = layerConf().getActivationFn().backprop(z, epsilon).getFirst(); //TODO handle activation function params

        if (maskArray != null) {
            applyMask(delta);
        }

        Gradient ret = new DefaultGradient();

        if(hasBias()){
            INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
            delta.sum(biasGrad, 0); //biasGrad is initialized/zeroed first
            ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);
        }

        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, true, workspaceMgr);

        INDArray epsilonNext = workspaceMgr.createUninitialized(ArrayType.ACTIVATION_GRAD, delta.dataType(), new long[]{W.size(0), delta.size(0)}, 'f');
        if(hasLayerNorm()) {
            INDArray g = getParam(DefaultParamInitializer.GAIN_KEY);

            INDArray dldg = gradientViews.get(DefaultParamInitializer.GAIN_KEY);
            Nd4j.getExecutioner().exec(new LayerNormBp(preNorm, g, delta, delta, dldg, true, 1));
            ret.gradientForVariable().put(DefaultParamInitializer.GAIN_KEY, dldg);

        }

        epsilonNext = W.mmuli(delta.transpose(),epsilonNext).transpose();   //W.mmul(delta.transpose()).transpose();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY); //f order
        Nd4j.gemm(input.castTo(weightGrad.dataType()), delta, weightGrad, true, false, 1.0, 0.0);           //TODO avoid castTo?
        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);

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
        return preOutputWithPreNorm(training, false, workspaceMgr).getFirst();
    }

    protected Pair<INDArray, INDArray> preOutputWithPreNorm(boolean training, boolean forBackprop, LayerWorkspaceMgr workspaceMgr) {
        assertInputSet(forBackprop);
        applyDropOutIfNecessary(training, workspaceMgr);
        INDArray W = getParamWithNoise(DefaultParamInitializer.WEIGHT_KEY, training, workspaceMgr);
        INDArray b = getParamWithNoise(DefaultParamInitializer.BIAS_KEY, training, workspaceMgr);
        INDArray g = (hasLayerNorm() ? getParam(DefaultParamInitializer.GAIN_KEY) : null);

        INDArray input = this.input.castTo(dataType);

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


        INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, W.dataType(), input.size(0), W.size(1));
        input.castTo(ret.dataType()).mmuli(W, ret);     //TODO Can we avoid this cast? (It sohuld be a no op if not required, however)

        INDArray preNorm = ret;
        if(hasLayerNorm()){
            preNorm = (forBackprop ? ret.dup(ret.ordering()) : ret);
            Nd4j.getExecutioner().exec(new LayerNorm(preNorm, g, ret, true, 1));
        }

        if(hasBias()){
            ret.addiRowVector(b);
        }

        if (maskArray != null) {
            applyMask(ret);
        }

        return new Pair<>(ret, preNorm);
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
    public double calcRegularizationScore(boolean backpropParamsOnly){
        double scoreSum = 0.0;
        for (Map.Entry<String, INDArray> e : paramTable().entrySet()) {
            List<Regularization> l = layerConf().getRegularizationByParam(e.getKey());
            if(l == null || l.isEmpty()){
                continue;
            }
            for(Regularization r : l){
                scoreSum += r.score(e.getValue(), getIterationCount(), getEpochCount());
            }
        }
        return scoreSum;
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
            log.error("",e);
        }

        return layer;

    }

    /**
     * The number of parameters for the model
     *
     * @return the number of parameters for the model
     */
    @Override
    public long numParams() {
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
    public void clear(){
        super.clear();
        weightNoiseParams.clear();
    }

    @Override
    public void clearNoiseWeightParams(){
        weightNoiseParams.clear();
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

    /**
     * Does this layer support and is it enabled layer normalization? Only Dense and SimpleRNN Layers support
     * layer normalization.
     *
     * @return True if layer normalization is enabled on this layer, false otherwise
     */
    public boolean hasLayerNorm(){
        // Overridden by layers supporting layer normalization.
        return false;
    }
}
