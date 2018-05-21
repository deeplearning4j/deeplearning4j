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

import lombok.AccessLevel;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.Setter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.*;

/**
 * A layer with input and output, no parameters or gradients
 */
@Data
@NoArgsConstructor
public abstract class AbstractLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.Layer> implements Layer {

    @Setter(AccessLevel.NONE)
    protected INDArray input;
    protected INDArray preOutput;
    protected NeuralNetConfiguration conf;
    protected INDArray dropoutMask;
    protected boolean dropoutApplied = false;
    protected Collection<TrainingListener> trainingListeners = new ArrayList<>();
    protected int index = 0;
    protected INDArray maskArray;
    protected MaskState maskState;
    protected CacheMode cacheMode = CacheMode.NONE;

    protected int iterationCount;
    protected int epochCount;

    public AbstractLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
        cacheMode = conf.getCacheMode();
    }

    public AbstractLayer(NeuralNetConfiguration conf, INDArray input) {
        this(conf);
        this.input = input;
    }

    @Override
    public void setCacheMode(CacheMode mode) {
        if (mode == null)
            mode = CacheMode.NONE;

        this.cacheMode = mode;
    }

    protected LayerConfT layerConf() {
        return (LayerConfT) this.conf.getLayer();
    }

    protected String layerId() {
        String name = this.conf().getLayer().getLayerName();
        return "(layer name: " + (name == null ? "\"\"" : name) + ", layer index: " + index + ", layer type: " +
                getClass().getSimpleName() + ")";
    }

    public INDArray getInput() {
        return input;
    }

    /**
     * Init the model
     */
    @Override
    public void init() {

    }

    @Override
    public abstract Layer clone();

    @Override
    public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        this.input = workspaceMgr.leverageTo(ArrayType.INPUT, input);
        dropoutApplied = false;
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
    public Collection<TrainingListener> getListeners() {
        return trainingListeners;
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
        this.trainingListeners = listeners != null ? listeners : new ArrayList<TrainingListener>();
    }

    /**
     * This method ADDS additional TrainingListener to existing listeners
     *
     * @param listeners
     */
    @Override
    public void addListeners(TrainingListener... listeners) {
        if (this.trainingListeners == null) {
            setListeners(listeners);
            return;
        }

        Collections.addAll(trainingListeners, listeners);
    }

    @Override
    public void setListeners(TrainingListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void update(Gradient gradient) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException();
    }


    @Override
    public ConvexOptimizer getOptimizer() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        this.conf = conf;
    }

    /**Returns the parameters of the neural network as a flattened row vector
     * @return the parameters of the neural network
     */
    @Override
    public INDArray params() {
        return null;
    }

    @Override
    public INDArray getParam(String param) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParam(String key, INDArray val) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParams(INDArray params) {
        if (params != null) {
            throw new UnsupportedOperationException("Not supported");
        }
    }

    protected void setParams(INDArray params, char order) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        if (params != null) {
            throw new UnsupportedOperationException("Not supported");
        }
    }

    @Override
    public INDArray getGradientsViewArray() {
        return null;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        if (gradients != null) {
            throw new UnsupportedOperationException("Not supported");
        }
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        if (paramTable != null && !paramTable.isEmpty()) {
            throw new UnsupportedOperationException("Not supported");
        }
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
        return Collections.emptyMap();
    }

    protected void applyMask(INDArray to) {
        to.muliColumnVector(maskArray);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        setInput(input, workspaceMgr);
        return activate(training, workspaceMgr);
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        return 0.0;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        return 0.0;
    }

    @Override
    public int batchSize() {
        return (int) input.size(0);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }


    @Override
    public void clear() {
        input = null;
        maskArray = null;
        maskState = null;
    }

    protected void applyDropOutIfNecessary(boolean training, LayerWorkspaceMgr workspaceMgr){
        if(training && !dropoutApplied && layerConf().getIDropout() != null ){
            input = layerConf().getIDropout().applyDropout(workspaceMgr.dup(ArrayType.INPUT, input, input.ordering()),
                    getIterationCount(), getEpochCount(), true);
            dropoutApplied = true;
        }
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
        return 0;
    }

    @Override
    public int numParams(boolean backwards) {
        return numParams();
    }

    @Override
    public void fit(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("Not supported");
    }


    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public INDArray input() {
        return input;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void setInputMiniBatchSize(int size) {}

    @Override
    public int getInputMiniBatchSize() {
        // FIXME: int cast
        return (int) input.size(0);
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        this.maskArray = maskArray;
    }

    @Override
    public INDArray getMaskArray() {
        return maskArray;
    }


    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        //Most layers: CNN, dense, activation, etc - set mask array, mask state and then leave the mask unmodified

        this.maskArray = maskArray;
        this.maskState = currentMaskState;

        return new Pair<>(maskArray, currentMaskState);
    }


    @Override
    public Gradient gradient() {
        throw new UnsupportedOperationException(
                        "Not supported for this layer, or should be overridden for layers requiring it");
    }

    @Override
    public void fit() {
        throw new UnsupportedOperationException(
                        "Not supported for this layer, or should be overridden for layers requiring it");
    }

    @Override
    public double score() {
        throw new UnsupportedOperationException(
                        "Not supported for this layer, or should be overridden for layers requiring it");
    }

    @Override
    public void accumulateScore(double accum) {
        throw new UnsupportedOperationException(
                        "Not supported for this layer, or should be overridden for layers requiring it");
    }


    @Override
    public void applyConstraints(int iteration, int epoch){
        if(layerConf().getConstraints() != null){
            for(LayerConstraint lc : layerConf().getConstraints()){
                lc.applyConstraint(this, iteration, epoch);
            }
        }
    }

    public void assertInputSet(boolean backprop){
        if(input == null){
            if(backprop){
                throw new IllegalStateException("Cannot perform backprop in layer " + getClass().getSimpleName()
                        + ": layer input field is not set");
            } else {
                throw new IllegalStateException("Cannot perform forward pass in layer " + getClass().getSimpleName()
                        + ": layer input field is not set");
            }
        }
    }
}
