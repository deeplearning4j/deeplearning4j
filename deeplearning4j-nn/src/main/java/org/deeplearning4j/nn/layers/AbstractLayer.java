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

import lombok.Data;
import lombok.NoArgsConstructor;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.nd4j.linalg.primitives.Pair;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.util.Dropout;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

/**
 * A layer with input and output, no parameters or gradients
 */
@Data
@NoArgsConstructor
public abstract class AbstractLayer<LayerConfT extends org.deeplearning4j.nn.conf.layers.Layer> implements Layer {

    protected INDArray input, preOutput;
    protected NeuralNetConfiguration conf;
    protected INDArray dropoutMask;
    protected boolean dropoutApplied = false;
    protected Collection<IterationListener> iterationListeners = new ArrayList<>();
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
        return "(layer name: " + (name == null ? "\"\"" : name) + ", layer index: " + index + ")";
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
    public void setInput(INDArray input) {
        this.input = input;
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
    public Collection<IterationListener> getListeners() {
        return iterationListeners;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.iterationListeners = listeners != null ? listeners : new ArrayList<IterationListener>();
    }

    /**
     * This method ADDS additional IterationListener to existing listeners
     *
     * @param listeners
     */
    @Override
    public void addListeners(IterationListener... listeners) {
        if (this.iterationListeners == null) {
            setListeners(listeners);
            return;
        }

        for (IterationListener listener : listeners)
            iterationListeners.add(listener);
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        setListeners(Arrays.asList(listeners));
    }

    @Override
    public Gradient error(INDArray errorSignal) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    @Deprecated
    public INDArray derivativeActivation(INDArray input) {
        throw new UnsupportedOperationException("Deprecated - " + layerId());
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public void computeGradientAndScore() {
        throw new UnsupportedOperationException("Not supported");
    }


    @Override
    public INDArray preOutput(INDArray x, TrainingMode training) {
        return preOutput(x, training == TrainingMode.TRAIN);
    }

    @Override
    public INDArray activate(TrainingMode training) {
        return activate(training == TrainingMode.TRAIN);
    }

    @Override
    public INDArray activate(INDArray input, TrainingMode training) {
        return activate(input, training == TrainingMode.TRAIN);
    }

    /**
     * iterate one iteration of the network
     *
     * @param input  the input to iterate on
     */
    @Override
    public void iterate(INDArray input) {
        throw new UnsupportedOperationException();
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

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        if (x == null) {
            throw new IllegalArgumentException("Cannot do forward pass with null input " + layerId());
        }
        setInput(x);
        return preOutput(training);
    }


    public abstract INDArray preOutput(boolean training);

    protected void applyMask(INDArray to) {
        to.muliColumnVector(maskArray);
    }

    @Override
    public INDArray activate(INDArray input) {
        setInput(input);
        return activate(true);
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        setInput(input);
        return activate(training);
    }

    @Override
    public INDArray activate() {
        return activate(false);
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
    public INDArray preOutput(INDArray x) {
        return preOutput(x, true);
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
        return input.size(0);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return conf;
    }


    @Override
    public void clear() {
        input = null;
    }

    protected void applyDropOutIfNecessary(boolean training){//} int iteration, int epoch) {
        if(training && !dropoutApplied && layerConf().getIDropout() != null ){
            //TODO: Epoch + iteration counters...
            if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceExternal)) {
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal)
                        .notifyScopeBorrowed()) {
                    input = layerConf().getIDropout().applyDropout(input, getIterationCount(), getEpochCount(), false);
                }
            } else {
                input = layerConf().getIDropout().applyDropout(input, getIterationCount(), getEpochCount(), false);
            }
            dropoutApplied = true;
        }
    }

    /**
     * Averages the given logistic regression from a mini batch into this layer
     * @param l the logistic regression layer to average into this layer
     * @param batchSize  the batch size
     */
    @Override
    public void merge(Layer l, int batchSize) {
        throw new UnsupportedOperationException();
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
    public void fit(INDArray input) {
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
        return input.size(0);
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
}
