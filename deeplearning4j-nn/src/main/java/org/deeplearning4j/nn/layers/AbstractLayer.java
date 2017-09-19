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

import lombok.*;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.activations.ActivationsFactory;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.layers.LayerConstraint;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
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

    @Getter(AccessLevel.PROTECTED)
    protected Activations input;
    @Getter(AccessLevel.NONE) @Setter(AccessLevel.NONE)
    protected INDArray preOutput;
    protected NeuralNetConfiguration conf;
    protected boolean preprocessorApplied = false;
    protected boolean dropoutApplied = false;
    @Getter @Setter
    protected int index = 0;
    protected int numInputs = 1;
    protected int numOutput = 1;
//    protected INDArray maskArray;
//    protected MaskState maskState;
    protected CacheMode cacheMode = CacheMode.NONE;

    protected int iterationCount;
    protected int epochCount;

    public AbstractLayer(NeuralNetConfiguration conf) {
        this.conf = conf;
        cacheMode = conf.getCacheMode();
    }

    @Override
    public int numInputs() {
        return numInputs;
    }

    @Override
    public int numOutputs() {
        return numOutput;
    }

    @Override
    public String getName(){
        return conf.getLayer().getLayerName();
    }

    @Override
    public void setInputs(INDArray... inputs){
        if(inputs == null) {
            input = null;
            return;
        }

        if(inputs.length != numInputs()){
            throw new IllegalArgumentException("Cannot set inputs: length " + numInputs() + " expected, got " + inputs.length);
        }
        if(input == null){
            input = ActivationsFactory.getInstance().create(inputs, null, null);
        } else {
            for (int i = 0; i < inputs.length; i++) {
                input.set(i, inputs[i]);
            }
        }
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

    public void setInput(INDArray input){
        setInput(0, input);
    }

    protected void setInput(Activations input){
        this.input = input;
    }

    @Override
    public void setInput(int inputNumber, INDArray input){
        if(inputNumber >= numInputs())
            throw new IllegalArgumentException("Index must be in range 0 to " + (numInputs()-1) + ": got " + inputNumber);
        if(this.input == null){
            this.input = ActivationsFactory.getInstance().create(numInputs());
            this.input.set(inputNumber, input);
        } else {
            this.input.set(inputNumber, input);
        }
        dropoutApplied = false;
        preprocessorApplied = false;
    }

    @Override
    public INDArray getInput(int inputNumber){
        if(inputNumber >= numInputs())
            throw new IllegalArgumentException("Index must be in range 0 to " + (numInputs()-1) + ": got " + inputNumber);
        if(input == null){
            return null;
        }
        return this.input.get(inputNumber);
    }

    @Override
    public void update(Gradient gradient) {
        throw new UnsupportedOperationException();
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
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return Collections.emptyMap();
    }

    protected void applyMask(INDArray to) {
        to.muliColumnVector(input.getMask(0));
    }

    @Override
    public Activations activate(Activations input, boolean training) {
        setInput(0, input.get(0));
        return activate(training);
    }

    @Override
    public Activations activate(Activations input){
        return activate(input, false);
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
    public NeuralNetConfiguration conf() {
        return conf;
    }


    @Override
    public void clear() {
        if(input != null){
            input.clear();
        }
        dropoutApplied = false;
        preprocessorApplied = false;
        preOutput = null;
    }

    protected void applyDropOutIfNecessary(boolean training){//} int iteration, int epoch) {
        if(training && !dropoutApplied && layerConf().getIDropout() != null ){
            //TODO: Epoch + iteration counters...
            if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspaceExternal)) {
                try (MemoryWorkspace ws = Nd4j.getWorkspaceManager()
                        .getWorkspaceForCurrentThread(ComputationGraph.workspaceExternal)
                        .notifyScopeBorrowed()) {
                    INDArray postDropout = layerConf().getIDropout().applyDropout(input.get(0), getIterationCount(), getEpochCount(), false);
                    input.set(0, postDropout);
                }
            } else {
                INDArray postDropout = layerConf().getIDropout().applyDropout(input.get(0), getIterationCount(), getEpochCount(), false);
                input.set(0, postDropout);
            }
            dropoutApplied = true;
        }
    }

    protected void applyPreprocessorIfNecessary(boolean training){
        if(!preprocessorApplied && layerConf().getPreProcessor() != null){
            input = layerConf().getPreProcessor().preProcess(input, input.get(0).size(0), training);        //TODO MINIBATCH SIZE
            preprocessorApplied = true;
        }
    }

    protected Gradients backpropPreprocessor(Gradients gradients){
        if(layerConf().getPreProcessor() != null){
            return layerConf().getPreProcessor().backprop(gradients, input.get(0).size(0));       //TODO MINIBATCH SIZE
        }
        return gradients;
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
    public INDArray input() {
        if(input == null){
            return null;
        }
        return input.get(0);
    }

    @Override
    public void setInputMiniBatchSize(int size) {}

    @Override
    public int getInputMiniBatchSize() {
        if(input == null || input.get(0) == null)
            return -1;
        return input.get(0).size(0);
    }

    @Override
    public void setMaskArray(int idx, INDArray maskArray) {
        if(idx != 0)
            throw new IllegalStateException("Index must be 0");

        if(input == null){
            this.input = ActivationsFactory.getInstance().create(null, maskArray, null);
        } else {
            this.input.setMask(0, maskArray);
        }
    }

    @Override
    public INDArray getMaskArray(int idx) {
        if(idx != 0)
            throw new IllegalStateException("Index must be 0");
        if(input == null){
            return null;
        }
        return input.getMask(0);
    }


//    @Override
//    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
//                    int minibatchSize) {
//        //Most layers: CNN, dense, activation, etc - set mask array, mask state and then leave the mask unmodified
//
//        this.maskArray = maskArray;
//        this.maskState = currentMaskState;
//
//        return new Pair<>(maskArray, currentMaskState);
//    }


    @Override
    public Gradient gradient() {
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


    @Override
    public InputPreProcessor getPreProcessor() {
        if(conf != null){
            return conf.getLayer().getPreProcessor();
        }
        return null;
    }
}
