/*-
 *
 *  * Copyright 2016 Skymind,Inc.
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

package org.deeplearning4j.nn.graph.vertex.impl;

import lombok.Data;
import lombok.EqualsAndHashCode;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.activations.Activations;
import org.deeplearning4j.nn.api.gradients.Gradients;
import org.deeplearning4j.nn.api.gradients.GradientsFactory;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.InputPreProcessor;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.graph.vertex.BaseGraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;

/**
 * LayerVertex is a GraphVertex with a neural network Layer (and, optionally an {@link InputPreProcessor}) in it
 *
 * @author Alex Black
 * @deprecated No longer required - kept for legacy reasons
 */
@Data
@EqualsAndHashCode(callSuper = true)
@Deprecated
public class LayerVertex extends BaseGraphVertex {

    private Layer layer;
    private final InputPreProcessor layerPreProcessor;

    /**
     * Create a network input vertex:
     */
    public LayerVertex(ComputationGraph graph, String name, int vertexIndex, int numInputs, Layer layer,
                    InputPreProcessor layerPreProcessor, boolean outputVertex) {
        super(graph, name, vertexIndex, numInputs);
        this.graph = graph;
        this.vertexName = name;
        this.index = vertexIndex;
        this.layer = layer;
        this.layerPreProcessor = layerPreProcessor;
        this.outputVertex = outputVertex;
    }


    @Override
    public int numInputs(){
        return layer.numInputs();
    }

    @Override
    public int numOutputs(){
        return layer.numOutputs();
    }

    @Override
    public int batchSize(){
        return layer.batchSize();
    }

    @Override
    public INDArray input(){
        return layer.input();
    }


    // ----- Parameter Methods -----
    @Override
    public INDArray params(){
        return layer.params();
    }

    @Override
    public int numParams(){
        return layer.numParams();
    }

    @Override
    public int numParams(boolean backwards){
        return layer.numParams(backwards);
    }

    @Override
    public void setParams(INDArray params){
        layer.setParams(params);
    }

    @Override
    public void setParamsViewArray(INDArray params){
        layer.setParamsViewArray(params);
    }

    @Override
    public INDArray getParam(String param){
        return layer.getParam(param);
    }

    @Override
    public Map<String, INDArray> paramTable(){
        return layer.paramTable();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly){
        return layer.paramTable(backpropParamsOnly);
    }

//    @Override
//    public void setParamTable(Map<String, INDArray> paramTable){
//        layer.setParamTable(paramTable);
//    }

    @Override
    public void setParam(String key, INDArray val){
        layer.setParam(key, val);
    }

    @Override
    public double calcL2(boolean backpropOnlyParams){
        return layer.calcL2(backpropOnlyParams);
    }

    @Override
    public double calcL1(boolean backpropOnlyParams){
        return layer.calcL1(backpropOnlyParams);
    }





    // ----- Forward Pass Methods -----

    @Override
    public Activations activate(boolean training){
        return layer.activate(training);
    }

    @Override
    public Activations activate(Activations input, boolean training){
        return layer.activate(input, training);
    }

    @Override
    public Activations activate(Activations input){
        return layer.activate(input);
    }

    @Override
    public void setInputs(INDArray... inputs){
        layer.setInputs(inputs);
    }

    @Override
    public INDArray getInput(int inputNumber){
        return layer.getInput(inputNumber);
    }

    @Override
    public void setInputMiniBatchSize(int size){
        layer.setInputMiniBatchSize(size);
    }

    @Override
    public int getInputMiniBatchSize(){
        return layer.getInputMiniBatchSize();
    }

    @Override
    public void setMaskArray(int idx, INDArray maskArray){
        layer.setMaskArray(idx, maskArray);
    }

    @Override
    public INDArray getMaskArray(int idx){
        return layer.getMaskArray(idx);
    }



    // ----- Gradient Methods -----

    @Override
    public Gradient gradient(){
        return layer.gradient();
    }


    @Override
    public INDArray getGradientsViewArray(){
        return layer.getGradientsViewArray();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients){
        layer.setBackpropGradientsViewArray(gradients);
    }

    @Override
    public void update(Gradient gradient){
        layer.update(gradient);
    }


    // ----- General Methods -----

    @Override
    public String getName(){
        return layer.getName();
    }

    @Override
    public void clear(){
        layer.clear();
    }


    @Override
    public void applyConstraints(int iteration, int epoch){
        layer.applyConstraints(iteration, epoch);
    }

    @Override
    public NeuralNetConfiguration conf(){
        return layer.conf();
    }

    @Override
    public void setConf(NeuralNetConfiguration conf){
        layer.setConf(conf);
    }

    @Override
    public void setCacheMode(CacheMode mode){
        layer.setCacheMode(mode);
    }

    @Override
    public void setIndex(int index){
        layer.setIndex(index);
    }

    @Override
    public int getIndex(){
        return layer.getIndex();
    }

    @Override
    public int getIterationCount(){
        return layer.getIterationCount();
    }

    @Override
    public int getEpochCount(){
        return layer.getEpochCount();
    }

    @Override
    public void setIterationCount(int iterationCount){
        layer.setIterationCount(iterationCount);
    }

    @Override
    public void setEpochCount(int epochCount){
        layer.setEpochCount(epochCount);
    }

    @Override
    public boolean isPretrainLayer(){
        return layer.isPretrainLayer();
    }


    @Override
    public void clearNoiseWeightParams(){
        layer.clearNoiseWeightParams();
    }

    @Override
    protected Pair<INDArray, MaskState> feedForwardMaskArrays(INDArray[] maskArrays, MaskState currentMaskState, int minibatchSize) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradients backpropGradient(Gradients gradient) {
        return layer.backpropGradient(gradient);
    }
}
