/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.deeplearning4j.nn.layers.wrapper;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.MaskState;
import org.deeplearning4j.nn.api.TrainingConfig;
import org.deeplearning4j.nn.conf.CacheMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.Collection;
import java.util.Map;

/**
 * Abstract wrapper layer. The idea: this class passes through all methods to the underlying layer.
 * Then, subclasses of BaseWrapperLayer can selectively override specific methods, rather than having
 * to implement every single one of the passthrough methods in each subclass.
 *
 * @author Alex Black
 */
@Data
public abstract class BaseWrapperLayer implements Layer {

    protected Layer underlying;

    public BaseWrapperLayer(@NonNull Layer underlying){
        this.underlying = underlying;
    }

    @Override
    public void setCacheMode(CacheMode mode) {
        underlying.setCacheMode(mode);
    }

    @Override
    public double calcL2(boolean backpropOnlyParams) {
        return underlying.calcL2(backpropOnlyParams);
    }

    @Override
    public double calcL1(boolean backpropOnlyParams) {
        return underlying.calcL1(backpropOnlyParams);
    }

    @Override
    public Type type() {
        return underlying.type();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        return underlying.backpropGradient(epsilon, workspaceMgr);
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        return underlying.activate(training, workspaceMgr);
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {
        return underlying.activate(input, training, workspaceMgr);
    }

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException("Not supported");   //If required, implement in subtype (so traspose is wrapped)
    }

    @Override
    public Layer clone() {
        throw new UnsupportedOperationException("Clone not supported");
    }

    @Override
    public Collection<TrainingListener> getListeners() {
        return underlying.getListeners();
    }

    @Override
    public void setListeners(TrainingListener... listeners) {
        underlying.setListeners(listeners);
    }

    @Override
    public void addListeners(TrainingListener... listener) {
        underlying.addListeners(listener);
    }

    @Override
    public void fit() {
        underlying.fit();
    }

    @Override
    public void update(Gradient gradient) {
        underlying.update(gradient);
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        underlying.update(gradient, paramType);
    }

    @Override
    public double score() {
        return underlying.score();
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        underlying.computeGradientAndScore(workspaceMgr);
    }

    @Override
    public void accumulateScore(double accum) {
        underlying.accumulateScore(accum);
    }

    @Override
    public INDArray params() {
        return underlying.params();
    }

    @Override
    public int numParams() {
        return underlying.numParams();
    }

    @Override
    public int numParams(boolean backwards) {
        return underlying.numParams();
    }

    @Override
    public void setParams(INDArray params) {
        underlying.setParams(params);
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        underlying.setParamsViewArray(params);
    }

    @Override
    public INDArray getGradientsViewArray() {
        return underlying.getGradientsViewArray();
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        underlying.setBackpropGradientsViewArray(gradients);
    }

    @Override
    public void fit(INDArray data, LayerWorkspaceMgr workspaceMgr) {
        underlying.fit(data, workspaceMgr);
    }

    @Override
    public Gradient gradient() {
        return underlying.gradient();
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return underlying.gradientAndScore();
    }

    @Override
    public int batchSize() {
        return underlying.batchSize();
    }

    @Override
    public NeuralNetConfiguration conf() {
        return underlying.conf();
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        underlying.setConf(conf);
    }

    @Override
    public INDArray input() {
        return underlying.input();
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return underlying.getOptimizer();
    }

    @Override
    public INDArray getParam(String param) {
        return underlying.getParam(param);
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return underlying.paramTable();
    }

    @Override
    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        return underlying.paramTable(backpropParamsOnly);
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        underlying.setParamTable(paramTable);
    }

    @Override
    public void setParam(String key, INDArray val) {
        underlying.setParam(key, val);
    }

    @Override
    public void clear() {
        underlying.clear();
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        underlying.applyConstraints(iteration, epoch);
    }

    @Override
    public void init() {
        underlying.init();
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
        underlying.setListeners(listeners);
    }

    @Override
    public void setIndex(int index) {
        underlying.setIndex(index);
    }

    @Override
    public int getIndex() {
        return underlying.getIndex();
    }

    @Override
    public int getIterationCount() {
        return underlying.getIterationCount();
    }

    @Override
    public int getEpochCount() {
        return underlying.getEpochCount();
    }

    @Override
    public void setIterationCount(int iterationCount) {
        underlying.setIterationCount(iterationCount);
    }

    @Override
    public void setEpochCount(int epochCount) {
        underlying.setEpochCount(epochCount);
    }

    @Override
    public void setInput(INDArray input, LayerWorkspaceMgr workspaceMgr) {
        underlying.setInput(input, workspaceMgr);
    }

    @Override
    public void setInputMiniBatchSize(int size) {
        underlying.setInputMiniBatchSize(size);
    }

    @Override
    public int getInputMiniBatchSize() {
        return underlying.getInputMiniBatchSize();
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        underlying.setMaskArray(maskArray);
    }

    @Override
    public INDArray getMaskArray() {
        return underlying.getMaskArray();
    }

    @Override
    public boolean isPretrainLayer() {
        return underlying.isPretrainLayer();
    }

    @Override
    public void clearNoiseWeightParams() {
        underlying.clearNoiseWeightParams();
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState, int minibatchSize) {
        return underlying.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
    }

    @Override
    public void allowInputModification(boolean allow) {
        underlying.allowInputModification(allow);
    }

    @Override
    public LayerHelper getHelper() {
        return underlying.getHelper();
    }

    @Override
    public TrainingConfig getConfig() {
        return underlying.getConfig();
    }
}
