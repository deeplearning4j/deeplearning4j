/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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
package org.deeplearning4j.rl4j.network;

import lombok.Getter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A {@link INetworkHandler} implementation to be used with {@link MultiLayerNetwork MultiLayerNetworks}
 */
public class MultiLayerNetworkHandler implements INetworkHandler {

    private final MultiLayerNetwork model;

    @Getter
    private final boolean recurrent;
    private final MultiLayerConfiguration configuration;
    private final String labelName;
    private final String gradientName;
    private final int inputFeatureIdx;

    /**
     *
     * @param model The {@link MultiLayerNetwork} to use internally
     * @param labelName The name of the label (in {@link FeaturesLabels}) to use as the network's input.
     * @param gradientName The name of the gradient (in {@link Gradients}) to use as the network's output.
     * @param inputFeatureIdx The channel index to use as the input of the model
     */
    public MultiLayerNetworkHandler(MultiLayerNetwork model,
                                    String labelName,
                                    String gradientName,
                                    int inputFeatureIdx) {
        this.model = model;
        recurrent = model.getOutputLayer() instanceof RnnOutputLayer;
        configuration = model.getLayerWiseConfigurations();
        this.labelName = labelName;
        this.gradientName = gradientName;
        this.inputFeatureIdx = inputFeatureIdx;
    }

    @Override
    public void notifyGradientCalculation() {
        Iterable<TrainingListener> listeners = model.getListeners();

        if (listeners != null) {
            for (TrainingListener l : listeners) {
                l.onGradientCalculation(model);
            }
        }
    }

    @Override
    public void notifyIterationDone() {
        BaseNetwork.ModelCounters modelCounters = getModelCounters();
        Iterable<TrainingListener> listeners = model.getListeners();
        if (listeners != null) {
            for (TrainingListener l : listeners) {
                l.iterationDone(model, modelCounters.getIterationCount(), modelCounters.getEpochCount());
            }
        }
    }

    @Override
    public void performFit(FeaturesLabels featuresLabels) {
        INDArray features = featuresLabels.getFeatures().get(inputFeatureIdx);
        INDArray labels = featuresLabels.getLabels(labelName);
        model.fit(features, labels);
    }

    @Override
    public void performGradientsComputation(FeaturesLabels featuresLabels) {
        model.setInput(featuresLabels.getFeatures().get(inputFeatureIdx));
        model.setLabels(featuresLabels.getLabels(labelName));
        model.computeGradientAndScore();
    }

    private BaseNetwork.ModelCounters getModelCounters() {
        return new BaseNetwork.ModelCounters(configuration.getIterationCount(), configuration.getEpochCount());
    }

    @Override
    public void applyGradient(Gradients gradients, long batchSize) {
        BaseNetwork.ModelCounters modelCounters = getModelCounters();
        int iterationCount = modelCounters.getIterationCount();
        Gradient gradient = gradients.getGradient(gradientName);
        model.getUpdater().update(model, gradient, iterationCount, modelCounters.getEpochCount(), (int)batchSize, LayerWorkspaceMgr.noWorkspaces());
        model.params().subi(gradient.gradient());
        configuration.setIterationCount(iterationCount + 1);
    }

    @Override
    public INDArray[] recurrentStepOutput(Observation observation) {
        return new INDArray[] { model.rnnTimeStep(observation.getChannelData(inputFeatureIdx)) };
    }

    @Override
    public INDArray[] batchOutput(Features features) {
        return new INDArray[] { model.output(features.get(inputFeatureIdx)) };
    }

    @Override
    public INDArray[] stepOutput(Observation observation) {
        return new INDArray[] { model.output(observation.getChannelData(inputFeatureIdx)) };
    }

    @Override
    public void resetState() {
        model.rnnClearPreviousState();
    }

    @Override
    public INetworkHandler clone() {
        return new MultiLayerNetworkHandler(model.clone(), labelName, gradientName, inputFeatureIdx);
    }

    @Override
    public void copyFrom(INetworkHandler from) {
        model.setParams(((MultiLayerNetworkHandler) from).model.params());
    }

    @Override
    public void fillGradientsResponse(Gradients gradients) {
        gradients.putGradient(gradientName, model.gradient());
    }

}
