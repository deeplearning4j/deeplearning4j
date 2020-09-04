/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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
package org.deeplearning4j.rl4j.network;

import lombok.Getter;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.observation.Observation;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;

/**
 * A Q-Network that uses a {@link ComputationGraph} internally
 */
public class ComputationGraphQNetwork implements ITrainableNeuralNet<ComputationGraphQNetwork> {

    @Getter
    private final boolean recurrent;

    private final ComputationGraph model;

    public ComputationGraphQNetwork(ComputationGraph model) {
        this.model = model;
        this.recurrent = model.getOutputLayer(0) instanceof RnnOutputLayer;
    }

    @Override
    public void fit(FeaturesLabels featuresLabels) {
        INDArray[] features = new INDArray[] { featuresLabels.getFeatures() };
        INDArray[] labels = new INDArray[] { featuresLabels.getLabels(CommonLabelNames.QValues) };
        model.fit(features, labels);
    }

    @Override
    public Gradients computeGradients(FeaturesLabels updateLabels) {
        model.setInput(0, updateLabels.getFeatures());
        model.setLabels(updateLabels.getLabels(CommonLabelNames.QValues));
        model.computeGradientAndScore();
        Collection<TrainingListener> valueIterationListeners = model.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener l : valueIterationListeners) {
                l.onGradientCalculation(model);
            }
        }

        Gradients result = new Gradients(updateLabels.getBatchSize());
        result.putGradient(CommonGradientNames.QValues, model.gradient());

        return result;
    }

    @Override
    public void applyGradients(Gradients gradients) {
        ComputationGraphConfiguration cgConf = model.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();

        Gradient gradient = gradients.getGradient(CommonGradientNames.QValues);
        model.getUpdater().update(gradient, iterationCount, epochCount, (int)gradients.getBatchSize(), LayerWorkspaceMgr.noWorkspaces());
        model.params().subi(gradient.gradient());
        Collection<TrainingListener> iterationListeners = model.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener listener : iterationListeners) {
                listener.iterationDone(model, iterationCount, epochCount);
            }
        }
        cgConf.setIterationCount(iterationCount + 1);
    }

    @Override
    public void copyFrom(ComputationGraphQNetwork from) {
        model.setParams(from.model.params());
    }

    @Override
    public ComputationGraphQNetwork clone() {
        return new ComputationGraphQNetwork(model.clone());
    }

    @Override
    public NeuralNetOutput output(Observation observation) {
        if(!isRecurrent()) {
            return output(observation.getData());
        }

        return packageResult(model.rnnTimeStep(observation.getData())[0]);
    }

    @Override
    public NeuralNetOutput output(INDArray batch) {
        return packageResult(model.output(batch)[0]);
    }

    private NeuralNetOutput packageResult(INDArray qValues) {
        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.QValues, qValues);

        return result;
    }

    @Override
    public void reset() {
        model.rnnClearPreviousState();
    }

}