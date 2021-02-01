/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.rl4j.network.ac;

import lombok.Getter;
import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.rl4j.agent.learning.update.Features;
import org.deeplearning4j.rl4j.agent.learning.update.FeaturesLabels;
import org.deeplearning4j.rl4j.agent.learning.update.Gradients;
import org.deeplearning4j.rl4j.network.CommonGradientNames;
import org.deeplearning4j.rl4j.network.CommonLabelNames;
import org.deeplearning4j.rl4j.network.CommonOutputNames;
import org.deeplearning4j.rl4j.network.NeuralNetOutput;
import org.deeplearning4j.rl4j.observation.Observation;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Collection;

/**
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/23/16.
 */
@Deprecated
public class ActorCriticSeparate<NN extends ActorCriticSeparate> implements IActorCritic<NN> {

    final protected MultiLayerNetwork valueNet;
    final protected MultiLayerNetwork policyNet;
    @Getter
    final protected boolean recurrent;

    public ActorCriticSeparate(MultiLayerNetwork valueNet, MultiLayerNetwork policyNet) {
        this.valueNet = valueNet;
        this.policyNet = policyNet;
        this.recurrent = valueNet.getOutputLayer() instanceof RnnOutputLayer;
    }

    public NeuralNetwork[] getNeuralNetworks() {
        return new NeuralNetwork[] { valueNet, policyNet };
    }

    public static ActorCriticSeparate load(String pathValue, String pathPolicy) throws IOException {
        return new ActorCriticSeparate(ModelSerializer.restoreMultiLayerNetwork(pathValue),
                                       ModelSerializer.restoreMultiLayerNetwork(pathPolicy));
    }

    public void reset() {
        if (recurrent) {
            valueNet.rnnClearPreviousState();
            policyNet.rnnClearPreviousState();
        }
    }

    public void fit(INDArray input, INDArray[] labels) {
        valueNet.fit(input, labels[0]);
        policyNet.fit(input, labels[1]);
    }

    public INDArray[] outputAll(INDArray batch) {
        if (recurrent) {
            return new INDArray[] {valueNet.rnnTimeStep(batch), policyNet.rnnTimeStep(batch)};
        } else {
            return new INDArray[] {valueNet.output(batch), policyNet.output(batch)};
        }
    }

    public NN clone() {
        NN nn = (NN)new ActorCriticSeparate(valueNet.clone(), policyNet.clone());
        nn.valueNet.setListeners(valueNet.getListeners());
        nn.policyNet.setListeners(policyNet.getListeners());
        return nn;
    }

    @Override
    public void fit(FeaturesLabels featuresLabels) {
        valueNet.fit(featuresLabels.getFeatures().get(0), featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value));
        policyNet.fit(featuresLabels.getFeatures().get(0), featuresLabels.getLabels(CommonLabelNames.ActorCritic.Policy));
    }

    @Override
    public Gradients computeGradients(FeaturesLabels featuresLabels) {
        valueNet.setInput(featuresLabels.getFeatures().get(0));
        valueNet.setLabels(featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value));
        valueNet.computeGradientAndScore();
        Collection<TrainingListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener l : valueIterationListeners) {
                l.onGradientCalculation(valueNet);
            }
        }

        policyNet.setInput(featuresLabels.getFeatures().get(0));
        policyNet.setLabels(featuresLabels.getLabels(CommonLabelNames.ActorCritic.Policy));
        policyNet.computeGradientAndScore();
        Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (TrainingListener l : policyIterationListeners) {
                l.onGradientCalculation(policyNet);
            }
        }

        Gradients result = new Gradients(featuresLabels.getBatchSize());
        result.putGradient(CommonGradientNames.ActorCritic.Value, valueNet.gradient());
        result.putGradient(CommonGradientNames.ActorCritic.Policy, policyNet.gradient());
        return result;
    }

    @Override
    public void applyGradients(Gradients gradients) {
        int batchSize = (int)gradients.getBatchSize();
        MultiLayerConfiguration valueConf = valueNet.getLayerWiseConfigurations();
        int valueIterationCount = valueConf.getIterationCount();
        int valueEpochCount = valueConf.getEpochCount();
        Gradient valueGradient = gradients.getGradient(CommonGradientNames.ActorCritic.Value);
        valueNet.getUpdater().update(valueNet, valueGradient, valueIterationCount, valueEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        valueNet.params().subi(valueGradient.gradient());
        Collection<TrainingListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener listener : valueIterationListeners) {
                listener.iterationDone(valueNet, valueIterationCount, valueEpochCount);
            }
        }
        valueConf.setIterationCount(valueIterationCount + 1);

        MultiLayerConfiguration policyConf = policyNet.getLayerWiseConfigurations();
        int policyIterationCount = policyConf.getIterationCount();
        int policyEpochCount = policyConf.getEpochCount();
        Gradient policyGradient = gradients.getGradient(CommonGradientNames.ActorCritic.Policy);
        policyNet.getUpdater().update(policyNet, policyGradient, policyIterationCount, policyEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        policyNet.params().subi(policyGradient.gradient());
        Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (TrainingListener listener : policyIterationListeners) {
                listener.iterationDone(policyNet, policyIterationCount, policyEpochCount);
            }
        }
        policyConf.setIterationCount(policyIterationCount + 1);
    }

    public void copyFrom(NN from) {
        valueNet.setParams(from.valueNet.params());
        policyNet.setParams(from.policyNet.params());
    }

    @Deprecated
    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        valueNet.setInput(input);
        valueNet.setLabels(labels[0]);
        valueNet.computeGradientAndScore();
        Collection<TrainingListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener l : valueIterationListeners) {
                    l.onGradientCalculation(valueNet);
            }
        }

        policyNet.setInput(input);
        policyNet.setLabels(labels[1]);
        policyNet.computeGradientAndScore();
        Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (TrainingListener l : policyIterationListeners) {
                l.onGradientCalculation(policyNet);
            }
        }
        return new Gradient[] {valueNet.gradient(), policyNet.gradient()};
    }

    @Deprecated
    public void applyGradient(Gradient[] gradient, int batchSize) {
        MultiLayerConfiguration valueConf = valueNet.getLayerWiseConfigurations();
        int valueIterationCount = valueConf.getIterationCount();
        int valueEpochCount = valueConf.getEpochCount();
        valueNet.getUpdater().update(valueNet, gradient[0], valueIterationCount, valueEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        valueNet.params().subi(gradient[0].gradient());
        Collection<TrainingListener> valueIterationListeners = valueNet.getListeners();
        if (valueIterationListeners != null && valueIterationListeners.size() > 0) {
            for (TrainingListener listener : valueIterationListeners) {
                listener.iterationDone(valueNet, valueIterationCount, valueEpochCount);
            }
        }
        valueConf.setIterationCount(valueIterationCount + 1);

        MultiLayerConfiguration policyConf = policyNet.getLayerWiseConfigurations();
        int policyIterationCount = policyConf.getIterationCount();
        int policyEpochCount = policyConf.getEpochCount();
        policyNet.getUpdater().update(policyNet, gradient[1], policyIterationCount, policyEpochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        policyNet.params().subi(gradient[1].gradient());
        Collection<TrainingListener> policyIterationListeners = policyNet.getListeners();
        if (policyIterationListeners != null && policyIterationListeners.size() > 0) {
            for (TrainingListener listener : policyIterationListeners) {
                listener.iterationDone(policyNet, policyIterationCount, policyEpochCount);
            }
        }
        policyConf.setIterationCount(policyIterationCount + 1);
    }

    public double getLatestScore() {
        return valueNet.score();
    }

    public void save(OutputStream stream) throws IOException {
        throw new UnsupportedOperationException("Call save(streamValue, streamPolicy)");
    }

    public void save(String path) throws IOException {
        throw new UnsupportedOperationException("Call save(pathValue, pathPolicy)");
    }

    public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {
        ModelSerializer.writeModel(valueNet, streamValue, true);
        ModelSerializer.writeModel(policyNet, streamPolicy, true);
    }

    public void save(String pathValue, String pathPolicy) throws IOException {
        ModelSerializer.writeModel(valueNet, pathValue, true);
        ModelSerializer.writeModel(policyNet, pathPolicy, true);
    }

    @Override
    public NeuralNetOutput output(Observation observation) {
        if(!isRecurrent()) {
            return output(observation.getChannelData(0));
        }

        INDArray observationData = observation.getChannelData(0);
        return packageResult(valueNet.rnnTimeStep(observationData), policyNet.rnnTimeStep(observationData));
    }

    @Override
    public NeuralNetOutput output(INDArray batch) {
        return packageResult(valueNet.output(batch), policyNet.output(batch));
    }

    @Override
    public NeuralNetOutput output(Features features) {
        throw new NotImplementedException("Not implemented in legacy classes");
    }

    private NeuralNetOutput packageResult(INDArray value, INDArray policy) {
        NeuralNetOutput result = new NeuralNetOutput();
        result.put(CommonOutputNames.ActorCritic.Value, value);
        result.put(CommonOutputNames.ActorCritic.Policy, policy);

        return result;
    }
}


