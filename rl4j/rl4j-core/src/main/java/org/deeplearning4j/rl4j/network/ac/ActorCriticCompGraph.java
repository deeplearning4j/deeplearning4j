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

package org.deeplearning4j.rl4j.network.ac;

import lombok.Getter;
import org.apache.commons.lang3.NotImplementedException;
import org.deeplearning4j.nn.api.NeuralNetwork;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.recurrent.RnnOutputLayer;
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
 * @author rubenfiszel (ruben.fiszel@epfl.ch) on 8/9/16.
 *
 * Standard implementation of ActorCriticCompGraph
 */
@Deprecated
public class ActorCriticCompGraph implements IActorCritic<ActorCriticCompGraph> {

    final protected ComputationGraph cg;
    @Getter
    final protected boolean recurrent;

    public ActorCriticCompGraph(ComputationGraph cg) {
        this.cg = cg;
        this.recurrent = cg.getOutputLayer(0) instanceof RnnOutputLayer;
    }

    public NeuralNetwork[] getNeuralNetworks() {
        return new NeuralNetwork[] { cg };
    }

    public static ActorCriticCompGraph load(String path) throws IOException {
        return new ActorCriticCompGraph(ModelSerializer.restoreComputationGraph(path));
    }

    public void fit(INDArray input, INDArray[] labels) {
        cg.fit(new INDArray[] {input}, labels);
    }

    public void reset() {
        if (recurrent) {
            cg.rnnClearPreviousState();
        }
    }

    public INDArray[] outputAll(INDArray batch) {
        if (recurrent) {
            return cg.rnnTimeStep(batch);
        } else {
            return cg.output(batch);
        }
    }

    public ActorCriticCompGraph clone() {
        ActorCriticCompGraph nn = new ActorCriticCompGraph(cg.clone());
        nn.cg.setListeners(cg.getListeners());
        return nn;
    }

    @Override
    public void fit(FeaturesLabels featuresLabels) {
        INDArray[] features = new INDArray[] { featuresLabels.getFeatures().get(0) };
        INDArray[] labels = new INDArray[] { featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value), featuresLabels.getLabels(CommonLabelNames.ActorCritic.Policy) };
        cg.fit(features, labels);
    }

    @Override
    public Gradients computeGradients(FeaturesLabels featuresLabels) {
        cg.setInput(0, featuresLabels.getFeatures().get(0));
        cg.setLabels(featuresLabels.getLabels(CommonLabelNames.ActorCritic.Value), featuresLabels.getLabels(CommonLabelNames.ActorCritic.Policy));
        cg.computeGradientAndScore();
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener l : iterationListeners) {
                l.onGradientCalculation(cg);
            }
        }

        Gradients result = new Gradients(featuresLabels.getBatchSize());
        result.putGradient(CommonGradientNames.ActorCritic.Combined, cg.gradient());

        return result;
    }

    @Override
    public void applyGradients(Gradients gradients) {
        ComputationGraphConfiguration cgConf = cg.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();

        Gradient gradient = gradients.getGradient(CommonGradientNames.ActorCritic.Combined);
        cg.getUpdater().update(gradient, iterationCount, epochCount, (int)gradients.getBatchSize(), LayerWorkspaceMgr.noWorkspaces());
        cg.params().subi(gradient.gradient());
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener listener : iterationListeners) {
                listener.iterationDone(cg, iterationCount, epochCount);
            }
        }
        cgConf.setIterationCount(iterationCount + 1);
    }

    public void copyFrom(ActorCriticCompGraph from) {
        cg.setParams(from.cg.params());
    }

    @Deprecated
    public Gradient[] gradient(INDArray input, INDArray[] labels) {
        cg.setInput(0, input);
        cg.setLabels(labels);
        cg.computeGradientAndScore();
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener l : iterationListeners) {
                l.onGradientCalculation(cg);
            }
        }
        return new Gradient[] {cg.gradient()};
    }


    public void applyGradient(Gradient[] gradient, int batchSize) {
        if (recurrent) {
            // assume batch sizes of 1 for recurrent networks,
            // since we are learning each episode as a time serie
            batchSize = 1;
        }
        ComputationGraphConfiguration cgConf = cg.getConfiguration();
        int iterationCount = cgConf.getIterationCount();
        int epochCount = cgConf.getEpochCount();
        cg.getUpdater().update(gradient[0], iterationCount, epochCount, batchSize, LayerWorkspaceMgr.noWorkspaces());
        cg.params().subi(gradient[0].gradient());
        Collection<TrainingListener> iterationListeners = cg.getListeners();
        if (iterationListeners != null && iterationListeners.size() > 0) {
            for (TrainingListener listener : iterationListeners) {
                listener.iterationDone(cg, iterationCount, epochCount);
            }
        }
        cgConf.setIterationCount(iterationCount + 1);
    }

    public double getLatestScore() {
        return cg.score();
    }

    public void save(OutputStream stream) throws IOException {
        ModelSerializer.writeModel(cg, stream, true);
    }

    public void save(String path) throws IOException {
        ModelSerializer.writeModel(cg, path, true);
    }

    public void save(OutputStream streamValue, OutputStream streamPolicy) throws IOException {
        throw new UnsupportedOperationException("Call save(stream)");
    }

    public void save(String pathValue, String pathPolicy) throws IOException {
        throw new UnsupportedOperationException("Call save(path)");
    }

    @Override
    public NeuralNetOutput output(Observation observation) {
        if(!isRecurrent()) {
            return output(observation.getChannelData(0));
        }

        INDArray[] cgOutput = cg.rnnTimeStep(observation.getChannelData(0));
        return packageResult(cgOutput[0], cgOutput[1]);
    }

    @Override
    public NeuralNetOutput output(INDArray batch) {
        INDArray[] cgOutput = cg.output(batch);
        return packageResult(cgOutput[0], cgOutput[1]);
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

