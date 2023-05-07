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

package org.deeplearning4j.optimize.solvers;

import lombok.Getter;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.StepFunction;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.stepfunctions.NegativeDefaultStepFunction;
import org.deeplearning4j.optimize.stepfunctions.NegativeGradientStepFunction;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.common.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Base optimizer
 * @author Adam Gibson
 */
public abstract class BaseOptimizer implements ConvexOptimizer {

    protected NeuralNetConfiguration conf;
    protected static final Logger log = LoggerFactory.getLogger(BaseOptimizer.class);
    @Getter
    protected StepFunction stepFunction;
    protected Collection<TrainingListener> trainingListeners = new ArrayList<>();
    protected Model model;
    protected Updater updater;
    protected ComputationGraphUpdater computationGraphUpdater;
    protected double step;
    private int batchSize;
    protected double score, oldScore;
    protected double stepMax = Double.MAX_VALUE;
    public final static String GRADIENT_KEY = "g";
    public final static String SCORE_KEY = "score";
    public final static String PARAMS_KEY = "params";
    public final static String SEARCH_DIR = "searchDirection";
    protected Map<String, Object> searchState = new ConcurrentHashMap<>();


    /**
     *
     * @param conf
     * @param stepFunction
     * @param trainingListeners
     * @param model
     */
    public BaseOptimizer(NeuralNetConfiguration conf, StepFunction stepFunction,
                         Collection<TrainingListener> trainingListeners, Model model) {
        this.conf = conf;
        this.stepFunction = (stepFunction != null ? stepFunction : getDefaultStepFunctionForOptimizer(this.getClass()));
        this.trainingListeners = trainingListeners != null ? trainingListeners : new ArrayList<TrainingListener>();
        this.model = model;
    }



    @Override
    public double score() {
        throw new UnsupportedOperationException("Not yet reimplemented");
    }

    @Override
    public Updater getUpdater() {
        return getUpdater(true);
    }

    @Override
    public Updater getUpdater(boolean initializeIfReq) {
        if (updater == null && initializeIfReq) {
            updater = UpdaterCreator.getUpdater(model);
        }
        return updater;
    }

    @Override
    public void setUpdater(Updater updater) {
        this.updater = updater;
    }



    @Override
    public ComputationGraphUpdater getComputationGraphUpdater() {
        return getComputationGraphUpdater(true);
    }

    @Override
    public ComputationGraphUpdater getComputationGraphUpdater(boolean initializIfReq) {
        if (computationGraphUpdater == null && model instanceof ComputationGraph && initializIfReq) {
            computationGraphUpdater = new ComputationGraphUpdater((ComputationGraph) model);
        }
        return computationGraphUpdater;
    }

    @Override
    public void setUpdaterComputationGraph(ComputationGraphUpdater updater) {
        this.computationGraphUpdater = updater;
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
        if (listeners == null)
            this.trainingListeners = Collections.emptyList();
        else
            this.trainingListeners = listeners;
    }

    @Override
    public NeuralNetConfiguration getConf() {
        return conf;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        oldScore = score;
        model.computeGradientAndScore(workspaceMgr);

        if (trainingListeners != null && !trainingListeners.isEmpty()) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                for (TrainingListener l : trainingListeners) {
                    l.onGradientCalculation(model);
                }
            }
        }

        Pair<Gradient, Double> pair = model.gradientAndScore();
        score = pair.getSecond();
        updateGradientAccordingToParams(pair.getFirst(), model, model.batchSize(), workspaceMgr);
        return pair;
    }

    /**
     * Optimize call. This runs the optimizer.
     * @return whether it converged or not
     */
    // TODO add flag to allow retaining state between mini batches and when to apply updates
    @Override
    public boolean optimize(LayerWorkspaceMgr workspaceMgr) {
        throw new UnsupportedOperationException("BackTrackLineSearch has been removed. Use SGD.");
    }

    protected void postFirstStep(INDArray gradient) {
        //no-op
    }

    @Override
    public int batchSize() {
        return batchSize;
    }

    @Override
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }


    /**
     * Pre preProcess to setup initial searchDirection approximation
     */
    @Override
    public void preProcessLine() {
        //no-op
    }

    /**
     * Post step to update searchDirection with new gradient and parameter information
     */
    @Override
    public void postStep(INDArray gradient) {
        //no-op
    }


    @Override
    public void updateGradientAccordingToParams(Gradient gradient, Model model, int batchSize, LayerWorkspaceMgr workspaceMgr) {
        if (model instanceof ComputationGraph) {
            ComputationGraph graph = (ComputationGraph) model;
            if (computationGraphUpdater == null) {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    computationGraphUpdater = new ComputationGraphUpdater(graph);
                }
            }
            computationGraphUpdater.update(gradient, getIterationCount(model), getEpochCount(model), batchSize, workspaceMgr);
        } else {
            if (updater == null) {
                try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    updater = UpdaterCreator.getUpdater(model);
                }
            }
            Layer layer = (Layer) model;

            updater.update(layer, gradient, getIterationCount(model), getEpochCount(model), batchSize, workspaceMgr);
        }
    }

    /**
     * Setup the initial search state
     * @param pair
     */
    @Override
    public void setupSearchState(Pair<Gradient, Double> pair) {
        INDArray gradient = pair.getFirst().gradient(conf.variables());
        INDArray params = model.params().dup(); //Need dup here: params returns an array that isn't a copy (hence changes to this are problematic for line search methods)
        searchState.put(GRADIENT_KEY, gradient);
        searchState.put(SCORE_KEY, pair.getSecond());
        searchState.put(PARAMS_KEY, params);
    }


    public static StepFunction getDefaultStepFunctionForOptimizer(Class<? extends ConvexOptimizer> optimizerClass) {
        if (optimizerClass == StochasticGradientDescent.class) {
            return new NegativeGradientStepFunction();
        } else {
            return new NegativeDefaultStepFunction();
        }
    }

    public static int getIterationCount(Model model) {
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getIterationCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getIterationCount();
        } else {
            return model.conf().getIterationCount();
        }
    }

    public static void incrementIterationCount(Model model, int incrementBy) {
        if (model instanceof MultiLayerNetwork) {
            MultiLayerConfiguration conf = ((MultiLayerNetwork) model).getLayerWiseConfigurations();
            conf.setIterationCount(conf.getIterationCount() + incrementBy);
        } else if (model instanceof ComputationGraph) {
            ComputationGraphConfiguration conf = ((ComputationGraph) model).getConfiguration();
            conf.setIterationCount(conf.getIterationCount() + incrementBy);
        } else {
            model.conf().setIterationCount(model.conf().getIterationCount() + incrementBy);
        }
    }

    public static int getEpochCount(Model model){
        if (model instanceof MultiLayerNetwork) {
            return ((MultiLayerNetwork) model).getLayerWiseConfigurations().getEpochCount();
        } else if (model instanceof ComputationGraph) {
            return ((ComputationGraph) model).getConfiguration().getEpochCount();
        } else {
            return model.conf().getEpochCount();
        }
    }

    public static void applyConstraints(Model model){
        int iter = getIterationCount(model);
        int epoch = getEpochCount(model);
        model.applyConstraints(iter, epoch);
    }

}
