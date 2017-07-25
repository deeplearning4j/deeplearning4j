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

package org.deeplearning4j.nn.multilayer;


import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.berkeley.Triple;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.*;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.OneTimeLogger;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.heartbeat.Heartbeat;
import org.nd4j.linalg.heartbeat.reports.Environment;
import org.nd4j.linalg.heartbeat.reports.Event;
import org.nd4j.linalg.heartbeat.reports.Task;
import org.nd4j.linalg.heartbeat.utils.EnvironmentUtils;
import org.nd4j.linalg.heartbeat.utils.TaskUtils;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.memory.abstracts.DummyWorkspace;
import org.nd4j.linalg.util.FeatureUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;

import static org.deeplearning4j.nn.graph.ComputationGraph.workspaceConfigurationCache;


/**
 * MultiLayerNetwork is a neural network with multiple layers in a stack, and usually an output layer.
 * For neural networks with a more complex connection architecture, use {@link org.deeplearning4j.nn.graph.ComputationGraph}
 * which allows for an arbitrary directed acyclic graph connection structure between layers.
 * MultiLayerNetwork is trainable via backprop, with optional pretraining, depending on the type of layers it contains.
 *
 * @author Adam Gibson
 */
public class MultiLayerNetwork implements Serializable, Classifier, Layer, NeuralNetwork {
    private static final Logger log = LoggerFactory.getLogger(MultiLayerNetwork.class);

    //the hidden neural network layers (including output layer)
    protected Layer[] layers;
    protected LinkedHashMap<String, Layer> layerMap = new LinkedHashMap<>();

    //Current training data: input features and labels
    protected INDArray input, labels;

    protected boolean initCalled = false;
    private Collection<IterationListener> listeners = new ArrayList<>();
    private Collection<TrainingListener> trainingListeners = new ArrayList<>();

    protected NeuralNetConfiguration defaultConfiguration;
    protected MultiLayerConfiguration layerWiseConfigurations;
    protected Gradient gradient;
    protected INDArray epsilon;
    protected double score;
    @Setter
    protected boolean initDone = false;
    protected INDArray flattenedParams; //Params for all layers are a view/subset of this array
    @Getter
    protected transient INDArray flattenedGradients; //Gradients for all layers are a view/subset of this array

    protected ThreadLocal<Long> lastEtlTime = new ThreadLocal<>();

    /*
      Binary drop connect mask
     */
    protected INDArray mask;

    protected int layerIndex; //For Layer.get/setIndex()

    protected transient Solver solver; //Used to call optimizers during backprop

    protected final static String workspaceExternal = "LOOP_EXTERNAL";
    protected final static String workspaceFeedForward = "LOOP_FF";
    protected final static String workspaceBackProp = "LOOP_BP";
    public final static String workspaceTBPTT = "LOOP_TBPTT";

    protected final static WorkspaceConfiguration workspaceConfigurationExternal = WorkspaceConfiguration.builder()
                    .initialSize(0).overallocationLimit(0.3).policyLearning(LearningPolicy.FIRST_LOOP)
                    .policyReset(ResetPolicy.BLOCK_LEFT).policySpill(SpillPolicy.REALLOCATE)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE).build();

    protected WorkspaceConfiguration workspaceConfigurationFeedForward = WorkspaceConfiguration.builder().initialSize(0)
                    .overallocationLimit(0.2).policyReset(ResetPolicy.BLOCK_LEFT)
                    .policyLearning(LearningPolicy.OVER_TIME).policySpill(SpillPolicy.REALLOCATE)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE).build();

    protected final static WorkspaceConfiguration workspaceConfigurationTBPTT = WorkspaceConfiguration.builder()
                    .initialSize(0).overallocationLimit(0.2).policyReset(ResetPolicy.BLOCK_LEFT)
                    .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
                    .policyLearning(LearningPolicy.OVER_TIME).build();

    public MultiLayerNetwork(MultiLayerConfiguration conf) {
        this.layerWiseConfigurations = conf;
        this.defaultConfiguration = conf.getConf(0).clone();
    }

    /**
     * This method sets specified CacheMode for all layers within network
     *
     * @param mode
     */
    public void setCacheMode(CacheMode mode) {
        if (mode == null)
            mode = CacheMode.NONE;

        for (Layer layer : layers) {
            layer.setCacheMode(mode);
        }
    }

    public void setLastEtlTime(long time) {
        lastEtlTime.set(time);
    }

    public long getLastEtlTime() {
        Long time = lastEtlTime.get();
        return time == null ? 0L : time;
    }

    /**
     * Initialize the network based on the configuration
     *
     * @param conf   the configuration json
     * @param params the parameters
     */
    public MultiLayerNetwork(String conf, INDArray params) {
        this(MultiLayerConfiguration.fromJson(conf));
        init();
        setParameters(params);
    }


    /**
     * Initialize the network based on the configuraiton
     *
     * @param conf   the configuration
     * @param params the parameters
     */
    public MultiLayerNetwork(MultiLayerConfiguration conf, INDArray params) {
        this(conf);
        init();
        setParameters(params);
    }


    protected void intializeConfigurations() {

        if (layerWiseConfigurations == null)
            layerWiseConfigurations = new MultiLayerConfiguration.Builder().build();

        if (layers == null)
            layers = new Layer[getnLayers()];

        if (defaultConfiguration == null)
            defaultConfiguration = new NeuralNetConfiguration.Builder().build();
    }


    /**
     * Perform layerwise pretraining on all pre-trainable layers in the network (VAEs, RBMs, Autoencoders, etc)<br>
     * Note that pretraining will be performed on one layer after the other, resetting the DataSetIterator between iterations.<br>
     * For multiple epochs per layer, appropriately wrap the iterator (for example, a MultipleEpochsIterator) or train
     * each layer manually using {@link #pretrainLayer(int, DataSetIterator)}
     *
     * @param iter Training data
     */
    public void pretrain(DataSetIterator iter) {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        if (!layerWiseConfigurations.isPretrain())
            return;

        for (int i = 0; i < getnLayers(); i++) {
            pretrainLayer(i, iter);
        }
    }

    /**
     * Perform layerwise unsupervised training on a single pre-trainable layer in the network (VAEs, RBMs, Autoencoders, etc)<br>
     * If the specified layer index (0 to numLayers - 1) is not a pretrainable layer, this is a no-op.
     *
     * @param layerIdx Index of the layer to train (0 to numLayers-1)
     * @param iter Training data
     */
    public void pretrainLayer(int layerIdx, DataSetIterator iter) {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        if (!layerWiseConfigurations.isPretrain())
            return;
        if (layerIdx >= layers.length) {
            throw new IllegalArgumentException(
                            "Cannot pretrain layer: layerIdx (" + layerIdx + ") >= numLayers (" + layers.length + ")");
        }

        Layer layer = layers[layerIdx];
        if (!layer.isPretrainLayer())
            return;

        if (!iter.hasNext() && iter.resetSupported()) {
            iter.reset();
        }

        MemoryWorkspace workspace =
                layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        ComputationGraph.workspaceConfigurationExternal, ComputationGraph.workspaceExternal);
        MemoryWorkspace cache = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(ComputationGraph.workspaceConfigurationCache, ComputationGraph.workspaceCache);

        log.info("Starting unsupervised training on layer " + layerIdx);
        while (iter.hasNext()) {
            DataSet next = iter.next();

            try (MemoryWorkspace wsCache = cache.notifyScopeEntered()) {
                try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                    input = next.getFeatureMatrix();
                    pretrainLayer(layerIdx, input);
                }
            }
        }
    }

    /**
     * Perform layerwise unsupervised training on a single pre-trainable layer in the network (VAEs, RBMs, Autoencoders, etc)<br>
     * If the specified layer index (0 to numLayers - 1) is not a pretrainable layer, this is a no-op.
     *
     * @param layerIdx Index of the layer to train (0 to numLayers-1)
     * @param features Training data array
     */
    public void pretrainLayer(int layerIdx, INDArray features) {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        if (!layerWiseConfigurations.isPretrain())
            return;
        if (layerIdx >= layers.length) {
            throw new IllegalArgumentException(
                            "Cannot pretrain layer: layerIdx (" + layerIdx + ") >= numLayers (" + layers.length + ")");
        }

        INDArray layerInput = features;
        if (layerIdx == 0 && getLayerWiseConfigurations().getInputPreProcess(0) != null) {
            layerInput = getLayerWiseConfigurations().getInputPreProcess(0).preProcess(input, input.size(0));
        }

        Layer layer = layers[layerIdx];
        if (!layer.isPretrainLayer())
            return;
        layer.conf().setPretrain(true);

        MemoryWorkspace workspace = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE
                ? new DummyWorkspace()
                : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                ? Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceExternal)
                : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                workspaceConfigurationFeedForward, workspaceFeedForward);

        MemoryWorkspace pretrain = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE
                ? new DummyWorkspace()
                : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                ? Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceExternal)
                : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                workspaceConfigurationFeedForward, ComputationGraph.workspacePretrain);

        try (MemoryWorkspace wsP = pretrain.notifyScopeEntered()) {
            //Do forward pass to the layer to be pretrained
            for (int j = 0; j < layerIdx; j++) {
                try (MemoryWorkspace wsFF = workspace.notifyScopeEntered()) {
                    if (Nd4j.getWorkspaceManager().checkIfWorkspaceExists(ComputationGraph.workspacePretrain))
                        layerInput = activationFromPrevLayer(j, layerInput, true).leverageTo(ComputationGraph.workspacePretrain);
                    else
                        layerInput = activationFromPrevLayer(j, layerInput, true);
                }
            }
            layer.fit(layerInput);
        }

        // Turn off pretrain after it is complete
        layer.conf().setPretrain(false);
    }


    /**
     * @deprecated use {@link #pretrain(DataSetIterator)} or {@link #pretrainLayer(int, DataSetIterator)} or {@link #pretrainLayer(int, INDArray)}.
     * Pretraining each layer in a row on a single minibatch (as per this method) instead of N epochs per layer is not advisable.
     */
    @Deprecated
    public void pretrain(INDArray input) {
        if (!layerWiseConfigurations.isPretrain())
            return;
        if (flattenedGradients == null) {
            initGradientsView();
        }

        MemoryWorkspace workspace = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE
                ? new DummyWorkspace()
                : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                ? Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceExternal)
                : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                workspaceConfigurationFeedForward, workspaceFeedForward);

        MemoryWorkspace pretrain = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE
                ? new DummyWorkspace()
                : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                ? Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceExternal)
                : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                workspaceConfigurationFeedForward, ComputationGraph.workspacePretrain);

        /* During pretrain, feed forward expected activations of network, use activation cooccurrences during pretrain  */

        int miniBatchSize = input.size(0);
        INDArray layerInput = null;
        Layer layer;
        int nPretrainLayers = getnLayers();
        if (getLayer(getnLayers() - 1) instanceof IOutputLayer)
            nPretrainLayers--;

        try (MemoryWorkspace wsP = pretrain.notifyScopeEntered()) {
            for (int i = 0; i < nPretrainLayers; i++) {
                try (MemoryWorkspace wsFF = workspace.notifyScopeEntered()) {
                    layer = getLayer(i);
                    if (i == 0) {
                        if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
                            layerInput = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input, miniBatchSize).leverageTo(ComputationGraph.workspacePretrain);
                        } else {
                            layerInput = input.leverageTo(ComputationGraph.workspacePretrain);
                        }
                    } else {
                        layerInput = activationFromPrevLayer(i - 1, layerInput, true).leverageTo(ComputationGraph.workspacePretrain);
                    }
                    layer.conf().setPretrain(true);
                    layer.fit(layerInput);
                    layer.conf().setPretrain(false);
                }
            }
        }
    }

    @Override
    public int batchSize() {
        return input.size(0);
    }

    @Override
    public NeuralNetConfiguration conf() {
        return defaultConfiguration;
    }

    @Override
    public void setConf(NeuralNetConfiguration conf) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray input() {
        return input;
    }

    @Override
    public void validateInput() {

    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return solver.getOptimizer();
    }

    @Override
    public INDArray getParam(String param) {
        //Get params for MultiLayerNetwork sub layers.
        //Parameter keys here: same as MultiLayerNetwork.backprop().
        int idx = param.indexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: not have layer separator: \"" + param + "\"");
        int layerIdx = Integer.parseInt(param.substring(0, idx));
        String newKey = param.substring(idx + 1);

        return layers[layerIdx].getParam(newKey);
    }

    @Override
    public void initParams() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        //Get all parameters from all layers
        Map<String, INDArray> allParams = new LinkedHashMap<>();
        for (int i = 0; i < layers.length; i++) {
            Map<String, INDArray> paramMap = layers[i].paramTable(backpropParamsOnly);
            for (Map.Entry<String, INDArray> entry : paramMap.entrySet()) {
                String newKey = i + "_" + entry.getKey();
                allParams.put(newKey, entry.getValue());
            }
        }
        return allParams;
    }

    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        throw new UnsupportedOperationException();

    }

    @Override
    public void setParam(String key, INDArray val) {
        //Set params for MultiLayerNetwork sub layers.
        //Parameter keys here: same as MultiLayerNetwork.backprop().
        int idx = key.indexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: not have layer separator: \"" + key + "\"");
        int layerIdx = Integer.parseInt(key.substring(0, idx));
        String newKey = key.substring(idx + 1);

        layers[layerIdx].setParam(newKey, val);
    }


    public MultiLayerConfiguration getLayerWiseConfigurations() {
        return layerWiseConfigurations;
    }

    public void setLayerWiseConfigurations(MultiLayerConfiguration layerWiseConfigurations) {
        this.layerWiseConfigurations = layerWiseConfigurations;
    }

    /**
     * Base class for initializing the neuralNets based on the input.
     * This is meant for capturing numbers such as input columns or other things.
     *
     * @param input the input matrix for training
     */
    public void initializeLayers(INDArray input) {
        if (input == null)
            throw new IllegalArgumentException("Unable to initialize neuralNets with empty input");

        this.input = input;
        setInputMiniBatchSize(input.size(0));

        if (!initCalled)
            init();
    }

    /**
     * Initialize the MultiLayerNetwork. This should be called once before the network is used.
     */
    public void init() {
        init(null, false);
    }

    /**
     * Initialize the MultiLayerNetwork, optionally with an existing parameters array.
     * If an existing parameters array is specified, it will be used (and the values will not be modified) in the network;
     * if no parameters array is specified, parameters will be initialized randomly according to the network configuration.
     *
     * @param parameters              Network parameter. May be null. If null: randomly initialize.
     * @param cloneParametersArray    Whether the parameter array (if any) should be cloned, or used directly
     */
    public void init(INDArray parameters, boolean cloneParametersArray) {
        if (layerWiseConfigurations == null || layers == null)
            intializeConfigurations();
        if (initCalled)
            return;

        OneTimeLogger.info(log, "Starting MultiLayerNetwork with WorkspaceModes set to [training: {}; inference: {}]",
                        layerWiseConfigurations.getTrainingWorkspaceMode(),
                        layerWiseConfigurations.getInferenceWorkspaceMode());

        if (layerWiseConfigurations.getCacheMode() == CacheMode.HOST) {
            workspaceConfigurationCache.setPolicyMirroring(MirroringPolicy.HOST_ONLY);
        }

        int nLayers = getnLayers();

        if (nLayers < 1)
            throw new IllegalStateException("Unable to create network: number of layers is less than 1");

        if (this.layers == null || this.layers[0] == null) {
            if (this.layers == null)
                this.layers = new Layer[nLayers];

            //First: Work out total length of (backprop) params
            int paramLength = 0;
            int[] nParamsPerLayer = new int[nLayers];
            for (int i = 0; i < nLayers; i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                nParamsPerLayer[i] = conf.getLayer().initializer().numParams(conf);
                paramLength += nParamsPerLayer[i];
            }

            //Create parameters array, if required
            boolean initializeParams;
            if (parameters != null) {
                if (!parameters.isRowVector())
                    throw new IllegalArgumentException("Invalid parameters: should be a row vector");
                if (parameters.length() != paramLength)
                    throw new IllegalArgumentException("Invalid parameters: expected length " + paramLength
                                    + ", got length " + parameters.length());

                if (cloneParametersArray)
                    flattenedParams = parameters.dup();
                else
                    flattenedParams = parameters;

                initializeParams = false;
            } else {
                flattenedParams = Nd4j.create(1, paramLength);
                initializeParams = true;
            }

            //Set RNG seed, for repeatability between initializations when set
            if (initializeParams) {
                Nd4j.getRandom().setSeed(getDefaultConfiguration().getSeed());
            }

            // construct multi-layer
            int paramCountSoFar = 0;
            for (int i = 0; i < nLayers; i++) {
                INDArray paramsView;
                if (nParamsPerLayer[i] > 0) {
                    paramsView = flattenedParams.get(NDArrayIndex.point(0),
                                    NDArrayIndex.interval(paramCountSoFar, paramCountSoFar + nParamsPerLayer[i]));
                } else {
                    paramsView = null;
                }
                paramCountSoFar += nParamsPerLayer[i];

                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                layers[i] = conf.getLayer().instantiate(conf, listeners, i, paramsView, initializeParams);
                layerMap.put(conf.getLayer().getLayerName(), layers[i]);
            }
            initCalled = true;
        }

        //Set parameters in MultiLayerNetwork.defaultConfiguration for later use in BaseOptimizer.setupSearchState() etc
        //Keyed as per backprop()
        defaultConfiguration.clearVariables();
        List<String> variables = defaultConfiguration.variables(false);
        for (int i = 0; i < layers.length; i++) {
            for (String s : layers[i].conf().variables()) {
                variables.add(i + "_" + s);
            }
        }

        // now we init solver & optimizer
        if (solver == null) {
            try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                solver.initOptimizer();
            }
        }
    }

    /**
     * This method allows you to specificy GradientsAccumulator instance to be used with this model
     *
     * PLEASE NOTE: Do not use this method unless you understand how to use GradientsAccumulator & updates sharing.
     * PLEASE NOTE: Do not use this method on standalone model
     *
     * @param accumulator
     */
    public void setGradientsAccumulator(GradientsAccumulator accumulator) {
        if (!isInitCalled())
            init();

        solver.getOptimizer().setGradientsAccumulator(accumulator);
    }

    public boolean isInitCalled() {
        return initCalled;
    }

    /**
     * This method: initializes the flattened gradients array (used in backprop) and sets the appropriate subset in all layers.
     * As a general rule, this shouldn't ever need to be called manually when doing training via fit(DataSet) or fit(DataSetIterator)
     */
    public void initGradientsView() {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            if (layers == null)
                init();

            int nLayers = layers.length;

            //First: Work out total length of params
            int paramLength = 0;
            int[] nParamsPerLayer = new int[nLayers];
            for (int i = 0; i < nLayers; i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                nParamsPerLayer[i] = conf.getLayer().initializer().numParams(conf);
                paramLength += nParamsPerLayer[i];
            }

            flattenedGradients = Nd4j.zeros(new int[] {1, paramLength}, 'f'); //No need to initialize, as each layer will do it each iteration anyway

            int backpropParamsSoFar = 0;
            for (int i = 0; i < layers.length; i++) {
                if (nParamsPerLayer[i] == 0)
                    continue; //This layer doesn't have any parameters...
                INDArray thisLayerGradView = flattenedGradients.get(NDArrayIndex.point(0),
                                NDArrayIndex.interval(backpropParamsSoFar, backpropParamsSoFar + nParamsPerLayer[i]));
                layers[i].setBackpropGradientsViewArray(thisLayerGradView);
                backpropParamsSoFar += nParamsPerLayer[i];
            }
        }
    }


    /**
     * Triggers the activation of the last hidden layer ie: not logistic regression
     *
     * @return the activation of the last hidden layer given the last input to the network
     */
    public INDArray activate() {
        return getLayers()[getLayers().length - 1].activate();
    }

    /**
     * Triggers the activation for a given layer
     *
     * @param layer the layer to activate on
     * @return the activation for a given layer
     */
    public INDArray activate(int layer) {
        return getLayer(layer).activate();
    }

    @Override
    public INDArray activate(INDArray input) {
        throw new UnsupportedOperationException();
    }

    /**
     * Triggers the activation of the given layer
     *
     * @param layer the layer to trigger on
     * @param input the input to the hidden layer
     * @return the activation of the layer based on the input
     */
    public INDArray activate(int layer, INDArray input) {
        return getLayer(layer).activate(input);
    }

    @Override
    public INDArray activationMean() {
        //TODO determine how to pass back all activationMean for MLN
        throw new UnsupportedOperationException();
        //        List<INDArray> avgActivations =  new ArrayList<>();
        //
        //        for( Layer layer: getLayers() ){
        //            avgActivations.add(layer.activationMean());
        //            }
        //        return Nd4j.toFlattened(avgActivations);
    }

    /**
     * Sets the input and labels from this dataset
     *
     * @param data the dataset to initialize with
     */
    public void initialize(DataSet data) {
        setInput(data.getFeatureMatrix());
        feedForward(getInput());
        this.labels = data.getLabels();
        if (getOutputLayer() instanceof IOutputLayer) {
            IOutputLayer ol = (IOutputLayer) getOutputLayer();
            ol.setLabels(labels);
        }
    }


    /**
     * Compute input linear transformation (z) from previous layer
     * Apply pre processing transformation where necessary
     *
     * @param curr  the current layer
     * @param input the input
     * @param training training or test mode
     * @return the activation from the previous layer
     */
    public INDArray zFromPrevLayer(int curr, INDArray input, boolean training) {
        if (getLayerWiseConfigurations().getInputPreProcess(curr) != null)
            input = getLayerWiseConfigurations().getInputPreProcess(curr).preProcess(input, input.size(0));

        INDArray ret = layers[curr].preOutput(input, training);
        return ret;
    }

    /**
     * Calculate activation from previous layer including pre processing where necessary
     *
     * @param curr  the current layer
     * @param input the input
     * @return the activation from the previous layer
     */
    public INDArray activationFromPrevLayer(int curr, INDArray input, boolean training) {
        if (getLayerWiseConfigurations().getInputPreProcess(curr) != null)
            input = getLayerWiseConfigurations().getInputPreProcess(curr).preProcess(input, getInputMiniBatchSize());
        INDArray ret = layers[curr].activate(input, training);
        return ret;
    }

    /**
     * Calculate activation for few layers at once. Suitable for autoencoder partial activation.
     *
     * In example: in 10-layer deep autoencoder, layers 0 - 4 inclusive are used for encoding part, and layers 5-9 inclusive are used for decoding part.
     *
     * @param from first layer to be activated, inclusive
     * @param to last layer to be activated, inclusive
     * @return the activation from the last layer
     */
    public INDArray activateSelectedLayers(int from, int to, INDArray input) {
        if (input == null)
            throw new IllegalStateException("Unable to perform activation; no input found");
        if (from < 0 || from >= layers.length || from >= to)
            throw new IllegalStateException("Unable to perform activation; FROM is out of layer space");
        if (to < 1 || to >= layers.length)
            throw new IllegalStateException("Unable to perform activation; TO is out of layer space");

        INDArray res = input;
        for (int l = from; l <= to; l++) {
            res = this.activationFromPrevLayer(l, res, false);
        }
        return res;
    }

    /**
     * * Compute input linear transformation (z) of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> computeZ(boolean training) {
        INDArray currentInput = this.input;
        INDArray currentZ;

        List<INDArray> activations = new ArrayList<>();
        activations.add(currentInput);

        for (int i = 0; i < layers.length; i++) {
            //It's inefficient, but we do need to do forward pass twice, as some layers (like LSTMs)
            // don't decompose into out = activationFn(preOut)
            currentZ = zFromPrevLayer(i, currentInput, training);
            currentInput = activationFromPrevLayer(i, currentInput, training);
            activations.add(currentZ);
        }
        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> computeZ(INDArray input, boolean training) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        else if (this.getLayerWiseConfigurations().getInputPreProcess(0) != null)
            setInput(getLayerWiseConfigurations().getInputPreProcess(0).preProcess(input, getInputMiniBatchSize()));
        else
            setInput(input);
        return computeZ(training);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(INDArray input, boolean train) {
        setInput(input);
        return feedForward(train);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(boolean train) {
        return feedForwardToLayer(layers.length - 1, train);
    }

    /** Compute the activations from the input to the specified layer.<br>
     * To compute activations for all layers, use feedForward(...) methods<br>
     * Note: output list includes the original input. So list.get(0) is always the original input, and
     * list.get(i+1) is the activations of the ith layer.
     * @param layerNum Index of the last layer to calculate activations for. Layers are zero-indexed.
     *                 feedForwardToLayer(i,input) will return the activations for layers 0..i (inclusive)
     * @param input Input to the network
     * @return list of activations.
     */
    public List<INDArray> feedForwardToLayer(int layerNum, INDArray input) {
        return feedForwardToLayer(layerNum, input, false);
    }

    /** Compute the activations from the input to the specified layer.<br>
     * To compute activations for all layers, use feedForward(...) methods<br>
     * Note: output list includes the original input. So list.get(0) is always the original input, and
     * list.get(i+1) is the activations of the ith layer.
     * @param layerNum Index of the last layer to calculate activations for. Layers are zero-indexed.
     *                 feedForwardToLayer(i,input) will return the activations for layers 0..i (inclusive)
     * @param input Input to the network
     * @param train true for training, false for test (i.e., false if using network after training)
     * @return list of activations.
     */
    public List<INDArray> feedForwardToLayer(int layerNum, INDArray input, boolean train) {
        setInput(input);
        return feedForwardToLayer(layerNum, train);
    }

    /** Compute the activations from the input to the specified layer, using the currently set input for the network.<br>
     * To compute activations for all layers, use feedForward(...) methods<br>
     * Note: output list includes the original input. So list.get(0) is always the original input, and
     * list.get(i+1) is the activations of the ith layer.
     * @param layerNum Index of the last layer to calculate activations for. Layers are zero-indexed.
     *                 feedForwardToLayer(i,input) will return the activations for layers 0..i (inclusive)
     * @param train true for training, false for test (i.e., false if using network after training)
     * @return list of activations.
     */
    public List<INDArray> feedForwardToLayer(int layerNum, boolean train) {
        // TODO: maybe remove that?
        INDArray currInput =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE || !input.isAttached()
                                        ? input : input.migrate();
        List<INDArray> activations = new ArrayList<>();
        activations.add(currInput);


        MemoryWorkspace workspace = layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE
                        ? new DummyWorkspace()
                        : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                                        ? Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceExternal)
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationFeedForward, workspaceFeedForward);

        for (int i = 0; i <= layerNum; i++) {
            // log.info("Activating layer: {}", i);
            try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                currInput = activationFromPrevLayer(i, currInput, train).leverageTo(workspaceExternal);
                //currInput = activationFromPrevLayer(i, currInput, train);
                //applies drop connect to the activation
                activations.add(currInput);
            }
        }

        if (!train)
            if (layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SEPARATE)
                Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceFeedForward).initializeWorkspace();

        return activations;
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward() {
        return feedForward(false);
    }

    /**
     * Compute activations from input to output of the output layer
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(INDArray input) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        else if (this.getLayerWiseConfigurations().getInputPreProcess(0) != null)
            setInput(getLayerWiseConfigurations().getInputPreProcess(0).preProcess(input, input.size(0)));
        else
            setInput(input);
        return feedForward();
    }

    /** Compute the activations from the input to the output layer, given mask arrays (that may be null)
     * The masking arrays are used in situations such an one-to-many and many-to-one rucerrent neural network (RNN)
     * designs, as well as for supporting time series of varying lengths within the same minibatch for RNNs.
     */
    public List<INDArray> feedForward(INDArray input, INDArray featuresMask, INDArray labelsMask) {
        setLayerMaskArrays(featuresMask, labelsMask);
        List<INDArray> list = feedForward(input);
        clearLayerMaskArrays();
        return list;
    }


    @Override
    public Gradient gradient() {
        return gradient;
    }

    public INDArray epsilon() {
        return epsilon;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }


    @Override
    public MultiLayerNetwork clone() {
        MultiLayerConfiguration conf = this.layerWiseConfigurations.clone();
        MultiLayerNetwork ret = new MultiLayerNetwork(conf);
        ret.init(this.params().dup(), false);

        if (solver != null) {
            //If  solver is null: updater hasn't been initialized -> getUpdater call will force initialization, however
            Updater u = this.getUpdater();
            INDArray updaterState = u.getStateViewArray();
            if (updaterState != null) {
                ret.getUpdater().setStateViewArray(ret, updaterState.dup(), false);
            }
        }

        if (hasAFrozenLayer()) {
            //correct layers to frozen layers
            Layer[] clonedLayers = ret.getLayers();
            for (int i = 0; i < layers.length; i++) {
                if (layers[i] instanceof FrozenLayer) {
                    clonedLayers[i] = new FrozenLayer(ret.getLayer(i));
                }
            }
            ret.setLayers(clonedLayers);
        }
        return ret;
    }

    private boolean hasAFrozenLayer() {
        for (int i = 0; i < layers.length - 1; i++) {
            if (layers[i] instanceof FrozenLayer)
                return true;
        }
        return false;
    }


    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various neuralNets(w,hbias NOT VBIAS) and output layer
     *
     * @return the params for this neural net
     */
    public INDArray params(boolean backwardOnly) {
        if (backwardOnly)
            return params();

        List<INDArray> params = new ArrayList<>();
        for (Layer layer : getLayers()) {
            INDArray layerParams = layer.params();
            if (layerParams != null)
                params.add(layerParams); //may be null: subsampling etc layers
        }

        return Nd4j.toFlattened('f', params);
    }


    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various neuralNets(w,hbias NOT VBIAS) and output layer
     *
     * @return the params for this neural net
     */
    @Override
    public INDArray params() {
        return flattenedParams;
    }

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray
     * which then be unpacked internally
     * relative to the expected ordering of the model
     *
     * @param params the parameters for the model
     */
    @Override
    public void setParams(INDArray params) {
        if (flattenedParams == params) {
            return; //No op
        }

        if (flattenedParams != null && params.length() == flattenedParams.length()) {
            if (params != flattenedParams) {
                flattenedParams.assign(params);
            }
        } else {
            if (flattenedParams == null)
                flattenedParams = params.dup();
            int idx = 0;
            for (int i = 0; i < getLayers().length; i++) {
                Layer layer = getLayer(i);
                int range = layer.numParams();
                if (range <= 0)
                    continue; //Some layers: no parameters (subsampling, etc)
                INDArray get = params.get(NDArrayIndex.point(0), NDArrayIndex.interval(idx, range + idx));
                layer.setParams(get);
                idx += range;
            }
        }
    }

    @Override
    public void setParamsViewArray(INDArray params) {
        throw new UnsupportedOperationException("Not yet implemented");
    }

    @Override
    public INDArray getGradientsViewArray() {
        return flattenedGradients;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradients) {
        int paramsSoFar = 0;
        for (Layer layer : layers) {
            if (layer.numParams() == 0)
                continue;
            layer.setBackpropGradientsViewArray(gradients.get(NDArrayIndex.point(0),
                            NDArrayIndex.interval(paramsSoFar, paramsSoFar + layer.numParams())));
            paramsSoFar += layer.numParams();
        }
    }

    /**
     * Returns a 1 x m vector where the vector is composed of
     * a flattened vector of all of the weights for the
     * various neuralNets and output layer
     *
     * @return the params for this neural net
     */
    @Override
    public int numParams() {
        if (isInitCalled())
            return numParams(false);
        else
            log.info("Model is not initialized. Initialize net with init()");
        return 0;
    }

    @Override
    public int numParams(boolean backwards) {
        int length = 0;
        for (int i = 0; i < layers.length; i++)
            length += layers[i].numParams(backwards);

        return length;
    }

    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(org.nd4j.linalg.dataset.api.DataSet data) {
        return f1Score(data.getFeatures(), data.getLabels());
    }

    @Override
    public void fit(DataSetIterator iterator) {
        // we're wrapping all iterators into AsyncDataSetIterator to provide background prefetch - where appropriate
        DataSetIterator iter;
        boolean destructable = false;
        if (iterator.asyncSupported()) {
            iter = new AsyncDataSetIterator(iterator, Math.min(Nd4j.getAffinityManager().getNumberOfDevices() * 2, 2),
                            layerWiseConfigurations.getTrainingWorkspaceMode() != WorkspaceMode.NONE);
            destructable = true;
        } else {
            iter = iterator;
        }

        for (TrainingListener tl : trainingListeners) {
            tl.onEpochStart(this);
        }

        if (layerWiseConfigurations.isPretrain()) {
            pretrain(iter);
            if (iter.resetSupported()) {
                iter.reset();
            }
            //            while (iter.hasNext()) {
            //                DataSet next = iter.next();
            //                if (next.getFeatureMatrix() == null || next.getLabels() == null)
            //                    break;
            //                setInput(next.getFeatureMatrix());
            //                setLabels(next.getLabels());
            //                finetune();
            //            }
        }


        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);
        MemoryWorkspace cache =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        ComputationGraph.workspaceConfigurationCache,
                                                        ComputationGraph.workspaceCache);

        if (layerWiseConfigurations.isBackprop()) {
            update(TaskUtils.buildTask(iter));
            if (!iter.hasNext() && iter.resetSupported()) {
                iter.reset();
            }
            long time1 = System.currentTimeMillis();
            while (iter.hasNext()) {

                DataSet next = iter.next();
                long time2 = System.currentTimeMillis();

                lastEtlTime.set((time2 - time1));

                if (next.getFeatureMatrix() == null || next.getLabels() == null)
                    break;

                // TODO: basically we want to wrap internals of this loop into workspace


                boolean hasMaskArrays = next.hasMaskArrays();

                if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
                    doTruncatedBPTT(next.getFeatureMatrix(), next.getLabels(), next.getFeaturesMaskArray(),
                                    next.getLabelsMaskArray());
                } else {
                    if (hasMaskArrays)
                        setLayerMaskArrays(next.getFeaturesMaskArray(), next.getLabelsMaskArray());

                    setInput(next.getFeatureMatrix());
                    setLabels(next.getLabels());

                    if (solver == null) {
                        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                                            .build();
                        }
                    }

                    try (MemoryWorkspace wsCache = cache.notifyScopeEntered()) {
                        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                            solver.optimize();
                        }
                    }
                }

                if (hasMaskArrays)
                    clearLayerMaskArrays();

                time1 = System.currentTimeMillis();
            }
        } else if (layerWiseConfigurations.isPretrain()) {
            log.warn("Warning: finetune is not applied.");
        }

        if (trainingListeners.size() > 0) {
            for (TrainingListener tl : trainingListeners) {
                tl.onEpochEnd(this);
            }
        }

        clearLayersStates();

        if (destructable)
            ((AsyncDataSetIterator) iter).shutdown();
    }

    /** Calculate and set gradients for MultiLayerNetwork, based on OutputLayer and labels*/
    protected void backprop() {
        Pair<Gradient, INDArray> pair = calcBackpropGradients(null, true);
        this.gradient = (pair == null ? null : pair.getFirst());
        this.epsilon = (pair == null ? null : pair.getSecond());
    }

    /** Calculate gradients and errors. Used in two places:
     * (a) backprop (for standard multi layer network learning)
     * (b) backpropGradient (layer method, for when MultiLayerNetwork is used as a layer)
     * @param epsilon Errors (technically errors .* activations). Not used if withOutputLayer = true
     * @param withOutputLayer if true: assume last layer is output layer, and calculate errors based on labels. In this
     *                        case, the epsilon input is not used (may/should be null).
     *                        If false: calculate backprop gradients
     * @return Gradients and the error (epsilon) at the input
     */
    protected Pair<Gradient, INDArray> calcBackpropGradients(INDArray epsilon, boolean withOutputLayer) {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        String multiGradientKey;
        Gradient gradient = new DefaultGradient(flattenedGradients);
        Layer currLayer;



        //calculate and apply the backward gradient for every layer
        /**
         * Skip the output layer for the indexing and just loop backwards updating the coefficients for each layer.
         * (when withOutputLayer == true)
         *
         * Activate applies the activation function for each layer and sets that as the input for the following layer.
         *
         * Typical literature contains most trivial case for the error calculation: wT * weights
         * This interpretation transpose a few things to get mini batch because ND4J is rows vs columns organization for params
         */
        int numLayers = getnLayers();
        //Store gradients is a list; used to ensure iteration order in DefaultGradient linked hash map. i.e., layer 0 first instead of output layer
        LinkedList<Triple<String, INDArray, Character>> gradientList = new LinkedList<>();

        int layerFrom;
        Pair<Gradient, INDArray> currPair;
        if (withOutputLayer) {
            if (!(getOutputLayer() instanceof IOutputLayer)) {
                log.warn("Warning: final layer isn't output layer. You cannot use backprop without an output layer.");
                return null;
            }

            IOutputLayer outputLayer = (IOutputLayer) getOutputLayer();
            if (labels == null)
                throw new IllegalStateException("No labels found");
            outputLayer.setLabels(labels);
            currPair = outputLayer.backpropGradient(null);

            for (Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
                String origName = entry.getKey();
                multiGradientKey = String.valueOf(numLayers - 1) + "_" + origName;
                gradientList.addLast(new Triple<>(multiGradientKey, entry.getValue(),
                                currPair.getFirst().flatteningOrderForVariable(origName)));
            }
            if (getLayerWiseConfigurations().getInputPreProcess(numLayers - 1) != null)
                currPair = new Pair<>(currPair.getFirst(),
                                this.layerWiseConfigurations.getInputPreProcess(numLayers - 1)
                                                .backprop(currPair.getSecond(), getInputMiniBatchSize()));

            layerFrom = numLayers - 2;
        } else {
            currPair = new Pair<>(null, epsilon);
            layerFrom = numLayers - 1;
        }

        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SINGLE
                                                        ? Nd4j.getWorkspaceManager()
                                                                        .getWorkspaceForCurrentThread(workspaceExternal)
                                                        //: Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(wsConf, workspaceBackProp);
                                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                                        workspaceConfigurationFeedForward,
                                                                        workspaceFeedForward);

        // Calculate gradients for previous layers & drops output layer in count
        for (int j = layerFrom; j >= 0; j--) {
            try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                currLayer = getLayer(j);
                if (currLayer instanceof FrozenLayer) {
                    break;
                }
                currPair = currLayer.backpropGradient(currPair.getSecond());
                if (currPair.getSecond() != null) {
                    //May be null for embedding layer, etc
                    currPair.setSecond(currPair.getSecond().leverageTo(workspaceExternal));
                }

                LinkedList<Triple<String, INDArray, Character>> tempList = new LinkedList<>();
                for (Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
                    String origName = entry.getKey();
                    multiGradientKey = String.valueOf(j) + "_" + origName;
                    tempList.addFirst(new Triple<>(multiGradientKey, entry.getValue(),
                                    currPair.getFirst().flatteningOrderForVariable(origName)));
                }
                for (Triple<String, INDArray, Character> triple : tempList)
                    gradientList.addFirst(triple);

                //Pass epsilon through input processor before passing to next layer (if applicable)
                if (getLayerWiseConfigurations().getInputPreProcess(j) != null)
                    currPair = new Pair<>(currPair.getFirst(), getLayerWiseConfigurations().getInputPreProcess(j)
                                    .backprop(currPair.getSecond(), getInputMiniBatchSize()));

                //log.info("This layer space: {}", ((Nd4jWorkspace) ws).getThisCycleAllocations());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        if (layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.SEPARATE) {
            Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(workspaceFeedForward).initializeWorkspace();
        }

        //Add gradients to Gradients (map), in correct order
        for (Triple<String, INDArray, Character> triple : gradientList) {
            gradient.setGradientFor(triple.getFirst(), triple.getSecond(), triple.getThird());
        }

        return new Pair<>(gradient, currPair.getSecond());
    }

    protected void doTruncatedBPTT(INDArray input, INDArray labels, INDArray featuresMaskArray,
                    INDArray labelsMaskArray) {
        if (input.rank() != 3 || labels.rank() != 3) {
            log.warn("Cannot do truncated BPTT with non-3d inputs or labels. Expect input with shape [miniBatchSize,nIn,timeSeriesLength], got "
                            + Arrays.toString(input.shape()) + "\tand labels with shape "
                            + Arrays.toString(labels.shape()));
            return;
        }
        if (input.size(2) != labels.size(2)) {
            log.warn("Input and label time series have different lengths: {} input length, {} label length",
                            input.size(2), labels.size(2));
            return;
        }

        int fwdLen = layerWiseConfigurations.getTbpttFwdLength();
        update(TaskUtils.buildTask(input, labels));
        int timeSeriesLength = input.size(2);
        int nSubsets = timeSeriesLength / fwdLen;
        if (timeSeriesLength % fwdLen != 0)
            nSubsets++; //Example: 100 fwdLen with timeSeriesLength=100 -> want 2 subsets (1 of size 100, 1 of size 20)

        rnnClearPreviousState();

        workspaceConfigurationExternal.setCyclesBeforeInitialization(0);
        workspaceConfigurationExternal.setPolicyLearning(LearningPolicy.OVER_TIME);

        MemoryWorkspace workspaceT =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationTBPTT, workspaceTBPTT);
        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);

        try (MemoryWorkspace wsT = workspaceT.notifyScopeEntered()) {
            for (int i = 0; i < nSubsets; i++) {
                try (MemoryWorkspace wsE = workspace.notifyScopeEntered()) {
                    int startTimeIdx = i * fwdLen;
                    int endTimeIdx = startTimeIdx + fwdLen;
                    if (endTimeIdx > timeSeriesLength)
                        endTimeIdx = timeSeriesLength;

                    INDArray inputSubset = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));
                    INDArray labelSubset = labels.get(NDArrayIndex.all(), NDArrayIndex.all(),
                                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));

                    setInput(inputSubset);
                    setLabels(labelSubset);

                    INDArray featuresMaskSubset = null;
                    INDArray labelsMaskSubset = null;
                    if (featuresMaskArray != null) {
                        featuresMaskSubset = featuresMaskArray.get(NDArrayIndex.all(),
                                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
                    }
                    if (labelsMaskArray != null) {
                        labelsMaskSubset = labelsMaskArray.get(NDArrayIndex.all(),
                                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
                    }
                    if (featuresMaskSubset != null || labelsMaskSubset != null)
                        setLayerMaskArrays(featuresMaskSubset, labelsMaskSubset);

                    if (solver == null) {
                        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                                            .build();
                        }
                    }
                    solver.optimize();

                    //Finally, update the state of the RNN layers:
                    updateRnnStateWithTBPTTState();
                }
            }
        }

        if (layerWiseConfigurations.getTrainingWorkspaceMode() != WorkspaceMode.NONE) {
            workspace.initializeWorkspace();
            workspaceT.initializeWorkspace();
        }

        rnnClearPreviousState();
        if (featuresMaskArray != null || labelsMaskArray != null)
            clearLayerMaskArrays();
    }

    public void updateRnnStateWithTBPTTState() {
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof RecurrentLayer) {
                RecurrentLayer l = ((RecurrentLayer) layers[i]);
                l.rnnSetPreviousState(l.rnnGetTBPTTState());
            } else if (layers[i] instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) layers[i]).updateRnnStateWithTBPTTState();
            }
        }
    }

    /** Equivalent to backprop(), but calculates gradient for truncated BPTT instead. */
    protected void truncatedBPTTGradient() {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        String multiGradientKey;
        gradient = new DefaultGradient(flattenedGradients);
        Layer currLayer;

        if (!(getOutputLayer() instanceof IOutputLayer)) {
            log.warn("Warning: final layer isn't output layer. You cannot use backprop (truncated BPTT) without an output layer.");
            return;
        }

        IOutputLayer outputLayer = (IOutputLayer) getOutputLayer();
        if (labels == null)
            throw new IllegalStateException("No labels found");
        if (outputLayer instanceof BaseLayer
                        && ((BaseLayer) outputLayer.conf().getLayer()).getWeightInit() == WeightInit.ZERO) {
            throw new IllegalStateException("Output layer weights cannot be initialized to zero when using backprop.");
        }

        outputLayer.setLabels(labels);

        //calculate and apply the backward gradient for every layer
        int numLayers = getnLayers();
        //Store gradients is a list; used to ensure iteration order in DefaultGradient linked hash map. i.e., layer 0 first instead of output layer
        LinkedList<Pair<String, INDArray>> gradientList = new LinkedList<>();

        Pair<Gradient, INDArray> currPair = outputLayer.backpropGradient(null);

        for (Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
            multiGradientKey = String.valueOf(numLayers - 1) + "_" + entry.getKey();
            gradientList.addLast(new Pair<>(multiGradientKey, entry.getValue()));
        }

        if (getLayerWiseConfigurations().getInputPreProcess(numLayers - 1) != null)
            currPair = new Pair<>(currPair.getFirst(), this.layerWiseConfigurations.getInputPreProcess(numLayers - 1)
                            .backprop(currPair.getSecond(), getInputMiniBatchSize()));

        // Calculate gradients for previous layers & drops output layer in count
        for (int j = numLayers - 2; j >= 0; j--) {
            currLayer = getLayer(j);
            if (currLayer instanceof RecurrentLayer) {
                currPair = ((RecurrentLayer) currLayer).tbpttBackpropGradient(currPair.getSecond(),
                                layerWiseConfigurations.getTbpttBackLength());
            } else {
                currPair = currLayer.backpropGradient(currPair.getSecond());
            }

            LinkedList<Pair<String, INDArray>> tempList = new LinkedList<>();
            for (Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
                multiGradientKey = String.valueOf(j) + "_" + entry.getKey();
                tempList.addFirst(new Pair<>(multiGradientKey, entry.getValue()));
            }

            for (Pair<String, INDArray> pair : tempList)
                gradientList.addFirst(pair);

            //Pass epsilon through input processor before passing to next layer (if applicable)
            if (getLayerWiseConfigurations().getInputPreProcess(j) != null)
                currPair = new Pair<>(currPair.getFirst(), getLayerWiseConfigurations().getInputPreProcess(j)
                                .backprop(currPair.getSecond(), getInputMiniBatchSize()));
        }

        //Add gradients to Gradients, in correct order
        for (Pair<String, INDArray> pair : gradientList)
            gradient.setGradientFor(pair.getFirst(), pair.getSecond());
    }


    /**
     *
     * @return
     */
    public Collection<IterationListener> getListeners() {
        return listeners;
    }

    @Override
    public void setListeners(Collection<IterationListener> listeners) {
        this.listeners = listeners;

        if (layers == null) {
            init();
        }
        for (Layer layer : layers) {
            layer.setListeners(listeners);
        }

        if (solver != null) {
            solver.setListeners(listeners);
        }

        this.trainingListeners.clear();
        if (listeners != null) {
            for (IterationListener il : listeners) {
                if (il instanceof TrainingListener) {
                    this.trainingListeners.add((TrainingListener) il);
                }
            }
        }
    }

    /**
     * This method ADDS additional IterationListener to existing listeners
     *
     * @param listeners
     */
    @Override
    public void addListeners(IterationListener... listeners) {
        if (this.listeners == null) {
            setListeners(listeners);
            return;
        }

        for (IterationListener listener : listeners) {
            this.listeners.add(listener);
            if (listener instanceof TrainingListener) {
                this.trainingListeners.add((TrainingListener) listener);
            }
        }

        if (solver != null) {
            solver.setListeners(this.listeners);
        }
    }

    @Override
    public void setListeners(IterationListener... listeners) {
        Collection<IterationListener> cListeners = new ArrayList<>();
        //Check: user might have done setListeners(null) thinking this would clear the current listeners.
        //This results in an IterationListener[1] with a single null value -> results in a NPE later
        if (listeners != null && listeners.length > 0) {
            for (IterationListener i : listeners) {
                if (i != null)
                    cListeners.add(i);
            }
        }
        setListeners(cListeners);
    }


    /**
     * Run SGD based on the given labels
     */
    public void finetune() {
        if (!layerWiseConfigurations.isBackprop()) {
            log.warn("Warning: finetune is not applied.");
            return;
        }
        if (!(getOutputLayer() instanceof IOutputLayer)) {
            log.warn("Output layer not instance of output layer returning.");
            return;
        }
        if (flattenedGradients == null) {
            initGradientsView();
        }

        if (labels == null)
            throw new IllegalStateException("No labels found");

        log.info("Finetune phase");
        IOutputLayer output = (IOutputLayer) getOutputLayer();
        if (output.conf().getOptimizationAlgo() != OptimizationAlgorithm.HESSIAN_FREE) {
            feedForward();
            output.fit(output.input(), labels);
        } else {
            throw new UnsupportedOperationException();
        }
    }


    /**
     * Returns the predictions for each example in the dataset
     *
     * @param d the matrix to predict
     * @return the prediction for the dataset
     */
    @Override
    public int[] predict(INDArray d) {
        INDArray output = output(d, Layer.TrainingMode.TEST);
        int[] ret = new int[d.size(0)];
        if (d.isRowVector())
            ret[0] = Nd4j.getBlasWrapper().iamax(output);
        else {
            for (int i = 0; i < ret.length; i++)
                ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        }
        return ret;
    }

    /**
     * Return predicted label names
     *
     * @param dataSet to predict
     * @return the predicted labels for the dataSet
     */
    @Override
    public List<String> predict(org.nd4j.linalg.dataset.api.DataSet dataSet) {
        int[] intRet = predict(dataSet.getFeatures());
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < intRet.length; i++) {
            ret.add(i, dataSet.getLabelName(intRet[i]));
        }
        return ret;
    }



    /**
     * Returns the probabilities for each label
     * for each example row wise
     *
     * @param examples the examples to classify (one example in each row)
     * @return the likelihoods of each example and each label
     */
    @Override
    public INDArray labelProbabilities(INDArray examples) {
        List<INDArray> feed = feedForward(examples);
        IOutputLayer o = (IOutputLayer) getOutputLayer();
        return o.labelProbabilities(feed.get(feed.size() - 1));
    }

    /**
     * Fit the model
     *
     * @param data   the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray data, INDArray labels) {
        fit(data, labels, null, null);
    }

    /**
     * Fit the model
     *
     * @param features   the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     * @param featuresMask The mask array for the features (used for variable length time series, etc). May be null.
     * @param labelsMask The mask array for the labels (used for variable length time series, etc). May be null.
     */
    public void fit(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask) {

        setInput(features);
        setLabels(labels);
        if (featuresMask != null || labelsMask != null) {
            this.setLayerMaskArrays(featuresMask, labelsMask);
        }
        update(TaskUtils.buildTask(features, labels));

        MemoryWorkspace workspace =
                layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        workspaceConfigurationExternal, workspaceExternal);

        MemoryWorkspace cache =
                layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                        ComputationGraph.workspaceConfigurationCache,
                        ComputationGraph.workspaceCache);

        if (layerWiseConfigurations.isPretrain()) {
            try (MemoryWorkspace wsCache = cache.notifyScopeEntered()) {
                try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                    pretrain(features);
                }
            }
        }

        if (layerWiseConfigurations.isBackprop()) {
            if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
                doTruncatedBPTT(features, labels, featuresMask, labelsMask);
            } else {
                if (solver == null) {
                    try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                    }
                }

                try (MemoryWorkspace wsCache = cache.notifyScopeEntered()) {
                    try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
                        solver.optimize();
                    }
                }
            }
        }

        if (featuresMask != null || labelsMask != null) {
            clearLayerMaskArrays();
        }

        clearLayersStates();
    }

    /**
     * Fit the unsupervised model
     *
     * @param data the examples to classify (one example in each row)
     */

    @Override
    public void fit(INDArray data) {
        setInput(data);
        if (!layerWiseConfigurations.isPretrain())
            throw new IllegalStateException(
                            "Set pretrain to true in the configuration in order to pretrain the model.");
        update(TaskUtils.buildTask(data));
        pretrain(data);
    }

    @Override
    public void iterate(INDArray input) {
        pretrain(input);
    }


    /**
     * Fit the model
     *
     * @param data the data to train on
     */
    @Override
    public void fit(org.nd4j.linalg.dataset.api.DataSet data) {
        if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {

            doTruncatedBPTT(data.getFeatures(), data.getLabels(), data.getFeaturesMaskArray(),
                            data.getLabelsMaskArray());

        } else {
            //Standard training
            boolean hasMaskArrays = data.hasMaskArrays();
            if (hasMaskArrays)
                setLayerMaskArrays(data.getFeaturesMaskArray(), data.getLabelsMaskArray());
            fit(data.getFeatures(), data.getLabels());
            if (hasMaskArrays)
                clearLayerMaskArrays();

        }

        clearLayersStates();
    }

    /**
     * Fit the model
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        org.deeplearning4j.nn.conf.layers.OutputLayer layerConf =
                        (org.deeplearning4j.nn.conf.layers.OutputLayer) getOutputLayer().conf().getLayer();
        fit(examples, FeatureUtil.toOutcomeMatrix(labels, layerConf.getNOut()));
    }


    /**
     * Label the probabilities of the input
     *
     * @param input    the input to label
     * @param train whether the output
     *             is test or train. This mainly
     *             affect hyper parameters such as
     *             drop out where certain things should
     *             be applied with activations
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray input, TrainingMode train) {
        return output(input, train == TrainingMode.TRAIN);
    }

    /**
     * Label the probabilities of the input
     *
     * @param input    the input to label
     * @param train whether the output
     *             is test or train. This mainly
     *             affect hyper parameters such as
     *             drop out where certain things should
     *             be applied with activations
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray input, boolean train) {
        WorkspaceMode cMode = layerWiseConfigurations.getTrainingWorkspaceMode();
        layerWiseConfigurations.setTrainingWorkspaceMode(layerWiseConfigurations.getInferenceWorkspaceMode());
        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);

        try (MemoryWorkspace wsE = workspace.notifyScopeEntered()) {
            INDArray ret = silentOutput(input, train).detach();

            layerWiseConfigurations.setTrainingWorkspaceMode(cMode);
            return ret;
        }
    }

    protected INDArray silentOutput(INDArray input, boolean train) {
        List<INDArray> activations = feedForward(input, train);

        //last activation is output
        return activations.get(activations.size() - 1);
    }

    /** Calculate the output of the network, with masking arrays. The masking arrays are used in situations such
     * as one-to-many and many-to-one recurrent neural network (RNN) designs, as well as for supporting time series
     * of varying lengths within the same minibatch.
     */
    public INDArray output(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask) {
        WorkspaceMode cMode = layerWiseConfigurations.getTrainingWorkspaceMode();
        layerWiseConfigurations.setTrainingWorkspaceMode(layerWiseConfigurations.getInferenceWorkspaceMode());
        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);

        try (MemoryWorkspace wsE = workspace.notifyScopeEntered()) {
            INDArray ret = silentOutput(input, train, featuresMask, labelsMask).detach();

            layerWiseConfigurations.setTrainingWorkspaceMode(cMode);
            return ret;
        }
    }

    protected INDArray silentOutput(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask) {


        setLayerMaskArrays(featuresMask, labelsMask);
        INDArray out = silentOutput(input, train);
        clearLayerMaskArrays();
        return out;
    }

    /**
     * Label the probabilities of the input
     *
     * @param input the input to label
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(INDArray input) {
        return output(input, TrainingMode.TEST);
    }

    /**
     * Label the probabilities of the input
     *
     * @param iterator test data to evaluate
     * @return a vector of probabilities
     * given each label.
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray output(DataSetIterator iterator, boolean train) {
        List<INDArray> outList = new ArrayList<>();
        while (iterator.hasNext()) {
            DataSet next = iterator.next();

            if (next.getFeatureMatrix() == null || next.getLabels() == null)
                break;

            INDArray features = next.getFeatures();

            if (next.hasMaskArrays()) {
                INDArray fMask = next.getFeaturesMaskArray();
                INDArray lMask = next.getLabelsMaskArray();
                outList.add(this.output(features, train, fMask, lMask));

            } else {
                outList.add(output(features, train));
            }
        }
        return Nd4j.vstack(outList.toArray(new INDArray[0]));
    }

    public INDArray output(DataSetIterator iterator) {
        return output(iterator, false);
    }


    /**
     * Reconstructs the input.
     * This is equivalent functionality to a
     * deep autoencoder.
     *
     * @param x        the input to transform
     * @param layerNum the layer to output for encoding
     * @return a reconstructed matrix
     * relative to the size of the last hidden layer.
     * This is great for data compression and visualizing
     * high dimensional data (or just doing dimensionality reduction).
     * <p>
     * This is typically of the form:
     * [0.5, 0.5] or some other probability distribution summing to one
     */
    public INDArray reconstruct(INDArray x, int layerNum) {
        List<INDArray> forward = feedForward(x);
        return forward.get(layerNum - 1);
    }


    /**
     * Prints the configuration
     */
    public void printConfiguration() {
        StringBuilder sb = new StringBuilder();
        int count = 0;
        for (NeuralNetConfiguration conf : getLayerWiseConfigurations().getConfs()) {
            sb.append(" Layer " + count++ + " conf " + conf);
        }

        log.info(sb.toString());
    }


    /**
     * Assigns the parameters of this model to the ones specified by this
     * network. This is used in loading from input streams, factory methods, etc
     *
     * @param network the network to getFromOrigin parameters from
     */
    public void update(MultiLayerNetwork network) {
        this.defaultConfiguration =
                        (network.defaultConfiguration != null ? network.defaultConfiguration.clone() : null);
        if (network.input != null)
            setInput(network.input.dup()); //Dup in case of dropout etc
        this.labels = network.labels;
        if (network.layers != null) {
            layers = new Layer[network.layers.length];
            for (int i = 0; i < layers.length; i++) {
                layers[i] = network.layers[i].clone();
            }
        } else {
            this.layers = null;
        }
        if (network.solver != null) {
            //Network updater state: should be cloned over also
            INDArray updaterView = network.getUpdater().getStateViewArray();
            if (updaterView != null) {
                //                Updater newUpdater = new MultiLayerUpdater(this, updaterView.dup());
                Updater newUpdater = new MultiLayerUpdater(this);
                newUpdater.setStateViewArray(this, updaterView.dup(), false);
                this.setUpdater(newUpdater);
            }
        } else {
            this.solver = null;
        }
    }


    /**
     * Sets the input and labels and returns a score for the prediction
     * wrt true labels
     *
     * @param input  the input to score
     * @param labels the true labels
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(INDArray input, INDArray labels) {
        feedForward(input);
        setLabels(labels);
        Evaluation eval = new Evaluation();
        eval.eval(labels, labelProbabilities(input));
        return eval.f1();
    }

    /**
     * Returns the number of possible labels
     *
     * @return the number of possible labels for this classifier
     */
    @Override
    public int numLabels() {
        return labels.columns();
    }

    /**Sets the input and labels and returns a score for the prediction with respect to the true labels<br>
     * This is equivalent to {@link #score(DataSet, boolean)} with training==true.
     * @param data the data to score
     * @return the score for the given input,label pairs
     * @see #score(DataSet, boolean)
     */
    public double score(DataSet data) {
        return score(data, false);
    }

    /**Calculate the score (loss function) of the prediction with respect to the true labels<br>
     * @param data data to calculate score for
     * @param training If true: score during training. If false: score at test time. This can affect the application of
     *                 certain features, such as dropout and dropconnect (which are applied at training time only)
     * @return the score (value of the loss function)
     */
    public double score(DataSet data, boolean training) {
        boolean hasMaskArray = data.hasMaskArrays();
        if (hasMaskArray)
            setLayerMaskArrays(data.getFeaturesMaskArray(), data.getLabelsMaskArray());

        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);

        try (MemoryWorkspace ws = workspace.notifyScopeEntered()) {
            // activation for output layer is calculated in computeScore
            List<INDArray> activations = feedForwardToLayer(layers.length - 2, data.getFeatureMatrix(), training);
            int n = activations.size();
            setLabels(data.getLabels());
            if (getOutputLayer() instanceof IOutputLayer) {
                IOutputLayer ol = (IOutputLayer) getOutputLayer();
                INDArray olInput = activations.get(n - 1);
                if (getLayerWiseConfigurations().getInputPreProcess(n - 1) != null) {
                    olInput = getLayerWiseConfigurations().getInputPreProcess(n - 1).preProcess(olInput, input.size(0));
                }
                ol.setInput(olInput); //Feedforward doesn't include output layer for efficiency
                ol.setLabels(data.getLabels());
                ol.computeScore(calcL1(true), calcL2(true), training);
                this.score = ol.score();
            } else {
                log.warn("Cannot calculate score wrt labels without an OutputLayer");
                return 0.0;
            }
        }

        if (hasMaskArray)
            clearLayerMaskArrays();

        return score();
    }

    public INDArray scoreExamples(DataSetIterator iter, boolean addRegularizationTerms) {
        List<INDArray> out = new ArrayList<>();

        while (iter.hasNext()) {
            out.add(scoreExamples(iter.next(), addRegularizationTerms));
        }
        return Nd4j.toFlattened('f', out);
    }

    /**Calculate the score for each example in a DataSet individually. Unlike {@link #score(DataSet)} and {@link #score(DataSet, boolean)}
     * this method does not average/sum over examples. This method allows for examples to be scored individually (at test time only), which
     * may be useful for example for autoencoder architectures and the like.<br>
     * Each row of the output (assuming addRegularizationTerms == true) is equivalent to calling score(DataSet) with a single example.
     * @param data The data to score
     * @param addRegularizationTerms If true: add l1/l2 regularization terms (if any) to the score. If false: don't add regularization terms
     * @return An INDArray (column vector) of size input.numRows(); the ith entry is the score (loss value) of the ith example
     */
    public INDArray scoreExamples(DataSet data, boolean addRegularizationTerms) {
        boolean hasMaskArray = data.hasMaskArrays();
        if (hasMaskArray)
            setLayerMaskArrays(data.getFeaturesMaskArray(), data.getLabelsMaskArray());
        feedForward(data.getFeatureMatrix(), false);
        setLabels(data.getLabels());

        INDArray out;
        if (getOutputLayer() instanceof IOutputLayer) {
            IOutputLayer ol = (IOutputLayer) getOutputLayer();
            ol.setLabels(data.getLabels());
            double l1 = (addRegularizationTerms ? calcL1(true) : 0.0);
            double l2 = (addRegularizationTerms ? calcL2(true) : 0.0);
            out = ol.computeScoreForExamples(l1, l2);
        } else {
            throw new UnsupportedOperationException(
                            "Cannot calculate score with respect to labels without an OutputLayer");
        }
        if (hasMaskArray)
            clearLayerMaskArrays();
        return out;
    }


    @Override
    public void fit() {
        fit(input, labels);
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not implemented");
    }


    /**
     * Score of the model (relative to the objective function)
     *
     * @return the score of the model (relative to the objective function)
     */
    @Override
    public double score() {
        return score;
    }


    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public void computeGradientAndScore() {
        //Calculate activations (which are stored in each layer, and used in backprop)
        if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
            List<INDArray> activations = rnnActivateUsingStoredState(getInput(), true, true);
            if (trainingListeners.size() > 0) {
                for (TrainingListener tl : trainingListeners) {
                    tl.onForwardPass(this, activations);
                }
            }
            truncatedBPTTGradient();
        } else {
            //First: do a feed-forward through the network
            //Note that we don't actually need to do the full forward pass through the output layer right now; but we do
            // need the input to the output layer to be set (such that backprop can be done)
            List<INDArray> activations = feedForwardToLayer(layers.length - 2, true);
            if (trainingListeners.size() > 0) {
                //TODO: We possibly do want output layer activations in some cases here...
                for (TrainingListener tl : trainingListeners) {
                    tl.onForwardPass(this, activations);
                }
            }
            INDArray actSecondLastLayer = activations.get(activations.size() - 1);
            if (layerWiseConfigurations.getInputPreProcess(layers.length - 1) != null)
                actSecondLastLayer = layerWiseConfigurations.getInputPreProcess(layers.length - 1)
                                .preProcess(actSecondLastLayer, getInputMiniBatchSize());
            getOutputLayer().setInput(actSecondLastLayer);
            //Then: compute gradients
            backprop();
        }

        //Calculate score
        if (!(getOutputLayer() instanceof IOutputLayer)) {
            throw new DL4JException(
                            "Cannot calculate gradient and score with respect to labels: final layer is not an IOutputLayer");
        }
        score = ((IOutputLayer) getOutputLayer()).computeScore(calcL1(true), calcL2(true), true);

        //Listeners
        if (trainingListeners.size() > 0) {
            try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                for (TrainingListener tl : trainingListeners) {
                    tl.onBackwardPass(this);
                }
            }
        }
    }

    @Override
    public void accumulateScore(double accum) {

    }

    /**
     * Clear the inputs. Clears optimizer state.
     */
    public void clear() {
        for (Layer layer : layers)
            layer.clear();

        input = null;
        labels = null;
        solver = null;
    }

    /**
     * Averages the given logistic regression
     * from a mini batch in to this one
     *
     * @param layer     the logistic regression to average in to this one
     * @param batchSize the batch size
     * @deprecated Not supported and not used
     */
    @Override
    @Deprecated
    public void merge(Layer layer, int batchSize) {
        throw new UnsupportedOperationException();
    }

    /**
     * Deprecated: Merges this network with the other one.
     *
     * @param network   the network to merge with
     * @param batchSize the batch size (number of training examples)
     *                  to average by
     * @deprecated As of 0.7.3 - Feb 2017. No longer used; parameter averaging is performed via alternative means/methods
     */
    @Deprecated
    public void merge(MultiLayerNetwork network, int batchSize) {
        if (network.layers.length != layers.length)
            throw new IllegalArgumentException("Unable to merge networks that are not of equal length");
        for (int i = 0; i < getnLayers(); i++) {
            Layer n = layers[i];
            Layer otherNetwork = network.layers[i];
            n.merge(otherNetwork, batchSize);

        }

        getOutputLayer().merge(network.getOutputLayer(), batchSize);
    }


    /**
     * Note that if input isn't null
     * and the neuralNets are null, this is a way
     * of initializing the neural network
     *
     * @param input
     */
    public void setInput(INDArray input) {
        this.input = input;
        if (this.layers == null) {
            log.info("setInput: {}", Nd4j.getMemoryManager().getCurrentWorkspace());
            this.initializeLayers(getInput());
        }
        if (input != null) {
            if (input.length() == 0)
                throw new IllegalArgumentException(
                                "Invalid input: length 0 (shape: " + Arrays.toString(input.shape()) + ")");
            setInputMiniBatchSize(input.size(0));
        }
    }


    /**
     * Get the output layer
     *
     * @return
     */
    public Layer getOutputLayer() {
        return getLayers()[getLayers().length - 1];
    }


    /**
     * Sets parameters for the model.
     * This is used to manipulate the weights and biases across
     * all neuralNets (including the output layer)
     *
     * @param params a parameter vector equal 1,numParameters
     */
    public void setParameters(INDArray params) {
        setParams(params);
    }

    @Override
    public void applyLearningRateScoreDecay() {
        for (Layer layer : layers) {
            if (!layer.conf().getLearningRateByParam().isEmpty()) {
                for (Map.Entry<String, Double> lrPair : layer.conf().getLearningRateByParam().entrySet()) {
                    layer.conf().setLearningRateByParam(lrPair.getKey(),
                                    lrPair.getValue() * (layer.conf().getLrPolicyDecayRate() + Nd4j.EPS_THRESHOLD));
                }
            }
        }
    }

    public NeuralNetConfiguration getDefaultConfiguration() {
        return defaultConfiguration;
    }

    public INDArray getLabels() {
        return labels;
    }

    public INDArray getInput() {
        return input;
    }


    /**
     *
     * @param labels
     */
    public void setLabels(INDArray labels) {
        this.labels = labels;
    }

    /**
     * Get the number of layers in the network
     *
     * @return the number of layers in the network
     */
    public int getnLayers() {
        return layerWiseConfigurations.getConfs().size();
    }

    /**
     *
     * @return
     */
    public synchronized Layer[] getLayers() {
        return layers;
    }

    public Layer getLayer(int i) {
        return layers[i];
    }

    public Layer getLayer(String name) {
        return layerMap.get(name);
    }

    public List<String> getLayerNames() {
        return new ArrayList<>(layerMap.keySet());
    }

    public void setLayers(Layer[] layers) {
        this.layers = layers;
    }

    public INDArray getMask() {
        return mask;
    }

    public void setMask(INDArray mask) {
        this.mask = mask;
    }

    public INDArray getMaskArray() {
        return mask;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public Pair<INDArray, MaskState> feedForwardMaskArray(INDArray maskArray, MaskState currentMaskState,
                    int minibatchSize) {
        if (maskArray == null) {
            for (int i = 0; i < layers.length; i++) {
                layers[i].feedForwardMaskArray(null, null, minibatchSize);
            }
        } else {
            //Do a forward pass through each preprocessor and layer
            for (int i = 0; i < layers.length; i++) {
                InputPreProcessor preProcessor = getLayerWiseConfigurations().getInputPreProcess(i);

                if (preProcessor != null) {
                    Pair<INDArray, MaskState> p =
                                    preProcessor.feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
                    if (p != null) {
                        maskArray = p.getFirst();
                        currentMaskState = p.getSecond();
                    } else {
                        maskArray = null;
                        currentMaskState = null;
                    }
                }

                Pair<INDArray, MaskState> p =
                                layers[i].feedForwardMaskArray(maskArray, currentMaskState, minibatchSize);
                if (p != null) {
                    maskArray = p.getFirst();
                    currentMaskState = p.getSecond();
                } else {
                    maskArray = null;
                    currentMaskState = null;
                }
            }
        }

        return new Pair<>(maskArray, currentMaskState);
    }

    //==========
    //Layer methods

    @Override
    public Gradient error(INDArray errorSignal) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Type type() {
        return Type.MULTILAYER;
    }

    @Override
    public INDArray derivativeActivation(INDArray input) {
        throw new UnsupportedOperationException();
    }

    @Override
    public Gradient calcGradient(Gradient layerError, INDArray activation) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray preOutput(INDArray x) {
        INDArray lastLayerActivation = x;
        for (int i = 0; i < layers.length - 1; i++) {
            if (getLayerWiseConfigurations().getInputPreProcess(i) != null)
                lastLayerActivation = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(lastLayerActivation,
                                getInputMiniBatchSize());
            lastLayerActivation = layers[i].activate(lastLayerActivation);
        }
        if (getLayerWiseConfigurations().getInputPreProcess(layers.length - 1) != null)
            lastLayerActivation = getLayerWiseConfigurations().getInputPreProcess(layers.length - 1)
                            .preProcess(lastLayerActivation, getInputMiniBatchSize());
        return layers[layers.length - 1].preOutput(lastLayerActivation);
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

    @Override
    public Layer transpose() {
        throw new UnsupportedOperationException();
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon) {
        if (getOutputLayer() instanceof IOutputLayer)
            throw new UnsupportedOperationException("Cannot calculate gradients based on epsilon with OutputLayer");

        return calcBackpropGradients(epsilon, false);
    }

    @Override
    public void setIndex(int index) {
        layerIndex = index;
    }

    @Override
    public int getIndex() {
        return layerIndex;
    }

    @Override
    public double calcL2(boolean backpropParamsOnly) {
        double l2 = 0.0;
        for (int i = 0; i < layers.length; i++) {
            l2 += layers[i].calcL2(backpropParamsOnly);
        }
        return l2;
    }

    @Override
    public double calcL1(boolean backpropParamsOnly) {
        double l1 = 0.0;
        for (int i = 0; i < layers.length; i++) {
            l1 += layers[i].calcL1(backpropParamsOnly);
        }
        return l1;
    }

    @Override
    public void update(Gradient gradient) {
        if (gradient.gradient().length() != numParams(true))
            throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams(true));
        for (Map.Entry<String, INDArray> entry : gradient.gradientForVariable().entrySet()) {
            String key = entry.getKey();
            INDArray val = entry.getValue();
            int idx = key.indexOf('_');
            if (idx == -1)
                throw new IllegalStateException("Invalid param key: not have layer separator: \"" + key + "\"");
            Integer layerId = Integer.parseInt(key.substring(0, idx));
            String paramType = key.substring(idx + 1);
            // Update MLN gradient
            this.gradient.gradientForVariable().put(key, val);
            // Update layer params
            layers[layerId].update(val, paramType);
        }
        // Update layerwise gradient view
        setBackpropGradientsViewArray(gradient.gradient());

    }

    @Override
    public INDArray preOutput(INDArray x, boolean training) {
        throw new UnsupportedOperationException();

    }

    @Override
    public INDArray activate(boolean training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input, boolean training) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void setInputMiniBatchSize(int size) {
        if (layers != null)
            for (Layer l : layers)
                l.setInputMiniBatchSize(size);
    }

    @Override
    public int getInputMiniBatchSize() {
        return input.size(0);
    }

    @Override
    public void setMaskArray(INDArray maskArray) {
        throw new UnsupportedOperationException();
    }

    /**
     *
     * If this MultiLayerNetwork contains one or more RNN layers: conduct forward pass (prediction)
     * but using previous stored state for any RNN layers. The activations for the final step are
     * also stored in the RNN layers for use next time rnnTimeStep() is called.<br>
     * This method can be used to generate output one or more steps at a time instead of always having to do
     * forward pass from t=0. Example uses are for streaming data, and for generating samples from network output
     * one step at a time (where samples are then fed back into the network as input)<br>
     * If no previous state is present in RNN layers (i.e., initially or after calling rnnClearPreviousState()),
     * the default initialization (usually 0) is used.<br>
     * Supports mini-batch (i.e., multiple predictions/forward pass in parallel) as well as for single examples.<br>
     * @param input Input to network. May be for one or multiple time steps. For single time step:
     *  input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1]. miniBatchSize=1 for single example.<br>
     *  For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]
     * @return Output activations. If output is RNN layer (such as RnnOutputLayer): if input has shape [miniBatchSize,inputSize]
     * i.e., is 2d, output has shape [miniBatchSize,outputSize] (i.e., also 2d).<br>
     * Otherwise output is 3d [miniBatchSize,outputSize,inputTimeSeriesLength] when using RnnOutputLayer.
     */
    public INDArray rnnTimeStep(INDArray input) {
        this.setInputMiniBatchSize(input.size(0)); //Necessary for preprocessors/reshaping
        this.input = input;
        boolean inputIs2d = input.rank() == 2;
        for (int i = 0; i < layers.length; i++) {
            if (getLayerWiseConfigurations().getInputPreProcess(i) != null)
                input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input, getInputMiniBatchSize());
            if (layers[i] instanceof RecurrentLayer) {
                input = ((RecurrentLayer) layers[i]).rnnTimeStep(input);
            } else if (layers[i] instanceof MultiLayerNetwork) {
                input = ((MultiLayerNetwork) layers[i]).rnnTimeStep(input);
            } else {
                input = layers[i].activate(input, false);
            }
        }
        if (inputIs2d && input.rank() == 3 && layers[layers.length - 1].type() == Type.RECURRENT) {
            //Return 2d output with shape [miniBatchSize,nOut]
            // instead of 3d output with shape [miniBatchSize,nOut,1]
            return input.tensorAlongDimension(0, 1, 0);
        }

        this.input = null;
        return input;
    }

    /**Get the state of the RNN layer, as used in rnnTimeStep().
     * @param layer Number/index of the layer.
     * @return Hidden state, or null if layer is not an RNN layer
     */
    public Map<String, INDArray> rnnGetPreviousState(int layer) {
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        if (!(layers[layer] instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");
        return ((RecurrentLayer) layers[layer]).rnnGetPreviousState();
    }

    /**Set the state of the RNN layer.
     * @param layer The number/index of the layer.
     * @param state The state to set the specified layer to
     */
    public void rnnSetPreviousState(int layer, Map<String, INDArray> state) {
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        if (!(layers[layer] instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");

        RecurrentLayer r = (RecurrentLayer) layers[layer];
        r.rnnSetPreviousState(state);
    }

    /** Clear the previous state of the RNN layers (if any).
     */
    public void rnnClearPreviousState() {
        if (layers == null)
            return;
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] instanceof RecurrentLayer)
                ((RecurrentLayer) layers[i]).rnnClearPreviousState();
            else if (layers[i] instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) layers[i]).rnnClearPreviousState();
            }
        }
    }

    /** Similar to rnnTimeStep and feedForward() methods. Difference here is that this method:<br>
     * (a) like rnnTimeStep does forward pass using stored state for RNN layers, and<br>
     * (b) unlike rnnTimeStep does not modify the RNN layer state<br>
     * Therefore multiple calls to this method with the same input should have the same output.<br>
     * Typically used during training only. Use rnnTimeStep for prediction/forward pass at test time.
     * @param input Input to network
     * @param training Whether training or not
     * @param storeLastForTBPTT set to true if used as part of truncated BPTT training
     * @return Activations for each layer (including input, as per feedforward() etc)
     */
    public List<INDArray> rnnActivateUsingStoredState(INDArray input, boolean training, boolean storeLastForTBPTT) {
        INDArray currInput = input;
        List<INDArray> activations = new ArrayList<>();
        activations.add(currInput);

        for (int i = 0; i < layers.length; i++) {
            if (getLayerWiseConfigurations().getInputPreProcess(i) != null)
                currInput = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(currInput, input.size(0));
            if (layers[i] instanceof RecurrentLayer) {
                currInput = ((RecurrentLayer) layers[i]).rnnActivateUsingStoredState(currInput, training,
                                storeLastForTBPTT);
            } else if (layers[i] instanceof MultiLayerNetwork) {
                List<INDArray> temp = ((MultiLayerNetwork) layers[i]).rnnActivateUsingStoredState(currInput, training,
                                storeLastForTBPTT);
                currInput = temp.get(temp.size() - 1);
            } else {
                currInput = layers[i].activate(currInput, training);
            }
            activations.add(currInput);
        }
        return activations;
    }

    /** Get the updater for this MultiLayerNetwork
     * @return Updater for MultiLayerNetwork
     */
    public synchronized Updater getUpdater() {
        if (solver == null) {
            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
            solver.getOptimizer().setUpdater(UpdaterCreator.getUpdater(this));
        }
        return solver.getOptimizer().getUpdater();
    }

    /** Set the updater for the MultiLayerNetwork */
    public void setUpdater(Updater updater) {
        if (solver == null) {
            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
        }
        solver.getOptimizer().setUpdater(updater);
    }

    /**Set the mask arrays for features and labels. Mask arrays are typically used in situations such as one-to-many
     * and many-to-one learning with recurrent neural networks, as well as for supporting time series of varying lengths
     * within the same minibatch.<br>
     * For example, with RNN data sets with input of shape [miniBatchSize,nIn,timeSeriesLength] and outputs of shape
     * [miniBatchSize,nOut,timeSeriesLength], the features and mask arrays will have shape [miniBatchSize,timeSeriesLength]
     * and contain values 0 or 1 at each element (to specify whether a given input/example is present - or merely padding -
     * at a given time step).<br>
     * <b>NOTE</b>: This method is not usually used directly. Instead, methods such as {@link #feedForward(INDArray, INDArray, INDArray)}
     * and {@link #output(INDArray, boolean, INDArray, INDArray)} handle setting of masking internally.
     * @param featuresMaskArray Mask array for features (input)
     * @param labelsMaskArray Mask array for labels (output)
     * @see #clearLayerMaskArrays()
     */
    public void setLayerMaskArrays(INDArray featuresMaskArray, INDArray labelsMaskArray) {
        if (featuresMaskArray != null) {

            //New approach: use feedForwardMaskArray method
            feedForwardMaskArray(featuresMaskArray, MaskState.Active, featuresMaskArray.size(0));


            /*
            //feedforward layers below a RNN layer: need the input (features) mask array
            //Reason: even if the time series input is zero padded, the output from the dense layers are
            // non-zero (i.e., activationFunction(0*weights + bias) != 0 in general)
            //This assumes that the time series input is masked - i.e., values are 0 at the padded time steps,
            // so we don't need to do anything for the recurrent layer
            
            //Now, if mask array is 2d -> need to reshape to 1d (column vector) in the exact same order
            // as is done for 3d -> 2d time series reshaping
            INDArray reshapedFeaturesMask = TimeSeriesUtils.reshapeTimeSeriesMaskToVector(featuresMaskArray);
            
            for( int i=0; i<layers.length-1; i++ ){
                Type t = layers[i].type();
                if( t == Type.CONVOLUTIONAL || t == Type.FEED_FORWARD ){
                    layers[i].setMaskArray(reshapedFeaturesMask);
                } else if( t == Type.RECURRENT ) break;
            
            }
            */
        }
        if (labelsMaskArray != null) {
            if (!(getOutputLayer() instanceof IOutputLayer))
                return;
            layers[layers.length - 1].setMaskArray(labelsMaskArray);
        }
    }

    /** Remove the mask arrays from all layers.<br>
     * See {@link #setLayerMaskArrays(INDArray, INDArray)} for details on mask arrays.
     */
    public void clearLayerMaskArrays() {
        for (Layer layer : layers) {
            layer.setMaskArray(null);
        }
    }

    /**
     * Evaluate the network (classification performance)
     *
     * @param iterator Iterator to evaluate on
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public Evaluation evaluate(DataSetIterator iterator) {
        return evaluate(iterator, null);
    }

    /**
     * Evaluate the network for regression performance
     * @param iterator Data to evaluate on
     * @return
     */
    public RegressionEvaluation evaluateRegression(DataSetIterator iterator) {
        return doEvaluation(iterator, new RegressionEvaluation(iterator.totalOutcomes()))[0];
    }

    /**
     * Evaluate the network (must be a binary classifier) on the specified data, using the {@link ROC} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROC}
     * @return ROC evaluation on the given dataset
     */
    public ROC evaluateROC(DataSetIterator iterator, int rocThresholdSteps) {
        return doEvaluation(iterator, new ROC(rocThresholdSteps))[0];
    }

    /**
     * Evaluate the network on the specified data, using the {@link ROCMultiClass} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROCMultiClass}
     * @return Multi-class ROC evaluation on the given dataset
     */
    public ROCMultiClass evaluateROCMultiClass(DataSetIterator iterator, int rocThresholdSteps) {
        return doEvaluation(iterator, new ROCMultiClass(rocThresholdSteps))[0];
    }

    /**
     * Perform evaluation using an arbitrary IEvaluation instance.
     *
     * @param iterator   data to evaluate on
     */
    public <T extends IEvaluation> T[] doEvaluation(DataSetIterator iterator, T... evaluations) {
        if (!iterator.hasNext() && iterator.resetSupported()) {
            iterator.reset();
        }

        DataSetIterator iter = iterator.asyncSupported() ? new AsyncDataSetIterator(iterator, 2, true) : iterator;

        WorkspaceMode cMode = layerWiseConfigurations.getTrainingWorkspaceMode();
        layerWiseConfigurations.setTrainingWorkspaceMode(layerWiseConfigurations.getInferenceWorkspaceMode());

        MemoryWorkspace workspace =
                        layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE ? new DummyWorkspace()
                                        : Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(
                                                        workspaceConfigurationExternal, workspaceExternal);

        while (iter.hasNext()) {
            DataSet next = iter.next();

            if (next.getFeatureMatrix() == null || next.getLabels() == null)
                break;

            try (MemoryWorkspace wsB = workspace.notifyScopeEntered()) {

                INDArray features = next.getFeatures();
                INDArray labels = next.getLabels();
                INDArray lMask = next.getLabelsMaskArray();

                INDArray out;
                if (next.hasMaskArrays()) {
                    INDArray fMask = next.getFeaturesMaskArray();
                    out = this.silentOutput(features, false, fMask, lMask);
                } else {
                    out = this.silentOutput(features, false);
                }

                try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    for (T evaluation : evaluations)
                        evaluation.eval(labels, out, lMask);
                }
            }

            clearLayerMaskArrays();
        }

        if (iterator.asyncSupported())
            ((AsyncDataSetIterator) iter).shutdown();

        layerWiseConfigurations.setTrainingWorkspaceMode(cMode);

        return evaluations;
    }

    /**
     * Evaluate the network on the provided data set. Used for evaluating the performance of classifiers
     *
     * @param iterator Data to undertake evaluation on
     * @return Evaluation object, summarizing the results of the evaluation on the provided DataSetIterator
     */
    public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList) {
        return evaluate(iterator, labelsList, 1);
    }

    @Override
    public INDArray updaterState() {
        return getUpdater() != null ? getUpdater().getStateViewArray() : null;
    }

    @Override
    public void fit(MultiDataSet dataSet) {
        if (dataSet.getFeatures().length == 1 && dataSet.getLabels().length == 1) {
            INDArray features = null;
            INDArray labels = null;
            INDArray fMask = null;
            INDArray lMask = null;

            if (dataSet.getFeaturesMaskArrays() != null)
                fMask = dataSet.getFeaturesMaskArrays()[0];

            if (dataSet.getFeaturesMaskArrays() != null)
                lMask = dataSet.getLabelsMaskArrays()[0];

            features = dataSet.getFeatures()[0];
            labels = dataSet.getLabels()[0];

            DataSet ds = new DataSet(features, labels, fMask, lMask);
            fit(ds);
        }
        throw new DL4JInvalidInputException(
                        "MultiLayerNetwork can't handle MultiDataSet. Please consider use of ComputationGraph");
    }

    @Override
    public void fit(MultiDataSetIterator iterator) {
        fit(new MultiDataSetWrapperIterator(iterator));
    }

    @Override
    public <T extends IEvaluation> T[] doEvaluation(MultiDataSetIterator iterator, T[] evaluations) {
        return doEvaluation(new MultiDataSetWrapperIterator(iterator), evaluations);
    }

    /**
     * Evaluate the network (for classification) on the provided data set, with top N accuracy in addition to standard accuracy.
     * For 'standard' accuracy evaluation only, use topN = 1
     *
     * @param iterator   Iterator (data) to evaluate on
     * @param labelsList List of labels. May be null.
     * @param topN       N value for top N accuracy evaluation
     * @return Evaluation object, summarizing the results of the evaluation on the provided DataSetIterator
     */
    public Evaluation evaluate(DataSetIterator iterator, List<String> labelsList, int topN) {
        if (layers == null || !(getOutputLayer() instanceof IOutputLayer)) {
            throw new IllegalStateException("Cannot evaluate network with no output layer");
        }
        if (labelsList == null)
            labelsList = iterator.getLabels();

        Evaluation e = new Evaluation(labelsList, topN);
        doEvaluation(iterator, e);

        return e;
    }

    private void update(Task task) {
        if (!initDone) {
            initDone = true;
            Heartbeat heartbeat = Heartbeat.getInstance();
            task = ModelSerializer.taskByModel(this);
            Environment env = EnvironmentUtils.buildEnvironment();
            heartbeat.reportEvent(Event.STANDALONE, env, task);
        }
    }

    /**
     * String detailing the architecture of the multilayernetwork.
     * Columns are LayerIndex with layer type, nIn, nOut, Total number of parameters and the Shapes of the parameters
     * Will also give information about frozen layers, if any.
     * @return Summary as a string
     */
    public String summary() {
        String ret = "\n";
        ret += StringUtils.repeat("=", 140);
        ret += "\n";
        ret += String.format("%-40s%-15s%-15s%-30s\n", "LayerName (LayerType)", "nIn,nOut", "TotalParams",
                        "ParamsShape");
        ret += StringUtils.repeat("=", 140);
        ret += "\n";
        int frozenParams = 0;
        for (Layer currentLayer : layers) {
            String name = String.valueOf(currentLayer.getIndex());
            String paramShape = "-";
            String in = "-";
            String out = "-";
            String[] classNameArr = currentLayer.getClass().getName().split("\\.");
            String className = classNameArr[classNameArr.length - 1];
            String paramCount = String.valueOf(currentLayer.numParams());
            if (currentLayer.numParams() > 0) {
                paramShape = "";
                in = String.valueOf(((FeedForwardLayer) currentLayer.conf().getLayer()).getNIn());
                out = String.valueOf(((FeedForwardLayer) currentLayer.conf().getLayer()).getNOut());
                Set<String> paraNames = currentLayer.conf().getLearningRateByParam().keySet();
                for (String aP : paraNames) {
                    String paramS = ArrayUtils.toString(currentLayer.paramTable().get(aP).shape());
                    paramShape += aP + ":" + paramS + ", ";
                }
                paramShape = paramShape.subSequence(0, paramShape.lastIndexOf(",")).toString();
            }
            if (currentLayer instanceof FrozenLayer) {
                frozenParams += currentLayer.numParams();
                classNameArr = ((FrozenLayer) currentLayer).getInsideLayer().getClass().getName().split("\\.");
                className = "Frozen " + classNameArr[classNameArr.length - 1];
            }
            ret += String.format("%-40s%-15s%-15s%-30s", name + " (" + className + ")", in + "," + out, paramCount,
                            paramShape);
            ret += "\n";
        }
        ret += StringUtils.repeat("-", 140);
        ret += String.format("\n%30s %d", "Total Parameters: ", params().length());
        ret += String.format("\n%30s %d", "Trainable Parameters: ", params().length() - frozenParams);
        ret += String.format("\n%30s %d", "Frozen Parameters: ", frozenParams);
        ret += "\n";
        ret += StringUtils.repeat("=", 140);
        ret += "\n";
        return ret;
    }

    /**
     * This method just makes sure there's no state preserved within layers
     */
    protected void clearLayersStates() {
        for (int f = 0; f < layers.length; f++) {
            layers[f].setInput(null);
            layers[f].setMaskArray(null);
            layers[f].clear();
        }
    }

    /**
     * Indicates whether some other object is "equal to" this one.
     * <p>
     * The {@code equals} method implements an equivalence relation
     * on non-null object references:
     * <ul>
     * <li>It is <i>reflexive</i>: for any non-null reference value
     * {@code x}, {@code x.equals(x)} should return
     * {@code true}.
     * <li>It is <i>symmetric</i>: for any non-null reference values
     * {@code x} and {@code y}, {@code x.equals(y)}
     * should return {@code true} if and only if
     * {@code y.equals(x)} returns {@code true}.
     * <li>It is <i>transitive</i>: for any non-null reference values
     * {@code x}, {@code y}, and {@code z}, if
     * {@code x.equals(y)} returns {@code true} and
     * {@code y.equals(z)} returns {@code true}, then
     * {@code x.equals(z)} should return {@code true}.
     * <li>It is <i>consistent</i>: for any non-null reference values
     * {@code x} and {@code y}, multiple invocations of
     * {@code x.equals(y)} consistently return {@code true}
     * or consistently return {@code false}, provided no
     * information used in {@code equals} comparisons on the
     * objects is modified.
     * <li>For any non-null reference value {@code x},
     * {@code x.equals(null)} should return {@code false}.
     * </ul>
     * <p>
     * The {@code equals} method for class {@code Object} implements
     * the most discriminating possible equivalence relation on objects;
     * that is, for any non-null reference values {@code x} and
     * {@code y}, this method returns {@code true} if and only
     * if {@code x} and {@code y} refer to the same object
     * ({@code x == y} has the value {@code true}).
     * <p>
     * Note that it is generally necessary to override the {@code hashCode}
     * method whenever this method is overridden, so as to maintain the
     * general contract for the {@code hashCode} method, which states
     * that equal objects must have equal hash codes.
     *
     * @param obj the reference object with which to compare.
     * @return {@code true} if this object is the same as the obj
     * argument; {@code false} otherwise.
     * @see #hashCode()
     * @see HashMap
     */
    @Override
    public boolean equals(Object obj) {
        if (obj == null)
            return false;
        if (obj instanceof MultiLayerNetwork) {
            MultiLayerNetwork network = (MultiLayerNetwork) obj;
            boolean paramsEquals = network.params().equals(params());
            boolean confEquals = getLayerWiseConfigurations().equals(network.getLayerWiseConfigurations());
            boolean updaterEquals = getUpdater().equals(network.getUpdater());
            return paramsEquals && confEquals && updaterEquals;
        }
        return false;
    }
}
