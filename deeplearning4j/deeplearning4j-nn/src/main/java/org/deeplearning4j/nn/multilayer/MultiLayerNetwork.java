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

package org.deeplearning4j.nn.multilayer;


import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.MultiDataSetWrapperIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.exception.DL4JException;
import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.api.Updater;
import org.deeplearning4j.nn.api.layers.IOutputLayer;
import org.deeplearning4j.nn.api.layers.RecurrentLayer;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.Bidirectional;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.layers.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.layers.recurrent.BidirectionalLayer;
import org.deeplearning4j.nn.layers.LayerHelper;
import org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer;
import org.deeplearning4j.nn.updater.MultiLayerUpdater;
import org.deeplearning4j.nn.updater.UpdaterCreator;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.solvers.accumulation.GradientsAccumulator;
import org.deeplearning4j.util.CrashReportingUtil;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.util.NetworkUtils;
import org.deeplearning4j.util.OutputLayerUtil;
import org.nd4j.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
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
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.linalg.primitives.Triple;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.util.FeatureUtil;
import org.nd4j.linalg.workspace.ND4JWorkspaceException;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.nd4j.util.OneTimeLogger;

import java.io.*;
import java.util.*;


/**
 * MultiLayerNetwork is a neural network with multiple layers in a stack, and usually an output layer.<br>
 * For neural networks with a more complex connection architecture, use {@link org.deeplearning4j.nn.graph.ComputationGraph}
 * which allows for an arbitrary directed acyclic graph connection structure between layers.
 * MultiLayerNetwork is trainable via backprop, with optional unsupervised layerwise training, depending on the type of layers it contains.
 *
 * @author Adam Gibson
 */
@Slf4j
public class MultiLayerNetwork implements Serializable, Classifier, Layer, NeuralNetwork {

    //the hidden neural network layers (including output layer)
    protected Layer[] layers;
    protected LinkedHashMap<String, Layer> layerMap = new LinkedHashMap<>();

    //Current training data: input features and labels
    protected INDArray input, labels;

    protected boolean initCalled = false;
    protected Collection<TrainingListener> trainingListeners = new ArrayList<>();

    protected NeuralNetConfiguration defaultConfiguration;
    protected MultiLayerConfiguration layerWiseConfigurations;
    protected Gradient gradient;
    protected double score;
    @Setter
    protected boolean initDone = false;
    protected INDArray flattenedParams; //Params for all layers are a view/subset of this array
    @Getter
    protected transient INDArray flattenedGradients; //Gradients for all layers are a view/subset of this array

    protected boolean clearTbpttState = true;  //Mainly for unit testing (should be enabled otherwise)
    protected transient ThreadLocal<Long> lastEtlTime = new ThreadLocal<>();
    protected INDArray mask;

    protected int layerIndex; //For Layer.get/setIndex()

    protected transient Solver solver; //Used to call optimizers during backprop
    //Workspaces for CUDNN. Pass to LayerWorkspaceMgr for re-use in cudnn helpers
    @Getter
    protected transient Map<String,Pointer> helperWorkspaces = new HashMap<>();


    /**
     * Workspace for working memory for a single layer: forward pass and backward pass
     * Note that this is opened/closed once per op (activate/backpropGradient call)
     */
    protected static final String WS_LAYER_WORKING_MEM = "WS_LAYER_WORKING_MEM";
    /**
     * Workspace for storing all layers' activations - used only to store activations (layer inputs) as part of backprop
     * Not used for inference
     */
    protected static final String WS_ALL_LAYERS_ACT = "WS_ALL_LAYERS_ACT";
    /**
     * Next 2 workspaces: used for:
     * (a) Inference: holds activations for one layer only
     * (b) Backprop: holds activation gradients for one layer only
     * In both cases, they are opened and closed on every second layer
     */
    protected static final String WS_LAYER_ACT_1 = "WS_LAYER_ACT_1";
    protected static final String WS_LAYER_ACT_2 = "WS_LAYER_ACT_2";

    /**
     * Workspace for output methods that use OutputAdapter
     */
    protected static final String WS_OUTPUT_MEM = "WS_OUTPUT_MEM";

    /**
     * Workspace for working memory in RNNs - opened and closed once per RNN time step
     */
    protected static final String WS_RNN_LOOP_WORKING_MEM = "WS_RNN_LOOP_WORKING_MEM";


    protected WorkspaceConfiguration WS_LAYER_WORKING_MEM_CONFIG;

    protected static final WorkspaceConfiguration WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.05)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    protected WorkspaceConfiguration WS_LAYER_ACT_X_CONFIG;

    protected static final WorkspaceConfiguration WS_RNN_LOOP_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0).overallocationLimit(0.05).policyReset(ResetPolicy.BLOCK_LEFT)
            .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
            .policyLearning(LearningPolicy.FIRST_LOOP).build();


    public MultiLayerNetwork(MultiLayerConfiguration conf) {
        this.layerWiseConfigurations = conf;
        this.defaultConfiguration = conf.getConf(0).clone();

        //Working memory: should learn over course of: (a) full forward pass, and (b) full backward pass
        //Working memory should be opened once per layer and once per preprocessor, for each of forward and backward passes
        int numWorkingMem = 2 * (layerWiseConfigurations.getConfs().size() + layerWiseConfigurations.getInputPreProcessors().size());
        WS_LAYER_WORKING_MEM_CONFIG = getLayerWorkingMemWSConfig(numWorkingMem);
        WS_LAYER_ACT_X_CONFIG = getLayerActivationWSConfig(layerWiseConfigurations.getConfs().size());
    }

    protected static WorkspaceConfiguration getLayerWorkingMemWSConfig(int numWorkingMemCycles){
        return WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(numWorkingMemCycles)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
    }

    protected static WorkspaceConfiguration getLayerActivationWSConfig(int numLayers){
        //Activations memory: opened once per layer - for every second layer (preprocessors are within the loop).
        //Technically we could set learning to numLayers / 2, but will set to numLayers for simplicity, and also to
        // account for a backward pass
        return WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(numLayers)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
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

    /**
     * Set the last ETL time in milliseconds, for informational/reporting purposes. Generally used internally.
     * @param time    ETL time
     */
    public void setLastEtlTime(long time) {
        lastEtlTime.set(time);
    }

    /**
     * Get the last ETL time. This in informational, and is the amount of time in milliseconds that was required
     * to obtain the last DataSet/MultiDataSet during fitting.
     * A value consistently above 0 may indicate a data feeding bottleneck, or no asynchronous data prefetching (async
     * prefetch is enabled by default)
     * @return The last ETL time in milliseconds, if avaliable (or 0 if not)
     */
    public long getLastEtlTime() {
        Long time = lastEtlTime.get();
        return time == null ? 0L : time;
    }

    /**
     * Initialize the network based on the configuration (a MultiLayerConfiguration in JSON format) and parameters array
     *
     * @param conf   the configuration json
     * @param params the parameters for the network
     */
    public MultiLayerNetwork(String conf, INDArray params) {
        this(MultiLayerConfiguration.fromJson(conf));
        init();
        setParameters(params);
    }


    /**
     * Initialize the network based on the configuration and parameters array
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
     * Perform layerwise pretraining for one epoch - see {@link #pretrain(DataSetIterator, int)}
     */
    public void pretrain(DataSetIterator iter) {
        pretrain(iter, 1);
    }

    /**
     * Perform layerwise unsupervised training on all pre-trainable layers in the network (VAEs, Autoencoders, etc), for the specified
     * number of epochs each. For example, if numEpochs=3, then layer 0 will be fit for 3 epochs, followed by layer 1
     * for 3 epochs, and so on.<br>
     * Note that pretraining will be performed on one layer after the other. To perform unsupervised training on a single layer,
     * use {@link #pretrainLayer(int, DataSetIterator)}
     *
     * @param iter Training data
     */
    public void pretrain(DataSetIterator iter, int numEpochs){
        if (flattenedGradients == null) {
            initGradientsView();
        }

        for (int i = 0; i < getnLayers(); i++) {
            pretrainLayer(i, iter, numEpochs);
        }
    }

    /**
     * Fit for one epoch - see {@link #pretrainLayer(int, DataSetIterator, int)}
     */
    public void pretrainLayer(int layerIdx, DataSetIterator iter) {
        pretrainLayer(layerIdx, iter, 1);
    }

    /**
     * Perform layerwise unsupervised training on a single pre-trainable layer in the network (VAEs, Autoencoders, etc)
     * for the specified number of epochs<br>
     * If the specified layer index (0 to numLayers - 1) is not a pretrainable layer, this is a no-op.
     *
     * @param layerIdx  Index of the layer to train (0 to numLayers-1)
     * @param iter      Training data
     * @param numEpochs Number of epochs to fit the specified layer for
     */
    public void pretrainLayer(int layerIdx, DataSetIterator iter, int numEpochs) {
        Preconditions.checkState(numEpochs > 0, "Number of epochs (%s) must be a positive number", numEpochs);

        if (flattenedGradients == null) {
            initGradientsView();
        }
        if (layerIdx >= layers.length) {
            throw new IllegalArgumentException(
                    "Cannot pretrain layer: layerIdx (" + layerIdx + ") >= numLayers (" + layers.length + ")");
        }

        Layer layer = layers[layerIdx];
        if (!layer.isPretrainLayer())
            return;

        if(numEpochs > 1 && !iter.resetSupported())
            throw new IllegalStateException("Cannot fit multiple epochs (" + numEpochs + ") on an iterator that doesn't support resetting");

        if (!iter.hasNext() && iter.resetSupported()) {
            iter.reset();
        }

        log.info("Starting unsupervised training on layer " + layerIdx + " for " + numEpochs + " epochs");
        for(int i=0; i<numEpochs; i++ ) {
            if(i > 0)
                iter.reset();

            while (iter.hasNext()) {
                DataSet next = iter.next();
                input = next.getFeatures();
                pretrainLayer(layerIdx, input);
            }
        }

        int ec = getLayer(layerIdx).conf().getEpochCount() + 1;
        getLayer(layerIdx).conf().setEpochCount(ec);
    }

    /**
     * Perform layerwise unsupervised training on a single pre-trainable layer in the network (VAEs, Autoencoders, etc)<br>
     * If the specified layer index (0 to numLayers - 1) is not a pretrainable layer, this is a no-op.
     *
     * @param layerIdx Index of the layer to train (0 to numLayers-1)
     * @param features Training data array
     */
    public void pretrainLayer(int layerIdx, INDArray features) {
        setInput(features);
        setLayerMaskArrays(null, null);

        if (flattenedGradients == null) {
            initGradientsView();
        }
        if (layerIdx >= layers.length) {
            throw new IllegalArgumentException(
                    "Cannot pretrain layer: layerIdx (" + layerIdx + ") >= numLayers (" + layers.length + ")");
        }

        LayerWorkspaceMgr workspaceMgr;
        if(layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .defaultWorkspace(WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        Layer layer = layers[layerIdx];
        if (!layer.isPretrainLayer())
            return;

        //Do forward pass to the layer to be pretrained
        INDArray outputOfPrevLayer;
        if(layerIdx == 0) {
            outputOfPrevLayer = input;
        } else {
            //Yes, this part of training - but we'll do forward psas as inference mode when doing layerwise training
            // to effectively freeze earlier layers and not apply dropout etc
            outputOfPrevLayer = outputOfLayerDetached(false, FwdPassType.STANDARD, layerIndex-1, features, null, null, null);
        }

        try(MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {
            if (layerWiseConfigurations.getInputPreProcess(layerIdx) != null) {

                // FIXME: int cast
                outputOfPrevLayer = layerWiseConfigurations.getInputPreProcess(layerIdx).preProcess(outputOfPrevLayer, (int) input.size(0),
                        LayerWorkspaceMgr.noWorkspaces(helperWorkspaces));
            }

            layer.fit(outputOfPrevLayer, workspaceMgr);
        }
    }

    @Override
    public int batchSize() {
        // FIXME: int cast
        return (int) input.size(0);
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
    public ConvexOptimizer getOptimizer() {
        return solver.getOptimizer();
    }

    /**
     * Get one parameter array for the network.<br>
     * In MultiLayerNetwork, parameters are keyed like "0_W" and "0_b" to mean "weights of layer index 0" and "biases
     * of layer index 0" respectively. Numbers increment sequentially, and the suffixes ("W", "b" etc) depend on the
     * layer type, and are defined in the relevant parameter initializers for each layer.<br>
     * Note that the returned INDArrays are views of the underlying network parameters, so modifications of the returned
     * arrays will impact the parameters of the network.
     *
     * @param param the key of the parameter
     * @return The specified parameter array for the network
     * @see #paramTable() paramTable() method, for a map of all parameters
     */
    @Override
    public INDArray getParam(String param) {
        //Get params for MultiLayerNetwork sub layers.
        int idx = param.indexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: does not have layer separator: \"" + param + "\"");
        int layerIdx = Integer.parseInt(param.substring(0, idx));
        String newKey = param.substring(idx + 1);

        return layers[layerIdx].getParam(newKey);
    }

    /**
     * Return a map of all parameters in the network. Parameter names are as described in {@link #getParam(String)}.
     * As per {@link #getParam(String)} the returned arrays are views - modifications to these will impact
     * the underlying network parameters
     * @return A map of all parameters in the network
     */
    @Override
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    /**
     * Returns a map of all parameters in the network as per {@link #paramTable()}.<br>
     * Optionally (with backpropParamsOnly=true) only the 'backprop' parameters are returned - that is, any parameters
     * involved only in unsupervised layerwise pretraining not standard inference/backprop are excluded from the returned list.
     * @param backpropParamsOnly If true, return backprop params only. If false: return all params
     * @return Parameters for the network
     */
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

    /**
     * Intended for internal use
     */
    @Override
    public boolean updaterDivideByMinibatch(String paramName) {
        int idx = paramName.indexOf('_');
        int layerIdx = Integer.parseInt(paramName.substring(0, idx));
        String subName = paramName.substring(idx+1);
        return getLayer(layerIdx).updaterDivideByMinibatch(subName);
    }

    /**
     * Set the parameters of the netowrk. Note that the parameter keys must match the format as described in {@link #getParam(String)}
     * and {@link #paramTable()}. Note that the values of the parameters used as an argument to this method are copied -
     * i.e., it is safe to later modify/reuse the values in the provided paramTable without this impacting the network.
     *
     * @param paramTable    Parameters to set
     */
    @Override
    public void setParamTable(Map<String, INDArray> paramTable) {
        Map<String, INDArray> currParamTable = paramTable();
        if (!currParamTable.keySet().equals(paramTable.keySet())) {
            throw new IllegalArgumentException("Cannot set param table: parameter keys do not match.\n" + "Current: "
                    + currParamTable.keySet() + "\nTo set: " + paramTable.keySet());
        }

        for (String s : paramTable.keySet()) {
            INDArray curr = currParamTable.get(s);
            INDArray toSet = paramTable.get(s);
            if (!Arrays.equals(curr.shape(), toSet.shape())) {
                throw new IllegalArgumentException("Cannot set parameter table: parameter \"" + s + "\" shapes "
                        + "do not match. Current = " + Arrays.toString(curr.shape()) + ", to set = "
                        + Arrays.toString(toSet.shape()));
            }
        }

        //Now that we've checked ALL params (to avoid leaving net in half-modified state)
        for (String s : paramTable.keySet()) {
            INDArray curr = currParamTable.get(s);
            INDArray toSet = paramTable.get(s);
            curr.assign(toSet);
        }
    }

    /**
     * Set the values of a single parameter. See {@link #setParamTable(Map)} and {@link #getParam(String)} for more
     * details.
     * @param key the key of the parameter to set
     * @param val the new values for the parameter
     */
    @Override
    public void setParam(String key, INDArray val) {
        //Set params for MultiLayerNetwork sub layers.
        int idx = key.indexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: not have layer separator: \"" + key + "\"");
        int layerIdx = Integer.parseInt(key.substring(0, idx));
        String newKey = key.substring(idx + 1);

        layers[layerIdx].setParam(newKey, val);
    }

    /**
     * Get the configuration for the network
     * @return Network configuration
     */
    public MultiLayerConfiguration getLayerWiseConfigurations() {
        return layerWiseConfigurations;
    }

    /**
     * This method is intended for internal/developer use only.
     */
    public void setLayerWiseConfigurations(MultiLayerConfiguration layerWiseConfigurations) {
        this.layerWiseConfigurations = layerWiseConfigurations;
    }

    /**
     * Initialize the MultiLayerNetwork. This should be called once before the network is used.
     * This is functionally equivalent to calling {@code init(null, false)}.
     * @see MultiLayerNetwork#init(INDArray, boolean)
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

        if (layerMap == null)
            layerMap = new LinkedHashMap<>();

        if (layerWiseConfigurations.getTrainingWorkspaceMode() == null)
            layerWiseConfigurations.setTrainingWorkspaceMode(WorkspaceMode.NONE);

        if (layerWiseConfigurations.getInferenceWorkspaceMode() == null)
            layerWiseConfigurations.setInferenceWorkspaceMode(WorkspaceMode.NONE);

        if (layerWiseConfigurations.getCacheMode() == null)
            layerWiseConfigurations.setCacheMode(CacheMode.NONE);

        OneTimeLogger.info(log, "Starting MultiLayerNetwork with WorkspaceModes set to [training: {}; inference: {}], cacheMode set to [{}]",
                layerWiseConfigurations.getTrainingWorkspaceMode(),
                layerWiseConfigurations.getInferenceWorkspaceMode(),
                layerWiseConfigurations.getCacheMode());

        int nLayers = getnLayers();

        if (nLayers < 1)
            throw new IllegalStateException("Unable to create network: number of layers is less than 1");

        if (this.layers == null || this.layers[0] == null) {
            if (this.layers == null)
                this.layers = new Layer[nLayers];

            //First: Work out total length of params
            long paramLength = 0;
            val nParamsPerLayer = new long[nLayers];
            for (int i = 0; i < nLayers; i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                nParamsPerLayer[i] = conf.getLayer().initializer().numParams(conf);
                paramLength += nParamsPerLayer[i];
            }

            //Create parameters array, if required
            boolean initializeParams;
            if (parameters != null) {
                if (!parameters.isRowVectorOrScalar())
                    throw new IllegalArgumentException("Invalid parameters: should be a row vector");
                if (parameters.length() != paramLength)
                    throw new IllegalArgumentException("Invalid parameters: expected length " + paramLength
                            + ", got length " + parameters.length());

                if (cloneParametersArray)
                    flattenedParams = parameters.dup();
                else
                    flattenedParams = parameters;

                initializeParams = false;
            } else if(paramLength > 0){
                flattenedParams = Nd4j.create(1, paramLength);
                initializeParams = true;
            } else {
                //Edge case: 0 params in network
                flattenedParams = null;
                initializeParams = false;
            }

            //Set RNG seed, for repeatability between initializations when set
            if (initializeParams) {
                Nd4j.getRandom().setSeed(getDefaultConfiguration().getSeed());
            }

            // construct multi-layer
            long paramCountSoFar = 0;
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
                layers[i] = conf.getLayer().instantiate(conf, trainingListeners, i, paramsView, initializeParams);
                layerMap.put(conf.getLayer().getLayerName(), layers[i]);
            }
            initCalled = true;
        }

        //Set parameters in MultiLayerNetwork.defaultConfiguration for later use in BaseOptimizer.setupSearchState() etc
        defaultConfiguration.clearVariables();
        List<String> variables = defaultConfiguration.variables(false);
        for (int i = 0; i < layers.length; i++) {
            if(layers[i] == null){
                throw new IllegalStateException("Encountered null layer during initialization for layer " + i +
                        ": " + layerWiseConfigurations.getConf(i).getLayer().getClass().getSimpleName() + " initialization " +
                        "returned null layer?");
            }

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

        //Mark that input modification is allowed.
        //TODO When is it safe to NOT skip the very first layer? It's not always safe...
        // For example dropout + iterating over List<DataSet> that is used for multiple epochs...
        for( int i=1; i<layers.length; i++ ){
            layers[i].allowInputModification(true);
        }

        synchronizeIterEpochCounts();
    }

    /**
     * This method allows you to specificy GradientsAccumulator instance to be used with this model<br>
     * <br>
     * PLEASE NOTE: Do not use this method unless you understand how to use GradientsAccumulator & updates sharing.<br>
     * PLEASE NOTE: Do not use this method on standalone model
     *
     * @param accumulator    Gradient accumulator to use for the network
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
            long paramLength = 0;
            val nParamsPerLayer = new long[nLayers];
            for (int i = 0; i < nLayers; i++) {
                NeuralNetConfiguration conf = layerWiseConfigurations.getConf(i);
                nParamsPerLayer[i] = conf.getLayer().initializer().numParams(conf);
                paramLength += nParamsPerLayer[i];
            }

            if(paramLength > 0) {
                flattenedGradients = Nd4j.zeros(new long[]{1, paramLength}, 'f'); //No need to initialize, as each layer will do it each iteration anyway
            }

            long paramsSoFar = 0;
            for (int i = 0; i < layers.length; i++) {
                if (nParamsPerLayer[i] == 0)
                    continue; //This layer doesn't have any parameters...
                INDArray thisLayerGradView = flattenedGradients.get(NDArrayIndex.point(0),
                        NDArrayIndex.interval(paramsSoFar, paramsSoFar + nParamsPerLayer[i]));
                layers[i].setBackpropGradientsViewArray(thisLayerGradView);
                paramsSoFar += nParamsPerLayer[i];
            }
        }
    }

    protected INDArray activationFromPrevLayer(int curr, INDArray input, boolean training, LayerWorkspaceMgr mgr) {
        if (getLayerWiseConfigurations().getInputPreProcess(curr) != null) {
            input = getLayerWiseConfigurations().getInputPreProcess(curr).preProcess(input, getInputMiniBatchSize(), mgr);
        }

        INDArray ret = layers[curr].activate(input, training, mgr);
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

        try {
            LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces(helperWorkspaces);   //TODO

            INDArray res = input;
            for (int l = from; l <= to; l++) {
                res = this.activationFromPrevLayer(l, res, false, mgr);
            }
            return res;
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Compute all layer activations, from input to output of the output layer.
     * Note that the input is included in the list: thus feedForward(in,train).get(0) is the inputs,
     * .get(1) is the activations of layer 0, and so on.
     *
     * @param train Training: if true, perform forward pass/inference at training time. Usually, inference is performed
     *              with train = false. This impacts whether dropout etc is applied or not.
     * @return The list of activations for each layer, including the input
     */
    public List<INDArray> feedForward(INDArray input, boolean train) {
        setInput(input);
        return feedForward(train);
    }

    /**
     * Compute activations from input to output of the output layer.
     * As per {@link #feedForward(INDArray, boolean)} but using the inputs that have previously been set using {@link #setInput(INDArray)}
     *
     * @return the list of activations for each layer
     */
    public List<INDArray> feedForward(boolean train) {
        try {
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, layers.length-1,
                    input, mask, null, true);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Perform feed-forward, optionally (not) clearing the layer input arrays.<br>
     * Note: when using clearInputs=false, there can be some performance and memory overhead: this is because the arrays are
     * defined outside of workspaces (which are enabled by default) - otherwise, old/invalidated arrays could still be
     * accessed after calling this method. Consequently: Don't use clearInputs=false unless you have a use case that
     * requires them to remain after feed-forward has been completed
     *
     * @param train       training mode (true) or test mode (false)
     * @param clearInputs If false: don't clear the layer inputs
     * @return Activations from feed-forward
     */
    public List<INDArray> feedForward(boolean train, boolean clearInputs){
        try{
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, layers.length-1, input, mask, null, clearInputs);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
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
        try{
            return ffToLayerActivationsDetached(false, FwdPassType.STANDARD, false, layerNum, input, mask, null, true);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
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
        try {
            int layerVertexIdx = layers[layerNum].getIndex();
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, layerVertexIdx, input, mask, null, true);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
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
        try {
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, layerNum, input, mask, null, true);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }


    protected void validateArrayWorkspaces(LayerWorkspaceMgr mgr, INDArray array, ArrayType arrayType, int layerIdx,
                                           boolean isPreprocessor, String op){
        try{
            mgr.validateArrayLocation(arrayType, array, false, layerIdx > 0);
        } catch (ND4JWorkspaceException e){
            String layerName = layers[layerIdx].conf().getLayer().getLayerName();
            String clazz;
            if(isPreprocessor){
                clazz = layerWiseConfigurations.getInputPreProcess(layerIdx).getClass().getName();
            } else {
                clazz = layers[layerIdx].getClass().getName();
            }
            throw new IllegalStateException(op + ": array (" + arrayType + ") workspace validation failed (" +
                    (isPreprocessor ? "preprocessor" : "layer ") + layerIdx + (layerName != null ? " - layer name \"" +
                    layerName + "\"" : "") + " - class: " + clazz + ") - array is defined in incorrect workspace", e);
        }
    }

    /**
     * Feed-forward through the network - returning all array activations in a list, detached from any workspace.
     * Note that no workspace should be active externally when calling this method (an exception will be thrown
     * if a workspace is open externally)
     *
     * @param train             Training mode (true) or test/inference mode (false)
     * @param fwdPassType       Type of forward pass to perform (STANDARD or RNN_ACTIVATE_WITH_STORED_STATE only)
     * @param storeLastForTBPTT ONLY used if fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE
     * @param layerIndex        Index (inclusive) to stop forward pass at. For all layers, use numLayers-1
     * @param input             Input to the network
     * @param fMask             Feature mask array. May be null.
     * @param lMask             Label mask array. May be null.
     * @param clearInputs       Whether the layer inputs should be cleared
     * @return List of activations (including the input), detached from any workspace
     */
    protected synchronized List<INDArray> ffToLayerActivationsDetached(boolean train, @NonNull FwdPassType fwdPassType,
                                                          boolean storeLastForTBPTT, int layerIndex, @NonNull INDArray input,
                                                          INDArray fMask, INDArray lMask, boolean clearInputs){
        setInput(input);
        setLayerMaskArrays(fMask, lMask);

        //Verify that no workspace is open externally
        WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in ffToLayerActivationsDetached");

        LayerWorkspaceMgr workspaceMgr;
        WorkspaceMode wsm = (train ? layerWiseConfigurations.getTrainingWorkspaceMode() : layerWiseConfigurations.getInferenceWorkspaceMode());
        if(wsm == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .noWorkspaceFor(ArrayType.ACTIVATIONS)
                    .with(ArrayType.INPUT, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(input.isAttached()){
                //Don't leverage out of async DataSetIterator workspaces
                workspaceMgr.setNoLeverageOverride(input.data().getParentWorkspace().getId());
            }

            if(!clearInputs){
                workspaceMgr.setScopedOutFor(ArrayType.INPUT);
            }
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        List<INDArray> out = new ArrayList<>();
        out.add(workspaceMgr.leverageTo(ArrayType.INPUT, input));    //Should  be unnecessary (and no op), if layer is implemented correctly

        for( int i=0; i<=layerIndex; i++ ){
            try(MemoryWorkspace wsFFWorking = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)){
                if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
                    input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input, getInputMiniBatchSize(), workspaceMgr);
                    //Validation: Exception if invalid (bad preprocessor implementation)
                    validateArrayWorkspaces(workspaceMgr, input, ArrayType.ACTIVATIONS, i, true, "Feed forward to layer (inference)");
                }

                if(fwdPassType == FwdPassType.STANDARD){
                    input = layers[i].activate(input, train, workspaceMgr);
                } else if (fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE) {
                    if (layers[i] instanceof RecurrentLayer) {
                        input = ((RecurrentLayer) layers[i]).rnnActivateUsingStoredState(input, train,
                                storeLastForTBPTT, workspaceMgr);
                    } else if (layers[i] instanceof MultiLayerNetwork) {
                        List<INDArray> temp = ((MultiLayerNetwork) layers[i]).rnnActivateUsingStoredState(input, train, storeLastForTBPTT);
                        input = temp.get(temp.size() - 1);
                    } else {
                        input = layers[i].activate(input, train, workspaceMgr);
                    }
                } else {
                    throw new IllegalStateException("Forward pass type not supported for this method: " + fwdPassType);
                }

                //Validation: Exception if invalid (bad layer implementation)
                validateArrayWorkspaces(workspaceMgr, input, ArrayType.ACTIVATIONS, i, false, "Feed forward to layer (inference)");

                out.add(input);
            }
            if(clearInputs) {
                layers[i].clear();
            }
        }

        return out;
    }

    /**
     * Feed-forward through the network at training time - returning a list of all activations in a workspace (WS_ALL_LAYERS_ACT)
     * if workspaces are enabled for training; or detached if no workspaces are used.<br>
     * Note: if using workspaces for training, this method requires that WS_ALL_LAYERS_ACT is open externally.<br>
     * If using NO workspaces, requires that no external workspace is open<br>
     * Note that this method does NOT clear the inputs to each layer - instead, they are in the WS_ALL_LAYERS_ACT workspace
     * for use in later backprop.
     *
     * @param layerIndex        Index (inclusive) to stop forward pass at. For all layers, use numLayers-1
     * @param fwdPassType       Type of forward pass to perform (STANDARD or RNN_ACTIVATE_WITH_STORED_STATE only)
     * @param storeLastForTBPTT ONLY used if fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE
     * @param input             Input to network
     * @param fMask             Feature mask array. May be null
     * @param lMask             Label mask aray. May be null.
     * @return
     */
    protected synchronized List<INDArray> ffToLayerActivationsInWs(int layerIndex, @NonNull FwdPassType fwdPassType, boolean storeLastForTBPTT,
                                                      @NonNull INDArray input, INDArray fMask, INDArray lMask){
        setInput(input);
        setLayerMaskArrays(fMask, lMask);

        LayerWorkspaceMgr workspaceMgr;
        if(layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in ffToLayerActivationsInWs when training workspace is set to NONE");
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(input.isAttached()){
                //Don't leverage out of async DataSetIterator workspaces
                workspaceMgr.setNoLeverageOverride(input.data().getParentWorkspace().getId());
            }

            if(layerWiseConfigurations.getCacheMode() != CacheMode.NONE){
                //For now: store cache mode activations in activations workspace
                workspaceMgr.setWorkspace(ArrayType.FF_CACHE, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG);
                workspaceMgr.setWorkspace(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG);
            }

            WorkspaceUtils.assertOpenAndActive(WS_ALL_LAYERS_ACT, "ffToLayerActivationsInWs method requires workspace WS_ALL_LAYERS_ACT to be open");
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        List<INDArray> out = new ArrayList<>();
        out.add(workspaceMgr.leverageTo(ArrayType.INPUT, input));    //Probably unnecessary usually

        for( int i=0; i<=layerIndex; i++ ){
            try(MemoryWorkspace wsFFWorking = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)){
                if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
                    input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input, getInputMiniBatchSize(), workspaceMgr);
                    //Validation: Exception if invalid (bad preprocessor implementation)
                    validateArrayWorkspaces(workspaceMgr, input, ArrayType.ACTIVATIONS, i, true, "Feed forward to layer (training)");
                }

                if(fwdPassType == FwdPassType.STANDARD){
                    input = layers[i].activate(input, true, workspaceMgr);
                } else if(fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE){
                    if (layers[i] instanceof RecurrentLayer) {
                        input = ((RecurrentLayer) layers[i]).rnnActivateUsingStoredState(input, true, storeLastForTBPTT, workspaceMgr);
                    } else if (layers[i] instanceof MultiLayerNetwork) {
                        List<INDArray> temp = ((MultiLayerNetwork) layers[i]).rnnActivateUsingStoredState(input, true, storeLastForTBPTT);
                        input = temp.get(temp.size() - 1);
                    } else {
                        input = layers[i].activate(input, true, workspaceMgr);
                    }
                } else {
                    throw new IllegalStateException("FwdPassType not supported for this method: " + fwdPassType);
                }

                if(input == null){
                    throw new IllegalStateException("Layer " + i + " returned null activations");
                }

                //Validation: Exception if invalid (bad layer implementation)
                validateArrayWorkspaces(workspaceMgr, input, ArrayType.ACTIVATIONS, i, false, "Feed forward to layer (training)");
                validateArrayWorkspaces(workspaceMgr, layers[i].input(), ArrayType.INPUT, i, false, "Feed forward to layer (training)");

                out.add(input);
            }
        }

        return out;
    }

    /**
     * Provide the output of the specified layer, detached from any workspace. This is most commonly used at inference/test
     * time, and is more memory efficient than {@link #ffToLayerActivationsDetached(boolean, FwdPassType, boolean, int, INDArray, INDArray, INDArray, boolean)}
     * and {@link #ffToLayerActivationsInWs(int, FwdPassType, boolean, INDArray, INDArray, INDArray)}.<br>
     * This method clears all layer inputs.
     *
     * NOTE: in general, no workspaces should be activated externally for this method!
     * This method handles the workspace activation as required
     *
     * @param train             Training mode (true) or test/inference mode (false)
     * @param fwdPassType       Type of forward pass to perform (STANDARD, RNN_TIMESTEP or RNN_ACTIVATE_WITH_STORED_STATE)
     * @param layerIndex        Index (inclusive) to stop forward pass at. For all layers, use numLayers-1
     * @param input             Input to the network
     * @param featureMask       Input/feature mask array. May be null.
     * @param labelsMask        Labels mask array. May be null
     * @param outputWorkspace   Optional - if provided, outputs should be placed in this workspace. NOTE: this workspace
     *                          must be open
     * @return                  Output of the specified layer, detached from any workspace
     */
    protected INDArray outputOfLayerDetached(boolean train, @NonNull FwdPassType fwdPassType, int layerIndex, @NonNull INDArray input,
                                             INDArray featureMask, INDArray labelsMask, MemoryWorkspace outputWorkspace){
        setInput(input);
        setLayerMaskArrays(featureMask, labelsMask);

        /*
        Idea here: we want to minimize memory, and return only the final array
        Approach to do this: keep activations in memory only as long as we need them.
        In MultiLayerNetwork, the output activations of layer X are used as input to layer X+1
        Which means: the workspace for layer X has to be open for both layers X and X+1 forward pass.

        Here, we'll use two workspaces for activations:
        1. For even index layers, activations WS that opens on start of even layer fwd pass, closes at end of odd layer fwd pass
        2. For odd index layers, activations WS that opens on start of odd layer fwd pass, closes at end of even layer fwd pass

        Additionally, we'll reconfigure the workspace manager for the *final* layer, so that we don't have to detach
         */
        if(outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in outputOfLayerDetached", true);
        } else {
            Preconditions.checkState(outputWorkspace.isScopeActive(), "Workspace \"" + outputWorkspace.getId() +
                    "\" was provided for the network/layer outputs. When provided, this workspace must be opened before " +
                    "calling the output method; furthermore, closing the workspace is the responsibility of the user");
        }

        LayerWorkspaceMgr mgrEven;
        LayerWorkspaceMgr mgrOdd;

        WorkspaceMode wsm = train ? layerWiseConfigurations.getTrainingWorkspaceMode() : layerWiseConfigurations.getInferenceWorkspaceMode();
        if(wsm == WorkspaceMode.NONE){
            mgrEven = LayerWorkspaceMgr.noWorkspaces();
            mgrOdd = mgrEven;

            //Check for external workspace - doesn't make sense to have one with workspace mode NONE
            if(outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)){
                throw new IllegalStateException("Workspace \"" + outputWorkspace.getId() +
                        "\" was provided for the network/layer outputs, however " + (train ? "training" : "inference") +
                        " workspace mode is set to NONE. Cannot put output activations into the specified workspace if" +
                        "workspaces are disabled for the network. use getConfiguration().setTraining/InferenceWorkspaceMode(WorkspaceMode.ENABLED)");
            }
        } else {
            mgrEven = LayerWorkspaceMgr.builder()
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_LAYER_ACT_1, WS_LAYER_ACT_X_CONFIG)
                    .with(ArrayType.INPUT, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG)            //Inputs should always be in the previous WS
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            mgrOdd = LayerWorkspaceMgr.builder()
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG)
                    .with(ArrayType.INPUT, WS_LAYER_ACT_1, WS_LAYER_ACT_X_CONFIG)            //Inputs should always be in the previous WS
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();
        }
        mgrEven.setHelperWorkspacePointers(helperWorkspaces);
        mgrOdd.setHelperWorkspacePointers(helperWorkspaces);

        MemoryWorkspace wsActCloseNext = null;
        MemoryWorkspace temp = null;
        MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        try {
            for (int i = 0; i <= layerIndex; i++) {
                LayerWorkspaceMgr mgr = (i % 2 == 0 ? mgrEven : mgrOdd);

                //Edge case: for first layer with dropout, inputs can't be in previous workspace (as it hasn't been opened yet)
                //Hence: put inputs in working memory
                if(i == 0 && wsm != WorkspaceMode.NONE){
                    mgr.setWorkspace(ArrayType.INPUT, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG);
                }

                try (MemoryWorkspace wsFFWorking = mgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) { //Working memory: opened/closed once per layer
                    //Activations workspaces: opened/closed every second layer.
                    //So mgrEven (WS_LAYER_ACT_1) open at start of 0, 2, 4, 8; closed at end of 1, 3, 5, 7 etc
                    //and mgrOdd (WS_LAYER_ACT_2) opened at start of 1, 3, 5, 7; closed at end of 2, 4, 6, 8 etc
                    temp = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS);

                    //Note that because we're opening activation workspaces not in a simple nested order, we'll manually
                    // override the previous workspace setting. Otherwise, when we close these workspaces, the "current"
                    // workspace may be set to the incorrect one
                    temp.setPreviousWorkspace(initialWorkspace);


                    if(i == 0 && input.isAttached()){
                        //Don't leverage out of async DataSetIterator workspaces
                        mgr.setNoLeverageOverride(input.data().getParentWorkspace().getId());
                    }

                    if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
                        input = getLayerWiseConfigurations().getInputPreProcess(i).preProcess(input, getInputMiniBatchSize(), mgr);
                        //Validation: Exception if invalid (bad preprocessor implementation)
                        validateArrayWorkspaces(mgr, input, ArrayType.ACTIVATIONS, i, true, "Output of layer (inference)");
                    }

                    if ( i == layerIndex ) {
                        if(outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)){
                            //Place activations in user-specified workspace
                            mgr.setWorkspace(ArrayType.ACTIVATIONS, outputWorkspace.getId(), outputWorkspace.getWorkspaceConfiguration());
                        } else {
                            //Final activations: should be detached
                            mgr.setScopedOutFor(ArrayType.ACTIVATIONS);
                        }
                    }

                    if(fwdPassType == FwdPassType.STANDARD){
                        //Standard feed-forward case
                        input = layers[i].activate(input, train, mgr);
                    } else if(fwdPassType == FwdPassType.RNN_TIMESTEP){
                        //rnnTimeStep case
                        if (layers[i] instanceof RecurrentLayer) {
                            input = ((RecurrentLayer) layers[i]).rnnTimeStep(reshapeTimeStepInput(input), mgr);
                        } else if(layers[i] instanceof BaseWrapperLayer && ((BaseWrapperLayer)layers[i]).getUnderlying() instanceof RecurrentLayer){
                            RecurrentLayer rl = ((RecurrentLayer) ((BaseWrapperLayer)layers[i]).getUnderlying());
                            input = rl.rnnTimeStep(reshapeTimeStepInput(input), mgr);
                        } else if (layers[i] instanceof MultiLayerNetwork) {
                            input = ((MultiLayerNetwork) layers[i]).rnnTimeStep(reshapeTimeStepInput(input));
                        } else {
                            input = layers[i].activate(input, false, mgr);
                        }
                    } else {
                        throw new IllegalArgumentException("Unsupported forward pass type for this method: " + fwdPassType);
                    }
                    layers[i].clear();
                    //Validation: Exception if invalid (bad layer implementation)
                    validateArrayWorkspaces(mgr, input, ArrayType.ACTIVATIONS, i, false, "Output of layer (inference)");

                    if(wsActCloseNext != null){
                        wsActCloseNext.close();
                    }
                    wsActCloseNext = temp;
                    temp = null;
                }

                //Edge case: for first layer with dropout, inputs can't be in previous workspace (as it hasn't been opened yet)
                //Hence: put inputs in working memory -> set back to default for next use of workspace mgr
                if(i == 0 && wsm != WorkspaceMode.NONE){
                    mgr.setWorkspace(ArrayType.INPUT, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG);            //Inputs should always be in the previous WS
                }
            }

        } finally {
            if(wsActCloseNext != null){
                wsActCloseNext.close();
            }
            if(temp != null){
                //Should only be non-null on exception
                while(temp.isScopeActive()){
                    //For safety, should never occur in theory: a single close() call may not be sufficient, if
                    // workspace scope was borrowed and not properly closed when exception occurred
                    temp.close();
                }
            }

            Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);

            if(outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
                WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active at the end of outputOfLayerDetached", true);
            } else {
                Preconditions.checkState(outputWorkspace.isScopeActive(), "Expected output workspace to still be open" +
                        "at end of outputOfLayerDetached, but it is closed. This suggests an implementation or layer workspace problem");
            }
        }

        return input;
    }

    private INDArray reshapeTimeStepInput(INDArray input) {
        if (input.rank() == 2) { // dynamically reshape to 3D input with one time-step.
            long[] inShape = input.shape();
            input = input.reshape(inShape[0], inShape[1], 1);
        }
        return input;
    }

    /**
     * Compute activations of all layers from input (inclusive) to output of the final/output layer.
     * Equivalent to calling {@link #feedForward(boolean)} with train=false
     *
     * @return the list of activations for each layer, including the input
     */
    public List<INDArray> feedForward() {
        return feedForward(false);
    }

    /**
     * Compute activations of all layers from input (inclusive) to output of the final/output layer.
     * Equivalent to calling {@link #feedForward(INDArray, boolean)} with train = false
     *
     * @return the list of activations for each layer, including the input
     */
    public List<INDArray> feedForward(INDArray input) {
        if (input == null)
            throw new IllegalStateException("Unable to perform feed forward; no input found");
        setInput(input);
        return feedForward();
    }

    /**
     * Compute the activations from the input to the output layer, given mask arrays (that may be null)
     * The masking arrays are used in situations such an one-to-many and many-to-one rucerrent neural network (RNN)
     * designs, as well as for supporting time series of varying lengths within the same minibatch for RNNs.
     * Other than mask arrays, this is equivalent to calling {@link #feedForward(INDArray, boolean)} with train = false
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

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }


    /**
     * Clone the MultiLayerNetwork
     * @return A cloned MultiLayerNetwork with a copy of the configuration, parameters and updater identical to the current network.
     */
    @Override
    public MultiLayerNetwork clone() {
        if(!initCalled)
            init();
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

    protected boolean hasAFrozenLayer() {
        for (int i = 0; i < layers.length - 1; i++) {
            if (layers[i] instanceof FrozenLayer)
                return true;
        }
        return false;
    }


    /**
     * Returns a 1 x m vector where the vector is composed of a flattened vector of all of the parameters (weights and
     * biases etc) for all parameters in the network. Note that this method is generally reserved for developer and
     * internal use - see {@link #getParam(String)} and {@link #paramTable()} for a more useful/interpretable
     * representation of the parameters.<br>
     * Note that with backwardsOnly = false the parameter vector is not a copy, and changes to the returned INDArray
     * will impact the network parameters.
     *
     * @param backwardOnly Return a copy of the parameters excluding any parameters used only for unsupervised layers'
     *                     unsupervised training (such as decoder parameters in an autoencoder layer
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
     * Returns a 1 x m vector where the vector is composed of a flattened vector of all of the parameters in the network.<br>
     * See {@link #getParam(String)} and {@link #paramTable()} for a more useful/interpretable representation of the parameters.<br>
     * Note that the parameter vector is not a copy, and changes to the returned INDArray will impact the network parameters.
     *
     * @return the parameters for this neural net
     */
    @Override
    public INDArray params() {
        return flattenedParams;
    }

    /**
     * Set the parameters for this model.
     * This expects a linear ndarray which then be unpacked internally relative to the expected ordering of the model.<br>
     * See also: {@link #setParamTable(Map)} and {@link #setParam(String, INDArray)}
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
                long range = layer.numParams();
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

    @Override
    public TrainingConfig getConfig() {
        throw new UnsupportedOperationException("Not supported");
    }

    /**
     * Returns the number of parameters in the network
     *
     * @return The number of parameters
     */
    @Override
    public long numParams() {
        if(!isInitCalled())
            init();
        return flattenedParams == null ? 0 : flattenedParams.length();  //Maybe nul for 0 params net
    }

    /**
     * Returns the number of parameters in the network
     *
     * @param  backwards If true: exclude any parameters uned only in unsupervised layerwise training (such as the decoder
     *                   parameters in an autoencoder)
     * @return The number of parameters
     */
    @Override
    public long numParams(boolean backwards) {
        int length = 0;
        for (int i = 0; i < layers.length; i++)
            length += layers[i].numParams(backwards);

        return length;
    }

    /**
     * Sets the input and labels and returns the F1 score for the prediction with respect to the true labels
     *
     * @param data the data to score
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(org.nd4j.linalg.dataset.api.DataSet data) {
        return f1Score(data.getFeatures(), data.getLabels());
    }

    /**
     * Perform minibatch training on all minibatches in the DataSetIterator, for the specified number of epochs.
     * Equvalent to calling {@link #fit(DataSetIterator)} numEpochs times in a loop
     *
     * @param iterator  Training data (DataSetIterator). Iterator must support resetting
     * @param numEpochs Number of training epochs, >= 1
     */
    public void fit(@NonNull DataSetIterator iterator, int numEpochs){
        Preconditions.checkArgument(numEpochs > 0, "Number of epochs much be > 0. Got numEpochs = %s", numEpochs);
        Preconditions.checkArgument(numEpochs == 1 || iterator.resetSupported(), "Cannot perform multiple epochs training using" +
                "iterator thas does not support resetting (iterator.resetSupported() returned false)");

        for(int i=0; i<numEpochs; i++ ){
            fit(iterator);
        }
    }

    /**
     * Perform minibatch training on all minibatches in the DataSetIterator for 1 epoch.<br>
     * Note that this method does not do layerwise  pretraining.<br>
     * For pretraining use method pretrain.. {@link #pretrain(DataSetIterator)}<br>
     * @param iterator Training data (DataSetIterator)
     */
    @Override
    public void fit(DataSetIterator iterator) {
        try{
            fitHelper(iterator);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private synchronized void fitHelper(DataSetIterator iterator){
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

        LayerWorkspaceMgr workspaceMgr;
        if(getLayerWiseConfigurations().getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    //Note for updater working memory, we have the option to re-use WS_ALL_LAYERS_ACT or FF/BP_WORKING_MEM
                    // as these should be closed by the time updaters are executed
                    //Generally, WS_ALL_LAYERS_ACT will be the larger of the two, so we'll use this
                    .with(ArrayType.UPDATER_WORKING_MEM, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        update(TaskUtils.buildTask(iter));
        if (!iter.hasNext() && iter.resetSupported()) {
            iter.reset();
        }
        long time1 = System.currentTimeMillis();
        while (iter.hasNext()) {

            DataSet next = iter.next();
            long time2 = System.currentTimeMillis();

            lastEtlTime.set((time2 - time1));

            if (next.getFeatures() == null || next.getLabels() == null)
                break;

            // TODO: basically we want to wrap internals of this loop into workspace


            boolean hasMaskArrays = next.hasMaskArrays();

            if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
                doTruncatedBPTT(next.getFeatures(), next.getLabels(), next.getFeaturesMaskArray(),
                        next.getLabelsMaskArray(), workspaceMgr);
            } else {
                if (hasMaskArrays)
                    setLayerMaskArrays(next.getFeaturesMaskArray(), next.getLabelsMaskArray());

                setInput(next.getFeatures());
                setLabels(next.getLabels());

                if (solver == null) {
                    try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                        solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                                .build();
                    }
                }

                //TODO CACHE
                solver.optimize(workspaceMgr);
            }

            if (hasMaskArrays)
                clearLayerMaskArrays();

            time1 = System.currentTimeMillis();
            synchronizeIterEpochCounts();
        }

        if (!trainingListeners.isEmpty()) {
            for (TrainingListener tl : trainingListeners) {
                tl.onEpochEnd(this);
            }
        }

        clearLayersStates();

        if (destructable)
            ((AsyncDataSetIterator) iter).shutdown();

        incrementEpochCount();
    }

    /**
     * Calculate parameter gradients and input activation gradients given the input and labels, and optionally mask arrays
     *
     * @param features  Features for gradient calculation
     * @param label     Labels for gradient
     * @param fMask     Features mask array (may be null)
     * @param labelMask Label mask array (may be null)
     * @return A pair of gradient arrays: parameter gradients (in Gradient object) and input activation gradients
     */
    public Pair<Gradient,INDArray> calculateGradients(@NonNull INDArray features, @NonNull INDArray label,
                                                      INDArray fMask, INDArray labelMask) {
        try{
            return calculateGradientsHelper(features, label, fMask, labelMask);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private Pair<Gradient,INDArray> calculateGradientsHelper(INDArray features, INDArray label, INDArray fMask,
                                                             INDArray labelMask){
        setInput(features);
        setLabels(label);
        setLayerMaskArrays(fMask, labelMask);

        LayerWorkspaceMgr mgr;
        if(layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            mgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            mgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(layerWiseConfigurations.getCacheMode() != null){
                //For now: store cache mode activations in activations workspace
                mgr.setWorkspace(ArrayType.FF_CACHE, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG);
            }
        }
        mgr.setHelperWorkspacePointers(helperWorkspaces);

        //Calculate activations (which are stored in each layer, and used in backprop)
        try(MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
            //First: do a feed-forward through the network
            //Note that we don't actually need to do the full forward pass through the output layer right now; but we do
            // need the input to the output layer to be set (such that backprop can be done)
            List<INDArray> activations = ffToLayerActivationsInWs(layers.length - 2, FwdPassType.STANDARD, false, input, mask, fMask);
            if (!trainingListeners.isEmpty()) {
                //TODO: We possibly do want output layer activations in some cases here...
                for (TrainingListener tl : trainingListeners) {
                    tl.onForwardPass(this, activations);
                }
            }
            INDArray inputToOutputLayer = activations.get(activations.size() - 1);
            if (layerWiseConfigurations.getInputPreProcess(layers.length - 1) != null) {
                inputToOutputLayer = layerWiseConfigurations.getInputPreProcess(layers.length - 1)
                        .preProcess(inputToOutputLayer, getInputMiniBatchSize(), mgr);
                //Validate activations location
            }
            getOutputLayer().setInput(inputToOutputLayer, mgr);

            Pair<Gradient,INDArray> p = calcBackpropGradients(null, true, false, true);
            if(p.getSecond() != null){
                p.setSecond( p.getSecond().detach());
            }
            return p;
        }
    }

    /** Calculate gradients and errors. Used in two places:
     * (a) backprop (for standard multi layer network learning)
     * (b) backpropGradient (layer method, for when MultiLayerNetwork is used as a layer)
     * @param epsilon Errors (technically errors .* activations). Not used if withOutputLayer = true
     * @param withOutputLayer if true: assume last layer is output layer, and calculate errors based on labels. In this
     *                        case, the epsilon input is not used (may/should be null).
     *                        If false: calculate backprop gradients
     * @param returnInputActGrad If true: terun the input activation gradients (detached). False: don't return
     * @return Gradients and the error (epsilon) at the input
     */
    protected Pair<Gradient, INDArray> calcBackpropGradients(INDArray epsilon, boolean withOutputLayer, boolean tbptt,
                                                             boolean returnInputActGrad) {
        if (flattenedGradients == null) {
            initGradientsView();
        }
        String multiGradientKey;
        Gradient gradient = new DefaultGradient(flattenedGradients);

        LayerWorkspaceMgr mgrEven;
        LayerWorkspaceMgr mgrOdd;

        if(layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            mgrEven = LayerWorkspaceMgr.noWorkspaces();
            mgrOdd = mgrEven;
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in calcBackpropGradients when " +
                    "training workspace is set to none");
        } else {
            /*
            Workspaces for backprop in MLN share some features with outputOfLayerDetached, in terms of the
            "two alternating workspaces" idea (but for activation gradients here, instead of activations there).

            Workspace design for backprop:
            First: we calculate all activations, and ensure they are in WS_ALL_LAYERS_ACT. We assume this is done
                   EXTERNALLY to this method
            Then: we iterate backwards over layers.

            Activations gradient workspaces: opened/closed every second layer.
            mgrEven (WS_LAYER_ACT_1) activation grad WS opens at start of 8, 4, 2, 0; closed at end of 7, 5, 3, 1 etc
            mgrOdd (WS_LAYER_ACT_2) activation grad WS opens at start of 7, 3, 5, 1; closed at end of 6, 4, 2, 0 etc

             */

            mgrEven = LayerWorkspaceMgr.builder()
                    //Activations in context of backprop (preOut methods etc) are not used outside of the layer itself
                    .with(ArrayType.ACTIVATIONS, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG) //Usually not required here. Exception: OutputLayer dropout
                    .with(ArrayType.ACTIVATION_GRAD, WS_LAYER_ACT_1, WS_LAYER_ACT_X_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            mgrOdd = LayerWorkspaceMgr.builder()
                    //Activations in context of backprop (preOut methods etc) are not used outside of the layer itself
                    .with(ArrayType.ACTIVATIONS, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG) //Usually not required here. Exception: OutputLayer dropout
                    .with(ArrayType.ACTIVATION_GRAD, WS_LAYER_ACT_2, WS_LAYER_ACT_X_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(epsilon == null) {
                //If epsilon is non-null: external errors use case -> inputs are already detached
                WorkspaceUtils.assertOpenActiveAndCurrent(WS_ALL_LAYERS_ACT, "calcBackpropGradients method requires workspace WS_ALL_LAYERS_ACT" +
                        " to be open when workspaces are used");
            }
        }
        mgrEven.setHelperWorkspacePointers(helperWorkspaces);
        mgrOdd.setHelperWorkspacePointers(helperWorkspaces);

        //calculate and apply the backward gradient for every layer
        /*
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


        Pair<Gradient, INDArray> currPair = null;
        MemoryWorkspace wsActGradCloseNext = null;
        MemoryWorkspace wsActGradTemp = null;
        MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        try {
            for (int i = layers.length - 1; i >= 0; i--) {
                if (layers[i] instanceof FrozenLayer) {
                    break;
                }

                LayerWorkspaceMgr workspaceMgr = (i % 2 == 0 ? mgrEven : mgrOdd);

                if (withOutputLayer && i == layers.length - 1) {
                    if (!(getOutputLayer() instanceof IOutputLayer)) {
                        log.warn("Warning: final layer isn't output layer. You cannot use backprop without an output layer.");
                        return null;
                    }

                    IOutputLayer outputLayer = (IOutputLayer) getOutputLayer();
                    if (labels == null && outputLayer.needsLabels())
                        throw new IllegalStateException("No labels found");
                    outputLayer.setLabels(labels);
                }

                //Open activation gradients WS *then* BP working memory, so BP working memory is opened last for use in layers
                wsActGradTemp = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
                try(MemoryWorkspace wsBPWorking = workspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)){

                    //Note that because we're opening activation workspaces not in a simple nested order, we'll manually
                    // override the previous workspace setting. Otherwise, when we close these workspaces, the "current"
                    // workspace may be set to the incorrect one
                    wsActGradTemp.setPreviousWorkspace(initialWorkspace);
                    wsBPWorking.setPreviousWorkspace(initialWorkspace);

                    INDArray eps = (i == layers.length - 1 ? epsilon : currPair.getRight());  //eps is null for OutputLayer

                    if(!tbptt){
                        //Standard case
                        currPair = layers[i].backpropGradient(eps, workspaceMgr);
                    } else {
                        //TBPTT gradient
                        if (layers[i] instanceof RecurrentLayer) {
                            currPair = ((RecurrentLayer) layers[i]).tbpttBackpropGradient(currPair.getSecond(),
                                    layerWiseConfigurations.getTbpttBackLength(), workspaceMgr);
                        } else {
                            currPair = layers[i].backpropGradient(currPair.getSecond(), workspaceMgr);
                        }
                    }

                    if(currPair.getSecond() != null) {
                        //Edge case: may be null for Embedding layer, for example
                        validateArrayWorkspaces(workspaceMgr, currPair.getSecond(), ArrayType.ACTIVATION_GRAD, numLayers - 1,
                                false, "Backprop");
                    }

                    for (Map.Entry<String, INDArray> entry : currPair.getFirst().gradientForVariable().entrySet()) {
                        String origName = entry.getKey();
                        multiGradientKey = String.valueOf(i) + "_" + origName;
                        gradientList.addLast(new Triple<>(multiGradientKey, entry.getValue(),
                                currPair.getFirst().flatteningOrderForVariable(origName)));
                    }
                    if (getLayerWiseConfigurations().getInputPreProcess(i) != null) {
                        currPair = new Pair<>(currPair.getFirst(),
                                this.layerWiseConfigurations.getInputPreProcess(i)
                                        .backprop(currPair.getSecond(), getInputMiniBatchSize(), workspaceMgr));
                        if (i > 0 && currPair.getSecond() != null){
                            validateArrayWorkspaces(workspaceMgr, currPair.getSecond(), ArrayType.ACTIVATION_GRAD, i,
                                    true, "Backprop");
                        }
                    }

                    if(i == 0 ){
                        if(returnInputActGrad && currPair.getSecond() != null){
                            currPair.setSecond(currPair.getSecond().detach());
                        } else {
                            currPair.setSecond(null);
                        }
                    }

                    if(wsActGradCloseNext != null){
                        wsActGradCloseNext.close();
                    }
                    wsActGradCloseNext = wsActGradTemp;
                    wsActGradTemp = null;
                }
            }
        } finally {
            if(wsActGradCloseNext != null){
                wsActGradCloseNext.close();
            }
            if(wsActGradTemp != null){
                //Should only be non-null on exception
                wsActGradTemp.close();
            }
            Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);
        }

        if (layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE) {
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in calcBackpropGradients when " +
                    "training workspace is set to none");
        } else {
            if(epsilon == null) {
                //If epsilon != null: external errors use case (inputs are detached instead)
                WorkspaceUtils.assertOpenActiveAndCurrent(WS_ALL_LAYERS_ACT, "calcBackpropGradients: WS_ALL_LAYERS_ACT is no" +
                        " longer the currently open/active workspace");
            }
        }

        //Add gradients to Gradients (map), in correct order
        for (Triple<String, INDArray, Character> triple : gradientList) {
            gradient.setGradientFor(triple.getFirst(), triple.getSecond(), triple.getThird());
        }

        return new Pair<>(gradient, currPair.getSecond());
    }

    protected void doTruncatedBPTT(INDArray input, INDArray labels, INDArray featuresMaskArray,
                                   INDArray labelsMaskArray, LayerWorkspaceMgr workspaceMgr) {
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
        val timeSeriesLength = input.size(2);
        long nSubsets = timeSeriesLength / fwdLen;
        if (timeSeriesLength % fwdLen != 0)
            nSubsets++; //Example: 100 fwdLen with timeSeriesLength=120 -> want 2 subsets (1 of size 100, 1 of size 20)

        rnnClearPreviousState();

        for (int i = 0; i < nSubsets; i++) {
            long startTimeIdx = i * fwdLen;
            long endTimeIdx = startTimeIdx + fwdLen;
            if (endTimeIdx > timeSeriesLength)
                endTimeIdx = timeSeriesLength;

            // FIXME: int cast
            INDArray[] subsets = getSubsetsForTbptt((int) startTimeIdx, (int) endTimeIdx, input, labels,
                    featuresMaskArray, labelsMaskArray);

            setInput(subsets[0]);
            setLabels(subsets[1]);
            setLayerMaskArrays(subsets[2], subsets[3]);

            if (solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                            .build();
                }
            }
            solver.optimize(workspaceMgr);

            //Finally, update the state of the RNN layers:
            updateRnnStateWithTBPTTState();
        }

        rnnClearPreviousState();
        clearLayerMaskArrays();
    }

    private INDArray[] getSubsetsForTbptt(int startTimeIdx, int endTimeIdx, INDArray input, INDArray labels,
                                          INDArray fMask, INDArray lMask ){
        INDArray[] out = new INDArray[4];
        out[0] = input.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        out[1] = labels.get(NDArrayIndex.all(), NDArrayIndex.all(),
                NDArrayIndex.interval(startTimeIdx, endTimeIdx));

        if (fMask != null) {
            out[2] = fMask.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        }
        if (lMask != null) {
            out[3] = lMask.get(NDArrayIndex.all(),
                    NDArrayIndex.interval(startTimeIdx, endTimeIdx));
        }

        return out;
    }

    /**
     * Intended for internal/developer use
     */
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

    /**
     * Get the {@link TrainingListener}s set for this network, if any
     * @return listeners set for this network
     */
    public Collection<TrainingListener> getListeners() {
        return trainingListeners;
    }

    /**
     * @deprecated Use {@link #getListeners()}
     */
    @Deprecated
    public Collection<TrainingListener> getTrainingListeners() {
        return trainingListeners;
    }

    @Override
    public void setListeners(Collection<TrainingListener> listeners) {
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
            this.trainingListeners.addAll(listeners);
        }
    }

    /**
     * This method ADDS additional TrainingListener to existing listeners
     *
     * @param listeners
     */
    @Override
    public void addListeners(TrainingListener... listeners) {
        Collections.addAll(trainingListeners, listeners);

        // fixme this is wrong, since it removes existing listeners from the solver
        if (solver != null) {
            solver.setListeners(this.trainingListeners);
        }
    }

    @Override
    public void setListeners(TrainingListener... listeners) {
        Collection<TrainingListener> cListeners = new ArrayList<>();
        //Check: user might have done setListeners(null) thinking this would clear the current listeners.
        //This results in an TrainingListener[1] with a single null value -> results in a NPE later
        if (listeners != null && listeners.length > 0) {
            for (TrainingListener i : listeners) {
                if (i != null)
                    cListeners.add(i);
            }
        }
        setListeners(cListeners);
    }

    /**
     * Usable only for classification networks in conjunction with OutputLayer. Cannot be used with RnnOutputLayer,
     * CnnLossLayer, or networks used for regression.<br>
     * To get the raw output activations of the output layer, use {@link #output(INDArray)} or similar.<br>
     * <br>
     * Equivalent to argmax(this.output(input)): Returns the predicted class indices corresponding to the predictions
     * for each example in the features array.
     *
     * @param d The input features to perform inference on
     * @return The predicted class index for each example
     */
    @Override
    public int[] predict(INDArray d) {
        INDArray output = output(d, Layer.TrainingMode.TEST);

        // FIXME: int cast
        int[] ret = new int[(int) d.size(0)];
        if (d.isRowVectorOrScalar())
            ret[0] = Nd4j.getBlasWrapper().iamax(output);
        else {
            for (int i = 0; i < ret.length; i++)
                ret[i] = Nd4j.getBlasWrapper().iamax(output.getRow(i));
        }
        return ret;
    }

    /**
     * As per {@link #predict(INDArray)} but the returned values are looked up from the list of label names
     * in the provided DataSet
     */
    @Override
    public List<String> predict(org.nd4j.linalg.dataset.api.DataSet dataSet) {
        Preconditions.checkState(dataSet.getLabelNamesList() != null, "This method can only be used when the DataSet contains a label name list");
        int[] intRet = predict(dataSet.getFeatures());
        List<String> ret = new ArrayList<>();
        for (int i = 0; i < intRet.length; i++) {
            ret.add(i, dataSet.getLabelName(intRet[i]));
        }
        return ret;
    }

    /**
     * Fit the model for one iteration on the provided data
     *
     * @param data   the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     */
    @Override
    public void fit(INDArray data, INDArray labels) {
        fit(data, labels, null, null);
    }

    /**
     * Fit the model for one iteration on the provided data
     *
     * @param features   the examples to classify (one example in each row)
     * @param labels the example labels(a binary outcome matrix)
     * @param featuresMask The mask array for the features (used for variable length time series, etc). May be null.
     * @param labelsMask The mask array for the labels (used for variable length time series, etc). May be null.
     */
    public synchronized void fit(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask) {
        try{
            fitHelper(features, labels, featuresMask, labelsMask);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private void fitHelper(INDArray features, INDArray labels, INDArray featuresMask, INDArray labelsMask){
        if(numParams() == 0){
            //No op: can't fit a network with 0 parameters
            return;
        }

        setInput(features);
        setLabels(labels);
        this.setLayerMaskArrays(featuresMask, labelsMask);
        update(TaskUtils.buildTask(features, labels));

        LayerWorkspaceMgr workspaceMgr;
        if(layerWiseConfigurations.getTrainingWorkspaceMode() == null){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    //Note for updater working memory, we have the option to re-use WS_ALL_LAYERS_ACT or FF/BP_WORKING_MEM
                    // these should be closed by the time updaters are executed
                    //Generally, WS_ALL_LAYERS_ACT will be the larger of the two, so we'll use this
                    .with(ArrayType.UPDATER_WORKING_MEM, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        if (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT) {
            doTruncatedBPTT(features, labels, featuresMask, labelsMask, workspaceMgr);
        } else {
            if (solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                }
            }
            //TODO CACHE WORKSPACE, IF USED???
            solver.optimize(workspaceMgr);
        }

        clearLayerMaskArrays();
        clearLayersStates();
        synchronizeIterEpochCounts();
    }

    @Override
    public void fit(INDArray data, LayerWorkspaceMgr workspaceMgr){
        throw new UnsupportedOperationException("Not supported: use pretrainLayer");
    }


    /**
     * Fit the model for one iteration on the provided data
     *
     * @param data the data to train on
     */
    @Override
    public void fit(org.nd4j.linalg.dataset.api.DataSet data) {
        fit(data.getFeatures(), data.getLabels(), data.getFeaturesMaskArray(), data.getLabelsMaskArray());
    }

    /**
     * Fit the model for one iteration on the provided data
     *
     * @param examples the examples to classify (one example in each row)
     * @param labels   the labels for each example (the number of labels must match
     */
    @Override
    public void fit(INDArray examples, int[] labels) {
        org.deeplearning4j.nn.conf.layers.OutputLayer layerConf =
                (org.deeplearning4j.nn.conf.layers.OutputLayer) getOutputLayer().conf().getLayer();

        // FIXME: int cast
        fit(examples, FeatureUtil.toOutcomeMatrix(labels, (int) layerConf.getNOut()));
    }


    /**
     * Perform inference on the provided input/features - i.e., perform forward pass using the provided input/features
     * and return the output of the final layer.
     *
     * @param input Input to the network
     * @param train whether the output is test or train. This mainly affect hyper parameters such as dropout and
     *              batch normalization, which have different behaviour for test vs. train
     * @return The network predictions - i.e., the activations of the final layer
     */
    public INDArray output(INDArray input, TrainingMode train) {
        return output(input, train == TrainingMode.TRAIN);
    }

    /**
     * Perform inference on the provided input/features - i.e., perform forward pass using the provided input/features
     * and return the output of the final layer.
     *
     * @param input Input to the network
     * @param train whether the output is test or train. This mainly affect hyper parameters such as dropout and
     *              batch normalization, which have different behaviour for test vs. train
     * @return The network predictions - i.e., the activations of the final layer
     */
    public INDArray output(INDArray input, boolean train) {
        return output(input, train, null, null);
    }

    /**
     * Calculate the output of the network, with masking arrays. The masking arrays are used in situations such
     * as one-to-many and many-to-one recurrent neural network (RNN) designs, as well as for supporting time series
     * of varying lengths within the same minibatch.
     */
    public INDArray output(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask) {
        return output(input, train, featuresMask, labelsMask, null);
    }

    /**
     * Get the network output, which is optionally placed in the specified memory workspace.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param input           Input to the network
     * @param train           True for train, false otherwise
     * @param outputWorkspace May be null. If not null: the workspace MUST be opened before calling this method.
     * @return The output/activations from the network (either detached or in the specified workspace if provided)
     */
    public INDArray output(INDArray input, boolean train, MemoryWorkspace outputWorkspace) {
        return output(input, train, null, null, outputWorkspace);
    }

    /**
     * Get the network output, which is optionally placed in the specified memory workspace.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param input           Input to the network
     * @param train           True for train, false otherwise
     * @param outputWorkspace May be null. If not null: the workspace MUST be opened before calling this method.
     * @return The output/activations from the network (either detached or in the specified workspace if provided)
     */
    public synchronized INDArray output(INDArray input, boolean train, INDArray featuresMask, INDArray labelsMask, MemoryWorkspace outputWorkspace) {
        try {
            return outputOfLayerDetached(train, FwdPassType.STANDARD, layers.length - 1, input, featuresMask, labelsMask, outputWorkspace);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * This method uses provided OutputAdapter to return custom object built from INDArray
     *
     * PLEASE NOTE: This method uses dedicated Workspace for output generation to avoid redundant allocations
     *
     * @param inputs Input arrays to the netwonk
     * @param inputMasks Optional input mask arrays (may be null)
     * @param labelMasks Optional label mask arrays (may be null
     * @param outputAdapter OutputAdapter<T> instance
     * @param <T> T extends Object
     * @return T instance produced by OutputAdapter
     */
    public synchronized <T> T output(@NonNull INDArray inputs, INDArray inputMasks, INDArray labelMasks, @NonNull OutputAdapter<T> outputAdapter) {
        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_ALL_LAYERS_ACT_CONFIG, WS_OUTPUT_MEM)) {
            if (outputAdapter instanceof ModelAdapter)
                return ((ModelAdapter<T>) outputAdapter).apply(this, new INDArray[]{inputs}, new INDArray[]{ inputMasks}, new INDArray[]{labelMasks});
            else
                return outputAdapter.apply(output(inputs, false, inputMasks, labelMasks, ws));
        }
    }

    /**
     * Perform inference on the provided input/features - i.e., perform forward pass using the provided input/features
     * and return the output of the final layer. Equivalent to {@link #output(INDArray, boolean)} with train=false - i.e.,
     * this method is used for inference.
     *
     * @param input Input to the network
     * @return The network predictions - i.e., the activations of the final layer
     */
    public INDArray output(INDArray input) {
        return output(input, TrainingMode.TEST);
    }

    /**
     * Generate the output for all examples/batches in the input iterator, and concatenate them into a single array.
     * See {@link #output(INDArray)}<br>
     * NOTE 1: The output array can require a considerable amount of memory for iterators with a large number of examples<br>
     * NOTE 2: This method cannot be used for variable length time series outputs, as this would require padding arrays
     * for some outputs, or returning a mask array (which cannot be done with this method). For variable length time
     * series applications, use one of the other output methods. This method also cannot be used with fully convolutional
     * networks with different output sizes (for example, segmentation on different input image sizes).
     *
     *
     * @param iterator Data to pass through the network
     * @return output for all examples in the iterator, concatenated into a
     */
    public INDArray output(DataSetIterator iterator, boolean train) {
        List<INDArray> outList = new ArrayList<>();
        long[] firstOutputShape = null;
        while (iterator.hasNext()) {
            DataSet next = iterator.next();
            INDArray features = next.getFeatures();

            if (features == null)
                continue;

            INDArray fMask = next.getFeaturesMaskArray();
            INDArray lMask = next.getLabelsMaskArray();
            INDArray output = this.output(features, train, fMask, lMask);
            outList.add(output);
            if(firstOutputShape == null){
                firstOutputShape = output.shape();
            } else {
                //Validate that shapes are the same (may not be, for some RNN variable length time series applications)
                long[] currShape = output.shape();
                Preconditions.checkState(firstOutputShape.length == currShape.length, "Error during forward pass:" +
                        "different minibatches have different output array ranks - first minibatch shape %s, last minibatch shape %s", firstOutputShape, currShape);
                for( int i=1; i<currShape.length; i++ ){    //Skip checking minibatch dimension, fine if this varies
                    Preconditions.checkState(firstOutputShape[i] == currShape[i], "Current output shape does not match first" +
                            " output array shape at position %s: all dimensions must match other than the first dimension.\n" +
                            " For variable length output size/length use cases such as for RNNs with multiple sequence lengths," +
                            " use one of the other (non iterator) output methods. First batch output shape: %s, current batch output shape: %s",
                            i, firstOutputShape, currShape);
                }
            }
        }
        return Nd4j.concat(0, outList.toArray(new INDArray[outList.size()]));
    }

    /**
     * Equivalent to {@link #output(DataSetIterator, boolean)} with train=false
     */
    public INDArray output(DataSetIterator iterator) {
        return output(iterator, false);
    }

    /**
     * Perform inference and then calculate the F1 score of the output(input) vs. the labels.
     *
     * @param input  the input to perform inference with
     * @param labels the true labels
     * @return the score for the given input,label pairs
     */
    @Override
    public double f1Score(INDArray input, INDArray labels) {
        feedForward(input);
        setLabels(labels);
        Evaluation eval = new Evaluation();
        eval.eval(labels, output(input));
        return eval.f1();
    }

    /**
     * @deprecated Will be removed in a future release
     */
    @Deprecated
    @Override
    public int numLabels() {
        return (int)labels.size(1);
    }

    /**
     * Sets the input and labels and calculates the score (value of the output layer loss function plus l1/l2 if applicable)
     * for the prediction with respect to the true labels<br>
     * This is equivalent to {@link #score(DataSet, boolean)} with training==false.
     * @param data the data to score
     * @return the score for the given input,label pairs
     * @see #score(DataSet, boolean)
     */
    public double score(DataSet data) {
        return score(data, false);
    }

    /**
     * Sets the input and labels and calculates the score (value of the output layer loss function plus l1/l2 if applicable)
     * for the prediction with respect to the true labels<br>
     * @param data data to calculate score for
     * @param training If true: score during training. If false: score at test time. This can affect the application of
     *                 certain features, such as dropout and dropconnect (which are applied at training time only)
     * @return the score (value of the loss function)
     */
    public double score(DataSet data, boolean training) {
        try{
            return scoreHelper(data, training);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private double scoreHelper(DataSet data, boolean training){
        boolean hasMaskArray = data.hasMaskArrays();
        if (hasMaskArray)
            setLayerMaskArrays(data.getFeaturesMaskArray(), data.getLabelsMaskArray());

        if (!(getOutputLayer() instanceof IOutputLayer)) {
            throw new IllegalStateException("Cannot calculate score if final layer is not an instance of IOutputLayer. " +
                    "Final layer is of type: " + getOutputLayer().getClass());
        }

        WorkspaceMode wsm = (training ? layerWiseConfigurations.getTrainingWorkspaceMode() : layerWiseConfigurations.getInferenceWorkspaceMode());
        LayerWorkspaceMgr mgr;
        if(wsm == WorkspaceMode.NONE){
            mgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            mgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    //TODO we can probably optimize this
                    .noWorkspaceFor(ArrayType.ACTIVATIONS)
                    .noWorkspaceFor(ArrayType.INPUT)
                    .build();
        }
        mgr.setHelperWorkspacePointers(helperWorkspaces);

        INDArray inputToOutputLayer = outputOfLayerDetached(training, FwdPassType.STANDARD,layers.length-2, data.getFeatures(),
                data.getFeaturesMaskArray(), data.getLabelsMaskArray(), null);

        // FIXME: int cast
        IOutputLayer ol = (IOutputLayer) getOutputLayer();
        if (getLayerWiseConfigurations().getInputPreProcess(layers.length - 1) != null) {
            inputToOutputLayer = getLayerWiseConfigurations().getInputPreProcess(layers.length - 1)
                    .preProcess(inputToOutputLayer, (int) data.getFeatures().size(0), mgr);
        }
        ol.setInput(inputToOutputLayer, mgr); //Feedforward doesn't include output layer for efficiency
        ol.setLabels(data.getLabels());
        double score;
        try(MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {
            score = ol.computeScore(calcRegularizationScore(true), training, mgr);
        }

        if (hasMaskArray)
            clearLayerMaskArrays();
        clearLayersStates();

        return score;
    }

    /**
     * As per {@link #scoreExamples(DataSet, boolean)} - the outputs (example scores) for all DataSets in the iterator are concatenated
     */
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
        try{
            return scoreExamplesHelper(data, addRegularizationTerms);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private INDArray scoreExamplesHelper(DataSet data, boolean addRegularizationTerms){
        INDArray inputLast = outputOfLayerDetached(false, FwdPassType.STANDARD,layers.length-2, data.getFeatures(),
                data.getFeaturesMaskArray(), data.getLabelsMaskArray(), null);
        setLabels(data.getLabels());
        setLayerMaskArrays(data.getFeaturesMaskArray(), data.getLabelsMaskArray());

        //TODO we might want workspaces here?
        LayerWorkspaceMgr mgr = LayerWorkspaceMgr.noWorkspaces();

        INDArray out;
        if (getOutputLayer() instanceof IOutputLayer) {
            IOutputLayer ol = (IOutputLayer) getOutputLayer();
            if(layerWiseConfigurations.getInputPreProcess(layers.length-1) != null){

                // FIXME: int cast
                inputLast = layerWiseConfigurations.getInputPreProcess(layers.length-1).preProcess(inputLast,
                        (int) data.getFeatures().size(0), mgr);
            }
            ol.setLabels(data.getLabels());
            ol.setInput(inputLast, mgr);
            double r = (addRegularizationTerms ? calcRegularizationScore(true) : 0);
            out = ol.computeScoreForExamples(r, mgr);
        } else {
            throw new UnsupportedOperationException(
                    "Cannot calculate score with respect to labels without an OutputLayer");
        }

        clearLayersStates();
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
     * Score of the model (relative to the objective function) - previously calculated on the last minibatch
     *
     * @return the score of the model (relative to the objective function)
     */
    @Override
    public double score() {
        return score;
    }

    /**
     * Intended for developer/internal use
     */
    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr layerWorkspaceMgr){
        computeGradientAndScore();
    }

    public void computeGradientAndScore() {

        if (!(getOutputLayer() instanceof IOutputLayer)) {
            throw new DL4JException(
                    "Cannot calculate gradient and score with respect to labels: final layer is not an IOutputLayer. " +
                            "Final layer class: " + getOutputLayer().getClass() + ". To calculate gradients and fit a network " +
                            "using backpropagation, the final layer must be an output layer");
        }

        //Note: Workspace manager is only ose here for score calculation... other workspace managers are used in the
        // various FF/backprop methds
        LayerWorkspaceMgr mgr;
        if(layerWiseConfigurations.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            mgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            mgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(layerWiseConfigurations.getCacheMode() != null){
                //For now: store cache mode activations in activations workspace
                mgr.setWorkspace(ArrayType.FF_CACHE, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG);
            }
        }

        boolean tbptt = layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT;
        FwdPassType fwdType = (tbptt ? FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE : FwdPassType.STANDARD);
        synchronizeIterEpochCounts();

        //Calculate activations (which are stored in each layer, and used in backprop)
        try(MemoryWorkspace ws = mgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
            //First: do a feed-forward through the network
            //Note that we don't actually need to do the full forward pass through the output layer right now; but we do
            // need the input to the output layer to be set (such that backprop can be done)
            List<INDArray> activations = ffToLayerActivationsInWs(layers.length - 2, fwdType, tbptt, input, mask, null);
            if (!trainingListeners.isEmpty()) {
                //TODO: We possibly do want output layer activations in some cases here...
                for (TrainingListener tl : trainingListeners) {
                    tl.onForwardPass(this, activations);
                }
            }
            INDArray inputToOutputLayer = activations.get(activations.size() - 1);
            if (layerWiseConfigurations.getInputPreProcess(layers.length - 1) != null) {
                inputToOutputLayer = layerWiseConfigurations.getInputPreProcess(layers.length - 1)
                        .preProcess(inputToOutputLayer, getInputMiniBatchSize(), mgr);
                //Validate activations location
            }
            getOutputLayer().setInput(inputToOutputLayer, mgr);
            //Then: compute gradients
            Pair<Gradient, INDArray> pair = calcBackpropGradients(null, true, false, false);
            this.gradient = (pair == null ? null : pair.getFirst());

            //Calculate score
            try(MemoryWorkspace wsFF = mgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {
                double r = calcRegularizationScore(true);
                score = ((IOutputLayer) getOutputLayer()).computeScore(r, true, mgr);
            }

            //Listeners
            if (!trainingListeners.isEmpty()) {
                try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (TrainingListener tl : trainingListeners) {
                        tl.onBackwardPass(this);
                    }
                }
            }
        }

        //Clear the post noise/dropconnect parameters on the output layer
        getOutputLayer().clearNoiseWeightParams();
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

    @Override
    public void applyConstraints(int iteration, int epoch) {
        for(Layer l : layers){
            l.applyConstraints(iteration, epoch);
        }
    }


    /**
     * Set the input array for the network
     *
     * @param input Input array to set
     */
    public void setInput(INDArray input) {
        this.input = input;
        if (this.layers == null) {
            init();
        }
        if (input != null) {
            if (input.length() == 0)
                throw new IllegalArgumentException(
                        "Invalid input: length 0 (shape: " + Arrays.toString(input.shape()) + ")");

            // FIXME: int cast
            setInputMiniBatchSize((int) input.size(0));
        }
    }

    @Override
    public void setInput(INDArray input, LayerWorkspaceMgr mgr){
        throw new UnsupportedOperationException("Not supported");
    }

    /**
     * Get the output layer - i.e., the last layer in the netwok
     *
     * @return
     */
    public Layer getOutputLayer() {
        Layer ret = getLayers()[getLayers().length - 1];
        if (ret instanceof FrozenLayerWithBackprop) {
            ret = ((FrozenLayerWithBackprop) ret).getInsideLayer();
        }
        return ret;
    }


    /**
     * See {@link #setParams(INDArray)}
     */
    public void setParameters(INDArray params) {
        setParams(params);
    }

    /**
     * Intended for internal/developer use
     */
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
     * @param labels Labels to set
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
     * @return The layers in the network
     */
    public synchronized Layer[] getLayers() {
        return layers;
    }

    public Layer getLayer(int i) {
        Preconditions.checkArgument(i >= 0 && i < layers.length, "Invalid layer index: layer index must be 0" +
                " to %s (inclusive), got index %s", layers.length-1, i);
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
    public void clearNoiseWeightParams() {
        for(Layer l : layers){
            l.clearNoiseWeightParams();
        }
    }

    @Override
    public void allowInputModification(boolean allow) {
        throw new UnsupportedOperationException("Not supported");
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

    @Override
    public LayerHelper getHelper() {
        throw new UnsupportedOperationException("Not supported");
    }

    //==========
    //Layer methods

    @Override
    public Type type() {
        return Type.MULTILAYER;
    }


    /**
     * Equivalent to {@link #output(INDArray)} using the input set via {@link #setInput(INDArray)}
     */
    public INDArray activate(TrainingMode training) {
        return output(input, training == TrainingMode.TRAIN);
    }

    /**
     * Equivalent to {@link #output(INDArray, TrainingMode)}
     */
    public INDArray activate(INDArray input, TrainingMode training) {
        return output(input, training == TrainingMode.TRAIN);
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        if (getOutputLayer() instanceof IOutputLayer)
            throw new UnsupportedOperationException("Cannot calculate gradients based on epsilon with OutputLayer");

        return calcBackpropGradients(epsilon, false, false, true);
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
    public int getIterationCount() {
        return getLayerWiseConfigurations().getIterationCount();
    }

    @Override
    public int getEpochCount() {
        return getLayerWiseConfigurations().getEpochCount();
    }

    @Override
    public void setIterationCount(int iterationCount) {
        getLayerWiseConfigurations().setIterationCount(iterationCount);
    }

    @Override
    public void setEpochCount(int epochCount) {
        getLayerWiseConfigurations().setEpochCount(epochCount);
    }

    @Override
    public double calcRegularizationScore(boolean backpropParamsOnly){
        double scoreSum = 0.0;
        for (int i = 0; i < layers.length; i++) {
            scoreSum += layers[i].calcRegularizationScore(backpropParamsOnly);
        }
        return scoreSum;
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
    public INDArray activate(boolean training, LayerWorkspaceMgr mgr) {
        throw new UnsupportedOperationException();
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr mgr) {
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
        if(!conf().isMiniBatch())
            return 1;

        // FIXME: int cast
        return (int) input.size(0);
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
     * @see #rnnTimeStep(INDArray, MemoryWorkspace) For outputting the activations in the specified workspace
     */
    public INDArray rnnTimeStep(INDArray input) {
        return rnnTimeStep(input, null);
    }

    /**
     * See {@link #rnnTimeStep(INDArray)} for details<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param input           Input activations
     * @param outputWorkspace Output workspace. May be null
     * @return The output/activations from the network (either detached or in the specified workspace if provided)
     */
    public INDArray rnnTimeStep(INDArray input, MemoryWorkspace outputWorkspace ) {
        try {
            boolean inputIs2d = input.rank() == 2;
            INDArray out = outputOfLayerDetached(false, FwdPassType.RNN_TIMESTEP, layers.length - 1, input, null, null, outputWorkspace);
            if (inputIs2d && out.rank() == 3 && layers[layers.length - 1].type() == Type.RECURRENT) {
                //Return 2d output with shape [miniBatchSize,nOut]
                // instead of 3d output with shape [miniBatchSize,nOut,1]
                return out.tensorAlongDimension(0, 1, 0);
            }
            return out;
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**Get the state of the RNN layer, as used in rnnTimeStep().
     * @param layer Number/index of the layer.
     * @return Hidden state, or null if layer is not an RNN layer
     */
    public Map<String, INDArray> rnnGetPreviousState(int layer) {
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        Layer l = layers[layer];
        if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer){
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
        }
        if (!(l instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");
        return ((RecurrentLayer) l).rnnGetPreviousState();
    }

    /**Set the state of the RNN layer.
     * @param layer The number/index of the layer.
     * @param state The state to set the specified layer to
     */
    public void rnnSetPreviousState(int layer, Map<String, INDArray> state) {
        if (layer < 0 || layer >= layers.length)
            throw new IllegalArgumentException("Invalid layer number");
        Layer l = layers[layer];
        if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer){
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
        }
        if (!(l instanceof RecurrentLayer))
            throw new IllegalArgumentException("Layer is not an RNN layer");
        RecurrentLayer r = (RecurrentLayer) l;
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
            } else if(layers[i] instanceof BaseWrapperLayer && ((BaseWrapperLayer)layers[i]).getUnderlying() instanceof RecurrentLayer){
                ((RecurrentLayer) ((BaseWrapperLayer)layers[i]).getUnderlying()).rnnClearPreviousState();
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
        return ffToLayerActivationsDetached(training, FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE, storeLastForTBPTT, layers.length-1, input, mask, null, false);
    }

    /** Get the updater for this MultiLayerNetwork
     * @return Updater for MultiLayerNetwork
     */
    public Updater getUpdater() {
        return getUpdater(true);
    }

    public Updater getUpdater(boolean initializeIfReq) {
        if (solver == null && initializeIfReq) {
            synchronized(this){
                if(solver == null) {    //May have been created while waiting for lock
                    solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                    solver.getOptimizer().setUpdater(UpdaterCreator.getUpdater(this));
                }
            }
        }
        if(solver != null) {
            return solver.getOptimizer().getUpdater();
        }
        return null;
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

            // FIXME: int cast
            //New approach: use feedForwardMaskArray method
            feedForwardMaskArray(featuresMaskArray, MaskState.Active, (int) featuresMaskArray.size(0));


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
    public <T extends Evaluation> T evaluate(DataSetIterator iterator) {
        return (T)evaluate(iterator, null);
    }

    /**
     * Evaluate the network for regression performance
     * @param iterator Data to evaluate on
     * @return
     */
    public <T extends RegressionEvaluation> T evaluateRegression(DataSetIterator iterator) {
        return (T)doEvaluation(iterator, new RegressionEvaluation(iterator.totalOutcomes()))[0];
    }

    /**
     * @deprecated To be removed - use {@link #evaluateROC(DataSetIterator, int)} to enforce selection of appropriate ROC/threshold configuration
     */
    @Deprecated
    public <T extends ROC> T evaluateROC(DataSetIterator iterator){
        return evaluateROC(iterator, 0);
    }

    /**
     * Evaluate the network (must be a binary classifier) on the specified data, using the {@link ROC} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROC} - see that class for details.
     * @return ROC evaluation on the given dataset
     */
    public <T extends ROC> T evaluateROC(DataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = getOutputLayer();
        if(getLayerWiseConfigurations().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), ROC.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.ROC(rocThresholdSteps))[0];
    }

    /**
     * @deprecated To be removed - use {@link #evaluateROCMultiClass(DataSetIterator, int)} to enforce selection of appropriate ROC/threshold configuration
     */
    @Deprecated
    public <T extends ROCMultiClass> T evaluateROCMultiClass(DataSetIterator iterator) {
        return evaluateROCMultiClass(iterator, 0);
    }

    /**
     * Evaluate the network on the specified data, using the {@link ROCMultiClass} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROCMultiClass}
     * @return Multi-class ROC evaluation on the given dataset
     */
    public <T extends ROCMultiClass> T evaluateROCMultiClass(DataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = getOutputLayer();
        if(getLayerWiseConfigurations().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), ROCMultiClass.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.ROCMultiClass(rocThresholdSteps))[0];
    }

    /**
     * Perform evaluation using an arbitrary IEvaluation instance.
     *
     * @param iterator   data to evaluate on
     */
    public <T extends IEvaluation> T[] doEvaluation(DataSetIterator iterator, T... evaluations) {
        try{
            return doEvaluationHelper(iterator, evaluations);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    public <T extends IEvaluation> T[] doEvaluationHelper(DataSetIterator iterator, T... evaluations) {
        if (!iterator.hasNext() && iterator.resetSupported()) {
            iterator.reset();
        }

        DataSetIterator iter = iterator.asyncSupported() ? new AsyncDataSetIterator(iterator, 2, true) : iterator;

        WorkspaceMode cMode = layerWiseConfigurations.getTrainingWorkspaceMode();
        layerWiseConfigurations.setTrainingWorkspaceMode(layerWiseConfigurations.getInferenceWorkspaceMode());

        //First: let's determine if we should do 'split feed forward' for long time series
        //The idea: RNN 20k time steps. Train using TBPTT length 100 -> 200 segments of length 100. If we naively
        // just use .output(INDArray) here, then our memory requirements are 200x larger than if we did the same
        // evaluation in segments...
        //Only do this if TBPTT is enabled - if not, it means we can train without TBPTT and hence should be able
        // to test without splitting also
        boolean useRnnSegments = (layerWiseConfigurations.getBackpropType() == BackpropType.TruncatedBPTT);

        MemoryWorkspace outputWs;
        if(getLayerWiseConfigurations().getInferenceWorkspaceMode() == WorkspaceMode.ENABLED){
            outputWs = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(WS_ALL_LAYERS_ACT_CONFIG, WS_OUTPUT_MEM);
        } else {
            outputWs = new DummyWorkspace();
        }

        while (iter.hasNext()) {
            DataSet next = iter.next();

            if (next.getFeatures() == null || next.getLabels() == null)
                continue;


            INDArray features = next.getFeatures();
            INDArray labels = next.getLabels();
            INDArray fMask = next.getFeaturesMaskArray();
            INDArray lMask = next.getLabelsMaskArray();


            if (!useRnnSegments) {
                //Standard/non-RNN case:
                try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                    INDArray out = outputOfLayerDetached(false, FwdPassType.STANDARD, layers.length - 1, features, fMask, lMask, ws);

                    try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                        for (T evaluation : evaluations)
                            evaluation.eval(labels, out, lMask);
                    }
                }
            } else {
                rnnClearPreviousState();


                //Get subset of features and labels:
                val fwdLen = layerWiseConfigurations.getTbpttFwdLength();
                val tsLength = features.size(2);
                long nSubsets = tsLength / fwdLen;
                if (tsLength % fwdLen != 0)
                    nSubsets++; //Example: 100 fwdLen with timeSeriesLength=120 -> want 2 subsets (1 of size 100, 1 of size 20)
                for (int i = 0; i < nSubsets; i++) {
                    val startTimeIdx = i * fwdLen;
                    val endTimeIdx = Math.min(startTimeIdx + fwdLen, tsLength);

                    // FIXME: int cast
                    INDArray[] subsets = getSubsetsForTbptt(startTimeIdx, (int) endTimeIdx, features, labels, fMask, lMask);

                    setLayerMaskArrays(subsets[2], subsets[3]);

                    try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                        INDArray outSub = rnnTimeStep(subsets[0], ws);
                        try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                            for (T evaluation : evaluations)
                                evaluation.eval(subsets[1], outSub, subsets[3]);
                        }
                    }
                }
            }

            //Clear inputs, masks etc. Important to avoid leaking invalidated/out of scope arrays between iterations
            clearLayersStates();
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
            INDArray features = dataSet.getFeatures(0);
            INDArray labels = dataSet.getLabels(0);
            INDArray fMask = null;
            INDArray lMask = null;

            if (dataSet.getFeaturesMaskArrays() != null)
                fMask = dataSet.getFeaturesMaskArrays()[0];

            if (dataSet.getFeaturesMaskArrays() != null)
                lMask = dataSet.getLabelsMaskArrays()[0];

            DataSet ds = new DataSet(features, labels, fMask, lMask);
            fit(ds);
        } else {
            throw new DL4JInvalidInputException(
                    "MultiLayerNetwork can't handle MultiDataSet with more than 1 features or labels array." +
                            "Please consider use of ComputationGraph");
        }
    }

    /**
     * Perform minibatch training on all minibatches in the MultiDataSetIterator, for the specified number of epochs.
     * Equvalent to calling {@link #fit(MultiDataSetIterator)} numEpochs times in a loop
     *
     * @param iterator  Training data (DataSetIterator). Iterator must support resetting
     * @param numEpochs Number of training epochs, >= 1
     */
    public void fit(@NonNull MultiDataSetIterator iterator, int numEpochs){
        Preconditions.checkArgument(numEpochs > 0, "Number of epochs much be > 0. Got numEpochs = %s", numEpochs);
        Preconditions.checkArgument(numEpochs == 1 || iterator.resetSupported(), "Cannot perform multiple epochs training using" +
                "iterator has does not support resetting (iterator.resetSupported() returned false)");

        for(int i = 0; i < numEpochs; i++) {
            fit(iterator);
        }
    }

    /**
     * Perform minibatch training on all minibatches in the MultiDataSetIterator.<br>
     * Note: The MultiDataSets in the MultiDataSetIterator must have exactly 1 input and output array (as
     * MultiLayerNetwork only supports 1 input and 1 output)
     *
     * @param iterator  Training data (DataSetIterator). Iterator must support resetting
     */
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
        if (labelsList == null) {
            try {
                labelsList = iterator.getLabels();
            } catch (Throwable t){ }    //Ignore, maybe UnsupportedOperationException etc
        }

        Layer outputLayer = getOutputLayer();
        if(getLayerWiseConfigurations().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), Evaluation.class);
        }

        Evaluation e = new org.deeplearning4j.eval.Evaluation(labelsList, topN);
        doEvaluation(iterator, e);

        return e;
    }

    protected void update(Task task) {
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
     * @see #memoryInfo(int, InputType)
     */
    public String summary() {
        return summary(null);
    }

    /**
     * String detailing the architecture of the multilayernetwork.
     * Will also display activation size when given an input type.
     * Columns are LayerIndex with layer type, nIn, nOut, Total number of parameters, Shapes of the parameters, Input activation shape, Output activation shape
     * Will also give information about frozen layers, if any.
     * @return Summary as a string
     * @see #memoryInfo(int, InputType)
     */
    public String summary(InputType inputType) {
        StringBuilder ret = new StringBuilder();
        ret.append("\n");

        List<String[]> lines = new ArrayList<>();
        if(inputType == null){
            lines.add(new String[]{"LayerName (LayerType)", "nIn,nOut", "TotalParams", "ParamsShape"});
        } else {
            lines.add(new String[]{"LayerName (LayerType)", "nIn,nOut", "TotalParams", "ParamsShape", "InputShape", "OutputShape"});
        }
        int[] maxLength = new int[inputType == null ? 4 : 6];
        String[] header = lines.get(0);
        for( int i=0; i<header.length; i++ ){
            maxLength[i] = header[i].length();
        }

        int frozenParams = 0;
        for (org.deeplearning4j.nn.api.Layer currentLayer : getLayers()) {
            String name = currentLayer.conf().getLayer().getLayerName();
            if (name == null) {
                name = String.valueOf(currentLayer.getIndex());
            }
            String paramShape = "-";
            String in = "-";
            String out = "-";
            String[] classNameArr = currentLayer.getClass().getName().split("\\.");
            String className = classNameArr[classNameArr.length - 1];
            String paramCount = String.valueOf(currentLayer.numParams());
            String inShape = "";
            String outShape = "";
            InputPreProcessor preProcessor;
            InputType outType;
            if (inputType != null) {
                preProcessor = getLayerWiseConfigurations().getInputPreProcess(currentLayer.getIndex());
                inShape = inputType.toString();
                if (preProcessor != null) {
                    inputType = preProcessor.getOutputType(inputType);
                    inShape += "--> "+ inputType.toString();
                }
                outType = currentLayer.conf().getLayer().getOutputType(currentLayer.getIndex(), inputType);
                outShape = outType.toString();
                inputType = outType;
            }
            if (currentLayer.numParams() > 0) {
                paramShape = "";
                if (currentLayer instanceof BidirectionalLayer) { // Bidirectional layer is not an FFL
                    BidirectionalLayer bi = (BidirectionalLayer) currentLayer;
                    in = String.valueOf(((Bidirectional)bi.conf().getLayer()).getNIn());
                    out = String.valueOf(((Bidirectional)bi.conf().getLayer()).getNOut());
                } else {
                    try {
                        in = String.valueOf(((FeedForwardLayer) currentLayer.conf().getLayer()).getNIn());
                        out = String.valueOf(((FeedForwardLayer) currentLayer.conf().getLayer()).getNOut());
                    }
                    catch (Exception e) { // Some layers, like PReLU, are just BaseLayers (but have parameters)
                    }
                }
                Set<String> paraNames = currentLayer.paramTable().keySet();
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

            String[] line;
            if (inputType == null) {
                line = new String[]{name + " (" + className + ")", in + "," + out, paramCount, paramShape};
            } else {
                line = new String[]{name + " (" + className + ")", in + "," + out, paramCount,paramShape,inShape,outShape};
            }
            for( int i=0; i<line.length; i++ ){
                maxLength[i] = Math.max(maxLength[i], line[i] == null ? 0 : line[i].length());
            }
            lines.add(line);
        }

        StringBuilder sbFormat = new StringBuilder();
        int totalLength = 0;
        int pos = 0;
        for(int length : maxLength){
            int currLength;
            if(pos++ == maxLength.length-1){
                currLength = length;
            } else {
                currLength = length+3;
            }
            sbFormat.append("%-").append(currLength).append("s");
            totalLength += currLength;
        }
        sbFormat.append("\n");
        String format = sbFormat.toString();



        ret.append(StringUtils.repeat("=", totalLength))
                .append("\n");

        boolean first = true;
        for(String[] line : lines){
            String formatted = String.format(format, (Object[])line);
            ret.append(formatted);
            if(first){
                ret.append(StringUtils.repeat("=", totalLength)).append("\n");
                first = false;
            }
        }

        ret.append(StringUtils.repeat("-", totalLength));
        ret.append(String.format("\n%30s %d", "Total Parameters: ", params().length()));
        ret.append(String.format("\n%30s %d", "Trainable Parameters: ", params().length() - frozenParams));
        ret.append(String.format("\n%30s %d", "Frozen Parameters: ", frozenParams));
        ret.append("\n");
        ret.append(StringUtils.repeat("=", totalLength));
        ret.append("\n");
        return ret.toString();
    }

    /**
     * Generate information regarding memory use for the network, for the given input type and minibatch size.
     * Note that when using workspaces or CuDNN, the network should be trained for some iterations so that the memory
     * workspaces have time to initialize. Without this, the memory requirements during training may be underestimated.
     *
     * Note also that this is the same information that is generated during an OOM crash when training or performing
     * inference.
     *
     * @param minibatch    Minibatch size to estimate memory for
     * @param inputType    Input type to the network
     * @return A String with information about network memory use information
     */
    public String memoryInfo(int minibatch, InputType inputType){
        return CrashReportingUtil.generateMemoryStatus(this, minibatch, inputType);
    }

    /**
     * This method just makes sure there's no state preserved within layers
     */
    public void clearLayersStates() {
        for (Layer layer : layers) {
            layer.clear();
            layer.clearNoiseWeightParams();
        }
    }

    /**
     * Increment the epoch count (in the underlying {@link MultiLayerConfiguration} by 1).
     * Note that this is done <i>automatically</i> when using iterator-based fitting methods, such as
     * {@link #fit(DataSetIterator)}. However, when using non-iterator fit methods (DataSet, INDArray/INDArray etc),
     * the network has no way to know when one epoch ends and another starts. In such situations, this method
     * can be used to increment the epoch counter.<br>
     * Note that the epoch counter is used for situations such as some learning rate schedules, and the like.
     *
     * The current epoch count can be obtained using {@code MultiLayerConfiguration.getLayerwiseConfiguration().getEpochCount()}
     */
    public void incrementEpochCount(){
        layerWiseConfigurations.setEpochCount(layerWiseConfigurations.getEpochCount() + 1);
        synchronizeIterEpochCounts();
    }


    protected void synchronizeIterEpochCounts() {
        //TODO: this is necessary for some schedules - but the redundant values are a little ugly...
        int currIter = getIterationCount();
        int currEpoch = getEpochCount();
        for(Layer l : layers) {
            l.setIterationCount(currIter);
            l.setEpochCount(currEpoch);
        }
    }

    /**
     * Save the MultiLayerNetwork to a file. Restore using {@link #load(File, boolean)}.
     * Note that this saves the updater (i.e., the state array for momentum/Adam/rmsprop etc), which is desirable
     * if further training will be undertaken.
     *
     * @param f File to save the network to
     * @see ModelSerializer ModelSerializer for more details (and saving/loading via streams)
     * @see #save(File, boolean)
     */
    public void save( File f ) throws IOException {
        save(f, true);
    }

    /**
     * Save the MultiLayerNetwork to a file. Restore using {@link #load(File, boolean)}.
     *
     * @param f File to save the network to
     * @param saveUpdater If true: save the updater (i.e., the state array for momentum/Adam/rmsprop etc), which should
     *                    usually be saved if further training is required
     * @see ModelSerializer ModelSerializer for more details (and saving/loading via streams)
     * @see #save(File, boolean)
     */
    public void save(File f, boolean saveUpdater) throws IOException{
        ModelSerializer.writeModel(this, f, saveUpdater);
    }

    /**
     * Restore a MultiLayerNetwork to a file, saved using {@link #save(File)} or {@link ModelSerializer}
     * @param f File to load the network from
     * @param loadUpdater If true: load the updater if it is available (i.e., the state array for momentum/Adam/rmsprop
     *                   etc) - use <i>false</i> if no further training is required, or <i>true</i> if further training
     *                    will be undertaken
     * @see ModelSerializer ModelSerializer for more details (and saving/loading via streams)
     */
    public static MultiLayerNetwork load(File f, boolean loadUpdater) throws IOException {
        return ModelSerializer.restoreMultiLayerNetwork(f, loadUpdater);
    }

    /**
     * Convert this MultiLayerNetwork to a ComputationGraph
     *
     * @return ComputationGraph equivalent to this network (including parameters and updater state)
     */
    public ComputationGraph toComputationGraph(){
        return NetworkUtils.toComputationGraph(this);
    }

    /**
     * Set the learning rate for all layers in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration.
     *
     * @param newLr New learning rate for all layers
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(int, double)
     */
    public void setLearningRate(double newLr){
        NetworkUtils.setLearningRate(this, newLr);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLr New learning rate schedule for all layers
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(int, double)
     */
    public void setLearningRate(ISchedule newLr){
        NetworkUtils.setLearningRate(this, newLr);
    }

    /**
     * Set the learning rate for a single layer in the network to the specified value. Note that if any learning rate
     * schedules are currently present, these will be removed in favor of the new (fixed) learning rate.<br>
     * <br>
     * <b>Note</b>: <i>This method not free from a performance point of view</i>: a proper learning rate schedule
     * should be used in preference to calling this method at every iteration. Note also that
     * {@link #setLearningRate(double)} should also be used in preference, when all layers need to be set to a new LR
     *
     * @param layerNumber Number of the layer to set the LR for
     * @param newLr New learning rate for a single layer
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(int, double)
     */
    public void setLearningRate(int layerNumber, double newLr){
        NetworkUtils.setLearningRate(this, layerNumber, newLr);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(ISchedule)} should also be used in preference, when all layers need
     * to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link MultiLayerConfiguration#setIterationCount(int)}
     * and {@link MultiLayerConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerNumber Number of the layer to set the LR schedule for
     * @param newLr New learning rate for a single layer
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(int, double)
     */
    public void setLearningRate(int layerNumber, ISchedule newLr){
        NetworkUtils.setLearningRate(this, layerNumber, newLr);
    }

    /**
     * Get the current learning rate, for the specified layer, from the network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     * @param layerNumber   Layer number to get the learning rate for
     * @return Learning rate for the specified layer, or null
     */
    public Double getLearningRate(int layerNumber){
        return NetworkUtils.getLearningRate(this, layerNumber);
    }

    /**
     * Return the layer size (number of units) for the specified layer.<br>
     * Note that the meaning of the "layer size" can depend on the type of layer. For example:<br>
     * - DenseLayer, OutputLayer, recurrent layers: number of units (nOut configuration option)<br>
     * - ConvolutionLayer: the channels (number of channels)<br>
     * - Subsampling layers, global pooling layers, etc: size of 0 is always returned<br>
     *
     * @param layer Index of the layer to get the size of. Must be in range 0 to nLayers-1 inclusive
     * @return Size of the layer
     */
    public int layerSize(int layer) {
        if (layer < 0 || layer > layers.length) {
            throw new IllegalArgumentException("Invalid layer index: " + layer + ". Layer index must be between 0 and "
                    + (layers.length - 1) + " inclusive");
        }
        org.deeplearning4j.nn.conf.layers.Layer conf = layers[layer].conf().getLayer();
        if (conf == null || !(conf instanceof FeedForwardLayer)) {
            return 0;
        }
        FeedForwardLayer ffl = (FeedForwardLayer) conf;

        // FIXME: int cast
        return (int) ffl.getNOut();
    }

    /**
     * Return the input size (number of inputs) for the specified layer.<br>
     * Note that the meaning of the "input size" can depend on the type of layer. For example:<br>
     * - DenseLayer, OutputLayer, etc: the feature vector size (nIn configuration option)<br>
     * - Recurrent layers: the feature vector size <i>per time step</i> (nIn configuration option)<br>
     * - ConvolutionLayer: the channels (number of channels)<br>
     * - Subsampling layers, global pooling layers, etc: size of 0 is always returned<br>
     *
     * @param layer Index of the layer to get the size of. Must be in range 0 to nLayers-1 inclusive
     * @return Size of the layer
     */
    public int layerInputSize(int layer) {
        if (layer < 0 || layer > layers.length) {
            throw new IllegalArgumentException("Invalid layer index: " + layer + ". Layer index must be between 0 and "
                    + (layers.length - 1) + " inclusive");
        }
        org.deeplearning4j.nn.conf.layers.Layer conf = layers[layer].conf().getLayer();
        if (conf == null || !(conf instanceof FeedForwardLayer)) {
            return 0;
        }
        FeedForwardLayer ffl = (FeedForwardLayer) conf;

        // FIXME: int cast
        return (int) ffl.getNIn();
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

    private void writeObject(ObjectOutputStream oos) throws IOException {
        ModelSerializer.writeModel(this, oos, true);
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        val mln = ModelSerializer.restoreMultiLayerNetwork(ois, true);

        this.defaultConfiguration = mln.defaultConfiguration.clone();
        this.layerWiseConfigurations = mln.layerWiseConfigurations.clone();
        this.init();
        this.flattenedParams.assign(mln.flattenedParams);

        int numWorkingMem = 2 * (layerWiseConfigurations.getConfs().size() + layerWiseConfigurations.getInputPreProcessors().size());
        WS_LAYER_WORKING_MEM_CONFIG = getLayerWorkingMemWSConfig(numWorkingMem);
        WS_LAYER_ACT_X_CONFIG = getLayerActivationWSConfig(layerWiseConfigurations.getConfs().size());

        if (mln.getUpdater() != null && mln.getUpdater(false).getStateViewArray() != null)
            this.getUpdater(true).getStateViewArray().assign(mln.getUpdater(false).getStateViewArray());
    }

}
