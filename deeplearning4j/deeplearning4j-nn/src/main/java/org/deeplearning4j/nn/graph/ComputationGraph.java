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

package org.deeplearning4j.nn.graph;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.lang3.StringUtils;
import org.bytedeco.javacpp.Pointer;
import org.deeplearning4j.exception.DL4JInvalidConfigException;
import org.deeplearning4j.util.*;
import org.nd4j.adapters.OutputAdapter;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.dataset.AsyncMultiDataSetIterator;
import org.deeplearning4j.exception.DL4JException;
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
import org.deeplearning4j.nn.graph.util.ComputationGraphUtil;
import org.deeplearning4j.nn.graph.util.GraphIndices;
import org.deeplearning4j.nn.graph.vertex.GraphVertex;
import org.deeplearning4j.nn.graph.vertex.VertexIndices;
import org.deeplearning4j.nn.graph.vertex.impl.FrozenVertex;
import org.deeplearning4j.nn.graph.vertex.impl.InputVertex;
import org.deeplearning4j.nn.graph.vertex.impl.LayerVertex;
import org.deeplearning4j.nn.layers.FrozenLayer;
import org.deeplearning4j.nn.layers.FrozenLayerWithBackprop;
import org.deeplearning4j.nn.layers.recurrent.BidirectionalLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.updater.graph.ComputationGraphUpdater;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.deeplearning4j.optimize.Solver;
import org.deeplearning4j.optimize.api.ConvexOptimizer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.common.base.Preconditions;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.evaluation.classification.ROCMultiClass;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.memory.enums.ResetPolicy;
import org.nd4j.linalg.api.memory.enums.SpillPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.adapter.MultiDataSetIteratorAdapter;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.DataSetUtil;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.exception.ND4JArraySizeException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.api.memory.abstracts.DummyWorkspace;
import org.nd4j.common.primitives.Pair;
import org.nd4j.common.primitives.Triple;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.workspace.ND4JWorkspaceException;
import org.nd4j.linalg.workspace.WorkspaceUtils;
import org.nd4j.common.util.OneTimeLogger;
import org.nd4j.linalg.workspace.WorkspacesCloseable;

import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;

import static org.deeplearning4j.nn.workspace.ArrayType.*;
import static org.deeplearning4j.nn.workspace.ArrayType.FF_CACHE;

@Slf4j
public class ComputationGraph implements Serializable, Model, NeuralNetwork {

    protected ComputationGraphConfiguration configuration;
    protected boolean initCalled = false;
    protected transient Solver solver; //Used to call optimizers during backprop
    protected INDArray flattenedParams; //Params for all layers are a view/subset of this array
    @Getter
    protected transient INDArray flattenedGradients; //Gradients for all layers are a view/subset of this array
    @Getter
    @Setter
    protected Gradient gradient;
    protected double score;
    @Setter
    private boolean initDone = false;
    @Getter
    @Setter
    protected boolean clearTbpttState = true;  //Mainly for unit testing (should be enabled otherwise)
    //Workspaces for CUDNN. Pass to LayerWorkspaceMgr for re-use in cudnn helpers
    @Getter
    protected transient Map<String,Pointer> helperWorkspaces = new HashMap<>();

    private transient final AtomicLong occupiedBy = new AtomicLong(-1);

    /**
     * Workspace for working memory for a single layer: forward pass and backward pass
     * Note that this is opened/closed once per op (activate/backpropGradient call)
     */
    public static final String WS_LAYER_WORKING_MEM = "WS_LAYER_WORKING_MEM";
    /**
     * Workspace for storing all layers' activations - used only to store activations (layer inputs) as part of backprop
     * Not used for inference
     */
    protected static final String WS_ALL_LAYERS_ACT = "WS_ALL_LAYERS_ACT";
    /**
     * Workspace for working memory in RNNs - opened and closed once per RNN time step
     */
    protected static final String WS_RNN_LOOP_WORKING_MEM = "WS_RNN_LOOP_WORKING_MEM";

    /**
     * Workspace for output methods that use OutputAdapter
     */
    protected static final String WS_OUTPUT_MEM = "WS_OUTPUT_MEM";

    protected final WorkspaceConfiguration WS_LAYER_WORKING_MEM_CONFIG;

    protected static final WorkspaceConfiguration WS_ALL_LAYERS_ACT_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0)
            .overallocationLimit(0.05)
            .policyLearning(LearningPolicy.FIRST_LOOP)
            .policyReset(ResetPolicy.BLOCK_LEFT)
            .policySpill(SpillPolicy.REALLOCATE)
            .policyAllocation(AllocationPolicy.OVERALLOCATE)
            .build();

    protected final WorkspaceConfiguration WS_LAYER_ACT_X_CONFIG;

    protected static final WorkspaceConfiguration WS_RNN_LOOP_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
            .initialSize(0).overallocationLimit(0.05).policyReset(ResetPolicy.BLOCK_LEFT)
            .policyAllocation(AllocationPolicy.OVERALLOCATE).policySpill(SpillPolicy.REALLOCATE)
            .policyLearning(LearningPolicy.FIRST_LOOP).build();


    protected transient ThreadLocal<Long> lastEtlTime = new ThreadLocal<>();

    /**
     * All GraphVertex objects in the network.
     */
    protected GraphVertex[] vertices;
    /**
     * Map of vertices by name
     */
    protected Map<String, GraphVertex> verticesMap;
    /**
     * Indexes of graph vertices, in topological order. The topological order defines the order in which forward pass
     * (and hence also backward pass, which is the opposite to this) is conducted in the network.
     */
    protected int[] topologicalOrder;
    /**
     * Topological sort and vertex index/name + name/index mapping
     */
    protected GraphIndices graphIndices;

    /**
     * A list of layers. Each of these layers is present in a GraphVertex, but are here for easy reference.
     * This array also defines the order in which the getLayer(int) method returns layers.
     */
    protected Layer[] layers;

    /**
     * The number of input arrays to the network. Many networks only have 1 input; however, a ComputationGraph may
     * have an arbitrary number (>=1) separate input arrays
     */
    private int numInputArrays;
    /**
     * The number of output arrays to the network. Many networks only have 1 output; however, a ComputationGraph may
     * have an arbitrary number (>=1) separate output arrays
     */
    private int numOutputArrays;

    //Current inputs, labels, input mask arrays and label mask arrays
    private transient INDArray[] inputs;
    private transient INDArray[] labels;
    private transient INDArray[] inputMaskArrays;
    private transient INDArray[] labelMaskArrays;

    private transient int[] outputLayerIdxs;

    private NeuralNetConfiguration defaultConfiguration;
    private Collection<TrainingListener> trainingListeners = new ArrayList<>();


    public ComputationGraph(ComputationGraphConfiguration configuration) {
        this.configuration = configuration;
        this.numInputArrays = configuration.getNetworkInputs().size();
        this.numOutputArrays = configuration.getNetworkOutputs().size();
        this.inputs = new INDArray[numInputArrays];
        this.labels = new INDArray[numOutputArrays];
        this.defaultConfiguration = configuration.getDefaultConfiguration();

        //Working memory: should learn over course of: (a) full forward pass, and (b) full backward pass
        //Working memory should be opened once per vertex, for each of forward and backward passes
        int numWorkingMem = 2 * configuration.getVertices().size();
        WS_LAYER_WORKING_MEM_CONFIG = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(numWorkingMem)
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();

        //Activations memory: opened once per layer - for every second layer (preprocessors are within the loop).
        //Technically we could set learning to numLayers / 2, but will set to numLayers for simplicity, and also to
        // account for a backward pass
        WS_LAYER_ACT_X_CONFIG = WorkspaceConfiguration.builder()
                .initialSize(0)
                .overallocationLimit(0.02)
                .policyLearning(LearningPolicy.OVER_TIME)
                .cyclesBeforeInitialization(configuration.getVertices().size())
                .policyReset(ResetPolicy.BLOCK_LEFT)
                .policySpill(SpillPolicy.REALLOCATE)
                .policyAllocation(AllocationPolicy.OVERALLOCATE)
                .build();
    }

    /**
     * This method allows to set ETL field time, useful for performance tracking
     *
     * @param time
     */
    public void setLastEtlTime(long time) {
        lastEtlTime.set(time);
    }

    /**
     * This method returns ETL time field value
     *
     * @return
     */
    public long getLastEtlTime() {
        Long time = lastEtlTime.get();
        return time == null ? 0L : time;
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
     * This method returns configuration of this ComputationGraph
     *
     * @return
     */
    public ComputationGraphConfiguration getConfiguration() {
        return configuration;
    }

    /**
     * Returns the number of layers in the ComputationGraph
     */
    public int getNumLayers() {
        return (layers != null ? layers.length : 0);
    }

    /**
     * Get the layer by the number of that layer, in range 0 to getNumLayers()-1
     * NOTE: This is different from the internal GraphVertex index for the layer
     */
    public Layer getLayer(int idx) {
        return layers[idx];
    }

    /**
     * Get all layers in the ComputationGraph
     */
    public Layer[] getLayers() {
        return layers;
    }

    /**
     * Get a given layer by name.
     */
    public Layer getLayer(String name) {
        Preconditions.checkState(verticesMap.containsKey(name), "Layer with name %s does not exist in the network", name);
        return verticesMap.get(name).getLayer();
    }

    /**
     * Returns an array of all GraphVertex objects.
     */
    public GraphVertex[] getVertices() {
        return vertices;
    }

    /**
     * Return a given GraphVertex by name, or null if no vertex with that name exists
     */
    public GraphVertex getVertex(String name) {
        return verticesMap.get(name);
    }

    /**
     * The number of inputs to this network
     */
    public int getNumInputArrays() {
        return numInputArrays;
    }

    /**
     * The number of output (arrays) for this network
     */
    public int getNumOutputArrays() {
        return numOutputArrays;
    }

    /**
     * Set the specified input for the ComputationGraph
     */
    public void setInput(int inputNum, INDArray input) {
        if (inputs == null) {
            //May be null after clear()
            inputs = new INDArray[numInputArrays];
        }
        inputs[inputNum] = input;
    }

    /**
     * Set all inputs for the ComputationGraph network
     */
    public void setInputs(INDArray... inputs) {
        if (inputs != null && inputs.length != this.numInputArrays) {
            throw new IllegalArgumentException("Invalid input array: network has " + numInputArrays
                    + " inputs, but array is of length " + inputs.length);
        }

        this.inputs = inputs;
    }

    /**
     * Get the previously set input for the ComputationGraph
     */
    public INDArray getInput(int inputNum) {
        if (inputs == null)
            return null;
        return inputs[inputNum];
    }

    /**
     * Get the previously set inputs for the ComputationGraph
     */
    public INDArray[] getInputs() {
        return inputs;
    }

    /**
     * Get the previously set feature/input mask arrays for the ComputationGraph
     */
    public INDArray[] getInputMaskArrays() {
        return inputMaskArrays;
    }

    /**
     * Get the previously set label/output mask arrays for the ComputationGraph
     */
    public INDArray[] getLabelMaskArrays() {
        return labelMaskArrays;
    }

    /**
     * Set the specified label for the ComputationGraph
     */
    public void setLabel(int labelNum, INDArray label) {
        labels[labelNum] = label;
    }

    /**
     * Set all labels for the ComputationGraph network
     */
    public void setLabels(INDArray... labels) {
        if (labels != null && labels.length != this.numOutputArrays) {
            throw new IllegalArgumentException("Invalid output array: network has " + numOutputArrays
                    + " outputs, but array is of length " + labels.length);
        }
        this.labels = labels;
    }


    @Override
    public Updater createUpdater() {
        return new ComputationGraphUpdater(this);
    }

    /**
     * Initialize the ComputationGraph network
     */
    public void init() {
        init(null, false);
    }

    /**
     * Initialize the ComputationGraph, optionally with an existing parameters array.
     * If an existing parameters array is specified, it will be used (and the values will not be modified) in the network;
     * if no parameters array is specified, parameters will be initialized randomly according to the network configuration.
     *
     * @param parameters           Network parameter. May be null. If null: randomly initialize.
     * @param cloneParametersArray Whether the parameter array (if any) should be cloned, or used directly
     */
    public void init(INDArray parameters, boolean cloneParametersArray) {
        if (initCalled)
            return;

        DataType netDtype = getConfiguration().getDataType();
        if(parameters != null && parameters.dataType() != netDtype){
            Preconditions.checkState(parameters.rank() == 2 && parameters.size(0) == 1, "Invalid parameters array: should be rank 2 with shape [1,numParams]. Got %ndShape", parameters);
            if(cloneParametersArray) {
                try(MemoryWorkspace ws = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                    parameters = parameters.castTo(netDtype);
                }
            } else {
                throw new IllegalStateException("Error initializing network: Network datatype is set to " + netDtype
                        + " but provided array has datatype " + parameters.dataType() + " with cloneParametersArray argument" +
                        " set to false. Cannot initialize net with specified datatype array if that array does not match network datatype");
            }
        }

        if (configuration.getTrainingWorkspaceMode() == null)
            configuration.setTrainingWorkspaceMode(WorkspaceMode.NONE);

        if (configuration.getInferenceWorkspaceMode() == null)
            configuration.setInferenceWorkspaceMode(WorkspaceMode.NONE);

        if (configuration.getCacheMode() == null)
            configuration.setCacheMode(CacheMode.NONE);

        OneTimeLogger.info(log, "Starting ComputationGraph with WorkspaceModes set to [training: {}; inference: {}], cacheMode set to [{}]",
                configuration.getTrainingWorkspaceMode(), configuration.getInferenceWorkspaceMode(), configuration.getCacheMode());

        //First: build topological ordering, based on configuration. Used for forward pass, backprop and order of parameters/gradients
        GraphIndices indices = calculateIndices();
        topologicalOrder = indices.getTopologicalSortOrder();

        //Initialization: create the GraphVertex objects, based on configuration structure
        Map<String, org.deeplearning4j.nn.conf.graph.GraphVertex> configVertexMap = configuration.getVertices();

        //Names of all of the (data) inputs to the ComputationGraph
        List<String> networkInputNames = configuration.getNetworkInputs();

        //Inputs for each layer and GraphNode:
        Map<String, List<String>> vertexInputs = configuration.getVertexInputs();
        this.vertices = new GraphVertex[networkInputNames.size() + configuration.getVertices().size()];

        //All names: inputs, layers and graph nodes (index to name map)
        Map<String, Integer> allNamesReverse = new HashMap<>();

        //Create network input vertices:
        int vertexNumber = 0;
        for (String name : networkInputNames) {
            GraphVertex gv = new InputVertex(this, name, vertexNumber, null, netDtype); //Output vertices: set later
            allNamesReverse.put(name, vertexNumber);
            vertices[vertexNumber++] = gv;
        }

        //Go through layers, and work out total number of parameters. Then allocate full parameters array
        long numParams = 0;
        long[] numParamsForVertex = new long[topologicalOrder.length];
        int i = 0;
        for (; i < configuration.getNetworkInputs().size(); i++) {
            numParamsForVertex[i] = 0; //No parameters for input vertices
        }
        for(; i < topologicalOrder.length; i++) {
            String name = indices.getIdxToName().get(i);
            org.deeplearning4j.nn.conf.graph.GraphVertex n = configVertexMap.get(name);
            n.setDataType(netDtype);
            numParamsForVertex[i] = n.numParams(true);
            if(numParamsForVertex[i] < 0)
                throw new DL4JInvalidConfigException("Layer " + name + " had parameters < 0 " + numParamsForVertex[i]);
            numParams += numParamsForVertex[i];
        }

        boolean initializeParams;
        if (parameters != null) {
            if (numParams > 0 && !parameters.isRowVectorOrScalar())
                throw new IllegalArgumentException("Invalid parameters: should be a row vector");
            if (parameters.length() != numParams)
                throw new IllegalArgumentException("Invalid parameters: expected length " + numParams + ", got length "
                        + parameters.length());

            if (cloneParametersArray)
                flattenedParams = parameters.dup();
            else
                flattenedParams = parameters;

            initializeParams = false;
        } else if(numParams > 0){
            flattenedParams = Nd4j.create(netDtype, numParams);
            initializeParams = true;
        } else {
            flattenedParams = null;
            initializeParams = false;
        }

        //Set RNG seed, for repeatability between initializations when set
        if (initializeParams) {
            Nd4j.getRandom().setSeed(conf().getSeed());
        }

        if(flattenedParams == null)
            flattenedParams = Nd4j.zeros(DataType.FLOAT,0);

        INDArray flattenedParamsReshape = flattenedParams.reshape(flattenedParams.length());
        //Given the topological ordering: work out the subset of the parameters array used for each layer
        // Then extract out for use when initializing the Layers
        INDArray[] paramsViewForVertex = new INDArray[topologicalOrder.length];
        long paramOffsetSoFar = 0;
        i = 0;
        for (int vertexIdx : topologicalOrder) {
            long nParamsThisVertex = numParamsForVertex[vertexIdx];
            if (nParamsThisVertex != 0) {
                paramsViewForVertex[vertexIdx] = flattenedParamsReshape.get(
                        NDArrayIndex.interval(paramOffsetSoFar, paramOffsetSoFar + nParamsThisVertex));
            }
            i++;
            paramOffsetSoFar += nParamsThisVertex;
        }


        int numLayers = 0;
        List<Layer> tempLayerList = new ArrayList<>();
        defaultConfiguration.clearVariables();
        List<String> variables = defaultConfiguration.variables(false);
        i = configuration.getNetworkInputs().size();
        for(; i < topologicalOrder.length; i++) {
            String name = indices.getIdxToName().get(i);
            org.deeplearning4j.nn.conf.graph.GraphVertex n = configVertexMap.get(name);

            GraphVertex gv = n.instantiate(this, name, vertexNumber, paramsViewForVertex[vertexNumber],
                    initializeParams, netDtype);

            if(gv == null){
                throw new IllegalStateException("Encountered null layer/vertex during initialization for layer \"" + name +
                        "\": " + n.getClass().getSimpleName() + " initialization returned null layer/vertex?");
            }

            if (gv.hasLayer()) {
                numLayers++;
                Layer l = gv.getLayer();
                tempLayerList.add(l);
                List<String> layerVariables = l.conf().variables();
                if (layerVariables != null) {
                    for (String s : layerVariables) {
                        variables.add(gv.getVertexName() + "_" + s);
                    }
                }
            }

            allNamesReverse.put(name, vertexNumber);
            vertices[vertexNumber++] = gv;
        }
        layers = tempLayerList.toArray(new Layer[numLayers]);

        //Create the lookup table, so we can find vertices easily by name
        verticesMap = new HashMap<>();
        for (GraphVertex gv : vertices) {
            verticesMap.put(gv.getVertexName(), gv);
        }

        //Now: do another pass to set the input and output indices, for each vertex
        // These indices are used during forward and backward passes
        //To get output indices: need to essentially build the graph in reverse...
        Map<String, List<String>> verticesOutputTo = new HashMap<>(); //Key: vertex. Values: vertices that this node is an input for
        for (GraphVertex gv : vertices) {
            String vertexName = gv.getVertexName();
            List<String> vertexInputNames;
            vertexInputNames = vertexInputs.get(vertexName);

            if (vertexInputNames == null)
                continue;

            //Build reverse network structure:
            for (String s : vertexInputNames) {
                List<String> list = verticesOutputTo.get(s);
                if (list == null) {
                    list = new ArrayList<>();
                    verticesOutputTo.put(s, list);
                }
                list.add(vertexName); //Edge: s -> vertexName
            }
        }


        for (GraphVertex gv : vertices) {
            String vertexName = gv.getVertexName();
            int vertexIndex = gv.getVertexIndex();
            List<String> vertexInputNames;
            vertexInputNames = vertexInputs.get(vertexName);

            if (vertexInputNames == null)
                continue;

            VertexIndices[] inputIndices = new VertexIndices[vertexInputNames.size()];
            for (int j = 0; j < vertexInputNames.size(); j++) {
                String inName = vertexInputNames.get(j);
                int inputVertexIndex = allNamesReverse.get(inName);

                //Here: we have x -> gv connection
                //gv might have multiple inputs, not just x
                //Need to know which input x is
                int inputNumber = vertexInputs.get(vertexName).indexOf(inName);

                if (inputNumber == -1)
                    throw new IllegalStateException("Could not find vertex " + vertexIndex + " in the list of inputs "
                            + "for vertex " + gv.getVertexName() + "; error in graph structure?");

                inputIndices[j] = new VertexIndices(inputVertexIndex, inputNumber);
            }

            gv.setInputVertices(inputIndices);
        }

        //Handle the outputs for this vertex
        for (GraphVertex gv : vertices) {
            String vertexName = gv.getVertexName();

            List<String> thisVertexOutputsTo = verticesOutputTo.get(vertexName);

            if (thisVertexOutputsTo == null || thisVertexOutputsTo.isEmpty())
                continue; //Output vertex
            VertexIndices[] outputIndices = new VertexIndices[thisVertexOutputsTo.size()];
            int j = 0;
            for (String s : new HashSet<>(thisVertexOutputsTo)) {
                //First, we have gv -> s
                //Which input in s does gv connect to? s may in general have multiple inputs...
                List<String> nextVertexInputNames = vertexInputs.get(s);

                for (int k = 0; k < nextVertexInputNames.size(); k++) {
                    if(vertexName.equals(nextVertexInputNames.get(k))){
                        int outputVertexIndex = allNamesReverse.get(s);
                        outputIndices[j++] = new VertexIndices(outputVertexIndex, k);
                    }
                }
            }
            gv.setOutputVertices(outputIndices);
        }

        //Mark any output vertices as outputs:
        for (String s : configuration.getNetworkOutputs()) {
            GraphVertex gv = verticesMap.get(s);
            gv.setOutputVertex(true);
        }

        // now we init solver & optimizer
        if (solver == null) {
            try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                solver.initOptimizer();
            }
        }

        //Mark which layers can safely modify their input in-place. This is a performance optimization for
        // dropout and similar operations.
        // Safe when the input is: (a) it's not a graph input, and (b) isn't shared by any other layers/vertices

        Map<String,List<String>> seenAsInputTo = new HashMap<>();
        for(Map.Entry<String,List<String>> entry : configuration.getVertexInputs().entrySet()){
            for(String s : entry.getValue() ){
                if (!seenAsInputTo.containsKey(s)) {
                    seenAsInputTo.put(s, new ArrayList<String>());
                }
                List<String> seen = seenAsInputTo.get(s);
                seen.add(entry.getKey());
            }
        }

        for(Layer l : layers){
            String layerName = l.conf().getLayer().getLayerName();
            List<String> inputs = configuration.getVertexInputs().get(layerName);
            String in = inputs.get(0);  //For now: layers should have exactly 1 input

            if(configuration.getNetworkInputs().contains(in)){
                //TODO When is it safe to NOT allow input modifucation? It's not always safe...
                // For example dropout + iterating over List<MultiDataSet> that is used for multiple epochs...
                continue;
            }

            List<String> seen = seenAsInputTo.get(in);
            if(seen.size() == 1){
                l.allowInputModification(true);
            } else {
                //For the count > 1 case, we can work out if it's the last one in the topological order... at which point,
                // it should be safe to use
                int thisIdx = indices.getNameToIdx().get(layerName);
                int thisTopoPos = ArrayUtils.indexOf(indices.getTopologicalSortOrder(), thisIdx);
                int maxTopoPosition = -1;
                for(String s : seen){
                    int idx = indices.getNameToIdx().get(s);
                    int topoPos = ArrayUtils.indexOf(indices.getTopologicalSortOrder(), idx);
                    maxTopoPosition = Math.max(maxTopoPosition, topoPos);
                }

                if(thisTopoPos == maxTopoPosition){
                    //Last one in the topo sort... all other layers have already consumed this input by the time this layer's
                    // forward pass is done
                    l.allowInputModification(true);
                }   //Otherwise: keep default of false
            }
        }

        synchronizeIterEpochCounts();
        initCalled = true;
    }

    /**
     * This method: initializes the flattened gradients array (used in backprop) and sets the appropriate subset in all layers.
     * As a general rule, this shouldn't ever need to be called manually when doing training via fit(DataSet), fit(DataSetIterator)
     * or fit(MultiDataSet) methods
     */
    public void initGradientsView() {
        try (MemoryWorkspace ws = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            if (!initCalled)
                init();

            GraphIndices indices = calculateIndices();

            //Go through layers, and work out total number of parameters. Then allocate full parameters array
            long numParams = 0;
            long[] numParamsForVertex = new long[topologicalOrder.length];
            int i = 0;
            for (; i < configuration.getNetworkInputs().size(); i++) {
                numParamsForVertex[i] = 0; //No parameters for input vertices
            }
            Map<String, org.deeplearning4j.nn.conf.graph.GraphVertex> configVertexMap = configuration.getVertices();
            for (; i < topologicalOrder.length; i++) {
                String name = indices.getIdxToName().get(i);
                org.deeplearning4j.nn.conf.graph.GraphVertex n = configVertexMap.get(name);
                numParamsForVertex[i] = n.numParams(true);
                numParams += numParamsForVertex[i];
            }

            if(numParams > 0) {
                flattenedGradients = Nd4j.create(flattenedParams.dataType(), 1, numParams);
            }

            if(flattenedGradients == null)
                flattenedGradients = Nd4j.zeros(DataType.FLOAT,0);

            INDArray flattenedGradientsReshape = flattenedGradients.reshape(flattenedGradients.length());
            //Given the topological ordering: work out the subset of the gradient array used for each layer, and set it
            long paramOffsetSoFar = 0;
            i = 0;
            for (int vertexIdx : topologicalOrder) {
                long nParamsThisVertex = numParamsForVertex[vertexIdx];
                if (nParamsThisVertex != 0) {
                    INDArray gradientView = flattenedGradientsReshape.get(
                            NDArrayIndex.interval(paramOffsetSoFar, paramOffsetSoFar + nParamsThisVertex));
                    vertices[vertexIdx].setBackpropGradientsViewArray(gradientView);
                }
                i++;
                paramOffsetSoFar += nParamsThisVertex;
            }
        }
    }

    protected int[] getOutputLayerIndices(){
        if(outputLayerIdxs == null) {
            outputLayerIdxs = new int[numOutputArrays];
            int i = 0;
            for (String s : configuration.getNetworkOutputs()) {
                outputLayerIdxs[i++] = verticesMap.get(s).getVertexIndex();
            }
        }
        return  outputLayerIdxs;
    }

    /**
     * Perform layerwise pretraining for one epoch - see {@link #pretrain(DataSetIterator, int)}
     */
    public void pretrain(DataSetIterator iter) {
        pretrain(iter, 1);
    }

    /**
     * Pretrain network with a single input and single output. DataSetIterators can only be used if the number of input
     * arrays for the ComputationGraph is 1.<br>
     * This method performs layerwise pretraining on all pre-trainable layers in the network (VAEs, Autoencoders, etc), for the specified
     * number of epochs each. For example, if numEpochs=3, then layer 0 will be fit for 3 epochs, followed by layer 1
     * for 3 epochs, and so on.<br>
     * For networks with more than one input use {@link #pretrain(MultiDataSetIterator)}
     */
    public void pretrain(DataSetIterator iter, int numEpochs) {
        if (numInputArrays != 1) {
            throw new UnsupportedOperationException(
                    "Cannot train ComputationGraph network with  multiple inputs using a DataSetIterator");
        }

        pretrain(ComputationGraphUtil.toMultiDataSetIterator(iter), numEpochs);
    }

    /**
     * Pretrain network with multiple inputs and/or outputs
     */
    public void pretrain(MultiDataSetIterator iter) {
        pretrain(iter, 1);
    }

    /**
     * Pretrain network with multiple inputs and/or outputs<br>
     * This method performs layerwise pretraining on all pre-trainable layers in the network (VAEs, Autoencoders, etc), for the specified
     * number of epochs each. For example, if numEpochs=3, then layer 0 will be fit for 3 epochs, followed by layer 1
     * for 3 epochs, and so on.<br>
     * Non-pretrainable layers are ignored
     *
     * @param iter      Training data
     * @param numEpochs Number of epochs to fit each layer with
     * @see #pretrainLayer(String, MultiDataSetIterator)
     */
    public void pretrain(MultiDataSetIterator iter, int numEpochs) {
        try{
            pretrainHelper(iter, numEpochs);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private void pretrainHelper(MultiDataSetIterator iter, int numEpochs){
        if (flattenedGradients == null) {
            initGradientsView();
        }

        //Assume here that all layers are pretrainable layers
        for (int i = 0; i < topologicalOrder.length; i++) {
            if (!vertices[i].hasLayer())
                continue;
            if (vertices[i].getLayer() instanceof IOutputLayer)
                continue; //Don't pretrain output layer
            if (!vertices[i].getLayer().isPretrainLayer())
                continue; //Skip layers that aren't pretrainable

            pretrainLayerHelper(vertices[i].getVertexName(), iter, numEpochs);
        }
    }

    /**
     * Pretrain a specified layer with the given DataSetIterator
     *
     * @param layerName       Layer name
     * @param dataSetIterator Data
     */
    public void pretrainLayer(String layerName, DataSetIterator dataSetIterator) {
        if (numInputArrays != 1) {
            throw new UnsupportedOperationException(
                    "Cannot train ComputationGraph network with  multiple inputs using a DataSetIterator");
        }

        pretrainLayer(layerName, ComputationGraphUtil.toMultiDataSetIterator(dataSetIterator));
    }

    /**
     * Pretrain a specified layer with the given MultiDataSetIterator
     *
     * @param layerName Layer name
     * @param iter      Training data
     */
    public void pretrainLayer(String layerName, MultiDataSetIterator iter) {
        try{
            pretrainLayerHelper(layerName, iter, 1);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private void pretrainLayerHelper(String layerName, MultiDataSetIterator iter, int numEpochs){
        if (flattenedGradients == null) {
            initGradientsView();
        }

        if (!verticesMap.containsKey(layerName)) {
            throw new IllegalStateException("Invalid vertex name: " + layerName + " - all vertex names: " +
                    verticesMap.keySet());
        }
        if (!verticesMap.get(layerName).hasLayer()) {
            //No op
            return;
        }

        GraphVertex toTrain = verticesMap.get(layerName);
        int idx = toTrain.getVertexIndex();

        LayerWorkspaceMgr workspaceMgr;
        if(configuration.getTrainingWorkspaceMode() == WorkspaceMode.NONE){
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    //Use FF/BP working memory for updater also
                    .with(ArrayType.UPDATER_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .build();
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        if(!iter.hasNext() && iter.resetSupported())
            iter.reset();

        MultiDataSetIterator withAsync = iter.asyncSupported() ? new AsyncMultiDataSetIterator(iter) : iter;

        while(withAsync.hasNext()) {
            MultiDataSet mds = withAsync.next();
            try(MemoryWorkspace ws = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {
                //FF - note should be using TEST mode here for the layers that feed into the specified layer
                Map<String, INDArray> activations = ffToLayerActivationsInWS(false, idx, new int[]{idx}, FwdPassType.STANDARD,
                        false, mds.getFeatures(), mds.getFeaturesMaskArrays(), mds.getLabelsMaskArrays(), true);

                //Get input to the current layer
                VertexIndices[] inputsToLayer = toTrain.getInputVertices();
                for (VertexIndices vi : inputsToLayer) {
                    String inName = vertices[vi.getVertexIndex()].getVertexName();
                    INDArray act = activations.get(inName);
                    toTrain.setInput(vi.getVertexEdgeNumber(), act, workspaceMgr);
                }

                Layer layer = toTrain.getLayer();
                layer.fit(layer.input(), workspaceMgr);
            }
        }
    }

    /**
     * Fit the ComputationGraph using a DataSet.
     * Note that this method can only be used with ComputationGraphs with 1 input and 1 output.
     * For networks with more than one input or output, use {@link #fit(MultiDataSetIterator)}
     */
    public void fit(DataSet dataSet) {
        if (numInputArrays != 1 || numOutputArrays != 1)
            throw new UnsupportedOperationException("Cannot train ComputationGraph network with "
                    + " multiple inputs or outputs using a DataSet");

        boolean hasMaskArrays = dataSet.hasMaskArrays();
        if (hasMaskArrays) {
            INDArray[] fMask = (dataSet.getFeaturesMaskArray() != null ? new INDArray[]{dataSet.getFeaturesMaskArray()}
                    : null);
            INDArray[] lMask = (dataSet.getLabelsMaskArray() != null ? new INDArray[]{dataSet.getLabelsMaskArray()}
                    : null);
            fit(new INDArray[]{dataSet.getFeatures()}, new INDArray[]{dataSet.getLabels()}, fMask, lMask);
        } else {
            fit(new INDArray[]{dataSet.getFeatures()}, new INDArray[]{dataSet.getLabels()});
        }

        if (hasMaskArrays)
            clearLayerMaskArrays();

        clearLayersStates();
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
     * Fit the ComputationGraph using a DataSetIterator.<br>
     * Note that this method can only be used with ComputationGraphs with 1 input and 1 output<br>
     * Method doesn't do layerwise  pretraining.<br>
     * For pretraining use method pretrain.. {@link #pretrain(DataSetIterator)}<br>
     * @param iterator Training data (DataSetIterator)
     */
    public void fit(@NonNull DataSetIterator iterator) {
        fit(new MultiDataSetIteratorAdapter(iterator));
    }

    /**
     * Fit the ComputationGraph using a MultiDataSet
     */
    public void fit(MultiDataSet multiDataSet) {
        fit(multiDataSet.getFeatures(), multiDataSet.getLabels(), multiDataSet.getFeaturesMaskArrays(),
                multiDataSet.getLabelsMaskArrays());
        if (multiDataSet.hasMaskArrays())
            clearLayerMaskArrays();
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
                "iterator thas does not support resetting (iterator.resetSupported() returned false)");

        for(int i=0; i<numEpochs; i++ ){
            fit(iterator);
        }
    }

    /**
     * Fit the ComputationGraph using a MultiDataSetIterator
     * Method doesn't do layerwise  pretraining.<br>
     * For pretraining use method pretrain.. {@link #pretrain(MultiDataSetIterator)}<br>
     * @param multi Training data (MultiDataSetIterator)
     */
    public  void fit(MultiDataSetIterator multi) {
        if (flattenedGradients == null) {
            initGradientsView();
        }

        if(!multi.hasNext() && multi.resetSupported()){
            multi.reset();
        }

        for (TrainingListener tl : trainingListeners) {
            tl.onEpochStart(this);
        }

        boolean destructable = false;

        MultiDataSetIterator multiDataSetIterator;
        if (multi.asyncSupported()) {
            multiDataSetIterator = new AsyncMultiDataSetIterator(multi, Math.max(Nd4j.getAffinityManager().getNumberOfDevices() * 2, 2), true);
            destructable = true;
        } else
            multiDataSetIterator = multi;

        long time1 = System.currentTimeMillis();
        while(multiDataSetIterator.hasNext()){
            MultiDataSet mds = multiDataSetIterator.next();
            long time2 = System.currentTimeMillis();
            lastEtlTime.set((time2 - time1));

            fit(mds.getFeatures(),mds.getLabels(), mds.getFeaturesMaskArrays(), mds.getLabelsMaskArrays());
            time1 = System.currentTimeMillis();
        }

        if (destructable)
            ((AsyncMultiDataSetIterator) multiDataSetIterator).shutdown();

        for (TrainingListener tl : trainingListeners) {
            tl.onEpochEnd(this);
        }

        incrementEpochCount();
    }
    /**
     * Fit the ComputationGraph given arrays of inputs and labels.
     *
     * @param inputs The network inptus
     * @param labels The labels
     */
    public void fit(INDArray[] inputs, INDArray[] labels) {
        fit(inputs, labels, null, null);
    }

    /**
     * Fit the ComputationGraph using the specified inputs and labels (and mask arrays)
     *
     * @param inputs            The network inputs (features)
     * @param labels            The network labels
     * @param featureMaskArrays Mask arrays for inputs/features. Typically used for RNN training. May be null.
     * @param labelMaskArrays   Mas arrays for the labels/outputs. Typically used for RNN training. May be null.
     */
    public void fit(INDArray[] inputs, INDArray[] labels, INDArray[] featureMaskArrays, INDArray[] labelMaskArrays) {
        try{
            fitHelper(inputs, labels, featureMaskArrays, labelMaskArrays);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private  void fitHelper(INDArray[] inputs, INDArray[] labels, INDArray[] featureMaskArrays, INDArray[] labelMaskArrays) {
        if (numParams() == 0) {
            return; //Edge case: net with no params: fitting is a no-op
        }

        if (flattenedGradients == null) {
            initGradientsView();
        }

        setInputs(inputs);
        setLabels(labels);
        setLayerMaskArrays(featureMaskArrays, labelMaskArrays);

        LayerWorkspaceMgr workspaceMgr;
        if(configuration.getTrainingWorkspaceMode() == WorkspaceMode.NONE) {
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

        if (configuration.getBackpropType() == BackpropType.TruncatedBPTT) {
            doTruncatedBPTT(inputs, labels, featureMaskArrays, labelMaskArrays, workspaceMgr);
        } else {
            if (solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
                }
            }

            solver.optimize(workspaceMgr);



        }

        if (featureMaskArrays != null || labelMaskArrays != null) {
            clearLayerMaskArrays();
        }

        clearLayersStates();
        synchronizeIterEpochCounts();
    }



    /**
     * Calculate a topological sort order for the vertices in the graph.
     * Note that this is used for
     * (a) working out what order to do forward pass,
     * (b) what order to do backprop (i.e., reverse of this)
     * (c) order to flatten parameters (and gradients)
     * <p>
     * Specifically, gradients/params/forward pass are executed on vertex[topologicalSortOrder[i]], for i=0..nVertices-1
     */
    public int[] topologicalSortOrder() {
        return calculateIndices().getTopologicalSortOrder();
    }

    /**
     * Calculate the indices needed for the network:<br>
     * (a) topological sort order<br>
     * (b) Map: vertex index -> vertex name<br>
     * (c) Map: vertex name -> vertex index<br>
     *
     * @return Calculated indices
     */
    public GraphIndices calculateIndices(){
        if(graphIndices != null)
            return graphIndices;


        //Get cached topological sort order from config, if present
        if(configuration.getTopologicalOrder() != null && configuration.getTopologicalOrderStr() != null){
            int[] t = configuration.getTopologicalOrder();
            List<String> s = configuration.getTopologicalOrderStr();
            Map<String,Integer> m1 = new HashMap<>();
            Map<Integer,String> m2 = new HashMap<>();
            for( int i=0; i<t.length; i++ ){
                m1.put(s.get(i), t[i]);
                m2.put(t[i], s.get(i));
            }

            graphIndices = GraphIndices.builder()
                    .topologicalSortOrder(t)
                    .nameToIdx(m1)
                    .idxToName(m2)
                    .build();
            return graphIndices;
        }


        //https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
        Map<String, org.deeplearning4j.nn.conf.graph.GraphVertex> nodeMap = configuration.getVertices();
        List<String> networkInputNames = configuration.getNetworkInputs();
        int numVertices = networkInputNames.size() + configuration.getVertices().size();
        int[] out = new int[numVertices];
        int outCounter = 0;

        //First: represent the graph more usefully as a Map<Integer,Set<Integer>>, where map represents edges i -> j
        // key represents j, set is set of i (inputs) for vertices j
        Map<Integer, String> vertexNamesMap = new HashMap<>();
        Map<String, Integer> vertexNamesMap2 = new HashMap<>();
        int i = 0;
        for (String inputName : configuration.getNetworkInputs()) {
            vertexNamesMap.put(i, inputName);
            vertexNamesMap2.put(inputName, i);
            i++;
        }
        for (Map.Entry<String, org.deeplearning4j.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()) {
            String name = entry.getKey();
            vertexNamesMap.put(i, name);
            vertexNamesMap2.put(name, i);
            i++;
        }

        Map<Integer, Set<Integer>> inputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex receives input from
        Map<Integer, Set<Integer>> outputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex outputs to

        for (String s : configuration.getNetworkInputs()) {
            int idx = vertexNamesMap2.get(s);
            inputEdges.put(idx, null);
        }

        for (Map.Entry<String, org.deeplearning4j.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()) {
            String thisVertexName = entry.getKey();
            int idx = vertexNamesMap2.get(thisVertexName);
            List<String> inputsToThisVertex = configuration.getVertexInputs().get(thisVertexName);

            if (inputsToThisVertex == null || inputsToThisVertex.isEmpty()) {
                inputEdges.put(idx, null);
                continue;
            }

            Set<Integer> inputSet = new HashSet<>();
            for (String s : inputsToThisVertex) {
                Integer inputIdx = vertexNamesMap2.get(s);
                inputSet.add(inputIdx);
                Set<Integer> outputSetForInputIdx = outputEdges.get(inputIdx);
                if (outputSetForInputIdx == null) {
                    outputSetForInputIdx = new HashSet<>();
                    outputEdges.put(inputIdx, outputSetForInputIdx);
                }
                outputSetForInputIdx.add(idx); //input vertex outputs to the current vertex
            }

            inputEdges.put(idx, inputSet);
        }

        //Now: do topological sort
        //Set of all nodes with no incoming edges: (this would be: input vertices)
        LinkedList<Integer> noIncomingEdges = new LinkedList<>();
        for (Map.Entry<Integer, Set<Integer>> entry : inputEdges.entrySet()) {
            Set<Integer> inputsFrom = entry.getValue();
            if (inputsFrom == null || inputsFrom.isEmpty()) {
                noIncomingEdges.add(entry.getKey());
            }
        }

        while (!noIncomingEdges.isEmpty()) {
            int next = noIncomingEdges.removeFirst();
            out[outCounter++] = next; //Add to sorted list

            Set<Integer> vertexOutputsTo = outputEdges.get(next);

            //Remove edges next -> vertexOuputsTo[...] from graph;
            if (vertexOutputsTo != null) {
                for (Integer v : vertexOutputsTo) {
                    Set<Integer> set = inputEdges.get(v);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncomingEdges.add(v); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        //If any edges remain in the graph: graph has cycles:
        for (Map.Entry<Integer, Set<Integer>> entry : inputEdges.entrySet()) {
            Set<Integer> set = entry.getValue();
            if (set == null)
                continue;
            if (!set.isEmpty())
                throw new IllegalStateException(
                        "Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                                + "cycle includes vertex \"" + vertexNamesMap.get(entry.getKey())
                                + "\")");
        }

        //Store: the topological sort order in the configuraation... this is to ensure that when the config is
        // deserialized, it has exactly the same topological sort order on all platforms
        List<String> s = new ArrayList<>(out.length);
        for( int idx : out){
            s.add(vertexNamesMap.get(idx));
        }
        configuration.setTopologicalOrder(out);
        configuration.setTopologicalOrderStr(s);

        graphIndices = GraphIndices.builder()
                .topologicalSortOrder(out)
                .nameToIdx(vertexNamesMap2)
                .idxToName(vertexNamesMap)
                .build();
        return graphIndices;
    }

    @Override
    public void computeGradientAndScore(LayerWorkspaceMgr workspaceMgr) {
        computeGradientAndScore();
    }

    public void computeGradientAndScore() {
        synchronizeIterEpochCounts();

        LayerWorkspaceMgr workspaceMgr;
        if(configuration.getTrainingWorkspaceMode() == WorkspaceMode.NONE) {
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


        boolean tbptt = configuration.getBackpropType() == BackpropType.TruncatedBPTT;
        FwdPassType fwdType = (tbptt ? FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE : FwdPassType.STANDARD);
        synchronizeIterEpochCounts();

        //Calculate activations (which are stored in each layer, and used in backprop)
        try(MemoryWorkspace wsAllActivations = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATIONS)) {

            Map<String, INDArray> activations = ffToLayerActivationsInWS(true, -1, getOutputLayerIndices(),
                    fwdType, tbptt, inputs, inputMaskArrays, labelMaskArrays, false);
            if (!trainingListeners.isEmpty()) {
                try (MemoryWorkspace workspace = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    for (TrainingListener tl : trainingListeners) {
                        tl.onForwardPass(this, activations);
                    }
                }
            }
            calcBackpropGradients(false, false);


            //Score: sum of the scores for the various output layers...
            double r = calcRegularizationScore(true);

            score = 0.0;
            int outNum = 0;
            for (String s : configuration.getNetworkOutputs()) {
                GraphVertex gv = verticesMap.get(s);
                if (gv instanceof LayerVertex) {
                    //At this point: the input to the output layer might not be set on the layer itself - just the vertex
                    LayerVertex lv = (LayerVertex) gv;
                    if (!lv.isSetLayerInput()) {
                        lv.applyPreprocessorAndSetInput(workspaceMgr);
                    }
                }
                Layer vertexLayer = gv.getLayer();
                if (vertexLayer instanceof FrozenLayerWithBackprop) {
                    vertexLayer = ((FrozenLayerWithBackprop) vertexLayer).getInsideLayer();
                }
                vertexLayer.setMaskArray((labelMaskArrays == null) ? null : labelMaskArrays[outNum]);

                score += ((IOutputLayer) vertexLayer).computeScore(r, true, workspaceMgr);


                //Only want to add l1/l2 component once...
                r = 0.0;
                outNum++;
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

        for(GraphVertex gv : vertices) {
            gv.clear();
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(null);

    }


    /**
     * Conduct forward pass using a single input array. Note that this method can only be used with ComputationGraphs
     * with a single input array.
     *
     * @param input The input array
     * @param layerTillIndex the layer to feed forward to
     * @param train If true: do forward pass at training time
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray input, int layerTillIndex,boolean train) {
        if (numInputArrays != 1)
            throw new UnsupportedOperationException("Cannot feedForward with single input for graph network with "
                    + numInputArrays + " expected inputs");
        setInput(0, input);
        return feedForward(train,layerTillIndex);
    }



    /**
     * Conduct forward pass using an array of inputs. This overload allows the forward pass to be conducted, optionally
     * (not) clearing the layer input arrays.<br>
     * Note: when using clearInputs=false, there can be some performance and memory overhead: this is because the arrays are
     * defined outside of workspaces (which are enabled by default) - otherwise, old/invalidated arrays could still be
     * accessed after calling this method. Consequently: Don't use clearInputs=false unless you have a use case that
     * requires them to remain after feed-forward has been completed
     *
     * @param input An array of ComputationGraph inputs
     * @param layerTillIndex the index of the layer to feed forward to
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @param clearInputs If true (default for other methods): clear the inputs of all layers after doing forward
     *                    pass. False don't clear layer inputs.
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray[] input, int layerTillIndex,boolean train, boolean clearInputs) {
        setInputs(input);
        try {
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, layerTillIndex, null,
                    input, inputMaskArrays, labelMaskArrays, clearInputs);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }


    /**
     * Conduct forward pass using an array of inputs
     *
     * @param input An array of ComputationGraph inputs
     * @param layerTillIndex the index of the layer to feed forward to
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray[] input, int layerTillIndex,boolean train) {
        setInputs(input);
        return feedForward(train, layerTillIndex);
    }


    /**
     * Conduct forward pass using the stored inputs
     *
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @param layerTillIndex the index of the layer to feed forward to
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(boolean train,int layerTillIndex) {
        int graphVertexIndexOfLayer = layers[layerTillIndex].getIndex();
        try{
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, graphVertexIndexOfLayer,
                    null, inputs, inputMaskArrays, labelMaskArrays, true);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }



    /**
     * Conduct forward pass using a single input array. Note that this method can only be used with ComputationGraphs
     * with a single input array.
     *
     * @param input The input array
     * @param train If true: do forward pass at training time
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray input, boolean train) {
        if (numInputArrays != 1)
            throw new UnsupportedOperationException("Cannot feedForward with single input for graph network with "
                    + numInputArrays + " expected inputs");
        setInput(0, input);
        return feedForward(train);
    }

    /**
     * Conduct forward pass using an array of inputs
     *
     * @param input An array of ComputationGraph inputs
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray[] input, boolean train) {
        return feedForward(input, train, true);
    }

    /**
     * Conduct forward pass using an array of inputs. This overload allows the forward pass to be conducted, optionally
     * (not) clearing the layer input arrays.<br>
     * Note: this method should NOT be used with clearInputs = true, unless you know what you are doing. Specifically:
     * when using clearInputs=false, in combination with workspaces, the layer input fields may leak outside of the
     * workspaces in which they were defined - potentially causing a crash. See <a href="https://deeplearning4j.konduit.ai/config/config-memory/config-workspaces">
     *     https://deeplearning4j.konduit.ai/config/config-memory/config-workspaces</a>
     * for more details
     *
     * @param input An array of ComputationGraph inputs
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @param clearInputs If true (default for other methods): clear the inputs of all layers after doing forward
     *                    pass. False don't clear layer inputs.
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(INDArray[] input, boolean train, boolean clearInputs) {
        setInputs(input);
        try {
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, vertices.length - 1,
                    null, input, inputMaskArrays, labelMaskArrays, clearInputs);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Conduct forward pass using the stored inputs, at test time
     *
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward() {
        return feedForward(false);
    }

    /**
     * Conduct forward pass using the stored inputs
     *
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @return A map of activations for each layer (not each GraphVertex). Keys = layer name, values = layer activations
     */
    public Map<String, INDArray> feedForward(boolean train) {
        try {
            return ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false, vertices.length - 1,
                    null, inputs, inputMaskArrays, labelMaskArrays, true);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * @param train                            True: training time. False: test time
     * @param excludeOutputLayers              Should we exclude the output layers during forward pass? (usually: false)
     * @param includeNonLayerVertexActivations Include non-layer vertices in the output may?
     * @return Map of activations. Key: vertex name. Value: activations.
     */
    public Map<String, INDArray> feedForward(boolean train, boolean excludeOutputLayers,
                                             boolean includeNonLayerVertexActivations) {
        int[] exclude = null;
        if(excludeOutputLayers){
            exclude = getOutputLayerIndices();
        }

        Map<String,INDArray> m = ffToLayerActivationsDetached(train, FwdPassType.STANDARD, false,
                vertices.length-1, exclude, inputs, inputMaskArrays, labelMaskArrays, true);
        if(includeNonLayerVertexActivations){
            return m;
        } else {
            //Include only layers - in previous versions, we've always included inputs too for this method...
            Map<String,INDArray> out = new HashMap<>();
            for(Map.Entry<String,INDArray> e : m.entrySet()){
                GraphVertex v = verticesMap.get(e.getKey());
                if(v instanceof LayerVertex || v instanceof InputVertex){
                    out.put(e.getKey(), e.getValue());
                }
            }
            return out;
        }
    }

    /**
     * Return an array of network outputs (predictions) at test time, given the specified network inputs
     * Network outputs are for output layers only.
     *
     * @param input Inputs to the network
     * @return Output activations (order: same as defined in network configuration)
     */
    public INDArray[] output(INDArray... input) {
        return output(false, input, inputMaskArrays, labelMaskArrays);
    }

    /**
     * A convenience method that returns a single INDArray, instead of an INDArray[].
     * Useful for ComputationGraphs that have only a single output.
     * Otherwise identical to {@link #output(INDArray...)}
     *
     * @param input Inputs to the network
     * @return Output activations array
     */
    public INDArray outputSingle(INDArray... input) {
        return outputSingle(false, input);
    }

    /**
     * Return an array of network outputs (predictions), given the specified network inputs
     * Network outputs are for output layers only.
     *
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @param input Inputs to the network
     * @return Output activations (order: same as defined in network configuration)
     */
    public INDArray[] output(boolean train, INDArray... input) {
        return output(train, (MemoryWorkspace)null, input);
    }

    /**
     * Return an array of network outputs (predictions), given the specified network inputs
     * Network outputs are for output layers only.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param train           If true: do forward pass at training time; false: do forward pass at test time
     * @param outputWorkspace May be null. If not null: the workspace MUST be opened before calling this method.
     * @param input           Inputs to the network
     * @return Output activations (order: same as defined in network configuration)
     */
    public INDArray[] output(boolean train, MemoryWorkspace outputWorkspace, INDArray... input) {
        return output(train, input, inputMaskArrays, labelMaskArrays, outputWorkspace);
    }

    /**
     * Return an array of network outputs (predictions), given the specified network inputs
     * Network outputs are for output layers only.
     *
     * @param train      If true: forward pass for training mode. False: test mode
     * @param input      Input arrays to the netwonk
     * @param inputMasks Optional input mask arrays (may be null)
     * @return           Network output activations
     */
    public INDArray[] output(boolean train, @NonNull INDArray[] input, INDArray[] inputMasks){
        return output(train, input, inputMasks, null);
    }

    /**
     * Return an array of network outputs (predictions), given the specified network inputs
     * Network outputs are for output layers only.
     *
     * @param train      If true: forward pass for training mode. False: test mode
     * @param input      Input arrays to the netwonk
     * @param inputMasks Optional input mask arrays (may be null)
     * @param labelMasks Optional label mask arrays (may be null
     * @return           Network output activations
     */
    public INDArray[] output(boolean train, @NonNull INDArray[] input, INDArray[] inputMasks, INDArray[] labelMasks) {
        return output(train, input, inputMasks, labelMasks, null);
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
    public  <T> T output(@NonNull INDArray[] inputs, INDArray[] inputMasks, INDArray[] labelMasks, @NonNull OutputAdapter<T> outputAdapter) {
        try (val ws = Nd4j.getWorkspaceManager().getAndActivateWorkspace(WS_ALL_LAYERS_ACT_CONFIG, WS_OUTPUT_MEM)) {
            if (outputAdapter instanceof ModelAdapter)
                return ((ModelAdapter<T>) outputAdapter).apply(this, inputs, inputMasks, labelMasks);
            else
                return outputAdapter.apply(output(false, inputs, inputMasks, labelMasks, ws));
        }
    }

    /**
     * Return an array of network outputs (predictions), given the specified network inputs
     * Network outputs are for output layers only.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param train           If true: forward pass for training mode. False: test mode
     * @param input           Input arrays to the netwonk
     * @param inputMasks      Optional input mask arrays (may be null)
     * @param labelMasks      Optional label mask arrays (may be null
     * @param outputWorkspace May be null. If not null: the workspace MUST be opened before calling this method.
     * @return Network output activations
     */
    public  INDArray[] output(boolean train, @NonNull INDArray[] input, INDArray[] inputMasks, INDArray[] labelMasks, MemoryWorkspace outputWorkspace){
        try {
            setLayerMaskArrays(inputMasks, labelMasks);
            INDArray[] out = outputOfLayersDetached(train, FwdPassType.STANDARD, getOutputLayerIndices(), input, inputMasks, labelMasks, true, false, outputWorkspace);
            clearLayerMaskArrays();
            clearLayersStates();
            return out;
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * A convenience method that returns a single INDArray, instead of an INDArray[].
     * Useful for ComputationGraphs that have only a single output.
     * Otherwise identical to {@link #output(boolean, INDArray...)}
     *
     * @param train If true: do forward pass at training time; false: do forward pass at test time
     * @param input Inputs to the network
     * @return Output activations array
     */
    public INDArray outputSingle(boolean train, INDArray... input) {
        return outputSingle(train, true, input);
    }

    /**
     * Identical to {@link #outputSingle(boolean, boolean, INDArray...)} but has the option of not clearing the input
     * arrays (useful when later backpropagating external errors). Most users should use {@link #outputSingle(boolean, INDArray...)}
     * in preference to this method.
     */
    public INDArray outputSingle(boolean train, boolean clearInputs, INDArray... input) {
        if (numOutputArrays != 1) {
            throw new IllegalStateException(
                    "Cannot use outputSingle with ComputationGraph that does not have exactly 1 output. nOutputs: "
                            + numOutputArrays);
        }
        return output(train, clearInputs, input)[0];
    }

    /**
     * An output method for the network, with optional clearing of the layer inputs.<br>
     * Note: most users should use {@link #output(boolean, INDArray...)} or similar methods, unless they are doing
     * non-standard operations (like providing the input arrays externally)
     *
     * @param train       If true: output during training. False: output during testing. Affects some things such as
     *                    dropout
     * @param clearInputs If true: clear the input arrays for all layers. False: leave the input arrays as-is - which
     *                    can be useful for "external errors" (no output layer) backprop use cases
     * @param input       Input to the network
     * @return            Output from the network
     */
    public  INDArray[] output(boolean train, boolean clearInputs, INDArray... input) {
        boolean detachedInputs = !clearInputs;  //If !clearInputs, then inputs should be detached (otherwise: will be out of scope)
        try {

            return outputOfLayersDetached(train, FwdPassType.STANDARD, getOutputLayerIndices(), input, null, null, clearInputs, detachedInputs, null);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Generate the output for all examples/batches in the input iterator, and concatenate them into a single array
     * per network output
     *
     * @param iterator Data to pass through the network
     * @return output for all examples in the iterator
     */
    public INDArray[] output(DataSetIterator iterator){
        return output(new MultiDataSetIteratorAdapter(iterator));
    }

    /**
     * Generate the output for all examples/batches in the input iterator, and concatenate them into a single array
     * per network output
     *
     * @param iterator Data to pass through the network
     * @return output for all examples in the iterator
     */
    public INDArray[] output(MultiDataSetIterator iterator){
        List<INDArray[]> outputs = new ArrayList<>();
        while(iterator.hasNext()){
            MultiDataSet next = iterator.next();
            INDArray[] out = output(false, next.getFeatures(), next.getFeaturesMaskArrays(), next.getLabelsMaskArrays());
            outputs.add(out);
        }
        INDArray[][] arr = outputs.toArray(new INDArray[outputs.size()][0]);
        return DataSetUtil.mergeFeatures(arr, null).getFirst();
    }

    /**
     * Generate the output for all examples/batches in the input iterator, and concatenate them into a single array.
     * Can only be used with ComputationGraphs with 1 output
     *
     * @param iterator Data to pass through the network
     * @return output for all examples in the iterator
     */
    public INDArray outputSingle(DataSetIterator iterator){
        Preconditions.checkArgument(numOutputArrays == 1, "Cannot use this method with nets that have more" +
                " than 1 output array. This network has %s outputs", numOutputArrays);
        return output(iterator)[0];
    }

    /**
     * Generate the output for all examples/batches in the input iterator, and concatenate them into a single array.
     * Can only be used with ComputationGraphs with 1 output
     *
     * @param iterator Data to pass through the network
     * @return output for all examples in the iterator
     */
    public INDArray outputSingle(MultiDataSetIterator iterator){
        Preconditions.checkArgument(numOutputArrays == 1, "Cannot use this method with nets that have more" +
                " than 1 output array. This network has %s outputs", numOutputArrays);
        return output(iterator)[0];
    }

    /**
     * Get the activations for the specific layers only
     * @param layers       Layers to get the specified activations for
     * @param train        If true: train mode. False: test (inference) mode
     * @param features     Features array
     * @param featureMasks Feature masks array. May be null
     * @return Activations of the selected layers, in the same order as the "layers" arg/list
     */
    public INDArray[] output(List<String> layers, boolean train, INDArray[] features, INDArray[] featureMasks){
        Preconditions.checkState(layers != null && layers.size() > 0, "Layers must not be null: got later names %s", layers);
        int[] layerNums = new int[layers.size()];
        for( int i = 0; i < layers.size(); i++) {
            String n = layers.get(i);
            Preconditions.checkState(verticesMap.containsKey(n), "Layer with name %s not found in network", n);
            layerNums[i] = verticesMap.get(n).getVertexIndex();
        }
        INDArray[] out = outputOfLayersDetached(train, FwdPassType.STANDARD, layerNums, features, featureMasks, null, true,
                false, null);
        return out;
    }


    protected void validateArrayWorkspaces(LayerWorkspaceMgr mgr, INDArray array, ArrayType arrayType, String vertexName, boolean isInputVertex, String op){
        try{
            mgr.validateArrayLocation(arrayType, array, false, isInputVertex);
        } catch (ND4JWorkspaceException e) {
            String clazz;
            GraphVertex v = verticesMap.get(vertexName);
            if(v instanceof LayerVertex) {
                clazz = v.getLayer().getClass().getSimpleName();
            } else {
                clazz = v.getClass().getSimpleName();
            }
            throw new IllegalStateException(op + ": array (" + arrayType + ") workspace validation failed (vertex " +
                    vertexName + " - class: " + clazz + ") - array is defined in incorrect workspace", e);
        }
    }

    /**
     * Feed-forward through the network - returning all array activations detached from any workspace.
     * Note that no workspace should be active externally when calling this method (an exception will be thrown
     * if a workspace is open externally)
     *
     * @param train             Training mode (true) or test/inference mode (false)
     * @param fwdPassType       Type of forward pass to perform (STANDARD or RNN_ACTIVATE_WITH_STORED_STATE only)
     * @param storeLastForTBPTT ONLY used if fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE
     * @param layerIndex        Index (inclusive) to stop forward pass at. For all layers, use numLayers-1
     * @param excludeIdxs       Layers (vertices) to exclude from forward pass. These layers will be skipped, and hence
     *                          are usually output layers or at the end of the network. May be null.
     * @param features          Input feature arrays
     * @param fMask             Feature mask arrays. May be null.
     * @param lMask             Label mask array. May be null.
     * @param clearLayers       Whether the layer inputs should be cleared
     * @return Map of activations (including the input), detached from any workspace
     */
    protected  Map<String,INDArray> ffToLayerActivationsDetached(boolean train, @NonNull FwdPassType fwdPassType, boolean storeLastForTBPTT,
                                                                 int layerIndex, int[] excludeIdxs, @NonNull INDArray[] features,
                                                                 INDArray[] fMask, INDArray[] lMask, boolean clearLayers) {
        if(layerIndex < 0 || layerIndex >= topologicalOrder.length) {
            throw new IllegalArgumentException("Invalid layer index - index must be >= 0 and < " + topologicalOrder.length
                    + ", got index " + layerIndex);
        }

        setInputs(features);
        setLayerMaskArrays(fMask, lMask);

        //Verify that no workspace is open externally
        WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active before call to ffToLayerActivationsDetached", true);

        LayerWorkspaceMgr workspaceMgr;
        WorkspaceMode wsm = (train ? configuration.getTrainingWorkspaceMode() : configuration.getInferenceWorkspaceMode());
        if (wsm == WorkspaceMode.NONE) {
            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            workspaceMgr = LayerWorkspaceMgr.builder()
                    .noWorkspaceFor(ArrayType.ACTIVATIONS)
                    .with(ArrayType.INPUT, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(features[0].isAttached()) {
                //Don't leverage out of async DataMultiSetIterator workspaces
                workspaceMgr.setNoLeverageOverride(features[0].data().getParentWorkspace().getId());
            }
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        Map<String, INDArray> activations = new HashMap<>();

        //Add the inputs:
        for( int i = 0; i < features.length; i++) {
            activations.put(configuration.getNetworkInputs().get(i), features[i]);
        }

        boolean traceLog = log.isTraceEnabled();
        // workspaceMgr.keepOpen(ArrayType.values());

        //Do forward pass according to the topological ordering of the network
        for (int i = 0; i <= layerIndex; i++) {
            GraphVertex current = vertices[topologicalOrder[i]];
            String vName = current.getVertexName();
            int vIdx = current.getVertexIndex();

            if(excludeIdxs != null && ArrayUtils.contains(excludeIdxs, vIdx)) {
                continue;
            }

            if(traceLog) {
                log.trace("About forward pass: {} (\"{}\") - {}", i, vName, current.getClass().getSimpleName());
            }



            VertexIndices[] inputsTo = current.getOutputVertices();

            INDArray out;
            if(current.isInputVertex()) {
                out = inputs[vIdx];
            } else {

                if(fwdPassType == FwdPassType.STANDARD) {
                    //Standard feed-forward case
                    out = current.doForward(train, workspaceMgr);
                } else if(fwdPassType == FwdPassType.RNN_TIMESTEP) {
                    if (current.hasLayer()) {
                        //Layer
                        INDArray input = current.getInputs()[0];
                        Layer l = current.getLayer();
                        if (l instanceof RecurrentLayer) {
                            out = ((RecurrentLayer) l).rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                        }  else if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer && ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying() instanceof RecurrentLayer){
                            RecurrentLayer rl = ((RecurrentLayer) ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying());
                            out = rl.rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                        } else if (l instanceof MultiLayerNetwork) {
                            out = ((MultiLayerNetwork) l).rnnTimeStep(reshapeTimeStepInput(input));
                        } else {
                            //non-recurrent layer
                            out = current.doForward(train, workspaceMgr);
                        }
                    } else {
                        //GraphNode
                        out = current.doForward(train, workspaceMgr);
                    }
                } else if(fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE) {
                    if (current.hasLayer()) {
                        Layer l = current.getLayer();
                        if (l instanceof RecurrentLayer) {
                            out = ((RecurrentLayer) l).rnnActivateUsingStoredState(current.getInputs()[0], train, storeLastForTBPTT, workspaceMgr);
                        } else if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer && ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying() instanceof RecurrentLayer) {
                            RecurrentLayer rl = (RecurrentLayer) ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
                            out = rl.rnnActivateUsingStoredState(current.getInputs()[0], train,storeLastForTBPTT, workspaceMgr);
                        } else if (l instanceof MultiLayerNetwork) {
                            List<INDArray> temp = ((MultiLayerNetwork) l).rnnActivateUsingStoredState(
                                    current.getInputs()[0], train, storeLastForTBPTT);
                            out = temp.get(temp.size() - 1);
                        } else {
                            //non-recurrent layer
                            out = current.doForward(train, workspaceMgr);
                        }
                    } else {
                        out = current.doForward(train, workspaceMgr);
                    }
                } else {
                    throw new IllegalArgumentException("Unsupported forward pass type for this method: " + fwdPassType);
                }

                validateArrayWorkspaces(workspaceMgr, out, ArrayType.ACTIVATIONS, vName, false, "Feed forward (inference)");
            }

            activations.put(current.getVertexName(), out.detach());

            if(inputsTo != null) {  //May be null for output vertices (which don't feed into any other vertices)
                for (VertexIndices v : inputsTo) {
                    //Note that we don't have to do anything special here: the activations are always detached in
                    // this method
                    int inputToIndex = v.getVertexIndex();
                    int vIdxEdge = v.getVertexEdgeNumber();
                    vertices[inputToIndex].setInput(vIdxEdge, out, workspaceMgr);
                }
            }

            if(clearLayers) {
                current.clear();
            }


            if(traceLog) {
                log.trace("Completed forward pass: {} (\"{}\") - {}", i, vName, current.getClass().getSimpleName());
            }
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(null);

        return activations;
    }

    /**
     * Feed-forward through the network - if workspaces are used, all returned activations will be present in workspace
     * WS_ALL_LAYERS_ACT.<br>
     * Note: if using workspaces for training, requires that WS_ALL_LAYERS_ACT is open externally.
     * If using NO workspaces, requires that no external workspace is open
     *
     * @param train             Training mode (true) or test/inference mode (false)
     * @param layerIndex        Index (inclusive) to stop forward pass at. For all layers, use -1
     * @param excludeIdxs       Layers (vertices) to exclude from forward pass. These layers will be skipped, and hence
     *                          are usually output layers or at the end of the network. May be null.
     * @param fwdPassType       Type of forward pass to perform (STANDARD or RNN_ACTIVATE_WITH_STORED_STATE only)
     * @param storeLastForTBPTT ONLY used if fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE
     * @param input             Input feature arrays
     * @param fMask             Feature mask arrays. May be null.
     * @param lMask             Label mask array. May be null.
     * @param clearInputs       Whether the layer inputs should be cleared
     * @return Map of activations (including the input), in workspace WS_ALL_LAYERS_ACT if workspaces are used (detached
     * otherwise)
     */
    protected  Map<String,INDArray> ffToLayerActivationsInWS(boolean train, int layerIndex, int[] excludeIdxs,
                                                             FwdPassType fwdPassType, boolean storeLastForTBPTT,
                                                             INDArray[] input, INDArray[] fMask, INDArray[] lMask, boolean clearInputs) {
        if(layerIndex != -1 && (layerIndex < 0 || layerIndex >= topologicalOrder.length)){
            throw new IllegalArgumentException("Invalid input index - index must be >= 0 and < " + topologicalOrder.length
                    + ", got index " + layerIndex);
        }
        setInputs(input);
        setLayerMaskArrays(fMask, lMask);

        LayerWorkspaceMgr workspaceMgr;
        WorkspaceMode wsm = (train ? configuration.getTrainingWorkspaceMode() : configuration.getInferenceWorkspaceMode());
        if(wsm == WorkspaceMode.NONE) {
            //Verify that no workspace is open externally
            WorkspaceUtils.assertNoWorkspacesOpen("Expected no workspace active in ffToLayerActivationsDetached", true);

            workspaceMgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            WorkspaceUtils.assertOpenAndActive(WS_ALL_LAYERS_ACT, "ffToLayerActivationsInWs method requires workspace WS_ALL_LAYERS_ACT to be open");

            workspaceMgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();

            if(input[0].isAttached()) {
                //Don't leverage out of async DataMultiSetIterator workspaces
                workspaceMgr.setNoLeverageOverride(input[0].data().getParentWorkspace().getId());
            }

            if(configuration.getCacheMode() != CacheMode.NONE) {
                //For now: store cache mode activations in activations workspace
                workspaceMgr.setWorkspace(ArrayType.FF_CACHE, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG);
            }
        }
        workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

        boolean traceLog = log.isTraceEnabled();
        Map<String, INDArray> activations = new HashMap<>();
        //Do forward pass according to the topological ordering of the network
        int stopIndex;
        if (layerIndex > 0) {
            stopIndex = ArrayUtils.indexOf(topologicalOrder, layerIndex);
        } else {
            stopIndex = topologicalOrder.length - 1;
        }
        for (int i = 0; i <= stopIndex; i++) {
            GraphVertex current = vertices[topologicalOrder[i]];
            String vName = current.getVertexName();
            int vIdx = current.getVertexIndex();

            if(traceLog) {
                log.trace("About forward pass: {} (\"{}\") - {}", i, vName, current.getClass().getSimpleName());
            }

            if(excludeIdxs != null && ArrayUtils.contains(excludeIdxs, vIdx)) {
                continue;
            }


            VertexIndices[] inputsTo = current.getOutputVertices();

            try(MemoryWorkspace wsFFWorking = workspaceMgr.notifyScopeEntered(ArrayType.FF_WORKING_MEM)) {

                INDArray out;
                if (current.isInputVertex()) {
                    out = inputs[vIdx];
                } else {
                    if (fwdPassType == FwdPassType.STANDARD) {
                        out = current.doForward(train, workspaceMgr);
                    } else if (fwdPassType == FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE) {
                        if (current.hasLayer()) {
                            Layer l = current.getLayer();
                            if (l instanceof RecurrentLayer) {
                                out = ((RecurrentLayer) l).rnnActivateUsingStoredState(
                                        current.getInputs()[0], train,
                                        storeLastForTBPTT, workspaceMgr);
                            } else if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer &&
                                    ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying() instanceof RecurrentLayer) {
                                RecurrentLayer rl = (RecurrentLayer) ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying();
                                out = rl.rnnActivateUsingStoredState(current.getInputs()[0], train, storeLastForTBPTT, workspaceMgr);
                            } else if (l instanceof MultiLayerNetwork) {
                                List<INDArray> temp = ((MultiLayerNetwork) l).rnnActivateUsingStoredState(
                                        current.getInputs()[0], train, storeLastForTBPTT);
                                out = temp.get(temp.size() - 1);
                            } else {
                                //non-recurrent layer
                                out = current.doForward(train, workspaceMgr);
                            }
                        } else {
                            out = current.doForward(train, workspaceMgr);
                        }
                    } else {
                        throw new IllegalStateException("FwdPassType not supported for this method: " + fwdPassType);
                    }

                    validateArrayWorkspaces(workspaceMgr, out, ArrayType.ACTIVATIONS, vName, false, "Feed forward (inference)");
                }

                activations.put(current.getVertexName(), out);

                if (inputsTo != null) {
                    //Can be null for output layers
                    for (VertexIndices v : inputsTo) {
                        //Note that we don't have to do anything special here: the activations are always detached in
                        // this method
                        int inputToIndex = v.getVertexIndex();
                        int vIdxEdge = v.getVertexEdgeNumber();
                        vertices[inputToIndex].setInput(vIdxEdge, out, workspaceMgr);
                    }
                }

                if (clearInputs) {
                    current.clear();
                }


                if (traceLog) {
                    log.trace("Completed forward pass: {} (\"{}\") - {}", i, vName, current.getClass().getSimpleName());
                }
            }
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(null);

        return activations;
    }


    /**
     * Provide the output of the specified layers, detached from any workspace. This is most commonly used at inference/test
     * time, and is more memory efficient than {@link #ffToLayerActivationsDetached(boolean, FwdPassType, boolean, int, int[], INDArray[], INDArray[], INDArray[], boolean)}
     * and {@link #ffToLayerActivationsInWS(boolean, int, int[], FwdPassType, boolean, INDArray[], INDArray[], INDArray[], boolean)}.<br>
     * This method clears all layer inputs.
     *
     * NOTE: in general, no workspaces should be activated externally for this method!
     * This method handles the workspace activation as required
     *
     * @param train             Training mode (true) or test/inference mode (false)
     * @param fwdPassType       Type of forward pass to perform (STANDARD or RNN_TIMESTEP only)
     * @param layerIndexes      Indexes of the layers to get the activations for
     * @param features          Input features for the network
     * @param fMask             Input/feature mask array. May be null.
     * @param lMasks            Labels mask array. May be null
     * @param clearLayerInputs  If true: the layer input fields will be cleared
     * @param detachedInputs    If true: the layer input fields will be detached. Usually used for external errors cases
     * @param outputWorkspace   Optional - if provided, outputs should be placed in this workspace. NOTE: this workspace
     *                          must be open
     * @return                  Output of the specified layers, detached from any workspace
     */
    protected INDArray[] outputOfLayersDetached(boolean train, @NonNull FwdPassType fwdPassType, @NonNull int[] layerIndexes, @NonNull INDArray[] features,
                                                INDArray[] fMask, INDArray[] lMasks, boolean clearLayerInputs, boolean detachedInputs, MemoryWorkspace outputWorkspace){
        if(features.length != numInputArrays) {
            throw new IllegalArgumentException("Invalid number of input arrays: network has " + numInputArrays
                    + " inputs, got " + features.length + " input arrays");
        }
        for( int i = 0; i < layerIndexes.length; i++) {
            if(layerIndexes[i] < 0 || layerIndexes[i] >= topologicalOrder.length) {
                throw new IllegalArgumentException("Invalid input index - index must be >= 0 and < " + topologicalOrder.length
                        + ", got index " + layerIndexes[i]);
            }
        }
        setInputs(features);
        setLayerMaskArrays(fMask, lMasks);

        MemoryWorkspace outputPrevious = null;
        if(outputWorkspace == null || outputWorkspace instanceof DummyWorkspace) {
        } else {
            Preconditions.checkState(outputWorkspace.isScopeActive(), "Workspace \"" + outputWorkspace.getId() +
                    "\" was provided for the network/layer outputs. When provided, this workspace must be opened before " +
                    "calling the output method; furthermore, closing the workspace is the responsibility of the user");
            outputPrevious = outputWorkspace.getParentWorkspace();
        }


        //First: for each vertex, determine the highest index of the vertex that consumes it's output
        //Then: for each vertex, determine the forward pass step that each vertex's output has been fully consumed on
        //In other words, if vertex X -> Y and X -> Z, and topological sort order is X,a,Y,b,Z,
        //Then we know X's output activations have been fully consumed by step index 4 in the topological sort
        //thus vertexOutputsFullyConsumedByStep[X.index] = IndexOf(topologicalSort, Z.index)

        //Position in array: index of vertex. Value at position: the step (in topological order) that the activations
        // have been consumed by
        //Put another way: this is the step that it's safe to deallocate the layer's activations by closing the
        // corresponding workspace
        int[] vertexOutputsFullyConsumedByStep = new int[topologicalOrder.length];
        for(GraphVertex gv : vertices) {
            int idx = gv.getVertexIndex();
            int maxStepOfOutputTo = -1;
            VertexIndices[] outputsTo = gv.getOutputVertices();
            if(outputsTo != null) {
                //May be null for final/output layers
                for (VertexIndices vi : outputsTo) {
                    int posInTopoSort = ArrayUtils.indexOf(topologicalOrder, vi.getVertexIndex());
                    if (posInTopoSort == -1) {
                        throw new IllegalStateException("Did not find vertex " + vi.getVertexIndex() + " in topological sort array");
                    }
                    maxStepOfOutputTo = Math.max(maxStepOfOutputTo, posInTopoSort);
                }
            } else {
                maxStepOfOutputTo = topologicalOrder.length - 1;
            }
            vertexOutputsFullyConsumedByStep[idx] = maxStepOfOutputTo;
        }

        //Do forward pass according to the topological ordering of the network
        INDArray[] outputs = new INDArray[layerIndexes.length];
        int stopIndex = -1;
        for( int i = 0; i < layerIndexes.length; i++) {
            stopIndex = Math.max(stopIndex, ArrayUtils.indexOf(topologicalOrder, layerIndexes[i]));
        }
        List<LayerWorkspaceMgr> allWorkspaceManagers = new ArrayList<>();
        List<LayerWorkspaceMgr> freeWorkspaceManagers = new ArrayList<>();  //Basically used as a stack

        WorkspaceMode wsm = (train ? configuration.getTrainingWorkspaceMode() : configuration.getInferenceWorkspaceMode());
        boolean noWS = wsm == WorkspaceMode.NONE;
        LayerWorkspaceMgr allNone = noWS ? LayerWorkspaceMgr.noWorkspaces() : null;
        MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();
        Throwable t = null;
        try {
            for (int i = 0; i <= stopIndex; i++) {
                GraphVertex current = vertices[topologicalOrder[i]];
                GraphVertex prev = i > 0 ? vertices[topologicalOrder[i - 1]] : null;

                String vName = current.getVertexName();
                int vIdx = current.getVertexIndex();

                //First: determine what workspace manager we should use for forward pass in this vertex
                LayerWorkspaceMgr workspaceMgr;
                if (noWS) {
                    workspaceMgr = allNone;
                } else {
                    //First: is there a free forward pass workspace we can use?
                    if (freeWorkspaceManagers.size() > 0) {
                        workspaceMgr = freeWorkspaceManagers.remove(freeWorkspaceManagers.size() - 1);
                    } else {
                        //No existing free workspace managers for forward pass - create a new one...
                        String wsName = "WS_LAYER_ACT_" + allWorkspaceManagers.size();
                        workspaceMgr = LayerWorkspaceMgr.builder()
                                .with(ArrayType.INPUT, wsName, WS_LAYER_ACT_X_CONFIG)
                                .with(ArrayType.ACTIVATIONS, wsName, WS_LAYER_ACT_X_CONFIG)
                                .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                                .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                                .build();

                        if (detachedInputs) {
                            //Sometimes (like: external errors use cases) we don't want the activations/inputs to be
                            // in a workspace
                            workspaceMgr.setScopedOutFor(ArrayType.INPUT);
                            workspaceMgr.setScopedOutFor(ArrayType.ACTIVATIONS);
                        } else {
                            //Don't leverage out of async MultiDataSetIterator workspaces
                            if (features[0].isAttached()) {
                                workspaceMgr.setNoLeverageOverride(features[0].data().getParentWorkspace().getId());
                            }
                        }

                        allWorkspaceManagers.add(workspaceMgr);
                    }
                }

                // workspaceMgr.keepOpen(ArrayType.values());
                workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

                //Is this one of the layers/vertices that we want the output for?
                boolean isRequiredOutput = false;
                String origWSAct = null;
                WorkspaceConfiguration origWSActConf = null;
                if (ArrayUtils.contains(layerIndexes, vIdx)) {
                    isRequiredOutput = true;

                    if (outputWorkspace != null && !(outputWorkspace instanceof DummyWorkspace)) {
                        //Place activations in user-specified workspace
                        origWSAct = workspaceMgr.getWorkspaceName(ArrayType.ACTIVATIONS);
                        origWSActConf = workspaceMgr.getConfiguration(ArrayType.ACTIVATIONS);
                        workspaceMgr.setWorkspace(ArrayType.ACTIVATIONS, outputWorkspace.getId(), outputWorkspace.getWorkspaceConfiguration());
                    } else {
                        //Standard case
                        if (!workspaceMgr.isScopedOut(ArrayType.ACTIVATIONS)) {
                            //Activations/output to return: don't want this in any workspace
                            origWSAct = workspaceMgr.getWorkspaceName(ArrayType.ACTIVATIONS);
                            origWSActConf = workspaceMgr.getConfiguration(ArrayType.ACTIVATIONS);
                            workspaceMgr.setScopedOutFor(ArrayType.ACTIVATIONS);
                        }
                    }
                }


                VertexIndices[] inputsTo = current.getOutputVertices();

                INDArray out = null;
                if (current.isInputVertex()) {
                    out = features[vIdx];

                } else {

                    if (fwdPassType == FwdPassType.STANDARD) {
                        //Standard feed-forward case

                        if(i > 0 && current.hasLayer() && prev.hasLayer() &&
                                ConvolutionUtils.layerHasConvolutionLayout(prev.getLayer().conf().getLayer())
                                && ConvolutionUtils.layerHasConvolutionLayout(current.getLayer().conf().getLayer())) {

                            /**
                             * Not QUITE the proper fix, but getting close.
                             * Able to detect this happens mid graph and do something about it.
                             * Need to play with output sizes a bit to make sure we put the right parameters in there to get
                             * correct behavior.
                             */
                            CNN2DFormat preLayerFormat = ConvolutionUtils.getFormatForLayer(prev.getLayer().conf().getLayer());
                            CNN2DFormat currLayerFormat = ConvolutionUtils.getFormatForLayer(current.getLayer().conf().getLayer());
                            if(preLayerFormat != currLayerFormat) {
                                int inputIdx = -1;
                                for(int inputVertex = 0; inputVertex < current.getInputVertices().length; inputVertex++) {
                                    if(current.getInputVertices()[inputVertex].getVertexIndex() == prev.getVertexIndex()) {
                                        inputIdx = inputVertex;
                                    }
                                }

                                //NHWC case
                                if(preLayerFormat == CNN2DFormat.NCHW) {
                                    current.setInput(inputIdx,current.getInputs()[inputIdx].permute(0,3,1,2),workspaceMgr);
                                }
                                //NCHW case
                                else if(preLayerFormat == CNN2DFormat.NHWC) {
                                    current.setInput(inputIdx,current.getInputs()[inputIdx].permute(0,2,3,1),workspaceMgr);

                                }
                                else
                                    throw new IllegalStateException("No CNN2DDataFormat type found for previous layer!");

                                out = current.doForward(train, workspaceMgr);
                            }
                            else
                                out = current.doForward(train, workspaceMgr);
                        } else    if(i > 0 && current.hasLayer() && prev.hasLayer() &&
                                Convolution1DUtils.hasRnnDataFormat(prev.getLayer().conf().getLayer())
                                && Convolution1DUtils.hasRnnDataFormat(current.getLayer().conf().getLayer())) {
                            RNNFormat preLayerFormat = Convolution1DUtils.getRnnFormatFromLayer(prev.getLayer().conf().getLayer());
                            RNNFormat currLayerFormat = Convolution1DUtils.getRnnFormatFromLayer(current.getLayer().conf().getLayer());
                            int inputIdx = -1;
                            for(int inputVertex = 0; inputVertex < current.getInputVertices().length; inputVertex++) {
                                if(current.getInputVertices()[inputVertex].getVertexIndex() == prev.getVertexIndex()) {
                                    inputIdx = inputVertex;
                                }
                            }
                            //permute for next layer
                            if(preLayerFormat != currLayerFormat) {
                                current.setInput(inputIdx,current.getInputs()[inputIdx].permute(0,2,1),workspaceMgr);
                            }
                            out = current.doForward(train, workspaceMgr);


                        }  else {
                            out = current.doForward(train, workspaceMgr);
                        }

                    } else if (fwdPassType == FwdPassType.RNN_TIMESTEP) {
                        if (current.hasLayer()) {
                            //Layer
                            INDArray input = current.getInputs()[0];
                            Layer l = current.getLayer();
                            if (l instanceof RecurrentLayer) {
                                out = ((RecurrentLayer) l).rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                            } else if (l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer && ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying() instanceof RecurrentLayer) {
                                RecurrentLayer rl = ((RecurrentLayer) ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer) l).getUnderlying());
                                out = rl.rnnTimeStep(reshapeTimeStepInput(input), workspaceMgr);
                            } else if (l instanceof MultiLayerNetwork) {
                                out = ((MultiLayerNetwork) l).rnnTimeStep(reshapeTimeStepInput(input));
                            } else {
                                //non-recurrent layer
                                out = current.doForward(train, workspaceMgr);
                            }
                        } else {
                            //GraphNode
                            out = current.doForward(train, workspaceMgr);
                        }
                    } else {
                        throw new IllegalArgumentException("Unsupported forward pass type for this method: " + fwdPassType);
                    }

                    validateArrayWorkspaces(workspaceMgr, out, ArrayType.ACTIVATIONS, vName, false, "Feed forward (inference)");
                }

                if (inputsTo != null) {  //Output vertices may not input to any other vertices
                    for (VertexIndices v : inputsTo) {
                        //Note that we don't have to do anything special here: the activations are always detached in
                        // this method
                        int inputToIndex = v.getVertexIndex();
                        int vIdxEdge = v.getVertexEdgeNumber();
                        vertices[inputToIndex].setInput(vIdxEdge, out, workspaceMgr);
                    }
                }

                if (clearLayerInputs) {
                    current.clear();
                }

                if (isRequiredOutput) {
                    outputs[ArrayUtils.indexOf(layerIndexes, vIdx)] = out;
                    if (origWSAct != null) {
                        //Reset the configuration, as we may reuse this workspace manager...
                        workspaceMgr.setWorkspace(ArrayType.ACTIVATIONS, origWSAct, origWSActConf);
                    }
                }


            }
        } catch (Throwable t2) {
            t = t2;
        } finally {
            Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);

            if(t != null){
                if(t instanceof RuntimeException) {
                    throw ((RuntimeException)t);
                }
                throw new RuntimeException("Error during neural network forward pass", t);
            }

        }


        Nd4j.getMemoryManager().setCurrentWorkspace(outputPrevious);


        return outputs;
    }

    private INDArray reshapeTimeStepInput(INDArray input) {
        if (input.rank() == 2) { // dynamically reshape to 3D input with one time-step.
            long[] inShape = input.shape();
            input = input.reshape(inShape[0], inShape[1], 1);
        }
        return input;
    }


    /**
     * Calculate the gradient of the network with respect to some external errors.
     * Note that this is typically used for things like reinforcement learning, not typical networks that include
     * an OutputLayer or RnnOutputLayer
     *
     * @param epsilons Epsilons (errors) at the output. Same order with which the output layers are defined in configuration setOutputs(String...)
     * @return Gradient for the network
     */
    public Gradient backpropGradient(INDArray... epsilons) {
        if (epsilons == null || epsilons.length != numOutputArrays)
            throw new IllegalArgumentException(
                    "Invalid input: must have epsilons length equal to number of output arrays");


        try {
            calcBackpropGradients(true, configuration.getBackpropType() == BackpropType.TruncatedBPTT, epsilons);
            return gradient;
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Do backprop (gradient calculation)
     *
     * @param truncatedBPTT    false: normal backprop. true: calculate gradients using truncated BPTT for RNN layers
     * @param externalEpsilons null usually (for typical supervised learning). If not null (and length > 0) then assume that
     *                         the user has provided some errors externally, as they would do for example in reinforcement
     *                         learning situations.
     */
    protected void calcBackpropGradients(boolean clearLayers, boolean truncatedBPTT, INDArray... externalEpsilons) {
        if (flattenedGradients == null) {
            initGradientsView();
        }

        /*
         Design for workspaces use in backprop for ComputationGraph is similar to MultiLayerNetwork and shares some
         features with outputOfLayersDetached

         Specifically:
         1. We assume forward pass has already been done, and hence layer input fields are set (with all arrays/activations in
            workspace WS_ALL_LAYERS_ACT if appropriate)
         2. We use a set of small workspaces to contain the activation gradients for a single layer
            These are opened once per layer, and are closed only once the corresponding activation gradients have been
            consumed by all layers
         */




        //Validate the network configuration for external errors - no output layers
        if(externalEpsilons != null && externalEpsilons.length > 0) {
            List<String> outputLayers = configuration.getNetworkOutputs();
            for(String s : outputLayers ){
                GraphVertex gv = getVertex(s);
                if(gv instanceof LayerVertex && ((LayerVertex)gv).getLayer() instanceof IOutputLayer){
                    throw new IllegalStateException("Cannot perform backprop with external errors in conjunction with an output layer:" +
                            " output layers cannot use external errors for backprop. Layer name: " + s);
                }
            }

        }

        //Position in array: index of vertex. Value at position: the step (in topological order) that the activation
        // gradients of the specified vertex have been consumed by
        //Put another way: this is the step that it's safe to deallocate the layer's activation gradients by closing the
        // corresponding workspace
        //TODO we can probably cache this...
        int[] vertexActGradsFullyConsumedByStep = new int[topologicalOrder.length];
        for(GraphVertex gv : vertices){
            int idx = gv.getVertexIndex();
            int minStepOfInputFrom = Integer.MAX_VALUE;
            VertexIndices[] inputsFrom = gv.getInputVertices();
            if(inputsFrom != null) {
                //inputsFrom may be null for input vertex
                for (VertexIndices vi : inputsFrom) {
                    int posInTopoSort = ArrayUtils.indexOf(topologicalOrder, vi.getVertexIndex());
                    if (posInTopoSort == -1) {
                        throw new IllegalStateException("Did not find vertex " + vi.getVertexIndex() + " in topological sort array");
                    }
                    minStepOfInputFrom = Math.min(minStepOfInputFrom, posInTopoSort);
                }
            }

            if(minStepOfInputFrom == Integer.MAX_VALUE){
                //Input vertex, etc
                vertexActGradsFullyConsumedByStep[idx] = 0;
            } else {
                vertexActGradsFullyConsumedByStep[idx] = minStepOfInputFrom;
            }
        }


        boolean noWS = configuration.getInferenceWorkspaceMode() == WorkspaceMode.NONE;
        LayerWorkspaceMgr allNone = noWS ? LayerWorkspaceMgr.noWorkspaces() : null;

        List<LayerWorkspaceMgr> allWorkspaceManagers = new ArrayList<>();
        List<LayerWorkspaceMgr> freeWorkspaceManagers = new ArrayList<>();  //Basically used as a stack
        Map<MemoryWorkspace, LayerWorkspaceMgr> openActivationsWorkspaces = new IdentityHashMap<>();
        List<MemoryWorkspace>[] closeAtEndIteraton = (List<MemoryWorkspace>[])new List[topologicalOrder.length];

        //Do backprop, in reverse topological order
        LinkedList<Triple<String, INDArray, Character>> gradients = new LinkedList<>();
        boolean[] setVertexEpsilon = new boolean[topologicalOrder.length]; //If true: already set epsilon for this vertex; later epsilons should be *added* to the existing one, not set
        MemoryWorkspace initialWorkspace = Nd4j.getMemoryManager().getCurrentWorkspace();

        boolean traceLog = log.isTraceEnabled();

        Throwable t = null;
        try {
            for (int i = topologicalOrder.length - 1; i >= 0; i--) {
                boolean hitFrozen = false;
                GraphVertex current = vertices[topologicalOrder[i]];
                int vIdx = current.getVertexIndex();
                String vertexName = current.getVertexName();

                if (traceLog) {
                    log.trace("About backprop: {} (\"{}\") - {}", i, vertexName, current.getClass().getSimpleName());
                }

                //FIXME: make the frozen vertex feature extraction more flexible
                if (current.hasLayer() && current.getLayer() instanceof FrozenLayer || current instanceof FrozenVertex) {
                    hitFrozen = true;
                }



                //First: determine what workspace manager we should use for the activation gradients from this vertex
                LayerWorkspaceMgr workspaceMgr;
                if (noWS) {
                    workspaceMgr = allNone;
                } else {
                    //First: is there a free activation gradient workspace we can use?
                    if (freeWorkspaceManagers.size() > 0) {
                        workspaceMgr = freeWorkspaceManagers.remove(freeWorkspaceManagers.size() - 1);
                    } else {
                        //No existing free workspace managers for forward pass - create a new one...
                        String wsName = "WS_LAYER_ACT_" + allWorkspaceManagers.size();
                        workspaceMgr = LayerWorkspaceMgr.builder()
                                .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                                .with(ArrayType.ACTIVATION_GRAD, wsName, WS_LAYER_ACT_X_CONFIG)
                                .with(ArrayType.ACTIVATIONS, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG) //For forward pass in the context of BP
                                .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                                .with(ArrayType.BP_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                                .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                                .with(ArrayType.RNN_BP_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                                .build();

                        allWorkspaceManagers.add(workspaceMgr);
                    }
                }
                workspaceMgr.setHelperWorkspacePointers(helperWorkspaces);

                for(LayerWorkspaceMgr layerWorkspaceMgr : allWorkspaceManagers) {
                    // layerWorkspaceMgr.keepOpen(ArrayType.values());
                }

                if (current.isOutputVertex()) {
                    //Two reasons for a vertex to be an output vertex:
                    //(a) it's an output layer (i.e., instanceof IOutputLayer), or
                    //(b) it's a normal layer, but it has been marked as an output layer for use in external errors - for reinforcement learning, for example

                    int thisOutputNumber = configuration.getNetworkOutputs().indexOf(current.getVertexName());
                    Layer currentLayer = current.getLayer();
                    if (currentLayer instanceof FrozenLayerWithBackprop) {
                        currentLayer = ((FrozenLayerWithBackprop) currentLayer).getInsideLayer();
                    }
                    if (currentLayer instanceof IOutputLayer) {
                        IOutputLayer outputLayer = (IOutputLayer) currentLayer;

                        INDArray currLabels = labels[thisOutputNumber];
                        outputLayer.setLabels(currLabels);
                    } else {
                        if ((externalEpsilons == null || externalEpsilons.length == 0)
                                && labels[thisOutputNumber] != null) {
                            throw new DL4JException("Layer \"" + current.getVertexName() + "\" of type "
                                    + current.getLayer().getClass().getSimpleName()
                                    + " is set as network output "
                                    + "(but isn't an IOutputLayer). Only IOutputLayer layers can be fit via backprop with"
                                    + " a labels array. ");
                        }
                        current.setEpsilon(externalEpsilons[thisOutputNumber]);
                        setVertexEpsilon[topologicalOrder[i]] = true;
                    }
                }

                //Actually execute backprop for the specified vertex
                //First: Open the relevant workspace for the activations.
                //Note that this will be closed only once the current vertex's activations have been consumed
                MemoryWorkspace wsActivationGrads = workspaceMgr.notifyScopeEntered(ArrayType.ACTIVATION_GRAD);
                openActivationsWorkspaces.put(wsActivationGrads, workspaceMgr);

                //Note that because we're opening activation gradient workspaces not in any defined order (i.e., workspace
                // use isn't simply nested), we'll manually override the previous workspace setting. Otherwise, when we
                // close these workspaces, the "current" workspace may be set to the incorrect one
                wsActivationGrads.setPreviousWorkspace(initialWorkspace);

                int closeableAt = vertexActGradsFullyConsumedByStep[vIdx];
                if (closeableAt >= 0) {
                    if (closeAtEndIteraton[closeableAt] == null) {
                        closeAtEndIteraton[closeableAt] = new ArrayList<>();
                    }
                    closeAtEndIteraton[closeableAt].add(wsActivationGrads);
                }

                Pair<Gradient, INDArray[]> pair;
                INDArray[] epsilons;
                try (MemoryWorkspace wsWorkingMem = workspaceMgr.notifyScopeEntered(ArrayType.BP_WORKING_MEM)) {

                    pair = current.doBackward(truncatedBPTT, workspaceMgr);
                    epsilons = pair.getSecond();

                    //Validate workspace location for the activation gradients:
                    for (INDArray epsilon : epsilons) {
                        if (epsilon != null) {
                            //May be null for EmbeddingLayer, etc
                            validateArrayWorkspaces(workspaceMgr, epsilon, ArrayType.ACTIVATION_GRAD, vertexName, false, "Backprop");
                        }
                    }
                }

                //Inputs to the current GraphVertex:
                VertexIndices[] inputVertices = current.getInputVertices();

                //Set epsilons for the vertices that provide inputs to this vertex:
                if (inputVertices != null) {
                    int j = 0;
                    for (VertexIndices v : inputVertices) {
                        GraphVertex gv = vertices[v.getVertexIndex()];
                        if (setVertexEpsilon[gv.getVertexIndex()]) {
                            //This vertex: must output to multiple vertices... we want to add the epsilons here
                            INDArray currentEps = gv.getEpsilon();
                            if(currentEps == null){
                                //Edge case: this can be null for dual embedding layer case - in -> e1, in -> e2
                                gv.setEpsilon(currentEps);
                            } else {
                                gv.setEpsilon(currentEps.addi(epsilons[j++]));  //TODO is this always safe?
                            }
                        } else {
                            gv.setEpsilon(epsilons[j++]);
                        }
                        setVertexEpsilon[gv.getVertexIndex()] = true;
                    }
                }

                if (pair.getFirst() != null) {
                    Gradient g = pair.getFirst();
                    Map<String, INDArray> map = g.gradientForVariable();
                    LinkedList<Triple<String, INDArray, Character>> tempList = new LinkedList<>();
                    for (Map.Entry<String, INDArray> entry : map.entrySet()) {
                        String origName = entry.getKey();
                        String newName = current.getVertexName() + "_" + origName;
                        tempList.addFirst(new Triple<>(newName, entry.getValue(),
                                g.flatteningOrderForVariable(origName)));
                    }
                    for (Triple<String, INDArray, Character> triple : tempList)
                        gradients.addFirst(triple);
                }


                if (traceLog) {
                    log.trace("Completed backprop: {} (\"{}\") - {}", i, vertexName, current.getClass().getSimpleName());
                }
            }
        } catch (Throwable t2) {
            t = t2;
        } finally {
            Nd4j.getMemoryManager().setCurrentWorkspace(initialWorkspace);

            if(t != null){
                if(t instanceof RuntimeException) {
                    throw ((RuntimeException)t);
                }
                throw new RuntimeException("Error during neural network backpropagation calculation", t);
            }
        }

        //Now, add the gradients in the order we need them in for flattening (same as params order)
        Gradient gradient = new DefaultGradient(flattenedGradients);
        for (Triple<String, INDArray, Character> tr : gradients) {
            gradient.setGradientFor(tr.getFirst(), tr.getSecond(), tr.getThird());
        }

        this.gradient = gradient;

        if(truncatedBPTT && clearTbpttState) {
            rnnClearPreviousState();
        }

        //Clear inputs and epsilons:
        if(clearLayers) {
            for (GraphVertex gv : vertices) {
                gv.clear();
            }
        }

        Nd4j.getMemoryManager().setCurrentWorkspace(null);


    }

    @Override
    public ComputationGraph clone() {
        ComputationGraph cg = new ComputationGraph(configuration.clone());
        cg.init(params().dup(), false);
        if (solver != null) {
            //If  solver is null: updater hasn't been initialized -> getUpdater call will force initialization, however
            ComputationGraphUpdater u = this.getUpdater();
            INDArray updaterState = u.getStateViewArray();
            if (updaterState != null) {
                cg.getUpdater().setStateViewArray(updaterState.dup());
            }
        }
        cg.trainingListeners = this.trainingListeners;
        for (int i = 0; i < topologicalOrder.length; i++) {
            if (!vertices[topologicalOrder[i]].hasLayer())
                continue;
            String layerName = vertices[topologicalOrder[i]].getVertexName();
            if (getLayer(layerName) instanceof FrozenLayer) {
                cg.getVertex(layerName).setLayerAsFrozen();
            }
        }
        return cg;
    }


    public double calcRegularizationScore(boolean backpropParamsOnly){
        double scoreSum = 0.0;
        for (int i = 0; i < layers.length; i++) {
            scoreSum += layers[i].calcRegularizationScore(backpropParamsOnly);
        }
        return scoreSum;
    }

    /**
     * Set the trainingListeners for the ComputationGraph (and all layers in the network)
     */
    public void setListeners(Collection<TrainingListener> listeners) {
        if (layers == null)
            init();

        for (Layer l : layers) {
            l.setListeners(listeners);
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
     * Set the trainingListeners for the ComputationGraph (and all layers in the network)
     */
    public void setListeners(TrainingListener... listeners) {
        List<TrainingListener> list = new ArrayList<>();
        //Check: user might have done setListeners(null) thinking this would clear the current listeners.
        //This results in an TrainingListener[1] with a single null value -> results in a NPE later
        if (listeners != null && listeners.length > 0) {
            for (TrainingListener i : listeners) {
                if (i != null)
                    list.add(i);
            }
        }
        setListeners(list);
    }

    /**
     * This method ADDS additional TrainingListener to existing listeners
     *
     * @param listeners Listeners to add
     */
    @Override
    public void addListeners(TrainingListener... listeners) {
        if (this.trainingListeners == null) {
            setListeners(listeners);
            return;
        } else {
            List<TrainingListener> newListeners = new ArrayList<>(this.trainingListeners);   //To avoid immutable list issues
            Collections.addAll(newListeners, listeners);
            setListeners(newListeners);
        }

        if (solver != null) {
            solver.setListeners(this.trainingListeners);
        }
    }

    /**
     * Get the trainingListeners for the ComputationGraph
     */
    public Collection<TrainingListener> getListeners() {
        return trainingListeners;
    }

    /**
     * Get the ComputationGraphUpdater for the network. Creates one on demand, if required
     */
    public ComputationGraphUpdater getUpdater() {
        return getUpdater(true);
    }

    /**
     * Get the ComputationGraphUpdater for this network
     * @param initializeIfAbsent If true: create the updater if one is absent. False: return null if absent.
     * @return Updater
     */
    public ComputationGraphUpdater getUpdater(boolean initializeIfAbsent) {
        if (solver == null && initializeIfAbsent) {
            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
            solver.getOptimizer().setUpdater(new ComputationGraphUpdater(this));
        }
        if(solver != null) {
            return (ComputationGraphUpdater) solver.getOptimizer().getUpdater(initializeIfAbsent);
        }
        return null;
    }

    /**
     * Set the computationGraphUpdater for the network
     */
    public void setUpdater(ComputationGraphUpdater updater) {
        if (solver == null) {
            solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this).build();
        }
        solver.getOptimizer().setUpdater(updater);
    }

    /**
     * Get the specified output layer, by index. The index of the output
     * layer may be 0 to {@link #getNumOutputArrays()}-1
     */
    public Layer getOutputLayer(int outputLayerIdx) {
        if (outputLayerIdx >= numOutputArrays)
            throw new IllegalArgumentException("Invalid index: cannot get output layer " + outputLayerIdx
                    + ", total number of network outputs = " + numOutputArrays);
        return getLayer(configuration.getNetworkOutputs().get(outputLayerIdx));
    }

    /**
     * @deprecated To be removed. Use {@link #params()}
     */
    @Deprecated
    public INDArray params(boolean backwardOnly) {
        return params();
    }

    /**
     * Sets the input and labels and returns a score for the prediction with respect to the true labels<br>
     * This is equivalent to {@link #score(DataSet, boolean)} with training==true.<br>
     * <b>NOTE:</b> this version of the score function can only be used with ComputationGraph networks that have
     * a single input and a single output.
     *
     * @param dataSet the data to score
     * @return the score for the given input,label pairs
     * @see #score(DataSet, boolean)
     */
    public double score(DataSet dataSet) {
        return score(dataSet, false);
    }

    /**
     * Sets the input and labels and returns a score for the prediction with respect to the true labels<br>
     * <b>NOTE:</b> this version of the score function can only be used with ComputationGraph networks that have
     * a single input and a single output. Use {@link #score(MultiDataSet, boolean)} for multiple input/output networks
     *
     * @param dataSet  the data to score
     * @param training whether score is being calculated at training time (true) or test time (false)
     * @return the score for the given input,label pairs
     * @see #score(DataSet, boolean)
     */
    public double score(DataSet dataSet, boolean training) {
        if (numInputArrays != 1 || numOutputArrays != 1)
            throw new UnsupportedOperationException("Cannot score ComputationGraph network with "
                    + " DataSet: network does not have 1 input and 1 output arrays");
        return score(ComputationGraphUtil.toMultiDataSet(dataSet), training);
    }

    /**
     * Score the network given the MultiDataSet, at test time
     */
    public double score(MultiDataSet dataSet) {
        return score(dataSet, false);
    }

    /**
     * Sets the input and labels and returns a score for the prediction with respect to the true labels<br>
     *
     * @param dataSet  the data to score
     * @param training whether score is being calculated at training time (true) or test time (false)
     * @return the score for the given input,label pairs
     */
    public double score(MultiDataSet dataSet, boolean training) {
        try {
            return scoreHelper(dataSet, training);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private double scoreHelper(MultiDataSet dataSet, boolean training) {
        LayerWorkspaceMgr mgr;
        WorkspaceMode wsm = (training ? configuration.getTrainingWorkspaceMode() : configuration.getInferenceWorkspaceMode());
        if(wsm == WorkspaceMode.NONE) {
            mgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            mgr = LayerWorkspaceMgr.builder()
                    .noWorkspaceFor(ArrayType.ACTIVATIONS)
                    .noWorkspaceFor(ArrayType.INPUT)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();
        }
        mgr.setHelperWorkspacePointers(helperWorkspaces);

        boolean hasMaskArrays = dataSet.hasMaskArrays();
        if (hasMaskArrays) {
            setLayerMaskArrays(dataSet.getFeaturesMaskArrays(), dataSet.getLabelsMaskArrays());
        }

        double score = 0.0;
        setInputs(dataSet.getFeatures());
        //TODO Can possibly optimize this, in terms of memory use/workspaces
        Map<String, INDArray> stringINDArrayMap = ffToLayerActivationsDetached(training,
                FwdPassType.STANDARD,
                false, vertices.length - 1,
                getOutputLayerIndices(),
                dataSet.getFeatures(),
                dataSet.getFeaturesMaskArrays(),
                dataSet.getLabelsMaskArrays(),
                false);


        INDArray[] labels = dataSet.getLabels();
        setLabels(labels);

        //Score: sum of the scores for the various output layers...
        double r = calcRegularizationScore(true);

        int i = 0;
        for (String s : configuration.getNetworkOutputs()) {
            GraphVertex gv = verticesMap.get(s);
            Layer outLayer = gv.getLayer();
            if (outLayer == null || !(outLayer instanceof IOutputLayer)) {
                log.warn("Cannot calculate score: vertex \"" + s + "\" is not an output layer");
                return 0.0;
            }

            IOutputLayer ol = (IOutputLayer) outLayer;
            ol.setLabels(labels[i++]);

            score += ((LayerVertex) gv).computeScore(r, training, mgr);

            //Only want to add l1/l2 once...
            r = 0.0;
        }


        clearLayersStates();    //Clean up layer inputs/mask arrays - may be invalidated by workspace
        return score;
    }

    /**
     * Calculate the score for each example in a DataSet individually. Unlike {@link #score(DataSet)} and {@link #score(DataSet, boolean)}
     * this method does not average/sum over examples. This method allows for examples to be scored individually (at test time only), which
     * may be useful for example for autoencoder architectures and the like.<br>
     * Each row of the output (assuming addRegularizationTerms == true) is equivalent to calling score(DataSet) with a single example.
     *
     * @param data                   The data to score
     * @param addRegularizationTerms If true: add l1/l2 regularization terms (if any) to the score. If false: don't add regularization terms
     * @return An INDArray (column vector) of size input.numRows(); the ith entry is the score (loss value) of the ith example
     */
    public INDArray scoreExamples(DataSet data, boolean addRegularizationTerms) {
        if (numInputArrays != 1 || numOutputArrays != 1)
            throw new UnsupportedOperationException("Cannot score ComputationGraph network with "
                    + " DataSet: network does not have 1 input and 1 output arrays");
        return scoreExamples(ComputationGraphUtil.toMultiDataSet(data), addRegularizationTerms);
    }

    /**
     * Calculate the score for each example in a DataSet individually. Unlike {@link #score(MultiDataSet)} and {@link #score(MultiDataSet, boolean)}
     * this method does not average/sum over examples. This method allows for examples to be scored individually (at test time only), which
     * may be useful for example for autoencoder architectures and the like.<br>
     * Each row of the output (assuming addRegularizationTerms == true) is equivalent to calling score(MultiDataSet) with a single example.
     *
     * @param dataSet                The data to score
     * @param addRegularizationTerms If true: add l1/l2 regularization terms (if any) to the score. If false: don't add regularization terms
     * @return An INDArray (column vector) of size input.numRows(); the ith entry is the score (loss value) of the ith example
     */
    public INDArray scoreExamples(MultiDataSet dataSet, boolean addRegularizationTerms) {
        try{
            return scoreExamplesHelper(dataSet, addRegularizationTerms);
        } catch (OutOfMemoryError e) {
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private INDArray scoreExamplesHelper(MultiDataSet dataSet, boolean addRegularizationTerms) {
        LayerWorkspaceMgr mgr;
        if(configuration.getInferenceWorkspaceMode() == WorkspaceMode.NONE) {
            mgr = LayerWorkspaceMgr.noWorkspaces();
        } else {
            mgr = LayerWorkspaceMgr.builder()
                    .with(ArrayType.ACTIVATIONS, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.INPUT, WS_ALL_LAYERS_ACT, WS_ALL_LAYERS_ACT_CONFIG)
                    .with(ArrayType.FF_WORKING_MEM, WS_LAYER_WORKING_MEM, WS_LAYER_WORKING_MEM_CONFIG)
                    .with(ArrayType.RNN_FF_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM, WS_RNN_LOOP_WORKING_MEM_CONFIG)
                    .build();
        }
        mgr.setHelperWorkspacePointers(helperWorkspaces);

        boolean hasMaskArrays = dataSet.hasMaskArrays();
        if (hasMaskArrays) {
            setLayerMaskArrays(dataSet.getFeaturesMaskArrays(), dataSet.getLabelsMaskArrays());
        }

        INDArray out = null;
        setInputs(dataSet.getFeatures());

        //Need to feed forward, but not the output layers
        //TODO maybe optimize? We only need *some* of the activations in the WS...
        //mgr.keepOpen(ArrayType.values());
        ffToLayerActivationsInWS(false, vertices.length - 1, getOutputLayerIndices(), FwdPassType.STANDARD, false,
                dataSet.getFeatures(), dataSet.getFeaturesMaskArrays(), dataSet.getLabelsMaskArrays(), false);

        INDArray[] labels = dataSet.getLabels();
        setLabels(labels);


        double r = (addRegularizationTerms ? calcRegularizationScore(true) : 0.0);
        int i = 0;
        for (String s : configuration.getNetworkOutputs()) {
            GraphVertex gv = verticesMap.get(s);
            Layer outLayer = gv.getLayer();
            if (outLayer == null || !(outLayer instanceof IOutputLayer)) {
                throw new UnsupportedOperationException(
                        "Cannot calculate score: vertex \"" + s + "\" is not an output layer");
            }

            IOutputLayer ol = (IOutputLayer) outLayer;
            ol.setLabels(labels[i++]);

            INDArray scoreCurrLayer;;

            scoreCurrLayer =((LayerVertex) gv).computeScoreForExamples(r, mgr);

            if (out == null)
                out = scoreCurrLayer.detach();
            else
                out.addi(scoreCurrLayer);

            //Only want to add l1/l2 once...
            r = 0.0;
        }


        if (dataSet.hasMaskArrays())
            clearLayerMaskArrays();
        clearLayersStates();
        return out;
    }


    //------------------------------------------------------
    //Model methods:

    @Override
    public void fit() {
        fit(inputs, labels, inputMaskArrays, labelMaskArrays);
    }

    @Override
    public void update(INDArray gradient, String paramType) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void update(Gradient gradient) {
        if (gradient.gradient().length() != numParams(true))
            throw new IllegalArgumentException("Invalid input: expect gradients array of length " + numParams(true));
        for (Map.Entry<String, INDArray> entry : gradient.gradientForVariable().entrySet()) {
            String key = entry.getKey();
            INDArray val = entry.getValue();
            int idx = key.lastIndexOf('_');
            if (idx == -1)
                throw new IllegalStateException("Invalid param key: not have layer separator: \"" + key + "\"");
            String layerName = key.substring(0, idx);
            String paramType = key.split("_")[1];
            // Update graph gradient
            this.gradient.gradientForVariable().put(key, val);
            // Update layer params
            getLayer(layerName).update(val, paramType);
        }
        // Update layerwise gradient view
        setBackpropGradientsViewArray(gradient.gradient());
    }


    @Override
    public double score() {
        return score;
    }

    public void setScore(double score) {
        this.score = score;
    }

    @Override
    public INDArray params() {
        if(flattenedParams == null)
            return Nd4j.zeros(DataType.FLOAT,0);

        if(flattenedParams.rank() > 1 && !flattenedParams.wasClosed())
            return flattenedParams.reshape(flattenedParams.length());
        return flattenedParams;
    }

    @Override
    public INDArray updaterState() {
        return getUpdater() != null ? getUpdater().getUpdaterStateViewArray() : null;
    }

    @Override
    public long numParams() {
        return numParams(true);
    }

    @Override
    public long numParams(boolean backwards) {
        int nParams = 0;
        for (Layer layer : layers) {
            nParams += layer.numParams(backwards);
        }
        return nParams;
    }

    @Override
    public void setParams(INDArray params) {
        if (params == flattenedParams)
            return; //No op

        if (this.flattenedParams != null && this.flattenedParams.length() == params.length()) {
            this.flattenedParams.assign(params.reshape(flattenedParams.shape()));
            return;
        }

        INDArray paramsViewReshape = params.reshape(params.length());
        int idx = 0;
        for (int i = 0; i < topologicalOrder.length; i++) {
            if (!vertices[topologicalOrder[i]].hasLayer())
                continue;

            Layer layer = vertices[topologicalOrder[i]].getLayer();
            long range = layer.numParams();
            if (range <= 0)
                continue; //Some layers: no parameters (subsampling etc)
            INDArray get = paramsViewReshape.get(NDArrayIndex.interval(idx, range + idx));
            layer.setParams(get);
            idx += range;
        }
    }

    @Override
    public void setParamsViewArray(INDArray gradient) {
        throw new UnsupportedOperationException("Not supported");
    }

    @Override
    public INDArray getGradientsViewArray() {
        return flattenedGradients;
    }

    @Override
    public void setBackpropGradientsViewArray(INDArray gradient) {
        INDArray gradientReshape = gradient.reshape(gradient.length());
        int paramsSoFar = 0;
        for (int i = 0; i < topologicalOrder.length; i++) {
            if (!vertices[topologicalOrder[i]].hasLayer())
                continue;

            Layer layer = vertices[topologicalOrder[i]].getLayer();
            long range = layer.numParams();
            if (range <= 0)
                continue; //Some layers: no parameters (subsampling etc)
            layer.setBackpropGradientsViewArray(gradientReshape.get(
                    NDArrayIndex.interval(paramsSoFar, paramsSoFar + range)));
            paramsSoFar += range;
        }
    }

    @Override
    public void fit(INDArray data, LayerWorkspaceMgr workspaceMgr){
        throw new UnsupportedOperationException("Cannot pretrain ComputationGraph with single INDArray");
    }

    @Override
    public Gradient gradient() {
        return gradient;
    }

    @Override
    public Pair<Gradient, Double> gradientAndScore() {
        return new Pair<>(gradient(), score());
    }

    @Override
    public int batchSize() {
        //In 99+% of cases, the input and labels dimension 0 size should be identical
        //The only real exceptions: space to batch, and batch to space layers
        //In those cases, we should base it on the labels size, as this impacts gradient calculation
        return labels == null || labels[0] == null ? (int) inputs[0].size(0) : (int)labels[0].size(0);
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
        if (numInputArrays == 1)
            return (inputs != null ? inputs[0] : null);
        else
            throw new UnsupportedOperationException(
                    "Cannot return single input: ComputationGraph  has multiple inputs");
    }

    @Override
    public ConvexOptimizer getOptimizer() {
        return solver.getOptimizer();
    }

    @Override
    public INDArray getParam(String paramName) {
        //        throw new UnsupportedOperationException("Not implemented");
        int idx = paramName.lastIndexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: not have layer separator: \"" + paramName + "\"");
        String layerName = paramName.substring(0, idx);
        String paramType = paramName.substring(idx + 1);
        return getLayer(layerName).getParam(paramType);

    }

    @Override
    public Map<String, INDArray> paramTable() {
        return paramTable(false);
    }

    public Map<String, INDArray> paramTable(boolean backpropParamsOnly) {
        //Get all parameters from all layers/vertices
        Map<String, INDArray> allParams = new LinkedHashMap<>();
        for(GraphVertex gv : vertices){
            Map<String, INDArray> paramMap = gv.paramTable(backpropParamsOnly);
            for (Map.Entry<String, INDArray> entry : paramMap.entrySet()) {
                String newKey = gv.getVertexName() + "_" + entry.getKey();
                allParams.put(newKey, entry.getValue());
            }
        }
        return allParams;
    }

    @Override
    public void setParamTable(@NonNull Map<String, INDArray> paramTable) {
        Map<String,INDArray> m = paramTable();
        Preconditions.checkArgument(paramTable.keySet().equals(m.keySet()), "Cannot set param table: parameter set keys are not equal");
        Map<String,INDArray> current = paramTable();
        //Check shapes before doing partial assigment to avoid leaving net in incorrect state
        for(String s : current.keySet()){
            INDArray arrCurrent = current.get(s);
            INDArray arrNew = paramTable.get(s);
            val shapeCurrent = arrCurrent.shape();
            val shapeNew = arrNew.shape();
            Preconditions.checkState(Arrays.equals(shapeCurrent, shapeNew), "Cannot set parameters: shape array for " +
                    "parameter \"%s\" does not match existing shape: parameter shape = %s, new param shape = %s", s, shapeCurrent, arrNew);
        }

        for(String s : current.keySet()) {
            INDArray arrCurrent = current.get(s);
            INDArray arrNew = paramTable.get(s);
            arrCurrent.assign(arrNew);
        }
    }

    @Override
    public void setParam(String key, INDArray val) {
        //        throw new UnsupportedOperationException("Not implemented");
        int idx = key.lastIndexOf('_');
        if (idx == -1)
            throw new IllegalStateException("Invalid param key: not have layer separator: \"" + key + "\"");
        String layerName = key.substring(0, idx);
        String paramType = key.substring(idx + 1);
        getLayer(layerName).setParam(paramType, val);
    }

    @Override
    public void clear() {
        inputs = null;
        labels = null;
        inputMaskArrays = null;
        labelMaskArrays = null;
    }

    @Override
    public void applyConstraints(int iteration, int epoch) {
        for(Layer l : layers){
            l.applyConstraints(iteration, epoch);
        }
    }

    //------------------------------------------------------------------------------
    //RNN-specific functionality

    /**
     * If this ComputationGraph contains one or more RNN layers: conduct forward pass (prediction)
     * but using previous stored state for any RNN layers. The activations for the final step are
     * also stored in the RNN layers for use next time rnnTimeStep() is called.<br>
     * This method can be used to generate output one or more steps at a time instead of always having to do
     * forward pass from t=0. Example uses are for streaming data, and for generating samples from network output
     * one step at a time (where samples are then fed back into the network as input)<br>
     * If no previous state is present in RNN layers (i.e., initially or after calling rnnClearPreviousState()),
     * the default initialization (usually 0) is used.<br>
     * Supports mini-batch (i.e., multiple predictions/forward pass in parallel) as well as for single examples.<br>
     *
     * @param inputs Input to network. May be for one or multiple time steps. For single time step:
     *               input has shape [miniBatchSize,inputSize] or [miniBatchSize,inputSize,1]. miniBatchSize=1 for single example.<br>
     *               For multiple time steps: [miniBatchSize,inputSize,inputTimeSeriesLength]
     * @return Output activations. If output is RNN layer (such as RnnOutputLayer): if all inputs have shape [miniBatchSize,inputSize]
     * i.e., is 2d, then outputs have shape [miniBatchSize,outputSize] (i.e., also 2d) instead of [miniBatchSize,outputSize,1].<br>
     * Otherwise output is 3d [miniBatchSize,outputSize,inputTimeSeriesLength] when using RnnOutputLayer (or unmodified otherwise).
     */
    public INDArray[] rnnTimeStep(INDArray... inputs) {
        return rnnTimeStepHelper(null, inputs);
    }

    /**
     * See {@link #rnnTimeStep(INDArray...)} for details.<br>
     * If no memory workspace is provided, the output will be detached (not in any workspace).<br>
     * If a memory workspace is provided, the output activation array (i.e., the INDArray returned by this method)
     * will be placed in the specified workspace. This workspace must be opened by the user before calling this method -
     * and the user is responsible for (a) closing this workspace, and (b) ensuring the output array is not used out
     * of scope (i.e., not used after closing the workspace to which it belongs - as this is likely to cause either
     * an exception when used, or a crash).
     *
     * @param inputs          Input activations
     * @param outputWorkspace Output workspace. May be null
     * @return The output/activations from the network (either detached or in the specified workspace if provided)
     */
    public INDArray[] rnnTimeStep(MemoryWorkspace outputWorkspace, INDArray... inputs){
        try{
            return rnnTimeStepHelper(outputWorkspace, inputs);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    private INDArray[] rnnTimeStepHelper(MemoryWorkspace outputWs, INDArray... inputs){
        boolean inputIs2d = true;
        for (INDArray i : inputs) {
            if (i.rank() != 2) {
                inputIs2d = false;
                break;
            }
        }

        INDArray[] outputs = outputOfLayersDetached(false, FwdPassType.RNN_TIMESTEP, getOutputLayerIndices(), inputs, null, null, true, false, outputWs);

        //As per MultiLayerNetwork.rnnTimeStep(): if inputs are all 2d, then outputs are all 2d
        if (inputIs2d) {
            for (int i = 0; i < outputs.length; i++) {
                if (outputs[i].rank() == 3 && outputs[i].size(2) == 1) {
                    //Return 2d output with shape [miniBatchSize,nOut]
                    // instead of 3d output with shape [miniBatchSize,nOut,1]
                    outputs[i] = outputs[i].tensorAlongDimension(0, 1, 0);
                }
            }
        }

        this.inputs = null;
        return outputs;
    }

    /**
     * Get the state of the RNN layer, as used in {@link #rnnTimeStep(INDArray...)}.
     *
     * @param layer Number/index of the layer.
     * @return Hidden state, or null if layer is not an RNN layer
     */
    public Map<String, INDArray> rnnGetPreviousState(int layer) {
        return rnnGetPreviousState(layers[layer].conf().getLayer().getLayerName());
    }

    /**
     * Get the state of the RNN layer, as used in {@link #rnnTimeStep(INDArray...)}.
     *
     * @param layerName name of the layer
     * @return Hidden state, or null if layer is not an RNN layer
     */
    public Map<String, INDArray> rnnGetPreviousState(String layerName) {
        Layer l = verticesMap.get(layerName).getLayer();
        if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer){
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
        }
        if (l == null || !(l instanceof RecurrentLayer))
            return null;
        return ((RecurrentLayer) l).rnnGetPreviousState();
    }

    /**
     * Get a map of states for ALL RNN layers, as used in {@link #rnnTimeStep(INDArray...)}.
     * Layers that are not RNN layers will not have an entry in the returned map
     *
     * @return Map of states (keyed by layer name) or null if layer is not an RNN layer
     * @see #rnnSetPreviousStates(Map)
     */
    public Map<String, Map<String, INDArray>> rnnGetPreviousStates() {
        Map<String, Map<String, INDArray>> states = new HashMap<>();
        for (Layer l : layers) {
            if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer){
                l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
            }
            if (l instanceof RecurrentLayer) {
                states.put(l.conf().getLayer().getLayerName(), ((RecurrentLayer) l).rnnGetPreviousState());
            }
        }
        return states;
    }

    /**
     * Set the state of the RNN layer, for use in {@link #rnnTimeStep(INDArray...)}
     *
     * @param layer The number/index of the layer.
     * @param state The state to set the specified layer to
     */
    public void rnnSetPreviousState(int layer, Map<String, INDArray> state) {
        rnnSetPreviousState(layers[layer].conf().getLayer().getLayerName(), state);
    }

    /**
     * Set the state of the RNN layer, for use in {@link #rnnTimeStep(INDArray...)}
     *
     * @param layerName The name of the layer.
     * @param state     The state to set the specified layer to
     */
    public void rnnSetPreviousState(String layerName, Map<String, INDArray> state) {
        Layer l = verticesMap.get(layerName).getLayer();
        if(l instanceof org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer){
            l = ((org.deeplearning4j.nn.layers.wrapper.BaseWrapperLayer)l).getUnderlying();
        }
        if (l == null || !(l instanceof RecurrentLayer)) {
            throw new UnsupportedOperationException(
                    "Layer \"" + layerName + "\" is not a recurrent layer. Cannot set state");
        }
        ((RecurrentLayer) l).rnnSetPreviousState(state);
    }

    /**
     * Set the states for all RNN layers, for use in {@link #rnnTimeStep(INDArray...)}
     *
     * @param previousStates The previous time step states for all layers (key: layer name. Value: layer states)
     * @see #rnnGetPreviousStates()
     */
    public void rnnSetPreviousStates(Map<String, Map<String, INDArray>> previousStates) {
        for (Map.Entry<String, Map<String, INDArray>> entry : previousStates.entrySet()) {
            rnnSetPreviousState(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Clear the previous state of the RNN layers (if any), used in {@link #rnnTimeStep(INDArray...)}
     */
    public void rnnClearPreviousState() {
        if (layers == null)
            return;
        for (Layer layer : layers) {
            if (layer instanceof RecurrentLayer)
                ((RecurrentLayer) layer).rnnClearPreviousState();
            else if (layer instanceof MultiLayerNetwork) {
                ((MultiLayerNetwork) layer).rnnClearPreviousState();
            }
        }
    }

    /**
     * Fit the network using truncated BPTT
     */
    protected void doTruncatedBPTT(INDArray[] inputs, INDArray[] labels, INDArray[] featureMasks,
                                   INDArray[] labelMasks, LayerWorkspaceMgr workspaceMgr) {
        if (flattenedGradients == null) {
            initGradientsView();
        }

        //Approach used here to implement truncated BPTT: if input is 3d, split it. Otherwise: input is unmodified
        long timeSeriesLength = -1;
        for (INDArray in : inputs) {
            if (in.rank() != 3)
                continue;
            if (timeSeriesLength == -1)
                timeSeriesLength = in.size(2);
            else if (timeSeriesLength != in.size(2)) {
                log.warn("Cannot do TBPTT with time series of different lengths");
                return;
            }
        }
        for (INDArray out : labels) {
            if (out.rank() != 3)
                continue;
            if (timeSeriesLength == -1)
                timeSeriesLength = out.size(2);
            else if (timeSeriesLength != out.size(2)) {
                log.warn("Cannot do TBPTT with time series of different lengths");
                return;
            }
        }

        long fwdLen = configuration.getTbpttFwdLength();
        long nSubsets = timeSeriesLength / fwdLen;
        if (timeSeriesLength % fwdLen != 0)
            nSubsets++;

        rnnClearPreviousState();

        for (int i = 0; i < nSubsets; i++) {
            long startTimeIdx = i * fwdLen;
            long endTimeIdx = startTimeIdx + fwdLen;
            if (endTimeIdx > timeSeriesLength)
                endTimeIdx = timeSeriesLength;

            if (startTimeIdx > Integer.MAX_VALUE)
                throw new ND4JArraySizeException();
            List<INDArray[]> list = getSubsetsForTbptt((int) startTimeIdx, endTimeIdx, inputs, labels, featureMasks, labelMasks);

            setInputs(list.get(0));
            setLabels(list.get(1));
            setLayerMaskArrays(list.get(2), list.get(3));

            if (solver == null) {
                try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
                    solver = new Solver.Builder().configure(conf()).listeners(getListeners()).model(this)
                            .build();
                }
            }
            solver.optimize(workspaceMgr);

            //Finally, update the state of the RNN layers:
            rnnUpdateStateWithTBPTTState();
        }

        if(clearTbpttState) {
            rnnClearPreviousState();
        }
        clearLayerMaskArrays();
    }

    private List<INDArray[]> getSubsetsForTbptt(int startTimeIdx, long endTimeIdx, INDArray[] inputs, INDArray[] labels,
                                                INDArray[] featureMasks, INDArray[] labelMasks){
        INDArray[] newInputs = new INDArray[inputs.length];
        INDArray[] newLabels = new INDArray[labels.length];
        INDArray[] newFeatureMasks = (featureMasks != null ? new INDArray[featureMasks.length] : null);
        INDArray[] newLabelMasks = (labelMasks != null ? new INDArray[labelMasks.length] : null);

        for (int j = 0; j < inputs.length; j++) {
            if (inputs[j].rank() != 3)
                newInputs[j] = inputs[j];
            else {
                newInputs[j] = inputs[j].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        for (int j = 0; j < labels.length; j++) {
            if (labels[j].rank() != 3)
                newLabels[j] = labels[j];
            else {
                newLabels[j] = labels[j].get(NDArrayIndex.all(), NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        if (featureMasks != null) {
            for (int j = 0; j < featureMasks.length; j++) {
                if (featureMasks[j] == null)
                    continue;
                newFeatureMasks[j] = featureMasks[j].get(NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }
        if (labelMasks != null) {
            for (int j = 0; j < labelMasks.length; j++) {
                if (labelMasks[j] == null)
                    continue;
                newLabelMasks[j] = labelMasks[j].get(NDArrayIndex.all(),
                        NDArrayIndex.interval(startTimeIdx, endTimeIdx));
            }
        }

        return Arrays.asList(newInputs, newLabels, newFeatureMasks, newLabelMasks);
    }

    /**
     * Similar to rnnTimeStep and feedForward() methods. Difference here is that this method:<br>
     * (a) like rnnTimeStep does forward pass using stored state for RNN layers, and<br>
     * (b) unlike rnnTimeStep does not modify the RNN layer state<br>
     * Therefore multiple calls to this method with the same input should have the same output.<br>
     * Typically used during training only. Use rnnTimeStep for prediction/forward pass at test time.
     *
     * @param inputs            Input to network
     * @param training          Whether training or not
     * @param storeLastForTBPTT set to true if used as part of truncated BPTT training
     * @return Activations for each layer (including input, as per feedforward() etc)
     */
    public Map<String, INDArray> rnnActivateUsingStoredState(INDArray[] inputs, boolean training,
                                                             boolean storeLastForTBPTT) {
        return ffToLayerActivationsDetached(training, FwdPassType.RNN_ACTIVATE_WITH_STORED_STATE, storeLastForTBPTT, vertices.length-1,
                null, inputs, inputMaskArrays, labelMaskArrays, true);
    }

    /**
     * Set the mask arrays for features and labels. Mask arrays are typically used in situations such as one-to-many
     * and many-to-one learning with recurrent neural networks, as well as for supporting time series of varying lengths
     * within the same minibatch.<br>
     * For example, with RNN data sets with input of shape [miniBatchSize,nIn,timeSeriesLength] and outputs of shape
     * [miniBatchSize,nOut,timeSeriesLength], the features and mask arrays will have shape [miniBatchSize,timeSeriesLength]
     * and contain values 0 or 1 at each element (to specify whether a given input/example is present - or merely padding -
     * at a given time step).<br>
     * <b>NOTE</b>: This method is not usually used directly. Instead, the various feedForward and fit methods handle setting
     * of masking internally.
     *
     * @param featureMaskArrays Mask array for features (input)
     * @param labelMaskArrays   Mask array for labels (output)
     * @see #clearLayerMaskArrays()
     */
    public void setLayerMaskArrays(INDArray[] featureMaskArrays, INDArray[] labelMaskArrays) {
        this.clearLayerMaskArrays();
        this.inputMaskArrays = featureMaskArrays;
        this.labelMaskArrays = labelMaskArrays;

        if (featureMaskArrays != null) {
            if (featureMaskArrays.length != numInputArrays) {
                throw new IllegalArgumentException("Invalid number of feature mask arrays");
            }

            long minibatchSize = -1;
            for (INDArray i : featureMaskArrays) {
                if (i != null) {
                    minibatchSize = i.size(0);
                }
            }

            //Here: need to do forward pass through the network according to the topological ordering of the network

            Map<Integer, Pair<INDArray, MaskState>> map = new HashMap<>();
            for (int i = 0; i < topologicalOrder.length; i++) {
                GraphVertex current = vertices[topologicalOrder[i]];

                if (current.isInputVertex()) {
                    INDArray fMask = featureMaskArrays[current.getVertexIndex()];
                    map.put(current.getVertexIndex(), new Pair<>(fMask, MaskState.Active));
                } else {
                    VertexIndices[] inputVertices = current.getInputVertices();

                    //Now: work out the mask arrays to feed forward...
                    INDArray[] inputMasks = null; //new INDArray[inputVertices.length];
                    MaskState maskState = null;
                    for (int j = 0; j < inputVertices.length; j++) {
                        Pair<INDArray, MaskState> p = map.get(inputVertices[j].getVertexIndex());
                        if (p != null) {
                            if (inputMasks == null) {
                                inputMasks = new INDArray[inputVertices.length];
                            }
                            inputMasks[j] = p.getFirst();
                            if (maskState == null || maskState == MaskState.Passthrough) {
                                maskState = p.getSecond();
                            }
                        }
                    }

                    if (minibatchSize > Integer.MAX_VALUE)
                        throw new ND4JArraySizeException();
                    Pair<INDArray, MaskState> outPair =
                            current.feedForwardMaskArrays(inputMasks, maskState, (int)minibatchSize);
                    map.put(topologicalOrder[i], outPair);
                }
            }
        }

        if (labelMaskArrays != null) {
            if (labelMaskArrays.length != numOutputArrays) {
                throw new IllegalArgumentException("Invalid number of label mask arrays");
            }
            for (int i = 0; i < labelMaskArrays.length; i++) {
                if (labelMaskArrays[i] == null) {
                    // This output doesn't have a mask, we can skip it.
                    continue;
                }
                String outputName = configuration.getNetworkOutputs().get(i);
                GraphVertex v = verticesMap.get(outputName);
                Layer ol = v.getLayer();
                ol.setMaskArray(labelMaskArrays[i]);
            }
        }
    }

    /**
     * Remove the mask arrays from all layers.<br>
     * See {@link #setLayerMaskArrays(INDArray[], INDArray[])} for details on mask arrays.
     */
    public void clearLayerMaskArrays() {
        for (Layer layer : layers) {
            layer.setMaskArray(null);
        }
        this.inputMaskArrays = null;
        this.labelMaskArrays = null;
    }

    /**
     * Update the internal state of RNN layers after a truncated BPTT fit call
     */
    protected void rnnUpdateStateWithTBPTTState() {
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
     * Evaluate the network (classification performance - single output ComputationGraphs only)
     *
     * @param iterator Iterator to evaluate on
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public <T extends Evaluation> T evaluate(DataSetIterator iterator) {
        return (T)evaluate(iterator, (List<String>)null);
    }

    /**
     * Evaluate the network (classification performance - single output ComputationGraphs only)
     *
     * @param iterator Iterator to evaluate on
     * @return Evaluation object; results of evaluation on all examples in the data set
     */
    public <T extends Evaluation> T  evaluate(MultiDataSetIterator iterator) {
        return evaluate(iterator, (List<String>)null);
    }

    /**
     * Evaluate the network on the provided data set (single output ComputationGraphs only). Used for evaluating
     * the performance of classifiers
     *
     * @param iterator Data to undertake evaluation on
     * @return Evaluation object, summarizing the results of the evaluation on the provided DataSetIterator
     */
    public <T extends Evaluation> T  evaluate(DataSetIterator iterator, List<String> labelsList) {
        return evaluate(iterator, labelsList, 1);
    }

    /**
     * Evaluate the network on the provided data set (single output ComputationGraphs only). Used for evaluating
     * the performance of classifiers
     *
     * @param iterator Data to undertake evaluation on
     * @return Evaluation object, summarizing the results of the evaluation on the provided DataSetIterator
     */
    public <T extends Evaluation> T evaluate(MultiDataSetIterator iterator, List<String> labelsList) {
        return evaluate(iterator, labelsList, 1);
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
    public <T extends Evaluation> T evaluate(DataSetIterator iterator, List<String> labelsList, int topN) {
        if (labelsList == null)
            labelsList = iterator.getLabels();

        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), Evaluation.class);
        }

        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.Evaluation(labelsList, topN))[0];
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
    public <T extends Evaluation> T evaluate(MultiDataSetIterator iterator, List<String> labelsList, int topN) {
        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), Evaluation.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.Evaluation(labelsList, topN))[0];
    }

    /**
     * Evaluate the (single output layer only) network for regression performance
     *
     * @param iterator Data to evaluate on
     * @return Regression evaluation
     */
    public <T extends RegressionEvaluation> T evaluateRegression(DataSetIterator iterator) {
        return evaluateRegression(iterator, null);
    }

    /**
     * Evaluate the (single output layer only) network for regression performance
     *
     * @param iterator Data to evaluate on
     * @return Regression evaluation
     */
    public <T extends RegressionEvaluation> T evaluateRegression(MultiDataSetIterator iterator) {
        return evaluateRegression(iterator, null);
    }

    /**
     * Evaluate the (single output layer only) network for regression performance
     *
     * @param iterator    Data to evaluate on
     * @param columnNames Column names for the regression evaluation. May be null.
     * @return Regression evaluation
     */
    public <T extends RegressionEvaluation> T evaluateRegression(DataSetIterator iterator, List<String> columnNames) {
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.RegressionEvaluation(columnNames))[0];
    }

    /**
     * Evaluate the (single output layer only) network for regression performance
     *
     * @param iterator Data to evaluate on
     * @return Regression evaluation
     */
    public <T extends RegressionEvaluation> T evaluateRegression(MultiDataSetIterator iterator, List<String> columnNames) {
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.RegressionEvaluation(columnNames))[0];
    }


    /**
     * @deprecated To be removed - use {@link #evaluateROC(DataSetIterator, int)} to enforce selection of appropriate ROC/threshold configuration
     */
    @Deprecated
    public <T extends ROC> T evaluateROC(DataSetIterator iterator) {
        return evaluateROC(iterator, 0);
    }

    /**
     * Evaluate the network (must be a binary classifier) on the specified data, using the {@link ROC} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROC}
     * @return ROC evaluation on the given dataset
     */
    public <T extends ROC> T evaluateROC(DataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), ROC.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.ROC(rocThresholdSteps))[0];
    }

    /**
     * @deprecated To be removed - use {@link #evaluateROC(DataSetIterator, int)} to enforce selection of appropriate ROC/threshold configuration
     */
    @Deprecated
    public <T extends ROC> T evaluateROC(MultiDataSetIterator iterator) {
        return evaluateROC(iterator, 0);
    }

    /**
     * Evaluate the network (must be a binary classifier) on the specified data, using the {@link ROC} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROC}
     * @return ROC evaluation on the given dataset
     */
    public <T extends ROC> T evaluateROC(MultiDataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
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
        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), ROCMultiClass.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.ROCMultiClass(rocThresholdSteps))[0];
    }

    /**
     * Evaluate the network on the specified data, using the {@link ROCMultiClass} class
     *
     * @param iterator          Data to evaluate on
     * @param rocThresholdSteps Number of threshold steps to use with {@link ROCMultiClass}
     * @return Multi-class ROC evaluation on the given dataset
     */
    public <T extends ROCMultiClass> T evaluateROCMultiClass(MultiDataSetIterator iterator, int rocThresholdSteps) {
        Layer outputLayer = getOutputLayer(0);
        if(getConfiguration().isValidateOutputLayerConfig()){
            OutputLayerUtil.validateOutputLayerForClassifierEvaluation(outputLayer.conf().getLayer(), ROCMultiClass.class);
        }
        return (T)doEvaluation(iterator, new org.deeplearning4j.eval.ROCMultiClass(rocThresholdSteps))[0];
    }

    /**
     * Perform evaluation on the given data (DataSetIterator) with the given {@link IEvaluation} instance
     *
     * @param iterator   Test data to evaluate on
     * @param evaluations IEvaluation instances
     * @param <T>        Type of the IEvaluation instance
     * @return The input IEvaluation instance, after performing evaluation on the test data
     */
    public <T extends IEvaluation> T[] doEvaluation(DataSetIterator iterator, T... evaluations) {
        return doEvaluation(new MultiDataSetIteratorAdapter(iterator), evaluations);
    }

    /**
     * Perform evaluation on the given data (MultiDataSetIterator) with the given {@link IEvaluation} instance
     *
     * @param iterator    Test data to evaluate on
     * @param evaluations IEvaluation insntance
     * @param <T>         Type of the IEvaluation instance
     * @return The input IEvaluation instance, after performing evaluation on the test data
     */
    public <T extends IEvaluation> T[] doEvaluation(MultiDataSetIterator iterator, T... evaluations) {
        try{
            return doEvaluationHelper(iterator, evaluations);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    /**
     * Perform evaluation for networks with multiple outputs.
     *
     * @param iterator    Data to evaluate
     * @param evaluations Evaluation instances. Key: the network output number (0 to numOutputs-1). Value: the IEvaluation
     *                    instances to perform evaluation with, for that output only. Note that not every output needs to
     *                    have an IEvaluation[] defined.
     * @return The same evaluation map, after performing evaluation
     */
    public <T extends IEvaluation> Map<Integer, T[]> evaluate(DataSetIterator iterator, Map<Integer,T[]> evaluations){
        return evaluate(new MultiDataSetIteratorAdapter(iterator), evaluations);
    }

    /**
     * Perform evaluation for networks with multiple outputs.
     *
     * @param iterator    Data to evaluate
     * @param evaluations Evaluation instances. Key: the network output number (0 to numOutputs-1). Value: the IEvaluation
     *                    instances to perform evaluation with, for that output only. Note that not every output needs to
     *                    have an IEvaluation[] defined.
     * @return The same evaluation map, after performing evaluation
     */
    public <T extends IEvaluation> Map<Integer, T[]> evaluate(MultiDataSetIterator iterator, Map<Integer,T[]> evaluations){
        try{
            return doEvaluationHelper(iterator, evaluations);
        } catch (OutOfMemoryError e){
            CrashReportingUtil.writeMemoryCrashDump(this, e);
            throw e;
        }
    }

    @SuppressWarnings("unchecked")
    @SafeVarargs
    private final <T extends IEvaluation> T[] doEvaluationHelper(MultiDataSetIterator iterator, T... evaluations) {
        Map<Integer,IEvaluation[]> map = Collections.singletonMap(0, (IEvaluation[])evaluations);
        return (T[])doEvaluationHelper(iterator, map).get(0);
    }

    private <T extends IEvaluation> Map<Integer,T[]> doEvaluationHelper(MultiDataSetIterator iterator, Map<Integer, T[]> evaluations){
        if (layers == null || !(getOutputLayer(0) instanceof IOutputLayer)) {
            throw new IllegalStateException("Cannot evaluate network with no output layer");
        }

        WorkspaceUtils.assertNoWorkspacesOpen("Expected no external workspaces open at start of evaluation (doEvaluationHelper)");

        if (iterator.resetSupported() && !iterator.hasNext())
            iterator.reset();

        MultiDataSetIterator iter =
                iterator.asyncSupported() ? new AsyncMultiDataSetIterator(iterator, 2, true) : iterator;

        WorkspaceMode cMode = configuration.getTrainingWorkspaceMode();
        configuration.setTrainingWorkspaceMode(configuration.getInferenceWorkspaceMode());

        boolean useRnnSegments = (configuration.getBackpropType() == BackpropType.TruncatedBPTT);

        MemoryWorkspace outputWs;
        if(getConfiguration().getInferenceWorkspaceMode() == WorkspaceMode.ENABLED){
            outputWs = Nd4j.getWorkspaceManager().getWorkspaceForCurrentThread(WS_ALL_LAYERS_ACT_CONFIG, WS_OUTPUT_MEM);
        } else {
            outputWs = new DummyWorkspace();
        }

        while (iter.hasNext()) {
            MultiDataSet next = iter.next();

            if (next.getFeatures() == null || next.getLabels() == null)
                continue;

            if (!useRnnSegments) {
                //Standard/non-RNN case

                //Assuming single output here
                INDArray[] features = next.getFeatures();
                INDArray[] featuresMasks = next.getFeaturesMaskArrays();
                INDArray[] labels = next.getLabels();
                INDArray[] labelMasks = next.getLabelsMaskArrays();
                List<Serializable> meta = next.getExampleMetaData();

                try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                    INDArray[] out = outputOfLayersDetached(false, FwdPassType.STANDARD, getOutputLayerIndices(), features, featuresMasks, labelMasks, true, false, ws);

                    for (Integer i : evaluations.keySet()) {
                        Preconditions.checkState(i >= 0 && i <labels.length, "Invalid output index: evaluation/output indices must be between 0" +
                                " and numOutputs-1 (%s), got index %s", numOutputArrays, (int)i);
                        IEvaluation[] evalsThisOutput = evaluations.get(i);
                        if (evalsThisOutput == null)
                            continue;

                        Preconditions.checkState(i >= 0 && i < getNumOutputArrays(), "Invalid output index: indices for outputs " +
                                "must be between 0 and %s inclusive - found index %s", numOutputArrays, (int) i);
                        INDArray currOut = out[i];
                        INDArray currLabel = labels[i];

                        try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                            for (IEvaluation evaluation : evalsThisOutput)
                                evaluation.eval(currLabel, currOut, next.getLabelsMaskArray(i), meta);
                        }
                    }
                }


            } else {
                rnnClearPreviousState();

                int fwdLen = configuration.getTbpttFwdLength();
                long tsLength = -1;
                long nF = next.getFeatures().length;
                for (int i = 0; i < nF; i++) {
                    if (next.getFeatures(i).rank() == 3) {
                        tsLength = next.getFeatures(i).size(2);
                    }
                }
                if (tsLength < 0) {
                    throw new IllegalStateException("Invalid configuration: detected TBPTT backprop type without" +
                            " time series features");
                }

                long nSubsets = tsLength / fwdLen;
                if (tsLength % fwdLen != 0)
                    nSubsets++; //Example: 100 fwdLen with timeSeriesLength=120 -> want 2 subsets (1 of size 100, 1 of size 20)
                for (int i = 0; i < nSubsets; i++) {
                    int startTimeIdx = i * fwdLen;
                    long endTimeIdx = Math.min(startTimeIdx + fwdLen, tsLength);

                    List<INDArray[]> subset = getSubsetsForTbptt(startTimeIdx, endTimeIdx, next.getFeatures(),
                            next.getLabels(), next.getFeaturesMaskArrays(), next.getLabelsMaskArrays());
                    setLayerMaskArrays(subset.get(2), subset.get(3));

                    try (MemoryWorkspace ws = outputWs.notifyScopeEntered()) {
                        INDArray[] outSub = rnnTimeStep(ws, subset.get(0));

                        for (Integer idx : evaluations.keySet()) {
                            IEvaluation[] evalsThisOutput = evaluations.get(idx);
                            if (evalsThisOutput == null)
                                continue;

                            INDArray labelSub = (subset.get(1) == null ? null : subset.get(1)[idx]);
                            INDArray maskSub = subset.get(3) == null ? null : subset.get(3)[idx];
                            INDArray currOut = outSub[idx];
                            try (MemoryWorkspace wsO = Nd4j.getWorkspaceManager().scopeOutOfWorkspaces()) {
                                for (IEvaluation evaluation : evalsThisOutput)
                                    evaluation.eval(labelSub, currOut, maskSub);
                            }
                        }
                    }
                }

                rnnClearPreviousState();
            }

            //Clear inputs, masks etc. Important to avoid leaking invalidated/out of scope arrays between iterations
            clearLayersStates();
        }

        if (iterator.asyncSupported())
            ((AsyncMultiDataSetIterator) iter).shutdown();

        configuration.setTrainingWorkspaceMode(cMode);

        return (Map<Integer, T[]>) evaluations;
    }

    /**
     * String detailing the architecture of the computation graph.
     * Vertices are printed in a topological sort order.
     * Columns are Vertex Names with layer/vertex type, nIn, nOut, Total number of parameters and the Shapes of the parameters
     * And the inputs to the vertex
     * Will also give information about frozen layers/vertices, if any.
     *
     * @return Summary as a string
     * @see #memoryInfo(int, InputType...)
     */
    public String summary() {
        return summary((InputType[])null);
    }

    /**
     * String detailing the architecture of the computation graph.
     * Will also display activation size when given an input type.
     * Vertices are printed in a topological sort order.
     * Columns are Vertex Names with layer/vertex type, nIn, nOut, Total number of parameters and the Shapes of the parameters
     * And the inputs to the vertex
     * Will also give information about frozen layers/vertices, if any.
     *
     * @return Summary as a string
     * @see #memoryInfo(int, InputType...)
     */
    public String summary(InputType... inputTypes) {
        StringBuilder ret = new StringBuilder();
        ret.append("\n");

        int frozenParams = 0;
        Map<String, InputType> vertexOutputs = new HashMap<>(); //vertex name and output types
        int currLayerIdx = -1;

        List<String[]> lines = new ArrayList<>();
        if(inputTypes == null){
            lines.add(new String[]{"VertexName (VertexType)", "nIn,nOut", "TotalParams", "ParamsShape", "Vertex Inputs"});
        } else {
            lines.add(new String[]{"VertexName (VertexType)", "nIn,nOut", "TotalParams", "ParamsShape", "Vertex Inputs", "InputShape", "OutputShape"});
        }
        int[] maxLength = new int[inputTypes == null || inputTypes.length == 0 ? 5 : 7];
        String[] header = lines.get(0);
        for( int i=0; i<header.length; i++ ){
            maxLength[i] = header[i].length();
        }

        if(topologicalOrder == null){
            GraphIndices indices = calculateIndices();
            topologicalOrder = indices.getTopologicalSortOrder();
        }

        for (int currVertexIdx : topologicalOrder) {

            GraphVertex currentVertex = vertices[currVertexIdx];
            String currentVertexName = currentVertex.getVertexName();

            //String vars for print
            String[] classNameArr = currentVertex.getClass().toString().split("\\.");
            String className = classNameArr[classNameArr.length - 1];
            String connections = "-";
            String inShape = "-";
            String outShape = "-";
            String paramCount = "-";
            String in = "-";
            String out = "-";
            String paramShape = "-";
            if (currentVertex.isInputVertex()) {
                if (inputTypes != null) vertexOutputs.put(currentVertexName, inputTypes[configuration.getNetworkInputs().indexOf(currentVertexName)]); //for input vertices the outputs are just the input types (only layer vertices have preprocessing?)
            } else {
                connections = configuration.getVertexInputs().get(currentVertexName).toString();
                List<InputType> inputTypeList = new ArrayList<>();
                if (currentVertex.hasLayer()) {
                    Layer currentLayer = ((LayerVertex) currentVertex).getLayer();
                    classNameArr = currentLayer.getClass().getName().split("\\.");
                    className = classNameArr[classNameArr.length - 1];
                    paramCount = String.format("%,d", currentLayer.numParams());
                    //layer with params
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
                        List<String> paraNames = currentLayer.conf().variables();
                        for (String aP : paraNames) {
                            String paramS = ArrayUtils.toString(currentLayer.paramTable().get(aP).shape());
                            paramShape += aP + ":" + paramS + ", ";
                        }
                        paramShape = paramShape.subSequence(0, paramShape.lastIndexOf(",")).toString();
                    }
                    //frozen layer
                    if (currentLayer instanceof FrozenLayer) {
                        frozenParams += currentLayer.numParams();
                        classNameArr = ((FrozenLayer) currentLayer).getInsideLayer().getClass().getName().split("\\.");
                        className = "Frozen " + classNameArr[classNameArr.length - 1];
                    }

                    if (inputTypes != null) {
                        //get input type
                        String inputVertexName = vertices[currentVertex.getInputVertices()[0].getVertexIndex()].getVertexName();
                        InputType currentInType = vertexOutputs.get(inputVertexName);
                        inShape = currentInType.toString();
                        inputTypeList.add(currentInType);

                        InputPreProcessor layerVertexPreProcesor = ((org.deeplearning4j.nn.conf.graph.LayerVertex)configuration.getVertices().get(currentVertexName)).getPreProcessor();
                        if (layerVertexPreProcesor != null) {
                            inShape += "-->" + layerVertexPreProcesor.getOutputType(currentInType);
                        }
                    }
                    currLayerIdx++;
                } else {
                    //get input type
                    if (inputTypes != null) {
                        VertexIndices[] inputVertices = currentVertex.getInputVertices();
                        if (inputVertices != null) {
                            for (int i = 0; i < inputVertices.length; i++) {
                                GraphVertex thisInputVertex = vertices[inputVertices[i].getVertexIndex()];
                                inputTypeList.add(vertexOutputs.get(thisInputVertex.getVertexName()));
                            }
                        }
                    }
                }
                if (inputTypes != null) {
                    InputType currentVertexOutputType = configuration.getVertices().get(currentVertexName).getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]));
                    outShape = currentVertexOutputType.toString();
                    vertexOutputs.put(currentVertexName, currentVertexOutputType);
                }
            }

            //Add on to summary string
            String[] line;
            if (inputTypes == null) {
                line = new String[]{currentVertexName + " (" + className + ")", in + "," + out, paramCount, paramShape, connections};
            } else {
                line = new String[]{currentVertexName + " (" + className + ")", in + "," + out, paramCount, paramShape, connections, inShape, outShape};
            }
            for( int i = 0; i < line.length; i++) {
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

        ret.append(StringUtils.repeat("-", totalLength))
                .append(String.format("\n%30s %,d", "Total Parameters: ", params().length()))
                .append(String.format("\n%30s %,d", "Trainable Parameters: ", params().length() - frozenParams))
                .append(String.format("\n%30s %,d", "Frozen Parameters: ", frozenParams))
                .append("\n")
                .append(StringUtils.repeat("=", totalLength))
                .append("\n");

        return ret.toString();
    }

    /**
     * Generate information regarding memory use for the network, for the given input types and minibatch size.
     * Note that when using workspaces or CuDNN, the network should be trained for some iterations so that the memory
     * workspaces have time to initialize. Without this, the memory requirements during training may be underestimated.
     *
     * Note also that this is the same information that is generated during an OOM crash when training or performing
     * inference.
     *
     * @param minibatch    Minibatch size to estimate memory for
     * @param inputTypes   Input types to the network
     * @return A String with information about network memory use information
     */
    public String memoryInfo(int minibatch, InputType... inputTypes){
        return CrashReportingUtil.generateMemoryStatus(this, minibatch, inputTypes);
    }

    /**
     * This method just makes sure there's no state preserved within layers
     */
    public void clearLayersStates() {
        for (Layer layer : layers) {
            layer.clear();
            layer.clearNoiseWeightParams();
        }

        for (GraphVertex vertex : vertices) {
            vertex.clearVertex();
        }
    }

    /**
     * Increment the epoch count (in the underlying {@link ComputationGraphConfiguration} by 1).
     * Note that this is done <i>automatically</i> when using iterator-based fitting methods, such as
     * {@link #fit(DataSetIterator)} or {@link #fit(MultiDataSet)}. However, when using non-iterator fit methods
     * (DataSet, MultiDataSet, INDArrays etc), the network has no way to know when one epoch ends and another starts.
     * In such situations, this method can be used to increment the epoch counter.<br>
     * Note that the epoch counter is used for situations such as some learning rate schedules, and the like.
     *
     * The current epoch count can be obtained using {@code ComputationGraph.getConfiguration().getEpochCount()}
     */
    public void incrementEpochCount(){
        configuration.setEpochCount(configuration.getEpochCount() + 1);
        synchronizeIterEpochCounts();
    }

    protected void synchronizeIterEpochCounts(){
        //TODO: this is necessrry for some schedules - but the redundant values are a little ugly...
        int currIter = getConfiguration().getIterationCount();
        int currEpoch = getConfiguration().getEpochCount();
        for(Layer l : layers){
            l.setIterationCount(currIter);
            l.setEpochCount(currEpoch);
        }
    }

    /**
     * Returns the number of iterations (parameter updates) that the ComputationGraph has done
     * @return Number of iterations
     */
    public int getIterationCount(){
        return configuration.getIterationCount();
    }

    /**
     * Returns the number of epochs that the ComputationGraph has done.
     * Note that the epoch count is incremented only when {@link #fit(DataSetIterator)}, {@link #fit(MultiDataSetIterator)},
     * {@link #fit(DataSetIterator, int)} or {@link #fit(MultiDataSetIterator, int)} are used.
     * The epoch count can also be manually incremented using {@link #incrementEpochCount()}
     * @return Number of epochs
     */
    public int getEpochCount(){
        return configuration.getEpochCount();
    }

    /**
     * Save the ComputationGraph to a file. Restore using {@link #load(File, boolean)}.
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
     * Save the ComputationGraph to a file. Restore using {@link #load(File, boolean)}.
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
     * Restore a ComputationGraph to a file, saved using {@link #save(File)} or {@link ModelSerializer}
     * @param f File to load the network from
     * @param loadUpdater If true: load the updater if it is available (i.e., the state array for momentum/Adam/rmsprop
     *                   etc) - use <i>false</i> if no further training is required, or <i>true</i> if further training
     *                    will be undertaken
     * @see ModelSerializer ModelSerializer for more details (and saving/loading via streams)
     */
    public static ComputationGraph load(File f, boolean loadUpdater) throws IOException {
        return ModelSerializer.restoreComputationGraph(f, loadUpdater);
    }

    /**
     * Return a copy of the network with the parameters and activations set to use the specified (floating point) data type.
     * If the existing datatype is the same as the requested dataype, the original network will be returned unchanged.
     * Only floating point datatypes (DOUBLE, FLOAT, HALF) may be used.
     *
     * @param dataType Datatype to convert the network to
     * @return The network, set to use the specified datatype for the parameters and activations
     */
    public ComputationGraph convertDataType(@NonNull DataType dataType) {
        Preconditions.checkState(dataType.isFPType(), "Invalid DataType: %s. Can only convert network to a floating point type", dataType);
        if(dataType == params().dataType()){
            return this;
        }

        INDArray newParams = params().castTo(dataType);
        String jsonConfig = getConfiguration().toJson();
        ComputationGraphConfiguration newConf = ComputationGraphConfiguration.fromJson(jsonConfig);
        newConf.setDataType(dataType);
        ComputationGraph newNet = new ComputationGraph(newConf);
        newNet.init(newParams, false);

        Updater u = getUpdater(false);
        if(u != null && u.getStateViewArray() != null){
            INDArray oldUpdaterState = u.getStateViewArray();
            newNet.getUpdater(true).getStateViewArray().assign(oldUpdaterState);
        }
        return newNet;

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
     * @see #setLearningRate(String, double)
     */
    public void setLearningRate(double newLr) {
        NetworkUtils.setLearningRate(this, newLr);
    }

    /**
     * Set the learning rate schedule for all layers in the network to the specified schedule.
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param newLr New learning rate schedule for all layers
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(String, double)
     */
    public void setLearningRate(ISchedule newLr) {
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
     * @param layerName Name of the layer to set the LR for
     * @param newLr     New learning rate for a single layer
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(String, double)
     */
    public void setLearningRate(String layerName, double newLr) {
        NetworkUtils.setLearningRate(this, layerName, newLr);
    }

    /**
     * Set the learning rate schedule for a single layer in the network to the specified value.<br>
     * Note also that {@link #setLearningRate(ISchedule)} should also be used in preference, when all layers need
     * to be set to a new LR schedule.<br>
     * This schedule will replace any/all existing schedules, and also any fixed learning rate values.<br>
     * Note also that the iteration/epoch counts will <i>not</i> be reset. Use {@link ComputationGraphConfiguration#setIterationCount(int)}
     * and {@link ComputationGraphConfiguration#setEpochCount(int)} if this is required
     *
     * @param layerName Name of the layer to set the LR schedule for
     * @param newLr     New learning rate for a single layer
     * @see #setLearningRate(ISchedule)
     * @see #setLearningRate(String, double)
     */
    public void setLearningRate(String layerName, ISchedule newLr) {
        NetworkUtils.setLearningRate(this, layerName, newLr);
    }

    /**
     * Get the current learning rate, for the specified layer, from the network.
     * Note: If the layer has no learning rate (no parameters, or an updater without a learning rate) then null is returned
     * @param layerName   Layer name
     * @return Learning rate for the specified layer, or null
     */
    public Double getLearningRate(String layerName){
        return NetworkUtils.getLearningRate(this, layerName);
    }

    /**
     * Return the layer size (number of units) for the specified layer.
     * Note that the meaning of the "layer size" can depend on the type of layer. For example:<br>
     * - DenseLayer, OutputLayer, recurrent layers: number of units (nOut configuration option)<br>
     * - ConvolutionLayer: the channels (number of channels)<br>
     * - Subsampling layers, global pooling layers, etc: size of 0 is always returned<br>
     *
     * @param layer Index of the layer to get the size of. Must be in range 0 to nLayers-1 inclusive
     * @return Size of the layer
     */
    public long layerSize(int layer) {
        if (layer < 0 || layer > layers.length) {
            throw new IllegalArgumentException("Invalid layer index: " + layer + ". Layer index must be between 0 and "
                    + (layers.length - 1) + " inclusive");
        }
        return layerSize(layers[layer].conf().getLayer().getLayerName());
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
    public long layerInputSize(int layer) {
        if (layer < 0 || layer > layers.length) {
            throw new IllegalArgumentException("Invalid layer index: " + layer + ". Layer index must be between 0 and "
                    + (layers.length - 1) + " inclusive");
        }
        return layerInputSize(layers[layer].conf().getLayer().getLayerName());
    }

    /**
     * Return the layer size (number of units) for the specified layer.<br>
     * Note that the meaning of the "layer size" can depend on the type of layer. For example:<br>
     * - DenseLayer, OutputLayer, recurrent layers: number of units (nOut configuration option)<br>
     * - ConvolutionLayer: the channels (number of channels)<br>
     * - Subsampling layers, global pooling layers, etc: size of 0 is always returned<br>
     *
     * @param layerName Name of the layer to get the size of
     * @return Size of the layer
     */
    public long layerSize(String layerName) {
        Layer l = getLayer(layerName);
        if(l == null){
            throw new IllegalArgumentException("No layer with name \"" + layerName + "\" exists");
        }
        org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
        if (conf == null || !(conf instanceof FeedForwardLayer)) {
            return 0;
        }
        FeedForwardLayer ffl = (FeedForwardLayer) conf;

        return ffl.getNOut();
    }

    /**
     * Return the input size (number of inputs) for the specified layer.<br>
     * Note that the meaning of the "input size" can depend on the type of layer. For example:<br>
     * - DenseLayer, OutputLayer, etc: the feature vector size (nIn configuration option)<br>
     * - Recurrent layers: the feature vector size <i>per time step</i> (nIn configuration option)<br>
     * - ConvolutionLayer: the channels (number of channels)<br>
     * - Subsampling layers, global pooling layers, etc: size of 0 is always returned<br>
     *
     * @param layerName Name of the layer to get the size of
     * @return Size of the layer
     */
    public long layerInputSize(String layerName) {
        Layer l = getLayer(layerName);
        if(l == null){
            throw new IllegalArgumentException("No layer with name \"" + layerName + "\" exists");
        }
        org.deeplearning4j.nn.conf.layers.Layer conf = l.conf().getLayer();
        if (conf == null || !(conf instanceof FeedForwardLayer)) {
            return 0;
        }
        FeedForwardLayer ffl = (FeedForwardLayer) conf;

        return ffl.getNIn();
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
        if (obj instanceof ComputationGraph) {
            ComputationGraph network = (ComputationGraph) obj;
            boolean paramsEquals = network.params().equals(params());
            boolean confEquals = getConfiguration().equals(network.getConfiguration());
            boolean updaterEquals = getUpdater().equals(network.getUpdater());
            return paramsEquals && confEquals && updaterEquals;
        }
        return false;
    }

    private void writeObject(ObjectOutputStream oos) throws IOException {
        ModelSerializer.writeModel(this, oos, true);
    }

    private void readObject(ObjectInputStream ois) throws ClassNotFoundException, IOException {
        val cg = ModelSerializer.restoreComputationGraph(ois, true);

        this.defaultConfiguration = cg.defaultConfiguration.clone();
        this.configuration = cg.configuration.clone();
        this.init();
        this.flattenedParams.assign(cg.flattenedParams);

        if (cg.getUpdater() != null && cg.getUpdater(false).getStateViewArray() != null)
            this.getUpdater(true).getStateViewArray().assign(cg.getUpdater(false).getStateViewArray());
    }

    /**
     * Close the network and deallocate all native memory, including: parameters, gradients, updater memory and workspaces
     * Note that the network should not be used again for any purpose after it has been closed
     */
    @Override
    public void close() {
        //Close the INDArray and dealloc
        if(flattenedParams.closeable())
            flattenedParams.close();

        if(flattenedGradients != null && flattenedGradients.closeable())
            flattenedGradients.close();

        Updater u = getUpdater(false);
        if(u != null && u.getStateViewArray() != null) {
            INDArray state = u.getStateViewArray();
            if(state.closeable())
                state.close();
        }

    }
}
