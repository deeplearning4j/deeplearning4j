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
package org.deeplearning4j.nn.conf;

import lombok.*;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.LayerVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.BaseLayer;
import org.deeplearning4j.nn.conf.layers.BasePretrainNetwork;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.memory.NetworkMemoryReport;
import org.nd4j.base.Preconditions;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * ComputationGraphConfiguration is a configuration object for neural networks with arbitrary connection structure.
 * It is analogous to {@link MultiLayerConfiguration}, but allows considerably greater flexibility for the network
 * architecture.<br>
 * Specifically, the network architecture is a directed acyclic graph, where each vertex in the graph is a {@link GraphVertex},
 * which may for example be a layer or a vertex/object that defines arbitrary forward and backward pass functionality.<br>
 * Note that the ComputationGraph may have an arbitrary number of inputs (multiple independent inputs, possibly of different
 * types), and an arbitrary number of outputs (for example, multiple {@link OutputLayer} instances.
 * Typical usage:<br>
 * {@code ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()....graphBuilder()...build();}
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class ComputationGraphConfiguration implements Serializable, Cloneable {
    private static Logger log = LoggerFactory.getLogger(ComputationGraphConfiguration.class);

    protected Map<String, GraphVertex> vertices = new LinkedHashMap<>();
    protected Map<String, List<String>> vertexInputs = new LinkedHashMap<>();

    @Getter
    @Setter
    protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected CacheMode cacheMode;

    /**
     * List of inputs to the network, by name
     */
    protected List<String> networkInputs;

    /**
     * List of network outputs, by name
     */
    protected List<String> networkOutputs;

    protected boolean pretrain = false;
    protected boolean backprop = true;
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;

    protected NeuralNetConfiguration defaultConfiguration;

    //Counter for the number of parameter updates so far
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;

    protected int[] topologicalOrder;
    protected List<String> topologicalOrderStr;

    /**
     * @return JSON representation of configuration
     */
    public String toYaml() {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        synchronized (mapper) {
            try {
                return mapper.writeValueAsString(this);
            } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Create a neural net configuration from json
     *
     * @param json the neural net configuration from json
     * @return {@link ComputationGraphConfiguration}
     */
    public static ComputationGraphConfiguration fromYaml(String json) {
        ObjectMapper mapper = NeuralNetConfiguration.mapperYaml();
        try {
            return mapper.readValue(json, ComputationGraphConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @return JSON representation of computation graph configuration
     */
    public String toJson() {
        //As per MultiLayerConfiguration.toJson()
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        synchronized (mapper) {
            //JSON mappers are supposed to be thread safe: however, in practice they seem to miss fields occasionally
            //when writeValueAsString is used by multiple threads. This results in invalid JSON. See issue #3243
            try {
                return mapper.writeValueAsString(this);
            } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
                throw new RuntimeException(e);
            }
        }
    }

    /**
     * Create a computation graph configuration from json
     *
     * @param json the neural net configuration from json
     * @return {@link ComputationGraphConfiguration}
     */
    public static ComputationGraphConfiguration fromJson(String json) {
        //As per MultiLayerConfiguration.fromJson()
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        ComputationGraphConfiguration conf;
        try {
            conf = mapper.readValue(json, ComputationGraphConfiguration.class);
        } catch (Exception e) {
            //Check if this exception came from legacy legacy deserializer...
            String msg = e.getMessage();
            if(msg != null && msg.contains("legacy")){
                throw new RuntimeException("Error deserializing ComputationGraphConfiguration - configuration may have a custom " +
                        "layer, vertex or preprocessor, in pre version 1.0.0-alpha JSON format. These layers can be " +
                        "deserialized by first registering them with NeuralNetConfiguration.registerLegacyCustomClassesForJSON(Class...)", e);
            }
            throw new RuntimeException(e);
        }

        //To maintain backward compatibility after activation function refactoring (configs generated with v0.7.1 or earlier)
        // Previously: enumeration used for activation functions. Now: use classes
        int layerCount = 0;
        Map<String, GraphVertex> vertexMap = conf.getVertices();
        JsonNode vertices = null;
        for (Map.Entry<String, GraphVertex> entry : vertexMap.entrySet()) {
            if (!(entry.getValue() instanceof LayerVertex)) {
                continue;
            }

            LayerVertex lv = (LayerVertex) entry.getValue();
            if (lv.getLayerConf() != null && lv.getLayerConf().getLayer() != null) {
                Layer layer = lv.getLayerConf().getLayer();

                if (layer instanceof BaseLayer && ((BaseLayer) layer).getActivationFn() == null) {
                    String layerName = layer.getLayerName();

                    try {
                        if (vertices == null) {
                            JsonNode jsonNode = mapper.readTree(json);
                            vertices = jsonNode.get("vertices");
                        }

                        JsonNode vertexNode = vertices.get(layerName);
                        JsonNode layerVertexNode = vertexNode.get("LayerVertex");
                        if (layerVertexNode == null || !layerVertexNode.has("layerConf")
                                        || !layerVertexNode.get("layerConf").has("layer")) {
                            continue;
                        }
                        JsonNode layerWrapperNode = layerVertexNode.get("layerConf").get("layer");

                        if (layerWrapperNode == null || layerWrapperNode.size() != 1) {
                            continue;
                        }

                        JsonNode layerNode = layerWrapperNode.elements().next();
                        JsonNode activationFunction = layerNode.get("activationFunction"); //Should only have 1 element: "dense", "output", etc

                        if (activationFunction != null) {
                            IActivation ia = Activation.fromString(activationFunction.asText()).getActivationFunction();
                            ((BaseLayer) layer).setActivationFn(ia);
                        }

                    } catch (IOException e) {
                        log.warn("Layer with null ActivationFn field or pre-0.7.2 activation function detected: could not parse JSON",
                                        e);
                    }
                }
            }
        }

        return conf;
    }

    @Override
    public String toString() {
        return toJson();
    }

    @Override
    public ComputationGraphConfiguration clone() {
        ComputationGraphConfiguration conf = new ComputationGraphConfiguration();

        conf.vertices = new LinkedHashMap<>();
        for (Map.Entry<String, GraphVertex> entry : this.vertices.entrySet()) {
            conf.vertices.put(entry.getKey(), entry.getValue().clone());
        }

        conf.vertexInputs = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : this.vertexInputs.entrySet()) {
            conf.vertexInputs.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }

        conf.networkInputs = new ArrayList<>();

        conf.networkInputs = new ArrayList<>(this.networkInputs);
        conf.networkOutputs = new ArrayList<>(this.networkOutputs);

        conf.pretrain = pretrain;
        conf.backprop = backprop;
        conf.backpropType = backpropType;
        conf.tbpttFwdLength = tbpttFwdLength;
        conf.tbpttBackLength = tbpttBackLength;
        conf.defaultConfiguration = defaultConfiguration.clone();
        conf.trainingWorkspaceMode = trainingWorkspaceMode;
        conf.inferenceWorkspaceMode = inferenceWorkspaceMode;
        conf.cacheMode = this.cacheMode;
        conf.defaultConfiguration.cacheMode = this.cacheMode;

        return conf;
    }


    /**
     * Check the configuration, make sure it is valid
     *
     * @throws IllegalStateException if configuration is not valid
     */
    public void validate() {
        validate(false, false);
    }

    /**
     * Check the configuration, make sure it is valid
     *
     * @param allowDisconnected If true: don't throw an exception on vertices that are 'disconnected'. A disconnected
     *                          vertex is one that is not an output, and doesn't connect to any other vertices. i.e.,
     *                          it's output activations don't go anywhere
     * @throws IllegalStateException if configuration is not valid
     */
    public void validate(boolean allowDisconnected, boolean allowNoOutput){

        if (networkInputs == null || networkInputs.isEmpty()) {
            throw new IllegalStateException( "Invalid configuration: network has no inputs. " +
                    "Use .addInputs(String...) to label (and give an ordering to) the network inputs");
        }
        if ((networkOutputs == null || networkOutputs.isEmpty()) && !allowNoOutput) {
            throw new IllegalStateException("Invalid configuration: network has no outputs." +
                    "Use .setOutput(String...) to specify (and give an ordering to) the output vertices, " +
                    "or use allowNoOutputs(true) to disable this check");
        }

        //Check uniqueness of names for inputs, layers, GraphNodes
        for (String s : networkInputs) {
            if (vertices.containsKey(s)) {
                throw new IllegalStateException("Invalid configuration: name \"" + s
                                + "\" is present in both network inputs and graph vertices/layers");
            }
        }

        //Check: each layer & node has at least one input
        for (Map.Entry<String, List<String>> e : vertexInputs.entrySet()) {
            String nodeName = e.getKey();
            if (e.getValue() == null || e.getValue().isEmpty()) {
                throw new IllegalStateException("Invalid configuration: vertex \"" + nodeName + "\" has no inputs");
            }
            for (String inputName : e.getValue()) {
                if (!vertices.containsKey(inputName) && !networkInputs.contains(inputName)) {
                    throw new IllegalStateException("Invalid configuration: Vertex \"" + nodeName + "\" has input \""
                                    + inputName + "\" that does not exist");
                }
            }
        }

        //Check output names:
        if(networkOutputs != null) {
            for (String s : networkOutputs) {
                if (!vertices.containsKey(s)) {
                    throw new IllegalStateException(
                            "Invalid configuration: Output name \"" + s + "\" is not a valid vertex");
                }
            }
        }

        //Check that there aren't any disconnected vertices
        if(!allowDisconnected){
            //A vertex is considered disconnected if it is (a) not an output vertex, and (b) isn't used an as input
            // to another layer

            Set<String> seenAsInput = new HashSet<>();
            seenAsInput.addAll(networkOutputs);
            for(Map.Entry<String,List<String>> e : vertexInputs.entrySet()){
                seenAsInput.addAll(e.getValue());
            }

            Set<String> disconnected = new HashSet<>();
            disconnected.addAll(networkInputs);
            disconnected.addAll(vertices.keySet());
            disconnected.removeAll(seenAsInput);
            if(!disconnected.isEmpty() && !allowNoOutput){  //If allowing no output: by definition we have disconnected vertices
                throw new IllegalStateException("Invalid configuration: disconnected vertices found - " + disconnected
                        + ". Disconnected vertices are those that do not connect to either another vertex, and are also"
                        + " not a network output. To disable this error (i.e., allow network configurations with" +
                        " disconnected vertices) use GraphBuilder.allowDisconnected(true)");
            }
        }

        //Check for no graph cycles: done in ComputationGraph.init()
    }

    /**
     * Add preprocessors automatically, given the specified types of inputs for the network. Inputs are specified using the
     * {@link InputType} class, in the same order in which the inputs were defined in the original configuration.<br>
     * For example, in a network with two inputs: a convolutional input (28x28x1 images) and feed forward inputs, use
     * {@code .addPreProcessors(InputType.convolutional(1,28,28),InputType.feedForward())}.<br>
     * For the CNN->Dense and CNN->RNN transitions, the nIns on the Dense/RNN layers will also be added automatically.
     * <b>NOTE</b>: This method will be called automatically when using the
     * {@link GraphBuilder#setInputTypes(InputType...)} functionality.
     * See that method for details.
     */
    public void addPreProcessors(InputType... inputTypes) {
        getLayerActivationTypes(true, inputTypes);
    }

    /**
     * For the given input shape/type for the network, return a map of activation sizes for each layer and vertex
     * in the graph. Note that this method will automatically add preprocessors if required, to handle (for example)
     * the transition between CNN and dense layers.
     * @param inputTypes                Input types for the network
     * @return A map of activation types for the graph (key: vertex name. value: type of activations out of that vertex)
     */
    public Map<String,InputType> getLayerActivationTypes(InputType... inputTypes){
        return getLayerActivationTypes(true, inputTypes);
    }

    /**
     * For the given input shape/type for the network, return a map of activation sizes for each layer and vertex
     * in the graph. Note that this method can also add preprocessors if required (to handle transitions between some
     * layer types such as convolutional -> dense, for example)
     * @param addPreprocIfNecessary     If true: add any required preprocessors, in the process of calculating the layer
     *                                  activation sizes
     * @param inputTypes                Input types for the network
     * @return A map of activation types for the graph (key: vertex name. value: type of activations out of that vertex)
     */
    public Map<String,InputType> getLayerActivationTypes(boolean addPreprocIfNecessary, InputType... inputTypes){

        if (inputTypes == null || inputTypes.length != networkInputs.size()) {
            throw new IllegalArgumentException(
                    "Invalid number of InputTypes: cannot add preprocessors if number of InputType "
                            + "objects differs from number of network inputs");
        }

        //Now: need to do essentially a forward pass through the network, to work out what type of preprocessors to add
        //To do this: need to know what the output types are for each GraphVertex.

        //Do topological sort
        List<String> topologicalOrdering = topologicalOrdering();

        //Now, given the topological sort: do equivalent of forward pass
        Map<String, InputType> vertexOutputs = new HashMap<>();
        int currLayerIdx = -1;
        for (String s : topologicalOrdering) {
            int inputIdx = networkInputs.indexOf(s);
            if (inputIdx != -1) {
                vertexOutputs.put(s, inputTypes[inputIdx]);
                continue;
            }

            GraphVertex gv = vertices.get(s);

            List<InputType> inputTypeList = new ArrayList<>();

            if (gv instanceof LayerVertex) {
                //Add preprocessor, if necessary:
                String in = vertexInputs.get(s).get(0);
                InputType layerInput = vertexOutputs.get(in);
                inputTypeList.add(layerInput);

                LayerVertex lv = (LayerVertex) gv;
                Layer l = lv.getLayerConf().getLayer();

                //Preprocessors - add if necessary
                if (lv.getPreProcessor() == null && addPreprocIfNecessary) {
                    //But don't override preprocessors that are manually defined; if none has been defined,
                    //add the appropriate preprocessor for this input type/layer combination
                    InputPreProcessor preproc = l.getPreProcessorForInputType(layerInput);
                    lv.setPreProcessor(preproc);
                }

                //Set nIn value for layer (if not already set)
                InputType afterPreproc = layerInput;
                if (lv.getPreProcessor() != null) {
                    InputPreProcessor ip = lv.getPreProcessor();
                    afterPreproc = ip.getOutputType(layerInput);
                }
                l.setNIn(afterPreproc, false);

                currLayerIdx++;
            } else {
                List<String> inputs = vertexInputs.get(s);
                if (inputs != null) {
                    for (String inputVertexName : inputs) {
                        inputTypeList.add(vertexOutputs.get(inputVertexName));
                    }
                }
            }

            InputType outputFromVertex =
                    gv.getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]));
            vertexOutputs.put(s, outputFromVertex);
        }

        return vertexOutputs;
    }

    private Map<String, List<String>> verticesOutputTo() {
        Map<String, List<String>> verticesOutputTo = new HashMap<>(); //Key: vertex. Values: vertices that this node is an input for
        for (Map.Entry<String, GraphVertex> entry : vertices.entrySet()) {
            String vertexName = entry.getKey();
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

        return verticesOutputTo;
    }

    private List<String> topologicalOrdering() {
        //First step: build network in reverse order (i.e., define map of a -> list(b) instead of list(a) -> b)
        Map<String, List<String>> verticesOutputTo = verticesOutputTo();
        LinkedList<String> noIncomingEdges = new LinkedList<>(networkInputs); //Set of all nodes with no incoming edges
        List<String> topologicalOrdering = new ArrayList<>();

        Map<String, Set<String>> inputEdges = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : vertexInputs.entrySet()) {
            inputEdges.put(entry.getKey(), new HashSet<>(entry.getValue()));
        }

        while (!noIncomingEdges.isEmpty()) {
            String next = noIncomingEdges.removeFirst();
            topologicalOrdering.add(next);

            //Remove edges next -> vertexOuputsTo[...] from graph;
            List<String> nextEdges = verticesOutputTo.get(next);

            if (nextEdges != null && !nextEdges.isEmpty()) {
                for (String s : nextEdges) {
                    Set<String> set = inputEdges.get(s);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncomingEdges.add(s); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        //If any edges remain in the graph: graph has cycles:
        for (Map.Entry<String, Set<String>> entry : inputEdges.entrySet()) {
            Set<String> set = entry.getValue();
            if (set == null)
                continue;
            if (!set.isEmpty())
                throw new IllegalStateException(
                                "Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                                                + "cycle includes vertex \"" + entry.getKey() + "\")");
        }

        return topologicalOrdering;
    }

    /**
     * Get a {@link MemoryReport} for the given computation graph configuration. This is used to estimate the
     * memory requirements for the given network configuration and input
     *
     * @param inputTypes Input types for the network
     * @return Memory report for the network
     */
    public NetworkMemoryReport getMemoryReport(InputType... inputTypes) {


        Map<String, MemoryReport> memoryReportMap = new LinkedHashMap<>();
        List<String> topologicalOrdering = topologicalOrdering();

        Map<String, InputType> vertexOutputs = new HashMap<>();
        int currLayerIdx = -1;
        for (String s : topologicalOrdering) {
            int inputIdx = networkInputs.indexOf(s);
            if (inputIdx != -1) {
                vertexOutputs.put(s, inputTypes[inputIdx]);
                continue;
            }

            GraphVertex gv = vertices.get(s);

            List<InputType> inputTypeList = new ArrayList<>();

            if (gv instanceof LayerVertex) {
                //Add preprocessor, if necessary:
                String in = vertexInputs.get(s).get(0);
                InputType layerInput = vertexOutputs.get(in);
                inputTypeList.add(layerInput);
                currLayerIdx++;
            } else {
                List<String> inputs = vertexInputs.get(s);
                if (inputs != null) {
                    for (String inputVertexName : inputs) {
                        inputTypeList.add(vertexOutputs.get(inputVertexName));
                    }
                }
            }



            InputType outputFromVertex =
                            gv.getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]));
            vertexOutputs.put(s, outputFromVertex);

            MemoryReport mr = gv.getMemoryReport(inputTypeList.toArray(new InputType[inputTypeList.size()]));

            memoryReportMap.put(s, mr);
        }

        return new NetworkMemoryReport(memoryReportMap, ComputationGraphConfiguration.class, "ComputationGraph",
                        inputTypes);
    }

    @Data
    public static class GraphBuilder {
        protected Map<String, GraphVertex> vertices = new LinkedHashMap<>();

        /**
         * Key: graph node. Values: input to that node
         */
        protected Map<String, List<String>> vertexInputs = new LinkedHashMap<>();

        protected List<String> networkInputs = new ArrayList<>();
        protected List<InputType> networkInputTypes = new ArrayList<>();
        protected List<String> networkOutputs = new ArrayList<>();

        protected boolean pretrain = false;
        protected boolean backprop = true;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;

        protected Map<String, InputPreProcessor> inputPreProcessors = new LinkedHashMap<>();

        protected NeuralNetConfiguration.Builder globalConfiguration;

        protected boolean allowDisconnected = false;
        protected boolean allowNoOutput = false;

        public GraphBuilder(NeuralNetConfiguration.Builder globalConfiguration) {
            this.globalConfiguration = globalConfiguration;
        }

        public GraphBuilder(ComputationGraphConfiguration newConf, NeuralNetConfiguration.Builder globalConfiguration) {

            ComputationGraphConfiguration clonedConf = newConf.clone();

            this.vertices = clonedConf.getVertices();
            this.vertexInputs = clonedConf.getVertexInputs();

            this.networkInputs = clonedConf.getNetworkInputs();
            this.networkOutputs = clonedConf.getNetworkOutputs();

            this.pretrain = clonedConf.isPretrain();
            this.backprop = clonedConf.isBackprop();
            this.backpropType = clonedConf.getBackpropType();
            this.tbpttFwdLength = clonedConf.getTbpttFwdLength();
            this.tbpttBackLength = clonedConf.getTbpttBackLength();
            this.globalConfiguration = globalConfiguration;
            //this.getGlobalConfiguration().setSeed(clonedConf.getDefaultConfiguration().getSeed());
        }

        /**
         * Specify the processors for a given layer
         * These are used at each layer for doing things like normalization and shaping of input.<br>
         * <b>Note</b>: preprocessors can also be defined using the {@link #addLayer(String, Layer, InputPreProcessor, String...)} method.
         *
         * @param layer     the name of the layer that this preprocessor will be used with
         * @param processor the preprocessor to use for the specified layer
         */
        public GraphBuilder inputPreProcessor(String layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer, processor);
            return this;
        }

        /**
         * Whether to do back prop (standard supervised learning) or not
         *
         * @param backprop whether to do back prop or not
         */
        public GraphBuilder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        /**
         * Whether to do layerwise pre training or not
         *
         * @param pretrain whether to do pre train or not
         */
        public GraphBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        /**
         * The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         *
         * @param type Type of backprop. Default: BackpropType.Standard
         */
        public GraphBuilder backpropType(BackpropType type) {
            this.backpropType = type;
            return this;
        }

        /**
         * When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         *
         * @param forwardLength Forward length > 0, >= backwardLength
         */
        public GraphBuilder tBPTTForwardLength(int forwardLength) {
            this.tbpttFwdLength = forwardLength;
            return this;
        }

        /**
         * When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         *
         * @param backwardLength <= forwardLength
         */
        public GraphBuilder tBPTTBackwardLength(int backwardLength) {
            this.tbpttBackLength = backwardLength;
            return this;
        }

        /**
         * When doing truncated backpropagation through time (tBPTT): how many steps should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * See: http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         *
         * @param tbpttLength length > 0
         */
        public GraphBuilder tBPTTLength(int tbpttLength){
            tBPTTForwardLength(tbpttLength);
            return tBPTTBackwardLength(tbpttLength);
        }

        /**
         * Add a layer, with no {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName   Name/label of the layer to add
         * @param layer       The layer configuration
         * @param layerInputs Inputs to this layer (must be 1 or more). Inputs may be other layers, GraphVertex objects,
         *                    on a combination of the two.
         * @see #addLayer(String, Layer, InputPreProcessor, String...)
         */
        public GraphBuilder addLayer(String layerName, Layer layer, String... layerInputs) {
            return addLayer(layerName, layer, null, layerInputs);
        }

        /**
         * Add a layer, with no {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName   Name/label of the layer to add
         * @param layer       The layer configuration
         * @param layerInputs Inputs to this layer (must be 1 or more). Inputs may be other layers, GraphVertex objects,
         *                    on a combination of the two.
         * @see #addLayer(String, Layer, InputPreProcessor, String...)
         */
        public GraphBuilder layer(String layerName, Layer layer, String... layerInputs) {
            return addLayer(layerName, layer, null, layerInputs);
        }

        /**
         * Add a layer and an {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName    Name/label of the layer to add
         * @param layer        The layer configuration
         * @param preProcessor The InputPreProcessor to use with this layer.
         * @param layerInputs  Inputs to this layer (must be 1 or more). Inputs may be other layers, GraphVertex objects,
         *                     on a combination of the two.
         */
        public GraphBuilder addLayer(String layerName, Layer layer, InputPreProcessor preProcessor,
                        String... layerInputs) {
            NeuralNetConfiguration.Builder builder = globalConfiguration.clone();
            builder.layer(layer);
            addVertex(layerName, new LayerVertex(builder.build(), preProcessor), layerInputs);
            layer.setLayerName(layerName);
            return this;
        }

        /**
         * Add a layer and an {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName    Name/label of the layer to add
         * @param layer        The layer configuration
         * @param preProcessor The InputPreProcessor to use with this layer.
         * @param layerInputs  Inputs to this layer (must be 1 or more). Inputs may be other layers, GraphVertex objects,
         *                     on a combination of the two.
         */
        public GraphBuilder layer(String layerName, Layer layer, InputPreProcessor preProcessor,
                                     String... layerInputs) {
            return addLayer(layerName, layer, preProcessor, layerInputs);
        }

        /**
         * Intended for use with the transfer learning API. Users discouraged from employing it directly.
         * Removes the specified vertex from the vertices list, it's connections and associated preprocessor
         * If the vertex removed is an output vertex it will also be removed from the list of outputs
         * @param vertexName Name of the vertex to remove
         */
        public GraphBuilder removeVertex(String vertexName) {
            removeVertex(vertexName, true);
            return this;
        }

        /**
         * Intended for use with the transfer learning API. Users discouraged from employing it directly.
         * Removes the specified vertex from the vertices list,
         * Removes it's connections (associated preprocessor and if an output also removes it from list of outputs) if "removeConnections" is specified as true
         * Specifying as false can leave the graph in an invalid state with references to vertices that donot exist unless a new vertex is added back in with the same name
         * @param removeConnections Specify true to remove connections
         * @param vertexName Name of the vertex to remove
         */
        public GraphBuilder removeVertex(String vertexName, boolean removeConnections) {
            vertices.remove(vertexName);
            vertexInputs.remove(vertexName);
            if (networkInputs.contains(vertexName)) {
                networkInputs.remove(vertexName);
            }
            if (removeConnections) {
                if (networkOutputs.contains(vertexName)) {
                    networkOutputs.remove(vertexName);
                }
                for (Map.Entry<String, List<String>> entry : this.vertexInputs.entrySet()) {
                    List inputs = entry.getValue();
                    if (inputs.contains(vertexName)) {
                        inputs.remove(vertexName);
                    }
                }
                if (inputPreProcessors.containsKey(vertexName)) {
                    inputPreProcessors.remove(vertexName);
                }
            }
            return this;
        }

        /**
         * Specify the inputs to the network, and their associated labels.
         *
         * @param inputNames The names of the inputs. This also defines their order
         */
        public GraphBuilder addInputs(String... inputNames) {
            Collections.addAll(networkInputs, inputNames);
            return this;
        }

        /**
         * Specify the inputs to the network, and their associated labels.
         *
         * @param inputNames The names of the inputs. This also defines their order
         */
        public GraphBuilder addInputs(Collection<String> inputNames) {
            networkInputs.addAll(inputNames);
            return this;
        }

        /**Specify the types of inputs to the network, so that:<br>
         * (a) preprocessors can be automatically added, and<br>
         * (b) the nIns (input size) for each layer can be automatically calculated and set<br>
         * The order here is the same order as .addInputs(). Thus, if you do .addInputs("a","b") and .setInputTypes(InputType.feedForward(),
         * InputType.convolutional(1,28,28)) then the input labelled "a" is a feed forward input, whereas the input labelled "b" in a CNN
         * input, with 28x28x1 images as input.<br>
         * <b>Note</b>: Using setInputTypes is not always necessary, but can be especially helpful for example with CNNs such that
         * the calculations on input/ouput sizes (width, height, channels, etc) don't need to be done manually.<br>
         * <b>Note 2</b>: If a preprocessor is manually added for a given layer, it will not be overridden by the automatic
         * addition of preprocessors.
         * <b>Note 3</b>: If a layer has an nIn set manually, this will not be overridden
         */
        public GraphBuilder setInputTypes(InputType... inputTypes) {
            if (inputTypes != null && inputTypes.length > 0)
                Collections.addAll(networkInputTypes, inputTypes);
            return this;
        }


        /**
         * Set the network output labels. These should be the names of the OutputLayer instances in the network
         *
         * @param outputNames The names of the output layers. This also defines their order.
         */
        public GraphBuilder setOutputs(String... outputNames) {
            networkOutputs.clear();
            Collections.addAll(networkOutputs, outputNames);

            return this;
        }

        /**
         * Add a {@link GraphVertex} to the network configuration. A GraphVertex defines forward and backward pass methods,
         * and can contain a {@link LayerVertex}, a {@link org.deeplearning4j.nn.conf.graph.ElementWiseVertex} to do element-wise
         * addition/subtraction, a {@link MergeVertex} to combine/concatenate the activations out of multiple layers or vertices,
         * a {@link org.deeplearning4j.nn.conf.graph.SubsetVertex} to select a subset of the activations out of another layer/GraphVertex.<br>
         * Custom GraphVertex objects (that extend the abstract {@link GraphVertex} class) may also be used.
         *
         * @param vertexName   The name of the GraphVertex to add
         * @param vertex       The GraphVertex to add
         * @param vertexInputs The inputs/activations to this GraphVertex
         */
        public GraphBuilder addVertex(String vertexName, GraphVertex vertex, String... vertexInputs) {
            vertices.put(vertexName, vertex);

            //Automatically insert a MergeNode if this vertex can only take 1 input (layer vertices, etc)
            if (vertex.maxVertexInputs() == 1 && vertexInputs != null && vertexInputs.length > 1) {
                String mergeName = vertexName + "-merge";
                addVertex(mergeName, new MergeVertex(), vertexInputs);
                this.vertexInputs.put(vertexName, Collections.singletonList(mergeName));
            } else if (vertexInputs != null) {
                this.vertexInputs.put(vertexName, Arrays.asList(vertexInputs));
            }
            return this;
        }

        /**
         * Used only during validation after building.<br>
         * If true: don't throw an exception on configurations containing vertices that are 'disconnected'. A disconnected
         * vertex is one that is not an output, and doesn't connect to any other vertices. i.e., it's output activations
         * don't go anywhere. Most users can (and should) leave this as the default value of false.
         *
         * @param allowDisconnected Whether to allow disconnected vertices, during validation
         */
        public GraphBuilder allowDisconnected(boolean allowDisconnected){
            this.allowDisconnected = allowDisconnected;
            return this;
        }

        /**
         * Used only during validation after building.<br>
         * If true: don't throw an exception on configurations without any outputs. This is enabled by default
         * to avoid creating invalid graphs, but can be disabled if required.<br>
         * Most users can (and should) leave this as the default value of false.
         *
         * @param allowNoOutput Whether to allow no outputs, during validation
         */
        public GraphBuilder allowNoOutput(boolean allowNoOutput){
            this.allowNoOutput = allowNoOutput;
            return this;
        }

        /**
         * For the (perhaps partially constructed) network configuration, return a map of activation sizes for each
         * layer and vertex in the graph.<br>
         * Note 1: The network configuration may be incomplete, but the inputs have been added to the layer already.<br>
         * Note 2: To use this method, the network input types must have been set using {@link #setInputTypes(InputType...)}
         * first
         * @return A map of activation types for the graph (key: vertex name. value: type of activations out of that vertex)
         */
        public Map<String,InputType> getLayerActivationTypes(){
            Preconditions.checkArgument(networkInputs != null && networkInputs.size() > 0,
                    "Cannot calculate activation types if no inputs have been set (use addInputs(String...))");
            Preconditions.checkArgument(networkInputTypes != null && networkInputTypes.size() == networkInputs.size(),
                    "Cannot calculate layer activation types if network if network input types have not" +
                            "been set (use ");

            //Instantiate temporary ComputationGraphConfiguration and calculate output shapes
            ComputationGraphConfiguration conf;
            try{
                conf = buildConfig();
            } catch (Exception e){
                throw new RuntimeException("Error calculating activation types for layers: error occured when constructing " +
                        "temporary ComputationGraphConfiguration)", e);
            }

            try{
                conf.validate(true, true);
            } catch (Exception e){
                throw new RuntimeException("Error calculating activation types for layers: validation of temporary" +
                        " ComputationGraphConfiguration failed", e);
            }

            return conf.getLayerActivationTypes(true, networkInputTypes.toArray(new InputType[networkInputTypes.size()]));
        }


        private ComputationGraphConfiguration buildConfig(){
            ComputationGraphConfiguration conf = new ComputationGraphConfiguration();
            conf.backprop = backprop;
            conf.pretrain = pretrain;
            conf.backpropType = backpropType;
            conf.tbpttBackLength = tbpttBackLength;
            conf.tbpttFwdLength = tbpttFwdLength;

            conf.networkInputs = networkInputs;
            conf.networkOutputs = networkOutputs;

            conf.vertices = this.vertices;
            conf.vertexInputs = this.vertexInputs;
            conf.trainingWorkspaceMode = globalConfiguration.trainingWorkspaceMode;
            conf.inferenceWorkspaceMode = globalConfiguration.inferenceWorkspaceMode;
            conf.cacheMode = globalConfiguration.cacheMode;

            conf.defaultConfiguration = globalConfiguration.build();
            conf.getDefaultConfiguration().setPretrain(pretrain);

            //Add preprocessors that were defined separately to the Layers to which they belong
            for (Map.Entry<String, InputPreProcessor> entry : inputPreProcessors.entrySet()) {
                GraphVertex gv = vertices.get(entry.getKey());
                if (gv instanceof LayerVertex) {
                    LayerVertex lv = (LayerVertex) gv;
                    lv.setPreProcessor(entry.getValue());
                } else {
                    throw new IllegalStateException(
                            "Invalid configuration: InputPreProcessor defined for GraphVertex \""
                                    + entry.getKey() + "\", but this vertex is not a LayerVertex");
                }

            }

            for (Map.Entry<String, GraphVertex> gv : vertices.entrySet()) {
                if (gv.getValue() instanceof LayerVertex) {
                    LayerVertex lv = (LayerVertex) gv.getValue();
                    Layer l = lv.getLayerConf().getLayer();
                    if (l instanceof BasePretrainNetwork)
                        lv.getLayerConf().setPretrain(pretrain);
                }

            }

            return conf;
        }


        /**
         * Create the ComputationGraphConfiguration from the Builder pattern
         */
        public ComputationGraphConfiguration build() {

            ComputationGraphConfiguration conf = buildConfig();
            conf.validate(allowDisconnected, allowNoOutput); //throws exception for invalid configuration

            //Automatically add preprocessors, set nIns for CNN->dense transitions, etc
            if (!networkInputTypes.isEmpty()) {
                conf.addPreProcessors(networkInputTypes.toArray(new InputType[networkInputs.size()]));
            }

            return conf;
        }
    }
}
