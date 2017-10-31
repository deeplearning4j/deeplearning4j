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
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.api.OptimizationConfig;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.memory.NetworkMemoryReport;
import org.nd4j.shade.jackson.databind.JsonNode;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/**
 * ComputationGraphConfiguration is a configuration object for neural networks with arbitrary connection structure.
 * It is analogous to {@link MultiLayerConfiguration}, but allows considerably greater flexibility for the network
 * architecture.<br>
 * Specifically, the network architecture is a directed acyclic graph, where each vertex in the graph is a {@link Layer},
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
@Slf4j
public class ComputationGraphConfiguration implements OptimizationConfig, Serializable, Cloneable {

    protected Map<String, Layer> vertices = new LinkedHashMap<>();
    protected Map<String, List<String>> vertexInputs = new LinkedHashMap<>();
    protected Map<String, Integer> vertexIndices;
    protected List<String> topologicalSortOrder;

    @Getter
    @Setter
    protected WorkspaceMode trainingWorkspaceMode;

    @Getter
    @Setter
    protected WorkspaceMode inferenceWorkspaceMode;

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

    //Counter for the number of parameter updates so far
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;

    //New fields, previously on defaultconfiguration, required for optimization:
    protected long seed;
    protected boolean miniBatch = true;
    protected boolean minimize = true;
    protected OptimizationAlgorithm optimizationAlgo;
    protected org.deeplearning4j.nn.conf.stepfunctions.StepFunction stepFunction;
    protected int maxNumLineSearchIterations;

    protected ComputationGraphConfiguration(GraphBuilder builder, NeuralNetConfiguration.Builder globalConfiguration){
        this.backprop = builder.backprop;
        this.pretrain = builder.pretrain;
        this.backpropType = builder.backpropType;
        this.tbpttBackLength = builder.tbpttBackLength;
        this.tbpttFwdLength = builder.tbpttFwdLength;

        this.networkInputs = builder.networkInputs;
        this.networkOutputs = builder.networkOutputs;

        this.vertices = builder.vertices;
        this.vertexInputs = builder.vertexInputs;
        this.trainingWorkspaceMode = globalConfiguration.getGlobalConf().getTrainingWorkspaceMode();
        this.inferenceWorkspaceMode = globalConfiguration.getGlobalConf().getInferenceWorkspaceMode();
        this.cacheMode = globalConfiguration.getGlobalConf().getCacheMode();

        //Add preprocessors that were defined separately to the Layers to which they belong
        //This "set separately" is now deprecated, and preprocessors should be set on the layers directly
        for (Map.Entry<String, InputPreProcessor> entry : builder.inputPreProcessors.entrySet()) {
            org.deeplearning4j.nn.conf.layers.Layer gv = vertices.get(entry.getKey());
            gv.setPreProcessor(entry.getValue());
        }

        this.validate(); //throws exception for invalid configuration

        //Apply global configuration to each of the layers (has to happen before adding preprocessors)
        for(Layer l : this.vertices.values()){
            l.applyGlobalConfiguration(globalConfiguration.globalConf);
        }

        //Automatically add preprocessors, set nIns for CNN->dense transitions, etc
        if (!builder.networkInputTypes.isEmpty()) {
            this.addPreProcessors(builder.networkInputTypes.toArray(new InputType[networkInputs.size()]));
        }

        //And apply global configuration to ComputationGraphConfiguration:
        GlobalConfiguration gc = globalConfiguration.getGlobalConf();
        this.seed = gc.getSeed();
        this.miniBatch = gc.getMiniBatch();
        this.minimize = gc.getMinimize();
        this.optimizationAlgo = gc.getOptimizationAlgo();
        this.stepFunction = gc.getStepFunction();
        this.maxNumLineSearchIterations = gc.getMaxNumLineSearchIterations();

        //Perform topological sort
        this.topologicalSortOrder = topologicalSort();
        getVertexIndices(); //Initialize
    }

    public Map<String,Integer> getVertexIndices(){
        //This design is partly for legacy reasons for import...
        if(vertexIndices != null)
            return vertexIndices;

        vertexIndices = new LinkedHashMap<>();

        //Assignment order:
        //1. Inputs, in the order they are specified
        //2. All other vertices, according to the iteration order of the vertices map
        int i=0;
        for(String s : networkInputs){
            vertexIndices.put(s, i++);
        }
        for(String s : vertices.keySet()){
            vertexIndices.put(s, i++);
        }

        return vertexIndices;
    }


    protected List<String> topologicalSort(){
        //https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
        Map<String, org.deeplearning4j.nn.conf.layers.Layer> nodeMap = getVertices();
        List<String> networkInputNames = getNetworkInputs();
        int numVertices = networkInputNames.size() + nodeMap.size();
        List<String> out = new ArrayList<>(numVertices);

        Map<String, List<String>> vertexInputs = new LinkedHashMap<>();     //Key: vertex name X. Values: all connections (Y -> X)
        Map<String, List<String>> vertexOutputs = new LinkedHashMap<>();    //Key: vertex name X. Values: all connections (X -> Y)

        //Input vertices: no inputs
        for (String s : getNetworkInputs()) {
            vertexInputs.put(s, Collections.<String>emptyList());
        }

        for (Map.Entry<String, org.deeplearning4j.nn.conf.layers.Layer> entry : nodeMap.entrySet()) {
            String thisVertexName = entry.getKey();
            List<String> inputsToThisVertex = getVertexInputs().get(thisVertexName);

            //Normalize the input names, to handle multiple output layers (input could be format like "myLayer/1" etc)
            List<String> inputsToThisVertexNormalized = (inputsToThisVertex == null ? null : new ArrayList<String>(inputsToThisVertex.size()));
            if(inputsToThisVertex != null){
                for(String s : inputsToThisVertex){
                    String normalized = ComputationGraphConfiguration.getLayerNameFromMultiOut(s);
                    if(!inputsToThisVertexNormalized.contains(normalized)){
                        inputsToThisVertexNormalized.add(normalized);
                    }
                }
            }


            if (inputsToThisVertexNormalized == null || inputsToThisVertexNormalized.isEmpty()) {
                //No inputs: skip
                vertexInputs.put(thisVertexName, Collections.<String>emptyList());
                continue;
            }

            vertexInputs.put(thisVertexName, inputsToThisVertexNormalized); //List of connections (Y -> X) where X == thisVertexName

            //Also add list of connections (X -> Y), where X == thisVertex
            for(String s : inputsToThisVertexNormalized){
                List<String> outputSetForInputIdx = vertexOutputs.get(s);
                if (outputSetForInputIdx == null) {
                    outputSetForInputIdx = new ArrayList<>();
                    vertexOutputs.put(s, outputSetForInputIdx);
                }
                outputSetForInputIdx.add(thisVertexName); //input vertex outputs to the current vertex
            }
        }

        //Now: do topological sort
        //Set of all nodes with no incoming edges: (this would be: input vertices)
        LinkedList<String> noIncomingEdges = new LinkedList<>();
        for (Map.Entry<String, List<String>> entry : vertexInputs.entrySet()) {
            List<String> inputsFrom = entry.getValue();
            if (inputsFrom == null || inputsFrom.isEmpty()) {
                noIncomingEdges.add(entry.getKey());
            }
        }

        while (!noIncomingEdges.isEmpty()) {
            String next = noIncomingEdges.removeFirst();
            out.add(next); //Add to sorted list

            List<String> vertexOutputsTo = vertexOutputs.get(next);

            //Remove all edges (next -> Y) from graph
            if (vertexOutputsTo != null) {
                for (String v : vertexOutputsTo) {
                    List<String> list = vertexInputs.get(v);
                    list.remove(next);
                    if (list.isEmpty()) {
                        noIncomingEdges.add(v); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        //If any edges remain in the graph: graph has cycles:
        for (Map.Entry<String, List<String>> entry : vertexInputs.entrySet()) {
            List<String> list = entry.getValue();
            if (list == null)
                continue;
            if (!list.isEmpty())
                throw new IllegalStateException(
                        "Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                                + "cycle includes vertex \"" + vertexInputs.get(entry.getKey())
                                + "\")");
        }

        return out;
    }

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
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        //To maintain backward compatibility after activation function refactoring (configs generated with v0.7.1 or earlier)
        // Previously: enumeration used for activation functions. Now: use classes
        int layerCount = 0;
        Map<String, Layer> vertexMap = conf.getVertices();
        JsonNode vertices = null;
        /*
        //LEGACY CODE FOR ACTIVATION FUNCTIONS
        for (Map.Entry<String, Layer> entry : vertexMap.entrySet()) {
            if (!(entry.getValue() instanceof LayerVertex)) {
                continue;
            }
            LayerVertex lv = (LayerVertex) entry.getValue();
            if (lv.getLayerConf() != null && lv.getLayerConf() != null) {
                Layer layer = lv.getLayerConf();

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
        */

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
        for (Map.Entry<String, Layer> entry : this.vertices.entrySet()) {
            conf.vertices.put(entry.getKey(), entry.getValue().clone());
        }

        conf.vertexInputs = new LinkedHashMap<>();
        for (Map.Entry<String, List<String>> entry : this.vertexInputs.entrySet()) {
            conf.vertexInputs.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }

        conf.vertexIndices = new LinkedHashMap<>();
        for (Map.Entry<String,Integer> entry : this.vertexIndices.entrySet()){
            conf.vertexIndices.put(entry.getKey(), entry.getValue());
        }

        conf.topologicalSortOrder = new ArrayList<>(getTopologicalSortOrder());

        conf.networkInputs = new ArrayList<>();

        conf.networkInputs = new ArrayList<>(this.networkInputs);
        conf.networkOutputs = new ArrayList<>(this.networkOutputs);

        conf.pretrain = pretrain;
        conf.backprop = backprop;
        conf.backpropType = backpropType;
        conf.tbpttFwdLength = tbpttFwdLength;
        conf.tbpttBackLength = tbpttBackLength;
        conf.trainingWorkspaceMode = trainingWorkspaceMode;
        conf.inferenceWorkspaceMode = inferenceWorkspaceMode;
        conf.cacheMode = this.cacheMode;

        conf.seed = seed;
        conf.miniBatch = miniBatch;
        conf.miniBatch = minimize;
        conf.optimizationAlgo = optimizationAlgo;
        conf.stepFunction = stepFunction;
        conf.maxNumLineSearchIterations = maxNumLineSearchIterations;

        //Note: intentionally *don't* include the iteration/epoch counts

        return conf;
    }

    /**
     * Convert the name from a possible multi-output format (like "myLayer/0", which specifies the first (0th) output
     * of the layer "myLayer") to the layer name only
     *
     * @param multiOut Possible multi-output layer name (OK to pass 'standard' layer namas also)
     * @return Layer name, with "/x" prefix removed if necessary
     */
    public static String getLayerNameFromMultiOut(String multiOut){
        if(multiOut.matches(".+/\\d")){
            return multiOut.substring(0, multiOut.length()-2);  //Strip last 2 characters - "/0" etc off
        }
        return multiOut;
    }

    public static int getOutputNumFromMultiOut(String  multiOut){
        if(multiOut.matches(".+/\\d")){
            String num = multiOut.substring(multiOut.length()-1);   //Get last character only
            return Integer.parseInt(num);
        }
        return 0;   //No specifier -> always 0
    }

    /**
     * Check the configuration, make sure it is valid
     *
     * @throws IllegalStateException if configuration is not valid
     */
    public void validate() {
        if (networkInputs == null || networkInputs.size() < 1) {
            throw new IllegalStateException(
                            "Invalid configuration: network has no inputs. Use .addInputs(String...) to label (and give an ordering to) the network inputs");
        }
        if (networkOutputs == null || networkOutputs.size() < 1) {
            throw new IllegalStateException(
                            "Invalid configuration: network has no outputs. Use .setOutput(String...) to specify (and give an ordering to) the output vertices");
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
                String noOutputIdxName = getLayerNameFromMultiOut(inputName);
                if (!vertices.containsKey(noOutputIdxName) && !networkInputs.contains(noOutputIdxName)) {
                    throw new IllegalStateException("Invalid configuration: Vertex \"" + nodeName + "\" has input \""
                                    + inputName + "\" that does not exist");
                }
            }
        }

        //Check output names:
        for (String s : networkOutputs) {
            if (!vertices.containsKey(s)) {
                throw new IllegalStateException(
                                "Invalid configuration: Output name \"" + s + "\" is not a valid vertex");
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

        if (inputTypes == null || inputTypes.length != networkInputs.size()) {
            throw new IllegalArgumentException(
                            "Invalid number of InputTypes: cannot add preprocessors if number of InputType "
                                            + "objects differs from number of network inputs");
        }

        //Now: need to do essentially a forward pass through the network, to work out what type of preprocessors to add
        //To do this: need to know what the output types are for each Layer.

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

            Layer gv = vertices.get(s);

            List<InputType> inputTypeList = new ArrayList<>();

            //Add preprocessor, if necessary:
            int nInputs = vertexInputs.get(s).size();
            InputType[] layerInputs = new InputType[nInputs];
            for( int i=0; i<nInputs; i++ ){
                layerInputs[i] = vertexOutputs.get(vertexInputs.get(s).get(i));
            }

            //Preprocessors - add if necessary
            if (gv.getPreProcessor() == null) {
                //But don't override preprocessors that are manually defined; if none has been defined,
                //add the appropriate preprocessor for this input type/layer combination
                InputPreProcessor preproc = gv.getPreProcessorForInputType(layerInputs);
                if (preproc != null) {
                    gv.setPreProcessor(preproc);
                    gv.setNIn(inputTypes, false); //Don't override the nIn setting, if it's manually set by the user
                } else {
                    gv.setNIn(inputTypes, false); //Don't override the nIn setting, if it's manually set by the user
                }
            }

            currLayerIdx++;

            InputType outputFromVertex =
                            gv.getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]))[0];  //TODO mulitple outputs
            vertexOutputs.put(s, outputFromVertex);
        }
    }

    private Map<String, List<String>> verticesOutputTo() {
        Map<String, List<String>> verticesOutputTo = new HashMap<>(); //Key: vertex. Values: vertices that this node is an input for
        for (Map.Entry<String, Layer> entry : vertices.entrySet()) {
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

            Layer gv = vertices.get(s);

            List<InputType> inputTypeList = new ArrayList<>();
            List<String> inputs = vertexInputs.get(s);
            if (inputs != null) {
                for (String inputVertexName : inputs) {
                    inputTypeList.add(vertexOutputs.get(inputVertexName));
                }
            }

            InputType outputFromVertex =
                            gv.getOutputType(currLayerIdx, inputTypeList.toArray(new InputType[inputTypeList.size()]))[0];  //TODO multiple outputs
            vertexOutputs.put(s, outputFromVertex);

            MemoryReport mr = gv.getMemoryReport(inputTypeList.toArray(new InputType[inputTypeList.size()]));

            memoryReportMap.put(s, mr);
        }

        return new NetworkMemoryReport(memoryReportMap, ComputationGraphConfiguration.class, "ComputationGraph",
                        inputTypes);
    }


    @Data
    public static class GraphBuilder {
        protected Map<String, Layer> vertices = new LinkedHashMap<>();

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

        @Deprecated
        protected Map<String, InputPreProcessor> inputPreProcessors = new LinkedHashMap<>();

        protected NeuralNetConfiguration.Builder globalConfiguration;

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
         * @deprecated Now: set input preprocessors on layers configurations/builders - {@link Layer.Builder#preProcessor(InputPreProcessor)}
         */
        @Deprecated
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
         * Add a layer, with no {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName   Name/label of the layer to add
         * @param layer       The layer configuration
         * @param layerInputs Inputs to this layer (must be 1 or more). Inputs may be other layers/vertices
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
         * @param layerInputs Inputs to this layer (must be 1 or more). Inputs may be other layers/vertices
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
         * @param layerInputs  Inputs to this layer (must be 1 or more). Inputs may be other layers/vertices
         */
        public GraphBuilder addLayer(String layerName, Layer layer, InputPreProcessor preProcessor,
                        String... layerInputs) {
            if(preProcessor != null) {
                layer.setPreProcessor(preProcessor);
            }
            layer.setLayerName(layerName);
            vertices.put(layerName, layer);
            if (layer.maxInputs() == 1 && layerInputs != null && layerInputs.length > 1) {
                String mergeName = layerName + "-merge";
                addLayer(mergeName, new MergeVertex(), layerInputs);
                this.vertexInputs.put(layerName, Collections.singletonList(mergeName));
            } else if (vertexInputs != null) {
                this.vertexInputs.put(layerName, Arrays.asList(layerInputs));
            }
            return this;
        }

        /**
         * Add a layer and an {@link InputPreProcessor}, with the specified name and specified inputs.
         *
         * @param layerName    Name/label of the layer to add
         * @param layer        The layer configuration
         * @param preProcessor The InputPreProcessor to use with this layer.
         * @param layerInputs  Inputs to this layer (must be 1 or more). Inputs may be other layers/vertices
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
         * the calculations on input/ouput sizes (width, height, depth, etc) don't need to be done manually.<br>
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

        public GraphBuilder add(String layerName, Layer layer, String... layerInputs){
            return layer(layerName, layer, layerInputs);
        }

        /**
         * Add a {@link Layer} (formerly a GraphVertex) to the network configuration.
         *
         * @param vertexName   The name of the Layer to add
         * @param vertex       The Layer to add
         * @param vertexInputs The inputs/activations to this Layer
         * @deprecated Use
         */
        @Deprecated
        public GraphBuilder addVertex(String vertexName, Layer vertex, String... vertexInputs){
            return layer(vertexName, vertex, vertexInputs);
        }

        /**
         * Create the ComputationGraphConfiguration from the Builder pattern
         */
        public ComputationGraphConfiguration build() {
            return new ComputationGraphConfiguration(this, globalConfiguration);
        }
    }
}
