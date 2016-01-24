/*
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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.introspect.AnnotatedClass;
import com.fasterxml.jackson.databind.jsontype.NamedType;
import lombok.*;
import org.apache.commons.lang3.ClassUtils;
import org.deeplearning4j.nn.conf.graph.GraphVertex;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.layers.Layer;
import org.nd4j.linalg.factory.Nd4j;
import org.reflections.Reflections;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

/** ComputationGraphConfiguration is a configuration object for neural networks with arbitrary connection structure.
 * It is analogous to {@link MultiLayerConfiguration}, but allows considerably greater flexibility for the network
 * architecture
 */
@Data @EqualsAndHashCode
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
public class ComputationGraphConfiguration implements Serializable, Cloneable {


    /** Map between layer numbers, and layer names */
    protected Map<Integer,String> layerNamesMap;
    protected Map<String,Integer> layerNumbersMap;

    protected Map<String,NeuralNetConfiguration> layers = new HashMap<>();
    protected Map<String,GraphVertex> graphNodes = new HashMap<>();

    protected Map<String,List<String>> layerInputs = new HashMap<>();
    protected Map<String,List<String>> graphNodeInputs = new HashMap<>();

    protected List<String> networkInputs;
    protected List<String> networkOutputs;


    protected boolean pretrain = true;
    protected boolean backprop = false;
    protected Map<String,InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;
    //whether to redistribute params or not
    protected boolean redistributeParams = false;


    /**
     * @return  JSON representation of configuration
     */
    public String toYaml() {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * Create a neural net configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.ComputationGraphConfiguration}
     */
    public static ComputationGraphConfiguration fromYaml(String json) {
        throw new UnsupportedOperationException("Not implemented");
    }

    /**
     * @return  JSON representation of computation graph configuration
     */
    public String toJson() {
        //As per MultiLayerConfiguration.toJson()
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.writeValueAsString(this);
        } catch (com.fasterxml.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Create a computation graph configuration from json
     * @param json the neural net configuration from json
     * @return {@link org.deeplearning4j.nn.conf.MultiLayerConfiguration}
     */
    public static ComputationGraphConfiguration fromJson(String json) {
        //As per MultiLayerConfiguration.fromJson()
        ObjectMapper mapper = NeuralNetConfiguration.mapper();
        try {
            return mapper.readValue(json, ComputationGraphConfiguration.class);
        } catch (IOException e) {
            //No op - try again after adding new subtypes
        }

        //Try: programmatically registering JSON subtypes for GraphVertex classes. This allows users to to add custom GraphVertex
        // implementations without needing to manually register subtypes
            //First: get all registered subtypes
        AnnotatedClass ac = AnnotatedClass.construct(GraphVertex.class,mapper.getSerializationConfig().getAnnotationIntrospector(),null);
        Collection<NamedType> types = mapper.getSubtypeResolver().collectAndResolveSubtypes(ac, mapper.getSerializationConfig(), mapper.getSerializationConfig().getAnnotationIntrospector());
        Set<Class<?>> registeredSubtypes = new HashSet<>();
        for(NamedType nt : types){
            registeredSubtypes.add(nt.getType());
        }

            //Second: get all subtypes of GraphVertex using reflection
        Reflections reflections = new Reflections();
        Set<Class<? extends GraphVertex>> subTypes = reflections.getSubTypesOf(GraphVertex.class);

            //Third: register all subtypes that are not already registered
        List<NamedType> toRegister = new ArrayList<>();
        for(Class<? extends GraphVertex> c : subTypes){
            if(!registeredSubtypes.contains(c)){
                String name;
                if(ClassUtils.isInnerClass(c)){
                    Class<?> c2 = c.getDeclaringClass();
                    name = c2.getSimpleName() + "$" + c.getSimpleName();
                } else {
                    name = c.getSimpleName();
                }
                toRegister.add(new NamedType(c, name));
            }
        }
        mapper = NeuralNetConfiguration.reinitMapperWithSubtypes(toRegister);


        try {
            return mapper.readValue(json, ComputationGraphConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public String toString() {
        return toJson();
    }

    @Override
    public ComputationGraphConfiguration clone(){
        ComputationGraphConfiguration conf = new ComputationGraphConfiguration();
        conf.layerNamesMap = (layerNamesMap != null ? new HashMap<>(this.layerNamesMap) : null);
        conf.layerNumbersMap = (layerNumbersMap != null ? new HashMap<>(this.layerNumbersMap) : null);
        conf.layers = new HashMap<>();
        for(Map.Entry<String,NeuralNetConfiguration> entry : this.layers.entrySet()){
            conf.layers.put(entry.getKey(),entry.getValue().clone());
        }
        conf.graphNodes = new HashMap<>();
        for(Map.Entry<String,GraphVertex> entry : this.graphNodes.entrySet()){
            conf.graphNodes.put(entry.getKey(),entry.getValue().clone());
        }
        conf.layerInputs = new HashMap<>();
        for( Map.Entry<String,List<String>> entry : this.layerInputs.entrySet() ){
            conf.layerInputs.put(entry.getKey(),new ArrayList<>(entry.getValue()));
        }
        conf.graphNodeInputs = new HashMap<>();
        for( Map.Entry<String,List<String>> entry : this.graphNodeInputs.entrySet() ){
            conf.graphNodeInputs.put(entry.getKey(), new ArrayList<>(entry.getValue()));
        }
        conf.networkInputs = new ArrayList<>(this.networkInputs);
        conf.networkOutputs = new ArrayList<>(this.networkOutputs);

        conf.pretrain = pretrain;
        conf.backprop = backprop;
        conf.inputPreProcessors = new HashMap<>();
        for(Map.Entry<String,InputPreProcessor> entry : inputPreProcessors.entrySet()){
            conf.inputPreProcessors.put(entry.getKey(),entry.getValue().clone());
        }
        conf.backpropType = backpropType;
        conf.tbpttFwdLength = tbpttFwdLength;
        conf.tbpttBackLength = tbpttBackLength;
        conf.redistributeParams = redistributeParams;



        return conf;
    }


    /** Check the configuration, make sure it is valid
     * @throws IllegalStateException if configuration is not valid
     * */
    public void validate(){
        if(networkInputs == null || networkInputs.size() < 1){
            throw new IllegalStateException("Invalid configuration: network has no inputs");
        }
        if(networkOutputs == null || networkOutputs.size() < 1){
            throw new IllegalStateException("Invalid configuration: network has no outputs");
        }

        //Check uniqueness of names for inputs, layers, GraphNodes
        for(String s : networkInputs){
            if(layers.containsKey(s)){
                throw new IllegalStateException("Invalid configuration: name \"" + s + "\" is present in both network inputs and layer names");
            }
            if(graphNodes.containsKey(s)){
                throw new IllegalStateException("Invalid configuration: name \"" + s + "\" is present in both network inputs and graph nodes");
            }
        }

        //Check: each layer & node has at least one input
        //and: check that all input keys/names for each layer actually exist
        for(Map.Entry<String,List<String>> e : layerInputs.entrySet() ){
            String layerName = e.getKey();
            if(e.getValue() == null || e.getValue().size() == 0){
                throw new IllegalStateException("Invalid configuration: layer \"" + layerName + "\" has no inputs");
            }
            for(String inputName : e.getValue()) {
                if (!layers.containsKey(inputName) && !graphNodes.containsKey(inputName) && !networkInputs.contains(inputName)) {
                    throw new IllegalStateException("Invalid configuration: layer \"" + layerName + "\" has input \"" +
                        inputName + "\" that does not exist");
                }
            }
        }
        for(Map.Entry<String,List<String>> e : graphNodeInputs.entrySet() ){
            String nodeName = e.getKey();
            if(e.getValue() == null || e.getValue().size() == 0){
                throw new IllegalStateException("Invalid configuration: graph node \"" + nodeName + "\" has no inputs");
            }
            for(String inputName : e.getValue()) {
                if (!layers.containsKey(inputName) && !graphNodes.containsKey(inputName) && !networkInputs.contains(inputName)) {
                    throw new IllegalStateException("Invalid configuration: GraphNode \"" + nodeName + "\" has input \"" +
                            inputName + "\" that does not exist");
                }
            }
        }

        //Check preprocessors
        for( String s : inputPreProcessors.keySet() ){
            if (!layers.containsKey(s) ) {
                throw new IllegalStateException("Invalid configuration: InputPreProcessor listed for layer \"" + s + "\" but layer \"" +
                        s + "\" does not exist");
            }
        }

        //Check output names:
        for(String s : networkOutputs){
            if(!layers.containsKey(s) && !graphNodes.containsKey(s)){
                throw new IllegalStateException("Invalid configuration: Output name \"" + s + "\" is not a Layer or GraphNode");
            }
        }

        //Check for no graph cycles: done in ComputationGraph.init()
    }

    /** Add FeedForward/RNN preprocessors automatically. */
    public void addPreProcessors(){

        System.out.println("addPreProcessors() functionality not yet implemented");
    }


    @Data
    public static class GraphBuilder {
        /** Map between layer numbers, and layer names */
        protected Map<Integer,String> layerNamesMap;
        protected Map<String,Integer> layerNumbersMap;

        protected Map<String,NeuralNetConfiguration.Builder> layers = new HashMap<>();
        protected Map<String,GraphVertex> graphNodes = new HashMap<>();

        /** Key: layer. Values: inputs to that layer */
        protected Map<String,List<String>> layerInputs = new HashMap<>();

        /** Key: graph node. Values: input to that node */
        protected Map<String,List<String>> graphNodeInputs = new HashMap<>();

        protected List<String> networkInputs = new ArrayList<>();
        protected List<String> networkOutputs = new ArrayList<>();

        protected boolean pretrain = true;
        protected boolean backprop = false;
        protected BackpropType backpropType = BackpropType.Standard;
        protected int tbpttFwdLength = 20;
        protected int tbpttBackLength = 20;

        protected Map<String,InputPreProcessor> inputPreProcessors = new HashMap<>();
        //whether to redistribute params or not
        protected boolean redistributeParams = false;

        //Whether to auto-add preprocessors - FF->RNN and RNN->FF etc
        protected boolean addPreProcessors = true;

        protected NeuralNetConfiguration.Builder globalConfiguration;


        public GraphBuilder(NeuralNetConfiguration.Builder globalConfiguration){
            this.globalConfiguration = globalConfiguration;
        }

        /**
         * Whether to redistribute parameters as a view or not
         * @param redistributeParams whether to redistribute parameters
         *                           as a view or not
         * @return
         */
        public GraphBuilder redistributeParams(boolean redistributeParams) {
            this.redistributeParams = redistributeParams;
            return this;
        }

        /**
         * Specify the processors.
         * These are used at each layer for doing things like normalization and
         * shaping of input.
         * @param processor what to use to preProcess the data.
         */
        public GraphBuilder inputPreProcessor(String layer, InputPreProcessor processor) {
            inputPreProcessors.put(layer,processor);
            return this;
        }

        /**
         * Whether to do back prop or not
         * @param backprop whether to do back prop or not
         * @return
         */
        public GraphBuilder backprop(boolean backprop) {
            this.backprop = backprop;
            return this;
        }

        /**Whether to do pre train or not
         * @param pretrain whether to do pre train or not
         * @return builder pattern
         */
        public GraphBuilder pretrain(boolean pretrain) {
            this.pretrain = pretrain;
            return this;
        }

        /**The type of backprop. Default setting is used for most networks (MLP, CNN etc),
         * but optionally truncated BPTT can be used for training recurrent neural networks.
         * If using TruncatedBPTT make sure you set both tBPTTForwardLength() and tBPTTBackwardLength()
         */
        public GraphBuilder backpropType(BackpropType type){
            this.backpropType = type;
            return this;
        }

        /**When doing truncated BPTT: how many steps of forward pass should we do
         * before doing (truncated) backprop?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * Typically tBPTTForwardLength parameter is same as the the tBPTTBackwardLength parameter,
         * but may be larger than it in some circumstances (but never smaller)<br>
         * Ideally your training data time series length should be divisible by this
         * This is the k1 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param forwardLength Forward length > 0, >= backwardLength
         */
        public GraphBuilder tBPTTForwardLength(int forwardLength){
            this.tbpttFwdLength = forwardLength;
            return this;
        }

        /**When doing truncated BPTT: how many steps of backward should we do?<br>
         * Only applicable when doing backpropType(BackpropType.TruncatedBPTT)<br>
         * This is the k2 parameter on pg23 of
         * http://www.cs.utoronto.ca/~ilya/pubs/ilya_sutskever_phd_thesis.pdf
         * @param backwardLength <= forwardLength
         */
        public GraphBuilder tBPTTBackwardLength(int backwardLength){
            this.tbpttBackLength = backwardLength;
            return this;
        }

        public GraphBuilder addLayer(String layerName, Layer layer, String... layerInputs ){
            NeuralNetConfiguration.Builder builder = globalConfiguration.clone();
            builder.layer(layer);
            layers.put(layerName, builder);

            //Automatically insert a MergeNode if layerInputs.length > 1
            //Layers can only have 1 input
            if(layerInputs != null && layerInputs.length > 1 ){
                String mergeName = layerName+"-merge";
                addNode(mergeName, new MergeVertex(), layerInputs );
                this.layerInputs.put(layerName,Collections.singletonList(mergeName));
            } else if(layerInputs != null) {
                this.layerInputs.put(layerName,Arrays.asList(layerInputs));
                layer.setLayerName(layerName);
            }
            return this;
        }

        public GraphBuilder addInputs( String... inputNames ){
            Collections.addAll(networkInputs,inputNames);
            return this;
        }

        public GraphBuilder setOutputs( String... outputNames ){
            Collections.addAll(networkOutputs,outputNames);
            return this;
        }

        public GraphBuilder addNode(String nodeName, GraphVertex node, String... nodeInputs ){
            graphNodes.put(nodeName,node);
            this.graphNodeInputs.put(nodeName, Arrays.asList(nodeInputs));
            return this;
        }

        /** Whether to automatically add preprocessors, between feed forward and RNN layers.<br>
         * This is enabled by default. The assumption here is that with RNN layers, the data input is a time series,
         * and thus RNN->FF preprocesors may need to be added on the input (if input feeds into for example a DenseLayer).
         * Any FF->RNN transitions will automatically add a {@link org.deeplearning4j.nn.conf.preprocessor.FeedForwardToRnnPreProcessor}<br>
         * <b>NOTE:</b> the assumptions made above may not for every network.
         * @param addPreProcessors If true: add RNN/FF preprocessors if applicable. False: add no preprocessors.
         * @return
         */
        public GraphBuilder addPreProcessors(boolean addPreProcessors){
            this.addPreProcessors = addPreProcessors;
            return this;
        }


        public ComputationGraphConfiguration build(){

            ComputationGraphConfiguration conf = new ComputationGraphConfiguration();
            conf.backprop = backprop;
            conf.pretrain = pretrain;
            conf.backpropType = backpropType;
            conf.tbpttBackLength = tbpttBackLength;
            conf.tbpttFwdLength = tbpttFwdLength;


            conf.layerNamesMap = layerNamesMap;
            conf.layerNumbersMap = layerNumbersMap;
            conf.layerInputs = layerInputs;
            conf.inputPreProcessors = inputPreProcessors;

            conf.networkInputs = networkInputs;
            conf.networkOutputs = networkOutputs;

            Map<String,NeuralNetConfiguration> layers = new HashMap<>();
            for( Map.Entry<String,NeuralNetConfiguration.Builder> entry : this.layers.entrySet() ){
                layers.put(entry.getKey(), entry.getValue().build());
            }

            conf.layers = layers;

            conf.graphNodes = this.graphNodes;
            conf.graphNodeInputs = this.graphNodeInputs;


            conf.validate();    //throws exception for invalid configuration

            if(addPreProcessors) conf.addPreProcessors();

            return conf;
        }
    }

}
