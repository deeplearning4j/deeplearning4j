
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


package org.deeplearning4j.nn.conf;

import lombok.*;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.distribution.Distribution;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.conf.memory.NetworkMemoryReport;
import org.deeplearning4j.nn.conf.serde.ComputationGraphConfigurationDeserializer;
import org.deeplearning4j.nn.conf.serde.JsonMappers;
import org.deeplearning4j.nn.conf.serde.MultiLayerConfigurationDeserializer;
import org.deeplearning4j.nn.weights.IWeightInit;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.OutputLayerUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossBinaryXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.nd4j.linalg.lossfunctions.impl.LossMSE;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;
import org.nd4j.shade.jackson.databind.*;
import org.nd4j.shade.jackson.databind.deser.BeanDeserializerModifier;
import org.nd4j.shade.jackson.databind.exc.InvalidTypeIdException;
import org.nd4j.shade.jackson.databind.module.SimpleModule;
import org.nd4j.shade.jackson.databind.node.ArrayNode;
import org.nd4j.shade.jackson.dataformat.yaml.YAMLFactory;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

@Data
@AllArgsConstructor(access = AccessLevel.PRIVATE)
@NoArgsConstructor
@Slf4j
public class MultiLayerConfiguration implements Serializable, Cloneable {

    protected List<NeuralNetConfiguration> confs;
    protected Map<Integer, InputPreProcessor> inputPreProcessors = new HashMap<>();
    protected BackpropType backpropType = BackpropType.Standard;
    protected int tbpttFwdLength = 20;
    protected int tbpttBackLength = 20;
    protected boolean validateOutputLayerConfig = true; //Default to legacy for pre 1.0.0-beta3 networks on deserialization

    @Getter
    @Setter
    protected WorkspaceMode trainingWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected WorkspaceMode inferenceWorkspaceMode = WorkspaceMode.ENABLED;

    @Getter
    @Setter
    protected CacheMode cacheMode;

    @Getter
    @Setter
    protected DataType dataType = DataType.FLOAT;   //Default to float for deserialization of beta3 and earlier nets

    //Counter for the number of parameter updates so far
    // This is important for learning rate schedules, for example, and is stored here to ensure it is persisted
    // for Spark and model serialization
    protected int iterationCount = 0;

    //Counter for the number of epochs completed so far. Used for per-epoch schedules
    protected int epochCount = 0;
    private static ObjectMapper mapper = mapper();
    private static ObjectMapper mapperYaml = mapperYaml();



    public static ObjectMapper mapperYaml() {
        ObjectMapper ret = new ObjectMapper(new YAMLFactory());
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);

        SimpleModule customDeserializerModule = new SimpleModule();
        customDeserializerModule.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                                                          JsonDeserializer<?> deserializer) {
                //Use our custom deserializers to handle backward compatibility for updaters -> IUpdater
                if (beanDesc.getBeanClass().equals(MultiLayerConfiguration.class)) {
                    return new MultiLayerConfigurationDeserializer(deserializer);
                }
                return deserializer;
            }
        });

        ret.registerModule(customDeserializerModule);
        return ret;
    }


    public static ObjectMapper mapper() {
        ObjectMapper ret = new ObjectMapper();
        ret.configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false);
        ret.configure(SerializationFeature.FAIL_ON_EMPTY_BEANS, false);
        ret.configure(MapperFeature.SORT_PROPERTIES_ALPHABETICALLY, true);
        ret.enable(SerializationFeature.INDENT_OUTPUT);
        SimpleModule customDeserializerModule = new SimpleModule();
        customDeserializerModule.setDeserializerModifier(new BeanDeserializerModifier() {
            @Override
            public JsonDeserializer<?> modifyDeserializer(DeserializationConfig config, BeanDescription beanDesc,
                                                          JsonDeserializer<?> deserializer) {
                //Use our custom deserializers to handle backward compatibility for updaters -> IUpdater
                if (beanDesc.getBeanClass().equals(MultiLayerConfiguration.class)) {
                    return new MultiLayerConfigurationDeserializer(deserializer);
                }
                return deserializer;
            }
        });

        ret.registerModule(customDeserializerModule);
        return ret;
    }

    public int getEpochCount() {
        return epochCount;
    }

    public void setEpochCount(int epochCount) {
        this.epochCount = epochCount;
        for (int i = 0; i < confs.size(); i++) {
            getConf(i).setEpochCount(epochCount);
        }
    }

    /**
     * @return JSON representation of NN configuration
     */
    public String toYaml() {
        try {
            return mapperYaml.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Create a neural net configuration from json
     *
     * @param json the neural net configuration from json
     * @return {@link MultiLayerConfiguration}
     */
    public static MultiLayerConfiguration fromYaml(String json) {
        try {
            return mapperYaml.readValue(json, MultiLayerConfiguration.class);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }


    /**
     * @return JSON representation of NN configuration
     */
    public String toJson() {
        //JSON mappers are supposed to be thread safe: however, in practice they seem to miss fields occasionally
        //when writeValueAsString is used by multiple threads. This results in invalid JSON. See issue #3243
        try {
            return mapper.writeValueAsString(this);
        } catch (org.nd4j.shade.jackson.core.JsonProcessingException e) {
            throw new RuntimeException(e);
        }

    }

    /**
     * Create a neural net configuration from json
     *
     * @param json the neural net configuration from json
     * @return {@link MultiLayerConfiguration}
     */
    public static  MultiLayerConfiguration fromJson(String json) {
          ObjectMapper mapper1 = mapper();
        MultiLayerConfiguration conf;
        try {
            conf = mapper1.readValue(json, MultiLayerConfiguration.class);
        } catch (InvalidTypeIdException e){
            if(e.getMessage().contains("@class")) {
                try {
                    //JSON may be legacy (1.0.0-alpha or earlier), attempt to load it using old format
                    return JsonMappers.getLegacyMapper().readValue(json, MultiLayerConfiguration.class);
                } catch (InvalidTypeIdException e2) {
                    //Check for legacy custom layers: "Could not resolve type id 'CustomLayer' as a subtype of [simple type, class org.deeplearning4j.nn.conf.layers.Layer]: known type ids = [Bidirectional, CenterLossOutputLayer, CnnLossLayer, ..."
                    //1.0.0-beta5: dropping support for custom layers defined in pre-1.0.0-beta format. Built-in layers from these formats still work
                    String msg = e2.getMessage();
                    if(msg != null && msg.contains("Could not resolve type id")){
                        throw new RuntimeException("Error deserializing MultiLayerConfiguration - configuration may have a custom " +
                                "layer, vertex or preprocessor, in pre version 1.0.0-beta JSON format.\nModels in legacy format with custom" +
                                " layers should be loaded in 1.0.0-beta to 1.0.0-beta4 and saved again, before loading in the current version of DL4J", e);
                    }
                    throw new RuntimeException(e2);
                } catch (IOException e2) {
                    throw new RuntimeException(e2);
                }
            }
            throw new RuntimeException(e);
        } catch (IOException e) {
            //Check if this exception came from legacy deserializer...
            String msg = e.getMessage();
            if (msg != null && msg.contains("legacy")) {
                throw new RuntimeException("Error deserializing MultiLayerConfiguration - configuration may have a custom " +
                        "layer, vertex or preprocessor, in pre version 1.0.0-alpha JSON format. These layers can be " +
                        "deserialized by first registering them with NeuralNetConfiguration.registerLegacyCustomClassesForJSON(Class...)", e);
            }
            throw new RuntimeException(e);
        }


        //To maintain backward compatibility after loss function refactoring (configs generated with v0.5.0 or earlier)
        // Previously: enumeration used for loss functions. Now: use classes
        // IN the past, could have only been an OutputLayer or RnnOutputLayer using these enums
        int layerCount = 0;
        JsonNode confs = null;
        for (NeuralNetConfiguration nnc : conf.getConfs()) {
            Layer l = nnc.getLayer();
            if (l instanceof BaseOutputLayer && ((BaseOutputLayer) l).getLossFn() == null) {
                //lossFn field null -> may be an old config format, with lossFunction field being for the enum
                //if so, try walking the JSON graph to extract out the appropriate enum value

                BaseOutputLayer ol = (BaseOutputLayer) l;
                try {
                    JsonNode jsonNode = mapper.readTree(json);
                    if (confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if (confs instanceof ArrayNode) {
                        ArrayNode layerConfs = (ArrayNode) confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if (outputLayerNNCNode == null)
                            return conf; //Should never happen...
                        JsonNode outputLayerNode = outputLayerNNCNode.get("layer");

                        JsonNode lossFunctionNode = null;
                        if (outputLayerNode.has("output")) {
                            lossFunctionNode = outputLayerNode.get("output").get("lossFunction");
                        } else if (outputLayerNode.has("rnnoutput")) {
                            lossFunctionNode = outputLayerNode.get("rnnoutput").get("lossFunction");
                        }

                        if (lossFunctionNode != null) {
                            String lossFunctionEnumStr = lossFunctionNode.asText();
                            LossFunctions.LossFunction lossFunction = null;
                            try {
                                lossFunction = LossFunctions.LossFunction.valueOf(lossFunctionEnumStr);
                            } catch (Exception e) {
                                log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",
                                        e);
                            }

                            if (lossFunction != null) {
                                switch (lossFunction) {
                                    case MSE:
                                        ol.setLossFn(new LossMSE());
                                        break;
                                    case XENT:
                                        ol.setLossFn(new LossBinaryXENT());
                                        break;
                                    case NEGATIVELOGLIKELIHOOD:
                                        ol.setLossFn(new LossNegativeLogLikelihood());
                                        break;
                                    case MCXENT:
                                        ol.setLossFn(new LossMCXENT());
                                        break;

                                    //Remaining: TODO
                                    case SQUARED_LOSS:
                                    case RECONSTRUCTION_CROSSENTROPY:
                                    default:
                                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not set loss function for {}",
                                                lossFunction);
                                        break;
                                }
                            }
                        }

                    } else {
                        log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON: layer 'confs' field is not an ArrayNode (is: {})",
                                (confs != null ? confs.getClass() : null));
                    }
                } catch (IOException e) {
                    log.warn("OutputLayer with null LossFunction or pre-0.6.0 loss function configuration detected: could not parse JSON",
                            e);
                    break;
                }
            }

            //Also, pre 0.7.2: activation functions were Strings ("activationFunction" field), not classes ("activationFn")
            //Try to load the old format if necessary, and create the appropriate IActivation instance
            if ((l instanceof BaseLayer) && ((BaseLayer) l).getActivationFn() == null) {
                try {
                    JsonNode jsonNode = mapper.readTree(json);
                    if (confs == null) {
                        confs = jsonNode.get("confs");
                    }
                    if (confs instanceof ArrayNode) {
                        ArrayNode layerConfs = (ArrayNode) confs;
                        JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                        if (outputLayerNNCNode == null)
                            return conf; //Should never happen...
                        JsonNode layerWrapperNode = outputLayerNNCNode.get("layer");

                        if (layerWrapperNode == null || layerWrapperNode.size() != 1) {
                            continue;
                        }

                        JsonNode layerNode = layerWrapperNode.elements().next();
                        JsonNode activationFunction = layerNode.get("activationFunction"); //Should only have 1 element: "dense", "output", etc

                        if (activationFunction != null) {
                            IActivation ia = Activation.fromString(activationFunction.asText()).getActivationFunction();
                            ((BaseLayer) l).setActivationFn(ia);
                        }
                    }

                } catch (IOException e) {
                    log.warn("Layer with null ActivationFn field or pre-0.7.2 activation function detected: could not parse JSON",
                            e);
                }
            }

            if(!handleLegacyWeightInitFromJson(json, l, mapper, confs, layerCount)) {
                return conf;
            }

            layerCount++;
        }
        return conf;
    }

    /**
     * Handle {@link WeightInit} and {@link Distribution} from legacy configs in Json format. Copied from handling of {@link Activation}
     * above.
     * @return True if all is well and layer iteration shall continue. False else-wise.
     */
    private static boolean handleLegacyWeightInitFromJson(String json, Layer l, ObjectMapper mapper, JsonNode confs, int layerCount) {
        if ((l instanceof BaseLayer) && ((BaseLayer) l).getWeightInitFn() == null) {
            try {
                JsonNode jsonNode = mapper.readTree(json);
                if (confs == null) {
                    confs = jsonNode.get("confs");
                }
                if (confs instanceof ArrayNode) {
                    ArrayNode layerConfs = (ArrayNode) confs;
                    JsonNode outputLayerNNCNode = layerConfs.get(layerCount);
                    if (outputLayerNNCNode == null)
                        return false; //Should never happen...
                    JsonNode layerWrapperNode = outputLayerNNCNode.get("layer");

                    if (layerWrapperNode == null || layerWrapperNode.size() != 1) {
                        return true;
                    }

                    JsonNode layerNode = layerWrapperNode.elements().next();
                    JsonNode weightInit = layerNode.get("weightInit"); //Should only have 1 element: "dense", "output", etc
                    JsonNode distribution = layerNode.get("dist");

                    Distribution dist = null;
                    if(distribution != null) {
                        dist = mapper.treeToValue(distribution, Distribution.class);
                    }

                    if (weightInit != null) {
                        IWeightInit wi = WeightInit.valueOf(weightInit.asText()).getWeightInitFunction(dist);
                        ((BaseLayer) l).setWeightInitFn(wi);
                    }
                }

            } catch (IOException e) {
                log.warn("Layer with null WeightInit detected: " + l.getLayerName() + ", could not parse JSON",
                        e);
            }
        }
        return true;

    }

    @Override
    public String toString() {
        return toJson();
    }

    public NeuralNetConfiguration getConf(int i) {
        return confs.get(i);
    }

    @Override
    public MultiLayerConfiguration clone() {
        try {
            MultiLayerConfiguration clone = (MultiLayerConfiguration) super.clone();

            if (clone.confs != null) {
                List<NeuralNetConfiguration> list = new ArrayList<>();
                for (NeuralNetConfiguration conf : clone.confs) {
                    list.add(conf.clone());
                }
                clone.confs = list;
            }

            if (clone.inputPreProcessors != null) {
                Map<Integer, InputPreProcessor> map = new HashMap<>();
                for (Map.Entry<Integer, InputPreProcessor> entry : clone.inputPreProcessors.entrySet()) {
                    map.put(entry.getKey(), entry.getValue().clone());
                }
                clone.inputPreProcessors = map;
            }

            clone.inferenceWorkspaceMode = this.inferenceWorkspaceMode;
            clone.trainingWorkspaceMode = this.trainingWorkspaceMode;
            clone.cacheMode = this.cacheMode;
            clone.validateOutputLayerConfig = this.validateOutputLayerConfig;
            clone.dataType = this.dataType;

            return clone;

        } catch (CloneNotSupportedException e) {
            throw new RuntimeException(e);
        }
    }

    public InputPreProcessor getInputPreProcess(int curr) {
        return inputPreProcessors.get(curr);
    }

    /**
     * Get a {@link MemoryReport} for the given MultiLayerConfiguration. This is used to estimate the
     * memory requirements for the given network configuration and input
     *
     * @param inputType Input types for the network
     * @return Memory report for the network
     */
    public NetworkMemoryReport getMemoryReport(InputType inputType) {

        Map<String, MemoryReport> memoryReportMap = new LinkedHashMap<>();
        int nLayers = confs.size();
        for (int i = 0; i < nLayers; i++) {
            String layerName = confs.get(i).getLayer().getLayerName();
            if (layerName == null) {
                layerName = String.valueOf(i);
            }

            //Pass input type through preprocessor, if necessary
            InputPreProcessor preproc = getInputPreProcess(i);
            //TODO memory requirements for preprocessor
            if (preproc != null) {
                inputType = preproc.getOutputType(inputType);
            }

            LayerMemoryReport report = confs.get(i).getLayer().getMemoryReport(inputType);
            memoryReportMap.put(layerName, report);

            inputType = confs.get(i).getLayer().getOutputType(i, inputType);
        }

        return new NetworkMemoryReport(memoryReportMap, MultiLayerConfiguration.class, "MultiLayerNetwork", inputType);
    }

    /**
     * For the given input shape/type for the network, return a list of activation sizes for each layer in the network.<br>
     * i.e., list.get(i) is the output activation sizes for layer i
     *
     * @param inputType Input type for the network
     * @return A lits of activation types for the network, indexed by layer number
     */
    public List<InputType> getLayerActivationTypes(@NonNull InputType inputType) {
        List<InputType> out = new ArrayList<>();
        int nLayers = confs.size();
        for (int i = 0; i < nLayers; i++) {
            InputPreProcessor preproc = getInputPreProcess(i);
            if (preproc != null) {
                inputType = preproc.getOutputType(inputType);
            }

            inputType = confs.get(i).getLayer().getOutputType(i, inputType);
            out.add(inputType);
        }
        return out;
    }

    @Data
    public static class Builder extends BaseBuilder {

        public MultiLayerConfiguration build() {
            //Validate BackpropType setting
            if ((tbpttBackLength != DEFAULT_TBPTT_LENGTH || tbpttFwdLength != DEFAULT_TBPTT_LENGTH) && backpropType != BackpropType.TruncatedBPTT) {
                log.warn("Truncated backpropagation through time lengths have been configured with values " + tbpttFwdLength
                        + " and " + tbpttBackLength + " but backprop type is set to " + backpropType + ". TBPTT configuration" +
                        " settings will only take effect if backprop type is set to BackpropType.TruncatedBPTT");
            }

            if(backpropType == BackpropType.TruncatedBPTT && validateTbpttConfig) {
                //Check for invalid combination - tbptt plus LastTimeStepLayer or
                for( int i = 0; i < confs.size(); i++) {
                    Layer l = confs.get(i).getLayer();
                    if(l instanceof LastTimeStep || l instanceof GlobalPoolingLayer) {
                        throw new IllegalStateException("Invalid network configuration detected: Truncated backpropagation through time (TBPTT)" +
                                " cannot be used with layer " + i + " of type " + l.getClass().getName() + ": TBPTT is incompatible with this layer type (which is designed " +
                                "to process entire sequences at once, and does support the type of sequence segments that TPBTT uses).\n" +
                                "This check can be disabled using validateTbpttConfig(false) but this is not recommended.");
                    }
                }
            }


            if (inputType == null && inputPreProcessors.get(0) == null) {
                //User hasn't set the InputType. Sometimes we can infer it...
                // For example, Dense/RNN layers, where preprocessor isn't set -> user is *probably* going to feed in
                // standard feedforward or RNN data
                //This isn't the most elegant implementation, but should avoid breaking backward compatibility here
                //Can't infer InputType for CNN layers, however (don't know image dimensions/depth)
                Layer firstLayer = confs.get(0).getLayer();
                if (firstLayer instanceof BaseRecurrentLayer) {
                    BaseRecurrentLayer brl = (BaseRecurrentLayer) firstLayer;
                    val nIn = brl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.recurrent(nIn, brl.getRnnDataFormat());
                    }
                } else if (firstLayer instanceof DenseLayer || firstLayer instanceof EmbeddingLayer
                        || firstLayer instanceof OutputLayer) {
                    //Can't just use "instanceof FeedForwardLayer" here. ConvolutionLayer is also a FeedForwardLayer
                    FeedForwardLayer ffl = (FeedForwardLayer) firstLayer;
                    val nIn = ffl.getNIn();
                    if (nIn > 0) {
                        inputType = InputType.feedForward(nIn);
                    }
                }
            }


            //Add preprocessors and set nIns, if InputType has been set
            // Builder.inputType field can be set in 1 of 4 ways:
            // 1. User calls setInputType directly
            // 2. Via ConvolutionLayerSetup -> internally calls setInputType(InputType.convolutional(...))
            // 3. Via the above code: i.e., assume input is as expected  by the RNN or dense layer -> sets the inputType field
            if (inputType != null) {
                InputType currentInputType = inputType;
                for (int i = 0; i < confs.size(); i++) {
                    Layer l = confs.get(i).getLayer();
                    if (inputPreProcessors.get(i) == null) {
                        //Don't override preprocessor setting, but set preprocessor if required...
                        InputPreProcessor inputPreProcessor = l.getPreProcessorForInputType(currentInputType);
                        if (inputPreProcessor != null) {
                            inputPreProcessors.put(i, inputPreProcessor);
                        }
                    }

                    InputPreProcessor inputPreProcessor = inputPreProcessors.get(i);
                    if (inputPreProcessor != null) {
                        currentInputType = inputPreProcessor.getOutputType(currentInputType);
                    }
                    if(i > 0) {
                        Layer layer = confs.get(i - 1).getLayer();
                        //convolution 1d is an edge case where it has rnn input type but the filters
                        //should be the output
                        if(layer instanceof Convolution1DLayer) {
                            if(l instanceof DenseLayer && inputType instanceof InputType.InputTypeRecurrent) {
                                FeedForwardLayer feedForwardLayer = (FeedForwardLayer) l;
                                if(inputType instanceof InputType.InputTypeRecurrent) {
                                    InputType.InputTypeRecurrent recurrent = (InputType.InputTypeRecurrent) inputType;
                                    feedForwardLayer.setNIn(recurrent.getTimeSeriesLength());
                                }
                            }
                            else
                                l.setNIn(currentInputType, overrideNinUponBuild); //Don't override the nIn setting, if it's manually set by the user
                        }
                        else
                            l.setNIn(currentInputType, overrideNinUponBuild); //Don't override the nIn setting, if it's manually set by the user

                    }
                    else
                        l.setNIn(currentInputType, overrideNinUponBuild); //Don't override the nIn setting, if it's manually set by the user


                    currentInputType = l.getOutputType(i, currentInputType);
                }

            }

            MultiLayerConfiguration conf = new MultiLayerConfiguration();
            conf.confs = this.confs;
            conf.inputPreProcessors = inputPreProcessors;
            conf.backpropType = backpropType;
            conf.tbpttFwdLength = tbpttFwdLength;
            conf.tbpttBackLength = tbpttBackLength;
            conf.trainingWorkspaceMode = trainingWorkspaceMode;
            conf.inferenceWorkspaceMode = inferenceWorkspaceMode;
            conf.cacheMode = cacheMode;
            conf.dataType = dataType;

            Nd4j.getRandom().setSeed(conf.getConf(0).getSeed());

            //Validate output layer configuration
            if (validateOutputConfig) {
                //Validate output layer configurations...
                for (NeuralNetConfiguration n : conf.getConfs()) {
                    Layer l = n.getLayer();
                    OutputLayerUtil.validateOutputLayer(l.getLayerName(), l); //No-op for non output/loss layers
                }
            }

            return conf;

        }
    }
}
